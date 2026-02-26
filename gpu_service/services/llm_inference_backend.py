"""
LLM Inference Backend - Production LLM inference for DevServer

Routes all LLM inference (safety, DSGVO, VLM, translation, interception, chat)
through the GPU Service's VRAMCoordinator instead of Ollama.

Follows TextBackend pattern (multi-model VRAMBackend) but WITHOUT
introspection flags (output_hidden_states, output_attentions) — pure inference.

Key differences from TextBackend:
- No introspection (saves VRAM)
- Auto-detects vision vs text models
- Extracts <think>...</think> blocks into separate 'thinking' field
- Accepts Ollama-style model names, resolves via LLM_MODEL_MAP
"""

import logging
import re
import threading
import time
from typing import Optional, List, Dict, Any
import asyncio

logger = logging.getLogger(__name__)


def _resolve_model_id(model_name: str) -> str:
    """Resolve Ollama-style model name to HuggingFace model ID."""
    from config import LLM_MODEL_MAP

    # Strip local/ prefix
    cleaned = model_name.replace("local/", "") if model_name.startswith("local/") else model_name
    # Try map lookup
    if cleaned in LLM_MODEL_MAP:
        return LLM_MODEL_MAP[cleaned]
    # Already a HF ID (contains /)
    return cleaned


def _extract_thinking(text: str) -> tuple:
    """Extract <think>...</think> blocks from response text.

    Returns (content_without_think, thinking_text_or_None).
    """
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        thinking = "\n".join(m.strip() for m in matches)
        content = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        return content, thinking
    return text, None


def _estimate_llm_vram(model_id: str, quantization: str = "bf16") -> float:
    """Estimate VRAM for an LLM model in GB."""
    from config import LLM_QUANT_MULTIPLIERS

    # Parameter-count heuristic from model name
    match = re.search(r'(\d+\.?\d*)[Bb]', model_id)
    if match:
        params_b = float(match.group(1))
        base_vram = params_b * 2.0  # ~2GB per billion params at bf16
    else:
        base_vram = 4.0  # Conservative default for unknown models
        logger.warning(f"[LLM-INF] Unknown model size for {model_id}, assuming {base_vram}GB")

    multiplier = LLM_QUANT_MULTIPLIERS.get(quantization, 1.0)
    return base_vram * multiplier


def _choose_quantization(model_id: str, available_vram_gb: float) -> tuple:
    """Choose optimal quantization level for available VRAM.

    Returns (quantization_string, estimated_vram_gb).
    """
    for quant in ["bf16", "int8", "int4"]:
        estimated = _estimate_llm_vram(model_id, quant)
        if estimated * 1.1 < available_vram_gb:
            return quant, estimated
    return "int4", _estimate_llm_vram(model_id, "int4")


def _is_vision_model(model_id: str) -> bool:
    """Detect if a model is a vision-language model by its ID."""
    vision_indicators = ['-vl', '-vision', 'llava', 'qwen2-vl', 'qwen2.5-vl',
                         'qwen3-vl', 'llama-3.2-vision', 'Vision']
    model_lower = model_id.lower()
    return any(ind.lower() in model_lower for ind in vision_indicators)


class LLMInferenceBackend:
    """Production LLM inference with VRAM coordination.

    Integrates with VRAMCoordinator for cross-backend eviction.
    Unlike TextBackend, does NOT enable hidden_states/attentions (pure inference).
    """

    def __init__(self):
        from config import LLM_DEVICE, LLM_DEFAULT_QUANTIZATION

        self.device = LLM_DEVICE
        self.default_quantization = LLM_DEFAULT_QUANTIZATION

        # Model cache (mirrors TextBackend pattern)
        self._models: Dict[str, tuple] = {}  # model_id -> (model, tokenizer/processor)
        self._model_last_used: Dict[str, float] = {}
        self._model_vram_mb: Dict[str, float] = {}
        self._model_in_use: Dict[str, int] = {}
        self._model_type: Dict[str, str] = {}  # "causal" | "vision"
        self._model_quant: Dict[str, str] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._model_locks_lock = threading.Lock()  # protects dict access only

        self._register_with_coordinator()

        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self._total_vram_gb = props.total_memory / 1024**3
            else:
                self._total_vram_gb = 0
        except ImportError:
            self._total_vram_gb = 0

        logger.info(f"[LLM-INF] Initialized: device={self.device}, total_vram={self._total_vram_gb:.1f}GB")

    def _register_with_coordinator(self):
        """Register with VRAM coordinator for cross-backend eviction."""
        try:
            from services.vram_coordinator import get_vram_coordinator
            coordinator = get_vram_coordinator()
            coordinator.register_backend(self)
            logger.info("[LLM-INF] Registered with VRAM coordinator")
        except Exception as e:
            logger.warning(f"[LLM-INF] Failed to register with VRAM coordinator: {e}")

    def _get_model_lock(self, model_id: str) -> threading.Lock:
        """Get or create a per-model lock."""
        with self._model_locks_lock:
            if model_id not in self._model_locks:
                self._model_locks[model_id] = threading.Lock()
            return self._model_locks[model_id]

    # =========================================================================
    # VRAMBackend Protocol Implementation
    # =========================================================================

    def get_backend_id(self) -> str:
        return "llm_inference"

    def get_registered_models(self) -> List[Dict[str, Any]]:
        from services.vram_coordinator import EvictionPriority
        return [
            {
                "model_id": mid,
                "vram_mb": self._model_vram_mb.get(mid, 0),
                "priority": EvictionPriority.NORMAL,
                "last_used": self._model_last_used.get(mid, 0),
                "in_use": self._model_in_use.get(mid, 0),
            }
            for mid in self._models
        ]

    def evict_model(self, model_id: str) -> bool:
        return self._unload_model_sync(model_id)

    # =========================================================================
    # Model Loading
    # =========================================================================

    def _get_free_vram_gb(self) -> float:
        import torch
        if not torch.cuda.is_available():
            return 0
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        return (total - allocated) / 1024**3

    def _load_model_sync(self, model_id: str, quantization: Optional[str] = None) -> bool:
        """Load a model with VRAM coordination and per-model locking."""
        import torch

        # Fast path: already loaded? (no lock needed)
        if model_id in self._models:
            self._model_last_used[model_id] = time.time()
            return True

        # Per-model lock: only THIS model blocks
        with self._get_model_lock(model_id):
            # Double-check under lock
            if model_id in self._models:
                self._model_last_used[model_id] = time.time()
                return True

            # Determine quantization
            free_vram = self._get_free_vram_gb()
            if quantization is None:
                quantization, estimated_gb = _choose_quantization(model_id, free_vram)
            else:
                estimated_gb = _estimate_llm_vram(model_id, quantization)

            required_mb = estimated_gb * 1024

            logger.info(
                f"[LLM-INF] Loading {model_id}: "
                f"quant={quantization}, estimated={estimated_gb:.1f}GB, free={free_vram:.1f}GB"
            )

            # Request VRAM from coordinator
            try:
                from services.vram_coordinator import get_vram_coordinator, EvictionPriority
                coordinator = get_vram_coordinator()
                coordinator.request_vram("llm_inference", required_mb, EvictionPriority.NORMAL)
            except Exception as e:
                logger.warning(f"[LLM-INF] VRAM coordinator request failed: {e}")

            # Detect model type and load
            try:
                is_vision = _is_vision_model(model_id)
                vram_before = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0

                # Build load kwargs — NO device_map, explicit .to(device) instead
                load_kwargs = {
                    "low_cpu_mem_usage": True,
                    "local_files_only": True,
                }

                if quantization == "bf16":
                    load_kwargs["dtype"] = torch.bfloat16
                elif quantization == "fp16":
                    load_kwargs["dtype"] = torch.float16
                elif quantization in ("int8", "int4", "nf4"):
                    from transformers import BitsAndBytesConfig
                    if quantization == "int8":
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    else:
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4" if quantization == "nf4" else "fp4"
                        )

                if is_vision:
                    from transformers import AutoModelForVision2Seq, AutoProcessor
                    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
                    model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)
                    model = model.to(self.device)
                    model.eval()
                    tokenizer_or_processor = processor
                    model_type = "vision"
                else:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    # NO output_hidden_states, NO output_attentions (pure inference)
                    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                    model = model.to(self.device)
                    model.eval()
                    tokenizer_or_processor = tokenizer
                    model_type = "causal"

                vram_after = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
                actual_vram_mb = (vram_after - vram_before) / (1024 * 1024)

                self._models[model_id] = (model, tokenizer_or_processor)
                self._model_last_used[model_id] = time.time()
                self._model_vram_mb[model_id] = actual_vram_mb
                self._model_in_use[model_id] = 0
                self._model_type[model_id] = model_type
                self._model_quant[model_id] = quantization

                logger.info(
                    f"[LLM-INF] Loaded {model_id}: type={model_type}, "
                    f"quant={quantization}, actual_vram={actual_vram_mb:.0f}MB"
                )
                return True

            except Exception as e:
                logger.error(f"[LLM-INF] Failed to load {model_id}: {e}")
                import traceback
                traceback.print_exc()
                return False

    def _unload_model_sync(self, model_id: str) -> bool:
        """Unload a model from memory."""
        import torch
        if model_id not in self._models:
            return False

        with self._get_model_lock(model_id):
            # Double-check under lock
            if model_id not in self._models:
                return False
            try:
                del self._models[model_id]
                self._model_last_used.pop(model_id, None)
                self._model_vram_mb.pop(model_id, None)
                self._model_in_use.pop(model_id, None)
                self._model_type.pop(model_id, None)
                self._model_quant.pop(model_id, None)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info(f"[LLM-INF] Unloaded {model_id}")
                return True
            except Exception as e:
                logger.error(f"[LLM-INF] Error unloading {model_id}: {e}")
                return False

    async def load_model(self, model_id: str, quantization: Optional[str] = None) -> bool:
        return await asyncio.to_thread(self._load_model_sync, model_id, quantization)

    async def unload_model(self, model_id: str) -> bool:
        return await asyncio.to_thread(self._unload_model_sync, model_id)

    def get_loaded_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "model_id": mid,
                "vram_mb": self._model_vram_mb.get(mid, 0),
                "quantization": self._model_quant.get(mid, "unknown"),
                "type": self._model_type.get(mid, "unknown"),
                "in_use": self._model_in_use.get(mid, 0) > 0,
                "last_used": self._model_last_used.get(mid, 0),
            }
            for mid in self._models
        ]

    # =========================================================================
    # Inference Methods
    # =========================================================================

    async def chat(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 500,
        repetition_penalty: Optional[float] = None,
        enable_thinking: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Messages-based chat inference.

        Args:
            model_name: Ollama-style or HF model name
            messages: List of {"role": "user/assistant/system", "content": "..."}
            images: Optional list of base64-encoded images (for vision models)
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            repetition_penalty: Penalty for repeated tokens (e.g. 1.5 for Qwen3)
            enable_thinking: Whether to enable <think> mode in chat template (default True)

        Returns:
            {"content": str, "thinking": str|None} or None on failure
        """
        model_id = _resolve_model_id(model_name)
        if not await self.load_model(model_id):
            logger.error(f"[LLM-INF] Failed to load model {model_id}")
            return None

        model, tokenizer_or_processor = self._models[model_id]
        self._model_in_use[model_id] = self._model_in_use.get(model_id, 0) + 1
        self._model_last_used[model_id] = time.time()

        try:
            import torch
            model_type = self._model_type.get(model_id, "causal")

            if model_type == "vision" and images:
                return await self._chat_vision(model, tokenizer_or_processor, messages, images, temperature, max_new_tokens)
            else:
                return await self._chat_text(model, tokenizer_or_processor, messages, temperature, max_new_tokens, repetition_penalty, enable_thinking)
        except Exception as e:
            logger.error(f"[LLM-INF] Chat inference failed ({model_id}): {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self._model_in_use[model_id] -= 1

    async def _chat_text(self, model, tokenizer, messages, temperature, max_new_tokens, repetition_penalty=None, enable_thinking=True):
        """Text-only chat inference."""
        import torch

        def _run():
            # Apply chat template
            if hasattr(tokenizer, 'apply_chat_template'):
                # Pass enable_thinking to suppress <think> mode when False
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if not enable_thinking:
                    template_kwargs["enable_thinking"] = False
                try:
                    input_text = tokenizer.apply_chat_template(messages, **template_kwargs)
                except TypeError:
                    # Tokenizer doesn't support enable_thinking kwarg — fall back
                    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback: concatenate messages
                input_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                input_text += "\nassistant:"

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if temperature > 0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                else:
                    gen_kwargs["do_sample"] = False

                if repetition_penalty is not None:
                    gen_kwargs["repetition_penalty"] = repetition_penalty

                outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)

            # Decode only new tokens
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            full_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            content, thinking = _extract_thinking(full_response)
            return {"content": content, "thinking": thinking}

        return await asyncio.to_thread(_run)

    async def _chat_vision(self, model, processor, messages, images, temperature, max_new_tokens):
        """Vision-language chat inference."""
        import torch
        from PIL import Image
        import base64
        import io

        def _run():
            # Decode base64 images
            pil_images = []
            for img_b64 in images:
                img_bytes = base64.b64decode(img_b64)
                pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            # Build conversation for vision model
            # Most VLMs use processor.apply_chat_template
            if hasattr(processor, 'apply_chat_template'):
                # Add image placeholders to user message
                vision_messages = []
                for msg in messages:
                    if msg["role"] == "user" and pil_images:
                        # Build content list with images and text
                        content_parts = []
                        for img in pil_images:
                            content_parts.append({"type": "image"})
                        content_parts.append({"type": "text", "text": msg["content"]})
                        vision_messages.append({"role": "user", "content": content_parts})
                    else:
                        vision_messages.append(msg)

                input_text = processor.apply_chat_template(vision_messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=input_text, images=pil_images if pil_images else None, return_tensors="pt").to(model.device)
            else:
                # Fallback
                text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                inputs = processor(text=text, images=pil_images[0] if pil_images else None, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                }
                if temperature > 0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                else:
                    gen_kwargs["do_sample"] = False

                outputs = model.generate(**inputs, **gen_kwargs)

            # Decode only new tokens
            input_len = inputs.get("input_ids", inputs.get("pixel_values")).shape[1] if "input_ids" in inputs else 0
            if "input_ids" in inputs:
                input_len = inputs["input_ids"].shape[1]
                new_tokens = outputs[0][input_len:]
            else:
                new_tokens = outputs[0]

            full_response = processor.decode(new_tokens, skip_special_tokens=True).strip()
            content, thinking = _extract_thinking(full_response)
            return {"content": content, "thinking": thinking}

        return await asyncio.to_thread(_run)

    async def generate(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 500,
        repetition_penalty: Optional[float] = None,
        enable_thinking: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Raw prompt generation (mirrors Ollama /api/generate).

        Args:
            model_name: Ollama-style or HF model name
            prompt: Raw prompt string
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            repetition_penalty: Penalty for repeated tokens (e.g. 1.5 for Qwen3)
            enable_thinking: Whether to enable <think> mode (for generate, suppresses via chat template wrapper)

        Returns:
            {"response": str, "thinking": str|None} or None on failure
        """
        model_id = _resolve_model_id(model_name)
        if not await self.load_model(model_id):
            logger.error(f"[LLM-INF] Failed to load model {model_id}")
            return None

        model, tokenizer = self._models[model_id]
        self._model_in_use[model_id] = self._model_in_use.get(model_id, 0) + 1
        self._model_last_used[model_id] = time.time()

        try:
            import torch

            def _run():
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    gen_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    if temperature > 0:
                        gen_kwargs["do_sample"] = True
                        gen_kwargs["temperature"] = temperature
                    else:
                        gen_kwargs["do_sample"] = False

                    if repetition_penalty is not None:
                        gen_kwargs["repetition_penalty"] = repetition_penalty

                    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)

                new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                full_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                content, thinking = _extract_thinking(full_response)
                return {"response": content, "thinking": thinking}

            return await asyncio.to_thread(_run)

        except Exception as e:
            logger.error(f"[LLM-INF] Generate failed ({model_id}): {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self._model_in_use[model_id] -= 1


# =============================================================================
# Singleton
# =============================================================================

_backend: Optional[LLMInferenceBackend] = None


def get_llm_inference_backend() -> LLMInferenceBackend:
    """Get LLMInferenceBackend singleton."""
    global _backend
    if _backend is None:
        _backend = LLMInferenceBackend()
    return _backend


def reset_llm_inference_backend():
    """Reset singleton (for testing)."""
    global _backend
    _backend = None
