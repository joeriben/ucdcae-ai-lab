"""
LLM Client — HTTP wrapper for GPU Service LLM inference

Follows DiffusersClient pattern (diffusers_client.py).
Primary: GPU Service (safetensors, VRAMCoordinator).
Fallback: Ollama (GGUF) on ConnectionError/Timeout.

LLM errors (OOM, bad model) are propagated, not fallen back.
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class LLMClient:
    """HTTP client for LLM inference via GPU Service with Ollama fallback."""

    def __init__(self):
        from config import GPU_SERVICE_URL, GPU_SERVICE_TIMEOUT, OLLAMA_API_BASE_URL
        self.gpu_url = GPU_SERVICE_URL.rstrip('/')
        self.timeout = GPU_SERVICE_TIMEOUT
        self.ollama_url = OLLAMA_API_BASE_URL.rstrip('/')
        logger.info(
            f"[LLM-CLIENT] Initialized: gpu={self.gpu_url}, "
            f"ollama={self.ollama_url}, timeout={self.timeout}s"
        )

    def _gpu_post(self, path: str, data: dict) -> Optional[dict]:
        """POST to GPU service. Returns JSON or None on connection failure."""
        import requests
        url = f"{self.gpu_url}{path}"
        try:
            resp = requests.post(url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout):
            logger.debug(f"[LLM-CLIENT] GPU service unreachable at {url}")
            return None  # Triggers Ollama fallback
        except Exception as e:
            logger.error(f"[LLM-CLIENT] GPU service error: {e}")
            return None

    def _gpu_get(self, path: str) -> Optional[dict]:
        """GET from GPU service."""
        import requests
        url = f"{self.gpu_url}{path}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def _ollama_chat(self, model: str, messages: list, images: Optional[list] = None,
                     temperature: float = 0.7, max_new_tokens: int = 500,
                     keep_alive: str = "10m",
                     repetition_penalty: Optional[float] = None,
                     enable_thinking: bool = True) -> Optional[Dict[str, Any]]:
        """Ollama fallback for chat."""
        import requests

        # Strip local/ prefix for Ollama
        ollama_model = model.replace("local/", "") if model.startswith("local/") else model

        # Inject images into first user message if provided
        ollama_messages = []
        for msg in messages:
            m = dict(msg)
            if images and m.get("role") == "user" and not any("images" in prev for prev in ollama_messages):
                m["images"] = images
            ollama_messages.append(m)

        options = {"temperature": temperature, "num_predict": max_new_tokens}
        if repetition_penalty is not None:
            options["repeat_penalty"] = repetition_penalty

        payload = {
            "model": ollama_model,
            "messages": ollama_messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options,
        }
        # Ollama Qwen3 thinking suppression: /no_think suffix or think param
        if not enable_thinking:
            payload["think"] = False

        try:
            resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            msg = resp.json().get("message", {})
            return {
                "content": msg.get("content", "").strip(),
                "thinking": msg.get("thinking", "").strip() or None,
            }
        except Exception as e:
            logger.error(f"[LLM-CLIENT] Ollama chat fallback failed ({ollama_model}): {e}")
            return None

    def _ollama_generate(self, model: str, prompt: str,
                         temperature: float = 0.7, max_new_tokens: int = 500,
                         keep_alive: str = "10m",
                         repetition_penalty: Optional[float] = None,
                         enable_thinking: bool = True) -> Optional[Dict[str, Any]]:
        """Ollama fallback for generate."""
        import requests

        ollama_model = model.replace("local/", "") if model.startswith("local/") else model

        options = {"temperature": temperature, "num_predict": max_new_tokens}
        if repetition_penalty is not None:
            options["repeat_penalty"] = repetition_penalty

        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options,
        }
        if not enable_thinking:
            payload["think"] = False

        try:
            resp = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            return {
                "response": resp.json().get("response", "").strip(),
                "thinking": None,
            }
        except Exception as e:
            logger.error(f"[LLM-CLIENT] Ollama generate fallback failed ({ollama_model}): {e}")
            return None

    # =========================================================================
    # Public API
    # =========================================================================

    async def is_available(self) -> bool:
        """Check if GPU Service LLM inference is available."""
        import asyncio
        try:
            result = await asyncio.to_thread(self._gpu_get, '/api/llm/available')
            return result is not None and result.get('available', False)
        except Exception:
            return False

    def chat(self, model: str, messages: list, images: Optional[list] = None,
             temperature: float = 0.7, max_new_tokens: int = 500,
             keep_alive: str = "10m",
             repetition_penalty: Optional[float] = None,
             enable_thinking: bool = True) -> Optional[Dict[str, Any]]:
        """Messages-based chat. Direct to Ollama (GPU Service LLM bypassed).

        Returns {"content": str, "thinking": str|None} or None on total failure.
        """
        # GPU Service LLM inference DISABLED — causes cascading failures.
        # Go directly to Ollama. Re-enable when GPU Service LLM is fixed.
        return self._ollama_chat(model, messages, images, temperature, max_new_tokens, keep_alive, repetition_penalty, enable_thinking)

    def generate(self, model: str, prompt: str,
                 temperature: float = 0.7, max_new_tokens: int = 500,
                 keep_alive: str = "10m",
                 repetition_penalty: Optional[float] = None,
                 enable_thinking: bool = True) -> Optional[Dict[str, Any]]:
        """Raw prompt generation. Direct to Ollama (GPU Service LLM bypassed).

        Returns {"response": str, "thinking": str|None} or None on total failure.
        """
        # GPU Service LLM inference DISABLED — causes cascading failures.
        # Go directly to Ollama. Re-enable when GPU Service LLM is fixed.
        return self._ollama_generate(model, prompt, temperature, max_new_tokens, keep_alive, repetition_penalty, enable_thinking)

    def list_models(self) -> list:
        """List loaded models on GPU service."""
        result = self._gpu_get('/api/llm/models')
        if result:
            return result.get("models", [])
        return []

    def unload_model(self, model_id: str) -> bool:
        """Force unload a model from GPU service."""
        result = self._gpu_post('/api/llm/unload', {"model_id": model_id})
        if result:
            return result.get("success", False)
        return False
