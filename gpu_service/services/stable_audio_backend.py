"""
Stable Audio Backend - Audio generation via Stable Audio Open (Diffusers)

Cross-aesthetic generation support: accepts pre-computed embeddings
for direct CLIP Vision → audio conditioning (Strategy A).

Features:
- StableAudioPipeline from diffusers
- Text + duration conditioning (T5-Base, 768d)
- 44100Hz stereo output, max 47.55s
- On-demand lazy loading
- VRAM management with coordinator integration
- Embedding injection for cross-aesthetic use

Usage:
    backend = get_stable_audio_backend()
    if await backend.is_available():
        audio_bytes = await backend.generate_audio(
            prompt="ocean waves crashing on rocks",
            duration_seconds=10.0
        )
"""

import logging
import time
from typing import Optional, Dict, Any, List
import asyncio

logger = logging.getLogger(__name__)


class StableAudioGenerator:
    """
    Audio generation using Stable Audio Open (stabilityai/stable-audio-open-1.0).

    Supports:
    - Text-to-audio generation via StableAudioPipeline
    - Pre-computed embedding injection (for cross-aesthetic strategies)
    - Lazy model loading for VRAM efficiency
    - VRAM coordinator integration for cross-backend eviction
    """

    def __init__(self):
        from config import (
            STABLE_AUDIO_MODEL_ID,
            STABLE_AUDIO_DEVICE,
            STABLE_AUDIO_DTYPE,
            STABLE_AUDIO_LAZY_LOAD,
            STABLE_AUDIO_MAX_DURATION,
            STABLE_AUDIO_SAMPLE_RATE,
        )

        self.model_id = STABLE_AUDIO_MODEL_ID
        self.device = STABLE_AUDIO_DEVICE
        self.dtype_str = STABLE_AUDIO_DTYPE
        self.lazy_load = STABLE_AUDIO_LAZY_LOAD
        self.max_duration = STABLE_AUDIO_MAX_DURATION
        self.sample_rate = STABLE_AUDIO_SAMPLE_RATE

        # Pipeline (lazy-loaded)
        self._pipeline = None
        self._is_loaded = False
        self._vram_mb: float = 0
        self._last_used: float = 0
        self._in_use: int = 0

        self._register_with_coordinator()

        logger.info(
            f"[STABLE-AUDIO] Initialized: model={self.model_id}, "
            f"device={self.device}, lazy_load={self.lazy_load}"
        )

    def _register_with_coordinator(self):
        try:
            from services.vram_coordinator import get_vram_coordinator
            coordinator = get_vram_coordinator()
            coordinator.register_backend(self)
            logger.info("[STABLE-AUDIO] Registered with VRAM coordinator")
        except Exception as e:
            logger.warning(f"[STABLE-AUDIO] Failed to register with VRAM coordinator: {e}")

    # =========================================================================
    # VRAMBackend Protocol
    # =========================================================================

    def get_backend_id(self) -> str:
        return "stable_audio"

    def get_registered_models(self) -> List[Dict[str, Any]]:
        from services.vram_coordinator import EvictionPriority

        if not self._is_loaded:
            return []

        return [
            {
                "model_id": self.model_id,
                "vram_mb": self._vram_mb,
                "priority": EvictionPriority.NORMAL,
                "last_used": self._last_used,
                "in_use": self._in_use,
            }
        ]

    def evict_model(self, model_id: str) -> bool:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.unload_pipeline())
        finally:
            loop.close()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def is_available(self) -> bool:
        try:
            from diffusers import StableAudioPipeline  # noqa: F401
            return True
        except ImportError:
            logger.error("[STABLE-AUDIO] diffusers not installed or StableAudioPipeline not available")
            return False

    async def _load_pipeline(self) -> bool:
        if self._is_loaded:
            return True

        try:
            import torch
            from diffusers import StableAudioPipeline

            # Request VRAM from coordinator (~4GB estimated)
            try:
                from services.vram_coordinator import get_vram_coordinator, EvictionPriority
                coordinator = get_vram_coordinator()
                coordinator.request_vram("stable_audio", 4000, EvictionPriority.NORMAL)
            except Exception as e:
                logger.warning(f"[STABLE-AUDIO] VRAM coordinator request failed: {e}")

            logger.info(f"[STABLE-AUDIO] Loading pipeline: {self.model_id}...")

            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            torch_dtype = dtype_map.get(self.dtype_str, torch.float16)

            vram_before = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0

            def _load():
                pipe = StableAudioPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                )
                pipe = pipe.to(self.device)
                return pipe

            self._pipeline = await asyncio.to_thread(_load)
            self._is_loaded = True
            self._last_used = time.time()

            vram_after = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
            self._vram_mb = (vram_after - vram_before) / (1024 * 1024)

            logger.info(f"[STABLE-AUDIO] Pipeline loaded (VRAM: {self._vram_mb:.0f}MB)")
            return True

        except Exception as e:
            logger.error(f"[STABLE-AUDIO] Failed to load pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def unload_pipeline(self) -> bool:
        if not self._is_loaded:
            return False

        try:
            import torch

            del self._pipeline
            self._pipeline = None
            self._is_loaded = False
            self._vram_mb = 0
            self._in_use = 0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("[STABLE-AUDIO] Pipeline unloaded")
            return True

        except Exception as e:
            logger.error(f"[STABLE-AUDIO] Error unloading pipeline: {e}")
            return False

    # =========================================================================
    # Generation
    # =========================================================================

    async def generate_audio(
        self,
        prompt: str,
        duration_seconds: float = 10.0,
        negative_prompt: str = "",
        steps: int = 100,
        cfg_scale: float = 7.0,
        seed: int = -1,
        output_format: str = "wav",
    ) -> Optional[bytes]:
        """
        Generate audio from text prompt.

        Args:
            prompt: Text description of desired audio
            duration_seconds: Duration in seconds (max 47.55)
            negative_prompt: Negative conditioning text
            steps: Number of inference steps (default 100)
            cfg_scale: Classifier-free guidance scale
            seed: Seed for reproducibility (-1 = random)
            output_format: 'wav' or 'mp3'

        Returns:
            Audio bytes or None on failure
        """
        try:
            import torch

            if not self._is_loaded:
                if not await self._load_pipeline():
                    return None

            self._in_use += 1
            self._last_used = time.time()

            try:
                duration_seconds = min(duration_seconds, self.max_duration)

                if seed == -1:
                    import random
                    seed = random.randint(0, 2**32 - 1)

                generator = torch.Generator(device=self.device).manual_seed(seed)

                logger.info(
                    f"[STABLE-AUDIO] Generating: prompt='{prompt[:80]}...', "
                    f"duration={duration_seconds}s, steps={steps}, cfg={cfg_scale}, seed={seed}"
                )

                def _generate():
                    with torch.no_grad():
                        result = self._pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            audio_end_in_s=duration_seconds,
                            num_inference_steps=steps,
                            guidance_scale=cfg_scale,
                            generator=generator,
                        )
                    return result.audios[0]  # [channels, samples]

                audio = await asyncio.to_thread(_generate)
                return self._encode_audio(audio, output_format, seed)

            finally:
                self._in_use -= 1

        except Exception as e:
            logger.error(f"[STABLE-AUDIO] Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def generate_from_embeddings(
        self,
        prompt_embeds,  # torch.Tensor [B, seq, 768]
        attention_mask=None,  # torch.Tensor [B, seq] or None
        seconds_start: float = 0.0,
        seconds_end: float = 10.0,
        negative_prompt: str = "",
        steps: int = 100,
        cfg_scale: float = 7.0,
        seed: int = -1,
    ) -> Optional[bytes]:
        """
        Generate audio from pre-computed embeddings (for cross-aesthetic use).

        The pipeline accepts prompt_embeds to bypass T5 text encoding.
        This enables CLIP image features (768d) to be injected directly
        as conditioning — the dimensional match with T5-Base is exact.

        CRITICAL: When using pre-computed embeddings with CFG (guidance_scale > 1),
        we must also provide negative_prompt_embeds as zeros. Otherwise the pipeline
        encodes negative_prompt text via T5, and the CFG computation becomes:
            output = T5("") + cfg * (CLIP_features - T5(""))
        Since CLIP and T5 are different feature spaces, the subtraction is meaningless
        and produces constant noise regardless of the input image.
        With zeros: output = cfg * CLIP_features (correct).

        Args:
            prompt_embeds: Pre-computed conditioning tensor [B, seq, 768]
            attention_mask: Attention mask for embeddings [B, seq]
            seconds_start: Audio start time
            seconds_end: Audio end time (max 47.55)
            negative_prompt: Ignored when prompt_embeds is provided (kept for API compat)
            steps: Number of inference steps
            cfg_scale: Classifier-free guidance scale
            seed: Seed for reproducibility (-1 = random)

        Returns:
            Audio bytes (WAV) or None on failure
        """
        try:
            import torch

            if not self._is_loaded:
                if not await self._load_pipeline():
                    return None

            self._in_use += 1
            self._last_used = time.time()

            try:
                seconds_end = min(seconds_end, self.max_duration)

                if seed == -1:
                    import random
                    seed = random.randint(0, 2**32 - 1)

                generator = torch.Generator(device=self.device).manual_seed(seed)

                # Move embeddings to device
                xf_dtype = self._pipeline.transformer.dtype
                prompt_embeds = prompt_embeds.to(device=self.device, dtype=xf_dtype)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=self.device)

                # Zero negative embeddings: keeps CFG in the same feature space
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_attention_mask = torch.ones_like(attention_mask)

                logger.info(
                    f"[STABLE-AUDIO] Generating from embeddings: "
                    f"shape={list(prompt_embeds.shape)}, "
                    f"duration={seconds_end - seconds_start}s, steps={steps}, seed={seed}"
                )

                def _generate():
                    with torch.no_grad():
                        result = self._pipeline(
                            prompt_embeds=prompt_embeds,
                            attention_mask=attention_mask,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_attention_mask=negative_attention_mask,
                            audio_start_in_s=seconds_start,
                            audio_end_in_s=seconds_end,
                            num_inference_steps=steps,
                            guidance_scale=cfg_scale,
                            generator=generator,
                        )
                    return result.audios[0]

                audio = await asyncio.to_thread(_generate)
                return self._encode_audio(audio, "wav", seed)

            finally:
                self._in_use -= 1

        except Exception as e:
            logger.error(f"[STABLE-AUDIO] Embedding generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def encode_prompt(self, text: str):
        """
        Encode text via Stable Audio's T5 tokenizer + T5EncoderModel.

        Exposes the T5 conditioning step so callers can manipulate
        the embedding before passing it to generate_from_embeddings().

        Args:
            text: Text prompt to encode

        Returns:
            Tuple of (prompt_embeds, attention_mask) or (None, None) on failure.
            prompt_embeds: Tensor [1, seq_len, 768]
            attention_mask: Tensor [1, seq_len]
        """
        try:
            import torch

            if not self._is_loaded:
                if not await self._load_pipeline():
                    return None, None

            self._last_used = time.time()

            pipe = self._pipeline
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder

            def _encode():
                with torch.no_grad():
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                    )
                    input_ids = inputs.input_ids.to(text_encoder.device)
                    attention_mask = inputs.attention_mask.to(text_encoder.device)

                    encoder_output = text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    # T5 hidden states: [1, seq_len, 768]
                    hidden_states = encoder_output.last_hidden_state

                    # Apply the projection model (StableAudioProjectionModel)
                    # In Stable Audio, text_projection is nn.Identity() (768→768)
                    projection = pipe.projection_model.text_projection
                    projected = projection(hidden_states)

                    return projected, attention_mask

            prompt_embeds, attention_mask = await asyncio.to_thread(_encode)
            logger.info(
                f"[STABLE-AUDIO] Encoded prompt: shape={list(prompt_embeds.shape)}, "
                f"text='{text[:60]}...'"
            )
            return prompt_embeds, attention_mask

        except Exception as e:
            logger.error(f"[STABLE-AUDIO] Prompt encoding error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    async def generate_audio_with_guidance(
        self,
        image_bytes: bytes,
        prompt: str = "",
        lambda_guidance: float = 0.1,
        warmup_steps: int = 10,
        total_steps: int = 50,
        duration_seconds: float = 10.0,
        cfg_scale: float = 7.0,
        seed: int = -1,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate audio with ImageBind gradient guidance.

        Implements the "Seeing and Hearing" (CVPR 2024) approach:
        During the first `warmup_steps` of the denoising process,
        the predicted clean latent is decoded, encoded via ImageBind's
        audio encoder, and a cosine similarity loss against the image
        embedding provides a gradient signal to steer the latents.

        Args:
            image_bytes: Input image bytes
            prompt: Optional text basis conditioning
            lambda_guidance: Gradient guidance strength (default 0.1)
            warmup_steps: Number of guidance steps (default 10)
            total_steps: Total inference steps (default 50)
            duration_seconds: Audio duration
            cfg_scale: CFG scale
            seed: Random seed

        Returns:
            Dict with audio_bytes, cosine_similarity, seed, generation_time_ms
        """
        import time as time_mod

        try:
            import torch
            import torch.nn.functional as F

            if not self._is_loaded:
                if not await self._load_pipeline():
                    return None

            self._in_use += 1
            self._last_used = time.time()
            start_time = time_mod.time()

            try:
                duration_seconds = min(duration_seconds, self.max_duration)

                if seed == -1:
                    import random
                    seed = random.randint(0, 2**32 - 1)

                # Get ImageBind backend for guidance
                from services.imagebind_backend import get_imagebind_backend
                ib = get_imagebind_backend()

                # Pre-compute image embedding (this stays fixed)
                ib_image_emb = await ib.encode_image(image_bytes)
                if ib_image_emb is None:
                    logger.error("[STABLE-AUDIO] ImageBind image encoding failed")
                    return None

                # Run the guided denoising loop
                # This mirrors StableAudioPipeline.__call__ exactly, but with
                # gradient guidance injected during warmup steps.
                def _guided_generate():
                    from diffusers.models.embeddings import get_1d_rotary_pos_embed

                    pipe = self._pipeline
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    do_cfg = cfg_scale > 1.0
                    text_input = prompt if prompt else ""

                    # --- Encode prompt (mirrors pipeline step 3) ---
                    # With do_cfg=True and no negative_prompt, encode_prompt returns
                    # just the positive prompt_embeds (no concatenation).
                    # The pipeline handles the uncond case separately below.
                    prompt_embeds = pipe.encode_prompt(
                        prompt=text_input,
                        device=self.device,
                        do_classifier_free_guidance=do_cfg,
                    )

                    # --- Encode duration (mirrors pipeline's encode_duration) ---
                    seconds_start_hidden_states, seconds_end_hidden_states = pipe.encode_duration(
                        audio_start_in_s=0.0,
                        audio_end_in_s=duration_seconds,
                        device=self.device,
                        do_classifier_free_guidance=False,  # don't double — we handle CFG below
                        batch_size=1,
                    )

                    # --- Build text_audio_duration_embeds and audio_duration_embeds ---
                    # text_audio_duration_embeds: [B, seq+2, dim] — cross-attention input
                    # audio_duration_embeds: [B, 1, 2*dim] — global conditioning input
                    text_audio_duration_embeds = torch.cat(
                        [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
                    )
                    audio_duration_embeds = torch.cat(
                        [seconds_start_hidden_states, seconds_end_hidden_states], dim=2
                    )

                    # CFG without negative prompt: uncond = zeros (mirrors pipeline logic)
                    if do_cfg:
                        negative_text_audio_duration_embeds = torch.zeros_like(text_audio_duration_embeds)
                        # Pipeline order: [uncond, cond]
                        text_audio_duration_embeds = torch.cat(
                            [negative_text_audio_duration_embeds, text_audio_duration_embeds], dim=0
                        )
                        audio_duration_embeds = torch.cat(
                            [audio_duration_embeds, audio_duration_embeds], dim=0
                        )

                    # --- Prepare latents (mirrors pipeline step 5) ---
                    waveform_length = int(pipe.transformer.config.sample_size)
                    latent_channels = pipe.transformer.config.in_channels
                    latents = torch.randn(
                        1, latent_channels, waveform_length,
                        generator=generator, device=self.device,
                        dtype=text_audio_duration_embeds.dtype,
                    )
                    # Bug 1 fix: scale initial noise by init_noise_sigma
                    # (sigma_max^2 + 1)^0.5 ≈ 80.006 for EDM scheduler
                    latents = latents * pipe.scheduler.init_noise_sigma

                    # --- Scheduler (mirrors pipeline step 4) ---
                    pipe.scheduler.set_timesteps(total_steps, device=self.device)
                    timesteps = pipe.scheduler.timesteps

                    # --- Rotary embedding (mirrors pipeline step 7) ---
                    rotary_embedding = get_1d_rotary_pos_embed(
                        pipe.rotary_embed_dim,
                        latents.shape[2] + audio_duration_embeds.shape[1],
                        use_real=True,
                        repeat_interleave_real=False,
                    )

                    final_cosine_sim = None

                    # --- Denoising loop (mirrors pipeline step 8) ---
                    # Correct algorithm (Xing et al. CVPR 2024 + EDM math):
                    # - Guidance modifies latents BEFORE scheduler.step()
                    # - scheduler.step() ALWAYS runs to keep state consistent
                    # - precondition_outputs() computes predicted clean sample (not DDPM formula)
                    for i, t in enumerate(timesteps):
                        is_warmup = i < warmup_steps

                        if is_warmup:
                            latents = latents.detach().requires_grad_(True)

                        # Expand for CFG
                        latent_input = torch.cat([latents] * 2) if do_cfg else latents
                        latent_input = pipe.scheduler.scale_model_input(latent_input, t)

                        # Transformer forward — with grad during warmup for EDM gradient flow
                        # At high sigma (early steps): c_skip ≈ 0, c_out ≈ 0.5
                        # → predicted_clean ≈ 0.5 * noise_pred — gradient flows through transformer
                        if is_warmup:
                            noise_pred = pipe.transformer(
                                latent_input,
                                t.unsqueeze(0),
                                encoder_hidden_states=text_audio_duration_embeds,
                                global_hidden_states=audio_duration_embeds,
                                rotary_embedding=rotary_embedding,
                                return_dict=False,
                            )[0]
                        else:
                            with torch.no_grad():
                                noise_pred = pipe.transformer(
                                    latent_input,
                                    t.unsqueeze(0),
                                    encoder_hidden_states=text_audio_duration_embeds,
                                    global_hidden_states=audio_duration_embeds,
                                    rotary_embedding=rotary_embedding,
                                    return_dict=False,
                                )[0]

                        # CFG — pipeline order is [uncond, cond]
                        if do_cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

                        # --- Gradient guidance (warmup only) ---
                        if is_warmup:
                            # Bug 2 fix: Use EDM preconditioning to compute predicted clean sample
                            # c_skip = σ_data² / (σ² + σ_data²), c_out = σ·σ_data / (σ² + σ_data²)^0.5
                            # predicted_clean = c_skip * sample + c_out * model_output
                            sigma = pipe.scheduler.sigmas[pipe.scheduler.step_index]
                            predicted_clean = pipe.scheduler.precondition_outputs(
                                latents, noise_pred, sigma
                            )

                            # Decode to waveform for ImageBind
                            audio_waveform = pipe.vae.decode(
                                predicted_clean.to(pipe.vae.dtype)
                            ).sample  # [1, channels, audio_samples]

                            # Mono, first 2 seconds at 16kHz for ImageBind
                            mono = audio_waveform.mean(dim=1)  # [1, samples]

                            # Encode audio via ImageBind (direct tensor path, no file I/O)
                            ib_audio_emb = None
                            try:
                                import torchaudio
                                from torchvision import transforms as tv_transforms
                                from imagebind.data import waveform2melspec, get_clip_timepoints
                                from imagebind.models.imagebind_model import ModalityType
                                from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

                                resampler = torchaudio.transforms.Resample(
                                    self.sample_rate, 16000
                                ).to(mono.device)
                                mono_16k = resampler(mono.float())

                                # Take first 2s (ImageBind clip duration)
                                max_samples = 2 * 16000
                                mono_16k = mono_16k[:, :max_samples]

                                # Build mel spectrograms directly (bypasses torchaudio.load/torchcodec)
                                clip_sampler = ConstantClipsPerVideoSampler(
                                    clip_duration=2, clips_per_video=3
                                )
                                duration_s = mono_16k.size(1) / 16000
                                clip_timepoints = get_clip_timepoints(clip_sampler, duration_s)

                                all_clips = []
                                for start, end in clip_timepoints:
                                    clip = mono_16k[:, int(start * 16000):int(end * 16000)]
                                    # waveform2melspec expects [channels, samples], returns [1, 128, 204]
                                    melspec = waveform2melspec(clip, 16000, 128, 204)
                                    all_clips.append(melspec)

                                normalize = tv_transforms.Normalize(mean=-4.268, std=9.138)
                                all_clips = [normalize(ac).to(self.device) for ac in all_clips]
                                # Stack: [clips=3, 1, 128, 204] → unsqueeze → [1, 3, 1, 128, 204]
                                mel_tensor = torch.stack(all_clips, dim=0).unsqueeze(0)

                                from services.imagebind_backend import get_imagebind_backend
                                ib_model = get_imagebind_backend()._model
                                inputs = {ModalityType.AUDIO: mel_tensor}
                                ib_audio_emb = ib_model(inputs)[ModalityType.AUDIO]

                            except Exception as e:
                                logger.warning(f"[STABLE-AUDIO] ImageBind audio encoding failed at step {i}: {e}")

                            if ib_audio_emb is not None:
                                cos_sim = F.cosine_similarity(
                                    ib_audio_emb, ib_image_emb.detach()
                                )
                                loss = 1.0 - cos_sim.mean()

                                grad = torch.autograd.grad(
                                    loss, latents, retain_graph=False
                                )[0]

                                # Apply guidance: modify latents BEFORE scheduler.step()
                                latents = (latents - lambda_guidance * grad).detach()
                                final_cosine_sim = cos_sim.mean().item()

                                logger.debug(
                                    f"[STABLE-AUDIO] Guidance step {i}: "
                                    f"cos_sim={cos_sim.mean().item():.4f}, "
                                    f"grad_norm={grad.norm().item():.4f}"
                                )
                            else:
                                latents = latents.detach()

                        # Bug 3+4 fix: ALWAYS call scheduler.step() to keep state consistent
                        # This increments step_index, updates model_outputs buffer,
                        # and uses the correct DPMSolver++ multi-step integration
                        noise_pred_detached = noise_pred.detach()
                        latents_detached = latents.detach()
                        with torch.no_grad():
                            latents = pipe.scheduler.step(
                                noise_pred_detached, t, latents_detached
                            ).prev_sample

                    # Final decode
                    with torch.no_grad():
                        audio = pipe.vae.decode(
                            latents.to(pipe.vae.dtype)
                        ).sample

                    # Trim to requested duration (mirrors pipeline step 9)
                    waveform_start = 0
                    waveform_end = int(duration_seconds * self.sample_rate)
                    audio = audio[:, :, waveform_start:waveform_end]

                    return audio.squeeze(0), final_cosine_sim  # [channels, samples], float

                audio, cosine_sim = await asyncio.to_thread(_guided_generate)

                audio_bytes = self._encode_audio(audio, "wav", seed)
                if audio_bytes is None:
                    return None

                elapsed_ms = int((time_mod.time() - start_time) * 1000)

                logger.info(
                    f"[STABLE-AUDIO] Guided generation complete: "
                    f"cos_sim={cosine_sim}, time={elapsed_ms}ms, seed={seed}"
                )

                return {
                    "audio_bytes": audio_bytes,
                    "cosine_similarity": cosine_sim,
                    "seed": seed,
                    "generation_time_ms": elapsed_ms,
                }

            finally:
                self._in_use -= 1

        except Exception as e:
            logger.error(f"[STABLE-AUDIO] Guided generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_vae(self):
        """Get the VAE (AutoencoderOobleck) for cross-aesthetic latent decoding."""
        if not self._is_loaded or self._pipeline is None:
            return None
        return self._pipeline.vae

    def _encode_audio(self, audio, output_format: str, seed: int) -> Optional[bytes]:
        """Encode audio tensor to bytes."""
        import io
        import numpy as np

        try:
            # audio shape: [channels, samples] as numpy or tensor
            # Pipeline outputs float16 when torch_dtype=float16 — soundfile needs float32
            if hasattr(audio, 'numpy'):
                audio_np = audio.cpu().float().numpy()
            elif hasattr(audio, 'cpu'):
                audio_np = audio.cpu().float().numpy()
            else:
                audio_np = np.array(audio)

            # Transpose for soundfile: [samples, channels]
            if audio_np.ndim == 2:
                audio_np = audio_np.T

            if output_format == "mp3":
                return self._encode_mp3(audio_np)
            else:
                return self._encode_wav(audio_np)

        except Exception as e:
            logger.error(f"[STABLE-AUDIO] Audio encoding error: {e}")
            return None

    def _encode_wav(self, audio_np) -> bytes:
        import io
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio_np, self.sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.getvalue()

    def _encode_mp3(self, audio_np) -> bytes:
        import io
        import tempfile
        import os

        # Write WAV first, then convert (pydub or direct ffmpeg)
        try:
            from pydub import AudioSegment
            wav_bytes = self._encode_wav(audio_np)
            audio_segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="192k")
            mp3_buffer.seek(0)
            return mp3_buffer.getvalue()
        except ImportError:
            logger.warning("[STABLE-AUDIO] pydub not available, falling back to WAV")
            return self._encode_wav(audio_np)


# =============================================================================
# Singleton
# =============================================================================

_backend: Optional[StableAudioGenerator] = None


def get_stable_audio_backend() -> StableAudioGenerator:
    global _backend
    if _backend is None:
        _backend = StableAudioGenerator()
    return _backend


def reset_stable_audio_backend():
    global _backend
    _backend = None
