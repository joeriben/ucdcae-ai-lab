"""
VLM Safety Check — Image safety via local VLM.

Extracted from schema_pipeline_routes.py to be reusable for both
post-generation checks and input image upload checks.

Uses LLMClient (GPU Service primary, Ollama fallback).
"""

import base64
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

VLM_PROMPTS = {
    'kids': (
        'Is the image free of violence, nudity, unsettling, scaring, '
        'potentially traumatizing, or other content unsafe or unhealthy '
        'for children age 6-12 years? Provide a clear decision: "safe" or "unsafe".'
    ),
    'youth': (
        'Is the image free of violence, nudity, unsettling, scaring, '
        'potentially traumatizing, or other content unsafe or unhealthy '
        'for teenagers age 14-18 years? Provide a clear decision: "safe" or "unsafe".'
    ),
}


def vlm_safety_check(image_path: str | Path, safety_level: str) -> tuple[bool, str, str]:
    """
    Check image safety via qwen3-vl. Returns (is_safe, reason, description). Fail-open.

    Args:
        image_path: Path to the image file on disk.
        safety_level: 'kids' or 'youth' (only these trigger VLM check).

    Returns:
        (is_safe, reason, description) — description is the VLM's image analysis.
        (True, '', '') on safe or error (fail-open).
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning("[VLM-SAFETY] Image file not found — skipping check")
            return (True, '', '')

        prompt_text = VLM_PROMPTS.get(safety_level)
        if not prompt_text:
            return (True, '', '')

        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        logger.info(f"[VLM-SAFETY] Checking image ({len(image_bytes)} bytes) with {config.VLM_SAFETY_MODEL} for safety_level={safety_level}")

        from my_app.services.llm_backend import get_llm_backend
        result = get_llm_backend().chat(
            model=config.VLM_SAFETY_MODEL,
            messages=[{'role': 'user', 'content': prompt_text}],
            images=[image_b64],
            temperature=0.0,
            max_new_tokens=2000,
        )

        if result is None:
            logger.warning("[VLM-SAFETY] LLM returned None (fail-open)")
            return (True, '', '')

        # qwen3 uses thinking mode: answer may be in 'content' or 'thinking'
        content = result.get('content', '').lower().strip()
        thinking = (result.get('thinking') or '').lower().strip()
        combined = content or thinking
        logger.info(f"[VLM-SAFETY] Model response: content={content!r}, thinking={thinking!r}")

        # Use thinking as image description (it contains the VLM's analysis)
        description = (result.get('thinking') or '').strip()

        if 'unsafe' in combined:
            return (False, f"VLM safety check ({config.VLM_SAFETY_MODEL}): image flagged as unsafe for {safety_level}", description)
        return (True, '', description)

    except Exception as e:
        # Fail-open: VLM failure should never block
        logger.warning(f"[VLM-SAFETY] Error during check (fail-open): {e}")
        return (True, '', '')
