"""
Stable Audio HTTP Client — GPU Service Backend

Calls the shared GPU service (port 17803) for local Stable Audio Open generation.
Replaces the legacy cloud API client with local inference via StableAudioPipeline.
"""

import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StableAudioClient:
    """HTTP client for Stable Audio on the local GPU service."""

    def __init__(self):
        from config import GPU_SERVICE_URL, GPU_SERVICE_TIMEOUT_AUDIO
        self.base_url = GPU_SERVICE_URL.rstrip('/')
        self.timeout = GPU_SERVICE_TIMEOUT_AUDIO

    def _post(self, path: str, data: dict) -> Optional[dict]:
        import requests
        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            logger.error(f"[STABLE-AUDIO-CLIENT] GPU service unreachable at {url}")
            return None
        except requests.Timeout:
            logger.error(f"[STABLE-AUDIO-CLIENT] Timeout after {self.timeout}s: {path}")
            return None
        except Exception as e:
            logger.error(f"[STABLE-AUDIO-CLIENT] Request failed: {e}")
            return None

    def _get(self, path: str) -> Optional[dict]:
        import requests
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[STABLE-AUDIO-CLIENT] GET failed: {e}")
            return None

    async def is_available(self) -> bool:
        """Check if GPU service has Stable Audio available."""
        import asyncio
        try:
            result = await asyncio.to_thread(self._get, '/api/stable_audio/available')
            return result is not None and result.get('available', False)
        except Exception:
            return False

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
        Generate audio from text prompt via local GPU service.

        Returns:
            Audio bytes or None on failure
        """
        import asyncio

        result = await asyncio.to_thread(self._post, '/api/stable_audio/generate', {
            'prompt': prompt,
            'duration_seconds': duration_seconds,
            'negative_prompt': negative_prompt,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'seed': seed,
            'output_format': output_format,
        })

        if result is None or not result.get('success'):
            logger.error("[STABLE-AUDIO-CLIENT] Generation failed")
            return None

        return base64.b64decode(result['audio_base64'])


# Singleton
_client: Optional[StableAudioClient] = None


def get_stable_audio_client() -> StableAudioClient:
    global _client
    if _client is None:
        _client = StableAudioClient()
    return _client
