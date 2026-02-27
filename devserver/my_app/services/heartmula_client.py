"""
HeartMuLa HTTP Client — Drop-in replacement for HeartMuLaMusicGenerator

Calls the shared GPU service (port 17803) via HTTP REST instead of
loading models in-process. All method signatures and return types
are identical to HeartMuLaMusicGenerator.
"""

import base64
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class HeartMuLaClient:
    """HTTP client for HeartMuLa on the shared GPU service.

    Drop-in replacement for HeartMuLaMusicGenerator.
    """

    def __init__(self):
        from config import GPU_SERVICE_URL, GPU_SERVICE_TIMEOUT_MUSIC
        self.base_url = GPU_SERVICE_URL.rstrip('/')
        self.timeout = GPU_SERVICE_TIMEOUT_MUSIC
        logger.info(f"[HEARTMULA-CLIENT] Initialized: url={self.base_url}, timeout={self.timeout}s")

    def _post(self, path: str, data: dict) -> Optional[dict]:
        """Synchronous POST to GPU service."""
        import requests
        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[HEARTMULA-CLIENT] Request failed: {e}")
            return None

    def _get(self, path: str) -> Optional[dict]:
        """Synchronous GET to GPU service."""
        import requests
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[HEARTMULA-CLIENT] GET failed: {e}")
            return None

    async def is_available(self) -> bool:
        """Check if GPU service is reachable and HeartMuLa is available."""
        import asyncio
        try:
            result = await asyncio.to_thread(self._get, '/api/heartmula/available')
            return result is not None and result.get('available', False)
        except Exception:
            return False

    async def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU memory information from the GPU service."""
        import asyncio
        result = await asyncio.to_thread(self._get, '/api/health')
        if result and 'gpu' in result:
            return result['gpu']
        return result or {"error": "GPU service unreachable"}

    async def unload_pipeline(self) -> bool:
        """Unload HeartMuLa pipeline from GPU service."""
        import asyncio
        result = await asyncio.to_thread(self._post, '/api/heartmula/unload', {})
        return result is not None and result.get('success', False)

    async def generate_music(
        self,
        lyrics: str,
        tags: str,
        temperature: float = 1.0,
        topk: int = 70,
        cfg_scale: float = 3.0,
        max_audio_length_ms: int = 240000,
        seed: Optional[int] = None,
        output_format: str = "mp3"
    ) -> Optional[bytes]:
        """Generate music. Returns audio bytes or None."""
        import asyncio
        result = await asyncio.to_thread(self._post, '/api/heartmula/generate', {
            'lyrics': lyrics,
            'tags': tags,
            'temperature': temperature,
            'topk': topk,
            'cfg_scale': cfg_scale,
            'max_audio_length_ms': max_audio_length_ms,
            'seed': seed,
            'output_format': output_format,
        })

        if result is None or not result.get('success'):
            return None

        return base64.b64decode(result['audio_base64'])
