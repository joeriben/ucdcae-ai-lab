"""
VRAM Monitor — consolidated view of GPU memory usage across backends.

DevServer orchestrates both Ollama (LLM) and GPU Service (media generation).
This module provides a unified view of VRAM usage for both, enabling
future VRAM budget management (Phase 4B).
"""

import logging
from typing import Dict, Any, List, Optional

import requests

logger = logging.getLogger(__name__)


class VRAMMonitor:
    """Monitors VRAM usage across Ollama and GPU Service."""

    def __init__(self):
        from config import OLLAMA_API_BASE_URL, GPU_SERVICE_URL
        self.ollama_url = OLLAMA_API_BASE_URL.rstrip('/')
        self.gpu_url = GPU_SERVICE_URL.rstrip('/')

    def get_ollama_models(self) -> List[Dict[str, Any]]:
        """GET /api/ps → list of loaded Ollama models with VRAM usage."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                vram_bytes = m.get("size_vram", 0)
                models.append({
                    "name": m.get("name", "unknown"),
                    "vram_mb": round(vram_bytes / (1024 * 1024), 1) if vram_bytes else 0,
                    "size_mb": round(m.get("size", 0) / (1024 * 1024), 1),
                    "expires_at": m.get("expires_at"),
                })
            return models
        except Exception as e:
            logger.debug(f"[VRAM-MONITOR] Ollama /api/ps failed: {e}")
            return []

    def get_gpu_service_status(self) -> Dict[str, Any]:
        """GET /api/health → GPU service health and loaded models."""
        try:
            resp = requests.get(f"{self.gpu_url}/api/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"[VRAM-MONITOR] GPU service /api/health failed: {e}")
            return {"reachable": False, "error": str(e)}

    def get_combined_status(self) -> Dict[str, Any]:
        """Consolidated VRAM view across both backends."""
        ollama_models = self.get_ollama_models()
        gpu_status = self.get_gpu_service_status()

        ollama_total_vram = sum(m["vram_mb"] for m in ollama_models)

        return {
            "ollama": {
                "reachable": len(ollama_models) > 0 or self._ollama_reachable(),
                "models": ollama_models,
                "total_vram_mb": round(ollama_total_vram, 1),
            },
            "gpu_service": {
                "reachable": gpu_status.get("status") == "healthy" or gpu_status.get("reachable", False) is not False,
                "raw": gpu_status,
            },
        }

    def _ollama_reachable(self) -> bool:
        """Quick reachability check for Ollama (no loaded models doesn't mean unreachable)."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False


# Module-level singleton
_monitor: Optional[VRAMMonitor] = None


def get_vram_monitor() -> VRAMMonitor:
    """Get VRAMMonitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = VRAMMonitor()
    return _monitor
