"""
Output-Config Selector: Select default Output-Config based on media type

Architecture Principle: Separation of Concerns
- Pre-pipeline configs (dada.json) suggest media type via media_preferences.default_output
- Pre-pipeline configs DO NOT choose specific models
- This module provides centralized default mapping: media_type → output_config
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MediaOutput:
    """Structured tracking of generated media"""
    media_type: str  # "image", "audio", "music", "video"
    prompt_id: str  # ComfyUI queue ID or API reference
    output_mapping: Dict[str, Any]  # How to extract media (from Output-Chunk)
    config_name: str  # Which output config was used
    status: str  # "queued", "generating", "completed", "failed"
    metadata: Optional[Dict[str, Any]] = None  # Additional info


@dataclass
class ExecutionContext:
    """Track expected and actual media throughout execution"""
    config_name: str
    expected_media_type: str  # From pre-pipeline config.media_preferences.default_output
    generated_media: list  # List[MediaOutput]
    text_outputs: list  # List[str] - track text at each pipeline step

    def add_media(self, media: MediaOutput):
        """Add generated media to context"""
        self.generated_media.append(media)
        logger.info(f"[EXECUTION-CONTEXT] Added {media.media_type} media: {media.prompt_id} (status: {media.status})")

    def add_text_output(self, text: str):
        """Add text output from pipeline step"""
        self.text_outputs.append(text)
        logger.debug(f"[EXECUTION-CONTEXT] Added text output: {text[:100]}...")

    def get_latest_media(self) -> Optional[MediaOutput]:
        """Get most recently generated media"""
        return self.generated_media[-1] if self.generated_media else None

    def get_latest_text(self) -> str:
        """Get most recent text output"""
        return self.text_outputs[-1] if self.text_outputs else ""


class OutputConfigSelector:
    """Select default Output-Config based on media type"""

    def __init__(self, schemas_path: Path):
        self.schemas_path = schemas_path
        self.defaults: Dict[str, Any] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load output_config_defaults.json"""
        defaults_path = self.schemas_path / "output_config_defaults.json"

        if not defaults_path.exists():
            logger.error(f"output_config_defaults.json not found at {defaults_path}")
            return

        try:
            with open(defaults_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Filter out metadata fields (start with _)
            self.defaults = {
                media_type: modes
                for media_type, modes in data.items()
                if not media_type.startswith('_')
            }

            logger.info(f"Loaded output_config_defaults: {len(self.defaults)} media types")
            for media_type, value in self.defaults.items():
                logger.debug(f"  {media_type}: {value}")

        except Exception as e:
            logger.error(f"Error loading output_config_defaults.json: {e}")
            self.defaults = {}

    def select_output_config(self, media_type: str) -> Optional[str]:
        """
        Select default Output-Config for given media type.

        Supports two value formats:
        - String: direct config name
        - Dict: VRAM-tier mapping (e.g. {"vram_96": "...", "vram_32": "..."})

        Args:
            media_type: "image", "audio", "music", "video", "text"

        Returns:
            Output-Config name (e.g., "sd35_large") or None if not available
        """
        if media_type not in self.defaults:
            logger.warning(f"Unknown media type: {media_type}")
            return None

        output_config = self.defaults[media_type]

        # Dict = VRAM-tier mapping
        if isinstance(output_config, dict):
            output_config = self._resolve_vram_tier_dict(output_config)

        if output_config:
            logger.info(f"[OUTPUT-CONFIG-SELECTOR] {media_type} → {output_config}")
        else:
            logger.warning(f"[OUTPUT-CONFIG-SELECTOR] No default for {media_type}")

        return output_config

    def _resolve_vram_tier_dict(self, tier_dict: dict) -> Optional[str]:
        """Resolve a VRAM-tier dict to a config name.

        Matches the detected VRAM tier, falling through to lower tiers.
        """
        vram_tier = self._get_vram_tier()
        tier_order = ["vram_96", "vram_48", "vram_32", "vram_24", "vram_16", "vram_8"]

        try:
            start_idx = tier_order.index(vram_tier)
        except ValueError:
            start_idx = len(tier_order) - 1

        for tier in tier_order[start_idx:]:
            if tier in tier_dict:
                config_name = tier_dict[tier]
                logger.info(f"[OUTPUT-CONFIG-SELECTOR] VRAM {vram_tier} → matched {tier} → {config_name}")
                return config_name

        return None

    def _get_vram_tier(self) -> str:
        """Get VRAM tier, cached after first detection."""
        if not hasattr(self, '_vram_tier_cache'):
            try:
                from my_app.routes.settings_routes import detect_gpu_vram
                result = detect_gpu_vram()
                self._vram_tier_cache = result.get("vram_tier", "vram_8")
            except Exception:
                self._vram_tier_cache = "vram_8"
        return self._vram_tier_cache

    def get_available_media_types(self) -> list:
        """Get list of supported media types"""
        return list(self.defaults.keys())

    def is_media_type_supported(self, media_type: str) -> bool:
        """Check if media type is supported"""
        if media_type not in self.defaults:
            return False

        output_config = self.defaults[media_type]
        if isinstance(output_config, dict):
            output_config = self._resolve_vram_tier_dict(output_config)
        return output_config is not None


# Singleton instance
_selector_instance = None


def get_output_config_selector(schemas_path: Path = None) -> OutputConfigSelector:
    """Get singleton OutputConfigSelector instance"""
    global _selector_instance

    if _selector_instance is None:
        if schemas_path is None:
            # Default to schemas/ relative to this file
            schemas_path = Path(__file__).parent.parent
        _selector_instance = OutputConfigSelector(schemas_path)

    return _selector_instance
