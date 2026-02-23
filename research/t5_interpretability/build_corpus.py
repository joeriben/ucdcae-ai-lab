#!/usr/bin/env python3
"""
Phase 1a: Bulk Corpus Assembly

Downloads AudioCaps (~46K), MusicCaps (~5.5K), WavCaps (~45K),
deduplicates, and saves to corpus.json.

Usage: venv/bin/python research/t5_interpretability/build_corpus.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Ensure config is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    AUDIOCAPS_DATASET, MUSICCAPS_DATASET,
    WAVCAPS_REPO, WAVCAPS_JSON_FILES, WAVCAPS_FALLBACK_DATASET,
    DATA_DIR, CORPUS_PATH,
)


def download_audiocaps() -> list[dict]:
    """Download AudioCaps: environmental sounds, voices, music, machines."""
    from datasets import load_dataset

    logger.info("Downloading AudioCaps...")
    entries = []
    ds = load_dataset(AUDIOCAPS_DATASET)

    for split_name in ds:
        for row in ds[split_name]:
            caption = row.get("caption", "").strip()
            if caption:
                entries.append({
                    "text": caption,
                    "source": "audiocaps",
                    "category": "bulk",
                    "subcategory": "environmental",
                })

    logger.info(f"AudioCaps: {len(entries)} captions from {len(ds)} splits")
    return entries


def download_musiccaps() -> list[dict]:
    """Download MusicCaps: music descriptions with aspect labels."""
    from datasets import load_dataset

    logger.info("Downloading MusicCaps...")
    entries = []
    ds = load_dataset(MUSICCAPS_DATASET)

    for split_name in ds:
        for row in ds[split_name]:
            caption = row.get("caption", "").strip()
            if caption:
                entries.append({
                    "text": caption,
                    "source": "musiccaps",
                    "category": "bulk",
                    "subcategory": "music",
                })

    logger.info(f"MusicCaps: {len(entries)} captions")
    return entries


def download_wavcaps() -> list[dict]:
    """Download WavCaps: AudioSet + BBC + FreeSound captions via JSON files."""
    from huggingface_hub import hf_hub_download

    logger.info("Downloading WavCaps JSON files...")
    entries = []

    try:
        for json_file in WAVCAPS_JSON_FILES:
            logger.info(f"  Fetching {json_file}...")
            local_path = hf_hub_download(
                repo_id=WAVCAPS_REPO,
                filename=json_file,
                repo_type="dataset",
            )
            with open(local_path, "r") as f:
                data = json.load(f)

            # WavCaps JSON structure: {"data": [{"caption": "..."}, ...]}
            items = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = items.get("data", [])

            for item in items:
                caption = ""
                if isinstance(item, dict):
                    caption = item.get("caption", item.get("text", "")).strip()
                elif isinstance(item, str):
                    caption = item.strip()

                if caption:
                    entries.append({
                        "text": caption,
                        "source": "wavcaps",
                        "category": "bulk",
                        "subcategory": "mixed",
                    })

        logger.info(f"WavCaps: {len(entries)} captions from {len(WAVCAPS_JSON_FILES)} files")

    except Exception as e:
        logger.warning(f"WavCaps primary download failed: {e}")
        logger.info("Trying fallback dataset (AudioSetCaps)...")
        entries = _download_wavcaps_fallback()

    return entries


def _download_wavcaps_fallback() -> list[dict]:
    """Fallback: download AudioSetCaps if WavCaps JSON not accessible."""
    from datasets import load_dataset

    ds = load_dataset(WAVCAPS_FALLBACK_DATASET)
    entries = []
    for split_name in ds:
        for row in ds[split_name]:
            caption = row.get("caption", row.get("text", "")).strip()
            if caption:
                entries.append({
                    "text": caption,
                    "source": "wavcaps_fallback",
                    "category": "bulk",
                    "subcategory": "mixed",
                })

    logger.info(f"WavCaps fallback: {len(entries)} captions")
    return entries


def deduplicate(entries: list[dict], min_length: int = 10) -> list[dict]:
    """Deduplicate by normalized text, remove short entries."""
    seen = set()
    result = []
    for entry in entries:
        key = entry["text"].strip().lower()
        if len(key) >= min_length and key not in seen:
            seen.add(key)
            result.append(entry)

    removed = len(entries) - len(result)
    logger.info(f"Deduplication: {len(entries)} → {len(result)} ({removed} removed)")
    return result


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download all three sources
    audiocaps = download_audiocaps()
    musiccaps = download_musiccaps()
    wavcaps = download_wavcaps()

    # Combine and deduplicate
    all_entries = audiocaps + musiccaps + wavcaps
    logger.info(f"Total before dedup: {len(all_entries)}")

    corpus = deduplicate(all_entries)

    # Save
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=None)  # compact, one object per logical line

    # Print distribution
    sources = {}
    for entry in corpus:
        src = entry["source"]
        sources[src] = sources.get(src, 0) + 1

    logger.info(f"Corpus saved to {CORPUS_PATH}")
    logger.info(f"Total: {len(corpus)} entries")
    for src, count in sorted(sources.items()):
        logger.info(f"  {src}: {count}")


if __name__ == "__main__":
    main()
