#!/usr/bin/env python3
"""
Binary Contrast Sonification

Encodes atomic prompt pairs ("sound smooth" vs "sound harsh") through T5,
computes the difference vector, and generates audio at various strengths
along that direction. If the direction is semantically meaningful,
the audio should change perceptibly from one pole to the other.

This eliminates the vocabulary-frequency confound of the cultural probing:
both words in each pair are common English, so any difference in
the embedding reflects semantic content, not corpus statistics.

Usage: venv/bin/python research/t5_interpretability/sonify_binary_contrasts.py
"""

import base64
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    T5_MODEL_ID, T5_MAX_LENGTH,
    GPU_SERVICE_URL, DATA_DIR,
)

# ── Binary Contrast Pairs ─────────────────────────────────────────────────────
# Each pair: (pole_a, pole_b) — atomic English adjectives
# Chosen for perceptual verifiability: you can HEAR the difference

CONTRASTS = [
    ("smooth", "harsh"),
    ("high", "low"),
    ("fast", "slow"),
    ("loud", "quiet"),
    ("bright", "dark"),
    ("sharp", "soft"),
    ("rhythmic", "sustained"),
    ("metallic", "wooden"),
    ("wet", "dry"),
    ("thick", "thin"),
]

STRENGTHS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
PROMPT_TEMPLATE = "sound"  # neutral anchor
DURATION = 5.0  # longer than the 2s features — give the sound time to develop
STEPS = 100
CFG = 7.0
OUTPUT_DIR = DATA_DIR / "sonification_binary"


def load_t5():
    """Load T5-Base encoder."""
    from transformers import T5EncoderModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_ID)
    model = T5EncoderModel.from_pretrained(T5_MODEL_ID).cuda().half()
    model.eval()
    return tokenizer, model


def encode_prompt(text: str, tokenizer, model) -> tuple[np.ndarray, np.ndarray]:
    """Encode a text prompt → [1, seq, 768] embedding + attention mask."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=T5_MAX_LENGTH,
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask.cuda(),
        )
    emb = outputs.last_hidden_state.cpu().float().numpy()  # [1, seq, 768]
    mask = inputs.attention_mask.cpu().float().numpy()
    return emb, mask


def _numpy_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_audio(embeddings: np.ndarray, mask: np.ndarray, seed: int = 42) -> bytes | None:
    """Call GPU service to generate audio from embeddings."""
    url = f"{GPU_SERVICE_URL}/api/stable_audio/generate_from_embeddings"
    payload = {
        "embeddings_b64": _numpy_to_b64(embeddings),
        "attention_mask_b64": _numpy_to_b64(mask),
        "duration_seconds": DURATION,
        "steps": STEPS,
        "cfg_scale": CFG,
        "seed": seed,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return base64.b64decode(data["audio_base64"])
        else:
            logger.error(f"Generation failed: {data.get('error')}")
            return None
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load T5
    tokenizer, model = load_t5()

    # Encode neutral anchor
    logger.info(f"Encoding neutral prompt: '{PROMPT_TEMPLATE}'")
    neutral_emb, neutral_mask = encode_prompt(PROMPT_TEMPLATE, tokenizer, model)

    total = len(CONTRASTS) * len(STRENGTHS)
    done = 0
    start_time = time.time()

    for pole_a, pole_b in CONTRASTS:
        # Encode both poles
        prompt_a = f"{PROMPT_TEMPLATE} {pole_a}"
        prompt_b = f"{PROMPT_TEMPLATE} {pole_b}"

        emb_a, _ = encode_prompt(prompt_a, tokenizer, model)
        emb_b, _ = encode_prompt(prompt_b, tokenizer, model)

        # Difference vector: direction from B to A
        # Positive strength = more pole_a, negative = more pole_b
        diff = emb_a - emb_b  # [1, seq, 768]
        diff_norm = np.linalg.norm(diff)

        logger.info(f"\n{'='*60}")
        logger.info(f"Contrast: {pole_a} ↔ {pole_b} (diff norm={diff_norm:.4f})")

        for strength in STRENGTHS:
            filename = f"{pole_a}_vs_{pole_b}_strength_{strength:+.1f}.wav"
            out_path = OUTPUT_DIR / filename

            if out_path.exists():
                done += 1
                continue

            # Inject: neutral + strength * normalized_diff
            modified = neutral_emb + strength * diff
            audio = generate_audio(modified, neutral_mask)

            if audio:
                with open(out_path, "wb") as f:
                    f.write(audio)

            done += 1
            elapsed = time.time() - start_time
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            logger.info(
                f"  [{done}/{total}] {filename} "
                f"{'OK' if audio else 'FAILED'} "
                f"(elapsed={elapsed:.0f}s, ETA={eta:.0f}s)"
            )

    # Also generate the pure pole prompts as text (not injection)
    logger.info(f"\n{'='*60}")
    logger.info("Generating pure text-prompted references...")

    for pole_a, pole_b in CONTRASTS:
        for pole in [pole_a, pole_b]:
            prompt = f"{PROMPT_TEMPLATE} {pole}"
            filename = f"reference_{pole}.wav"
            out_path = OUTPUT_DIR / filename

            if out_path.exists():
                continue

            emb, mask = encode_prompt(prompt, tokenizer, model)
            audio = generate_audio(emb, mask)

            if audio:
                with open(out_path, "wb") as f:
                    f.write(audio)
            logger.info(f"  {filename}: {'OK' if audio else 'FAILED'}")

    # Unload T5
    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    logger.info(f"\nDone in {elapsed:.0f}s. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
