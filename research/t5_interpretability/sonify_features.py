#!/usr/bin/env python3
"""
Phase 6: Feature Sonification

For top 50 most interpretable SAE features:
1. Encode neutral prompt "sound" via standalone T5
2. Extract decoder column (unit-norm 768d direction vector)
3. Inject: neutral_emb + strength * decoder_col at [-2, -1, 0, +1, +2]
4. Call GPU service HTTP endpoint per feature × strength
5. Save WAVs

Requires GPU service running on port 17803 with Stable Audio loaded.
~250 generations, ~2 hours.

Usage: venv/bin/python research/t5_interpretability/sonify_features.py
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
    SAE_D_MODEL, SAE_N_FEATURES, SAE_K,
    SAE_WEIGHTS_PATH, FEATURE_ATLAS_PATH,
    T5_MODEL_ID, T5_MAX_LENGTH,
    GPU_SERVICE_URL,
    SONIFICATION_DIR, SONIFICATION_TOP_FEATURES, SONIFICATION_STRENGTHS,
    SONIFICATION_DURATION_SECONDS, SONIFICATION_STEPS, SONIFICATION_CFG_SCALE,
    SONIFICATION_NEUTRAL_PROMPT,
    DATA_DIR,
)
from train_sae import TopKSAE


def encode_neutral_prompt() -> tuple[np.ndarray, np.ndarray]:
    """Encode neutral prompt via standalone T5-Base → [1, seq, 768] + attention mask."""
    from transformers import T5EncoderModel, AutoTokenizer

    logger.info(f"Encoding neutral prompt: '{SONIFICATION_NEUTRAL_PROMPT}'")
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_ID)
    model = T5EncoderModel.from_pretrained(T5_MODEL_ID).cuda().half()
    model.eval()

    inputs = tokenizer(
        SONIFICATION_NEUTRAL_PROMPT,
        return_tensors="pt",
        padding="max_length",
        max_length=T5_MAX_LENGTH,
        truncation=True,
    )
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [1, seq, 768]

    # Convert to float32 numpy for manipulation
    embeddings = hidden_states.cpu().float().numpy()
    mask = attention_mask.cpu().float().numpy()

    logger.info(f"Neutral embedding shape: {embeddings.shape}")

    # Unload T5 from GPU
    del model
    torch.cuda.empty_cache()

    return embeddings, mask


def get_top_features(n: int) -> list[int]:
    """Get top-N most interpretable feature indices from feature atlas."""
    with open(FEATURE_ATLAS_PATH) as f:
        atlas = json.load(f)

    # Score features by max category correlation
    correlations = atlas.get("feature_correlations", {})
    scored = []
    for feat_str, corrs in correlations.items():
        if corrs:
            max_r = max(abs(c["pearson_r"]) for c in corrs)
            scored.append((int(feat_str), max_r))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [feat_idx for feat_idx, _ in scored[:n]]
    logger.info(f"Top {n} features by interpretability: {top[:10]}...")
    return top


def extract_decoder_columns(sae: TopKSAE, feature_indices: list[int]) -> dict[int, np.ndarray]:
    """Extract decoder weight columns (768d direction vectors) for selected features."""
    # decoder.weight shape: [d_model, n_features]
    decoder_weight = sae.decoder.weight.data.cpu().float().numpy()  # [768, n_features]

    columns = {}
    for feat_idx in feature_indices:
        col = decoder_weight[:, feat_idx]  # [768]
        # Already unit-norm from training, but re-normalize to be safe
        col = col / (np.linalg.norm(col) + 1e-8)
        columns[feat_idx] = col

    return columns


def _numpy_to_b64(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string via .npy format."""
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_from_embeddings(
    embeddings: np.ndarray,
    attention_mask: np.ndarray,
    seed: int = 42,
) -> bytes | None:
    """Call GPU service to generate audio from embeddings."""
    url = f"{GPU_SERVICE_URL}/api/stable_audio/generate_from_embeddings"

    payload = {
        "embeddings_b64": _numpy_to_b64(embeddings),
        "attention_mask_b64": _numpy_to_b64(attention_mask),
        "duration_seconds": SONIFICATION_DURATION_SECONDS,
        "steps": SONIFICATION_STEPS,
        "cfg_scale": SONIFICATION_CFG_SCALE,
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
        logger.error(f"GPU service request failed: {e}")
        return None


def main():
    SONIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Encode neutral prompt
    neutral_emb, attention_mask = encode_neutral_prompt()

    # Step 2: Load SAE and extract decoder columns
    sae = TopKSAE(d_model=SAE_D_MODEL, n_features=SAE_N_FEATURES, k=SAE_K)
    sae.load_state_dict(torch.load(SAE_WEIGHTS_PATH, weights_only=True, map_location="cpu"))
    sae.eval()

    top_features = get_top_features(SONIFICATION_TOP_FEATURES)
    decoder_cols = extract_decoder_columns(sae, top_features)

    # Step 3: Generate audio for each feature × strength
    total = len(top_features) * len(SONIFICATION_STRENGTHS)
    done = 0
    failed = 0
    start_time = time.time()

    for feat_idx in top_features:
        col = decoder_cols[feat_idx]  # [768]

        for strength in SONIFICATION_STRENGTHS:
            out_path = SONIFICATION_DIR / f"feature_{feat_idx}_strength_{strength:.1f}.wav"

            # Skip if already generated
            if out_path.exists():
                done += 1
                continue

            # Inject feature direction into embedding
            # col is [768], neutral_emb is [1, seq, 768]
            # Add feature direction uniformly across all sequence positions
            modified_emb = neutral_emb.copy()
            modified_emb += strength * col[np.newaxis, np.newaxis, :]  # broadcast [1, 1, 768]

            # Generate
            audio_bytes = generate_from_embeddings(modified_emb, attention_mask, seed=42)

            if audio_bytes:
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
            else:
                failed += 1

            done += 1
            elapsed = time.time() - start_time
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            logger.info(
                f"[{done}/{total}] feature={feat_idx}, strength={strength:.1f}, "
                f"{'OK' if audio_bytes else 'FAILED'} "
                f"(elapsed={elapsed:.0f}s, ETA={eta:.0f}s)"
            )

    elapsed = time.time() - start_time
    logger.info(f"\nSonification complete: {done - failed}/{total} succeeded in {elapsed:.0f}s")
    if failed:
        logger.warning(f"{failed} generations failed")


if __name__ == "__main__":
    main()
