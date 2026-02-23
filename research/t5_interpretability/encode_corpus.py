#!/usr/bin/env python3
"""
Phase 2: Batch T5-Base Encoding

Loads T5-Base standalone (NOT via Stable Audio — text_projection = nn.Identity(),
so standalone hidden states are identical). Encodes all ~100K prompts in batches,
mean-pools over non-padding tokens, saves activations.

VRAM: T5-Base fp16 ~440MB. Total <1GB.
Time: ~100K ÷ 64 batch × ~50ms ≈ 1.5 minutes.

Usage: venv/bin/python research/t5_interpretability/encode_corpus.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    T5_MODEL_ID, T5_MAX_LENGTH, ENCODING_BATCH_SIZE,
    CORPUS_PATH, ACTIVATIONS_POOLED_PATH, CORPUS_INDEX_PATH, DATA_DIR,
)


def load_corpus() -> list[dict]:
    """Load corpus.json and return list of entry dicts."""
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    logger.info(f"Loaded corpus: {len(corpus)} entries")
    return corpus


def encode_all(corpus: list[dict]) -> torch.Tensor:
    """Encode all prompts through T5-Base, return mean-pooled activations [N, 768]."""
    from transformers import T5EncoderModel, AutoTokenizer

    logger.info(f"Loading T5-Base from {T5_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_ID)
    model = T5EncoderModel.from_pretrained(T5_MODEL_ID).cuda().half()
    model.eval()

    texts = [entry["text"] for entry in corpus]
    n = len(texts)
    all_pooled = []

    logger.info(f"Encoding {n} prompts in batches of {ENCODING_BATCH_SIZE}...")
    start = time.time()

    for i in range(0, n, ENCODING_BATCH_SIZE):
        batch_texts = texts[i:i + ENCODING_BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            max_length=T5_MAX_LENGTH,
            truncation=True,
        )
        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, seq, 768]

        # Mean-pool over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).half()  # [B, seq, 1]
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # [B, 768]
        lengths = attention_mask.sum(dim=1, keepdim=True).half()  # [B, 1]
        pooled = sum_hidden / lengths.clamp(min=1)  # [B, 768]

        all_pooled.append(pooled.cpu())

        if (i // ENCODING_BATCH_SIZE) % 100 == 0:
            elapsed = time.time() - start
            done = i + len(batch_texts)
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(f"  {done}/{n} ({rate:.0f} prompts/sec)")

    elapsed = time.time() - start
    logger.info(f"Encoding complete in {elapsed:.1f}s ({n / elapsed:.0f} prompts/sec)")

    activations = torch.cat(all_pooled, dim=0)  # [N, 768]
    return activations


def validate(activations: torch.Tensor, n_corpus: int):
    """Validate activations tensor."""
    assert activations.shape[0] == n_corpus, f"Shape mismatch: {activations.shape[0]} vs {n_corpus}"
    assert activations.shape[1] == 768, f"Unexpected dim: {activations.shape[1]}"
    assert not torch.isnan(activations).any(), "NaN detected in activations"
    assert not torch.isinf(activations).any(), "Inf detected in activations"
    logger.info(f"Validation passed: shape={list(activations.shape)}, dtype={activations.dtype}")
    logger.info(f"  mean={activations.mean().item():.4f}, std={activations.std().item():.4f}")
    logger.info(f"  min={activations.min().item():.4f}, max={activations.max().item():.4f}")


def save_corpus_index(corpus: list[dict]):
    """Save corpus index mapping row → metadata (without text, to save space)."""
    index = []
    for entry in corpus:
        index.append({
            "source": entry["source"],
            "category": entry["category"],
            "subcategory": entry.get("subcategory", ""),
        })
    with open(CORPUS_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=None)
    logger.info(f"Corpus index saved to {CORPUS_INDEX_PATH}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus()
    activations = encode_all(corpus)
    validate(activations, len(corpus))

    # Save activations as fp16
    torch.save(activations.half(), ACTIVATIONS_POOLED_PATH)
    logger.info(f"Activations saved to {ACTIVATIONS_POOLED_PATH}")
    logger.info(f"  File size: {ACTIVATIONS_POOLED_PATH.stat().st_size / (1024*1024):.1f} MB")

    save_corpus_index(corpus)


if __name__ == "__main__":
    main()
