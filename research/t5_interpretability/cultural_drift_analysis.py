#!/usr/bin/env python3
"""
Cultural Drift Analysis

For each semantic axis (perceptual, cultural-contextual, critical), measures
how moving along the axis shifts the embedding's proximity to 15 musical
tradition centroids in T5 embedding space.

This connects the LERP sonification findings with the cultural distance
analysis: when you turn the "rhythmic" slider, do you also drift toward
specific cultural traditions?

No audio generation — pure embedding-space analysis.

Usage: venv/bin/python research/t5_interpretability/cultural_drift_analysis.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    T5_MODEL_ID, T5_MAX_LENGTH,
    DATA_DIR, ACTIVATIONS_POOLED_PATH, CORPUS_INDEX_PATH,
)

# ── Axes ──────────────────────────────────────────────────────────────────────
# Three levels: perceptual, cultural-contextual, critically revealing

AXES = {
    # Level 1: Perceptual-physical
    "perceptual": [
        ("rhythmic ↔ sustained", "sound rhythmic", "sound sustained"),
        ("bright ↔ dark", "sound bright", "sound dark"),
        ("smooth ↔ harsh", "sound smooth", "sound harsh"),
        ("dense ↔ sparse", "sound dense", "sound sparse"),
        ("tonal ↔ noisy", "sound tonal", "sound noisy"),
        ("fast ↔ slow", "sound fast", "sound slow"),
        ("loud ↔ quiet", "sound loud", "sound quiet"),
        ("close ↔ distant", "sound close", "sound distant"),
    ],
    # Level 2: Culturally-contextual
    "cultural": [
        ("traditional ↔ modern", "traditional music", "modern music"),
        ("acoustic ↔ electronic", "acoustic music", "electronic music"),
        ("sacred ↔ secular", "sacred music", "secular music"),
        ("solo ↔ ensemble", "solo music", "ensemble music"),
        ("improvised ↔ composed", "improvised music", "composed music"),
        ("ceremonial ↔ everyday", "ceremonial music", "everyday music"),
        ("vocal ↔ instrumental", "vocal music", "instrumental music"),
    ],
    # Level 3: Critically revealing
    "critical": [
        ("complex ↔ simple", "complex music", "simple music"),
        ("beautiful ↔ ugly", "beautiful sound", "ugly sound"),
        ("professional ↔ amateur", "professional music", "amateur music"),
        ("authentic ↔ fusion", "authentic music", "fusion music"),
        ("refined ↔ raw", "refined music", "raw music"),
        ("music ↔ noise", "music", "noise"),
    ],
}

LERP_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

OUTPUT_DIR = DATA_DIR / "cultural_drift"


def load_t5():
    from transformers import T5EncoderModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_ID)
    model = T5EncoderModel.from_pretrained(T5_MODEL_ID).cuda().half()
    model.eval()
    return tokenizer, model


def encode_prompt_pooled(text: str, tokenizer, model) -> np.ndarray:
    """Encode text → mean-pooled T5 embedding [768]."""
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
    emb = outputs.last_hidden_state.cpu().float()  # [1, seq, 768]
    mask = inputs.attention_mask.cpu().float()  # [1, seq]

    # Mean-pool over non-padding tokens (same as encode_corpus.py)
    emb_masked = emb * mask.unsqueeze(-1)
    pooled = emb_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return pooled.squeeze(0).numpy()  # [768]


def compute_tradition_centroids_t5(
    activations: np.ndarray,
    index: list[dict],
) -> dict[str, np.ndarray]:
    """Compute tradition centroids in raw T5 space (768d)."""
    traditions: dict[str, list[int]] = {}

    for i, entry in enumerate(index):
        if entry["category"].startswith("pillar1_"):
            subcat = entry["subcategory"]
            if subcat not in traditions:
                traditions[subcat] = []
            traditions[subcat].append(i)

    centroids = {}
    for name, indices in sorted(traditions.items()):
        centroids[name] = activations[indices].mean(axis=0)
        logger.info(f"  {name}: {len(indices)} prompts")

    return centroids


def measure_drift(
    lerp_positions: list[np.ndarray],
    centroids: dict[str, np.ndarray],
) -> dict[str, list[float]]:
    """Measure cosine distance from each LERP position to each tradition."""
    drift = {}
    for name, centroid in centroids.items():
        distances = []
        for emb in lerp_positions:
            d = cosine_distance(emb, centroid)
            distances.append(float(d))
        drift[name] = distances
    return drift


def compute_drift_magnitude(drift: dict[str, list[float]]) -> dict[str, float]:
    """For each tradition, compute total drift (distance at t=1.0 minus t=0.0)."""
    magnitudes = {}
    for name, distances in drift.items():
        magnitudes[name] = distances[-1] - distances[0]  # negative = getting closer to pole A
    return magnitudes


def generate_report(
    all_results: dict,
    centroids: dict[str, np.ndarray],
    neutral_distances: dict[str, float],
) -> str:
    lines = [
        "# Cultural Drift Analysis\n",
        "For each semantic axis, measures how moving from pole B (t=0.0) to pole A (t=1.0)",
        "shifts the embedding's proximity to 15 musical tradition centroids in T5 space.\n",
        "Negative drift = axis pole A moves CLOSER to that tradition.",
        "Positive drift = axis pole A moves FURTHER from that tradition.\n",
        f"LERP positions: {LERP_POSITIONS}\n",
    ]

    # Baseline: neutral distances
    lines.append("## Baseline: Distance of 'sound'/'music' to Each Tradition\n")
    sorted_neutral = sorted(neutral_distances.items(), key=lambda x: x[1])
    for name, d in sorted_neutral:
        bar = "#" * int(d * 200)
        lines.append(f"- {name:>25}: {d:.4f} {bar}")
    lines.append("")

    # Per-level results
    for level_name, level_axes in AXES.items():
        lines.append(f"---\n\n## Level: {level_name.title()}\n")

        for axis_label, prompt_a, prompt_b in level_axes:
            key = axis_label
            if key not in all_results:
                continue

            result = all_results[key]
            drift = result["drift"]
            magnitudes = result["magnitudes"]

            lines.append(f"### {axis_label}\n")
            lines.append(f"- Pole A (t=1.0): `{prompt_a}`")
            lines.append(f"- Pole B (t=0.0): `{prompt_b}`")
            lines.append(f"- Embedding distance between poles: {result['pole_distance']:.4f}\n")

            # Sort by drift magnitude
            sorted_mag = sorted(magnitudes.items(), key=lambda x: x[1])

            # Traditions that move CLOSER to pole A
            approaching = [(n, m) for n, m in sorted_mag if m < -0.001]
            receding = [(n, m) for n, m in sorted_mag if m > 0.001]

            if approaching:
                lines.append(f"**Traditions approaching pole A** (`{prompt_a}`):\n")
                for name, m in approaching:
                    lines.append(f"- {name}: {m:+.4f}")
                lines.append("")

            if receding:
                lines.append(f"**Traditions receding from pole A** (`{prompt_a}`):\n")
                for name, m in receding:
                    lines.append(f"- {name}: {m:+.4f}")
                lines.append("")

            if not approaching and not receding:
                lines.append("No significant cultural drift detected.\n")

            # Gradient table for top movers
            top_movers = sorted_mag[:3] + sorted_mag[-3:]
            lines.append("| Tradition | t=0.00 | t=0.25 | t=0.50 | t=0.75 | t=1.00 | Drift |")
            lines.append("|---|---|---|---|---|---|---|")
            for name, _ in top_movers:
                dists = drift[name]
                mag = magnitudes[name]
                lines.append(
                    f"| {name} | {dists[0]:.4f} | {dists[1]:.4f} | "
                    f"{dists[2]:.4f} | {dists[3]:.4f} | {dists[4]:.4f} | {mag:+.4f} |"
                )
            lines.append("")

    # Cross-axis confounding summary
    lines.append("---\n\n## Confounding Summary\n")
    lines.append("Which traditions are most affected by which axes?\n")
    lines.append("| Tradition | Strongest Drift | Axis | Direction |")
    lines.append("|---|---|---|---|")

    tradition_max_drift: dict[str, tuple[str, float]] = {}
    for key, result in all_results.items():
        for name, mag in result["magnitudes"].items():
            if name not in tradition_max_drift or abs(mag) > abs(tradition_max_drift[name][1]):
                tradition_max_drift[name] = (key, mag)

    for name in sorted(tradition_max_drift.keys()):
        axis, mag = tradition_max_drift[name]
        direction = "approaching pole A" if mag < 0 else "receding from pole A"
        lines.append(f"| {name} | {mag:+.4f} | {axis} | {direction} |")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load T5
    logger.info("Loading T5...")
    tokenizer, model = load_t5()

    # Load tradition centroids from activations
    logger.info("Loading activations and computing tradition centroids...")
    activations = torch.load(ACTIVATIONS_POOLED_PATH, weights_only=True).float().numpy()
    with open(CORPUS_INDEX_PATH) as f:
        index = json.load(f)

    centroids = compute_tradition_centroids_t5(activations, index)
    logger.info(f"Computed {len(centroids)} tradition centroids in T5 space")

    # Encode neutral prompt for baseline
    neutral_emb = encode_prompt_pooled("sound", tokenizer, model)
    neutral_distances = {}
    for name, centroid in centroids.items():
        neutral_distances[name] = float(cosine_distance(neutral_emb, centroid))

    # Process all axes
    all_results = {}
    total_axes = sum(len(axes) for axes in AXES.values())
    done = 0

    for level_name, level_axes in AXES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Level: {level_name}")

        for axis_label, prompt_a, prompt_b in level_axes:
            done += 1
            logger.info(f"\n[{done}/{total_axes}] {axis_label}")
            logger.info(f"  A: '{prompt_a}' | B: '{prompt_b}'")

            # Encode poles (mean-pooled, same as activations)
            emb_a = encode_prompt_pooled(prompt_a, tokenizer, model)
            emb_b = encode_prompt_pooled(prompt_b, tokenizer, model)

            pole_distance = float(cosine_distance(emb_a, emb_b))
            logger.info(f"  Pole distance: {pole_distance:.4f}")

            # LERP at multiple positions
            lerp_embs = []
            for t in LERP_POSITIONS:
                emb = (1.0 - t) * emb_b + t * emb_a
                lerp_embs.append(emb)

            # Measure drift to each tradition
            drift = measure_drift(lerp_embs, centroids)
            magnitudes = compute_drift_magnitude(drift)

            # Log top movers
            sorted_mag = sorted(magnitudes.items(), key=lambda x: x[1])
            logger.info(f"  Approaching pole A: {sorted_mag[0][0]} ({sorted_mag[0][1]:+.4f})")
            logger.info(f"  Receding from pole A: {sorted_mag[-1][0]} ({sorted_mag[-1][1]:+.4f})")

            all_results[axis_label] = {
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "pole_distance": pole_distance,
                "drift": drift,
                "magnitudes": magnitudes,
            }

    # Unload T5
    del model
    torch.cuda.empty_cache()

    # Save raw results
    with open(OUTPUT_DIR / "drift_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate report
    report = generate_report(all_results, centroids, neutral_distances)
    report_path = OUTPUT_DIR / "cultural_drift_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"\n{'='*60}")
    logger.info(f"Report: {report_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
