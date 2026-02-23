#!/usr/bin/env python3
"""
Phase 7: Cultural Analysis

Using probing corpus results + SAE features:
1. Tradition centroids in 12,288d feature space
2. Pairwise cosine distances → 15×15 matrix
3. Default-encoding bias: distance of bare "music" to each tradition centroid
4. Bias dimensionality: how many features separate cultural groups
5. Statistical significance via permutation testing

Usage: venv/bin/python research/t5_interpretability/cultural_analysis.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SAE_D_MODEL, SAE_N_FEATURES, SAE_K, SAE_BATCH_SIZE,
    ACTIVATIONS_POOLED_PATH, SAE_WEIGHTS_PATH,
    CORPUS_PATH, CORPUS_INDEX_PATH,
    CULTURAL_REPORT_PATH, DATA_DIR,
    T5_MODEL_ID, T5_MAX_LENGTH,
)
from train_sae import TopKSAE


def load_and_encode() -> tuple[np.ndarray, list[dict], list[dict]]:
    """Load activations, run through SAE, return feature activations + metadata."""
    activations = torch.load(ACTIVATIONS_POOLED_PATH, weights_only=True).float()
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    with open(CORPUS_INDEX_PATH) as f:
        index = json.load(f)

    sae = TopKSAE(d_model=SAE_D_MODEL, n_features=SAE_N_FEATURES, k=SAE_K).cuda()
    sae.load_state_dict(torch.load(SAE_WEIGHTS_PATH, weights_only=True))
    sae.eval()

    # Encode all through SAE
    all_sparse = []
    n = activations.shape[0]
    with torch.no_grad():
        for i in range(0, n, SAE_BATCH_SIZE):
            batch = activations[i:i + SAE_BATCH_SIZE].cuda()
            sparse = sae.encode(batch)
            all_sparse.append(sparse.cpu().numpy())

    features = np.concatenate(all_sparse, axis=0)
    logger.info(f"SAE features: {features.shape}")
    return features, corpus, index


def compute_tradition_centroids(
    features: np.ndarray,
    index: list[dict],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Compute mean feature vector per tradition."""
    traditions = {}
    for i, entry in enumerate(index):
        if entry["category"].startswith("pillar1_"):
            subcat = entry["subcategory"]
            if subcat not in traditions:
                traditions[subcat] = []
            traditions[subcat].append(i)

    centroids = {}
    names = sorted(traditions.keys())
    for tname in names:
        indices = traditions[tname]
        centroids[tname] = features[indices].mean(axis=0)

    logger.info(f"Computed centroids for {len(centroids)} traditions")
    return centroids, names


def compute_pairwise_distances(
    centroids: dict[str, np.ndarray],
    names: list[str],
) -> np.ndarray:
    """15×15 pairwise cosine distance matrix."""
    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                a = centroids[names[i]]
                b = centroids[names[j]]
                cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                matrix[i, j] = 1.0 - cos_sim

    return matrix


def compute_default_encoding_bias(
    features: np.ndarray,
    corpus: list[dict],
    centroids: dict[str, np.ndarray],
    names: list[str],
) -> dict[str, float]:
    """Distance of bare "music" to each tradition centroid."""
    # Find "music" baseline prompts
    music_indices = [
        i for i, entry in enumerate(corpus)
        if entry["text"].strip().lower() in ("music", "musical performance", "a sound playing")
    ]

    if not music_indices:
        # Fallback: use all baseline prompts
        music_indices = [
            i for i, entry in enumerate(corpus)
            if entry.get("category") == "control_baseline"
        ]

    if not music_indices:
        logger.warning("No baseline prompts found for default encoding analysis")
        return {}

    music_centroid = features[music_indices].mean(axis=0)

    distances = {}
    for tname in names:
        a = music_centroid
        b = centroids[tname]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        distances[tname] = float(1.0 - cos_sim)

    # Sort by distance (closest = most "default")
    distances = dict(sorted(distances.items(), key=lambda x: x[1]))
    return distances


def compute_bias_dimensionality(
    centroids: dict[str, np.ndarray],
    names: list[str],
) -> dict:
    """How many SAE features are needed to distinguish traditions?"""
    # For each feature dimension: compute variance of centroid values across traditions
    centroid_matrix = np.array([centroids[name] for name in names])  # [15, n_features]

    # Variance across traditions per feature
    per_feature_var = centroid_matrix.var(axis=0)  # [n_features]

    # How many features have significant variance (> 1% of max)?
    threshold = per_feature_var.max() * 0.01
    discriminative_features = (per_feature_var > threshold).sum()

    # How many features carry 90% of discrimination?
    sorted_vars = np.sort(per_feature_var)[::-1]
    cumulative = np.cumsum(sorted_vars) / sorted_vars.sum()
    features_for_90pct = int(np.searchsorted(cumulative, 0.90)) + 1

    return {
        "discriminative_features": int(discriminative_features),
        "features_for_90pct": features_for_90pct,
        "total_features": len(per_feature_var),
        "top_discriminative_indices": [
            int(i) for i in np.argsort(per_feature_var)[-20:][::-1]
        ],
    }


def permutation_test(
    features: np.ndarray,
    index: list[dict],
    names: list[str],
    n_permutations: int = 1000,
) -> dict:
    """Statistical significance: are tradition differences real or chance?"""
    logger.info(f"Running permutation test ({n_permutations} permutations)...")

    # Collect all pillar1 indices grouped by tradition
    traditions = {}
    all_pillar1_indices = []
    for i, entry in enumerate(index):
        if entry["category"].startswith("pillar1_"):
            subcat = entry["subcategory"]
            if subcat not in traditions:
                traditions[subcat] = []
            traditions[subcat].append(i)
            all_pillar1_indices.append(i)

    if len(traditions) < 2:
        return {"p_value": 1.0, "observed_dispersion": 0.0}

    # Observed: mean pairwise cosine distance between tradition centroids
    centroids = {t: features[idx].mean(axis=0) for t, idx in traditions.items()}
    observed_distances = []
    trad_names = sorted(traditions.keys())
    for i in range(len(trad_names)):
        for j in range(i + 1, len(trad_names)):
            a = centroids[trad_names[i]]
            b = centroids[trad_names[j]]
            cos_d = 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            observed_distances.append(cos_d)
    observed_mean = np.mean(observed_distances)

    # Permutation: shuffle tradition labels, recompute
    rng = np.random.default_rng(42)
    sizes = [len(traditions[t]) for t in trad_names]
    n_more_extreme = 0

    for _ in range(n_permutations):
        shuffled = rng.permutation(all_pillar1_indices)
        perm_centroids = {}
        offset = 0
        for ti, t in enumerate(trad_names):
            perm_idx = shuffled[offset:offset + sizes[ti]]
            perm_centroids[t] = features[perm_idx].mean(axis=0)
            offset += sizes[ti]

        perm_distances = []
        for i in range(len(trad_names)):
            for j in range(i + 1, len(trad_names)):
                a = perm_centroids[trad_names[i]]
                b = perm_centroids[trad_names[j]]
                cos_d = 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                perm_distances.append(cos_d)

        if np.mean(perm_distances) >= observed_mean:
            n_more_extreme += 1

    p_value = (n_more_extreme + 1) / (n_permutations + 1)
    logger.info(f"Permutation test: p={p_value:.4f} (observed mean distance={observed_mean:.6f})")

    return {
        "p_value": float(p_value),
        "observed_mean_distance": float(observed_mean),
        "n_permutations": n_permutations,
    }


def generate_report(
    distance_matrix: np.ndarray,
    names: list[str],
    default_bias: dict,
    dimensionality: dict,
    perm_test: dict,
) -> str:
    """Generate cultural analysis markdown report."""
    lines = ["# Cultural Analysis Report\n"]
    lines.append("## T5 Audio-Semantic Space: Cultural Distance Geometry\n")

    # Statistical significance
    lines.append("## Statistical Significance\n")
    lines.append(f"- Permutation test p-value: **{perm_test.get('p_value', 'N/A')}**")
    lines.append(f"- Observed mean pairwise distance: {perm_test.get('observed_mean_distance', 'N/A'):.6f}")
    lines.append(f"- Permutations: {perm_test.get('n_permutations', 'N/A')}")
    sig = "YES" if perm_test.get("p_value", 1.0) < 0.01 else "NO"
    lines.append(f"- Significant at p < 0.01: **{sig}**\n")

    # Pairwise distance matrix
    lines.append("## Pairwise Cosine Distance Matrix\n")
    lines.append("Cosine distance between tradition centroids in SAE feature space.\n")

    # Header
    short_names = [n[:8] for n in names]
    header = "| |" + "|".join(f" {s:>8} " for s in short_names) + "|"
    sep = "|" + "|".join(["---"] * (len(names) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for i, name in enumerate(names):
        row = f"| {name[:8]:>8} |"
        for j in range(len(names)):
            if i == j:
                row += "    —     |"
            else:
                row += f" {distance_matrix[i,j]:.4f}  |"
        lines.append(row)
    lines.append("")

    # Closest and furthest pairs
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], distance_matrix[i, j]))
    pairs.sort(key=lambda x: x[2])

    lines.append("### Closest Tradition Pairs")
    for a, b, d in pairs[:5]:
        lines.append(f"- {a} ↔ {b}: {d:.4f}")
    lines.append("")

    lines.append("### Most Distant Tradition Pairs")
    for a, b, d in pairs[-5:]:
        lines.append(f"- {a} ↔ {b}: {d:.4f}")
    lines.append("")

    # Default encoding bias
    lines.append("## Default-Encoding Bias\n")
    lines.append('Distance of bare "music" centroid to each tradition centroid.\n')
    lines.append("Closer = T5 treats this tradition as more 'default-like'.\n")

    for tname, dist in default_bias.items():
        bar = "█" * int(dist * 200)  # visual bar
        lines.append(f"- {tname:>20}: {dist:.4f} {bar}")
    lines.append("")

    # Bias dimensionality
    lines.append("## Bias Dimensionality\n")
    lines.append(f"- Discriminative features (var > 1% of max): **{dimensionality['discriminative_features']}**")
    lines.append(f"- Features for 90% discrimination: **{dimensionality['features_for_90pct']}**")
    lines.append(f"- Total features: {dimensionality['total_features']}")
    lines.append("")

    few_or_many = "few" if dimensionality['features_for_90pct'] < 100 else "many"
    lines.append(f"Interpretation: Cultural bias is encoded in **{few_or_many}** features.")
    if few_or_many == "few":
        lines.append("This suggests strong, concentrated bias encoding — a small number of features act as 'cultural switches'.")
    else:
        lines.append("This suggests distributed bias encoding — cultural identity is spread across many features.")
    lines.append("")

    return "\n".join(lines)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load and encode
    features, corpus, index = load_and_encode()

    # Compute tradition centroids
    centroids, names = compute_tradition_centroids(features, index)

    # Pairwise distances
    distance_matrix = compute_pairwise_distances(centroids, names)

    # Default encoding bias
    default_bias = compute_default_encoding_bias(features, corpus, centroids, names)

    # Bias dimensionality
    dimensionality = compute_bias_dimensionality(centroids, names)

    # Permutation test
    perm_test = permutation_test(features, index, names, n_permutations=1000)

    # Generate report
    report = generate_report(distance_matrix, names, default_bias, dimensionality, perm_test)

    with open(CULTURAL_REPORT_PATH, "w") as f:
        f.write(report)
    logger.info(f"Cultural analysis report saved to {CULTURAL_REPORT_PATH}")

    # Summary
    logger.info("\n=== Key Findings ===")
    logger.info(f"p-value: {perm_test['p_value']:.4f}")
    if default_bias:
        closest = list(default_bias.items())[0]
        furthest = list(default_bias.items())[-1]
        logger.info(f"Closest to 'music' default: {closest[0]} ({closest[1]:.4f})")
        logger.info(f"Furthest from 'music' default: {furthest[0]} ({furthest[1]:.4f})")
    logger.info(f"Bias dimensionality: {dimensionality['features_for_90pct']} features for 90% discrimination")


if __name__ == "__main__":
    main()
