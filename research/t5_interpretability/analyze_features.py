#!/usr/bin/env python3
"""
Phase 5: Feature Interpretation

Runs all activations through trained SAE encoder → [N, 12288] sparse.
Per feature: top-20 activating prompts, category correlations (Pearson r),
cultural bias features, co-activation clustering, dead feature analysis.

Usage: venv/bin/python research/t5_interpretability/analyze_features.py
"""

import json
import logging
import sys
import time
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
    FEATURE_ATLAS_PATH, FEATURE_ATLAS_REPORT_PATH, DATA_DIR,
)
from train_sae import TopKSAE


def load_sae() -> TopKSAE:
    """Load trained SAE."""
    sae = TopKSAE(d_model=SAE_D_MODEL, n_features=SAE_N_FEATURES, k=SAE_K).cuda()
    sae.load_state_dict(torch.load(SAE_WEIGHTS_PATH, weights_only=True))
    sae.eval()
    logger.info("Loaded trained SAE")
    return sae


def encode_all(sae: TopKSAE, activations: torch.Tensor) -> np.ndarray:
    """Run all activations through SAE encoder → sparse features [N, n_features]."""
    n = activations.shape[0]
    all_sparse = []

    logger.info(f"Encoding {n} activations through SAE...")
    start = time.time()

    with torch.no_grad():
        for i in range(0, n, SAE_BATCH_SIZE):
            batch = activations[i:i + SAE_BATCH_SIZE].cuda()
            sparse = sae.encode(batch)  # [batch, n_features]
            all_sparse.append(sparse.cpu().numpy())

    features = np.concatenate(all_sparse, axis=0)  # [N, n_features]
    elapsed = time.time() - start
    logger.info(f"SAE encoding complete: {features.shape} in {elapsed:.1f}s")
    return features


def find_top_prompts(features: np.ndarray, corpus: list[dict], top_k: int = 20) -> dict:
    """Per-feature: top-K highest-activating prompts."""
    logger.info(f"Finding top-{top_k} prompts for {features.shape[1]} features...")
    n_features = features.shape[1]
    feature_tops = {}

    for feat_idx in range(n_features):
        col = features[:, feat_idx]

        # Skip dead features
        if col.max() == 0:
            continue

        top_indices = np.argsort(col)[-top_k:][::-1]
        feature_tops[feat_idx] = [
            {
                "index": int(idx),
                "text": corpus[idx]["text"],
                "category": corpus[idx].get("category", ""),
                "activation": float(col[idx]),
            }
            for idx in top_indices
            if col[idx] > 0
        ]

    alive = len(feature_tops)
    dead = n_features - alive
    logger.info(f"Alive features: {alive}, dead: {dead} ({dead/n_features*100:.1f}%)")
    return feature_tops


def compute_category_correlations(
    features: np.ndarray,
    index: list[dict],
) -> dict:
    """Per-feature: Pearson correlation with category membership vectors."""
    logger.info("Computing category correlations...")

    # Build category membership vectors
    categories = {}
    for i, entry in enumerate(index):
        cat = entry["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)

    # Filter to categories with enough members
    categories = {k: v for k, v in categories.items() if len(v) >= 20}

    n = features.shape[0]
    n_features = features.shape[1]

    # Build binary membership matrix [n_categories, N]
    cat_names = sorted(categories.keys())
    membership = np.zeros((len(cat_names), n), dtype=np.float32)
    for ci, cat in enumerate(cat_names):
        for idx in categories[cat]:
            membership[ci, idx] = 1.0

    # Compute correlations: for each feature, correlate with each category
    # Only for alive features (significant speedup)
    feature_correlations = {}
    feature_alive_mask = features.max(axis=0) > 0

    # Standardize features and membership for fast correlation
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True)
    feat_std[feat_std == 0] = 1
    features_z = (features - feat_mean) / feat_std

    mem_mean = membership.mean(axis=1, keepdims=True)
    mem_std = membership.std(axis=1, keepdims=True)
    mem_std[mem_std == 0] = 1
    membership_z = (membership - mem_mean) / mem_std

    # Batch correlation: [n_features, n_categories] = features_z.T @ membership_z.T / N
    corr_matrix = features_z.T @ membership_z.T / n  # [n_features, n_categories]

    for feat_idx in range(n_features):
        if not feature_alive_mask[feat_idx]:
            continue

        correlations = corr_matrix[feat_idx]
        # Top 5 most correlated categories
        top_cat_indices = np.argsort(np.abs(correlations))[-5:][::-1]
        feature_correlations[feat_idx] = [
            {"category": cat_names[ci], "pearson_r": float(correlations[ci])}
            for ci in top_cat_indices
            if abs(correlations[ci]) > 0.01
        ]

    logger.info(f"Computed correlations for {len(feature_correlations)} features × {len(cat_names)} categories")
    return feature_correlations


def find_cultural_bias_features(
    features: np.ndarray,
    index: list[dict],
) -> list[dict]:
    """Features that discriminate between cultural traditions."""
    logger.info("Finding cultural bias features...")

    # Group by tradition (subcategory within pillar1_* categories)
    traditions = {}
    for i, entry in enumerate(index):
        if entry["category"].startswith("pillar1_"):
            subcat = entry["subcategory"]
            if subcat not in traditions:
                traditions[subcat] = []
            traditions[subcat].append(i)

    if len(traditions) < 2:
        logger.warning("Not enough traditions for bias analysis")
        return []

    tradition_names = sorted(traditions.keys())
    logger.info(f"Traditions found: {tradition_names}")

    # Mean activation per tradition per feature
    tradition_means = {}
    for tname in tradition_names:
        indices = traditions[tname]
        tradition_means[tname] = features[indices].mean(axis=0)  # [n_features]

    # For each feature: compute max asymmetry between tradition pairs
    # Asymmetry = |mean_a - mean_b| / (mean_a + mean_b + eps)
    bias_features = []
    n_features = features.shape[1]

    for feat_idx in range(n_features):
        max_asym = 0
        max_pair = ("", "")
        for ai, ta in enumerate(tradition_names):
            ma = tradition_means[ta][feat_idx]
            for bi in range(ai + 1, len(tradition_names)):
                tb = tradition_names[bi]
                mb = tradition_means[tb][feat_idx]
                denom = abs(ma) + abs(mb) + 1e-8
                asym = abs(ma - mb) / denom
                if asym > max_asym:
                    max_asym = asym
                    max_pair = (ta, tb)

        if max_asym > 0.3:  # threshold for "bias feature"
            bias_features.append({
                "feature": feat_idx,
                "max_asymmetry": float(max_asym),
                "tradition_pair": list(max_pair),
                "tradition_means": {
                    t: float(tradition_means[t][feat_idx])
                    for t in tradition_names
                },
            })

    bias_features.sort(key=lambda x: x["max_asymmetry"], reverse=True)
    logger.info(f"Found {len(bias_features)} cultural bias features (asymmetry > 0.3)")
    return bias_features[:200]  # cap for JSON size


def cluster_features_by_coactivation(
    features: np.ndarray,
    n_clusters: int = 64,
    max_features_for_clustering: int = 2000,
) -> dict:
    """Cluster alive features by co-activation patterns."""
    from scipy.cluster.hierarchy import ward, fcluster
    from scipy.spatial.distance import pdist

    logger.info("Clustering features by co-activation...")

    # Only cluster alive features
    alive_mask = features.max(axis=0) > 0
    alive_indices = np.where(alive_mask)[0]

    if len(alive_indices) > max_features_for_clustering:
        # Sample the most active features
        feature_activity = features[:, alive_indices].sum(axis=0)
        top_active = np.argsort(feature_activity)[-max_features_for_clustering:]
        selected = alive_indices[top_active]
    else:
        selected = alive_indices

    if len(selected) < 2:
        return {"families": {}, "n_families": 0}

    # Co-activation matrix: correlation between feature activation vectors
    feat_subset = features[:, selected].T  # [n_selected, N]

    dist = pdist(feat_subset, metric="correlation")
    # Handle NaN distances (constant features)
    dist = np.nan_to_num(dist, nan=1.0)

    linkage = ward(dist)
    labels = fcluster(linkage, t=min(n_clusters, len(selected)), criterion="maxclust")

    families = {}
    for idx, (feat_idx, cluster_id) in enumerate(zip(selected, labels)):
        cid = int(cluster_id)
        if cid not in families:
            families[cid] = []
        families[cid].append(int(feat_idx))

    logger.info(f"Formed {len(families)} feature families from {len(selected)} features")
    return {"families": {str(k): v for k, v in families.items()}, "n_families": len(families)}


def generate_report(
    feature_tops: dict,
    feature_correlations: dict,
    bias_features: list[dict],
    families: dict,
    n_total_features: int,
) -> str:
    """Generate human-readable markdown report."""
    lines = ["# Feature Atlas Report\n"]

    # Overview
    alive = len(feature_tops)
    dead = n_total_features - alive
    lines.append(f"## Overview\n")
    lines.append(f"- Total features: {n_total_features:,}")
    lines.append(f"- Alive features: {alive:,} ({alive/n_total_features*100:.1f}%)")
    lines.append(f"- Dead features: {dead:,} ({dead/n_total_features*100:.1f}%)")
    lines.append(f"- Feature families: {families.get('n_families', 0)}")
    lines.append(f"- Cultural bias features: {len(bias_features)}\n")

    # Top 20 most interesting features (by category correlation strength)
    lines.append(f"## Top 20 Most Interpretable Features\n")
    scored = []
    for feat_idx, corrs in feature_correlations.items():
        if corrs:
            max_r = max(abs(c["pearson_r"]) for c in corrs)
            scored.append((feat_idx, max_r, corrs))
    scored.sort(key=lambda x: x[1], reverse=True)

    for rank, (feat_idx, max_r, corrs) in enumerate(scored[:20]):
        tops = feature_tops.get(feat_idx, [])
        top_texts = [t["text"][:80] for t in tops[:5]]
        top_cats = [f'{c["category"]} (r={c["pearson_r"]:.3f})' for c in corrs[:3]]

        lines.append(f"### Feature {feat_idx} (max |r| = {max_r:.3f})")
        lines.append(f"Top categories: {', '.join(top_cats)}")
        lines.append(f"Top prompts:")
        for t in top_texts:
            lines.append(f"  - {t}")
        lines.append("")

    # Cultural bias features
    if bias_features:
        lines.append(f"## Top Cultural Bias Features\n")
        for bf in bias_features[:10]:
            lines.append(
                f"### Feature {bf['feature']} "
                f"(asymmetry={bf['max_asymmetry']:.3f}, pair={bf['tradition_pair']})"
            )
            # Show per-tradition means
            means = bf["tradition_means"]
            sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)
            for tname, val in sorted_means[:5]:
                lines.append(f"  - {tname}: {val:.4f}")
            lines.append(f"  ...")
            for tname, val in sorted_means[-3:]:
                lines.append(f"  - {tname}: {val:.4f}")
            lines.append("")

    return "\n".join(lines)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    activations = torch.load(ACTIVATIONS_POOLED_PATH, weights_only=True).float()
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    with open(CORPUS_INDEX_PATH) as f:
        index = json.load(f)

    logger.info(f"Loaded: activations {activations.shape}, corpus {len(corpus)} entries")

    # Load and run SAE
    sae = load_sae()
    features = encode_all(sae, activations)

    # Analysis
    feature_tops = find_top_prompts(features, corpus)
    feature_correlations = compute_category_correlations(features, index)
    bias_features = find_cultural_bias_features(features, index)
    families = cluster_features_by_coactivation(features)

    # Save atlas
    atlas = {
        "n_features": SAE_N_FEATURES,
        "n_alive": len(feature_tops),
        "feature_tops": {str(k): v for k, v in feature_tops.items()},
        "feature_correlations": {str(k): v for k, v in feature_correlations.items()},
        "cultural_bias_features": bias_features,
        "feature_families": families,
    }

    with open(FEATURE_ATLAS_PATH, "w") as f:
        json.dump(atlas, f, indent=2)
    logger.info(f"Feature atlas saved to {FEATURE_ATLAS_PATH}")

    # Generate report
    report = generate_report(
        feature_tops, feature_correlations, bias_features, families, SAE_N_FEATURES
    )
    with open(FEATURE_ATLAS_REPORT_PATH, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {FEATURE_ATLAS_REPORT_PATH}")


if __name__ == "__main__":
    main()
