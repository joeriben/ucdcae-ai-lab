#!/usr/bin/env python3
"""
Phase 3: Dimension Atlas (Alammar-style, no SAE needed)

Operates on activations_pooled.pt (N × 768). Pure numpy/scipy, no GPU.

1. Per-dimension statistics (mean, std, skew, kurtosis)
2. 768×768 correlation matrix → hierarchical clustering (Ward)
3. Top-20 prompts per cluster
4. Probing category mean activations per dimension

Time: seconds.

Usage: venv/bin/python research/t5_interpretability/dimension_atlas.py
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
    ACTIVATIONS_POOLED_PATH, CORPUS_PATH, CORPUS_INDEX_PATH,
    DIMENSION_ATLAS_PATH, DATA_DIR,
)


def load_data():
    """Load activations and corpus."""
    activations = torch.load(ACTIVATIONS_POOLED_PATH, weights_only=True).float().numpy()
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    with open(CORPUS_INDEX_PATH) as f:
        index = json.load(f)
    logger.info(f"Loaded activations: {activations.shape}, corpus: {len(corpus)} entries")
    return activations, corpus, index


def compute_dim_statistics(activations: np.ndarray) -> list[dict]:
    """Per-dimension statistics across all prompts."""
    from scipy.stats import skew, kurtosis

    n_dims = activations.shape[1]
    stats = []

    means = activations.mean(axis=0)
    stds = activations.std(axis=0)
    skews = skew(activations, axis=0)
    kurts = kurtosis(activations, axis=0)

    for d in range(n_dims):
        stats.append({
            "dim": d,
            "mean": float(means[d]),
            "std": float(stds[d]),
            "skew": float(skews[d]),
            "kurtosis": float(kurts[d]),
        })

    logger.info(f"Computed statistics for {n_dims} dimensions")
    return stats


def cluster_dimensions(activations: np.ndarray, n_clusters: int = 32) -> dict:
    """Hierarchical clustering on 768×768 correlation matrix."""
    from scipy.cluster.hierarchy import ward, fcluster
    from scipy.spatial.distance import pdist

    logger.info("Computing correlation matrix and clustering...")

    # Correlation matrix (768 × 768): how dims co-activate across prompts
    corr = np.corrcoef(activations.T)  # [768, 768]

    # Convert correlation to distance (1 - |corr|) for clustering
    # Use condensed distance matrix
    dist = pdist(corr, metric="correlation")

    # Ward hierarchical clustering
    linkage = ward(dist)

    # Cut into clusters
    labels = fcluster(linkage, t=n_clusters, criterion="maxclust")

    # Organize clusters
    clusters = {}
    for dim_idx, cluster_id in enumerate(labels):
        cid = int(cluster_id)
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(dim_idx)

    logger.info(f"Formed {len(clusters)} clusters from 768 dimensions")
    for cid in sorted(clusters):
        logger.info(f"  Cluster {cid}: {len(clusters[cid])} dims")

    return {
        "clusters": {str(k): v for k, v in clusters.items()},
        "n_clusters": len(clusters),
    }


def find_top_prompts_per_cluster(
    activations: np.ndarray,
    corpus: list[dict],
    clusters: dict,
    top_k: int = 20,
) -> dict:
    """For each cluster, find top-K prompts that maximally activate that cluster's dims."""
    logger.info("Finding top prompts per cluster...")
    cluster_tops = {}

    for cid, dim_indices in clusters.items():
        # Mean activation across cluster dimensions for each prompt
        cluster_act = activations[:, dim_indices].mean(axis=1)  # [N]
        top_indices = np.argsort(cluster_act)[-top_k:][::-1]

        cluster_tops[cid] = [
            {
                "rank": rank,
                "index": int(idx),
                "text": corpus[idx]["text"],
                "category": corpus[idx].get("category", ""),
                "mean_activation": float(cluster_act[idx]),
            }
            for rank, idx in enumerate(top_indices)
        ]

    return cluster_tops


def compute_probing_analysis(
    activations: np.ndarray,
    index: list[dict],
) -> dict:
    """Per-category mean activation per dimension for probing entries."""
    logger.info("Computing probing analysis...")

    # Group indices by category
    cat_indices = {}
    for i, entry in enumerate(index):
        cat = entry["category"]
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)

    # Mean activation per dim per category
    probing = {}
    for cat, indices in cat_indices.items():
        if len(indices) < 5:
            continue
        cat_acts = activations[indices]  # [n_cat, 768]
        mean_acts = cat_acts.mean(axis=0)  # [768]

        # Find most discriminative dims (highest |mean - global_mean|)
        global_mean = activations.mean(axis=0)
        diff = np.abs(mean_acts - global_mean)
        top_dims = np.argsort(diff)[-20:][::-1]

        probing[cat] = {
            "n_prompts": len(indices),
            "top_discriminative_dims": [
                {
                    "dim": int(d),
                    "category_mean": float(mean_acts[d]),
                    "global_mean": float(global_mean[d]),
                    "abs_diff": float(diff[d]),
                }
                for d in top_dims
            ],
        }

    logger.info(f"Probing analysis: {len(probing)} categories")
    return probing


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    activations, corpus, index = load_data()

    # 1. Per-dim statistics
    dim_stats = compute_dim_statistics(activations)

    # 2. Dimension clustering
    clustering = cluster_dimensions(activations, n_clusters=32)

    # 3. Top prompts per cluster
    cluster_tops = find_top_prompts_per_cluster(
        activations, corpus, clustering["clusters"]
    )

    # 4. Probing analysis
    probing = compute_probing_analysis(activations, index)

    # Save atlas
    atlas = {
        "dim_statistics": dim_stats,
        "clustering": clustering,
        "cluster_top_prompts": cluster_tops,
        "probing_analysis": probing,
    }

    with open(DIMENSION_ATLAS_PATH, "w") as f:
        json.dump(atlas, f, indent=2)

    logger.info(f"Dimension atlas saved to {DIMENSION_ATLAS_PATH}")


if __name__ == "__main__":
    main()
