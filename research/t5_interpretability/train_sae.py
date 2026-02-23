#!/usr/bin/env python3
"""
Phase 4: TopK Sparse Autoencoder Training

Minimal TopK SAE (Gao et al. 2024 / OpenAI recipe):
- Input: activations_pooled.pt [N × 768]
- Expansion: 16× (768 → 12,288 features)
- TopK: k=64 (each input activates 64 of 12,288 features)
- Loss: MSE reconstruction (TopK handles sparsity, no L1 needed)
- Decoder weight normalization after each step

VRAM: <200MB. Time: ~25 seconds.

Usage: venv/bin/python research/t5_interpretability/train_sae.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SAE_D_MODEL, SAE_N_FEATURES, SAE_K,
    SAE_LR, SAE_BATCH_SIZE, SAE_EPOCHS, SAE_WEIGHT_DECAY,
    ACTIVATIONS_POOLED_PATH, SAE_WEIGHTS_PATH, SAE_TRAINING_LOG_PATH,
    DATA_DIR,
)


class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder following Anthropic/OpenAI recipe.

    Architecture:
        x → center(x) → encoder → topk → relu → decoder → uncenter
    """

    def __init__(self, d_model: int = 768, n_features: int = 12288, k: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        # Pre-encoder bias (learned data centering)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))

        # Encoder: d_model → n_features
        self.encoder = nn.Linear(d_model, n_features)

        # Decoder: n_features → d_model (no bias — pre_bias handles centering)
        self.decoder = nn.Linear(n_features, d_model, bias=False)

        # Initialize decoder columns to unit norm (Anthropic recipe)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, d_model]
        Returns:
            x_hat: [batch, d_model] — reconstruction
            sparse: [batch, n_features] — sparse feature activations
        """
        # Center
        centered = x - self.pre_bias

        # Encode
        z = self.encoder(centered)  # [batch, n_features]

        # TopK sparsity: keep only top-k activations, zero the rest
        topk_vals, topk_idx = z.topk(self.k, dim=-1)  # [batch, k]
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, topk_idx, F.relu(topk_vals))

        # Decode + uncenter
        x_hat = self.decoder(sparse) + self.pre_bias

        return x_hat, sparse

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode only (for analysis). Returns sparse activations."""
        centered = x - self.pre_bias
        z = self.encoder(centered)
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, topk_idx, F.relu(topk_vals))
        return sparse

    @torch.no_grad()
    def renorm_decoder(self):
        """Normalize decoder columns to unit norm (after each optimizer step)."""
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


def compute_metrics(sae: TopKSAE, activations: torch.Tensor, batch_size: int = 4096) -> dict:
    """Compute validation metrics on full dataset."""
    sae.eval()
    total_mse = 0.0
    total_l0 = 0.0
    feature_counts = torch.zeros(sae.n_features, device="cpu")
    n = activations.shape[0]

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = activations[i:i + batch_size].cuda()
            x_hat, sparse = sae(batch)

            mse = F.mse_loss(x_hat, batch, reduction="sum").item()
            total_mse += mse

            # L0: number of non-zero features per sample
            active = (sparse > 0).float()
            total_l0 += active.sum(dim=1).sum().item()

            # Track which features fire
            feature_counts += active.sum(dim=0).cpu()

    avg_mse = total_mse / n
    avg_l0 = total_l0 / n
    dead_features = (feature_counts == 0).sum().item()
    dead_pct = dead_features / sae.n_features * 100

    return {
        "mse": avg_mse,
        "l0": avg_l0,
        "dead_features": dead_features,
        "dead_features_pct": dead_pct,
    }


def train(sae: TopKSAE, activations: torch.Tensor) -> list[dict]:
    """Train SAE on activations. Returns training log."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR, weight_decay=SAE_WEIGHT_DECAY)
    n = activations.shape[0]
    n_batches = (n + SAE_BATCH_SIZE - 1) // SAE_BATCH_SIZE
    training_log = []

    logger.info(f"Training: {n} samples, {n_batches} batches/epoch, {SAE_EPOCHS} epochs")
    logger.info(f"SAE: {SAE_D_MODEL} → {SAE_N_FEATURES} (k={SAE_K})")

    start_time = time.time()

    for epoch in range(SAE_EPOCHS):
        sae.train()
        epoch_loss = 0.0

        # Shuffle
        perm = torch.randperm(n)

        for batch_idx in range(n_batches):
            start = batch_idx * SAE_BATCH_SIZE
            end = min(start + SAE_BATCH_SIZE, n)
            indices = perm[start:end]
            batch = activations[indices].cuda()

            x_hat, sparse = sae(batch)
            loss = F.mse_loss(x_hat, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Renorm decoder after each step
            sae.renorm_decoder()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / n_batches

        # Log every 10 epochs + first + last
        if epoch % 10 == 0 or epoch == SAE_EPOCHS - 1:
            metrics = compute_metrics(sae, activations)
            elapsed = time.time() - start_time

            log_entry = {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "elapsed_sec": round(elapsed, 1),
                **metrics,
            }
            training_log.append(log_entry)

            logger.info(
                f"Epoch {epoch:3d} | loss={avg_epoch_loss:.6f} | "
                f"MSE={metrics['mse']:.6f} | L0={metrics['l0']:.1f} | "
                f"dead={metrics['dead_features']} ({metrics['dead_features_pct']:.1f}%) | "
                f"{elapsed:.1f}s"
            )

    return training_log


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load activations
    logger.info(f"Loading activations from {ACTIVATIONS_POOLED_PATH}...")
    activations = torch.load(ACTIVATIONS_POOLED_PATH, weights_only=True).float()
    logger.info(f"Activations: {activations.shape}")

    # Create SAE
    sae = TopKSAE(d_model=SAE_D_MODEL, n_features=SAE_N_FEATURES, k=SAE_K).cuda()
    n_params = sum(p.numel() for p in sae.parameters())
    logger.info(f"SAE parameters: {n_params:,} ({n_params * 4 / 1024 / 1024:.1f} MB fp32)")

    # Train
    training_log = train(sae, activations)

    # Save weights
    torch.save(sae.state_dict(), SAE_WEIGHTS_PATH)
    logger.info(f"SAE weights saved to {SAE_WEIGHTS_PATH}")

    # Save training log
    with open(SAE_TRAINING_LOG_PATH, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Training log saved to {SAE_TRAINING_LOG_PATH}")

    # Final metrics summary
    final = training_log[-1]
    logger.info(f"\n=== Final Metrics ===")
    logger.info(f"MSE: {final['mse']:.6f} (target < 0.05)")
    logger.info(f"L0:  {final['l0']:.1f} (target ≈ {SAE_K})")
    logger.info(f"Dead features: {final['dead_features']} / {SAE_N_FEATURES} ({final['dead_features_pct']:.1f}%, target < 5%)")


if __name__ == "__main__":
    main()
