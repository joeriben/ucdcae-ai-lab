#!/usr/bin/env python3
"""
Phase 4: Sparse Autoencoder Training

ReLU + L1 SAE (Anthropic "Towards Monosemanticity" recipe):
- Input: activations_pooled.pt [N × 768]
- Expansion: 8× (768 → 6,144 features)
- Sparsity: L1 penalty on activations (target L0 ≈ 64)
- Loss: MSE reconstruction + L1 with warmup
- Decoder weight normalization after each step

TopK was attempted first but suffered catastrophic dead feature collapse
(99%+ dead) due to winner-take-all gradient dynamics. L1 provides continuous
gradient to all features, preventing collapse.

VRAM: <200MB. Time: ~3 minutes.

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
    SAE_L1_COEFF, SAE_L1_WARMUP_EPOCHS,
    ACTIVATIONS_POOLED_PATH, SAE_WEIGHTS_PATH, SAE_TRAINING_LOG_PATH,
    DATA_DIR,
)


class TopKSAE(nn.Module):
    """
    Sparse Autoencoder with ReLU + L1 sparsity.

    Architecture:
        x → center(x) → encoder → ReLU → decoder → uncenter

    Named TopKSAE for backward compatibility with analysis scripts.
    The encode() method applies a TopK selection for clean sparse codes
    at inference time, while training uses L1 for gradient health.
    """

    def __init__(self, d_model: int = 768, n_features: int = 6144, k: int = 64):
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

        # Kaiming init for encoder (important for L1 training)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        # Initialize decoder columns to unit norm (Anthropic recipe)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x: torch.Tensor):
        """
        Training forward pass: ReLU activation (no TopK).
        L1 loss applied externally to the sparse activations.

        Args:
            x: [batch, d_model]
        Returns:
            x_hat: [batch, d_model] — reconstruction
            features: [batch, n_features] — ReLU activations (dense-ish)
        """
        centered = x - self.pre_bias
        z = self.encoder(centered)  # [batch, n_features]
        features = F.relu(z)
        x_hat = self.decoder(features) + self.pre_bias
        return x_hat, features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference encoding with TopK selection for clean sparse output.
        Used by analysis scripts for interpretable feature activations.
        """
        centered = x - self.pre_bias
        z = self.encoder(centered)
        features = F.relu(z)

        # Apply TopK for clean sparse codes at inference
        topk_vals, topk_idx = features.topk(self.k, dim=-1)
        sparse = torch.zeros_like(features)
        sparse.scatter_(-1, topk_idx, topk_vals)
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
            x_hat, features = sae(batch)

            # Use TopK encoding for metrics (matches inference behavior)
            sparse = sae.encode(batch)

            mse = F.mse_loss(x_hat, batch, reduction="sum").item()
            total_mse += mse

            # L0: number of non-zero features per sample (after TopK)
            active = (sparse > 0).float()
            total_l0 += active.sum(dim=1).sum().item()

            # Track which features fire (using ReLU output, not TopK)
            alive = (features > 0).float()
            feature_counts += alive.sum(dim=0).cpu()

    avg_mse = total_mse / n
    avg_l0 = total_l0 / n
    mse_per_dim = avg_mse / sae.d_model

    # Dead = never fires on any sample (using ReLU, not TopK)
    dead_features = (feature_counts == 0).sum().item()
    dead_pct = dead_features / sae.n_features * 100

    # Mean L0 from ReLU (before TopK)
    relu_l0 = (feature_counts > 0).sum().item()  # how many features fire at all

    return {
        "mse": avg_mse,
        "mse_per_dim": mse_per_dim,
        "l0": avg_l0,
        "dead_features": dead_features,
        "dead_features_pct": dead_pct,
        "alive_features": sae.n_features - dead_features,
    }


def train(sae: TopKSAE, activations: torch.Tensor) -> list[dict]:
    """Train SAE with L1 sparsity + MSE reconstruction."""
    # Initialize pre_bias to data mean
    data_mean = activations.mean(dim=0).cuda()
    sae.pre_bias.data.copy_(data_mean)
    logger.info(f"Initialized pre_bias to data mean (norm={data_mean.norm():.4f})")

    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR, weight_decay=SAE_WEIGHT_DECAY)
    n = activations.shape[0]
    n_batches = (n + SAE_BATCH_SIZE - 1) // SAE_BATCH_SIZE
    training_log = []

    logger.info(f"Training: {n} samples, {n_batches} batches/epoch, {SAE_EPOCHS} epochs")
    logger.info(f"SAE: {SAE_D_MODEL} → {SAE_N_FEATURES} (ReLU + L1)")
    logger.info(f"L1 coeff: {SAE_L1_COEFF}, warmup: {SAE_L1_WARMUP_EPOCHS} epochs")

    start_time = time.time()

    for epoch in range(SAE_EPOCHS):
        sae.train()
        epoch_mse_loss = 0.0
        epoch_l1_loss = 0.0

        # L1 coefficient scheduling: warmup then full
        if epoch < SAE_L1_WARMUP_EPOCHS:
            l1_coeff = SAE_L1_COEFF * (epoch / SAE_L1_WARMUP_EPOCHS)
        else:
            l1_coeff = SAE_L1_COEFF

        # Shuffle
        perm = torch.randperm(n)

        for batch_idx in range(n_batches):
            start = batch_idx * SAE_BATCH_SIZE
            end = min(start + SAE_BATCH_SIZE, n)
            indices = perm[start:end]
            batch = activations[indices].cuda()

            x_hat, features = sae(batch)

            mse_loss = F.mse_loss(x_hat, batch)
            l1_loss = features.abs().mean()
            loss = mse_loss + l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Renorm decoder after each step
            sae.renorm_decoder()

            epoch_mse_loss += mse_loss.item()
            epoch_l1_loss += l1_loss.item()

        avg_mse = epoch_mse_loss / n_batches
        avg_l1 = epoch_l1_loss / n_batches

        # Log every 10 epochs + first + last
        if epoch % 10 == 0 or epoch == SAE_EPOCHS - 1:
            metrics = compute_metrics(sae, activations)
            elapsed = time.time() - start_time

            log_entry = {
                "epoch": epoch,
                "train_mse": avg_mse,
                "train_l1": avg_l1,
                "l1_coeff": l1_coeff,
                "elapsed_sec": round(elapsed, 1),
                **metrics,
            }
            training_log.append(log_entry)

            logger.info(
                f"Epoch {epoch:3d} | mse={avg_mse:.6f} | l1={avg_l1:.4f} (λ={l1_coeff:.4f}) | "
                f"L0={metrics['l0']:.1f} | alive={metrics['alive_features']}/{SAE_N_FEATURES} | "
                f"dead={metrics['dead_features_pct']:.1f}% | {elapsed:.1f}s"
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
    logger.info(f"MSE (per sample): {final['mse']:.4f}")
    logger.info(f"MSE (per dim):    {final['mse_per_dim']:.6f}")
    logger.info(f"L0 (TopK@{SAE_K}):    {final['l0']:.1f}")
    logger.info(f"Alive features:   {final['alive_features']} / {SAE_N_FEATURES} ({100-final['dead_features_pct']:.1f}%)")


if __name__ == "__main__":
    main()
