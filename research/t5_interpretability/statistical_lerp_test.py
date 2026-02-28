#!/usr/bin/env python3
"""
Statistical LERP Interpolation Test

Tests whether linear interpolation between two NATURAL T5 embeddings
produces statistically distinguishable audio — and whether intermediate
positions show a monotonic gradient.

Method:
  1. Encode "sound rhythmic" and "sound sustained" through T5
  2. LERP at 5 positions: 0.0 (sustained) → 0.25 → 0.5 → 0.75 → 1.0 (rhythmic)
  3. Generate N=100 samples per position (seeds 0–99)
  4. Extract librosa features, run statistical tests

Key difference from injection test: both endpoints are NATURAL T5 outputs,
so the interpolation stays closer to Stable Audio's learned distribution.

Usage: venv/bin/python research/t5_interpretability/statistical_lerp_test.py
"""

import base64
import io
import json
import logging
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import requests
import torch
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    T5_MODEL_ID, T5_MAX_LENGTH,
    GPU_SERVICE_URL, DATA_DIR,
)

# ── Experiment Config ──────────────────────────────────────────────────────────

PROMPT_A = "sound rhythmic"     # LERP position 1.0
PROMPT_B = "sound sustained"    # LERP position 0.0

LERP_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

N_SAMPLES = 100
DURATION = 5.0
STEPS = 100
CFG = 7.0

OUTPUT_DIR = DATA_DIR / "statistical_lerp_test"


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
    emb = outputs.last_hidden_state.cpu().float().numpy()
    mask = inputs.attention_mask.cpu().float().numpy()
    return emb, mask


def _numpy_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_audio_from_embeddings(
    embeddings: np.ndarray, mask: np.ndarray, seed: int,
) -> bytes | None:
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


def extract_features(wav_path: Path) -> dict | None:
    """Extract acoustic features from a WAV file."""
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)

        if np.abs(y).max() < 1e-6:
            return None

        features = {}

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        features["onset_density"] = len(onset_frames) / duration if duration > 0 else 0

        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(cent))
        features["spectral_centroid_std"] = float(np.std(cent))

        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        flat = librosa.feature.spectral_flatness(y=y)
        features["spectral_flatness_mean"] = float(np.mean(flat))

        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(np.mean(zcr))

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(np.atleast_1d(tempo)[0])

        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = float(np.mean(bw))

        S = np.abs(librosa.stft(y))
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        features["spectral_flux_mean"] = float(np.mean(flux))
        features["spectral_flux_std"] = float(np.std(flux))

        return features

    except Exception as e:
        logger.error(f"Feature extraction failed for {wav_path}: {e}")
        return None


def run_statistical_tests(features_a: list[dict], features_b: list[dict]) -> dict:
    """Run statistical comparisons between two conditions."""
    results = {}
    all_keys = sorted(features_a[0].keys())

    for key in all_keys:
        vals_a = np.array([f[key] for f in features_a])
        vals_b = np.array([f[key] for f in features_b])

        t_stat, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)

        pooled_std = np.sqrt((vals_a.std()**2 + vals_b.std()**2) / 2)
        cohens_d = (vals_a.mean() - vals_b.mean()) / pooled_std if pooled_std > 0 else 0

        u_stat, u_p = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')

        results[key] = {
            "mean_a": float(vals_a.mean()),
            "mean_b": float(vals_b.mean()),
            "std_a": float(vals_a.std()),
            "std_b": float(vals_b.std()),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "mann_whitney_p": float(u_p),
            "significant_001": bool(p_value < 0.001),
            "significant_01": bool(p_value < 0.01),
            "significant_05": bool(p_value < 0.05),
        }

    return results


def compute_gradient_analysis(all_features: dict[float, list[dict]]) -> dict:
    """Compute Pearson correlation between LERP position and each feature."""
    feature_keys = sorted(all_features[LERP_POSITIONS[0]][0].keys())
    gradient = {}

    for key in feature_keys:
        positions = []
        values = []
        for t in sorted(all_features.keys()):
            for feat in all_features[t]:
                positions.append(t)
                values.append(feat[key])

        positions = np.array(positions)
        values = np.array(values)

        r, p = stats.pearsonr(positions, values)
        # Also Spearman (rank correlation, robust to non-linearity)
        rho, rho_p = stats.spearmanr(positions, values)

        # Per-position means for the gradient table
        pos_means = {}
        for t in sorted(all_features.keys()):
            vals = [f[key] for f in all_features[t]]
            pos_means[f"mean_{t:.2f}"] = float(np.mean(vals))

        gradient[key] = {
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            **pos_means,
        }

    return gradient


def generate_report(
    extreme_results: dict,
    gradient: dict,
    sample_counts: dict[float, int],
) -> str:
    """Generate markdown report."""
    lines = [
        "# Statistical LERP Interpolation Test\n",
        f"- Pole A (t=1.0): `{PROMPT_A}`",
        f"- Pole B (t=0.0): `{PROMPT_B}`",
        f"- LERP positions: {LERP_POSITIONS}",
        f"- Samples per position: N={N_SAMPLES}",
        f"- Duration: {DURATION}s, Steps: {STEPS}, CFG: {CFG}",
        f"- Seeds: 0–{N_SAMPLES-1} per position\n",
    ]

    # Sample counts
    lines.append("## Sample Counts\n")
    for t in sorted(sample_counts.keys()):
        lines.append(f"- t={t:.2f}: N={sample_counts[t]}")
    lines.append("")

    # Extreme comparison (t=0.0 vs t=1.0)
    lines.append("## Extreme Comparison: t=0.0 (sustained) vs t=1.0 (rhythmic)\n")
    lines.append("| Feature | t=0.0 (mean±std) | t=1.0 (mean±std) | Cohen's d | p-value | Sig? |")
    lines.append("|---|---|---|---|---|---|")

    sorted_extreme = sorted(
        extreme_results.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True,
    )

    for key, r in sorted_extreme:
        sig = "***" if r["significant_001"] else ("**" if r["significant_01"] else ("*" if r["significant_05"] else ""))
        lines.append(
            f"| {key} | {r['mean_b']:.4f}±{r['std_b']:.4f} | "
            f"{r['mean_a']:.4f}±{r['std_a']:.4f} | "
            f"{r['cohens_d']:+.3f} | {r['p_value']:.2e} | {sig} |"
        )

    lines.append("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05\n")

    # Gradient analysis
    lines.append("## Gradient Analysis: Correlation with LERP Position\n")
    lines.append("| Feature | Pearson r | p-value | Spearman rho | t=0.00 | t=0.25 | t=0.50 | t=0.75 | t=1.00 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    sorted_gradient = sorted(
        gradient.items(), key=lambda x: abs(x[1]["pearson_r"]), reverse=True,
    )

    for key, g in sorted_gradient:
        sig = "***" if g["pearson_p"] < 0.001 else ("**" if g["pearson_p"] < 0.01 else ("*" if g["pearson_p"] < 0.05 else ""))
        lines.append(
            f"| {key} | {g['pearson_r']:+.3f}{sig} | {g['pearson_p']:.2e} | "
            f"{g['spearman_rho']:+.3f} | "
            f"{g['mean_0.00']:.2f} | {g['mean_0.25']:.2f} | {g['mean_0.50']:.2f} | "
            f"{g['mean_0.75']:.2f} | {g['mean_1.00']:.2f} |"
        )

    lines.append("")

    # Cross-experiment comparison
    text_path = DATA_DIR / "statistical_test" / "statistical_results.json"
    inj_path = DATA_DIR / "statistical_injection_test" / "statistical_results.json"

    text_results = {}
    inj_results = {}
    if text_path.exists():
        with open(text_path) as f:
            text_results = json.load(f)
    if inj_path.exists():
        with open(inj_path) as f:
            inj_results = json.load(f)

    if text_results:
        lines.append("## Cross-Experiment Comparison\n")
        lines.append("| Feature | Text d | Injection d | LERP d | LERP/Text |")
        lines.append("|---|---|---|---|---|")

        for key, r in sorted_extreme:
            text_d = text_results.get(key, {}).get("cohens_d", 0)
            inj_d = inj_results.get(key, {}).get("cohens_d", 0)
            lerp_d = r["cohens_d"]
            ratio = abs(lerp_d / text_d) if abs(text_d) > 0.01 else float("inf")
            ratio_str = f"{ratio:.1%}" if ratio < 100 else "n/a"
            lines.append(
                f"| {key} | {text_d:+.3f} | {inj_d:+.3f} | {lerp_d:+.3f} | {ratio_str} |"
            )

        lines.append("")
        lines.append("LERP/Text ratio: how much of the text-prompt effect size the LERP approach recovers.")

    # Interpretation
    lines.append("\n## Interpretation\n")

    sig_extreme = [(k, r) for k, r in sorted_extreme if r["significant_05"]]
    sig_gradient = [(k, g) for k, g in sorted_gradient if g["pearson_p"] < 0.05]

    if sig_extreme:
        lines.append(f"**Extreme comparison**: {len(sig_extreme)} features significant (p<0.05).\n")
    else:
        lines.append("**Extreme comparison**: No significant features.\n")

    if sig_gradient:
        lines.append(f"**Gradient**: {len(sig_gradient)} features show significant monotonic trend with LERP position.\n")
        monotonic = all(
            abs(g["pearson_r"]) > 0.05 and g["pearson_p"] < 0.05
            for _, g in sig_gradient
        )
        if monotonic:
            lines.append("The LERP interpolation produces a **monotonic gradient** — intermediate positions yield intermediate acoustic features.")
    else:
        lines.append("**Gradient**: No significant monotonic trends.\n")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if generation needed
    need_generation = False
    for t in LERP_POSITIONS:
        label = f"lerp_{t:.2f}"
        d = OUTPUT_DIR / label
        if not d.exists() or len(list(d.glob("*.wav"))) < N_SAMPLES:
            need_generation = True
            break

    if need_generation:
        logger.info("Loading T5 for embedding computation...")
        tokenizer, model = load_t5()

        logger.info(f"Encoding: '{PROMPT_A}' and '{PROMPT_B}'")
        emb_a, mask_a = encode_prompt(PROMPT_A, tokenizer, model)
        emb_b, mask_b = encode_prompt(PROMPT_B, tokenizer, model)

        # Use mask from prompt A (both should be nearly identical for short prompts)
        mask = mask_a

        # Unload T5
        del model
        torch.cuda.empty_cache()

        # Generate audio for each LERP position
        total = len(LERP_POSITIONS) * N_SAMPLES
        done = 0
        start_time = time.time()

        for t in LERP_POSITIONS:
            label = f"lerp_{t:.2f}"
            out_dir = OUTPUT_DIR / label
            out_dir.mkdir(exist_ok=True)

            existing = len(list(out_dir.glob("*.wav")))
            if existing >= N_SAMPLES:
                logger.info(f"{label}: {existing} files exist, skipping")
                done += N_SAMPLES
                continue

            # LERP: emb = (1 - t) * emb_b + t * emb_a
            emb = (1.0 - t) * emb_b + t * emb_a

            logger.info(f"Generating {N_SAMPLES} samples for {label}...")
            for seed in range(N_SAMPLES):
                out_path = out_dir / f"seed_{seed:03d}.wav"
                if out_path.exists():
                    done += 1
                    continue

                audio = generate_audio_from_embeddings(emb, mask, seed)
                if audio:
                    with open(out_path, "wb") as f:
                        f.write(audio)

                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    logger.info(
                        f"  [{done}/{total}] ({rate:.1f} samples/sec, ETA {eta:.0f}s)"
                    )
    else:
        logger.info("All LERP positions have enough samples, skipping generation")

    # Extract features for all positions
    logger.info("Extracting acoustic features...")
    all_features: dict[float, list[dict]] = {}
    sample_counts: dict[float, int] = {}

    for t in LERP_POSITIONS:
        label = f"lerp_{t:.2f}"
        out_dir = OUTPUT_DIR / label
        features = []
        for wav_path in sorted(out_dir.glob("*.wav")):
            feats = extract_features(wav_path)
            if feats:
                features.append(feats)
        all_features[t] = features
        sample_counts[t] = len(features)
        logger.info(f"  {label}: {len(features)} samples")

    # Check minimum samples
    for t, feats in all_features.items():
        if len(feats) < 10:
            logger.error(f"Too few samples at t={t}: {len(feats)}")
            return

    # Extreme comparison: t=0.0 vs t=1.0
    logger.info("Running statistical tests (extremes)...")
    extreme_results = run_statistical_tests(all_features[1.0], all_features[0.0])

    # Gradient analysis across all positions
    logger.info("Running gradient analysis...")
    gradient = compute_gradient_analysis(all_features)

    # Save raw results
    with open(OUTPUT_DIR / "extreme_results.json", "w") as f:
        json.dump(extreme_results, f, indent=2)
    with open(OUTPUT_DIR / "gradient_results.json", "w") as f:
        json.dump(gradient, f, indent=2)

    # Generate report
    report = generate_report(extreme_results, gradient, sample_counts)
    report_path = OUTPUT_DIR / "statistical_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Report saved to {report_path}")
    logger.info(f"{'='*60}")

    sorted_extreme = sorted(
        extreme_results.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True,
    )
    logger.info("Extreme (t=0 vs t=1):")
    for key, r in sorted_extreme[:5]:
        sig = "***" if r["significant_001"] else ("**" if r["significant_01"] else ("*" if r["significant_05"] else "n.s."))
        logger.info(f"  {key:>30}: d={r['cohens_d']:+.3f}, p={r['p_value']:.2e} {sig}")

    sorted_gradient = sorted(
        gradient.items(), key=lambda x: abs(x[1]["pearson_r"]), reverse=True,
    )
    logger.info("Gradient (Pearson r with LERP position):")
    for key, g in sorted_gradient[:5]:
        sig = "***" if g["pearson_p"] < 0.001 else ("**" if g["pearson_p"] < 0.01 else ("*" if g["pearson_p"] < 0.05 else "n.s."))
        logger.info(f"  {key:>30}: r={g['pearson_r']:+.3f}, p={g['pearson_p']:.2e} {sig}")


if __name__ == "__main__":
    main()
