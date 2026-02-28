#!/usr/bin/env python3
"""
Statistical Sonification Test

Generates N samples each for two contrasting text prompts (e.g. "sound rhythmic"
vs "sound sustained"), extracts acoustic features via librosa, and runs
statistical tests to determine if the distributions are separable.

This addresses the single-seed problem: one generation tells us nothing,
100 generations per condition give us statistical power.

Usage: venv/bin/python research/t5_interpretability/statistical_sonification_test.py
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
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GPU_SERVICE_URL, DATA_DIR

# ── Experiment Config ──────────────────────────────────────────────────────────

PROMPT_A = "sound rhythmic"
PROMPT_B = "sound sustained"
LABEL_A = "rhythmic"
LABEL_B = "sustained"

N_SAMPLES = 100          # per condition
DURATION = 5.0
STEPS = 100
CFG = 7.0

OUTPUT_DIR = DATA_DIR / "statistical_test"


def generate_audio(prompt: str, seed: int) -> bytes | None:
    """Generate audio via GPU service text prompt endpoint."""
    url = f"{GPU_SERVICE_URL}/api/stable_audio/generate"
    payload = {
        "prompt": prompt,
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

        # Skip silent files
        if np.abs(y).max() < 1e-6:
            return None

        features = {}

        # 1. Onset density (onsets per second) — key metric for rhythmic vs sustained
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        features["onset_density"] = len(onset_frames) / duration if duration > 0 else 0

        # 2. Spectral centroid (brightness, Hz)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(cent))
        features["spectral_centroid_std"] = float(np.std(cent))

        # 3. RMS energy
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        # 4. Spectral flatness (noise-like vs tonal, 0=tonal, 1=noise)
        flat = librosa.feature.spectral_flatness(y=y)
        features["spectral_flatness_mean"] = float(np.mean(flat))

        # 5. Zero crossing rate (roughness indicator)
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(np.mean(zcr))

        # 6. Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(np.atleast_1d(tempo)[0])

        # 7. Spectral bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = float(np.mean(bw))

        # 8. Spectral flux (rate of spectral change)
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

    # Get all feature names from first sample
    all_keys = sorted(features_a[0].keys())

    for key in all_keys:
        vals_a = np.array([f[key] for f in features_a])
        vals_b = np.array([f[key] for f in features_b])

        # Welch's t-test (unequal variance)
        t_stat, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((vals_a.std()**2 + vals_b.std()**2) / 2)
        cohens_d = (vals_a.mean() - vals_b.mean()) / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney U (non-parametric)
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


def generate_report(results: dict, n_a: int, n_b: int) -> str:
    """Generate markdown report."""
    lines = [
        f"# Statistical Sonification Test: {LABEL_A} vs {LABEL_B}\n",
        f"- Prompt A: `{PROMPT_A}` (N={n_a})",
        f"- Prompt B: `{PROMPT_B}` (N={n_b})",
        f"- Duration: {DURATION}s, Steps: {STEPS}, CFG: {CFG}",
        f"- Seeds: 0–{N_SAMPLES-1} per condition\n",
        "## Results\n",
        f"| Feature | {LABEL_A} (mean±std) | {LABEL_B} (mean±std) | Cohen's d | p-value | Sig? |",
        "|---|---|---|---|---|---|",
    ]

    # Sort by effect size (most interesting first)
    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True)

    for key, r in sorted_results:
        sig = "***" if r["significant_001"] else ("**" if r["significant_01"] else ("*" if r["significant_05"] else ""))
        lines.append(
            f"| {key} | {r['mean_a']:.4f}±{r['std_a']:.4f} | "
            f"{r['mean_b']:.4f}±{r['std_b']:.4f} | "
            f"{r['cohens_d']:+.3f} | {r['p_value']:.2e} | {sig} |"
        )

    lines.append("")
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05\n")

    # Interpretation
    lines.append("## Interpretation\n")
    significant = [(k, r) for k, r in sorted_results if r["significant_001"]]
    if significant:
        lines.append(f"**{len(significant)} features** show highly significant differences (p<0.001):\n")
        for key, r in significant:
            direction = LABEL_A if r["cohens_d"] > 0 else LABEL_B
            lines.append(f"- **{key}**: d={r['cohens_d']:+.3f} → higher in `{direction}`")
    else:
        lines.append("No features show significant differences at p<0.001.")
        weak = [(k, r) for k, r in sorted_results if r["significant_05"]]
        if weak:
            lines.append(f"\n{len(weak)} features significant at p<0.05 (weak).")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dir_a = OUTPUT_DIR / LABEL_A
    dir_b = OUTPUT_DIR / LABEL_B
    dir_a.mkdir(exist_ok=True)
    dir_b.mkdir(exist_ok=True)

    # Phase 1: Generate audio
    for label, prompt, out_dir in [(LABEL_A, PROMPT_A, dir_a), (LABEL_B, PROMPT_B, dir_b)]:
        existing = len(list(out_dir.glob("*.wav")))
        if existing >= N_SAMPLES:
            logger.info(f"{label}: {existing} files already exist, skipping generation")
            continue

        logger.info(f"Generating {N_SAMPLES} samples for '{prompt}'...")
        start = time.time()

        for seed in range(N_SAMPLES):
            out_path = out_dir / f"seed_{seed:03d}.wav"
            if out_path.exists():
                continue

            audio = generate_audio(prompt, seed)
            if audio:
                with open(out_path, "wb") as f:
                    f.write(audio)

            if (seed + 1) % 20 == 0:
                elapsed = time.time() - start
                rate = (seed + 1) / elapsed
                logger.info(f"  {seed+1}/{N_SAMPLES} ({rate:.1f} samples/sec)")

    # Phase 2: Extract features
    logger.info("Extracting acoustic features...")
    features_a = []
    features_b = []

    for label, out_dir, feature_list in [
        (LABEL_A, dir_a, features_a),
        (LABEL_B, dir_b, features_b),
    ]:
        for wav_path in sorted(out_dir.glob("*.wav")):
            feats = extract_features(wav_path)
            if feats:
                feature_list.append(feats)

    logger.info(f"Features extracted: {LABEL_A}={len(features_a)}, {LABEL_B}={len(features_b)}")

    if len(features_a) < 10 or len(features_b) < 10:
        logger.error("Too few samples for statistical analysis")
        return

    # Phase 3: Statistical tests
    logger.info("Running statistical tests...")
    results = run_statistical_tests(features_a, features_b)

    # Save raw results
    with open(OUTPUT_DIR / "statistical_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate and save report
    report = generate_report(results, len(features_a), len(features_b))
    report_path = OUTPUT_DIR / "statistical_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Report saved to {report_path}")
    logger.info(f"{'='*60}")

    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True)
    for key, r in sorted_results[:5]:
        sig = "***" if r["significant_001"] else ("**" if r["significant_01"] else ("*" if r["significant_05"] else "n.s."))
        logger.info(
            f"  {key:>30}: d={r['cohens_d']:+.3f}, p={r['p_value']:.2e} {sig}"
        )


if __name__ == "__main__":
    main()
