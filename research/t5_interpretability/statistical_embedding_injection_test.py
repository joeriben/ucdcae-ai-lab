#!/usr/bin/env python3
"""
Statistical Embedding Injection Test

Same design as statistical_sonification_test.py (N=100 per condition, librosa
features, Welch's t-test + Cohen's d), but uses EMBEDDING INJECTION instead
of text prompts.

Method:
  1. Encode "sound", "sound rhythmic", "sound sustained" through T5
  2. Compute difference vector: diff = emb("sound rhythmic") - emb("sound sustained")
  3. Condition A: neutral_emb + 1.0 * diff  (rhythmic direction)
  4. Condition B: neutral_emb - 1.0 * diff  (sustained direction)
  5. Generate 100 samples per condition (seeds 0–99)
  6. Extract librosa features, run statistical tests

This directly tests whether embedding-space manipulation produces
statistically distinguishable audio — the same question the binary contrast
experiment asked qualitatively, now with statistical power.

Usage: venv/bin/python research/t5_interpretability/statistical_embedding_injection_test.py
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

NEUTRAL_PROMPT = "sound"
PROMPT_A = "sound rhythmic"
PROMPT_B = "sound sustained"
LABEL_A = "injection_rhythmic"
LABEL_B = "injection_sustained"

INJECTION_STRENGTH = 1.0  # same as binary contrast default

N_SAMPLES = 100
DURATION = 5.0
STEPS = 100
CFG = 7.0

OUTPUT_DIR = DATA_DIR / "statistical_injection_test"


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
    emb = outputs.last_hidden_state.cpu().float().numpy()  # [1, seq, 768]
    mask = inputs.attention_mask.cpu().float().numpy()
    return emb, mask


def _numpy_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_audio_from_embeddings(
    embeddings: np.ndarray, mask: np.ndarray, seed: int
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

        # 1. Onset density
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        features["onset_density"] = len(onset_frames) / duration if duration > 0 else 0

        # 2. Spectral centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(cent))
        features["spectral_centroid_std"] = float(np.std(cent))

        # 3. RMS energy
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        # 4. Spectral flatness
        flat = librosa.feature.spectral_flatness(y=y)
        features["spectral_flatness_mean"] = float(np.mean(flat))

        # 5. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(np.mean(zcr))

        # 6. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(np.atleast_1d(tempo)[0])

        # 7. Spectral bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = float(np.mean(bw))

        # 8. Spectral flux
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


def generate_report(
    results: dict, n_a: int, n_b: int,
    diff_norm: float, text_results_path: Path | None = None,
) -> str:
    """Generate markdown report with optional comparison to text-prompt results."""
    lines = [
        f"# Statistical Embedding Injection Test: {LABEL_A} vs {LABEL_B}\n",
        f"- Neutral prompt: `{NEUTRAL_PROMPT}`",
        f"- Pole A: `{PROMPT_A}` → injection direction +{INJECTION_STRENGTH}",
        f"- Pole B: `{PROMPT_B}` → injection direction -{INJECTION_STRENGTH}",
        f"- Embedding difference norm: {diff_norm:.4f}",
        f"- Samples: N={n_a} (A), N={n_b} (B)",
        f"- Duration: {DURATION}s, Steps: {STEPS}, CFG: {CFG}",
        f"- Seeds: 0–{N_SAMPLES-1} per condition\n",
        "## Results\n",
        f"| Feature | {LABEL_A} (mean±std) | {LABEL_B} (mean±std) | Cohen's d | p-value | Sig? |",
        "|---|---|---|---|---|---|",
    ]

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
    significant = [(k, r) for k, r in sorted_results if r["significant_05"]]
    if significant:
        lines.append(f"**{len(significant)} features** show significant differences (p<0.05):\n")
        for key, r in significant:
            direction = LABEL_A if r["cohens_d"] > 0 else LABEL_B
            stars = "***" if r["significant_001"] else ("**" if r["significant_01"] else "*")
            lines.append(f"- **{key}**: d={r['cohens_d']:+.3f} → higher in `{direction}` {stars}")
    else:
        lines.append("**No features show significant differences at p<0.05.**")
        lines.append("Embedding injection does NOT produce statistically distinguishable audio.")

    # Compare with text-prompt results if available
    text_results = None
    text_results_path = DATA_DIR / "statistical_test" / "statistical_results.json"
    if text_results_path.exists():
        with open(text_results_path) as f:
            text_results = json.load(f)

    if text_results:
        lines.append("\n## Comparison: Embedding Injection vs Text Prompts\n")
        lines.append("| Feature | Text d | Injection d | Ratio |")
        lines.append("|---|---|---|---|")

        for key, r in sorted_results:
            if key in text_results:
                text_d = text_results[key]["cohens_d"]
                inj_d = r["cohens_d"]
                ratio = abs(inj_d / text_d) if abs(text_d) > 0.01 else float("inf")
                lines.append(
                    f"| {key} | {text_d:+.3f} | {inj_d:+.3f} | {ratio:.1%} |"
                )

        lines.append("")
        lines.append("Ratio = |injection_d / text_d|. Values near 0% = injection has no effect. "
                      "Values near 100% = injection works as well as text.")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dir_a = OUTPUT_DIR / LABEL_A
    dir_b = OUTPUT_DIR / LABEL_B
    dir_a.mkdir(exist_ok=True)
    dir_b.mkdir(exist_ok=True)

    # Phase 1: Compute embeddings
    existing_a = len(list(dir_a.glob("*.wav")))
    existing_b = len(list(dir_b.glob("*.wav")))
    need_generation = existing_a < N_SAMPLES or existing_b < N_SAMPLES

    if need_generation:
        logger.info("Loading T5 for embedding computation...")
        tokenizer, model = load_t5()

        logger.info(f"Encoding: '{NEUTRAL_PROMPT}', '{PROMPT_A}', '{PROMPT_B}'")
        neutral_emb, neutral_mask = encode_prompt(NEUTRAL_PROMPT, tokenizer, model)
        emb_a, _ = encode_prompt(PROMPT_A, tokenizer, model)
        emb_b, _ = encode_prompt(PROMPT_B, tokenizer, model)

        diff = emb_a - emb_b  # [1, seq, 768]
        diff_norm = float(np.linalg.norm(diff))
        logger.info(f"Difference vector norm: {diff_norm:.4f}")

        # Save diff norm for report
        with open(OUTPUT_DIR / "diff_norm.json", "w") as f:
            json.dump({"diff_norm": diff_norm}, f)

        # Condition A: neutral + strength * diff (rhythmic direction)
        emb_injection_a = neutral_emb + INJECTION_STRENGTH * diff
        # Condition B: neutral - strength * diff (sustained direction)
        emb_injection_b = neutral_emb - INJECTION_STRENGTH * diff

        # Unload T5
        del model
        torch.cuda.empty_cache()

        # Phase 2: Generate audio
        for label, emb, out_dir in [
            (LABEL_A, emb_injection_a, dir_a),
            (LABEL_B, emb_injection_b, dir_b),
        ]:
            existing = len(list(out_dir.glob("*.wav")))
            if existing >= N_SAMPLES:
                logger.info(f"{label}: {existing} files already exist, skipping")
                continue

            logger.info(f"Generating {N_SAMPLES} samples for '{label}'...")
            start = time.time()

            for seed in range(N_SAMPLES):
                out_path = out_dir / f"seed_{seed:03d}.wav"
                if out_path.exists():
                    continue

                audio = generate_audio_from_embeddings(emb, neutral_mask, seed)
                if audio:
                    with open(out_path, "wb") as f:
                        f.write(audio)

                if (seed + 1) % 20 == 0:
                    elapsed = time.time() - start
                    rate = (seed + 1) / elapsed
                    logger.info(f"  {seed+1}/{N_SAMPLES} ({rate:.1f} samples/sec)")
    else:
        logger.info(f"Both conditions have {N_SAMPLES}+ files, skipping generation")

    # Load diff_norm for report
    diff_norm_path = OUTPUT_DIR / "diff_norm.json"
    diff_norm = 0.0
    if diff_norm_path.exists():
        with open(diff_norm_path) as f:
            diff_norm = json.load(f)["diff_norm"]

    # Phase 3: Extract features
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

    # Phase 4: Statistical tests
    logger.info("Running statistical tests...")
    results = run_statistical_tests(features_a, features_b)

    # Save raw results
    with open(OUTPUT_DIR / "statistical_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate and save report
    report = generate_report(results, len(features_a), len(features_b), diff_norm)
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
