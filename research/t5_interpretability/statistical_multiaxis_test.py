#!/usr/bin/env python3
"""
Statistical Multi-Axis LERP Test

Tests whether combining multiple LERP axes simultaneously preserves
each axis's individual effect, or whether the combined embedding drifts
out-of-distribution and degrades.

Design: 2³ factorial experiment
  - 3 axes: rhythmic↔sustained, bright↔dark, smooth↔harsh
  - Each axis at 2 levels (pole A = +1, pole B = -1)
  - 8 conditions, N=50 per condition = 400 samples total

Combination method (additive):
  emb = neutral + (lerp1 - neutral) + (lerp2 - neutral) + (lerp3 - neutral)
  where each lerp_i is the T5 embedding of the selected pole.

Analysis:
  - Main effect of each axis (averaged over other axes' levels)
  - Interaction effects (does axis1's effect depend on axis2/3?)
  - Compare single-axis d vs multi-axis d

Usage: venv/bin/python research/t5_interpretability/statistical_multiaxis_test.py
"""

import base64
import io
import itertools
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

# 3 axes: (label, pole_A prompt, pole_B prompt)
AXES = [
    ("rhythmic_sustained", "sound rhythmic", "sound sustained"),
    ("bright_dark", "sound bright", "sound dark"),
    ("smooth_harsh", "sound smooth", "sound harsh"),
]

N_SAMPLES = 50  # per condition (8 conditions = 400 total)
DURATION = 5.0
STEPS = 100
CFG = 7.0

OUTPUT_DIR = DATA_DIR / "statistical_multiaxis_test"


def load_t5():
    from transformers import T5EncoderModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_ID)
    model = T5EncoderModel.from_pretrained(T5_MODEL_ID).cuda().half()
    model.eval()
    return tokenizer, model


def encode_prompt(text: str, tokenizer, model) -> tuple[np.ndarray, np.ndarray]:
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


def generate_audio(emb: np.ndarray, mask: np.ndarray, seed: int) -> bytes | None:
    url = f"{GPU_SERVICE_URL}/api/stable_audio/generate_from_embeddings"
    payload = {
        "embeddings_b64": _numpy_to_b64(emb),
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


def condition_label(levels: tuple[int, ...]) -> str:
    """Create a label like 'A1_B0_C1' from axis levels."""
    parts = []
    for i, level in enumerate(levels):
        pole = "A" if level == 1 else "B"
        parts.append(f"ax{i}{pole}")
    return "_".join(parts)


def compute_main_effects(
    all_features: dict[tuple, list[dict]],
) -> dict:
    """Compute main effect of each axis, averaged over other axes."""
    results = {}
    feature_keys = sorted(next(iter(all_features.values()))[0].keys())

    for axis_idx in range(len(AXES)):
        axis_name = AXES[axis_idx][0]

        # Collect all samples where this axis = A (1) vs B (0)
        feats_a = []
        feats_b = []
        for levels, features in all_features.items():
            if levels[axis_idx] == 1:
                feats_a.extend(features)
            else:
                feats_b.extend(features)

        axis_results = {}
        for key in feature_keys:
            vals_a = np.array([f[key] for f in feats_a])
            vals_b = np.array([f[key] for f in feats_b])

            t_stat, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)
            pooled_std = np.sqrt((vals_a.std()**2 + vals_b.std()**2) / 2)
            cohens_d = (vals_a.mean() - vals_b.mean()) / pooled_std if pooled_std > 0 else 0

            axis_results[key] = {
                "mean_a": float(vals_a.mean()),
                "mean_b": float(vals_b.mean()),
                "std_a": float(vals_a.std()),
                "std_b": float(vals_b.std()),
                "n_a": len(vals_a),
                "n_b": len(vals_b),
                "cohens_d": float(cohens_d),
                "p_value": float(p_value),
                "significant_001": bool(p_value < 0.001),
                "significant_01": bool(p_value < 0.01),
                "significant_05": bool(p_value < 0.05),
            }

        results[axis_name] = axis_results

    return results


def compute_single_axis_effects(
    all_features: dict[tuple, list[dict]],
) -> dict:
    """Compute effect of each axis when the other two are held at their midpoint-like average.

    Since we don't have a midpoint condition, we compare:
    - Axis varied, others both at A: e.g., (1,1,1) vs (0,1,1)
    - Axis varied, others both at B: e.g., (1,0,0) vs (0,0,0)
    Then average the two effect sizes.
    """
    results = {}
    feature_keys = sorted(next(iter(all_features.values()))[0].keys())

    for axis_idx in range(len(AXES)):
        axis_name = AXES[axis_idx][0]
        other_indices = [i for i in range(len(AXES)) if i != axis_idx]

        # For each combination of other axes' levels
        sub_effects = []
        for other_levels in itertools.product([0, 1], repeat=len(other_indices)):
            # Build level tuples for A and B of this axis
            levels_a = list(range(len(AXES)))
            levels_b = list(range(len(AXES)))
            for oi, ol in zip(other_indices, other_levels):
                levels_a[oi] = ol
                levels_b[oi] = ol
            levels_a[axis_idx] = 1
            levels_b[axis_idx] = 0
            levels_a = tuple(levels_a)
            levels_b = tuple(levels_b)

            if levels_a in all_features and levels_b in all_features:
                feats_a = all_features[levels_a]
                feats_b = all_features[levels_b]

                for key in feature_keys:
                    vals_a = np.array([f[key] for f in feats_a])
                    vals_b = np.array([f[key] for f in feats_b])
                    pooled_std = np.sqrt((vals_a.std()**2 + vals_b.std()**2) / 2)
                    d = (vals_a.mean() - vals_b.mean()) / pooled_std if pooled_std > 0 else 0
                    sub_effects.append({
                        "axis": axis_name,
                        "key": key,
                        "context": f"others={other_levels}",
                        "d": float(d),
                        "levels_a": levels_a,
                        "levels_b": levels_b,
                    })

        results[axis_name] = sub_effects

    return results


def generate_report(
    main_effects: dict,
    sub_effects: dict,
    sample_counts: dict[tuple, int],
) -> str:
    lines = [
        "# Multi-Axis LERP Interaction Test\n",
        f"- Neutral: `{NEUTRAL_PROMPT}`",
        f"- Axes:",
    ]
    for name, prompt_a, prompt_b in AXES:
        lines.append(f"  - {name}: `{prompt_a}` ↔ `{prompt_b}`")
    lines.append(f"- Combination: additive (neutral + Σ deltas)")
    lines.append(f"- Design: 2³ factorial, N={N_SAMPLES} per condition")
    lines.append(f"- Total: {sum(sample_counts.values())} samples\n")

    # Sample counts
    lines.append("## Conditions\n")
    lines.append("| Condition | Axis 0 | Axis 1 | Axis 2 | N |")
    lines.append("|---|---|---|---|---|")
    for levels in sorted(sample_counts.keys()):
        label = condition_label(levels)
        axis_labels = []
        for i, l in enumerate(levels):
            pole = AXES[i][1].split()[-1] if l == 1 else AXES[i][2].split()[-1]
            axis_labels.append(pole)
        lines.append(f"| {label} | {axis_labels[0]} | {axis_labels[1]} | {axis_labels[2]} | {sample_counts[levels]} |")
    lines.append("")

    # Main effects (averaged over other axes)
    lines.append("## Main Effects (each axis averaged over others)\n")

    # Load single-axis LERP reference for comparison
    lerp_path = DATA_DIR / "statistical_lerp_test" / "extreme_results.json"
    lerp_ref = {}
    if lerp_path.exists():
        with open(lerp_path) as f:
            lerp_ref = json.load(f)

    for axis_name in main_effects:
        pole_a = [a[1].split()[-1] for a in AXES if a[0] == axis_name][0]
        pole_b = [a[2].split()[-1] for a in AXES if a[0] == axis_name][0]
        lines.append(f"### {axis_name} ({pole_a} ↔ {pole_b})\n")
        lines.append("| Feature | Cohen's d | p-value | Sig? |")
        lines.append("|---|---|---|---|")

        sorted_feats = sorted(
            main_effects[axis_name].items(),
            key=lambda x: abs(x[1]["cohens_d"]),
            reverse=True,
        )

        for key, r in sorted_feats:
            if not r["significant_05"]:
                continue
            sig = "***" if r["significant_001"] else ("**" if r["significant_01"] else "*")
            lines.append(f"| {key} | {r['cohens_d']:+.3f} | {r['p_value']:.2e} | {sig} |")

        lines.append("")

    # Context-dependent effects (does axis effect depend on other axes?)
    lines.append("## Axis Effects by Context\n")
    lines.append("Does each axis's effect survive when other axes are also active?\n")

    # For the validated axis (rhythmic↔sustained), show effect in each context
    ref_axis = AXES[0][0]
    if ref_axis in sub_effects:
        lines.append(f"### {ref_axis} (reference axis, LERP d=2.434)\n")

        # Group by feature
        feature_keys = sorted(set(e["key"] for e in sub_effects[ref_axis]))

        # Pick the top features from LERP test
        top_features = ["spectral_centroid_std", "spectral_flatness_mean", "rms_std",
                        "spectral_flux_std", "onset_density", "rms_mean"]

        lines.append("| Feature | Others=BB | Others=BA | Others=AB | Others=AA | Ref (LERP) |")
        lines.append("|---|---|---|---|---|---|")

        for feat in top_features:
            row = [f"| {feat}"]
            for e in sub_effects[ref_axis]:
                if e["key"] == feat:
                    row.append(f"{e['d']:+.3f}")
            ref_d = lerp_ref.get(feat, {}).get("cohens_d", 0)
            row.append(f"{ref_d:+.3f}")
            lines.append(" | ".join(row) + " |")

        lines.append("")

    # Cross-experiment comparison
    text_path = DATA_DIR / "statistical_test" / "statistical_results.json"
    inj_path = DATA_DIR / "statistical_injection_test" / "statistical_results.json"

    text_results = {}
    if text_path.exists():
        with open(text_path) as f:
            text_results = json.load(f)

    if text_results and ref_axis in main_effects:
        lines.append("## Cross-Experiment Comparison (rhythmic↔sustained axis)\n")
        lines.append("| Feature | Text d | Injection d | LERP d | Multi-Axis d | Multi/Text |")
        lines.append("|---|---|---|---|---|---|")

        for key, r in sorted(
            main_effects[ref_axis].items(),
            key=lambda x: abs(x[1]["cohens_d"]),
            reverse=True,
        ):
            text_d = text_results.get(key, {}).get("cohens_d", 0)
            lerp_d = lerp_ref.get(key, {}).get("cohens_d", 0)
            inj_d = 0
            inj_path_f = DATA_DIR / "statistical_injection_test" / "statistical_results.json"
            if inj_path_f.exists():
                with open(inj_path_f) as f:
                    inj_data = json.load(f)
                inj_d = inj_data.get(key, {}).get("cohens_d", 0)
            multi_d = r["cohens_d"]
            ratio = abs(multi_d / text_d) if abs(text_d) > 0.01 else float("inf")
            ratio_str = f"{ratio:.1%}" if ratio < 100 else "n/a"
            lines.append(
                f"| {key} | {text_d:+.3f} | {inj_d:+.3f} | {lerp_d:+.3f} | {multi_d:+.3f} | {ratio_str} |"
            )

        lines.append("\nMulti/Text: how much of the text-prompt effect the multi-axis approach preserves.")

    # Summary
    lines.append("\n## Summary\n")

    sig_counts = {}
    for axis_name, feats in main_effects.items():
        sig_counts[axis_name] = sum(1 for r in feats.values() if r["significant_05"])

    for axis_name, count in sig_counts.items():
        lines.append(f"- **{axis_name}**: {count}/11 features significant")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # All 2³ conditions
    all_conditions = list(itertools.product([0, 1], repeat=len(AXES)))

    # Check generation status
    need_generation = False
    for levels in all_conditions:
        label = condition_label(levels)
        d = OUTPUT_DIR / label
        if not d.exists() or len(list(d.glob("*.wav"))) < N_SAMPLES:
            need_generation = True
            break

    # Prepare embeddings
    embeddings = {}
    if need_generation:
        logger.info("Loading T5...")
        tokenizer, model = load_t5()

        # Encode neutral and all poles
        logger.info(f"Encoding neutral: '{NEUTRAL_PROMPT}'")
        neutral_emb, neutral_mask = encode_prompt(NEUTRAL_PROMPT, tokenizer, model)

        pole_embeddings = {}
        for name, prompt_a, prompt_b in AXES:
            logger.info(f"Encoding axis {name}: '{prompt_a}' / '{prompt_b}'")
            emb_a, _ = encode_prompt(prompt_a, tokenizer, model)
            emb_b, _ = encode_prompt(prompt_b, tokenizer, model)
            pole_embeddings[name] = (emb_a, emb_b)

            # Log delta norms
            delta_a = emb_a - neutral_emb
            delta_b = emb_b - neutral_emb
            logger.info(f"  delta_A norm={np.linalg.norm(delta_a):.2f}, delta_B norm={np.linalg.norm(delta_b):.2f}")

        # Compute combined embeddings for each condition
        for levels in all_conditions:
            label = condition_label(levels)
            # Additive: neutral + Σ (pole_i - neutral)
            combined = neutral_emb.copy()
            for i, level in enumerate(levels):
                axis_name = AXES[i][0]
                emb_a, emb_b = pole_embeddings[axis_name]
                pole_emb = emb_a if level == 1 else emb_b
                combined = combined + (pole_emb - neutral_emb)
            embeddings[levels] = combined

        del model
        torch.cuda.empty_cache()

        # Generate audio
        total = len(all_conditions) * N_SAMPLES
        done = 0
        start_time = time.time()

        for levels in all_conditions:
            label = condition_label(levels)
            out_dir = OUTPUT_DIR / label
            out_dir.mkdir(exist_ok=True)

            existing = len(list(out_dir.glob("*.wav")))
            if existing >= N_SAMPLES:
                logger.info(f"{label}: {existing} files exist, skipping")
                done += N_SAMPLES
                continue

            logger.info(f"Generating {N_SAMPLES} samples for {label}...")

            for seed in range(N_SAMPLES):
                out_path = out_dir / f"seed_{seed:03d}.wav"
                if out_path.exists():
                    done += 1
                    continue

                audio = generate_audio(embeddings[levels], neutral_mask, seed)
                if audio:
                    with open(out_path, "wb") as f:
                        f.write(audio)

                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    logger.info(f"  [{done}/{total}] ({rate:.1f}/sec, ETA {eta:.0f}s)")
    else:
        logger.info("All conditions generated, skipping")

    # Extract features
    logger.info("Extracting features...")
    all_features: dict[tuple, list[dict]] = {}
    sample_counts: dict[tuple, int] = {}

    for levels in all_conditions:
        label = condition_label(levels)
        out_dir = OUTPUT_DIR / label
        features = []
        for wav_path in sorted(out_dir.glob("*.wav")):
            feats = extract_features(wav_path)
            if feats:
                features.append(feats)
        all_features[levels] = features
        sample_counts[levels] = len(features)
        logger.info(f"  {label}: {len(features)} samples")

    for levels, feats in all_features.items():
        if len(feats) < 10:
            logger.error(f"Too few samples for {condition_label(levels)}: {len(feats)}")
            return

    # Analysis
    logger.info("Computing main effects...")
    main_effects = compute_main_effects(all_features)

    logger.info("Computing context-dependent effects...")
    sub_effects = compute_single_axis_effects(all_features)

    # Save results
    with open(OUTPUT_DIR / "main_effects.json", "w") as f:
        json.dump(main_effects, f, indent=2)
    with open(OUTPUT_DIR / "sub_effects.json", "w") as f:
        json.dump(sub_effects, f, indent=2)

    # Report
    report = generate_report(main_effects, sub_effects, sample_counts)
    report_path = OUTPUT_DIR / "statistical_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"\n{'='*60}")
    logger.info(f"Report: {report_path}")
    logger.info(f"{'='*60}")

    # Print summary
    for axis_name, feats in main_effects.items():
        sorted_feats = sorted(feats.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True)
        top = sorted_feats[0]
        sig = "***" if top[1]["significant_001"] else ("**" if top[1]["significant_01"] else ("*" if top[1]["significant_05"] else "n.s."))
        logger.info(f"  {axis_name}: top={top[0]}, d={top[1]['cohens_d']:+.3f} {sig}")


if __name__ == "__main__":
    main()
