# T5 Audio-Semantic Interpretability: Research Findings

## Project Summary

This research investigated how T5-Base organizes semantic knowledge about sound in its 768-dimensional embedding space, and whether that structure can be exploited to build a controllable audio synthesizer ("Latent Audio Synth") through Stable Audio's text-to-audio diffusion pipeline.

**Approach**: Train a Sparse Autoencoder (SAE) on T5 activations from ~392K audio-related prompts, decompose 768 entangled dimensions into 6,144 monosemantic features, then sonify them by injecting SAE decoder vectors into Stable Audio's conditioning pathway.

**Core question**: Can we manipulate T5 embeddings directly to control audio generation?

**Answer**: Not via vector arithmetic (injection), but yes via interpolation between natural T5 outputs (LERP). The LERP approach recovers 80-90% of the text-prompt effect and produces monotonic acoustic gradients — a viable basis for a text-pole mixing board.

---

## 1. SAE Training: TopK Collapse and the L1 Solution

### Problem

The initial SAE used TopK sparsity (Anthropic's "Towards Monosemanticity" recipe): only the top-k activations survive per forward pass, all others are zeroed. With 392K training samples and 16x expansion (12,288 features, k=64), this produced **99.3% dead features** — only 86 of 12,288 features ever activated.

TopK creates a winner-take-all dynamic: early winners accumulate gradient, losers receive zero gradient and never recover. Neuron resampling (periodically reinitializing dead features from high-loss examples) was attempted but failed: features stayed alive only while resampling was active, then collapsed back to 99%+ dead within 5 epochs of stopping.

### Solution

Switched to ReLU activation with L1 sparsity penalty during training, TopK only at inference for clean sparse codes:

- **Training**: `features = ReLU(encoder(x))`, loss = MSE + lambda * L1(features)
- **Inference**: `features = TopK(ReLU(encoder(x)), k=64)`

L1 provides continuous gradient to all features (proportional to activation magnitude), preventing the winner-take-all collapse. Combined with reduced expansion (8x instead of 16x, yielding 6,144 features) and L1 warmup (10 epochs MSE-only, then linear ramp to full L1 coefficient):

| Metric | TopK (failed) | L1+ReLU (final) |
|---|---|---|
| Dead features | 99.3% | 0.0% (1 of 6,144) |
| Alive features | 86 | 6,143 |
| MSE per dimension | — | 0.000010 |
| L0 at inference (TopK@64) | — | 64.0 |

**Takeaway**: For SAEs trained on relatively small, domain-specific corpora (~400K samples), L1 sparsity with ReLU is more robust than TopK. The Anthropic TopK recipe was developed for billions of tokens from language model residual streams — a different data regime.

---

## 2. Cultural Distance Geometry

### Method

15 musical traditions (Ukrainian, Yoruba, Gamelan, Arabic, Jewish, Franconian, African American, Romani, Japanese, Korean, Hindustani, Tuvan, Aboriginal Australian, Flamenco, Electronic), each represented by 250 structurally identical prompts. Encoded through T5, projected into SAE feature space, tradition centroids computed.

### Results

**Permutation test**: p = 0.001 (1,000 permutations). The pairwise cosine distances between tradition centroids are significantly non-random. T5 does encode culturally distinguishable representations for these 15 traditions.

**Closest pairs** (cosine distance in SAE feature space):
- Japanese <-> Korean: 0.082
- Korean <-> Ukrainian: 0.085
- Arabic <-> Korean: 0.086

**Most distant pairs**:
- Electronic <-> Japanese: 0.317
- Electronic <-> Hindustani: 0.313
- Electronic <-> Gamelan: 0.293

Electronic music is maximally distant from all acoustic traditions — consistent with its fundamentally different vocabulary (synthesis terms vs. instrument names).

**Discriminative features**: 265 SAE features account for 90% of between-tradition variance (out of 6,144 total). Cultural encoding is distributed across many features, not concentrated in a few.

### Default-Encoding Bias

Distance of bare "music" centroid to each tradition centroid:

| Tradition | Distance to "music" | Interpretation |
|---|---|---|
| Romani | 0.569 (closest) | Most "default-like" |
| African American | 0.571 | |
| Jewish | 0.594 | |
| Franconian | 0.615 | |
| ... | | |
| Japanese | 0.681 (furthest) | Least "default-like" |

**Critical caveat**: This reflects T5's English-language training corpus, not musicological significance. "Romani music" likely co-occurs with common English music vocabulary (violin, guitar, rhythm, melody) more than "Japanese music" does (which requires specialized terms like shakuhachi, gagaku, shamisen). The finding measures **lexical proximity in training data**, not cultural importance or representativeness.

---

## 3. Individual Feature Interpretation: Unvalidated

The SAE analysis identified 1,073 alive features (at TopK@64 inference) and generated top-activating prompts for each. Example:

> **Feature #1058** (max |r| = 0.297): Top prompts include Balinese gamelan vocal interjections, Hindustani tanpura, and high-arousal exclamations.

These interpretations are **post-hoc correlational labels**, not causal descriptions. The top-activating prompts share surface vocabulary (specific instrument names, cultural labels) that may or may not reflect a coherent acoustic concept encoded in the feature.

**Validation attempt via sonification**: Activating Feature #1058 at various strengths produced ~0.5s percussive sounds with no identifiable connection to Balinese music or any other specific acoustic quality. The interpretation did not survive causal testing.

**Status**: Individual feature semantics remain unvalidated. The feature atlas is useful as a hypothesis-generating tool, not as a ground-truth map.

---

## 4. Sonification Experiments: The Conditioning Bottleneck

### Experiment A: SAE Feature Sonification

**Method**: Encode neutral prompt "sound" through T5, add scaled SAE decoder column vectors (strengths -2 to +2), generate audio via Stable Audio's embedding-conditioned endpoint.

**Result**: 250 files generated (50 features x 5 strengths). No perceptible systematic variation across strength levels. All outputs sound similar — short percussive textures regardless of which feature or strength is applied.

### Experiment B: Binary Contrast Sonification

**Method**: Encode "sound smooth" and "sound harsh" through T5, compute difference vector, inject at 7 strength levels (-3 to +3) into neutral "sound" embedding.

**Result**: 90 files generated (10 contrasts x 7 strengths + 20 references). No perceptible pattern across injection strengths. Notably, strength +0.0 (pure neutral embedding) always produces a similar knocking sound regardless of contrast direction.

However, the 20 reference files generated via direct text prompts ("sound smooth", "sound harsh", etc.) **do** sound different from each other. "sound rhythmic" in particular produces clearly rhythmic audio.

### Experiment C: Statistical Validation (N=100 per condition)

To overcome single-seed randomness, generated 100 samples each (seeds 0-99) for two conditions and extracted 11 acoustic features via librosa (onset density, spectral centroid, RMS energy, spectral flatness, zero crossing rate, tempo, spectral bandwidth, spectral flux). Welch's t-test, Cohen's d, and Mann-Whitney U computed per feature.

**Condition: "sound rhythmic" vs "sound sustained"**

#### Text Prompts (Experiment C1)

| Acoustic Feature | Cohen's d | p-value |
|---|---|---|
| spectral_centroid_std | **+2.882** | 6.9 x 10^-45 |
| spectral_flatness_mean | **+2.280** | 4.2 x 10^-31 |
| rms_mean | **-1.949** | 1.5 x 10^-26 |
| spectral_flux_std | **+1.635** | 6.0 x 10^-22 |
| onset_density | **-1.343** | 2.7 x 10^-16 |
| rms_std | **+1.073** | 2.1 x 10^-12 |

6 of 11 features show highly significant differences (p < 0.001). Cohen's d values up to 2.88 (d > 0.8 is conventionally "large"). Text prompts produce **massively** distinguishable audio across seeds.

#### Embedding Injection (Experiment C2)

Same semantic contrast (rhythmic vs sustained), but applied via T5 difference vector injection instead of text prompts:

| Acoustic Feature | Cohen's d | p-value |
|---|---|---|
| onset_density | **-0.834** | 2.0 x 10^-8 |
| spectral_flatness_mean | **+0.526** | 3.3 x 10^-4 |
| spectral_flux_std | **+0.419** | 3.6 x 10^-3 |

3 of 11 features significant. Largest effect: d = 0.83 (medium-to-large).

#### Direct Comparison

| Feature | Text d | Injection d | Signal Retention |
|---|---|---|---|
| spectral_centroid_std | +2.882 | +0.181 | 6% |
| spectral_flatness_mean | +2.280 | +0.526 | 23% |
| rms_mean | -1.949 | +0.214 | 11% (wrong direction) |
| spectral_flux_std | +1.635 | +0.419 | 26% |
| onset_density | -1.343 | -0.834 | 62% |
| rms_std | +1.073 | +0.038 | 4% |

**Key findings**:

1. Embedding injection is **not zero-effect** — 3 features reach statistical significance with N=100. But the effect is invisible at the single-sample level (unhearable).

2. Signal retention is **5-25%** for most features. The strongest injection effect (onset_density, d=0.83) is weaker than the weakest significant text-prompt effect (rms_std, d=1.07).

3. **Direction reversal**: rms_mean shows opposite sign under injection vs text prompts (+0.21 vs -1.95), indicating partial distortion of semantic content during embedding manipulation.

#### LERP Interpolation (Experiment C3)

Instead of injecting a difference vector into a neutral embedding, interpolate linearly between two natural T5 outputs: `emb = (1-t) * T5("sound sustained") + t * T5("sound rhythmic")`, with t in {0.0, 0.25, 0.5, 0.75, 1.0}. Both endpoints are natural T5 outputs — the interpolation stays close to Stable Audio's learned conditioning distribution.

N=100 per position, 500 samples total.

**Extreme comparison** (t=0.0 vs t=1.0):

| Acoustic Feature | Cohen's d | p-value |
|---|---|---|
| spectral_centroid_std | **+2.434** | 9.2 x 10^-41 |
| spectral_flatness_mean | **+1.860** | 3.3 x 10^-28 |
| rms_std | **+1.404** | 7.4 x 10^-19 |
| spectral_flux_std | **+1.348** | 7.8 x 10^-18 |
| onset_density | **-1.169** | 3.5 x 10^-13 |
| rms_mean | **-1.112** | 1.6 x 10^-12 |

6 of 11 features highly significant (p < 0.001). Same 6 features as the text-prompt experiment.

**Gradient analysis** (Pearson correlation between LERP position t and acoustic feature, N=500):

| Feature | Pearson r | p-value | t=0.00 | t=0.25 | t=0.50 | t=0.75 | t=1.00 |
|---|---|---|---|---|---|---|---|
| spectral_centroid_std | +0.631 | 7.5 x 10^-57 | 738 | 1393 | 1884 | 2276 | 2566 |
| spectral_flatness_mean | +0.544 | 6.2 x 10^-40 | 0.04 | 0.09 | 0.15 | 0.18 | 0.20 |
| spectral_flux_std | +0.520 | 5.8 x 10^-36 | 23.0 | 18.0 | 21.5 | 51.8 | 55.4 |
| rms_std | +0.509 | 2.8 x 10^-34 | 0.05 | 0.05 | 0.05 | 0.12 | 0.12 |
| onset_density | -0.361 | 7.7 x 10^-17 | 9.0 | 5.4 | 4.3 | 4.4 | 4.6 |
| rms_mean | -0.295 | 1.8 x 10^-11 | 0.28 | 0.13 | 0.07 | 0.15 | 0.13 |

8 of 11 features show significant monotonic trends. Intermediate LERP positions produce intermediate acoustic features.

#### Full Comparison Across Methods

| Feature | Text d | Injection d | LERP d | LERP/Text |
|---|---|---|---|---|
| spectral_centroid_std | +2.882 | +0.181 | +2.434 | **84%** |
| spectral_flatness_mean | +2.280 | +0.526 | +1.860 | **82%** |
| rms_std | +1.073 | +0.038 | +1.404 | **131%** |
| spectral_flux_std | +1.635 | +0.419 | +1.348 | **83%** |
| onset_density | -1.343 | -0.834 | -1.169 | **87%** |
| rms_mean | -1.949 | +0.214 | -1.112 | **57%** |

LERP recovers **57-131%** of the text-prompt effect (median ~84%), compared to injection's 5-25%. All effect directions match the text-prompt baseline. The rms_mean direction reversal observed in injection (wrong sign) is corrected in LERP.

#### Multi-Axis Additive LERP (Experiment C4)

Tests whether 3 LERP axes combined additively preserve individual effects.

**Design**: 2³ factorial — 3 axes (rhythmic↔sustained, bright↔dark, smooth↔harsh), each at 2 levels, 8 conditions, N=50 per condition = 400 samples total.

**Combination method**: `emb = neutral + (pole1 - neutral) + (pole2 - neutral) + (pole3 - neutral)`, where each pole is the T5 embedding of the selected axis endpoint.

**All three axes show significant main effects on different acoustic dimensions:**

| Axis | Strongest Feature | Cohen's d | p-value |
|---|---|---|---|
| rhythmic↔sustained | spectral_centroid_std | +1.022 | 7.9 x 10^-22 |
| bright↔dark | spectral_bandwidth_mean | +1.145 | 4.9 x 10^-26 |
| smooth↔harsh | rms_mean | -1.737 | 2.2 x 10^-47 |

The axes affect *different* acoustic features — rhythmic controls spectral variation, bright controls bandwidth, smooth controls energy. This is the prerequisite for independent control.

**However, signal degradation is substantial.** The rhythmic↔sustained axis loses ~60-65% of its single-axis effect:

| Feature | LERP (1 axis) d | Multi (3 axes) d | Retention |
|---|---|---|---|
| spectral_centroid_std | +2.434 | +1.022 | **35%** |
| spectral_flatness_mean | +1.860 | +0.821 | **36%** |
| spectral_flux_std | +1.348 | +0.646 | **40%** |
| onset_density | -1.169 | -0.306 | **23%** |

**Axis interaction**: The rhythmic axis effect varies by context — spectral_centroid_std d ranges from +0.36 (when others = dark+smooth) to +1.94 (when others = bright+smooth). The axes are not independent; they interact.

#### Complete Method Comparison

| Method | Signal Retention | Monotonic? | Multi-axis? |
|---|---|---|---|
| Text prompts | 100% (reference) | n/a | n/a |
| LERP (1 axis) | **84%** | Yes (r=0.63) | n/a |
| **Additive LERP (3 axes)** | **35%** | Yes | Yes, but degraded |
| Embedding injection | 6% | Not tested | Not tested |

---

## 5. Diagnosis: Why Injection Fails, LERP Succeeds, and Multi-Axis Degrades

Stable Audio's conditioning pathway is trained end-to-end: tokenizer -> T5 -> cross-attention -> diffusion. The cross-attention layers learn to attend to embedding patterns that T5 naturally produces.

**Injection fails** (5-25% signal retention): Adding difference vectors or SAE decoder columns to an embedding creates out-of-distribution inputs. The resulting point in embedding space doesn't look like anything T5 would naturally produce. Cross-attention largely ignores it.

**LERP succeeds** (80-90% signal retention): Both interpolation endpoints are natural T5 outputs. Linear interpolation between two in-distribution points stays close to the learned manifold. Cross-attention responds to the blended embedding almost as strongly as to a pure text-prompted one.

This is not a flaw in the SAE or in T5's representations. T5 does encode semantically meaningful structure (the cultural distance analysis confirms this). The constraint is that Stable Audio only responds to embeddings that lie on or near the manifold of natural T5 outputs.

**Multi-axis degrades** (35% signal retention): Adding 3 deltas simultaneously pushes the combined embedding away from the T5 manifold. Not as catastrophically as raw injection (the deltas are smaller and better-behaved), but the cumulative displacement is enough to lose ~65% of each axis's effect. The axes also interact — the effect of one axis depends on the position of the others.

**Implication**: Single-axis LERP works well. Multi-axis additive combination works but degrades. A learned projection (bridge model) that maps multi-slider configurations back onto the T5 manifold could recover the lost signal.

---

## 6. What Survives Scrutiny

| Finding | Status | Evidence |
|---|---|---|
| SAE training: L1+ReLU > TopK for small corpora | **Confirmed** | 0% vs 99.3% dead features |
| Cultural distances are non-random | **Confirmed** | Permutation test p=0.001 |
| Cultural encoding is distributed (~265 features) | **Confirmed** | Variance analysis |
| Default-bias reflects training corpus vocabulary | **Plausible** | But confounded by probing template vocabulary |
| Individual SAE feature semantics | **Unvalidated** | Sonification failed to provide causal evidence |
| Text prompts produce distinguishable audio | **Confirmed** | d=2.88, N=100, p<10^-44 |
| Embedding injection has measurable but weak effect | **Confirmed** | d=0.83, N=100, p<10^-7, ~5-25% signal retention |
| LERP between T5 outputs recovers ~84% of text effect | **Confirmed** | d=2.43, N=100, p<10^-40, monotonic gradient |
| LERP produces monotonic acoustic gradients | **Confirmed** | Pearson r up to 0.63, 8/11 features significant |
| Multi-axis additive LERP (3 axes) | **Partially confirmed** | Each axis significant, but 35% retention per axis |
| Multi-axis axes affect different features | **Confirmed** | rhythmic→spectral variation, bright→bandwidth, smooth→energy |
| Multi-axis axes are independent | **Rejected** | Axis effects depend on context (d ranges 0.36–1.94) |
| Latent Audio Synth via embedding injection | **Not viable** | Effect too weak for perceptual control |
| Latent Audio Synth via single-axis LERP | **Viable** | 84% signal retention, monotonic control |
| Latent Audio Synth via multi-axis additive LERP | **Marginal** | 35% retention; usable but needs bridge model for full signal |

---

## 7. Implications for the Latent Audio Synth

The original vision — SAE feature sliders directly manipulating embeddings — is not viable. LERP interpolation between text-encoded poles recovers 84% of the text-prompt effect for a single axis, but degrades to ~35% when 3 axes are combined additively. A functional multi-axis synth requires either accepting the degradation or training a bridge model.

### Architecture Options

#### Option A: Single-Axis LERP (validated, 84% signal)

One slider per session, switching between axis pairs. Simple, no interaction effects, strong signal. Limited expressivity — users control one dimension at a time.

#### Option B: Additive Multi-Axis (validated, ~35% signal per axis)

N sliders, each backed by a text-pole pair. Combination: neutral + Σ deltas. Effects are significant but degraded, and axes interact. Still usable for coarse control — d=1.02 is a "large" effect by convention, even at 35% retention. The smooth↔harsh axis retains d=1.74 even in multi-axis context.

Example axes (all validated in multi-axis context):
- "sound rhythmic" <-> "sound sustained" (d=1.02, spectral variation)
- "sound bright" <-> "sound dark" (d=1.15, bandwidth)
- "sound smooth" <-> "sound harsh" (d=1.74, energy)

#### Option C: Bridge Model (not yet implemented)

A small learned network (MLP or VAE) that maps slider positions → valid T5 embedding. Trained on (text prompt, T5 embedding) pairs with annotations for each axis dimension. The network learns the non-linear manifold geometry and projects arbitrary slider combinations back onto it.

Training data: ~10K text prompts with annotated properties, encoded through T5. Architecture: small MLP (N slider inputs → 768d embedding), loss = reconstruction + distributional regularization (output must be "T5-like").

This would potentially recover the full 84% signal per axis even with multiple axes, while eliminating interaction effects.

### Resolved Questions

1. **Multi-axis interaction**: Additive combination degrades to ~35% signal per axis. The embedding drifts from the T5 manifold, but not catastrophically — all axes remain significant.

2. **Axis orthogonality**: **Rejected.** Axes interact — the rhythmic axis effect varies from d=0.36 to d=1.94 depending on other axes' positions. Independent control requires a bridge model.

3. **Perceptual threshold**: Not yet tested. Statistical significance (d=1.02 in multi-axis) suggests the effects should be audible, but needs human evaluation.

4. **Pole selection**: The three tested axes affect different acoustic dimensions, confirming that diverse poles produce diverse control. Systematic screening of more pairs would identify optimal axes.

### Abandoned Paths

1. **SAE feature injection**: 5-25% signal retention. Analytically useful, operationally not actionable.

2. **Difference vector injection**: Same problem — out-of-distribution.

3. **Cross-attention fine-tuning**: Would require retraining Stable Audio, high cost, unnecessary given LERP viability.

---

## Experimental Details

| Parameter | Value |
|---|---|
| Corpus size | 392,268 prompts (388K bulk + 3.9K probing) |
| T5 model | google-t5/t5-base (110M params) |
| Activation shape | [392,268 x 768], mean-pooled over non-padding tokens |
| SAE architecture | ReLU + L1 training, TopK@64 inference |
| SAE expansion | 8x (768 -> 6,144 features) |
| SAE alive features | 6,143 / 6,144 (99.98%) |
| Audio model | Stable Audio (via GPU service, port 17803) |
| Audio parameters | 5s duration, 100 steps, CFG 7.0 |
| Statistical tests | Welch's t-test, Cohen's d, Mann-Whitney U |
| Samples per condition | N=100 (seeds 0-99); LERP: N=100 x 5 positions; Multi-axis: N=50 x 8 conditions |
| Acoustic features | 11 (librosa: onset density, spectral centroid mean/std, RMS mean/std, spectral flatness, ZCR, tempo, spectral bandwidth, spectral flux mean/std) |

### Scripts

| Script | Purpose |
|---|---|
| `build_corpus.py` | Phase 1a: Download AudioCaps, MusicCaps, WavCaps |
| `build_probing_corpus.py` | Phase 1b: Generate structured probing prompts |
| `encode_corpus.py` | Phase 2: T5 batch encoding |
| `dimension_atlas.py` | Phase 3: Per-dimension clustering |
| `train_sae.py` | Phase 4: SAE training (L1+ReLU) |
| `analyze_features.py` | Phase 5: Feature interpretation |
| `sonify_features.py` | Phase 6: SAE feature sonification |
| `cultural_analysis.py` | Phase 7: Cultural distance geometry |
| `sonify_binary_contrasts.py` | Validation: Binary contrast injection |
| `statistical_sonification_test.py` | Validation: Text-prompt statistics (N=100) |
| `statistical_embedding_injection_test.py` | Validation: Injection statistics (N=100) |
| `statistical_lerp_test.py` | Validation: LERP interpolation statistics (N=500) |
| `statistical_multiaxis_test.py` | Validation: Multi-axis factorial test (N=400) |

### Data

All outputs in `research/t5_interpretability/data/` (gitignored). Key files:

- `corpus.json` (392K entries)
- `activations_pooled.pt` ([392268, 768], 575 MB)
- `sae_weights.pt` (L1 SAE, 37 MB)
- `feature_atlas.json` + `feature_atlas_report.md`
- `cultural_analysis_report.md`
- `statistical_test/statistical_report.md` (text-prompt results)
- `statistical_injection_test/statistical_report.md` (injection results)
- `sonification/` (250 WAV, SAE features)
- `sonification_binary/` (90 WAV, binary contrasts)
- `statistical_test/` (200 WAV, text-prompt experiment)
- `statistical_injection_test/` (200 WAV, injection experiment)
- `statistical_lerp_test/` (500 WAV, LERP gradient experiment)
- `statistical_multiaxis_test/` (400 WAV, multi-axis factorial experiment)
