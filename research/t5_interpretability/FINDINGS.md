# T5 Audio-Semantic Interpretability: Research Findings

## Project Summary

This research investigated how T5-Base organizes semantic knowledge about sound in its 768-dimensional embedding space, and whether that structure can be exploited to build a controllable audio synthesizer ("Latent Audio Synth") through Stable Audio's text-to-audio diffusion pipeline.

**Approach**: Train a Sparse Autoencoder (SAE) on T5 activations from ~392K audio-related prompts, decompose 768 entangled dimensions into 6,144 monosemantic features, then sonify them by injecting SAE decoder vectors into Stable Audio's conditioning pathway.

**Core question**: Can we manipulate T5 embeddings directly to control audio generation?

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

---

## 5. Diagnosis: Why Embedding Injection Fails

Stable Audio's conditioning pathway is trained end-to-end: tokenizer -> T5 -> cross-attention -> diffusion. Direct embedding manipulation bypasses the tokenizer-T5 pipeline and injects vectors that lie outside the learned distribution of T5 outputs.

The cross-attention layers learn to attend to embedding patterns that T5 naturally produces. Artificial vector arithmetic (adding difference vectors or SAE decoder columns) creates out-of-distribution inputs that the cross-attention layers effectively ignore or dampen.

This is not a flaw in the SAE or in T5's representations. T5 does encode semantically meaningful structure (the cultural distance analysis confirms this). The bottleneck is Stable Audio's conditioning mechanism, which is not designed to respond to arbitrary embedding-space manipulations.

**Analogy**: T5's embedding space is like a well-organized library. The SAE successfully catalogs which books are on which shelves. But Stable Audio's reading mechanism only accepts books through the front door (text -> tokenizer -> T5). Handing books directly through a window (embedding injection) mostly gets ignored by the librarian.

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
| Latent Audio Synth via embedding injection | **Not viable** | Effect too weak for perceptual control |

---

## 7. Implications for the Latent Audio Synth

The original vision — a synthesizer where users manipulate SAE feature sliders to control audio generation — is not achievable with the current architecture (T5 embedding injection -> Stable Audio cross-attention). The conditioning bottleneck prevents embedding-space manipulations from producing perceptible acoustic changes.

### Possible Paths Forward

1. **Textual Inversion**: Instead of injecting vectors, map SAE features back to optimized token sequences that travel through the natural text -> T5 -> cross-attention pathway. This respects Stable Audio's learned distribution.

2. **Cross-Attention Fine-Tuning**: Train Stable Audio's cross-attention layers to respond more strongly to embedding variations, expanding the effective conditioning distribution.

3. **Alternative Audio Model**: Use an audio generation model whose conditioning is less strongly regularized — e.g., flow-matching models conditioned directly on continuous embeddings without cross-attention bottlenecks.

4. **Text-Prompt Interpolation**: Since text prompts produce strong effects (d=2.88), a "synth" that generates audio from dynamically composed text prompts (rather than manipulated embeddings) could leverage T5's semantic structure indirectly. The SAE feature atlas could inform which text dimensions to expose as controls.

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
| Samples per condition | N=100 (seeds 0-99) |
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
