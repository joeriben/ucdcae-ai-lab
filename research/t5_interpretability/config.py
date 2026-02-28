"""
T5 Interpretability Research — Shared Configuration

All paths relative to this file. All hyperparams in one place.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SONIFICATION_DIR = DATA_DIR / "sonification"

# Phase 1 outputs
CORPUS_PATH = DATA_DIR / "corpus.json"

# Phase 2 outputs
ACTIVATIONS_POOLED_PATH = DATA_DIR / "activations_pooled.pt"
CORPUS_INDEX_PATH = DATA_DIR / "corpus_index.json"

# Phase 3 outputs
DIMENSION_ATLAS_PATH = DATA_DIR / "dimension_atlas.json"

# Phase 4 outputs
SAE_WEIGHTS_PATH = DATA_DIR / "sae_weights.pt"
SAE_TRAINING_LOG_PATH = DATA_DIR / "training_log.json"

# Phase 5 outputs
FEATURE_ATLAS_PATH = DATA_DIR / "feature_atlas.json"
FEATURE_ATLAS_REPORT_PATH = DATA_DIR / "feature_atlas_report.md"

# Phase 7 outputs
CULTURAL_REPORT_PATH = DATA_DIR / "cultural_analysis_report.md"


# ── Dataset IDs ───────────────────────────────────────────────────────────────

AUDIOCAPS_DATASET = "d0rj/audiocaps"
MUSICCAPS_DATASET = "google/MusicCaps"
WAVCAPS_REPO = "cvssp/WavCaps"
WAVCAPS_FALLBACK_DATASET = "baijs/AudioSetCaps"

# WavCaps JSON files within the repo
WAVCAPS_JSON_FILES = [
    "json_files/AudioSet_SL/as_final.json",
    "json_files/BBC_Sound_Effects/bbc_final.json",
    "json_files/FreeSound/fsd_final.json",
    "json_files/SoundBible/sb_final.json",
]


# ── T5 Encoding ──────────────────────────────────────────────────────────────

T5_MODEL_ID = "google-t5/t5-base"
T5_D_MODEL = 768
T5_MAX_LENGTH = 512
ENCODING_BATCH_SIZE = 64


# ── SAE Hyperparameters ──────────────────────────────────────────────────────

SAE_D_MODEL = T5_D_MODEL          # 768
SAE_N_FEATURES = 6_144            # 8× expansion (16× had 99% dead features)
SAE_K = 64                        # TopK target sparsity (used as L0 target for L1 mode)
SAE_LR = 1e-4
SAE_BATCH_SIZE = 4096
SAE_EPOCHS = 150
SAE_WEIGHT_DECAY = 0
SAE_L1_COEFF = 5e-3               # L1 sparsity coefficient (tuned for L0 ≈ 64)
SAE_L1_WARMUP_EPOCHS = 10         # MSE-only warmup before adding L1


# ── Sonification ─────────────────────────────────────────────────────────────

GPU_SERVICE_URL = "http://localhost:17803"
SONIFICATION_TOP_FEATURES = 50
SONIFICATION_STRENGTHS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SONIFICATION_DURATION_SECONDS = 2.0
SONIFICATION_STEPS = 100
SONIFICATION_CFG_SCALE = 7.0
SONIFICATION_NEUTRAL_PROMPT = "sound"


# ── Probing Corpus ───────────────────────────────────────────────────────────

PROMPTS_PER_TEMPLATE = 50         # 50 prompts per template type per tradition
TEMPLATE_TYPES = 5                # instrument, ensemble, vocal, context, blind
PROMPTS_PER_TRADITION = PROMPTS_PER_TEMPLATE * TEMPLATE_TYPES  # 250
NUM_TRADITIONS = 15
PILLAR1_TOTAL = PROMPTS_PER_TRADITION * NUM_TRADITIONS          # 3750
PILLAR2_TARGET = 2000
CONTROLS_TARGET = 500
