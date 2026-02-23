# DevServer Architecture

**Part 31: GPU Service — Standalone Inference Server**

---

## Overview

The GPU Service (`gpu_service/`) is a standalone Flask/Waitress process on **port 17803** that handles all local GPU inference. Both dev (17802) and prod (17801) backends call it via HTTP REST. It runs from a shared venv — no separate virtual environment needed.

**Key Principle:** One GPU service serves all environments. Model weights live in the GPU service directory, not per-environment.

---

## Directory Structure

```
gpu_service/
├── app.py              # Flask app factory
├── config.py           # All configuration (models, paths, env vars)
├── server.py           # Waitress entry point
├── routes/
│   ├── diffusers_routes.py
│   ├── heartmula_routes.py
│   ├── mmaudio_routes.py
│   ├── imagebind_routes.py
│   ├── stable_audio_routes.py
│   ├── cross_aesthetic_routes.py
│   ├── text_routes.py
│   └── llm_inference_routes.py    # Production LLM inference (Session 202)
├── services/
│   ├── diffusers_backend.py       # SD3.5, Flux2 image generation
│   ├── heartmula_backend.py       # Music generation (HeartMuLa 3B)
│   ├── mmaudio_backend.py         # Video/Image-to-Audio (MMAudio)
│   ├── imagebind_backend.py       # ImageBind embedding extraction
│   ├── stable_audio_backend.py    # Stable Audio Open
│   ├── cross_aesthetic_backend.py # Cross-aesthetic generation
│   ├── text_backend.py            # LLM introspection (Latent Text Lab)
│   ├── llm_inference_backend.py   # Production LLM inference (Session 202)
│   ├── vram_coordinator.py        # Cross-backend VRAM management
│   └── attention_processors_sd3.py
├── weights/            # MMAudio model weights (NOT in git)
├── ext_weights/        # MMAudio auxiliary weights (NOT in git)
└── .checkpoints/       # ImageBind checkpoint (NOT in git)
```

---

## Model Weight Locations (CRITICAL for Deployment)

### Why weights live inside `gpu_service/`

MMAudio and ImageBind use **relative paths** (`./weights/`, `./ext_weights/`, `./.checkpoints/`) hardcoded in their upstream library code. The GPU service startup script (`2_start_gpu_service.sh`) sets the CWD to `gpu_service/` before launching Python. This means all relative paths resolve from there.

These paths **cannot be configured** without patching the upstream libraries.

### Weight inventory

| Directory | Library | File | Size | Source |
|-----------|---------|------|------|--------|
| `weights/` | MMAudio | `mmaudio_large_44k_v2.pth` | 3.9 GB | HuggingFace: `hkchengrex/MMAudio` |
| `ext_weights/` | MMAudio | `v1-44.pth` | 1.2 GB | HuggingFace: `hkchengrex/MMAudio` |
| `ext_weights/` | MMAudio | `synchformer_state_dict.pth` | 0.9 GB | GitHub: `hkchengrex/MMAudio` releases |
| `.checkpoints/` | ImageBind | `imagebind_huge.pth` | 4.5 GB | Meta: `dl.fbaipublicfiles.com` |

**Total: ~10.5 GB**

### Auto-download behavior

Both libraries attempt to download weights automatically on first use if they are missing:

- **MMAudio**: `model_config.download_if_needed()` in `mmaudio/utils/download_utils.py` checks MD5 hash and downloads from HuggingFace/GitHub if missing.
- **ImageBind**: `imagebind_model.imagebind_huge(pretrained=True)` in `imagebind/models/imagebind_model.py` downloads via `torch.hub.download_url_to_file()` if `.checkpoints/imagebind_huge.pth` is missing.

This means a fresh deployment will auto-download ~10.5 GB on first GPU service request that uses MMAudio or ImageBind. On a school network, this may take considerable time.

### Other model weights (NOT in `gpu_service/`)

These models are loaded via HuggingFace `from_pretrained()` and cached in `~/.cache/huggingface/`:

| Backend | Model | Cache Location | Size |
|---------|-------|----------------|------|
| Diffusers | SD3.5 Large | `~/.cache/huggingface/` | ~16 GB |
| Stable Audio | `stabilityai/stable-audio-open-1.0` | `~/.cache/huggingface/` | ~2.6 GB |
| CLIP Vision | `openai/clip-vit-large-patch14` | `~/.cache/huggingface/` | ~0.6 GB |
| HeartMuLa | HeartMuLa-oss-3B | `~/ai/heartlib/ckpt/` | ~12 GB |

### Deployment: external drive / production copy

When deploying to a separate directory (e.g., `/run/media/.../ai4artsed_production/`), the `gpu_service/` weight directories will be empty because they are gitignored. Options:

1. **Let auto-download handle it** — First start downloads ~10.5 GB. Simple but slow.
2. **Copy the weight files** — Copy `weights/`, `ext_weights/`, `.checkpoints/` from dev to production.
3. **Symlink** — Create symlinks from the production `gpu_service/{weights,ext_weights,.checkpoints}` to the dev copy (only works if both are on the same filesystem or the source is always mounted).

**Recommended for same-machine deployments:** Option 3 (symlinks) avoids 10.5 GB duplication.

**Recommended for external drive / different machine:** Option 1 (auto-download) or Option 2 (copy). Auto-download is simplest for amateur admins — it "just works" on first start, with no manual steps beyond waiting.

---

## Configuration

**File:** `gpu_service/config.py`

All settings use environment variables with sensible defaults. The `_AI_TOOLS_BASE` variable (default: `~/ai`) is the root for locating sibling repos (MMAudio, heartlib, ImageBind).

### Key variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_SERVICE_PORT` | `17803` | Service port |
| `AI_TOOLS_BASE` | `~/ai` | Root for sibling repos |
| `DIFFUSERS_ENABLED` | `true` | Enable Diffusers backends |
| `HEARTMULA_ENABLED` | `true` | Enable HeartMuLa music |
| `HEARTMULA_MODEL_PATH` | `{AI_TOOLS_BASE}/heartlib/ckpt` | HeartMuLa checkpoint |
| `STABLE_AUDIO_ENABLED` | `true` | Enable Stable Audio |
| `MMAUDIO_ENABLED` | `true` | Enable MMAudio |
| `MMAUDIO_REPO` | `{AI_TOOLS_BASE}/MMAudio` | MMAudio repo path |
| `IMAGEBIND_ENABLED` | `true` | Enable ImageBind |
| `CROSS_AESTHETIC_ENABLED` | `true` | Enable cross-aesthetic |
| `TEXT_ENABLED` | `true` | Enable Latent Text Lab |
| `LLM_INFERENCE_ENABLED` | `true` | Enable production LLM inference |

### Sibling repo dependencies (editable installs)

These repos must be installed as editable packages in the shared venv:

| Repo | Install command | Required by |
|------|----------------|-------------|
| `~/ai/MMAudio` | `pip install -e ~/ai/MMAudio` | MMAudio backend |
| `~/ai/ImageBind` | `pip install -e ~/ai/ImageBind` | ImageBind backend |
| `~/ai/heartlib` | `pip install --no-deps -e ~/ai/heartlib` | HeartMuLa backend |

---

## Startup

**Script:** `2_start_gpu_service.sh`

```bash
cd "$SCRIPT_DIR/gpu_service"          # CWD = gpu_service/ (required for relative weight paths)
"$SCRIPT_DIR/venv/bin/python" server.py  # Uses shared venv
```

The `cd gpu_service` is critical — MMAudio and ImageBind resolve weight paths relative to CWD.

---

## VRAM Management

The GPU service uses a `VRAMCoordinator` (singleton) that manages VRAM across all backends. Each backend implements the `VRAMBackend` protocol:

- `get_vram_info()` → current VRAM usage + eviction priority
- `unload()` → release all VRAM
- `is_loaded()` → whether model is in GPU memory

When a backend needs VRAM, the coordinator evicts the lowest-priority loaded backend. See Part 27 for the Diffusers-specific LRU cache details.

---

## Relationship to DevServer

```
┌──────────────────┐     HTTP REST      ┌──────────────────┐
│    DevServer     │ ──────────────────▶ │   GPU Service    │
│  (port 17801/02) │                     │   (port 17803)   │
│                  │                     │                  │
│  Orchestration   │                     │  Inference only  │
│  Safety checks   │                     │  No business     │
│  i18n, pedagogy  │                     │  logic           │
│  Pipeline exec   │                     │                  │
└──────────────────┘                     └──────────────────┘
```

DevServer calls the GPU service via HTTP clients (`DiffusersClient`, `HeartMuLaClient`, `TextClient`, `LLMClient`, etc.) that have identical async method signatures to the original in-process backends — zero changes to callers.

**Fallback:** If the GPU service is down, `is_available()` returns False and the DevServer falls back to ComfyUI/SwarmUI for media, or Ollama for LLM inference (see Part 08: Backend Routing).

### LLM Inference Backend (Session 202)

The `LLMInferenceBackend` routes all production LLM inference (safety, DSGVO, VLM, translation, interception, chat) through the VRAMCoordinator. It replaces direct Ollama calls in 7 DevServer files.

```
DevServer ──→ LLMClient ──→ GPU Service :17803  (safetensors, VRAMCoordinator)
                         ╰→ Ollama :11434       (GGUF, fallback on ConnectionError)
```

**Key differences from TextBackend:**
- No `output_hidden_states`/`output_attentions` (pure inference, not introspection)
- Auto-detects vision vs text models (`AutoModelForVision2Seq` vs `AutoModelForCausalLM`)
- Extracts `<think>...</think>` blocks centrally into separate `thinking` field
- Accepts Ollama-style model names, resolves via `LLM_MODEL_MAP`
- `LLMClient` falls back to Ollama per-call on `ConnectionError` — zero downtime if GPU service restarts

---

## Files

| File | Purpose |
|------|---------|
| `gpu_service/config.py` | All configuration |
| `gpu_service/server.py` | Waitress entry point |
| `gpu_service/app.py` | Flask app + route registration |
| `gpu_service/services/vram_coordinator.py` | Cross-backend VRAM management |
| `gpu_service/services/llm_inference_backend.py` | Production LLM inference backend |
| `gpu_service/routes/llm_inference_routes.py` | LLM inference REST endpoints |
| `2_start_gpu_service.sh` | Startup script (sets CWD) |
| `devserver/config.py` | `GPU_SERVICE_URL`, `GPU_SERVICE_TIMEOUT`, `LLM_SERVICE_PROVIDER` |
| `devserver/my_app/services/diffusers_client.py` | HTTP client (drop-in for DiffusersImageGenerator) |
| `devserver/my_app/services/heartmula_client.py` | HTTP client (drop-in for HeartMuLaBackend) |
| `devserver/my_app/services/text_client.py` | HTTP client (Latent Text Lab) |
| `devserver/my_app/services/llm_client.py` | HTTP client (LLM inference + Ollama fallback) |
| `devserver/my_app/services/llm_backend.py` | LLM client singleton factory |

---

**Document Status:** Active (2026-02-23)
**Maintainer:** AI4ArtsEd Development Team
