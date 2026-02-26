# Agentic AI4ArtsEd: Research Notes

Detaillierte Forschungsergebnisse zu UNO-Framework und Knowledge Distillation via QLoRA.
Referenz-Dokument fuer die Phase-Plaene. Siehe `MASTERPLAN.md` fuer Uebersicht.

---

## 1. Bestehende Experience-Daten (Codebase-Analyse)

### LivePipelineRecorder (Primaerquelle)
- **Location**: `exports/json/YYYY-MM-DD/{device_id}/run_{timestamp}/`
- **Volume**: 3.752 Runs, 7.1 GB
- **Pro Run**: `metadata.json` + `final/` (input, context_prompt, interception, generation_prompt, output) + `prompting_process/` (Iterationen)
- **metadata.json Felder**: run_id, timestamp, config_name, safety_level, device_id, entities[], models_used, interception_config

### Weitere Recorder
- **CanvasRecorder**: Gleiche Struktur, zusaetzlich `node_id`/`node_type`, `workflow.json`
- **LatentLabRecorder**: Frontend-POSTs Parameter + Ergebnisse, richste Parameterdaten
- **Favorites**: `exports/json/favorites.json` — ~20 Eintraege (sparse positive signal)

### Fehlende Feedback-Signale
| Signal | Typ | Status |
|--------|-----|--------|
| Favorites | Explizit positiv | Vorhanden (sparse) |
| Regeneration | Implizit negativ | Nicht getrackt (Seed-Tracking erweitern) |
| Session-Abbruch | Implizit negativ | Nicht getrackt |
| Thumbnail-Dwell | Implizit positiv | Nicht getrackt |
| Export | Explizit positiv | Trackbar |

### Production vs. Dev Filtering
- **Kein `environment`-Feld** in metadata.json
- **Device-ID-Heuristik**: `{uuid4}_{date}` = Browser/Workshop, `dev_*`/`api_*`/`canvas_*` = Testing
- **Empfehlung**: `environment`-Feld hinzufuegen (Zukunft); aktuell manuell filtern

---

## 2. Existierende Agentische Patterns

| Pattern | Ort | Typ |
|---------|-----|-----|
| Wikipedia Research Loop | `pipeline_executor.py:496-720` | ReAct-Agent (LLM emits `<wiki>`, system fetches, re-executes) |
| Canvas Workflow Builder | `canvas_executor.py` | DAG-Execution (50 Exec Limit) |
| Multi-Stage Safety | `stage_orchestrator.py` | Autonome Entscheidungen in Constrained Domain |
| VRAM-Tier Auto-Selection | `schema_pipeline_routes.py` | Environmental Sensing |
| Intelligent Seed Logic | `schema_pipeline_routes.py:1220-1247` | Autonomes State Management |
| Model Fallback Chains | `PromptInterceptionEngine`, `LLMClient` | Resilient Execution |

### 12 LLM-Interaktionspunkte
Stage1: DSGVO-Verify (qwen3:1.7b), Age-Filter (qwen3:4b), VLM-Safety (qwen3-vl:2b)
Stage2: PI (qwen3:4b), Optimization (qwen3:4b)
Stage3: Translation (qwen3:4b)
Stage5: Image Analysis (llama3.2-vision)
Chat: Trashy (qwen3:4b)
Canvas: Node Execution (various)
Training: Auto-Captioning (qwen3-vl:32b + mistral-nemo)
Code: p5.js/Tone.js (codestral)
Research: Latent Text Lab (GPU Service endpoints)

---

## 3. Background Processing Infrastruktur

### Existiert
- SSE mit `user_activity` Tracking (5min Timeout)
- `_last_used` Timestamps auf allen 7 GPU-Backends
- VRAMCoordinator mit LRU-Eviction
- Health-Endpoints (`/api/health`, `/api/health/vram`)
- Threading.Semaphore (Ollama max 3 concurrent)
- Background Threads (Training, Legacy-Migration)

### Existiert NICHT
- Background Scheduler/Timer
- Idle-Detection-Loop
- Task Queue
- Proaktives Model-Unloading
- System-weites Idle-Signal

### Designed aber nicht gebaut
- Async Job Queue (`docs/ASYNC_JOB_QUEUE_PROPOSAL.md`, Session 78)

---

## 4. UNO-Framework Deep Dive

### Was UNO loest
UNO ([arxiv 2602.06470](https://arxiv.org/abs/2602.06470)) verbessert deployed LLM-Systeme mit User-Logs. Kernproblem: Logs sind heterogen und verrauscht — naives Fine-Tuning verschlechtert Performance.

### Algorithmus

```
User Logs (Prompts, Outputs, Feedback)
         |
         v
1. RULE EXTRACTION: LLM destilliert Feedback zu semi-strukturierten Regeln R
         |
         v
2. DUAL-FEATURE CLUSTERING: v_i = [Norm(E(q_i)) + Norm(E(R_i))]
   Agglomerative, Ward-Linkage, epsilon_var = 4
   Encoder: Qwen3-Embedding-0.6B
         |
         v
3. COGNITIVE GAP: g_i = Dist(R_i^LLM, R_i)
   mu_k = mean(g_i fuer i in C_k)
   Threshold tau* = 0.45
         |
    mu_k < 0.45         mu_k >= 0.45
         |                    |
         v                    v
4a. PRIMARY (PEM)      4b. REFLECTIVE (REM)
    Expert LoRA             Critic LoRA
    DPO + NLL               NLL only
    Direct output            2-stage: generate -> critique -> regenerate
```

### Inference-Routing
1. Query -> naechster Cluster-Centroid (Euclidean)
2. Distanz > 1.2: Outlier -> Base-Modell
3. Low Gap: Expert LoRA direkt
4. High Gap: Base generiert -> Critic Feedback -> Base regeneriert

### Training-Hyperparameter (UNO Paper)
- LoRA rank: 64, Dropout: 0.05
- Learning rate: 5e-4, Epochs: 8
- DPO beta: 0.1, Loss: 0.5 DPO + 0.5 NLL
- Judge: Base LLM, 3 Samples, Score 1-10
- WinRate-Threshold: gamma = 0.53

### Mapping auf AI4ArtsEd
| UNO-Konzept | AI4ArtsEd-Aequivalent |
|-------------|---------------------|
| User Query q | Original User-Prompt |
| System Output y | Interception + generiertes Bild |
| User Feedback F | Favorites (+), Regeneration (-), Export (+), Dwell |
| Extracted Rules R | Transformations-Muster pro Config |

### Prototyp-Code: UNO-Lite

#### Schritt 1: Rule Extraction
```python
def extract_rules_from_run(run_metadata, input_text, interception_text):
    prompt = f"""Analyze this successful prompt transformation:
Original: {input_text}
Config: {run_metadata['interception_config']}
Result: {interception_text}
Extract 2-3 rules about WHY this was effective. JSON array of strings."""
    return llm_call(prompt)
```

#### Schritt 2: Clustering
```python
from sklearn.cluster import AgglomerativeClustering
prompt_emb = normalize(t5_encode(input_text))      # [768]
rule_emb = normalize(t5_encode(rules_text))         # [768]
dual_vector = np.concatenate([prompt_emb, rule_emb]) # [1536]
clustering = AgglomerativeClustering(distance_threshold=4.0, linkage='ward')
```

#### Schritt 3: Cognitive Gap
```python
def compute_cognitive_gap(cluster_runs, base_model):
    gaps = []
    for run in cluster_runs:
        predicted = base_model.predict_rules(run.input_text, run.config)
        gap = 1.0 - cosine_similarity(t5_encode(predicted), t5_encode(run.rules))
        gaps.append(gap)
    return np.mean(gaps)
```

### Ressourcen
| Schritt | Compute | VRAM |
|---------|---------|------|
| Rule Extraction (3.752 Runs) | ~2h | 4 GB |
| T5 Embedding | ~5 min | 2 GB |
| Clustering | ~10 sec | 0 GB |
| Cognitive Gap | ~30 min | 4 GB |
| LoRA Training (per Modul) | ~1h | 4 GB |

### Offene Fragen
1. Feedback-Sparsity (~20 Favorites) — Signale erweitern
2. Paedagogik vs. Optimierung — UNO optimiert User-Zufriedenheit, wir brauchen paedagogische Treue
3. 42 Configs als natuerliche Cluster vs. UNO-Subclustering
4. DPO ohne Paarung — synthetische Paare durch Re-Generation

---

## 5. Knowledge Distillation via QLoRA Deep Dive

### Paradigma: Teacher-Student

Claude Sonnet 4.6 (Teacher) generiert Gold-Standard English Transformations.
qwen3:4b (Student) lernt via QLoRA davon.

### English-Channel fuer PI

```
User (DE) -> Stage1 Safety -> Translate to EN (qwen3:4b) -> Stage2 PI (EN, LoRA) -> Stage3 -> Stage4
```

- Translation pro MediaBox (Schalter existiert bereits im Frontend)
- LoRA-Modell nur Englisch -> drastisch einfacheres Training
- Kulturelle Bezuege leben im Config-Kontext (englisch beschrieben), nicht in der Eingabesprache

### Trainings-Daten-Pipeline

#### Quelle A: Random Node (~2.000-3.000 Prompts)
- 3 Presets: `clean_image`, `photo`, `artform`
- LLM generiert frische Prompts (temperaturbasiert)
- `photo`: 15 Film-Typen x diverse Szenen
- Code: `canvas_executor.py:457` (`_execute_random_prompt()`)
- Presets: `canvas_executor.py` RANDOM_PROMPT_PRESETS dict

#### Quelle B: Production Runs (Workshop-Daten only!)
- Device-IDs `{uuid4}_{date}` = Browser Sessions
- AUSFILTERN: `dev_*`, `api_*`, `canvas_*` Prefixes
- Ins Englische uebersetzen falls noetig

#### Claude Sonnet 4.6 Gold-Generation
```python
DIVERSITY_AWARE_SYSTEM_PROMPT = """You are a pedagogical prompt transformer
committed to cultural diversity and equity.
CORE PRINCIPLES:
- NEVER "in the style of" -> artistic PRACTICE, TECHNIQUE, PERCEPTION
- ALL cultural traditions equally valid
- No exoticizing, no eurocentric defaults
- WIE-Regeln: perspective, sensory description, modes of perception
- Cultural references SPECIFIC, not stereotypical"""

# ~4.000 Prompts x 5 Configs = ~20.000 Paare
# Budget: ~$36-90 (Bedrock EU)
```

### Diversitaetssensibles Training (Forschungsbeitrag)

1. **Keine kulturelle Hierarchie**: Yoruba = Gamelan = Flamenco = Electronic
2. **NIEMALS "im Stil von"**: WIE-Regeln (Perspektive, Beschreibung, Wahrnehmung)
3. **Kulturelle Breite**: 15 musikalische Traditionen + visuelle Traditionen
4. **Anti-Default**: "beautiful landscape" != europaeisch, "music" != westlich tonal
5. **Potentiell erstes diversity-aware prompt transformation model** (publizierbar)

### Unsloth QLoRA Setup

```python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    "unsloth/Qwen3-4B-Instruct-unsloth-bnb-4bit",
    max_seq_length=2048, load_in_4bit=True
)
model = FastModel.get_peft_model(model,
    r=32, lora_alpha=64, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", bias="none"
)
# Training: batch=2, grad_accum=8, epochs=3, lr=2e-4, cosine scheduler
# VRAM: ~4 GB peak, ~90 min fuer ~8.000 Samples
```

### Deployment
```
# Ollama Modelfile (zero code-change):
FROM qwen3:4b
ADAPTER ./exports/lora_adapters/ai4artsed_interception_v1_gguf/
PARAMETER temperature 0.7
PARAMETER top_p 0.8
```
-> `qwen3:4b-ai4artsed-v1` in Ollama, config.py: `STAGE2_INTERCEPTION_MODEL_LORA`

### A/B-Testing: 10% LoRA vs. 90% Base
Metriken: Favoriten-Rate, Regeneration-Rate, Safety-Block-Rate, Shannon-Entropie (Mode Collapse), Kontext-Treue

### Risiken
| Risiko | Mitigation |
|--------|------------|
| Mode Collapse | Shannon-Entropie-Monitoring, Sofort-Rollback |
| Paedagogik-Drift | Claude Sonnet als Teacher IST paedagogisch instruiert |
| Overfitting | 8.000+ Samples, Early Stopping, max 3 Epochs |
| Translation-Verlust | Configs ERKLAEREN Kultur; eval mit kulturspez. Testset |
| Safety Regression | Hard-Constraint: >5% Anstieg -> Rollback |

### Qwen3-Spezifisch
- Thinking Mode AUS (enable_thinking=False)
- 4-bit QLoRA (Sweet Spot fuer 4B)
- max_seq_length=2048 (Prompts: 40-300 Tokens)
- train_on_responses_only (+~1% Genauigkeit)

---

## 6. State of the Art Quellen

### Kernreferenzen
- [UNO: Improve LLM Systems with User Logs](https://arxiv.org/abs/2602.06470) — Rule Extraction, Cognitive Gap, PEM/REM
- [ICLR 2026 RSI Workshop](https://recursive-workshop.github.io/) — Recursive Self-Improvement
- [Self-Improving AI Agents (Nakajima)](https://yoheinakajima.com/better-ways-to-build-self-improving-ai-agents/) — 6 Mechanismen

### LoRA/QLoRA
- [Unsloth](https://github.com/unslothai/unsloth) | [Qwen3 Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) | [Hyperparameters](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

### Agentic Patterns
- [Agentic AI Infrastructure 2025-2026](https://medium.com/@vinniesmandava/the-agentic-ai-infrastructure-landscape-in-2025-2026-a-strategic-analysis-for-tool-builders-b0da8368aee2)
- [Self-Healing Infrastructure](https://earezki.com/ai-news/2026-02-23-self-healing-infrastructure-with-agentic-ai-from-monitoring-to-autonomous-resolution/)
- [Meta-Prompting](https://intuitionlabs.ai/articles/meta-prompting-llm-self-optimization)
