# Phase 5: Knowledge Distillation (Forschungs-Phase)

**Abhaengigkeiten**: Phase 1 (Datenbasis), LoRA-Support-Plan (approved)
**Geschaetzter Umfang**: Forschungs-Phase (open-ended)

---

## Ziel

Diversitaetssensibles Prompt-Transformationsmodell via Knowledge Distillation: Claude Sonnet 4.6 (Teacher) -> qwen3:4b QLoRA (Student). English-Channel fuer PI. Potentiell erstes diversity-aware prompt transformation model.

## Architektur: English-Channel fuer PI

```
User (beliebige Sprache)
  -> Stage 1 Safety
  -> Translate to EN (qwen3:4b, existierend)      [pro MediaBox schaltbar]
  -> Stage 2 PI (EN, LoRA-Modell)                  [Spezialist, nur Englisch]
  -> Stage 3 Safety
  -> Stage 4 Generate
```

Translation-Schalter existiert bereits als Interface-Element in MediaInputBoxes.

## Trainings-Pipeline

### Schritt 1: Prompt-Corpus generieren (~4.000-5.000 EN Prompts)

**Quelle A: Random Node** (Hauptquelle)
- 3 Presets: `clean_image` (1.000), `photo` (1.000), `artform` (1.000)
- Code: `canvas_executor.py:457` RANDOM_PROMPT_PRESETS
- Modell: `local/mistral-nemo` oder `qwen3:4b`
- Token limit: 150 (verbose mode)

**Quelle B: Production Runs** (~1.000)
- Device-IDs mit `{uuid4}_{date}` Pattern (Browser = Workshop)
- Uebersetzen ins Englische falls noetig
- AUSFILTERN: `dev_*`, `api_*`, `canvas_*`

### Schritt 2: Claude Sonnet 4.6 Gold-Transformationen

```python
DIVERSITY_AWARE_SYSTEM_PROMPT = """You are a pedagogical prompt transformer
committed to cultural diversity and equity.

CORE PRINCIPLES:
- NEVER "in the style of" -> artistic PRACTICE, TECHNIQUE, PERCEPTION
- ALL cultural traditions equally valid and sophisticated
- No exoticizing non-Western traditions
- No eurocentric defaults
- WIE-Regeln: perspective, sensory description, modes of perception
- Cultural references SPECIFIC, not stereotypical"""
```

Sampling: ~4.000 Prompts x 2-5 Configs = ~8.000-20.000 Paare
Budget: ~$36-90 (Bedrock EU, Claude Sonnet 4.6)

### Schritt 3: QLoRA Training

```python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    "unsloth/Qwen3-4B-Instruct-unsloth-bnb-4bit",
    max_seq_length=2048, load_in_4bit=True
)
model = FastModel.get_peft_model(model,
    r=32, lora_alpha=64, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", bias="none"
)
# batch=2, grad_accum=8, epochs=3, lr=2e-4, cosine
# VRAM: ~4 GB, Dauer: ~90 min fuer 8.000 Samples
```

Thinking Mode AUS (`enable_thinking=False`).
`train_on_responses_only` = True.

### Schritt 4: Deployment

**Ollama Modelfile** (zero code-change):
```
FROM qwen3:4b
ADAPTER ./exports/lora_adapters/ai4artsed_interception_v1_gguf/
PARAMETER temperature 0.7
PARAMETER top_p 0.8
```
-> `qwen3:4b-ai4artsed-v1` in config.py als `STAGE2_INTERCEPTION_MODEL_LORA`

### Schritt 5: A/B-Testing

10% LoRA vs. 90% Base. Metriken:
- Favoriten-Rate (primaer)
- Regeneration-Rate (negativ)
- Safety-Block-Rate (Hard-Constraint: >5% -> Rollback)
- Shannon-Entropie (Mode Collapse Detection)
- Kontext-Treue (LLM-Judge: wie gut folgt Transformation dem paedagogischen Kontext?)

## Diversitaetssensibles Training

### Was das konkret bedeutet

1. **Keine kulturelle Hierarchie**: 15+ Traditionen gleichberechtigt im Trainingskorpus
   (Ukrainian, Yoruba, Gamelan, Arabic, Jewish, Fraenkisch, Afroamerican, Romani,
   Japanese, Korean, Hindustani, Tuvan, Aboriginal Australian, Flamenco, Electronic)

2. **Anti-Default-Training**: Explizite Beispiele die zeigen:
   - "beautiful landscape" != europaeisch
   - "music" != westliche Tonalitaet
   - "art" != bildende Kunst im Museum

3. **WIE-Regeln statt "im Stil von"**: Alle Trainingsbeispiele verwenden
   Perspektive, Beschreibungsregeln, Wahrnehmungsweisen — keine Stilreferenzen

4. **Evaluation**: Bias-Audit des trainierten Modells
   - Gleicher Prompt -> verschiedene kulturelle Kontexte -> Diversitaet der Outputs?
   - Kein kultureller Kontext -> defaultet das Modell eurozentrisch?

### Forschungsbeitrag

Potentiell das erste **diversity-aware prompt transformation model**:
- Nicht "neutral" (= implizit westlich-normativ)
- Sondern AKTIV diversitaetssensibel trainiert
- Evaluierbar: Base vs. LoRA auf Diversitaetsmetriken
- Publizierbar als Beitrag zu "diversitaetssensible AI in der Bildung"

## Iterativer Zyklus

```
Phase A: Corpus + Gold-Generation + Training + Deploy (10% A/B)
Phase B: 2-4 Wochen Daten sammeln
Phase C: Evaluation (Favoriten, Mode Collapse, Bias-Audit)
Phase D: Refinement (schwache Configs identifizieren, nachtrainieren)
Phase E: UNO-Integration (Cognitive Gap, cluster-spezifische Adapter)
```

## Risiken

| Risiko | Mitigation |
|--------|------------|
| Mode Collapse | Shannon-Entropie, Temperature-Tuning, Rollback |
| Teacher Bias (Claude-Stil) | Diverse Prompts + Temperature 0.7 |
| Translation-Verlust | Configs erklaeren Kultur auf Englisch |
| Overfitting | 8.000+ Samples, max 3 Epochs, Weight Decay |
| Safety Regression | Hard-Constraint im A/B |

## Betroffene Dateien

| Datei | Aktion |
|-------|--------|
| `devserver/my_app/services/distillation_pipeline.py` | NEU: Corpus + Gold-Gen |
| `devserver/my_app/services/lora_trainer.py` | NEU: Unsloth QLoRA Wrapper |
| `devserver/config.py` | EDIT: STAGE2_INTERCEPTION_MODEL_LORA |
| `devserver/my_app/routes/schema_pipeline_routes.py` | EDIT: A/B Model-Selection |
| `gpu_service/services/llm_inference_backend.py` | EDIT: LoRA-Adapter-Support (optional) |
| Ollama Modelfile | NEU |
