# Phase 3: Prompt Intelligence (UNO-Lite)

**Abhaengigkeiten**: Phase 1 (Experience Data)
**Geschaetzter Umfang**: ~3-4 Wochen

---

## Ziel

UNO-inspirierte Rule Extraction aus Erfahrungsdaten. Extrahierte Regeln fliessen als Experience Hints in Stage 2 zurueck. A/B-Framework zum Messen der Wirkung.

## Komponenten

### 1. Rule Extractor (`devserver/my_app/services/rule_extractor.py`)

LLM analysiert erfolgreiche Runs und extrahiert Transformationsregeln:

```python
class RuleExtractor:
    def extract_rules(self, runs_batch, stop_event):
        """Extrahiert Regeln aus einem Batch erfolgreicher Runs."""
        rules = []
        for run in runs_batch:
            if stop_event.is_set():
                break
            prompt = self._build_extraction_prompt(run)
            response = llm_call(prompt, model=config.LOCAL_DEFAULT_MODEL)
            rules.append({"run_id": run.run_id, "config": run.config, "rules": parse_json(response)})
        return rules
```

Output in `exports/experience/knowledge_base.json`:
```json
{
  "rules": [
    {
      "rule": "Landschafts-Prompts mit atmosphaerischen Adjektiven + Licht-Metaphern",
      "evidence": {"runs": 47, "favorites": 8},
      "configs": ["overdrive", "19th_century_photography"],
      "input_pattern": "landscape|Landschaft|Natur"
    }
  ],
  "extracted_at": "2026-02-26T10:00:00",
  "total_runs_analyzed": 189
}
```

### 2. Experience Hints (Stage 2 Integration)

Optionale Kontext-Erweiterung im `manipulate` Chunk-Template:

```
{{TASK_INSTRUCTION}}

{{CONTEXT}}

{{EXPERIENCE_HINTS}}    <-- NEU, optional

{{INPUT_TEXT}}
```

`EXPERIENCE_HINTS` werden nur injiziert wenn:
- A/B-Framework "experiment" Gruppe aktiv
- Relevante Regeln fuer diesen Prompt-Typ existieren (Similarity-Match)

**KRITISCH**: Experience Hints ergaenzen den paedagogischen Kontext, sie ersetzen ihn NICHT. Der `{{CONTEXT}}` (Interception-Config) bleibt unveraendert.

### 3. A/B-Framework

```python
# In schema_pipeline_routes.py:
def assign_ab_group(run_id):
    """Deterministische Zuordnung basierend auf run_id Hash."""
    return "experiment" if hash(run_id) % 10 == 0 else "control"
    # 10% Experiment, 90% Control
```

Geloggt in metadata.json: `"ab_group": "experiment"`, `"experience_hints_applied": true`

### 4. Metriken-Berechnung

Experience Engine (Phase 1) berechnet nach 2-4 Wochen:
- Favoriten-Rate: experiment vs. control
- Regeneration-Rate: experiment vs. control
- Safety-Block-Rate: experiment vs. control (Hard-Constraint)

### 5. Semantic Embeddings (Tier B) — fuer Similarity-Match

T5-Base Encoding aller Prompts (Idle-Skill):
```python
class BuildEmbeddingsSkill(Skill):
    name = "build_embeddings"
    permission = "AUTO"

    def execute(self, ctx, stop_event):
        # T5-Base ist bereits im GPU Service geladen
        prompts = load_all_prompt_texts()
        embeddings = gpu_service_t5_encode(prompts)  # Batch, ~5 min
        np.save("exports/experience/embeddings/prompt_embeddings.npy", embeddings)
```

## Betroffene Dateien

| Datei | Aktion |
|-------|--------|
| `devserver/my_app/services/rule_extractor.py` | NEU |
| `devserver/my_app/skills/extract_rules.py` | NEU |
| `devserver/my_app/skills/build_embeddings.py` | NEU |
| `devserver/schemas/chunks/manipulate.json` | EDIT: `{{EXPERIENCE_HINTS}}` Placeholder |
| `devserver/schemas/engine/chunk_builder.py` | EDIT: Experience Hints Resolution |
| `devserver/my_app/routes/schema_pipeline_routes.py` | EDIT: A/B-Zuordnung + Logging |

## Verification

1. Rule Extraction manuell ausfuehren -> `knowledge_base.json` enthaelt Regeln
2. A/B-Framework: 10 Runs -> 1 hat `ab_group: experiment` in metadata
3. Experiment-Run: `experience_hints_applied: true` in metadata
4. Nach 2 Wochen: `config_performance.json` zeigt experiment vs. control Split
