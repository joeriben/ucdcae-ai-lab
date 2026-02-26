# Phase 1: Experience Engine (Tier A)

**Abhaengigkeiten**: Phase 0 (Daemon + Idle-Detection)
**Geschaetzter Umfang**: ~2-3 Wochen

---

## Ziel

ExperienceAggregator verarbeitet die 3.752+ Runs zu strukturierten Erfahrungsdaten. Inkrementell, Idle-getriggert.

## Output-Struktur

```
exports/experience/
  knowledge_base.json         # Aggregierte Regeln & Muster
  config_performance.json     # Statistiken pro Interception-Config
  prompt_patterns.json        # Themencluster, Sprachverteilung
  failure_catalog.json        # Kategorisierte Fehler
  watermark.json              # Letzter verarbeiteter Zeitstempel
  session_summaries/          # Pro-Tag-Zusammenfassungen
    2026-02-25.json
```

## Komponenten

### 1. ExperienceAggregator (`devserver/my_app/services/experience_aggregator.py`)

```python
class ExperienceAggregator:
    def aggregate_new_runs(self, stop_event):
        """Verarbeitet Runs seit letztem Wasserzeichen."""
        watermark = self._load_watermark()
        for run_dir in self._iterate_runs_since(watermark):
            if stop_event.is_set():
                break
            self._process_run(run_dir)
        self._save_watermark()
        self._compute_statistics()
```

### 2. Pro-Run-Extraktion

Aus `metadata.json` + Entity-Files extrahiert:
```json
{
  "run_id": "run_...",
  "timestamp": "2026-02-25T16:43:14",
  "input_lang": "de",
  "prompt_length": 42,
  "interception_config": "19th_century_photography_landscape",
  "output_config": "sd35_large",
  "safety_passed": true,
  "safety_method": "fast_filter",
  "was_favorited": false,
  "had_error": false,
  "seed": 1834098993,
  "is_production": true
}
```

### 3. Production-Filterung

```python
def _is_production_run(self, device_id):
    """Workshop-Runs: Browser-generierte UUIDs. Dev: prefixed IDs."""
    if any(device_id.startswith(p) for p in ['dev_', 'api_', 'canvas_', 'test-']):
        return False
    # UUID4 pattern with date suffix = browser session
    return bool(re.match(r'[0-9a-f]{8}-[0-9a-f]{4}-', device_id))
```

### 4. Config-Performance-Statistiken

```json
{
  "overdrive": {
    "total_runs": 247,
    "production_runs": 189,
    "favorite_rate": 0.12,
    "error_rate": 0.02,
    "avg_regenerations": 1.3,
    "top_output_configs": ["sd35_large", "gpt5_image"],
    "top_input_languages": {"de": 142, "en": 31, "tr": 16}
  }
}
```

### 5. Feedback-Signal-Erweiterung

**Regeneration Detection**: In `schema_pipeline_routes.py`, beim Seed-Tracking:
```python
# Bestehende Logik (schema_pipeline_routes.py:1220-1247) erweitern:
if prompt_changed:
    # Neuer Prompt -> normaler Flow
    pass
else:
    # Gleicher Prompt, neuer Seed -> Regeneration = implizit negativ
    recorder.save_entity("regeneration_signal", "", metadata={"previous_seed": last_seed})
```

**Favorites-Cross-Reference**: `favorites.json` Run-IDs mit aggregierten Runs matchen.

### 6. `aggregate_experience` Skill

```python
class AggregateExperienceSkill(Skill):
    name = "aggregate_experience"
    permission = "AUTO"

    def should_run(self, ctx):
        return ctx.is_idle and self._hours_since_last_run() >= 24

    def execute(self, ctx, stop_event):
        aggregator = ExperienceAggregator()
        return aggregator.aggregate_new_runs(stop_event)
```

## Betroffene Dateien

| Datei | Aktion |
|-------|--------|
| `devserver/my_app/services/experience_aggregator.py` | NEU |
| `devserver/my_app/skills/aggregate_experience.py` | NEU |
| `devserver/my_app/routes/schema_pipeline_routes.py` | EDIT: Regeneration-Signal hinzufuegen |
| `devserver/my_app/services/pipeline_recorder.py` | Lesen (API nutzen) |
| `devserver/my_app/routes/favorites_routes.py` | Lesen (Cross-Reference) |

## Tier A+: Session-Aware Experience Summary (Kern-Innovation)

### Ziel
Das System entwickelt SENSIBILITAET fuer User-Intent — nicht was User tippen, sondern
was sie SUCHEN. Unterscheidung von Modi kreativer Engagement auf Session-Ebene.

### Modi kreativer Engagement

| Modus | Session-Signal |
|-------|---------------|
| Identitaetsausdruck | Persoenliche Referenzen, hohe Regeneration, config-stabil |
| Aesthetische Suche | Prompt-Variation bei gleichem Thema, Favorisierung, Seed-Iteration |
| Dekonstruktion | Wechselnde Configs, ungewoehnliche Prompts, keine Favorites |
| Exploration | Schneller Config-Wechsel, diverse Themen, kurze Verweilzeit |
| Grenzen testen | Safety-nahe Prompts, Provokation, research-Modus |
| Selbst-Artikulation | Prompts werden persoenlicher, emotionale Sprache, lange Verweilzeit |
| Optimierung | Gleicher Prompt, Parametervariationen, systematisch |

### Implementierung: Session-Gruppierung

Runs werden zu Sessions gruppiert (gleiche `device_id` + `date`, chronologisch sortiert).
Pro Session: Prompt-Sequenz, Config-Sequenz, Regeneration-Count, Favorites, Dauer.

### LLM-generierte Narrative Summary

Monatlich (Idle-Skill, CONFIRM): Ein leistungsfaehiges LLM (Claude Sonnet) analysiert
die Session-Daten und generiert eine NARRATIVE Zusammenfassung:

- Welche Modi kreativer Engagement beobachtet werden
- Wiederkehrende Verhaltensmuster (wie Sessions typisch beginnen/enden)
- Paedagogische Beobachtungen (was User SUCHEN, nicht was sie tippen)
- Sensibilitaets-Hinweise (Momente wo User feststecken, Durchbruch-Momente)

Output: `exports/experience/experience_summary.md` (menschenlesbar!)

### Wo das Wissen einfliesst

1. **Trashy System-Prompt**: Aktuelle Summary als Kontext -> kontextbewusste Hilfe
2. **Daemon Skills**: Session-Muster erkennen -> Modelle vorladen, Configs vorschlagen
3. **Config-Generator (Phase 4)**: Unabgedeckte Beduerfnisse identifizieren

### `generate_summary` Skill

```python
class GenerateSummarySkill(Skill):
    name = "generate_summary"
    permission = "CONFIRM"  # Nutzt Cloud-LLM (Kosten), deshalb CONFIRM

    def should_run(self, ctx):
        return ctx.is_idle and self._days_since_last_summary() >= 30

    def execute(self, ctx, stop_event):
        sessions = self._group_runs_into_sessions()
        summary = self._llm_analyze_sessions(sessions)
        self._save_summary(summary)
        self._update_trashy_context(summary)
```

---

## Verification

1. `aggregate_experience` Skill manuell triggern
2. `exports/experience/config_performance.json` enthaelt Statistiken fuer alle 42+ Configs
3. `watermark.json` zeigt letzten verarbeiteten Zeitstempel
4. Erneut ausfuehren -> nur neue Runs verarbeitet (inkrementell)
5. Regeneration-Signal: gleichen Prompt zweimal senden -> Signal in metadata
6. `generate_summary` -> `experience_summary.md` enthaelt narrative Analyse
7. Trashy-Antworten reflektieren das Platform-Wissen
