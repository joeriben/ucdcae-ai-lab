# DevServer Implementation TODOs
**Last Updated:** 2026-02-23
**Context:** Current priorities and active TODOs

---

## ✅ Kürzlich erledigt (Kurzreferenz)

| Session | Datum | Was |
|---------|-------|-----|
| 203 | 2026-02-23 | T5 Interpretability Research — 7-Phase Pipeline implementiert (Code complete, Erstdurchlauf steht aus) |
| 202 | 2026-02-23 | LLM Inference Migration — Ollama → GPU Service (6 Phasen, 4 neue Dateien, Ollama-Fallback) |
| 201 | 2026-02-23 | Hebrew + Arabic RTL Support (9 Sprachen, CSS logical properties, 23 LTR-pinned components) |
| 199 | 2026-02-23 | i18n Split — 8275-Zeilen-Monolith → per-language files + Batch Translation Workflow |
| 194 | 2026-02-22 | Forest MiniGame Flowerpot Cursor |
| 193 | 2026-02-22 | Wavetable Synthesis (Crossmodal Lab) |
| 192 | 2026-02-22 | LatentLabRecorder — Research Data Export |
| 190 | 2026-02-21 | Age-Filter Fail-Open Bug Fix + DSGVO Fallback Fix |
| 188 | 2026-02-20 | Crossmodal Lab — MMAudio + ImageBind installiert |

---

## 🔴 CRITICAL: Architektur-Verletzungen

### Proxy-Chunk-Pattern eliminieren

**Datum:** 2026-02-02
**Dokumentation:** `docs/ARCHITECTURE_VIOLATION_ProxyChunkPattern.md`

Das `output_image.json` Proxy-Chunk-Pattern verletzt die 3-Ebenen-Architektur:

- **Soll:** Pipeline entscheidet welche Chunks ausgeführt werden
- **Ist:** Config.OUTPUT_CHUNK entscheidet, Proxy-Chunk routet zu anderem Chunk
- **Scope:** 17 Output-Configs betroffen
- **Lösung:** Pipeline sollte `{{OUTPUT_CHUNK}}` direkt in `chunks` Array verwenden
- **Referenz:** `dual_text_media_generation` Pipeline zeigt korrektes Pattern

### Output-Chunks als Ausführungseinheiten wiederherstellen

**Datum:** 2026-02-02

Output-Chunks wurden zu Metadaten-Containern degradiert statt Ausführungseinheiten zu sein:

- `backend_router.py` enthält Backend-spezifische Logik die in Chunks gehört (`_process_diffusers_chunk()`, `_process_heartmula_chunk()`)
- Neue Backends erfordern Änderungen am zentralen Router statt "einfach einen Chunk hinzufügen"
- **Plan vorhanden:** `docs/plans/diffusers-chunk-migration.md` (User-Approval steht aus)

---

## 🔴 HIGH Priority

### SpaCy Startup-Check + requirements.txt bereinigen

**Datum:** 2026-02-18

Production war ohne SpaCy deployed → DSGVO-Schutz komplett deaktiviert.

1. **Startup-Check**: Prüfen ob SpaCy + 2 Modelle (`de_core_news_lg`, `xx_ent_wiki_sm`) installiert sind → Abbruch bei Fehlen
2. **requirements.txt**: Kommentare aktualisieren (alte 12 Modelle → 2 tatsächlich verwendete)
3. **Installationsskript** für `python -m spacy download`

### LoRA Support for Diffusers GPU Service

**Datum:** 2026-02-17
**Plan:** `docs/plans/lora-diffusers-support.md` (approved)

Diffusers GPU service hat NO LoRA support. `_process_diffusers_chunk` ignoriert `parameters['loras']` komplett.

**Scope (5 files):**
1. `gpu_service/config.py` — `LORA_DIR`
2. `gpu_service/services/diffusers_backend.py` — `_apply_loras()` / `_remove_loras()`
3. `gpu_service/routes/diffusers_routes.py` — `loras` from request
4. `devserver/my_app/services/diffusers_client.py` — `loras` in HTTP payloads
5. `devserver/schemas/engine/backend_router.py` — Extract + fix auto-detection routing

### Video Generation Wan 2.1 — PoC pending

**Datum:** 2026-02-15
**Plan:** `docs/plans/video_generation_wan21_diffusers.md`

Code komplett implementiert. Was noch fehlt:

1. PoC-Test: `venv/bin/python test_wan21_video.py` — wartet auf Modell-Download
2. 1.3B-Modell: Download prüfen (`ls ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/`)
3. Integration Test: GPU Service → DevServer → Video Pipeline end-to-end
4. Frontend: Vue-Komponente für Video-Anzeige

### Canvas Execution Feedback

**Datum:** 2026-01-26 (Session 136)

User starren 5+ Minuten auf den Bildschirm ohne Feedback. Minimum: Spinner/"Generating..." während Execution. Ideal: SSE-Stream mit Live-Updates (Node X von Y, aktueller Stage).

**Technische Optionen:**
- **Option A: Polling** — Frontend pollt `/api/canvas/status/{run_id}` alle 2s
- **Option B: SSE Stream** — Backend sendet `node_started`, `node_completed`, `error` Events

**Betroffene Dateien:**
- `devserver/my_app/routes/canvas_routes.py`
- `public/ai4artsed-frontend/src/stores/canvas.ts`
- `public/ai4artsed-frontend/src/views/canvas_workflow.vue`

### Safety-Architektur Refactoring

**Datum:** 2026-01-26
**Handover:** `docs/HANDOVER_SAFETY_REFACTORING.md`
**Plan:** `~/.claude/plans/wise-napping-metcalfe.md`

1. **KRITISCH: context_prompt nicht geprüft** — User-editierbarer Meta-Prompt wird nirgends safety-geprüft
2. **Namens-Inkonsistenz**: `/interception` macht Stage 1 + Stage 2
3. **Code-Duplikation**: Stage 1 Safety in 4 Endpoints embedded
4. **Frontend Bug**: MediaInputBox ignoriert 'blocked' SSE Events

### Stage 3 Negative Prompts nicht an Stage 4 weitergereicht

**Datum:** 2025-11-20 (Session 53) — **Bug besteht weiterhin** (Stand 2026-02-23 verifiziert)

Stage 3 generiert Negative Prompts basierend auf Safety Level (kids/youth), aber diese werden nie an Stage 4 übergeben:
- `safety_result['negative_prompt']` wird gespeichert aber nie verwendet
- Alle SD3.5 Images nutzen nur den hardcodierten Default: `"blurry, bad quality, watermark, text, distorted"`
- Kids/youth Safety-Filter sind damit nicht voll wirksam

**Fix:** `safety_result['negative_prompt']` an `pipeline_executor.execute_pipeline()` in Stage 4 übergeben.

---

## 🟡 MEDIUM Priority

### ✅ Mistral Large 2411 → 2512 Upgrade

**Datum:** 2026-02-27 — **Erledigt**

Live-Benchmark: 2411 vs 2512 bei Interception-Prompts → gleiche Latenz (~6s), bessere Output-Qualität (kompakter, englisch, stop statt length-cutoff). Upgrade durchgeführt:
- `devserver/hardware_matrix.json` (6 Einträge)
- `devserver/schemas/configs/interception/*.json` (4 Configs: lyrics_from_theme, lyrics_refinement, tag_suggestion_from_lyrics, tags_generation)
- `devserver/my_app/routes/canvas_routes.py` (Canvas Model-Liste)

### source_view in Favorites für korrektes Restore-Routing

**Status:** Implementiert, funktioniert noch nicht — Debugging nötig
**Datum:** 2026-02-14

- `source_view` Feld im gesamten Stack hinzugefügt (Frontend Store → API → JSON → Restore)
- Restore-Routing nutzt `source_view` noch nicht korrekt
- **Dateien:** `src/stores/favorites.ts`, `favorites_routes.py`, alle 5 Views

### Rare Earth Minigame — Phase 4 Testing

**Datum:** 2026-02-03

Code fertig (Phase 1-3 committed), Testing ausstehend:
- [ ] Manual testing im Frontend
- [ ] Balance prüfen (Degradation vs. Cleanup rates)
- [ ] Inactivity timeout (30s) verifizieren
- [ ] Truck animation testen
- [ ] Mobile Responsiveness prüfen
- [ ] Vue type-check

### Latent Lab UX Improvements

**Datum:** 2026-02-21

- ✅ Seed defaults (fixed)
- ✅ Sticky sub-tabs (fixed)
- ✅ Parameter hints (fixed)
- ✅ Crossmodal Lab explanation toggle (fixed)
- [ ] **Streamline "Erweiterte Einstellungen" collapse state** — persist via localStorage
- [ ] **Scientific references with DOI** in all labs (MMAudio=CVPR 2025, ImageBind=CVPR 2023, Stable Audio=ICML 2024)

### MMAudio / ImageBind Downloads ausstehend

**Datum:** 2026-02-20

Bei Festnetz-Verbindung:
- MMAudio-Weights (~3.9 GB) + VAE (~1.2 GB) — laden automatisch beim GPU Service Neustart
- ImageBind Checkpoint: `imagebind_huge.pth` (~4.5 GB) — lädt beim ersten Tab-3-Aufruf
- Gesamt: ~14.5 GB ausstehend

---

## 📋 LOW / PLANNED / DEFERRED

### "optimization" → "adaptation" Rename

**Datum:** 2026-01-29
Alle Backend-Referenzen von "optimization/optimize" zu "adaptation/adapt" umbenennen (Chunk-Dateien + Python-Code).

### Provider Routing ohne Präfix-Parsing

**Datum:** 2026-01-23
Routing basierend auf `EXTERNAL_LLM_PROVIDER` statt Model-String-Präfixen (`openrouter/`). Single Source of Truth.
**Dateien:** `prompt_interception_engine.py`, `settings_routes.py`

### PyTorch Stable Migration

**Datum:** 2026-02-22
Nightly `2.11.0.dev20260203+cu130` → Stable `2.11.0` (sollte seit Feb 16 released sein). Eigene Test-Session planen — nicht nebenbei.

### Internationalization — Primary Language Selector

**Datum:** 2025-11-02
Template-System für Educational Error Messages (hardcoded German → konfigurierbar). Nicht blocking für aktuelles Deployment.

### Debug-Stufen-System

**Plan:** `~/.claude/plans/dynamic-sprouting-stonebraker.md`
**Status:** DEFERRED — wartet auf Safety Regression Fix

### Pipeline-Autonomie Check

**Datum:** 2026-02-02
Prüfen ob in schema-/pipeline-/chunk-bezogenen Python-Files Funktionen absorbiert wurden, die zwischen Pipelines und "ihren" Chunks hätten realisiert werden sollen. Hängt zusammen mit Proxy-Chunk und Output-Chunk Refactoring.

---

## 🎮 Minigames / Waiting Animations

### Design-Prinzipien (übergreifend)

**Kern-Prinzip: "Sisyphus der Systeme"**

Alle Minigames folgen einem gemeinsamen pädagogischen Ansatz:
- **Abwärtsdynamik:** Keine vollständige Heilung möglich
- **User kann handeln:** Aber systemische Zerstörung läuft schneller als individuelle Aktion
- **Sisyphus-Metapher:** Kämpfen gegen eine Übermacht (wie in "Papers, Please", "This War of Mine")

**Offene Designfragen:**
- Resignation vs. Ermächtigung — Braucht es Hoffnungsmomente?
- "Was kann ich wirklich tun?" Sektion nach jedem Spiel?
- Angemessenheit für Kids (8-12) / Youth (13-17)?
- Balance: ehrlicher Realismus vs. pädagogische Ermächtigung?

### Fair Culture (Web Scraping Ethics) — PLANNED

**Datum:** 2026-02-03

Pädagogischer Content über Web-Scraping für generative AI, Künstler-Kompensation.
- Game Mechanic noch nicht designt
- Nächste Schritte: Recherche, Mechanik-Design, Content, Frontend-Integration

---

## 📁 Archiv

- **Sessions 1-14 Full History:** `docs/archive/devserver_todos_sessions_1-14.md`
- **Alle erledigten Items:** Dokumentiert im `DEVELOPMENT_LOG.md` (Sessions 12-203)
- **Architektur-Dokumentation:** `docs/ARCHITECTURE PART 01-20.md`

---

**Created:** 2025-10-26
**Major Cleanups:** 2025-11-02 (Session 14), 2026-02-23 (von ~2100 auf ~220 Zeilen)
