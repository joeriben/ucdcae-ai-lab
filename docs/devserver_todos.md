# DevServer Implementation TODOs
**Last Updated:** 2026-02-18
**Context:** Current priorities and active TODOs

---

## 📋 TODO: LLM Inference Migration — Ollama → GPU Service

**Status:** 📋 **TODO** — Plan approved, ready for implementation
**Datum:** 2026-02-22
**Priority:** HIGH (Architectural consolidation — single VRAM manager)
**Plan:** `/home/joerissen/.claude/plans/zazzy-exploring-penguin.md`

### Ziel

Alle LLM-Inferenz (Safety, DSGVO, VLM, Translation, Interception, Chat) von Ollama in den GPU Service migrieren. Ein Prozess, ein VRAM-Manager (VRAMCoordinator), ein Modellformat (safetensors). Ollama bleibt als Fallback für ressourcenschwache Deployments.

### Neue Dateien (4)

- `gpu_service/services/llm_inference_backend.py` — Neues Backend (Modell-Loading, Chat/Generate, VRAMCoordinator)
- `gpu_service/routes/llm_inference_routes.py` — 5 Endpoints (/api/llm/chat, /generate, /available, /models, /unload)
- `devserver/my_app/services/llm_client.py` — HTTP-Client mit Ollama-Fallback (wie DiffusersClient)
- `devserver/my_app/services/llm_backend.py` — Singleton-Factory

### 6 Migrationsphasen

1. **GPU Service Infrastruktur** — Backend + Routes + Config
2. **DevServer Client** — LLMClient + Factory + Config
3. **Safety migrieren** — stage_orchestrator.py (DSGVO, Age-Filter), ollama_service.check_safety()
4. **Translation migrieren** — ollama_service.translate_text()
5. **Vision-Modelle** — vlm_safety.py, image_analysis.py (qwen3-vl, llama3.2-vision)
6. **Interception + Chat** — prompt_interception_engine._call_ollama(), chat_routes

### Modell-Mapping (Ollama → HuggingFace)

- `llama-guard3:1b` → `meta-llama/Llama-Guard-3-1B`
- `qwen3:1.7b` → `Qwen/Qwen3-1.7B`
- `qwen3:4b` → `Qwen/Qwen3-4B-Instruct`
- `qwen3-vl:2b` → `Qwen/Qwen2.5-VL-3B-Instruct`
- `llama3.2-vision:latest` → `meta-llama/Llama-3.2-11B-Vision-Instruct`

---

## 📋 TODO: SpaCy Startup-Check + requirements.txt bereinigen

**Status:** 📋 **TODO**
**Datum:** 2026-02-18
**Priority:** HIGH (Production war ohne SpaCy deployed → DSGVO-Schutz komplett deaktiviert)

### Problem
SpaCy ist in `requirements.txt`, aber `pip install -r requirements.txt` installiert nicht die NER-Modelle (`de_core_news_lg`, `xx_ent_wiki_sm`). Production lief ohne SpaCy → kein DSGVO-NER-Check → Namen gingen ungeprüft an externe APIs.

### Aufgaben
1. **Startup-Check** im Backend: Beim Start prüfen ob SpaCy + die 2 erforderlichen Modelle installiert sind. Fehlt was → klare Fehlermeldung (nicht nur Warning, sondern Abbruch oder rote Warnung)
2. **requirements.txt bereinigen**: Kommentare zeigen noch die alten 12 Modelle — aktualisieren auf die 2 tatsächlich verwendeten (`de_core_news_lg`, `xx_ent_wiki_sm`)
3. **Installationsskript oder Post-Install-Hook** für die SpaCy-Modelle (sie werden per `python -m spacy download` installiert, nicht per pip)

---

## 📋 TODO: T5 Audio-Semantic Interpretability Research (SAE)

**Status:** 📋 **TODO** — Plan approved, ready for implementation
**Datum:** 2026-02-22
**Priority:** MEDIUM (Forschungsprojekt, nicht blocking)
**Plan:** `docs/plans/t5_interpretability_research.md`

### Forschungsfrage

Wie organisiert T5-Base semantisches Wissen über Klang in seinem Embedding-Raum (768d)?

### Ansatz

Sparse Autoencoder auf T5-Aktivierungen trainieren (101K Prompts), 768 entangled Dimensionen in ~12K monosemantische Features zerlegen, diese durch Stable Audio sonifizieren.

### Neuartigkeit

- SAE-Features durch Audio-Diffusionsmodell hörbar machen = unerforscht (SAEdit macht es für Bilder)
- T5-Conditioning-Space als Syntheseraum = neues Paradigma (IRCAM RAVE/AFTER nutzen VAE-Latenzräume)

### Prompt-Korpus

- **Bulk** (~95K): AudioCaps + MusicCaps + WavCaps
- **Probing Pillar 1** (3750): 15 Musiktraditionen × 250 Prompts (symmetrisch, keine Hierarchie): Ukrainisch, Yoruba, Gamelan, Arabisch, Jüdisch, Fränkisch, Afroamerikanisch, Romani, Japanisch, Koreanisch, Hindustani, Tuwinisch, Aboriginal Australian, Flamenco, Elektronisch
- **Probing Pillar 2** (2000): Materiell-physikalisch (Anregungsart, Material, Raum, Zeit, Spektrum, Dynamik)
- **Controls** (500): Baselines, Absurdes, Kompositorisch, Minimalpaare

### 7 Phasen

1. Corpus Assembly (`build_corpus.py`)
2. Batch Encoding (`encode_corpus.py`) — ~2 Min, <1 GB VRAM
3. Dimension Atlas (`dimension_atlas.py`) — Alammar-Stil, Sekunden
4. TopK SAE Training (`train_sae.py`) — ~30 Sek, <200 MB VRAM
5. Feature Interpretation (`analyze_features.py`)
6. Sonification (`sonify_features.py`) — ~2 Stunden (Stable Audio Inferenz)
7. Cultural Analysis (`cultural_analysis.py`)

### Constraints

- Keine neuen Dependencies (alles im bestehenden venv)
- Standalone in `research/t5_interpretability/`, nicht in DevServer/GPU Service integriert
- Compute: <3 Stunden total, <3 GB VRAM peak (RTX 6000 Blackwell 96GB)

---

## ✅ DONE: Crossmodal Lab — MMAudio + ImageBind installieren

**Status:** ✅ **DONE** — Session 188 (2026-02-20)
**Datum:** 2026-02-16 (erledigt 2026-02-20)
**Priority:** MEDIUM (Synth-Tab funktioniert bereits, die anderen zwei brauchen externe Pakete)

### Kontext

Crossmodal Lab v2 hat drei Tabs: Latent Audio Synth, MMAudio, ImageBind Guidance.
Synth funktioniert (nutzt nur Stable Audio Pipeline). MMAudio und ImageBind Guidance
brauchen externe Pakete + Modell-Checkpoints, die noch nicht installiert sind.

### Erledigt (Session 188)
- MMAudio: `~/ai/MMAudio` geklont, `--no-deps -e .` installiert (torch-Schutz)
- ImageBind: `~/ai/ImageBind` geklont, `--no-deps -e .` installiert + pytorchvideo
- **WICHTIG**: Beide mit `--no-deps` installiert um torch 2.11.0.dev nightly zu schützen
- torchcodec NICHT installiert (ABI-inkompatibel mit nightly, von MMAudio nicht gebraucht)
- `mmaudio_backend.py` komplett umgeschrieben: Originaler Code nutzte nicht-existentes `load_model` — jetzt korrekte API (`get_my_mmaudio`, `FeaturesUtils`, `FlowMatching`, `eval_utils.generate`)
- Production-Venv (`/run/media/joerissen/production/ai4artsed_production`) ebenfalls aktualisiert
- Model-Checkpoints werden beim ersten Aufruf automatisch heruntergeladen

### Offen
- **MMAudio Checkpoints**: Synchformer (950 MB) + CLIP ViT-H-14 (3.95 GB) heruntergeladen. MMAudio-Weights (~3.9 GB) + VAE (~1.2 GB) stehen noch aus — Download über 5G-Verbindung mit 30 GB Datenlimit pausiert. **Bei Festnetz-Verbindung GPU Service neu starten, Weights laden automatisch nach.**
- **ImageBind Checkpoint**: `imagebind_huge.pth` (~4.5 GB) lädt erst beim ersten Aufruf von Tab 3 (ImageBind Guidance). Ebenfalls auf Festnetz warten.
- **Ollama VRAM-Konflikt**: Ollama belegt ~62 GB VRAM auch ohne aktive Modelle (KV-Cache Vorab-Allokation). Vor MMAudio-Nutzung `sudo systemctl restart ollama` nötig. Langfrist-Lösung: VRAM Coordinator um Ollama-Adapter erweitern (`keep_alive:0` API)
- **Gesamter Download-Bedarf**: ~14.5 GB (MMAudio ~10 GB + ImageBind ~4.5 GB)

### MMAudio (CVPR 2025)

```bash
cd ~/ai && git clone https://github.com/hkchengrex/MMAudio
cd MMAudio && ~/ai/ai4artsed_development/venv/bin/pip install -e .
```

Checkpoints (~6 GB total):
- `mmaudio_large_44k_v2.pth` (3.9 GB)
- `v1-44.pth` (1.2 GB, VAE)
- BigVGAN 16kHz Vocoder (429 MB)
- Synchformer (907 MB)
- CLIP (auto-download ~1.7 GB)

### ImageBind (Meta, 2023)

```bash
~/ai/ai4artsed_development/venv/bin/pip install imagebind-package
```

Oder via Git:
```bash
cd ~/ai && git clone https://github.com/facebookresearch/ImageBind
cd ImageBind && ~/ai/ai4artsed_development/venv/bin/pip install -e .
```

Checkpoint: `imagebind_huge.pth` (~4.5 GB, auto-download)

### Betroffene Dateien

- `gpu_service/services/mmaudio_backend.py` — Import-Pfade ggf. anpassen nach Installation
- `gpu_service/services/imagebind_backend.py` — Import-Pfade ggf. anpassen nach Installation
- `gpu_service/config.py` — `MMAUDIO_REPO` Pfad pruefen

---

## 🟡 WIP: source_view in Favorites für korrektes Restore-Routing

**Status:** 🟡 **IMPLEMENTIERT, FUNKTIONIERT NOCH NICHT** - Debugging nötig
**Datum:** 2026-02-14
**Priority:** MEDIUM

### Was wurde gemacht
- `source_view` Feld im gesamten Stack hinzugefügt: Frontend Store → API → JSON → Restore-Logik
- Alle 5 Views übergeben ihren Route-Pfad beim Favorisieren
- Backend speichert `source_view` in favorites.json und nutzt es im Restore-Handler
- Surrealizer hat Restore-Watcher bekommen
- Fallback-Heuristik für alte Favorites ohne `source_view` bleibt erhalten

### Was noch nicht funktioniert
- Restore-Routing nutzt `source_view` noch nicht korrekt (Debugging nötig)
- Mögliche Ursachen: FooterGallery-Routing, Watcher-Timing, oder Backend-Datenfluss

### Betroffene Dateien
- `src/stores/favorites.ts` — `source_view` in Interface + `addFavorite`/`toggleFavorite`
- `favorites_routes.py` — POST speichert `source_view`, Restore liest es
- Alle 5 Views: text_transformation, image_transformation, surrealizer, music_generation, music_generation_v2

---

## 🔴 CRITICAL REFACTORING: Proxy-Chunk-Pattern eliminieren

**Status:** 🔴 **ARCHITEKTUR-VERLETZUNG** - Refactoring erforderlich
**Datum:** 2026-02-02
**Priority:** HIGH (blockiert saubere Backend-Integrationen)
**Dokumentation:** `docs/ARCHITECTURE_VIOLATION_ProxyChunkPattern.md`

### Problem

Das `output_image.json` Proxy-Chunk-Pattern verletzt die 3-Ebenen-Architektur:

- **Soll:** Pipeline entscheidet welche Chunks ausgeführt werden
- **Ist:** Config.OUTPUT_CHUNK entscheidet, Proxy-Chunk routet zu anderem Chunk

### Betroffene Dateien

- `single_text_media_generation.json` Pipeline
- `output_image.json` Proxy-Chunk
- 17 Output-Configs die das Pattern nutzen

### Lösung

Pipeline sollte `{{OUTPUT_CHUNK}}` direkt in `chunks` Array verwenden - kein Proxy nötig.

**Referenz-Implementierung:** `dual_text_media_generation` Pipeline (HeartMuLa) zeigt das korrekte Pattern.

---

## ✅ GELÖST: Träshy Ruheposition außerhalb Viewport

**Status:** ✅ **GELÖST** - Session 147
**Datum:** 2026-01-29

### Problem (gelöst)
Träshy (ChatOverlay) saß in seiner Ruheposition (collapsed state) halb außerhalb des Browserfensters.

### Lösung
`ChatOverlay.vue`: Positionierungslogik komplett überarbeitet:
- Prozentuale Positionierung durch Pixel-basierte Positionierung mit Clamping ersetzt
- Icon-Größe (100px max) wird jetzt berücksichtigt
- Sowohl collapsed als auch expanded State werden innerhalb des Viewports gehalten
- `minRight/maxRight` und `minBottom/maxBottom` Grenzen eingeführt

---

## 🔴 CRITICAL UX: Canvas Execution Feedback

**Status:** 📋 **TODO** - User erlebt 5 Minuten Stillstand ohne jegliche Rückmeldung
**Datum:** 2026-01-26 (Session 136)
**Priority:** HIGH (Usability-Blocker)

### Problem

Bei komplexen Canvas Workflows starren User 5+ Minuten auf den Bildschirm ohne jegliches Feedback:
- Keine Progress-Anzeige
- Keine Statusmeldungen
- Keine Bubble-Animation WÄHREND der Generierung (nur danach)
- Terminal zeigt Debug-Output, aber Frontend ist stumm

### Anforderungen

IRGENDEINE Form der Rückmeldung implementieren:

1. **Minimum Viable**: Spinner oder "Generating..." Text während Execution
2. **Besser**: Progress-Anzeige pro Node (Node X von Y wird verarbeitet)
3. **Ideal**: SSE-Stream mit Live-Updates vom Backend
   - Welcher Node wird gerade ausgeführt
   - Aktueller Stage (Interception/Translation/Generation)
   - Estimated time remaining (optional)

### Technische Optionen

**Option A: Polling (einfach)**
- Frontend pollt `/api/canvas/status/{run_id}` alle 2 Sekunden
- Backend trackt aktuellen Node in Session/Memory
- Pro: Einfach zu implementieren
- Con: Nicht real-time, zusätzliche Requests

**Option B: SSE Stream (sauber)**
- Backend sendet Events: `node_started`, `node_completed`, `error`
- Frontend zeigt Live-Progress
- Pro: Real-time, elegant
- Con: Mehr Aufwand, SSE-Infrastruktur nötig

**Option C: WebSocket (overkill)**
- Bidirektionale Kommunikation
- Pro: Maximal flexibel
- Con: Unnötig komplex für diesen Use Case

### Betroffene Dateien

**Backend:**
- `devserver/my_app/routes/canvas_routes.py` - Execution mit Progress-Tracking

**Frontend:**
- `public/ai4artsed-frontend/src/stores/canvas.ts` - Progress-State
- `public/ai4artsed-frontend/src/views/canvas_workflow.vue` - Progress-UI

### User-Zitat

> "User starren jetzt bei komplexem run für 5 Minuten auf Bildschirm ohne Aktion, ohne Feedback."
> "Es wäre ja nett zumindest einen gefilterten Stream vom Debug-Output zu sehen."

---

## 🔴 PRIORITY: Safety-Architektur Refactoring

**Status:** 📋 **GEPLANT** - Detaillierter Plan vorhanden
**Datum:** 2026-01-26
**Priority:** HIGH (Sicherheitslücke + Architektur-Inkonsistenz)

**Handover-Dokument:** `docs/HANDOVER_SAFETY_REFACTORING.md`
**Detaillierter Plan:** `~/.claude/plans/wise-napping-metcalfe.md`

### Gefundene Probleme

1. **KRITISCH: context_prompt nicht geprüft** - User-editierbarer Meta-Prompt wird nirgends safety-geprüft
2. **Namens-Inkonsistenz**: `/interception` macht Stage 1 + Stage 2 (nicht nur Interception)
3. **Code-Duplikation**: Stage 1 Safety in 4 Endpoints embedded statt zentral
4. **Frontend Bug**: MediaInputBox ignoriert 'blocked' SSE Events

### Ziel-Architektur

```
NACHHER (sauber getrennt):
├── /pipeline/safety            → NUR Stage 1 + context_prompt Support
├── /pipeline/stage2            → NUR Stage 2 (kein embedded Safety)
├── /pipeline/generation        → Stage 3 + Stage 4 + context_prompt
└── /pipeline/interception      → DEPRECATED
```

### Betroffene Dateien

**Backend:**
- `devserver/my_app/routes/schema_pipeline_routes.py` (5 Stellen)

**Frontend:**
- `public/ai4artsed-frontend/src/views/text_transformation.vue`

### Nächste Schritte

1. Handover lesen: `docs/HANDOVER_SAFETY_REFACTORING.md`
2. Plan lesen: `cat ~/.claude/plans/wise-napping-metcalfe.md`
3. Backend implementieren (Teil 1-4)
4. Frontend implementieren (Teil 5)
5. Verifizieren (4 Tests im Plan)

**Geschätzter Aufwand:** 3-4 Stunden

---

## 🔴 CRITICAL REFACTORING: Output-Chunks als Ausführungseinheiten

**Status:** 🔴 **ARCHITEKTUR-VERLETZUNG** - Output-Chunks wurden zu Metadaten-Containern degradiert
**Datum:** 2026-02-02
**Priority:** HIGH (blockiert neue Backend-Integrationen wie HeartMuLa)

### Das Problem

Output-Chunks sollten die **komplette Ausführung** übernehmen und ein fertiges Produkt liefern. Stattdessen:

1. **Chunks wurden zu reinen Metadaten-Containern** - enthalten nur `input_mappings`, `meta`, etc.
2. **Zentraler Code überladen** - `backend_router.py` hat jetzt `_process_diffusers_chunk()`, `_process_heartmula_chunk()`, etc.
3. **Unmaintainbar** - Für jedes neue Backend muss zentraler Code geändert werden

### Soll-Zustand (Ursprüngliche Architektur)

Output-Chunks enthalten **ausführbaren Code** oder **vollständige Konfiguration**:

| Chunk-Typ | Ausführungs-Config | Beispiel |
|-----------|-------------------|----------|
| ComfyUI | `workflow` | Komplettes ComfyUI-Workflow JSON |
| API | `api_config` | endpoint, headers, request_body |
| Diffusers | `diffusers_config` | model_id, pipeline_class |
| **HeartMuLa** | **FEHLT** | Sollte `heartmula_config` haben |

### Ist-Zustand (Architektur-Verletzung)

- **Diffusers-Chunks** haben zwar `diffusers_config`, aber die Ausführungslogik ist in `backend_router.py:_process_diffusers_chunk()`
- **HeartMuLa-Chunk** hat NUR Metadaten - keine Ausführungskonfiguration
- **Neue Backends** erfordern Änderungen am zentralen `backend_router.py`

### Konsequenz

Diese Praxis bringt das komplette Development durcheinander:
- Entwickler suchen Logik im Chunk, finden sie aber im Router
- Debugging wird schwieriger (Logik verstreut)
- Neue Backends können nicht "einfach einen Chunk hinzufügen"

### Betroffene Dateien

**Zentral (überladen):**
- `devserver/schemas/engine/backend_router.py` - Enthält Backend-spezifische Logik die in Chunks gehört

**Chunks (unvollständig):**
- `devserver/schemas/chunks/output_music_heartmula.json` - Nur Metadaten, keine Ausführungskonfiguration
- `devserver/schemas/chunks/output_image_*_diffusers.json` - Hat Config, aber Logik im Router

### TODO: Diffusers refactoren

Die Diffusers-Chunks haben `diffusers_config`, aber die Ausführungslogik (`_process_diffusers_chunk()`) gehört eigentlich IN den Chunk oder in ein Chunk-spezifisches Modul - nicht in den zentralen Router.

### TODO: HeartMuLa korrekt implementieren

HeartMuLa-Chunk braucht eine vollständige `heartmula_config` die definiert WIE heartmula aufgerufen wird, analog zu `diffusers_config` oder `api_config`.

---

## 🔧 Chck for Need for REFACTORING: Prinzip der Pipeline-Autonomie
Was in den über hundert
    Session ggf. etwas verwässert wurde ich meine Idee der "Binnen-Orchestrierung" der Pipelines in ihrer Domäne. Ich habe sie wie ausführenden Code gedacht, der
     eben diese Interveption-Aufgabe erledigt (input -> komplexer Prozess -> einfacher Output). In dieser Logik würde die Pipeline eben auch rekursive
    Chunk-Aufrufe einfach selbst orchestrieren.

    Ich denke das passt sehr gut zu unserem Paradigmenwechsel vom Backend-Orchestrator zur objektorientierten Frontend/VUE-Prozessinitiierung.

Konkret heißt das für mich, der ich die Codebasis nicht vollständig überblicke: sind ggf in den schema-/pipeline- und chunkbezogenen py-Codefiles ggf. Funktionen absorbiert, die eigentlich zwischen Pipelines und "ihren" Chunks hätten realisiert werden sollen?


## 🎯 CANVAS: Evaluation Nodes - Conditional Execution (Phase 3b) - DONE

**Status:** 🔴 **BLOCKED** - UI works, execution logic missing
**Priority:** HIGH (User-facing feature incomplete)
**Session:** 134
**Estimated Effort:** 2-3 hours

### Context

Evaluation nodes are implemented with 3 separate text outputs (Passthrough, Commented, Commentary), but conditional execution is not yet implemented.

**Current Behavior:**
- ✅ UI works: 3 output connectors (P, C, →)
- ✅ Backend generates 3 text outputs
- ❌ ALL paths execute (no branching logic)

**Expected Behavior:**
- Only active path (P or C) executes downstream
- Commentary path (→) ALWAYS executes
- Based on binary evaluation result

### Technical Requirements

**1. Connection Label Storage**
```typescript
// public/.../types/canvas.ts
interface CanvasConnection {
  sourceId: string
  targetId: string
  label?: 'passthrough' | 'commented' | 'commentary'  // NEW
  active?: boolean  // NEW - Set during execution
}
```

**2. Active Path Marking (Backend)**
```python
# devserver/.../canvas_routes.py
# After evaluation execution
active_path = 'passthrough' if binary_result else 'commented'

# Mark active connections
for conn in connections:
    if conn.sourceId == evaluation_node_id:
        if conn.label == active_path or conn.label == 'commentary':
            conn['active'] = True
        else:
            conn['active'] = False
```

**3. Conditional Execution**
```python
# In execution loop
for node_id in execution_order:
    # Check if all incoming connections are active
    incoming_conns = [c for c in connections if c.targetId == node_id]
    if incoming_conns:
        all_active = all(conn.get('active', True) for conn in incoming_conns)
        if not all_active:
            logger.info(f"[Canvas] Skipping {node_id} (inactive path)")
            continue

    # Execute node
    execute_node(node_id)
```

### Files to Modify

1. `public/ai4artsed-frontend/src/types/canvas.ts`
   - Add `label` and `active` to CanvasConnection

2. `devserver/my_app/routes/canvas_routes.py`
   - Store connection labels in workflow
   - Mark active connections after evaluation
   - Skip nodes on inactive paths

3. `public/ai4artsed-frontend/src/stores/canvasStore.ts` (if exists)
   - Store connection labels when creating connections

### Testing Checklist

- [ ] Evaluation with binary=true → only Passthrough path executes
- [ ] Evaluation with binary=false → only Commented path executes
- [ ] Commentary path ALWAYS executes (regardless of binary)
- [ ] Loop workflow: Input → Interception → Eval → Loop Controller → Interception
- [ ] Multiple evaluations in series (Fairness → Creativity)

### Related Docs

- `docs/ARCHITECTURE_CANVAS_EVALUATION_NODES.md` - Full architecture
- `docs/HANDOVER_SESSION_134.md` - Implementation details

---

## 🔄 CANVAS: Loop Controller Node (Phase 4) - DONE

**Status:** 📋 **PLANNED** - Depends on Phase 3b
**Priority:** MEDIUM (After conditional execution)
**Estimated Effort:** 3-4 hours

### Purpose

Enable feedback loops with max iteration limits for iterative content refinement.

### Features

- Max iterations counter (default: 3)
- Current iteration tracking
- Feedback target node selection (dropdown)
- Termination conditions:
  - `max_iterations` reached
  - `evaluation_passed` (binary=true)
  - `both` (either condition)

### UI Design

```
┌──────────────────────┐
│ 🔄 LOOP CONTROLLER   │
├──────────────────────┤
│ Max Iterations: [3]  │
│ Feedback to: [▼]     │ ← Dropdown of canvas nodes
│ Terminate: [Both ▼]  │
│                      │
│ Current: 0 / 3       │ ← Shows during execution
└──────────────────────┘
```

### Example Workflow

```
Input → Interception → Generation → Quality Eval
         ↑                              ↓ (Commented path)
         └───────── Loop Controller ────┘
                    (max 3 iterations)
```

**Behavior:**
1. Quality Eval fails (score < 7) → Commented output
2. Loop Controller receives Commented text
3. Increment iteration counter (1/3)
4. Check termination: Not reached → continue
5. Route to Interception with feedback
6. Repeat until: Quality passes OR max iterations reached

### Technical Implementation

**Node Type:**
```typescript
type: 'loop_controller'
maxIterations: number
currentIteration: number  // Managed by backend
feedbackTargetId: string  // Node ID to loop back to
terminationCondition: 'max_iterations' | 'evaluation_passed' | 'both'
```

**Backend Logic:**
```python
elif node_type == 'loop_controller':
    # Get iteration counter from state
    iteration = loop_state.get(node_id, {}).get('iteration', 0)
    max_iter = node.get('maxIterations', 3)

    # Check termination
    should_terminate = False

    if node.get('terminationCondition') == 'max_iterations':
        should_terminate = iteration >= max_iter
    elif node.get('terminationCondition') == 'evaluation_passed':
        # Check if previous eval passed
        prev_eval_binary = get_previous_eval_binary(node_id)
        should_terminate = prev_eval_binary
    elif node.get('terminationCondition') == 'both':
        should_terminate = iteration >= max_iter or get_previous_eval_binary(node_id)

    if should_terminate:
        # Exit loop - continue to next node
        logger.info(f"[Loop] Terminating after {iteration} iterations")
        results[node_id] = {'type': 'loop_exit', 'output': input_text}
    else:
        # Continue loop - re-queue feedback target
        iteration += 1
        loop_state[node_id] = {'iteration': iteration}
        logger.info(f"[Loop] Iteration {iteration}/{max_iter}")

        # Re-queue feedback target for execution
        feedback_target = node.get('feedbackTargetId')
        execution_queue.append(feedback_target)

        results[node_id] = {'type': 'loop_continue', 'output': input_text, 'iteration': iteration}
```

### Challenges

**Re-queueing Nodes:**
- Current execution uses topological sort (DAG)
- Loops create cycles → need execution queue instead
- Or: Special handling for loop edges (ignore in topo sort)

**State Management:**
- Iteration counter must persist across node executions
- Per-loop state dict: `{loop_controller_id: {iteration: N}}`

**Infinite Loop Prevention:**
- Hard limit: Max 10 iterations (configurable)
- Timeout: Max execution time per workflow

### Files to Modify

1. `public/.../types/canvas.ts` - loop_controller type
2. `public/.../StageModule.vue` - Loop controller UI
3. `public/.../ModulePalette.vue` - Add to palette
4. `devserver/.../canvas_routes.py` - Loop execution logic

---

## 🔧 REFACTORING: Provider Routing ohne Präfix-Parsing

**Status:** 📋 **PLANNED** - Aktuelle Lösung funktioniert, aber fragil
**Reported:** 2026-01-23
**Priority:** MEDIUM (Verbesserung, kein Blocker)

### Problem

Die aktuelle Provider-Detection in `prompt_interception_engine.py` basiert auf Model-String-Präfixen:

```python
# Aktuell (fragil)
if model.startswith("openrouter/"):
    return call_openrouter(model.removeprefix("openrouter/"))
elif model.startswith("anthropic/"):
    return call_anthropic(model)
```

**Schwachstellen:**
1. **Doppelte Wahrheit**: `EXTERNAL_LLM_PROVIDER` sagt "openrouter", aber Routing prüft Model-String
2. **Leicht zu vergessen**: Commit 6eec19d entfernte versehentlich `openrouter/` Präfixe
3. **Unübersichtlich**: `openrouter/anthropic/claude-sonnet-4.5` ist redundant

### Saubere Lösung

Routing basierend auf `EXTERNAL_LLM_PROVIDER` (Single Source of Truth):

```python
# Nachher (robust)
provider = config.EXTERNAL_LLM_PROVIDER
if provider == "openrouter":
    return call_openrouter(model)  # model = "anthropic/claude-sonnet-4.5"
elif provider == "anthropic":
    return call_anthropic(model)   # model = "anthropic/claude-3-5-sonnet-latest"
```

### Vorteile

- Model-Namen bleiben sauber (ohne `openrouter/` Präfix)
- Provider-Einstellung bestimmt Routing
- Weniger fehleranfällig bei Config-Änderungen
- `HARDWARE_MATRIX` Presets brauchen keine redundanten Präfixe

### Betroffene Dateien

1. `devserver/schemas/engine/prompt_interception_engine.py` - Routing-Logik
2. `devserver/my_app/routes/settings_routes.py` - HARDWARE_MATRIX (Präfixe entfernen)
3. Ggf. weitere Provider-Detection in anderen Dateien

### Aufwand

~2-3 Stunden (inkl. Testing)

---

## ✅ COMPLETED (2026-01-17): Unified Export Across Lab Endpoints

**Status:** ✅ **COMPLETE** - All backends tested and working
**Commit:** `7f07197`

### What Was Fixed

**Problem:** Export function BROKEN - entities scattered across multiple folders
- `/interception` created `run_xxx/` with input, safety, interception
- `/generation` created `run_yyy/` with output_image
- Result: Incomplete exports, no unified research data

**Solution:** Frontend passes `run_id` from `/interception` to `/generation`
- Backend loads existing Recorder via `load_recorder()`
- All entities saved to ONE unified folder

### Additional Fixes

**Multi-Backend Image Saving:**
- ❌ Before: Only SD3.5 (`swarmui_generated`) saved images
- ✅ After: ALL backends work:
  - SD3.5: Unchanged (via SwarmUI API)
  - QWEN/FLUX2: Read from `filesystem_path`
  - Gemini/GPT-Image: Decode from base64

**Files Modified:**
- `schema_pipeline_routes.py`: `load_recorder` import, filesystem_path + base64 handling
- `text_transformation.vue`: Pass `run_id` to /generation

### Architectural Cleanup ✅ COMPLETE (2026-01-17)

**Completed:** Eliminated `image_workflow` type - all image models now use `media_type: "image"`
- Changed `output_image_qwen.json`: `image_workflow` → `image`
- Changed `output_image_flux2.json`: `image_workflow` → `image`
- Simplified `backend_router.py` line 749

---

## 🚧 IN PROGRESS (Session 98 - 2025-12-13)

### Two Open Issues - partial_elimination
**Status:** ⚠️ **INCOMPLETE** - Two related issues need resolution
**Priority:** HIGH (composite important, routing required)
**Sessions:** 97 (abandoned), 98 (redesign + partial fixes)

**⚠️ READ THIS FIRST:** `docs/HANDOVER_Session_98_Two_Open_Issues.md`

### Issue #1: Composite Image Not Created (Backend)

**Problem:** Backend doesn't create composite image (code was reverted in Session 97)

**Solution:** Re-add 36 lines after `schema_pipeline_routes.py:2016`
```python
if len(media_files) > 1:
    composite_data = recorder.create_composite_image(...)
    recorder.save_entity(entity_type='output_image_composite', ...)
```

**Effort:** 5 minutes

### Issue #2: Vue Not in /execute/ Path (Frontend)

**Problem:** `partial_elimination.vue` in wrong location
- Current: `src/views/partial_elimination.vue`
- Expected: `src/views/execute/partial_elimination.vue` (or similar)
- Missing: Stage2 proxy config for routing

**Solution:**
1. Investigate existing `/execute/` routing pattern
2. Create proxy config (if needed)
3. Move Vue file to correct location
4. Update router

**Effort:** 30-60 minutes

### What Works Already ✅

- ✅ Frontend redesigned (design standards compliant)
- ✅ Dual-fetch logic (individuals + composite)
- ✅ New backend endpoint: `/api/pipeline/<run_id>/file/<filename>`
- ✅ Flexible 3-4 image grid layout
- ✅ All standard actions functional

**See:** Full details in `docs/HANDOVER_Session_98_Two_Open_Issues.md`

---

## ✅ COMPLETED (Session 91 - 2025-12-09)

### Model Availability Check - API-Based
**Status:** ✅ **COMPLETED** - All models correctly detected, frontend filters by availability
**Session:** 91 (2025-12-09)
**Duration:** ~95 minutes

**Implemented:**
- Backend: ModelAvailabilityService queries ComfyUI `/object_info` API
- Endpoint: `GET /api/models/availability` returns availability map
- Frontend: text_transformation.vue filters configs by availability
- Result: wan22_video and stableaudio_open correctly shown (Session 90 failed at this)

---

## 📋 FOLLOW-UP (Session 91+)

### Automate configsByCategory in text_transformation.vue
**Status:** 📋 **PLANNED** - After Session 91 success
**Priority:** MEDIUM
**Context:** User requested this AFTER model availability check working

**Current Problem:**
`text_transformation.vue` has hardcoded model lists:
```typescript
const configsByCategory = {
  image: [
    { id: 'flux2', label: 'Flux 2', ... },  // Hardcoded
    { id: 'sd35_large', label: 'SD 3.5', ... },  // Hardcoded
  ],
  video: [...],  // Hardcoded
  sound: [...]   // Hardcoded
}
```

**Desired Behavior:**
- Fetch all output configs from backend dynamically
- Categorize automatically by `media_preferences.default_output` (image/video/sound)
- Apply availability filter (from Session 91)
- New models appear automatically without Vue changes

**Implementation Plan:**

1. **Backend: Extend `/api/models/availability` endpoint** (or create new endpoint)
   - Return not just `{config_id: boolean}` but full metadata:
     ```json
     {
       "configs": [
         {
           "id": "flux2",
           "name": {"en": "Flux 2", "de": "Flux 2"},
           "media_type": "image",
           "logo": "/logos/flux_logo.png",
           "color": "#FF6B35",
           "available": true
         },
         ...
       ]
     }
     ```
   - Use existing `/pipeline_configs_with_properties` as reference
   - Filter to only Stage 4 output configs

2. **Frontend: Replace hardcoded configsByCategory**
   - Fetch configs on mount
   - Group by `media_type` (image/video/sound)
   - Apply availability filter
   - Map to existing Config interface format

3. **Benefits:**
   - Add new model in backend → automatically appears in UI
   - Consistent with Phase 1 property-based config loading
   - Logo, color, name changes centralized in backend configs

**Files to Modify:**
- `devserver/my_app/routes/config_routes.py` (extend /api/models/availability or new endpoint)
- `public/ai4artsed-frontend/src/views/text_transformation.vue` (replace configsByCategory)
- `public/ai4artsed-frontend/src/services/api.ts` (add interface for extended response)

**Testing:**
- All current configs still appear correctly
- Add new test config → appears without Vue changes
- Availability filter still works
- Logo and colors render correctly

---

## ✅ FIXED BUG (Session 135 - 2026-01-24)

### Prompt Optimization META-Instruction
**Status:** ✅ **FIXED** - Session 135
**Priority:** HIGH
**Reported:** 2025-11-26 | **Fixed:** 2026-01-24

**Problem:**
Die `prompt_optimization` META-Instruction in `instruction_selector.py` hatte hardcodierte kulturelle Mappings ("qiyun shengdong → dynamic brushstrokes") und ein image-spezifisches Output-Format.

**Root Cause:**
Die META-Instruction versuchte, kreative Interception-Outputs in fixierte Bildanweisungen zu hardcodieren. Die eigentlichen Regeln stehen bereits in den Output-Chunk `optimization_instruction` Feldern.

**Fix:**
Vereinfachte `prompt_optimization` zu einer simplen "Wende Context-Regeln an" Instruction:
```python
"prompt_optimization": {
    "description": "Apply media-specific optimization rules from Context",
    "default": """Transform the Input according to the rules in Context.

The Context contains media-specific transformation rules.
Apply these rules precisely to the Input.
Preserve the language of the Input.

Output ONLY the transformed result.
NO meta-commentary, NO headers, NO formatting."""
}
```

**Files Changed:**
- `devserver/schemas/engine/instruction_selector.py` (Zeilen 27-37)

**See:** `docs/HANDOVER_2026-01-24_Session134_OptimizationProblem.md` für Details

---

## 🔄 CURRENT WORK (Session 59)

### Stage 1-3 Translation Refactoring - **PLANNED**
**Status:** 📋 **READY TO IMPLEMENT** - Architecture corrected, clean develop branch
**Session:** 59 (2025-11-21)
**Priority:** HIGH (corrects flawed Session 56-58 mega-prompt architecture)

**The Problem:**
Session 56-58 planned to "eliminate Stage 3" by merging it into Stage 2 as a "mega-prompt" with built-in safety. This was a **pedagogical error** - users need to edit AFTER optimization but BEFORE final safety check. The architecture needs correction.

**The Correction:**
```
Stage 1: Safety ONLY (NO translation, bilingual DE/EN)
Stage 2: Interception + Optimization (in original language)
  → USER CAN EDIT HERE!
Stage 3: Translation (DE→EN) + Safety
Stage 4: Media Generation
```

**Implementation Tasks:**

1. **Stage 1: Remove Translation** (HIGH priority)
   - Create `/devserver/schemas/configs/pre_interception/gpt_oss_safety_only_bilingual.json`
   - Add `execute_stage1_safety_only_bilingual()` to `stage_orchestrator.py`
   - Update `/stage2` endpoint (line 252) to use safety-only function
   - Update `/execute` endpoint (line 541) to use safety-only function
   - Test: German input should NOT be translated in Stage 1

2. **Stage 2: Add Media Optimization** (MEDIUM priority)
   - Option A: Create optimization chunks (`optimize_image.json`, `optimize_audio.json`)
   - Option B: Add optimization to `manipulate.json` template
   - Update pipeline configs to include optimization step
   - Test: Output should be optimized for target media type

3. **Stage 3: Add Translation** (HIGH priority)
   - Create `/devserver/schemas/configs/pre_output/translation_de_en_stage3.json`
   - Add `execute_stage3_translation()` to `stage_orchestrator.py`
   - Update Stage 3-4 loop in `schema_pipeline_routes.py` (around line 700-800)
   - Add translation BEFORE safety check
   - Test: German text should be translated before media generation

4. **Frontend: Remove Context Translation** (LOW priority)
   - Check if `Phase2CreativeFlowView.vue` translates context before sending
   - Remove translation if present
   - Test: Frontend sends context in original language

5. **Testing & Validation** (HIGH priority)
   - Test complete flow: DE input → Safety → Interception → Edit → Translation → Safety → Media
   - Test user context rules: "alle mäuse haben ROTEN HUT" should be preserved
   - Test bilingual: Both DE and EN inputs should work
   - Test all safety levels: kids, youth, off

**Files to Create:**
- `devserver/schemas/configs/pre_interception/gpt_oss_safety_only_bilingual.json`
- `devserver/schemas/configs/pre_output/translation_de_en_stage3.json`
- (Optional) `devserver/schemas/chunks/optimize_image.json`

**Files to Modify:**
- `devserver/schemas/engine/stage_orchestrator.py` (add 2 functions)
- `devserver/my_app/routes/schema_pipeline_routes.py` (Stage 1 and Stage 3 calls)
- (Check) `public/ai4artsed-frontend/src/views/Phase2CreativeFlowView.vue`

**Documentation Updated:**
- ✅ ARCHITECTURE PART 01 (Version 2.1) - Reflects correct Stage 1-3 flow
- ✅ DEVELOPMENT_DECISIONS.md (Active Decision 1) - Explains why Session 56-58 was wrong

**Estimated Time:** 3-4 hours
**Complexity:** Medium (clear plan, well-understood changes)

---

## 🔄 PREVIOUS WORK (Session 52)

### Youth Mode & Pipeline Visualization - **IN PROGRESS**
**Status:** 🟡 **PARTIAL IMPLEMENTATION** - Core components ready, integration needed
**Session:** 52 (2025-11-20)
**Priority:** MEDIUM (pedagogical feature for ages 13-17)

**What Was Successfully Completed:**
1. ✅ **Admin Configuration System** (`devserver/config.py`)
   - `UI_MODE` setting: "kids" (8-12), "youth" (13-17), "expert" (teachers/devs)
   - `DEFAULT_SAFETY_LEVEL` setting: "kids", "youth", "adult", "off"
   - Expanded SAFETY_NEGATIVE_TERMS with detailed categorization
   - Clear documentation for admins

2. ✅ **Backend Pipeline Visualization Metadata** (`schema_pipeline_routes.py:823-905`)
   - `_get_step_icon()`: Returns emoji for chunk types (📝, 🎨, 🖼️)
   - `_get_child_friendly_label()`: Returns 2-3 word labels in DE/EN
   - `_get_pipeline_visualization_metadata()`: Reads pipeline structure from schemas
   - Enhanced `/api/schema/pipeline/execute` response with visualization data

3. ✅ **Frontend PipelineFlowVisualization Component**
   - File: `public/ai4artsed-frontend/src/components/PipelineFlowVisualization.vue` (~550 lines)
   - Horizontal flow: 💡 Input → 📝 Understanding → 🎨 Visuals → 🖼️ Image → [Result]
   - Status indicators: ⚙️ (running), ✓ (complete), ✗ (failed)
   - Responsive: horizontal (desktop), vertical (mobile)
   - Accessibility: high contrast, reduced motion support

4. ✅ **Frontend TypeScript Types** (`api.ts:70-122`)
   - `ExecutionStep`, `PipelineStepDefinition`, `PipelineVisualization` interfaces
   - Extended `PipelineExecuteResponse` with visualization metadata

5. ✅ **i18n Translations** (`i18n.ts:137-141, 276-280`)
   - Pipeline labels in German and English

**What's Broken/Missing:**
1. **🟡 Transform API Visualization** - **ROLLED BACK** due to 500 errors
   - Attempted to extend `/api/schema/pipeline/stage2` with pipeline_visualization
   - Caused AttributeError (needs careful testing)
   - Transform-only configs (dada, bauhaus) never show visualization currently

2. **🟡 UI Mode System Not Wired Up**
   - `UI_MODE` in config.py but not consumed anywhere
   - Need to add uiMode to userPreferences store
   - Need to add UI Mode toggle button in Phase2CreativeFlowView
   - Need conditional rendering: `v-if="uiMode !== 'kids'"`

3. **🟡 Safety Level Not Used**
   - `DEFAULT_SAFETY_LEVEL` defined but system still uses request-level safety_level
   - Need to wire up config value to Stage 3 safety checks

4. **🟡 Visualization Placement Limited**
   - Currently only works in media generation overlay (Phase 3/4)
   - Transform-only workflows never trigger `isGenerating`
   - Need visualization during Transform (Phase 2a) as well

**Architecture Understanding (Critical):**
- **Frontend Phases**: Property Selection → Creative Flow → Transform (2a) → Media Selection (2b) → Generation (3/4)
- **Backend Stages**: All pipelines go through 4 stages (some skip Stage 4 if no output_config)
- **Kids Mode (8-12)**: Simple, separated phases - NO pipeline visualization
- **Youth Mode (13-17)**: Educational flow WITH pipeline visualization - show HOW AI works
- **Expert Mode**: Full debug mode with all metadata

**Next Steps for Implementation:**
1. **Add uiMode to userPreferences store**
   - State: `uiMode` ("kids"/"youth"/"expert")
   - Functions: `setUiMode()`, `cycleUiMode()`
   - Persisted in localStorage

2. **Add UI Mode toggle** (Youth Mode implementation)
   - Button in top bar of Phase2CreativeFlowView
   - Cycles through modes: kids → youth → expert

3. **Extend Transform API** (CAREFULLY! Test incrementally!)
   - Add pipeline_visualization to `/api/schema/pipeline/stage2` response
   - Test thoroughly before deploying
   - Check for attribute errors on `config.pipeline_name`

4. **Show visualization in main layout**
   - After result panel, conditional: `v-if="pipelineVisualization && uiMode !== 'kids'"`
   - During transform AND after completion
   - Separate from media generation overlay

5. **Wire up DEFAULT_SAFETY_LEVEL**
   - Use config.DEFAULT_SAFETY_LEVEL in Stage 3 if no request-level override
   - Ensure backwards compatibility

**Files Modified (Session 52):**
- ✅ `devserver/config.py`: ADMIN section (KEPT after rollback)
- ⏸️ `devserver/my_app/routes/schema_pipeline_routes.py`: Transform changes (ROLLED BACK)
- ⏸️ `public/ai4artsed-frontend/src/services/api.ts`: TransformResponse (ROLLED BACK)
- ⏸️ `public/ai4artsed-frontend/src/stores/userPreferences.ts`: uiMode system (ROLLED BACK)
- ⏸️ `public/ai4artsed-frontend/src/views/Phase2CreativeFlowView.vue`: UI toggle (ROLLED BACK)

**Estimated Completion Time:** 4-6 hours
**Complexity:** Medium-High (requires careful Transform API work)

---

## 🚨 CRITICAL BUGS - HIGH PRIORITY

### Stage 3 Negative Prompts Not Passed to Stage 4 - **BUG**
**Status:** 🔴 **CRITICAL BUG** - Stage 3 safety negative prompts are ignored
**Discovered:** 2025-11-20 (Session 53)
**Priority:** HIGH (affects all image generation quality and safety)

**Problem:**
Stage 3 generates appropriate negative prompts based on safety level (kids/youth/off), but these are **never passed to Stage 4** (image generation). This means:
- Safety-appropriate negative prompts are ignored
- All SD3.5 images use only the hardcoded default: `"blurry, bad quality, watermark, text, distorted"`
- Kids/youth safety filters generate comprehensive negative prompts but they're discarded

**Current Behavior:**
1. **safety_level='off'**: Only basic quality negatives applied
2. **safety_level='kids'**: Stage 3 generates `"violence, violent, killing, murder, death, blood, gore, horror, scary, demon, evil, nude, naked, nsfw, sexual, abuse"` → **IGNORED**
3. **safety_level='youth'**: Stage 3 generates `"explicit, hardcore, brutal, pornographic, sexual, nsfw, rape, abuse, self-harm, suicide"` → **IGNORED**

**Root Cause:**
- `schema_pipeline_routes.py:862-867`: Stage 4 execution does not pass `negative_prompt` parameter
- `safety_result['negative_prompt']` from Stage 3 is stored but never used
- `backend_router.py:366` falls back to chunk default when no negative_prompt provided

**Files Involved:**
- `devserver/my_app/routes/schema_pipeline_routes.py:770-867` (Stage 3-4 integration)
- `devserver/schemas/engine/stage_orchestrator.py:477-605` (Stage 3 execution)
- `devserver/schemas/engine/backend_router.py:366` (negative_prompt fallback)
- `devserver/schemas/chunks/output_image_sd35_large.json:140` (hardcoded default)

**Required Fix:**
1. Pass `safety_result['negative_prompt']` to `pipeline_executor.execute_pipeline()` in Stage 4
2. Add `negative_prompt` parameter to Stage 4 execution call
3. Ensure backward compatibility (default to chunk default if Stage 3 skipped)
4. Test with all safety levels (off, kids, youth)

**Impact:**
- Image quality may be affected (missing quality-related negatives when safety=off)
- Safety filters not fully effective (unwanted content may appear despite Stage 3 checks)
- Kids/youth modes not properly protected

**Estimated Fix Time:** 1-2 hours
**Testing Required:** All safety levels + all output configs (SD3.5, FLUX, GPT-5)

---

## 🔧 CONFIGURATION & REFACTORING

### Stage 2 Model Selection in config.py - **COMPLETED** ✅
**Status:** ✅ **IMPLEMENTED** - STAGE2_MODE switch added to config.py
**Session:** 53 (2025-11-20)
**Priority:** MEDIUM (enables testing Claude Sonnet vs GPT-OSS 20b)

**What Was Implemented:**
1. ✅ Added `STAGE2_MODE` to `config.py` ("local" | "remote")
   - `local`: GPT-OSS 20b (Ollama, DSGVO-compliant, free)
   - `remote`: Claude Sonnet (OpenRouter, high quality, ~$3/M tokens)
2. ✅ Modified `chunk_builder.py` to read `STAGE2_MODE` for manipulate chunk
   - Overrides model and backend_type based on config
   - Logs which mode is active
3. ✅ Added comment to `manipulate.json` documenting override behavior

**Future Enhancement:**
- **TODO:** Make ALL model selections configurable in `config.py`
  - Currently: Only Stage 2 (manipulate chunk) is configurable
  - Goal: Allow admin to configure models for all stages/chunks in config.py
  - Example: `STAGE1_TRANSLATION_MODEL`, `STAGE1_SAFETY_MODEL`, `STAGE3_SAFETY_MODEL`, etc.
  - Benefit: Easy A/B testing, no code changes needed for model experiments

**Files Modified:**
- `devserver/config.py`: Added STAGE2_MODE (line 76)
- `devserver/schemas/engine/chunk_builder.py`: Added STAGE2_MODE logic (lines 179-200, 235-236)
- `devserver/schemas/chunks/manipulate.json`: Added documentation comment (line 5)

---

## ⚠️ DEACTIVATED FEATURES (Session 40)

### Vector Fusion UI - **INCOMPLETE / DEACTIVATED**
**Status:** 🔴 **DEACTIVATED** - Multiple unresolved bugs, low priority
**Session:** 40 (2025-11-09)
**Priority:** LOW (relatively unimportant feature per user)

**What Was Implemented:**
- ✅ **Complete UI**: Bubble-based flow visualization with 3 stages
  - Centered input bubble
  - Split flow visualization (Part A + Part B branches)
  - 4-image generation grid (Original, Part A, Part B, Fusion)
  - Vertical scrolling support
- ✅ **Frontend Components**: Phase2VectorFusionInterface.vue (~400 lines)
- ✅ **Pipeline Type Detection**: Frontend correctly detects `text_semantic_split` pipelines
- ✅ **API Integration**: executePipeline() calls working

**What Works:**
- UI renders correctly with bubble flow
- Split operation executes (LLM semantic splitting)
- Frontend receives split results (part_a, part_b)

**What's Broken:**
1. **🔴 Wrong Language Split**: German text is being split instead of English (translated text)
   - Split should operate on English text (after Stage 1 translation)
   - Currently operates on German original input
   - Root Cause: Stage 1→2 data flow passes wrong text field

2. **🔴 Image Generation Fails**: Backend uses deprecated workflow generator
   - Backend logs show: "Using deprecated comfyui_workflow_generator"
   - Ignores `output_vector_fusion_clip_sd35` chunk defined in pipeline
   - Workflow submitted to ComfyUI but backend doesn't wait for completion
   - Result: Frontend shows 4 image frames with question marks (no images)
   - Root Cause: Migration from deprecated generator to Output-Chunks incomplete

3. **🔴 Image Retrieval Missing**: No polling or download of generated images
   - Backend submits workflow successfully
   - ComfyUI generates images (visible in ComfyUI interface)
   - Backend never retrieves the results
   - Frontend never receives image URLs

**Technical Details:**
- Pipeline: `text_semantic_split` (Stage 2 semantic splitting)
- Configs Deactivated:
  - `splitandcombinelinear.json` (LERP interpolation)
  - `splitandcombinespherical.json` (SLERP interpolation)
- UI Component: `public/ai4artsed-frontend/src/components/Phase2VectorFusionInterface.vue`
- Output Chunk: `devserver/schemas/chunks/output_vector_fusion_clip_sd35.json` (not being used)

**Why Vector Fusion Requires English:**
- Vector Fusion works in CLIP embedding space
- CLIP embeddings are most precise with English text
- German prompts would produce incorrect semantic vectors
- Stage 1 translation is essential (skip_stage1: false is correct)

**Files Modified:**
- `text_semantic_split.json` - Changed pipeline_type to enable detection
- `Phase2VectorFusionInterface.vue` - Complete UI redesign
- `api.ts` - Fixed status vs success API type mismatch
- `Phase2CreativeFlowView.vue` - Added routing for Vector Fusion UI
- `PipelineExecutionView.vue` - Updated API response handling

**User Decision (Direct Quote):**
> "Wir machen das jetzt so: 1) vermekrke den Arbeitsstand auf einer Problemlist im ToDo. 2) Verschiebe diese workflows nach \"deactivated\". Ich habe keien Lust mit diesem relatriv unwichtigen Teil Stundenlang zu debuggen."

**Deactivated Files:**
- `devserver/schemas/configs/interception/deactivated/splitandcombinelinear.json`
- `devserver/schemas/configs/interception/deactivated/splitandcombinespherical.json`

**Next Steps (If Reactivated):**
1. Fix Stage 1→2 data flow to pass English translated text (not German original)
2. Complete migration from deprecated comfyui_workflow_generator to Output-Chunks system
3. Implement image retrieval polling in backend after ComfyUI workflow submission
4. Test full flow: Input → Translation → Split (EN) → Vector Fusion → Display images

**Estimated Fix Time:** 4-6 hours
**Complexity:** High (requires deep backend refactoring)

---

## ✅ COMPLETED FEATURES (Session 39)

### v2.0.0-alpha.1 Release - **COMPLETE**
**Status:** ✅ **SHIPPED** - First fully functional alpha release
**Session:** 39 (2025-11-09)
**Priority:** CRITICAL (milestone release)

**What Was Completed:**
1. **Critical Bugfix: media_type UnboundLocalError**
   - Extracted media_type determination before Stage 3-4 loop
   - Now stage4_only feature works correctly
   - File: `devserver/my_app/routes/schema_pipeline_routes.py` (lines 733-747)

2. **Git Merge & Release:**
   - Merged `feature/schema-architecture-v2` → `main` (113 commits)
   - Created annotated tag `v2.0.0-alpha.1`
   - Pushed to remote repository

3. **Visual Improvements:**
   - Added terminal run separator box (80-char with run metadata)

4. **Frontend Fixes:**
   - HomeView immediate redirect to /select (no old template flash)

5. **Development Workflow:**
   - Created `start_backend.sh` (backend only, port 17801)
   - Created `start_frontend.sh` (frontend only, port 5173)
   - Foreground execution with direct terminal output

**Stashed Features (WIP from Session 37):**
- SSE streaming (postponed per user request)
- Progressive image overlay UI
- Seed management UI (backend complete, frontend incomplete)

**Git Commits:**
- `fix: Extract media_type determination before Stage 3-4 loop`
- `feat: Add visual run separator box`
- `fix: HomeView immediate redirect to /select`
- `feat: Create separate start scripts`
- `docs: Archive Session 37 handover and obsolete docs`
- `Merge feature/schema-architecture-v2` (merge commit)
- `v2.0.0-alpha.1` (annotated tag)

**Testing:**
- ✅ Full 4-stage pipeline execution
- ✅ Image generation functional
- ✅ Files saved to disk
- ✅ Frontend displaying images
- ✅ Clean merge to main
- ✅ Tag pushed successfully

---

## ✅ COMPLETED FEATURES (Sessions 19-36)

### 1. Execution History Tracker - **COMPLETE**
**Status:** ✅ **COMPLETE** - Full Implementation & Testing Done
**Sessions:** 19-24 (2025-11-03)
**Priority:** HIGH (user needs this feature)

**What Was Implemented:**
- **Complete pedagogical journey tracking** from input to output across ALL 4 stages
- **Stage 1**: User input, translation, safety checks
- **Stage 2**: Interception iterations (including Stille Post recursive tracking)
- **Stage 3**: Pre-output safety checks (per output config)
- **Stage 4**: Media generation outputs (per output config)
- **Chronological order** with sequence numbers, timestamps, stage context
- **Two iteration types**: `stage_iteration` (Stage 2 recursive) and `loop_iteration` (Stage 3-4 multi-output)

**Implementation Phases Completed:**
- [x] **Phase 1** (Session 20): Core data structures (ExecutionItem, ExecutionRecord, enums)
- [x] **Phase 2** (Session 20): Tracker implementation & pipeline integration
- [x] **Phase 2.5** (Session 21): Metadata tracking expansion (model_used, backend_used, execution_time)
- [x] **Phase 3** (Session 22): Export API (REST endpoints /api/runs/*)
- [x] **Phase 3.5** (Session 22): Terminology fix (executions → pipeline_runs)
- [x] **Bug Fixes** (Session 23): Stille Post iteration tracking
- [x] **Minor Fixes** (Session 24): pipeline_complete loop_iteration, config_name in API

**Export Formats Implemented:**
- ✅ **JSON**: Fully implemented via /api/runs/{id}/export/json
- ⏸️ **XML/PDF/DOCX**: Placeholder (501 Not Implemented) - low priority

**Files Created:**
- `devserver/execution_history/__init__.py`
- `devserver/execution_history/models.py` (219 lines)
- `devserver/execution_history/tracker.py` (589 lines)
- `devserver/execution_history/storage.py` (240 lines)
- `devserver/my_app/routes/execution_routes.py` (334 lines)
- `docs/EXECUTION_TRACKER_ARCHITECTURE.md` (1200+ lines)
- `docs/ITEM_TYPE_TAXONOMY.md`
- `docs/TESTING_REPORT_SESSION_23.md`

**Documentation:**
- Session handover files: SESSION_19_HANDOVER.md through SESSION_22_HANDOVER.md
- Complete testing report: TESTING_REPORT_SESSION_23.md
- Development log: DEVELOPMENT_LOG.md (Sessions 19-24)

**API Endpoints Available:**
- GET /api/runs/list (with filtering and pagination)
- GET /api/runs/{execution_id}
- GET /api/runs/stats
- GET /api/runs/{execution_id}/export/{format}

**Testing Coverage:** ~70% complete
- ✅ Basic workflows (dada, bauhaus, etc.)
- ✅ Stille Post (8 recursive iterations)
- ✅ Loop iteration tracking (Stage 3-4 multi-output)
- ✅ Metadata tracking (model, backend, execution_time)
- ⏸️ Multi-output workflows (needs API clarification)
- ⏭️ Execution mode 'fast' (needs OpenRouter API key)

**Git Commits:**
- a7e5a3b - Phase 1: Core data structures
- 1907fb9 - Phase 2: Tracker integration
- c21bbd0, f5a94b5 - Phase 2.5: Metadata expansion
- 742f04a - Phase 3: Export API
- e3fa9f8 - Terminology fix
- 131427a, 54e8bb5, af22308 - Bug #1 fix & testing
- cbf622f - Minor fixes (OBSERVATION #1 & #2)

**Next Steps (Optional Enhancements):**
- Implement XML/PDF export (currently 501)
- Complete multi-output testing
- Test execution mode 'fast'
- Add frontend UI for browsing execution history

---

### 2. Unified Media Storage - **COMPLETE**
**Status:** ✅ **COMPLETE** - Full Implementation Done
**Session:** 27 (2025-11-04)
**Priority:** HIGH (fixes broken export functionality)

**What Was Implemented:**
- **Backend-agnostic media storage** for ComfyUI, OpenRouter, Replicate, etc.
- **Flat run-based structure**: `exports/json/{run_id}/`
- **Atomic research units**: All files per run in one folder
- **"Run" terminology** (not "execution" due to German connotations)
- **Automatic media download** during pipeline execution
- **UUID-based** for concurrent-safety (workshop scenario)

**Key Features:**
- Auto-detects URL vs prompt_id for media downloads
- Stores metadata.json with complete run information
- Serves media via `/api/media/*` endpoints
- Works with ANY backend (ComfyUI, OpenRouter, future backends)

**Files Created:**
- `devserver/my_app/services/media_storage.py` (414 lines)
- `docs/UNIFIED_MEDIA_STORAGE.md` (documentation)

**Files Modified:**
- `devserver/my_app/routes/schema_pipeline_routes.py` (integration)
- `devserver/my_app/routes/media_routes.py` (rewritten for local storage)

**Storage Structure:**
```
exports/json/{run_uuid}/
├── metadata.json           # Complete run metadata
├── input_text.txt         # Original user input
├── transformed_text.txt   # After interception
└── output_<type>.<format> # Generated media
```

**API Endpoints:**
- GET /api/media/image/<run_id>
- GET /api/media/audio/<run_id>
- GET /api/media/video/<run_id>
- GET /api/media/info/<run_id>
- GET /api/media/run/<run_id>

**Problems Fixed:**
- ✅ ComfyUI images now persisted locally
- ✅ OpenRouter images stored as actual files
- ✅ Export function works with persisted media
- ✅ Research data properly stored

**Documentation:**
- Session summary: SESSION_27_SUMMARY.md (archived)
- Technical docs: docs/UNIFIED_MEDIA_STORAGE.md

**Next Steps (Optional):**
- Update export_manager.py to use run_id
- Frontend verification of new storage structure

---

### 3. LivePipelineRecorder Migration - **FULLY COMPLETE**
**Status:** ✅ **MIGRATION COMPLETE** - MediaStorage Removed, Single System
**Sessions:** 29 (Initial), 37 (Migration Complete) - (2025-11-04, 2025-11-08)
**Priority:** CRITICAL (fixes complete desynchronization)

**Session 37 Final Migration (2025-11-08):**
- ✅ **MediaStorage completely removed** from pipeline execution
- ✅ **LivePipelineRecorder is now single source of truth**
- ✅ **Frontend displaying images correctly** after bug fix
- ✅ **No more dual-system complexity or duplicate files**

**What Was Completed:**
1. **Session 29:** Created LivePipelineRecorder with unified run_id
   - Sequential entity tracking (01_input.txt → 07_output_image.png)
   - Real-time API endpoints for frontend polling
   - Fixed media polling bug with `wait_for_completion()`

2. **Session 37:** Complete MediaStorage removal
   - Migrated download capabilities to LivePipelineRecorder
   - Added `download_and_save_from_comfyui()` method
   - Added `download_and_save_from_url()` method
   - Updated media_routes.py to use Recorder metadata format
   - Fixed frontend bug accessing media type from API response
   - Resolved Python bytecode caching issues

**Architecture Evolution:**
```
BEFORE (Session 29):
├─ ExecutionTracker (exec_*.json)      # OLD system
├─ MediaStorage (exports/json/)        # Creates runs, downloads media
└─ LivePipelineRecorder (exports/json/) # Copies from MediaStorage

AFTER (Session 37):
├─ ExecutionTracker (exec_*.json)      # Still exists for compatibility
└─ LivePipelineRecorder (exports/json/) # SINGLE SOURCE OF TRUTH
```

**Files Modified (Session 37):**
- `devserver/my_app/services/pipeline_recorder.py` - Added download methods (~200 lines)
- `devserver/my_app/routes/media_routes.py` - Complete rewrite for Recorder format
- `devserver/my_app/routes/schema_pipeline_routes.py` - Removed MediaStorage usage
- `public_dev/js/execution-handler.js` - Fixed media type access bug (line 448)

**API Endpoints:**
- GET /api/pipeline/{run_id}/status - Real-time execution state
- GET /api/pipeline/{run_id}/entity/{type} - Fetch specific entity
- GET /api/pipeline/{run_id}/entities - List all entities
- GET /api/media/image/{run_id} - Serve images from Recorder
- GET /api/media/info/{run_id} - Get media metadata from Recorder

**Test Results (Session 37):**
- Run ID: `1c173019-9437-43fe-bd57-e2612739a8c5`
- All 7 entities created with correct naming (01-07)
- Backend API working (HTTP 200 for all endpoints)
- Frontend displaying images correctly
- No duplicate files, single source of truth validated

**Documentation:**
- Technical docs: `docs/LIVE_PIPELINE_RECORDER.md`
- Session 29: `docs/archive/SESSION_29_*` (archived session docs)
- Session 37: `docs/DEVELOPMENT_LOG.md` (migration completion entry)

**Next Steps (Optional):**
- Consider deprecating old ExecutionTracker after more validation
- Add WebSocket support for real-time frontend updates
- Extend recorder format for video/audio outputs

---

### 4. Phase 2: Multilingual Context Editing - **BACKEND COMPLETE, FRONTEND BROKEN**
**Status:** 🚨 **BACKEND COMPLETE** - Frontend Implementation Wrong, Needs Redesign
**Session:** 36 (2025-11-08)
**Priority:** HIGH (core pedagogical feature) - **BLOCKED by frontend issues**

**What Was Implemented:**
- **Backend API endpoints** for multilingual meta-prompt access
- **Pipeline structure metadata** for dynamic UI rendering
- **Property system fixes** (calm→chill, i18n translation)
- **Frontend components** created (not yet tested end-to-end)

**Backend Endpoints Working:**
- `GET /api/config/<id>/context` - Returns multilingual meta-prompt {en, de}
- `GET /api/config/<id>/pipeline` - Returns pipeline structure metadata
  - `requires_interception_prompt` - Show/hide context editing bubble
  - `input_requirements` - How many text/image inputs needed

**Frontend Components Created (Previous Sessions):**
- `PipelineExecutionView.vue` - Main Phase 2 view with 3 bubbles
- `EditableBubble.vue` - Inline editing component
- `pipelineExecution` store - Phase 2 state management
- `userPreferences` store - Global language management

**🚨 CRITICAL: Phase 2 Frontend Implementation WRONG**
- **User Feedback:** "you did the stage2 design COMPLETELY wrong, ignoring the organic flow mockup, adding unwanted buttons, also instead of context-prompt there appears: 'Could you please provide the English text you'd like translated into German?'. I will not leave this to you, but add to bugs to be fixed."
- **Problems Identified:**
  - ❌ Ignored organic flow mockup specifications
  - ❌ Added unwanted buttons to the UI
  - ❌ Wrong placeholder text (translation prompt instead of context-prompt)
  - ❌ Fundamental UX design mismatch
- **File Affected:** `public/ai4artsed-frontend/src/views/PipelineExecutionView.vue`
- **Status:** User will handle the frontend implementation redesign
- **Impact:** Phase 2 end-to-end testing **BLOCKED** until frontend is fixed
- **Action Required:** Complete redesign following organic flow mockup
- **⚠️ DO NOT attempt to fix without explicit user instruction**

**Property System Fixes:**
- ✅ Removed all "calm" property IDs → "chill"
- ✅ Frontend now uses i18n translation (PropertyBubble, NoMatchState)
- ✅ Fixed PigLatin property pair violation (removed "chill", kept "chaotic")
- ✅ Properties display as translated labels ("wild" not "chaotic")

**Files Modified:**
- Backend: `schema_pipeline_routes.py`, `config_loader.py`, 1 + 10 configs
- Frontend: `PropertyBubble.vue`, `NoMatchState.vue`, 2 stores

**Testing Status:**
- ✅ Backend endpoints tested with curl
- ✅ Property translations fixed in code
- 🚨 Phase 2 end-to-end flow **BLOCKED** (frontend implementation wrong)
- ⚠️ Phase 1 → Phase 2 navigation NOT YET TESTED
- ⚠️ Language switching NOT YET TESTED
- ⚠️ Pipeline execution with edited context NOT YET TESTED

**Next Steps:**
1. **CRITICAL BLOCKER:** Fix Phase 2 Frontend Implementation
   - User will handle the PipelineExecutionView.vue redesign
   - Must follow organic flow mockup specifications
   - Remove unwanted buttons
   - Fix context-prompt placeholder text
   - **⚠️ DO NOT attempt without explicit user instruction**

2. **AFTER FRONTEND FIX:** Test Phase 2 end-to-end flow
   - Open `http://localhost:5173/select`
   - Select properties, click config tile
   - Should navigate to `/execute/<configId>`
   - Test language toggle, editing, execution

3. **IF WORKING:** Mark Phase 2 as complete, start Phase 3
4. **IF BROKEN:** Debug before proceeding (Phase 2 foundational for Phase 3)

**Git Commits (Session 36):**
- fix(backend): Add missing config_loader import and fix pipeline_name
- fix: Remove all 'calm' property IDs, replace with 'chill'
- fix(frontend): Use i18n translations for property display
- fix(piglatin): Remove 'chill' property - cannot coexist with 'chaotic'

**Documentation:**
- `docs/HANDOVER.md` - Complete session handover
- `docs/DEVELOPMENT_LOG.md` - Session 36 entry added
- `docs/ARCHITECTURE PART 04 - Pipeline-Types.md` - Pipeline metadata system

---

## 🔥 IMMEDIATE PRIORITIES (Session 36+)

### 2. Interface Design

**Goal 2: Design Educational Interface**

**Context from User:**
> "Now that the dev system works basically, our priority should be to develop the interface/frontend according to educational purposes. The schema-pipeline-system has been inspired by the idea that ENDUSER may edit or create new configs."

**Key Principles for UI Design:**

1. **Use Stage 2 pipelines as visual guides**
   - `text_transformation.json` shows the flow: input → manipulate → output
   - Pipeline metadata documents what happens at each step

2. **Make the 3-part structure visible and editable**
   - Show TASK_INSTRUCTION (from instruction_type)
   - Show CONTEXT (from config.context)
   - Show PROMPT (user input)
   - Allow editing of configs

3. **Educational transparency**
   - Students should see HOW their prompt is transformed
   - Students should be able to edit configs to create new styles
   - Students should understand the prompt interception concept

4. **Reference files for UI design**
   - `devserver/schemas/pipelines/*.json` - Flow structure
   - `devserver/schemas/configs/interception/*.json` - Config examples
   - `docs/ARCHITECTURE.md` Section 6 - instruction_selector.py docs

### 3. GPT-OSS Stage 3 Implementation (Deferred)

**From Session 14:** Replace llama-guard3 with GPT-OSS in Stage 3
**Status:** Deferred - Focus shifted to interface design
**See:** `docs/devserver_todos.md` for details

---

## 📝 Session 16 Completion Notes

**What Was Fixed:**
- ✅ Restored `single_text_media_generation.json` pipeline (accidentally deprecated in Session 15)
- ✅ Fixed Stage 4 error: "Config 'sd35_large' not found"
- ✅ Tested full 4-stage pipeline: Working correctly
- ✅ Committed fix: commit `6f7d30b`
- ✅ Updated SESSION_HANDOVER.md with Session 16→17 context
- ✅ Created PIPELINE_RENAME_PLAN.md (completed in Session 17)

**Key Learnings:**
- Pipeline naming is confusing: "single_prompt_generation" sounds like "generate a prompt" not "generate media FROM a prompt"
- The pipeline was critical for Stage 4 because it provides DIRECT media generation (no text transformation step)
- Output configs (sd35_large, gpt5_image) need this pipeline
- Never deprecate pipeline files without checking all config references first!

**Git Status:**
- Branch: `feature/schema-architecture-v2`
- Commit: `6f7d30b` - "fix: Restore single_prompt_generation pipeline"
- Pushed to remote: ✅

---

## 📁 Archived TODOs

**Archive Policy:** Completed tasks from old sessions archived for reference

**Archives:**
- **Sessions 1-14 (Full History):** `docs/archive/devserver_todos_sessions_1-14.md` (1406 lines)
  - Sessions 1-8: Various architecture work
  - Session 9: 4-Stage Architecture Refactoring
  - Session 10: Config Folder Restructuring
  - Session 11: Recursive Pipeline + Multi-Output Support
  - Session 12: Project Structure Cleanup + Export Sync
  - Session 13: GPT-OSS Model Research
  - Session 14: GPT-OSS Unified Stage 1 Activation

**See also:** `docs/DEVELOPMENT_LOG.md` for chronological session tracking with costs

---

## 🎯 PRIORITY 1 (Future): Internationalization - Primary Language Selector

**Status:** NEW TODO (from Session 14)
**Context:** German language is currently hardcoded in educational error messages
**Priority:** MEDIUM (works for German deployment, blocks international use)

**Current Issue:**
- Educational blocking messages hardcoded in German (stage_orchestrator.py:330-336)
- §86a StGB error template only in German
- System assumes German as primary language

**Proposed Solution:**

### 1. Add to `config.py`
```python
# Primary language for educational content and error messages
PRIMARY_LANGUAGE = "de"  # ISO 639-1 code: de, en, fr, es, etc.

# Supported languages for UI and error messages
SUPPORTED_LANGUAGES = ["de", "en"]
```

### 2. Create Language Templates Directory
```
devserver/schemas/language_templates/
├── de.json  # German templates (default)
├── en.json  # English templates
└── ...
```

### 3. Template Structure
```json
{
  "safety_blocked": {
    "heading": "Dein Prompt wurde blockiert",
    "why_rule": "WARUM DIESE REGEL?",
    "protection": "Wir schützen dich und andere vor gefährlichen Inhalten."
  }
}
```

### 4. Update Error Messages
- `stage_orchestrator.py`: Replace hardcoded strings with template system
- Load templates based on PRIMARY_LANGUAGE setting
- Fall back to English if language not supported

**Benefits:**
- Enables international deployment (UK, US, France, etc.)
- Maintains German compliance for German deployments
- Single config variable controls all language settings
- Easy to add new languages

**Timeline:** Future enhancement (not blocking current deployment)
**Estimated Time:** 3-4 hours

---

## ✅ RECENTLY COMPLETED

### Session 38 (2025-11-08): GPT-OSS Stage 3 + keep_alive Memory Management
**Status:** ✅ COMPLETE
**Priority:** HIGH (performance optimization + consistency)

**What Was Discovered:**
- GPT-OSS was already configured for Stage 3 via `model_override` in configs
- Only missing piece was `keep_alive` parameter for memory management

**What Was Implemented:**
- Added `keep_alive: "10m"` to all GPT-OSS configs and chunks:
  - `schemas/chunks/manipulate.json` (Stage 2 interception)
  - `schemas/configs/pre_interception/gpt_oss_unified.json` (Stage 1)
  - `schemas/configs/pre_output/text_safety_check_kids.json` (Stage 3)
  - `schemas/configs/pre_output/text_safety_check_youth.json` (Stage 3)

**Current State (All Stages Using GPT-OSS):**
- ✅ Stage 1: GPT-OSS:20b (Translation + §86a Safety) with keep_alive
- ✅ Stage 2: GPT-OSS:20b (Prompt Interception) with keep_alive
- ✅ Stage 3: GPT-OSS:20b (Pre-Output Safety kids/youth) with keep_alive
- ✅ Stage 4: Output generation (ComfyUI/API)

**Benefits Achieved:**
- ✅ Single model (GPT-OSS:20b) for all text processing and safety checks
- ✅ Model stays in VRAM for 10 minutes between calls (no loading overhead)
- ✅ Estimated ~2-3s performance improvement per request
- ✅ Unified safety approach across all stages
- ✅ Reduced VRAM thrashing (no model switching)

**Files Modified:**
- `devserver/schemas/chunks/manipulate.json`
- `devserver/schemas/configs/pre_interception/gpt_oss_unified.json`
- `devserver/schemas/configs/pre_output/text_safety_check_kids.json`
- `devserver/schemas/configs/pre_output/text_safety_check_youth.json`

**Testing:**
- ✅ All JSON configs validated
- ✅ Server restarted with new configs
- ✅ Pipeline execution tested successfully

**Git Commit:** [Pending]

---

### Session 17 (2025-11-03): Pipeline Rename to Input-Type Convention
**Status:** ✅ COMPLETE
**Commit:** `bff5da2` - "refactor: Rename pipelines to input-type naming convention"

**What Was Done:**
- Renamed `single_prompt_generation` → `single_text_media_generation`
- Updated 2 output configs: `sd35_large.json`, `gpt5_image.json`
- Updated 7 documentation files
- Deleted deprecated file: `single_prompt_generation.json.deprecated`
- Split ARCHITECTURE.md → ARCHITECTURE PART I.md + PART II.md

**New Pattern:** `[INPUT_TYPE(S)]_media_generation`
- Clear separation: "text" = input type, "media" = output type
- Scalable: Easy to add `image_text_media_generation`, `video_text_media_generation`, etc.
- Self-documenting: Name explicitly describes data flow

**Testing:**
- ✅ Config loader finds pipeline: single_text_media_generation
- ✅ sd35_large config references correct pipeline
- ✅ gpt5_image config references correct pipeline
- ✅ 7 pipelines loaded, 45 configs loaded

**Files Changed:** 13 files (+429 -90 lines)

### Session 14 (2025-11-02): GPT-OSS Unified Stage 1 Activation
**Status:** ✅ COMPLETE & TESTED
**Commit:** `839dc73`

**What Was Done:**
- Created unified GPT-OSS config with full §86a StGB legal text
- Added `execute_stage1_gpt_oss_unified()` in stage_orchestrator.py
- Updated schema_pipeline_routes.py to use unified function
- Tested successfully: ISIS blocking, Nazi code 88, legitimate prompts

**ISIS Failure Case from Session 13:** ✅ FIXED

**See:** `docs/DEVELOPMENT_LOG.md` Session 14 for full details

### Session 12 (2025-11-02): Project Structure Cleanup
**Status:** ✅ COMPLETE
**Commit:** `fe3b3c4`

**What Was Done:**
- Archived LoRA experiment + legacy docs
- Moved docs/ and public_dev/ to project root
- Robust start_devserver.sh
- Synced 109 export files from legacy

### Sessions 9-11 (2025-11-01): 4-Stage Architecture
**Status:** ✅ COMPLETE & TESTED

**What Was Implemented:**
- 4-Stage Architecture Refactoring (Stage 1-3 orchestration)
- Recursive Pipeline System ("Stille Post")
- Multi-Output Support (model comparison)

**See:** `docs/archive/devserver_todos_sessions_1-14.md` for full details

---

## 🎮 MINIGAMES / WAITING ANIMATIONS

### 🎯 Design Principles (Übergreifend)

**Kern-Prinzip: "Sisyphus der Systeme"**

Alle Minigames folgen einem gemeinsamen pädagogischen Ansatz:
- **Abwärtsdynamik:** Keine vollständige Heilung möglich
- **User kann handeln:** Aber systemische Zerstörung läuft schneller als individuelle Aktion
- **Realistische Darstellung:** Zeigt die echte Asymmetrie des Problems
- **Sisyphus-Metapher:** Kämpfen gegen eine Übermacht (wie in "Papers, Please", "This War of Mine")

**Beispiele:**
- **Trees:** 1 Sekunde pro Baumpflanzung, ABER Fabriken wachsen schneller
- **Seltene Erden:** Giftschlamm entfernen, ABER Abbau läuft weiter
- **Fair Culture:** (Noch zu definieren - ähnliches Prinzip)

---

**⚠️ KRITISCHE SELBSTREFLEXION - Zu klären:**

**Risiko 1: Resignation statt Handlung**
- Führt die Hoffnungslosigkeit zu Lähmung?
- Lernen Schüler "Es ist aussichtslos, also warum versuchen?"
- Ist das pädagogisch kontraproduktiv?

**Risiko 2: Fehlende Handlungsoptionen**
- Minigame zeigt Problem, aber keine Lösung
- Sollte es konkrete Exit-Strategien geben?
- Links zu realen Organisationen/Initiativen nach dem Spiel?

**Risiko 3: Zu deprimierend für Zielgruppe**
- Ist das für Kids (8-12) / Youth (13-17) angemessen?
- Balance zwischen Realismus und psychischer Belastung?
- Braucht es Hoffnungsmomente?

**Mögliche Lösungsansätze:**
- [ ] **"Was kann ich wirklich tun?"** - Sektion nach jedem Spiel
  - Recycling-Initiativen (z.B. Fairphone, refurbished Hardware)
  - Fair-Culture-Bewegungen (Künstler-Kollektive)
  - Politische Handlungsoptionen (Petitionen, Awareness-Kampagnen)
- [ ] **Kleine Siege zeigen:** User kann temporäre Verbesserungen erreichen
- [ ] **Kollektive Aktion:** "Du allein kannst es nicht schaffen, aber gemeinsam..."
- [ ] **Systemkritik statt Verzweiflung:** Fokus auf strukturelle Probleme, nicht individuelle Schuld

**Designfrage:** Wie balancieren wir **ehrlichen Realismus** mit **pädagogischer Ermächtigung**?

---

### Exploitation 1: Seltene Erden (Rare Earths)

**Status:** 🔧 **IN PROGRESS** - Implementation started (Session 156)
**Datum:** 2026-02-03
**Priority:** MEDIUM (pedagogical feature, not blocking core functionality)

**Konzept:** Pädagogisches Minigame über Seltene-Erden-Abbau und Umweltzerstörung

**Game Mechanic (v3 - FINAL DESIGN):**
- **Prinzip:** Umwelt vs. genAI - "Tauziehen"-Mechanik mit **Abwärtsdynamik**
- **Förderband (Conveyor Belt):** Abbau geschieht AUTOMATISCH
  - 3 farbige Kristalle (Nd, Dy, Tb) werden gefördert
  - Geschwindigkeit = f(GPU Temperatur) - je heißer, desto schneller
  - Füllt GPU-Chip kontinuierlich
- **Giftschlamm:** Fließt vom Förderband in den See
  - See färbt sich: blau → braun/grün
  - Himmel verdunkelt sich mit Verschmutzung
- **User-Tool:** Minecraft-like **Schaufel** (erscheint beim Click)
  - Click auf See → entfernt 10% Schlamm
  - 1 Sekunde Cooldown
  - Temporäre Heilung von 1-2 kranken Pflanzen (re-degradieren nach 5s)
- **Container:** Sammelt Schlamm
  - Bei 100% voll → **LKW fährt weg** (Animation)
  - Container leert sich, Loop geht weiter
- **Game Over:** Nur bei Inaktivität (30s kein Click)
  - NICHT bei Ökosystem-Kollaps (See kann 100% sein, Game läuft weiter)

**Visuelle Elemente:**
- ⛰️ Berg (links)
- 🏭 Förderband (oben rechts) mit 3 Kristallen
- 🌊 See (Mitte-rechts) - dynamische Farbe basierend auf Verschmutzung
- 🥄 Schaufel (erscheint beim Click)
- 🖥️ GPU-Chip (unten rechts) - zeigt 3 Edelsteine (Nd, Dy, Tb)
- 🌳🌿 Umwelt (Bäume/Büsche: grün/gesund → braun/krank → tot)
- 📦 Container (unten links)
- 🚚 LKW-Animation (fährt Container weg bei 100%)
- ☁️ Himmel (verdunkelt sich mit Verschmutzung)
- 📝 Info-Banner (wie bei anderen Games)

**Pädagogischer Kern:**
- Zeigt die **systemische Hoffnungslosigkeit** des Problems
- Abbau läuft schneller als Aufräumen (Sisyphus-Metapher)
- Verdeutlicht: AI-Nutzung → GPU-Nachfrage → Seltene-Erden-Abbau → Umweltzerstörung
- User kann handeln, aber nicht gewinnen (realistisch)

**Integration:** Als 4. Option in `RandomEdutainmentAnimation.vue` (neben pixel, iceberg, forest)

**Implementation Tasks:**
- [x] Design finalisiert (User-Feedback eingearbeitet)
- [x] Plan erstellt (`/home/joerissen/.claude/plans/atomic-beaming-seal.md`)
- [x] **Phase 1:** i18n keys hinzufügen (DE/EN) - Commit `69e25ed`
- [x] **Phase 2:** `RareEarthMiniGame.vue` erstellen (569 Zeilen) - Commit `b85552c`
  - [x] Component scaffold (props, composable, refs)
  - [x] Visual elements (sky, mountain, conveyor, lake, GPU, container, vegetation)
  - [x] Game loop (degradation, mining, sludge influx)
  - [x] Click handler (shovel animation, sludge removal)
  - [x] Truck animation (container full)
  - [x] Stats bar + UI (instructions, info banner, game over)
- [x] **Phase 3:** `RandomEdutainmentAnimation.vue` updaten - Commit `b85552c`
- [ ] **Phase 4:** Testing (balance, inactivity timeout, mobile responsive)
  - [ ] Manual testing im Frontend
  - [ ] Balance prüfen (Degradation vs. Cleanup rates)
  - [ ] Inactivity timeout (30s) verifizieren
  - [ ] Truck animation testen
  - [ ] Mobile Responsiveness prüfen
  - [ ] Vue type-check ausführen

---

### Exploitation 2: Fair Culture (Web Scraping Ethics)

**Status:** 📋 **PLANNED**
**Datum:** 2026-02-03
**Priority:** MEDIUM (pedagogical feature, not blocking core functionality)

**Konzept:** Pädagogischer Content über Web-Scraping für generative AI, mit Spiel-Mechanik zur Künstler-Kompensation

**Details:**
- **Type:** Waiting animation / minigame
- **Educational Goal:** Aufklärung über AI-Training-Data-Ethik und Künstler-Kompensation
- **Thema:** Wie AI-Modelle mit geklauten/gescrapten Kunstwerken trainiert werden
- **Game Mechanic:** Spieler könnten virtuelle Künstler "kompensieren" (Noch zu designen)
- **Integration Point:** Während AI-Model-Loading oder Generierungsprozessen
- **Pädagogischer Wert:** Kritisches Bewusstsein für Copyright und faire Entlohnung im AI-Zeitalter

**Mögliche Mechaniken:**
- Künstler-Profile mit echten Hintergründen (anonymisiert)
- "Kompensations-Punkte" sammeln während Wartezeit
- Visualisierung: Wie viele Kunstwerke für Training verwendet wurden
- Link zu Fair-Culture-Initiativen und Künstler-Kollektiven

**Nächste Schritte:**
- [ ] Recherche: Fair-Culture-Bewegung, Künstler-Initiativen
- [ ] Game Mechanic Design (Kompensations-System)
- [ ] Content: Künstler-Geschichten und Fakten über AI-Training
- [ ] Frontend: Integration in Waiting-Overlay
- [ ] Backend: Optional - Tracking welche Models genutzt werden

---

## 📝 Quick Reference

---

## 🟡 REFACTORING: "optimization" → "adaptation"

**Status:** 📋 **TODO** - Terminology cleanup
**Datum:** 2026-01-29 (Session 145)
**Priority:** LOW (Code-Hygiene)

### Problem

Backend verwendet "optimization" für Prompt-Adaption an Medienmodelle:
- `optimize_clip_prompt.json` → SD3.5
- `optimize_t5_prompt.json` → Flux
- Weitere für Video/Audio

Frontend/Dokumentation verwendet jetzt korrekt "Adaption/Adaptation".

### Aufgabe

Alle Backend-Referenzen von "optimization/optimize" zu "adaptation/adapt" umbenennen:
- Chunk-Dateien: `optimize_*.json` → `adapt_*.json`
- Code-Referenzen in Python
- Config-Keys

### Betroffene Dateien
```
devserver/schemas/chunks/optimize_clip_prompt*.json
devserver/schemas/chunks/optimize_t5_prompt*.json
devserver/my_app/routes/*.py (Referenzen)
devserver/my_app/engine/*.py (Referenzen)
```

---

**Current Architecture Status:**
- ✅ 4-Stage Pipeline System (Stages 1-4)
- ✅ Config-based system (Chunks → Pipelines → Configs)
- ✅ Backend abstraction (Ollama, ComfyUI, OpenRouter)
- ✅ GPT-OSS Stage 1 (Translation + §86a Safety)
- ✅ Stage 3 Hybrid Safety (fast string-match + LLM context)
- ✅ Multi-output support
- ✅ Recursive pipelines

**Next Up:**
1. Replace llama-guard3 with GPT-OSS in Stage 3
2. Implement keep_alive memory management
3. Add language template system

**Documentation:**
- Architecture: `docs/ARCHITECTURE.md`
- Development Log: `docs/DEVELOPMENT_LOG.md` (Sessions 12-14)
- Development Decisions: `docs/DEVELOPMENT_DECISIONS.md`
- Safety Architecture: `docs/safety-architecture-matters.md`

**Archived:**
- Old TODOs: `docs/archive/devserver_todos_sessions_1-14.md`
- Old Dev Log: `docs/archive/DEVELOPMENT_LOG_Sessions_1-11.md`

---

## 📋 TODO: Latent Lab UX Improvements

**Status:** 📋 **TODO**
**Datum:** 2026-02-21
**Priority:** MEDIUM (UX polish, not blocking functionality)

### 1. ✅ Seed defaults: set to -1 (random) where hardcoded to 42
Fixed in crossmodal_lab.vue (synth, MMAudio, guidance). Other labs already used -1.

### 2. ✅ Sticky sub-tabs: crossmodal lab sub-tab should persist
Solved by reorganization (`63dcc50`): `image_lab.vue` and `crossmodal_lab.vue` both use `localStorage` for tab persistence.

### 3. ✅ Streamline "Erweiterte Einstellungen" collapse state across labs
Done: `useDetailsState` composable syncs `<details>` open/closed state with localStorage. Applied to all 6 lab views (explanation + advanced settings + crossmodal dim-explorer/MIDI). Keys: `ll_{lab}_{section}` pattern.

### 4. Parameter hints for all latent lab elements
~~Every slider, parameter, and input in the latent lab should have a tooltip or expandable explanation. Currently inconsistent — some have `slider-hint`, some don't. Audit and fill gaps.~~

**4a. ✅ Parameter hints (slider-hint / control-hint) on every input**
Done in `22bf825`: ~30 new i18n hint keys (6 languages), hint spans added to all 6 Vue files (attention_cartography, concept_algebra, feature_probing, denoising_archaeology, crossmodal_lab, latent_text_lab). Shared hints for negative/steps/CFG/seed reused across image-generation labs.

**4b. ✅ Add `explanationToggle` section to crossmodal_lab.vue**
Done: Added collapsible explanation block with 4 Q&A sections (overview + one per sub-tab: Synth, MMAudio, ImageBind Guidance). i18n keys in all 6 languages, CSS matches green accent of crossmodal lab. Type check passes.

### 5. ✅ Scientific references with DOI in all labs
Done: DOI references added inside explanation blocks for all labs. Citations per lab: Attention (Hertz 2022, Tang 2022), Probing (Belinkov 2022, Zou 2023, Bau 2020), Algebra (Mikolov 2013, Liu 2022), Archaeology (Kwon 2023, Ho 2020), Text Lab RepEng (Zou 2023, Li 2023), Compare (Kornblith 2019, Olsson 2022), Bias (Bricken 2023, Zou 2023), ImageBind (Girdhar 2023). MMAudio already had inline citation. i18n: `referencesTitle` key in all 6 languages.

---

## 📋 TODO: Video Generation via Wan 2.1 Diffusers (VRAM-Tiered)

**Status:** 📋 **IMPLEMENTIERT, PoC PENDING** — Code fertig, Modell-Download läuft
**Datum:** 2026-02-15
**Priority:** HIGH (neue Medienfähigkeit)
**Plan:** `docs/plans/video_generation_wan21_diffusers.md`

### Was ist fertig

Code komplett implementiert in beiden Schichten (GPU Service + DevServer):
- GPU Service: `generate_video()` + `/api/diffusers/generate/video` Route
- DevServer: `DiffusersClient.generate_video()` + Python Output Chunk + 2 Output Configs
- VRAM-Tier-Routing: `output_config_defaults.json` mit Dict-Support (96GB→14B, 32GB→1.3B, 8GB→null)
- Backend Router: Video-Daten-Handling in `_execute_python_chunk()`

### Was noch fehlt

1. **PoC-Test**: `venv/bin/python test_wan21_video.py` — wartet auf Modell-Download
2. **Modell-Download**: Wan 2.1 T2V-1.3B läuft (T5-XXL Text Encoder ist ~25GB, teilweise aus 14B-Cache kopiert)
3. **14B-Modell**: Download unterbrochen (30GB Disk zu wenig für 85GB Modell), später auf anderem System fortsetzen
4. **Integration Test**: GPU Service → DevServer → Video Pipeline end-to-end
5. **Frontend**: Vue-Komponente für Video-Anzeige (noch nicht implementiert)

### Nächste Schritte nach Reboot

1. Prüfen ob 1.3B-Download fertig: `ls ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/`
2. PoC starten: `venv/bin/python test_wan21_video.py`
3. Bei Erfolg: GPU Service + DevServer starten, Integration testen

---

## 📋 TODO: LoRA Support for Diffusers GPU Service

**Status:** 📋 **TODO** — Plan approved, ready for implementation
**Datum:** 2026-02-17
**Priority:** HIGH (Blocks usage of user-trained LoRAs with Diffusers backend)
**Plan:** `docs/plans/lora-diffusers-support.md`

### Problem
Diffusers GPU service (port 17803) is now the primary backend for SD3.5 Large, but it has NO LoRA support. The entire LoRA data flow exists (interception configs → orchestrator → backend_router), but `_process_diffusers_chunk` silently ignores `parameters['loras']`. LoRA requests get forced to ComfyUI workflow path.

### Scope (5 files)
1. `gpu_service/config.py` — Add `LORA_DIR`
2. `gpu_service/services/diffusers_backend.py` — Add `_apply_loras()` / `_remove_loras()` + `loras` param to all generation methods
3. `gpu_service/routes/diffusers_routes.py` — Pass `loras` from request JSON
4. `devserver/my_app/services/diffusers_client.py` — Pass `loras` in HTTP payloads
5. `devserver/schemas/engine/backend_router.py` — Extract `parameters['loras']` in `_process_diffusers_chunk` + fix auto-detection routing

---

## 📋 TODO: PyTorch Stable Migration — Nightly → Stable Release

**Status:** 📋 **TODO** — Ready when convenient
**Datum:** 2026-02-22
**Priority:** LOW (Cosmetic — nightly works fine, but stable is cleaner)

### Problem
System runs on PyTorch nightly `2.11.0.dev20260203+cu130` (pinned to exact day — even 1 day difference can cause CUDA crashes on Blackwell). Originally necessary because stable PyTorch didn't support Blackwell (sm_120). Since PyTorch 2.7 (April 2025), Blackwell is officially supported in stable releases.

### Current State
- **Nightly**: `torch 2.11.0.dev20260203+cu130` (CUDA 13.0)
- **Latest Stable**: `torch 2.10.0` (Jan 2026), `2.11.0` geplant für Feb 16, 2026
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (sm_120, 96GB)
- **torchao Warning**: `Skipping import of cpp extensions` wegen Nightly-Inkompatibilität (harmlos)

### Migration Steps
1. Prüfen ob `torch 2.11.0` stable released ist (Feb 16 geplant)
2. Stable mit `cu128` oder `cu129` in separatem venv testen
3. SD3.5, Wan 2.1, HeartMuLa, Stable Audio, Cross-Aesthetic jeweils testen
4. Wenn alles funktioniert: venv migrieren, torchao upgraden
5. Nightly-Pinning-Warnung aus MEMORY.md entfernen

### Risiko
- CUDA 13.0 → 12.8/12.9 Downgrade: theoretisch kein Problem, aber Blackwell-spezifische Kernels könnten sich unterscheiden
- **Eigene Test-Session dafür planen**, nicht nebenbei machen

---

**Created:** 2025-10-26
**Last Major Cleanup:** 2025-11-02 Session 14
**Status:** Clean and concise (down from 1406 lines to ~230 lines)
