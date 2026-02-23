# Development Decisions - Current & Active
**AI4ArtsEd DevServer - Active Architectural Decisions**

> **IMPORTANT FOR ALL TASKS:**
> Every significant development decision MUST be documented here.
> Format: Date, Decision, Reasoning, Affected Files

**Full History:** See `docs/archive/DEVELOPMENT_DECISIONS_FULL.md` (2435 lines, Sessions 1-17)

---

## 📋 Quick Reference - Current Architecture

**Current System Status (as of 2025-12-02):**
- ✅ 4-Stage Pipeline Architecture (DevServer orchestrates Stages 1-4)
- ✅ GPT-OSS:20b for Stage 1 (Translation + §86a Safety unified)
- ✅ Config-based system (Interception configs, Output configs, Pre-output safety)
- ✅ Backend abstraction (Ollama, ComfyUI, SwarmUI APIs)
- ✅ Multi-output support (model comparison, batch generation)
- ✅ Recursive pipelines ("Stille Post" iterative transformation)
- ✅ Unified storage (symlink: prod → dev for shared research data)

**Deployment (Research Phase - 2025-11-16):**
- 🌐 Internet-facing via Cloudflare tunnel (multiple courses)
- 📱 Primary device: iPad Pro 10"
- 🔄 Legacy backend (port TBD) - Active for students
- 🔧 Dev backend (port 17801) - Development only
- 📊 Shared storage: `/home/joerissen/ai/ai4artsed_webserver/exports/`

---

## 🧠 LLM Inference Migration: Ollama → GPU Service (2026-02-23)

**Kontext:** 3 Inference-Backends (Ollama/GGUF, GPU Service/safetensors, SwarmUI/ComfyUI) konkurrierten blind um den gleichen GPU-VRAM. Ollama und GPU Service wussten nichts voneinander — ein Safety-Modell via Ollama konnte eine Diffusers-Pipeline aus dem VRAM verdrängen, ohne dass der VRAMCoordinator es mitbekam.

### Decision 1: LLMInferenceBackend als separater VRAMBackend (nicht TextBackend erweitern)

**Problem:** TextBackend lädt Modelle mit `output_hidden_states=True, output_attentions=True` für Latent-Text-Lab-Introspektion. Das kostet signifikant mehr VRAM als reines Inferencing.

**Decision:** Separater `LLMInferenceBackend` — gleiche VRAMBackend-Architektur, aber ohne Introspektions-Flags.

**Begründung:**
- TextBackend = pädagogische Introspektion (Attention Maps, Embedding-Interpolation, Bias-Probing)
- LLMInferenceBackend = Produktions-Inference (Safety, DSGVO, Translation, Interception)
- Verschiedene Concerns: Introspection braucht hidden_states (VRAM-teuer), Inference nicht
- Beide registrieren sich beim VRAMCoordinator → gegenseitige Eviction funktioniert

### Decision 2: Ollama-Fallback per-Call, nicht per-Session

**Problem:** GPU Service kann neustarten (VRAM-Cleanup, Updates). Alle LLM-Aufrufe würden in dieser Zeit fehlschlagen.

**Decision:** `LLMClient` versucht GPU Service pro Aufruf, fällt auf `ConnectionError`/`Timeout` per-Call auf Ollama zurück.

**Begründung:**
- Zero-Downtime: GPU Service Neustart → Ollama übernimmt nahtlos
- Keine Konfigurationsänderung nötig (kein "switch to Ollama mode")
- LLM-Fehler (OOM, falsches Modell) werden NICHT auf Ollama umgeleitet — nur Connectivity-Fehler

### Decision 3: Model-Name-Mapping statt Doppelte Konfiguration

**Problem:** DevServer config.py referenziert Ollama-Modellnamen (`qwen3:1.7b`), GPU Service braucht HuggingFace-IDs (`Qwen/Qwen3-1.7B`).

**Decision:** `LLM_MODEL_MAP` in `gpu_service/config.py` — bekannte Ollama-Namen werden automatisch auf HF-IDs gemappt. Unbekannte Namen werden as-is versucht (könnten bereits HF-IDs sein).

**Begründung:**
- DevServer-Config bleibt unverändert (Ollama-Namensstil beibehalten)
- Kein Admin muss HF-IDs kennen
- Neue Modelle: einfach Eintrag in Map hinzufügen

**Affected Files:** 4 neue + 9 modifizierte Dateien (siehe Session 202 Devlog)

---

## 🔬 Session Export: Device-ID statt User-ID als Filter (2026-02-21)

**Kontext:** Der "User"-Filter im Session Data Export (Forschungsdaten-Tab) war funktionslos — `user_id` ist fast immer "anonymous". Die Plattform hat aber ein Device-ID-System (Favorites/Browser-ID), das Sessions eindeutig einem Gerät zuordnet. `device_id` wird bereits in jeder `metadata.json` gespeichert (`pipeline_recorder.py`).

**Decision:** `user_id`-Filter komplett durch `device_id`-Filter ersetzen (Backend + Frontend).

**Begründung:**
- `user_id` = "anonymous" in 99%+ der Sessions → kein Filterwert
- `device_id` = pro Browser eindeutig → ermöglicht Gerät-basierte Analyse (z.B. "alle Sessions von iPad #1 im Kurs")
- Device-Dropdown zeigt nur Geräte im aktuellen Filter-Fokus (Datum/Config/Safety), da Backend unique values NACH Filterung sammelt

**Details:**
- Backend: Query-Param `user_id` → `device_id`, Filterung auf `metadata.get('device_id')`, Response `"devices"` statt `"users"`
- Frontend: Stats, Filter-Dropdown (gekürzt auf 8 Zeichen), Tabelle, Detail-Modal, PDF-Export
- Kein Breaking Change für bestehende `metadata.json` (Feld `device_id` war schon immer gespeichert)

**Affected Files:** `settings_routes.py`, `SessionExportView.vue`

---

## 🧠 LATENT TEXT LAB: Dekonstruktive LLM-Introspektion als GPU-Service-Proxy (2026-02-15)

**Kontext:** Die Plattform hatte dekonstruktive Tools für Bildmodelle (Attention Cartography, Feature Probing, Concept Algebra, Denoising Archaeology), aber keine Werkzeuge für Sprachmodelle. Lehrkräfte und Schüler konnten nicht beobachten, wie LLMs intern funktionieren — welche Biases kodiert sind, wie verschiedene Modelle dieselbe Information repräsentieren, oder wie sich gezielte Manipulationen auswirken.

### Decision 1: GPU-Service-Proxy statt In-Process-Execution

**Problem:** LLM-Modelle (LLaMA-8B, Mistral) benötigen 4-20GB VRAM und müssen mit dem VRAM-Koordinator interagieren, der im GPU Service läuft.

**Lösung:** DevServer → GPU Service HTTP-Proxy, identisch zum Muster bei Diffusers und HeartMuLa:
- `text_routes.py` (DevServer) = stateless proxy, jeder Endpoint ruft `TextClient._post()` auf
- `text_backend.py` (GPU Service) = alle Modelle, Tensoren, PyTorch-Hooks
- `text_client.py` (DevServer) = HTTP-Client mit Timeout-Handling

**Begründung:** DevServer ist der pädagogische Orchestrator, kein ML-Runtime. GPU-Inferenz gehört in den GPU Service — einheitlich für alle Modality (Bild, Musik, Text).

**Alternative verworfen:**
- ❌ LLM direkt im DevServer laden → VRAM-Koordination unmöglich, Prozessisolation verletzt

### Decision 2: Drei wissenschaftlich fundierte Tabs statt freier Exploration

**Problem:** Erste Prototypen (Session 175-176) boten generische Werkzeuge (Token Surgery, Embedding-Interpolation, Attention Maps). Das war technisch beeindruckend, aber pädagogisch undurchsichtig — Schüler wussten nicht, *was* sie damit untersuchen sollten.

**Lösung (Session 177 — wissenschaftliche Neufundierung):**

| Tab | Forschungsfrage | Paper |
|-----|----------------|-------|
| 1. Representation Engineering | "Kann man Konzept-Richtungen im Aktivierungsraum finden und Generation steuern?" | Zou 2023, Li 2024 |
| 2. Vergleichende Modell-Archäologie | "Wie repräsentieren verschiedene Modelle dieselbe Information?" | Belinkov 2022, Olsson 2022 |
| 3. Bias-Archäologie | "Welche systematischen Verzerrungen sind in den Gewichten kodiert?" | Zou 2023, Bricken 2023 |

Jeder Tab hat eine klare Forschungsfrage, ein definiertes Experiment-Protokoll, und vordefinierte Presets. Die früheren generischen Tools (Token Surgery, Interpolation, Attention Maps, Layer Analysis) bleiben als API-Endpoints erhalten, werden aber nicht mehr als eigenständige UI-Elemente exponiert.

**Begründung:**
- Geführte Forschung statt offene Exploration (Zielgruppe 13-17 Jahre)
- Preset-Experimente (Gender Bias, Sentiment, Domain) senken die Einstiegshürde
- Jeder Tab referenziert explizit die zugrundeliegende Forschung (Dropdown mit Paper-Referenzen)

### Decision 3: LLM-Interpretation statt Chat-Overlay

**Problem:** Bias-Archäologie zeigt rohe Generierungstexte (Baseline vs. Manipulation). Jugendliche sehen z.B. dass masculine-Suppression identisch zur Baseline ist, verstehen aber nicht *warum* (weil das Modell "they" als Default verwendet).

**Lösung:** Automatische LLM-Interpretation via `POST /api/text/interpret`:
- Reuse von `call_chat_helper()` (multi-provider LLM dispatch aus `chat_routes.py`)
- Pädagogischer System-Prompt (sachlich, 3-5 Sätze, Sprache der Eingabe)
- Direkt unter den Ergebnissen, ohne User-Interaktion (kein Chat-Overlay, kein Button)
- Fail-open: LLM-Fehler blockieren nie die Ergebnisanzeige

**Interpretation läuft auf DevServer, NICHT GPU Service:**
Die Interpretation nutzt das `CHAT_HELPER_MODEL` (z.B. Mistral Large via Ollama, Bedrock, OpenRouter) — das ist die pädagogische Schicht des DevServers. Der GPU Service bleibt für Tensor-Operationen reserviert.

**Alternativen verworfen:**
- ❌ Chat-Overlay (Träshy) → erfordert User-Interaktion, bricht den Experiment-Flow
- ❌ GPU-Service-seitige Interpretation → vermischt Tensor-Ops und Pädagogik
- ❌ Statische Erklärtexte → können nicht auf die tatsächlichen Ergebnisse eingehen

### Decision 4: Token-Resolution mit Varianten

**Problem:** BPE-Tokenizer kodieren `" he"` (mit Leerzeichen) und `"he"` (ohne) als verschiedene Token-IDs. Naive Token-Resolution (`tokenizer.encode("he")`) findet nur eine Variante → unvollständige Bias-Suppression.

**Lösung:** `_resolve_token_ids(tokenizer, words)` im GPU-Service resolvet für jedes Wort drei Varianten:
1. Bare: `"he"` → Token-ID 123
2. Space-prefixed: `" he"` → Token-ID 456
3. Capitalized: `"He"`, `" He"` → Token-IDs 789, 012

Alle gefundenen IDs werden gesammelt und für Boost/Suppress verwendet.

### Decision 5: Additive statt multiplikative Logit-Manipulation

**Problem (Session 177 Bug):** Multiplikative Manipulation (`logits *= factor`) verursacht Softmax-Kollaps — ein Token mit hohem Logit dominiert komplett, die Verteilung wird zu einem Dirac-Delta.

**Lösung:** Additive Manipulation (`logits += factor`). Verschiebt die Logits gleichmäßig, ohne die relative Skalierung zu zerstören. Suppression bleibt `-inf` (komplett blockieren).

**Betroffene Dateien:**
- `gpu_service/services/text_backend.py` — Core: TextBackend, _resolve_token_ids(), _get_decoder_layers()
- `gpu_service/routes/text_routes.py` — REST endpoints (TEXT_ENABLED guard)
- `devserver/my_app/routes/text_routes.py` — DevServer proxy + /interpret endpoint
- `devserver/my_app/services/text_client.py` — HTTP client
- `public/ai4artsed-frontend/src/views/latent_lab/latent_text_lab.vue` — Vue component (3 tabs)
- `public/ai4artsed-frontend/src/i18n.ts` — DE+EN translations
- `docs/ARCHITECTURE PART 28 - Latent-Lab.md` — Architecture documentation

---

## 🔐 RESEARCH-LEVEL-GATING: Canvas & Latent Lab hinter Safety-Level-Gate (2026-02-11)

**Kontext:** Canvas und Latent Lab nutzen direkte Pipeline-Aufrufe ohne vollständige 4-Stage-Safety (Stage 2 wird übersprungen, Stage 1/3 sind optional). Statt Safety in jeden experimentellen Endpoint nachzurüsten, wird der Zugang gegated: Diese Features sind nur ab Safety-Level `adult` verfügbar.

**Entscheidungen:**

### Decision 1: Safety-Level `off` → `research` umbenennen

Das alte Label `off` suggerierte "Development only / kaputt" — tatsächlich ist es ein bewusster Research-Modus für Erwachsene (16+). Neuer Name `research` kommuniziert den Zweck klarer. Hierarchie: `kids` < `youth` < `adult` < `research`.

**Betroffene Dateien:** `config.py`, `schema_pipeline_routes.py`, `stage_orchestrator.py`, `workflow_logic_service.py`, `export_manager.py`, `workflow_streaming_routes.py` — insgesamt ~25 Stellen (Vergleiche, Docstrings, Default-Werte).

### Decision 2: Feature-Gating statt Endpoint-Sicherung

**Problem:** Canvas und Latent Lab operieren absichtlich ohne Stage-2-Interception und mit optionaler Safety. Vollständige Safety nachzurüsten würde den pädagogisch-dekonstruktiven Charakter zerstören (z.B. Partial Elimination benötigt unverfälschte Vektoren).

**Lösung:** Zugangs-Gating auf Frontend-Ebene:
- `kids`/`youth` → Cards sichtbar aber deaktiviert (Opacity 0.4, Schloss-Icon, kein Klick)
- `adult` → Normal klickbar (adult hat eigene §86a + DSGVO Safety-Stages)
- `research` → Compliance-Dialog pro Session, dann klickbar

**Transparenz-Prinzip:** Locked Cards werden angezeigt, nicht ausgeblendet — Nutzer sehen, dass es mehr gibt, und verstehen warum es gesperrt ist.

### Decision 3: Session-basierte Compliance-Bestätigung (nur `research`)

Bei Safety-Level `research` müssen Nutzer pro Browser-Session eine Compliance-Bestätigung abgeben (Warnung: keine Filter aktiv, Altersempfehlung 16+). Die Bestätigung ist ein `ref` (kein `localStorage`) — Reset bei Page-Reload.

**Begründung:** `adult`-Level hat noch §86a + DSGVO Safety-Stages aktiv, daher kein Compliance-Dialog nötig. Nur `research` (= komplett ungefiltert) erfordert bewusste Bestätigung.

### Decision 4: Öffentlicher Safety-Level-Endpoint

`GET /api/settings/safety-level` — ohne Auth, da der Safety-Level kein Geheimnis ist (er bestimmt nur, welche Features sichtbar sind, nicht welche Daten zugänglich sind). Frontend-Store (`safetyLevel.ts`) fetcht beim App-Start und cached im Pinia-Store.

**Alternativen verworfen:**
- ❌ Safety in Latent-Lab-Endpoints nachrüsten → zerstört wissenschaftlichen Charakter
- ❌ Features komplett ausblenden statt locken → Nutzer wissen nicht, was es gibt
- ❌ Compliance per localStorage → zu persistent, Session-Reset ist bewusste Entscheidung
- ❌ Compliance auch für `adult` → unnötig, adult hat eigene Safety-Stages

**Betroffene Dateien:**
- Backend: `config.py`, `settings_routes.py`, `schema_pipeline_routes.py`, `stage_orchestrator.py`, `workflow_logic_service.py`, `export_manager.py`, `workflow_streaming_routes.py`
- Frontend: `stores/safetyLevel.ts` (NEU), `components/ResearchComplianceDialog.vue` (NEU), `views/LandingView.vue`, `router/index.ts`, `main.ts`, `i18n.ts`

---

## 🧪 LATENT LAB: Dekonstruktive Configs zu einem Modus zusammengefasst (2026-02-11)

**Kontext:** Die Plattform hatte mehrere separate dekonstruktive Workflows:
- **Hallucinator** (ehemals Surrealizer) — CLIP-L/T5 Extrapolation
- **Split & Combine** — Semantische Vektorfusion zweier Prompts
- **Partial Elimination** — Dimensionselimination im Vektorraum
- **Attention Cartography** — Cross-Attention Visualisierung
- **Feature Probing** — Embedding-Dimensionsanalyse + selektiver Transfer

Diese waren teils als eigenständige Views (`/surrealizer`), teils als Legacy-ComfyUI-Workflows, teils gar nicht über die UI erreichbar. Es fehlte ein konzeptueller Rahmen.

**Entscheidung: Ein "Latent Lab" als Forschungsmodus**

Alle dekonstruktiven, vektorraumbasierten Operationen werden unter `/latent-lab` als Tab-basierter Modus zusammengefasst. Das Latent Lab ist kein produktives Generierungstool, sondern ein Forschungsinstrument für:
1. **Attention Cartography** — Welche Tokens beeinflussen welche Bildregionen?
2. **Feature Probing** — Welche Embedding-Dimensionen kodieren welche Semantik?
3. **Concept Algebra** — Vektorarithmetik im Embedding-Raum (planned)
4. **Encoder Fusion** — Encoder-übergreifende Interpolation (planned)
5. **Denoising Archaeology** — Schichtweise Denoising-Analyse (planned)

**Begründung:**
- Gemeinsamer konzeptueller Rahmen: "Was passiert im Inneren des Modells?"
- Gemeinsames Safety-Profil: Stage-2-Bypass, da Prompts unverfälscht bleiben müssen
- Gemeinsame Zielgruppe: Fortgeschrittene Nutzer (→ `adult`/`research` Safety-Level)
- Klare Abgrenzung von produktiven Modi (Text/Bild/Musik-Transformation)

**Diffusers als flexible Plattform:**

Die Migration von ComfyUI-Workflows zu Diffusers (begonnen mit dem Hallucinator, Session 162) ermöglicht tiefere Modell-Introspektion. Diffusers bietet:
- Direkten Zugriff auf individuelle Text-Encoder (`pipe._get_clip_prompt_embeds()`, `pipe._get_t5_prompt_embeds()`)
- Hot-swappable Attention-Prozessoren (Custom `AttentionCaptureProcessor` statt SDPA)
- Tensor-Operationen ohne Workflow-Overhead (Embedding-Manipulation, Dimensionsanalyse)
- Programmierbare Pipeline-Schritte (Denoising-Loop Introspection)

ComfyUI ist node-graph-basiert — perfekt für "normales" Generieren, aber schlecht für Introspection, weil die internen Tensoren zwischen Nodes nicht sichtbar sind. Diffusers gibt programmatischen Zugriff auf alle Zwischenschritte.

**Hallucinator bleibt separat:** Der Hallucinator (`/surrealizer`) bleibt als eigenständige View bestehen — er ist das am meisten genutzte dekonstruktive Tool und hat einen eigenen kreativen Workflow (Alpha-Slider-Exploration). Integration ins Latent Lab ist für die Zukunft vorgesehen.

**Alternativen verworfen:**
- ❌ Jedes dekonstruktive Tool als eigene Top-Level-Route → zu viele Einträge in Navigation
- ❌ Alles in ComfyUI belassen → keine Tensor-Introspektion möglich
- ❌ Hallucinator sofort in Latent Lab integrieren → zu großer Umbau, eigenständiger Workflow

---

## 🏠 LANDING PAGE RESTRUCTURE: Feature-Dashboard + Kontextuelle Preset-Auswahl (2026-02-10)

**Kontext:** Die Plattform ist über ihren ursprünglichen Einstiegspunkt (`/select` = PropertyQuadrantsView) hinausgewachsen. Diese Seite zeigte Interception-Presets als Einstiegserlebnis — aber Canvas, HeartMuLa, Surrealizer und Latent Lab nutzen gar keine Interception-Presets. Zwei verschiedene Anliegen ("Welches Feature?" vs. "Welcher Interception-Stil?") waren auf einer Seite vermischt.

**Entscheidung:**
1. **Neue Landing Page** (`/`) als Feature-Dashboard mit 6 Karten (Text-Transformation, Bild-Transformation, Bildfusion, Musikgenerierung, Canvas Workflow, Latent Lab) — informiert über das Forschungsprojekt UND leitet zu den Features
2. **Preset-Auswahl wird kontextuell**: InterceptionPresetOverlay als Fullscreen-Bubble-Overlay, nur in den Views die es betrifft (text/image/multi-image transformation), ausgelöst durch Icon-Button in der Context-MediaInputBox
3. **`/select` komplett entfernt** — kein Redirect, einfach weg. Das Bubble-Visual lebt exklusiv im Overlay weiter
4. **Header-Reihenfolge** didaktisch: einfach→komplex (Text → Bild → Multi-Bild → Musik → Canvas → Latent Lab)

**Begründung:**
- Feature-Auswahl ≠ Preset-Auswahl — zwei verschiedene Entscheidungsebenen, die getrennt gehören
- Presets sind nur für Interception-Pipelines relevant (text_transformation, text_transformation_recursive) — andere Modi (Musik, Canvas, Latent Lab) haben eigene Konfigurationslogik
- Landing Page gibt Forschungskontext (BMBFSFJ-Förderung, pädagogischer Zweck) — wichtig für Workshop-Teilnehmer und externe Besucher
- Staggered Preview-Rotation (±800ms Jitter pro Karte) vermeidet uniformes Umschalten

**Alternativen verworfen:**
- ❌ `/select` als Redirect auf `/` behalten → unnötige Altlast
- ❌ Preset-Overlay global verfügbar → macht keinen Sinn in Musik/Canvas/Latent Lab
- ❌ Composable aus PropertyCanvas extrahieren → PropertyCanvas wird toter Code (kein Route mehr)

**Betroffene Dateien:**
- `src/views/LandingView.vue` (NEU)
- `src/components/InterceptionPresetOverlay.vue` (NEU)
- `src/components/MediaInputBox.vue` (showPresetButton Prop)
- `src/views/text_transformation.vue`, `image_transformation.vue`, `multi_image_transformation.vue` (Overlay-Verdrahtung)
- `src/App.vue` (Header-Icons: Reihenfolge, Latent Lab Mikroskop, LoRA Papagei)
- `src/router/index.ts` (Route-Änderungen)
- `src/i18n.ts` (landing + presetOverlay + multiImage Keys)

---

## 🔬 HALLUCINATOR: Diffusers Backend + Token-Level CLIP-L/T5 Extrapolation (2026-02-08)

**Status:** ✅ IMPLEMENTED
**Session:** 162

### Decision 1: Migrate from joint-embedding blending to individual-encoder token-level fusion

**The Hallucinator's surreal effect comes from extrapolating BETWEEN two different text encoder representations, not from blending joint SD3 embeddings.**

### Problem (vorher)

The Diffusers backend used `pipe.encode_prompt()` which returns **joint SD3 embeddings** — all three text encoders (CLIP-L + CLIP-G + T5) concatenated into one tensor `(1, 589, 4096)`. Blending two such tensors (one with CLIP active/T5 empty, one with CLIP empty/T5 active) had a destructive effect:

```
At α=20, CLIP region: -19 * CLIP(prompt) + 20 * CLIP("")
→ Pushes CLIP embeddings toward huge NEGATIVE values of the prompt
→ DESTROYS the signal instead of extrapolating between encoder spaces
→ α=10 already extreme, α=25 white/blank image
```

### Lösung (nachher)

Access individual text encoders via `pipe._get_clip_prompt_embeds()` and `pipe._get_t5_prompt_embeds()`, replicating the original ComfyUI `ai4artsed_t5_clip_fusion` node exactly:

1. CLIP-L encodes prompt independently → (1, 77, 768)
2. T5-XXL encodes prompt independently → (1, 512, 4096)
3. Pad CLIP-L to 4096d (zero-padded)
4. LERP first 77 tokens: `(1-α)·CLIP-L + α·T5` (extrapolation at α>1)
5. Append remaining T5 tokens (78-512) unchanged → semantic anchor
6. Same fusion for negative prompt, all 4 tensors bypass `encode_prompt()`

**At α=20:** `fused[0:77] = -19·CLIP-L + 20·T5` — pushes 19× past T5 into unexplored vector space. The model hallucinates because it must interpret out-of-distribution vectors.

### Decision 2: CLIP-L only — no CLIP-G anywhere in the fusion

The original ComfyUI workflow loads only `clip_l.safetensors` and `t5xxl_enconly.safetensors` — CLIP-G is absent from both embedding AND pooled output. We match this exactly:
- **Fused tokens:** CLIP-L (768d, zero-padded to 4096d) vs T5 (native 4096d)
- **Pooled output:** CLIP-L real (768d) + zeros (1280d) = 2048d — NO real CLIP-G pooled
- **Rationale:** Real CLIP-G pooled gives the DiT strong visual anchoring that fights extrapolation → incoherent results. The zeroed CLIP-G in pooled is essential for the surreal effect.

### Decision 3: Rename "Surrealizer" → "Hallucinator" (display name only)

The effect is technically **AI hallucination** (model interpreting vectors outside its training distribution), not stylistic surrealism. Renamed all user-facing text; internal IDs (`surrealizer`, file names, routes) kept unchanged to avoid breaking changes.

### Betroffene Dateien
- `devserver/my_app/services/diffusers_backend.py` — `generate_image_with_fusion()` rewritten
- `devserver/schemas/configs/interception/surrealizer.json` — description, context, name, tags
- `devserver/schemas/configs/output/surrealization_diffusers.json` — name, description
- `devserver/schemas/configs/output/surrealization_legacy.json` — name
- `devserver/schemas/chunks/output_image_surrealizer_diffusers.json` — description, alpha docs, notes
- `public/ai4artsed-frontend/src/i18n.ts` — DE+EN: new explanations, slider labels
- `public/ai4artsed-frontend/src/views/surrealizer.vue` — i18n'd slider labels, button text
- `public/ai4artsed-frontend/src/components/DokumentationModal.vue` — rewritten explanation
- `docs/ARCHITECTURE PART 22` — Diffusers backend section, technical analysis

---

## 🛡️ SAFETY: Post-Generation VLM Image Check + Safety-Architektur Klarstellung (2026-02-07)

**Status:** ✅ IMPLEMENTED
**Session:** 161

### Decision 1: Post-Generation VLM Safety Check

**Text-basierte Safety-Checks können nicht vorhersagen, was ein Bildgenerator tatsächlich erzeugt. Lösung: Das generierte Bild mit einem lokalen Vision-Language-Model (qwen3-vl:2b) analysieren, bevor es ans Frontend geht.**

### Problem (vorher)

Stage 1 und Stage 3 prüfen den **Prompt-Text** — aber ein harmloser Prompt ("visuell faszinierende Szene im Wald") kann ein verstörendes Bild produzieren. Es gab keine Prüfung des **tatsächlich generierten Bildes**.

### Lösung (nachher)

- `_vlm_safety_check_image()` in `schema_pipeline_routes.py` — direkte Ollama-Call nach Stage 4
- Liest Bild aus `recorder.get_entity_path('output_image')`, base64-encoded
- Empirisch getestete Prompts für kids (6-12) und youth (14-18)
- Nur für `media_type == 'image'` und `safety_level in ('kids', 'youth')`
- Fail-open bei Fehler (VLM-Ausfall blockt nicht)
- `VLM_SAFETY_MODEL = "qwen3-vl:2b"` in `config.py`

### Decision 2: Safety bedeutet verschiedenes an verschiedenen Stellen

Die Safety-Architektur hat **drei unabhängige Schutzebenen** mit unterschiedlichen Zielen:

| Schutzebene | Was wird geschützt | Wann aktiv | Wo |
|---|---|---|---|
| **§86a StGB** | Vor illegalen Inhalten (Nazi-Symbole, Terror) | IMMER (auch adult) | Stage 1 |
| **DSGVO** | Vor Verarbeitung persönlicher Daten | IMMER (auch adult) | Stage 1 (SpaCy NER) |
| **Jugendschutz** | Vor alters-unangemessenen Inhalten | kids/youth only | Stage 1 + 3 + VLM |

**Wichtig:** DSGVO-Safety ≠ Jugendschutz. §86a ist strafrechtlich, DSGVO ist datenschutzrechtlich, Jugendschutz ist pädagogisch. Alle drei koexistieren.

### Technische Erkenntnisse: qwen3-vl Thinking Mode

- qwen3-vl:2b nutzt standardmäßig Thinking Mode
- Antwort landet in `message.thinking`, nicht `message.content`
- `num_predict` muss hoch genug sein (500) für Thinking + Antwort
- Code prüft beide Felder (`content` und `thinking`)

### Betroffene Dateien
- `devserver/config.py` — `VLM_SAFETY_MODEL` Variable
- `devserver/my_app/routes/schema_pipeline_routes.py` — `_vlm_safety_check_image()` + Insertion in Streaming-Flow

### Offene Frage
- Video-Generierung: VLM-Check für Videos noch nicht implementiert (media_type != 'image' wird übersprungen)

---

## 📚 WIKIPEDIA: Opt-In per Config statt Opt-Out per Request (2026-02-06)

**Status:** ✅ IMPLEMENTED
**Session:** 160

### Decision

**Wikipedia-Research wird von opt-out (global aktiv, per Request abschaltbar) auf opt-in (per Config aktivierbar) umgestellt.**

### Problem (vorher)

Wikipedia-Instruktionen waren hardcoded in `manipulate.json` — jeder `manipulate`-Chunk-Call enthielt die gesamten Wikipedia-Anweisungen (70+ Sprachen, ~2KB Prompt-Text). Pipelines die kein Wikipedia brauchten (Musik, Code) mussten `skip_wikipedia: true` im Frontend-Request senden. Das war:
- Architektonisch fragwürdig (Feature wird zum Problem das man vermeiden muss)
- Fehleranfällig (vergessenes Flag → Wikipedia-Loop korrumpiert Output)
- Token-Verschwendung (Wikipedia-Instruktionen in jedem Prompt, auch bei Lyrics-Generierung)

### Lösung (nachher)

- Wikipedia-Instruktionen in eigenem Modul: `schemas/engine/wikipedia_prompt_helper.py`
- `manipulate.json` Template ist sauber: nur Task + Context + Prompt
- Config-level Steuerung: `"meta": {"wikipedia": true}` in Interception-Config JSONs
- `pipeline_executor._execute_single_step()` prüft Config-Flag, injiziert Instruktionen + Loop nur wenn aktiv
- `skip_wikipedia` komplett entfernt (Frontend + Backend)

### Betroffene Dateien
- `schemas/engine/wikipedia_prompt_helper.py` (NEU)
- `schemas/chunks/manipulate.json` (bereinigt)
- `schemas/engine/pipeline_executor.py` (opt-in Logik)
- 28 pädagogische Interception-Configs (`"wikipedia": true`)
- `schema_pipeline_routes.py` (skip_wikipedia entfernt)
- `music_generation.vue`, `music_generation_v2.vue` (skip_wikipedia entfernt)

### Prinzip
Wikipedia-Research ist ein pädagogisches Feature der `text_transformation.vue` für Kunst/Kultur-Configs. Es gehört nicht in den generischen `manipulate`-Chunk.

---

## 🎵 MUSIC-GENERATION: Unified Simple/Advanced Mode (2026-02-06)

**Status:** ✅ IMPLEMENTED
**Session:** 158

### Decision

**Beide Music-Generation-UIs (V1 + V2) bleiben erhalten und werden über einen Simple/Advanced Toggle auf einer gemeinsamen Seite angeboten.**

### Reasoning

**Pädagogische Analyse der beiden Ansätze:**

| Aspekt | V1 (Simple) | V2 (Advanced) |
|--------|-------------|---------------|
| Einstiegshürde | Niedrig | Mittel |
| Lerneffekt Musik | Keiner | Hoch (8 Dimensionen) |
| Scaffold ohne Lyrics | Keines | "Theme → Lyrics" |
| Tag-Wissen nötig | Ja (Freitext) | Nein (Chips) |
| ML-Parameter | Keine | Temp/TopK/CFG |

**Lösung: Benutzer wählt selbst**
- **Simple Mode** = V1: Schneller Einstieg, keine Erklärung nötig
- **Advanced Mode** = V2: Musikalisches Lernen, mehr Kontrolle

**Default-Presets für V2:**
- Audio Length: 3:20 (200s) — typische Songlänge
- Temperature: 1.0 — balancierte Kreativität
- Top-K: 65 — etwas fokussierter als 70
- CFG Scale: 2.75 — Mitte von 2.5-3.0 sweet spot

### Implementation

- `music_generation_unified.vue` als Wrapper
- Toggle persistiert in localStorage
- `/music-generation` → unified, `/music-generation-simple` + `/music-generation-advanced` für direkten Zugriff
- Custom Tags in MusicTagSelector für Power-User

---

## 🧠 LLM-STRATEGIE: Wechsel zu Mistral für VRAM-Optimierung (2026-01-29)

**Status:** ✅ DECIDED - Implementation pending
**Session:** 147

### Decision

**Strategische Entscheidung: Lokales LLM (gpt-OSS:120b) durch externes Mistral ersetzen, um VRAM für Bild-/Video-Modelle freizugeben.**

1. **Alle Meta-Prompts für Mistral optimieren**
   - Interception-Prompts, Safety-Checks, Translations
   - Ziel: Gleiche oder bessere Ergebnisse wie mit gpt-OSS:120b

2. **VRAM-Budget für ComfyUI maximieren**
   - Aktuell: 96GB - 65GB (LLM) = 31GB für Bild/Video
   - Zukünftig: 96GB - ~0GB (externes LLM) = 96GB für Bild/Video
   - Ermöglicht: Mehrere Modelle gleichzeitig geladen (wan22 + sd35 + flux + audio)

3. **Zentralen VRAM-Manager implementieren**
   - Ersetzt dezentrale `keep_alive` Settings
   - Koordiniert ComfyUI-Modell-Loading
   - Verhindert VRAM-Thrashing im Workshop-Betrieb

### Reasoning

**Problem-Analyse (Workshop 29.1.2026):**

| Szenario | Problem |
|----------|---------|
| Kleine LLMs (20b) | Qualität zu schlecht für Interception |
| Große LLMs (120b) | Funktioniert gut für Einzelsessions |
| Workshop (120b) | VRAM-Stau: 65GB LLM + wechselnde Bild-Modelle = Thrashing |
| Einzelsession (120b) | Kreativ-Flow behindert: Modellwechsel dauert "ewig" → User vermeidet intuitiv |

**DSGVO-Analyse externer LLM-Anbieter:**

| Anbieter | DSGVO-Status | Verfügbarkeit |
|----------|--------------|---------------|
| OpenAI | ❌ US-Server | Nicht nutzbar |
| Anthropic | ❌ US-Server | Nicht nutzbar |
| Google | ⚠️ Kompliziert | Nicht praktikabel |
| AWS Bedrock EU | ✅ EU-Server | Nur Enterprise-Verträge |
| **Mistral** | ✅ EU (Frankreich) | **Token-basiert für Kleinabnehmer** |

**Einzige DSGVO-konforme Option für Kleinabnehmer: Mistral**

**Qualitätseinschätzung:**
- gpt-OSS:120b: Sehr gute Ergebnisse für Interception
- Mistral Large: Nach bisherigen Tests etwas schwächer
- **Konsequenz:** Meta-Prompts müssen für Mistral optimiert werden

### Architektur-Implikation

**Dezentrales VRAM-Problem (Ist-Zustand):**
```
manipulate.json:          keep_alive: "10m"
safety_check_*.json:      keep_alive: "10m"
prompt_interception.py:   keep_alive: 0  (aktives Entladen!)
image_analysis.py:        keep_alive: "0s"
→ Keine Koordination zwischen Ollama und ComfyUI
→ VRAM-Thrashing bei parallelen Requests
```

**Zentraler VRAM-Manager (Soll-Zustand):**
```
VRAMManager:
  - Trackt verfügbares VRAM (96GB)
  - Reserviert Budget für ComfyUI-Modelle
  - Entscheidet welche Modelle geladen bleiben
  - Kein Ollama mehr → volle 96GB für ComfyUI
```

### Betroffene Dateien (Implementation)

**Phase 1: Mistral-Migration**
- `devserver/config.py` - Model-Konstanten auf Mistral
- `devserver/schemas/chunks/*.json` - Alle LLM-Chunks
- `devserver/schemas/engine/instruction_selector.py` - Meta-Prompts
- `devserver/schemas/engine/prompt_interception_engine.py` - Entfernen von keep_alive=0

**Phase 2: VRAM-Manager**
- `devserver/my_app/services/vram_manager.py` - NEU
- `devserver/my_app/services/comfyui_service.py` - Integration
- `devserver/schemas/engine/backend_router.py` - Integration

### Offene Fragen

1. Mistral-API-Key-Management (*.key Datei)
2. Fallback-Strategie wenn Mistral nicht erreichbar
3. Kosten-Monitoring für Token-Verbrauch
4. Prompt-Optimierung: Wie viel Aufwand für Mistral-Anpassung?

---

## 🔀 MODEL-ROUTING: Prefix-basierte Provider-Auswahl (2026-01-29)

**Status:** ✅ DECIDED & IMPLEMENTED
**Session:** 147

### Decision

**Prefix-basierte Model-Routing bleibt erhalten für maximale Flexibilität.**

Das Model-Prefix bestimmt explizit den Provider:
- `local/model-name` → Ollama (lokales LLM)
- `openrouter/provider/model` → OpenRouter API
- `anthropic/model` → Anthropic API direkt
- `mistral/model` → Mistral API direkt
- `bedrock/model` → AWS Bedrock

### Reasoning

1. **Canvas braucht explizite Auswahl:** User wählt gezielt Provider + Model
2. **Flexibilität:** Gleiche Models über verschiedene Provider verfügbar
3. **Transparenz:** Prefix macht den API-Endpunkt sichtbar
4. **Fallback-fähig:** Prefix kann zu OpenRouter führen wenn direkter Provider nicht verfügbar

### Implementation

Canvas zeigt für Anthropic-Models beide Optionen:
- "Claude Opus 4.5 (OpenRouter)" → `openrouter/anthropic/claude-opus-4.5`
- "Claude Opus 4.5 (Anthropic)" → `anthropic/claude-opus-4.5`

### Affected Files

- `devserver/schemas/engine/prompt_interception_engine.py:146-175` - Routing-Logik
- `devserver/my_app/routes/canvas_routes.py:32-43` - CURATED_TOP_MODELS Liste

---

## 📡 SSE-STREAMING: Real-Time Stage Progress für Generation (2026-01-30)

**Status:** ✅ DECIDED & IMPLEMENTED
**Session:** 148

### Decision

**SSE (Server-Sent Events) für den `/generation` Endpoint, um Badges zum richtigen Zeitpunkt anzuzeigen.**

Statt:
- ❌ Fake 300ms Delay für Safety Badge
- ❌ Badges erst nach kompletter Generation

Jetzt:
- ✅ `stage3_complete` Event → Badges sofort anzeigen
- ✅ `stage4_start` Event → Progress-Animation starten
- ✅ Echte Stage-Trennung im UI sichtbar

### Reasoning

**Problem:**
- "Translated" Badge erschien erst NACH der Bildgenerierung
- Safety Badge verwendete künstlichen 300ms Delay - nicht akkurat
- User hatte keine Ahnung was gerade passiert (Translation? Safety? Generation?)

**Warum SSE statt Split-Requests:**

| Ansatz | Pro | Contra |
|--------|-----|--------|
| **Split (Stage 3 → Stage 4)** | Einfach zu implementieren | Race Conditions, State-Sync |
| **SSE Streaming** | Single Connection, failsafe | Etwas komplexer |

SSE gewählt weil:
1. **Failsafe:** Wenn Connection abbricht, stoppt alles sauber
2. **Kein Race:** Keine zwei Requests die synchronisiert werden müssen
3. **Bestehendes Pattern:** `/interception` nutzt bereits SSE

**Warum "→ EN" statt Flag-Icon:**
- 🇬🇧 (Britische Flagge) war problematisch: Kolonialismus, UK-Zentrismus
- "→ EN" ist neutral, klar, ohne politische Konnotation

### Architecture Impact

**Stage-Separation jetzt sauber:**

| Stage | Funktion | Endpoint | SSE Event |
|-------|----------|----------|-----------|
| Stage 1 | Safety (§86a) | `/interception` | `stage_complete (1)` |
| Stage 2 | Interception | `/interception` | `stage_complete (2)` |
| Stage 3 | Translation + Safety | `/generation` | `stage3_complete` |
| Stage 4 | Generation | `/generation` | `stage4_complete` |

**Backend-Funktionen klar getrennt:**
- `execute_stage1_gpt_oss_unified()`
- `execute_pipeline()` (Stage 2)
- `execute_stage3_safety()`
- `execute_stage4_generation_only()`

### Implementation

**Backend (`schema_pipeline_routes.py`):**
- `execute_generation_streaming()` Generator-Funktion
- Events: `connected`, `stage3_start`, `stage3_complete`, `stage4_start`, `complete`, `blocked`, `error`

**Frontend:**
- Neuer Composable: `useGenerationStream.ts`
- Shared zwischen 3 Views: text_, image_, multi_image_transformation

### Affected Files

| File | Change |
|------|--------|
| `devserver/my_app/routes/schema_pipeline_routes.py` | SSE-Modus für /generation |
| `public/.../composables/useGenerationStream.ts` | NEU - Shared SSE Composable |
| `public/.../views/text_transformation.vue` | Composable eingebunden |
| `public/.../views/image_transformation.vue` | Composable eingebunden |
| `public/.../views/multi_image_transformation.vue` | Composable eingebunden |

---

## 🌍 ANTI-ORIENTALISM & EPISTEMIC JUSTICE: Cultural-Aware AI (2026-01-26)

**Status:** ✅ DECIDED & IMPLEMENTED
**Session:** 136
**Full Analysis:** See `docs/analysis/ORIENTALISM_PROBLEM_2026-01.md`

### Decision

**Two-part solution to prevent orientalist stereotypes in prompt interception:**

1. **Enhanced Meta-Prompt with Anti-Orientalism Rules**
   - Added CULTURAL RESPECT PRINCIPLES to `instruction_selector.py`
   - Explicit FORBIDDEN list: exoticizing, romanticizing, mystifying cultural practices
   - Equality principle: "Use the same neutral, fact-based approach as for Western contexts"
   - Applies universally to ALL interception configs

2. **Wikipedia Lookup in Cultural Reference Language**
   - LLM must use Wikipedia in the CULTURAL REFERENCE LANGUAGE, not prompt language
   - 70+ languages mapped: Africa (15+), Asia (30+), Americas (indigenous), Oceania
   - Example: German prompt about Nigeria → uses Hausa/Yoruba/Igbo Wikipedia (not German)
   - Example: German prompt about Peru → uses Quechua/Aymara Wikipedia (not German)

### Reasoning

**Problem Identified:**
User report: GPT-OSS:120b produced "enormer, furchtbarer exotistischer orientalistischer Kitsch" when processing Nigerian cultural festival prompt. LLMs defaulted to orientalist tropes (exotic, mysterious, timeless) even with factual data available.

**Root Cause:**
- LLMs trained on Western-centric corpora default to orientalist framing
- Wikipedia lookup alone insufficient - models need explicit anti-stereotype rules
- Using German/European Wikipedia for non-European topics perpetuates colonial knowledge hierarchies

**Why Epistemic Justice Matters:**
- **Linguistic Sovereignty:** Cultures have the right to be represented in their own languages
- **Local Knowledge:** Local-language Wikipedias written BY local communities FOR local contexts
- **Decolonizing AI:** Breaks colonial pattern of European languages as "universal" knowledge sources
- **Pedagogical Integrity:** AI4ArtsEd makes transformation choices visible and criticalizable - orientalist output undermines this goal

**Theoretical Foundation:**
Based on postcolonial theory (Said, Fanon, Spivak):
- Edward Said: "Orientalism" as Western construction of exotic Other
- Frantz Fanon: Dehumanization through exoticization
- Gayatri Chakravorty Spivak: Epistemic violence of representation

### Implementation

**Files Modified:**
- `devserver/schemas/engine/instruction_selector.py` - Anti-orientalism meta-prompt
- `devserver/schemas/chunks/manipulate.json` - Wikipedia cultural reference language mapping
- `docs/analysis/ORIENTALISM_PROBLEM_2026-01.md` - Complete analysis and testing strategy

**Testing:**
- ✅ Original failing case: "Das wichtigste Fest im Norden Nigerias"
- ✅ Result: Factual, respectful narrative WITHOUT orientalist tropes
- ✅ Improvement: From "furchtbarer exotistischer Kitsch" to culturally grounded output

### Impact

**Cultural Coverage (70+ Languages):**
- **Nigeria:** ha (Hausa), yo (Yoruba), ig (Igbo), en
- **India:** hi, ta, bn, te, mr, gu, kn, ml, pa, ur (10+ regional languages!)
- **China:** zh, zh-yue (Cantonese)
- **Latin America:** es, pt, qu (Quechua), ay (Aymara), nah (Nahuatl)
- **Indigenous North America:** ik (Inuktitut), chr (Cherokee)
- **Oceania:** mi (Māori), to (Tongan), sm (Samoan), fj (Fijian)
- **Africa:** sw (Swahili), am (Amharic), zu (Zulu), xh (Xhosa), ar, ber

**Pedagogical Significance:**
This is not just a technical fix - it's a fundamental ethical stance. AI4ArtsEd systematically respects cultural diversity and actively resists colonial knowledge patterns. The system now embodies **epistemic justice** as a core architectural principle.

**Related Decisions:**
- See "planetarizer.json" and "one_world.json" configs (already had specific anti-Othering rules)
- This decision extends those principles universally across ALL interception configs

---

## 🎯 CANVAS EVALUATION NODES: Unified 3-Output Architecture (2026-01-25)

**Status:** ✅ DECIDED & IMPLEMENTED
**Session:** 134
**Full Details:** See `docs/ARCHITECTURE_CANVAS_EVALUATION_NODES.md`

### Decision

**ONE unified Evaluation node** with 3 separate TEXT outputs, instead of 7 separate node types (5 evaluation + 2 fork).

### Reasoning

**Problem with Original Plan (7 Nodes):**
- Evaluation and Fork were conceptually ONE decision, split into TWO nodes
- Unclear data flow: What text flows through fork? Input? Commentary? Both?
- UI complexity: 7 new nodes for one logical operation
- Pedagogically unclear: "Fairness Check" + "Binary Fork" = 2 steps for 1 decision

**Why 3 Text Outputs?**
1. **Passthrough (P)**: Original input unchanged (active if binary=true)
   - Use: Evaluation passed → continue workflow
2. **Commented (C)**: Input + "\n\nFEEDBACK: {commentary}" (active if binary=false)
   - Use: Evaluation failed → loop back to Interception with feedback
3. **Commentary (→)**: Just the commentary text (ALWAYS active)
   - Use: Display/Collector for user transparency

**Pedagogical Advantage:**
- Explicit decision points in workflow
- Visible feedback (not "black box")
- Enables iterative improvement (feedback loops)
- One node = one conceptual decision

### Technical Implementation

**Frontend Structure:**
```typescript
// Node properties
evaluationType: 'fairness' | 'creativity' | 'equity' | 'quality' | 'custom'
evaluationPrompt: string
outputType: 'commentary' | 'score' | 'all'
enableBranching: boolean
branchCondition: 'binary' | 'threshold'
thresholdValue: number
trueLabel: string
falseLabel: string
```

**Backend Structure:**
```python
# Evaluation result
{
  'type': 'evaluation',
  'outputs': {
    'passthrough': input_text,  # Original
    'commented': f"{input_text}\n\nFEEDBACK: {commentary}",  # Input + feedback
    'commentary': commentary  # Just commentary
  },
  'metadata': {
    'binary': True/False,
    'score': 0-10,
    'active_path': 'passthrough' | 'commented'
  }
}
```

**Binary Logic:**
- LLM prompt explicitly requests: "Answer ONLY 'true' or 'false'"
- Fallback: No binary → use score (< 5.0 = fail)
- Default if no score: False (safer, triggers feedback)

### Affected Files

- `public/.../types/canvas.ts` - Unified evaluation type
- `public/.../StageModule.vue` - 3-output UI, evaluation config
- `public/.../ModulePalette.vue` - 7 nodes → 1 node
- `devserver/.../canvas_routes.py` - 3 text outputs, binary logic

### Trade-offs

**Chosen:**
- Unified node with dropdown (1 node type)
- 3 separate text outputs
- Binary + Commentary always generated

**Rejected:**
- 7 separate node types (too complex)
- 2 outputs only (loses transparency)
- Combined object output (text doesn't flow properly)
- Default binary=true (unsafe, hides problems)

### Next Steps (Phase 3b - Not Yet Implemented)

**Conditional Execution:**
- Currently: All 3 outputs execute connected nodes
- Goal: Only active path (P or C) executes downstream
- Commentary path always executes (for display/collector)
- Requires: Connection label tracking, active path marking

---

## 🔄 CANVAS EXECUTION: Tracer-Pattern vs. Kahn's Algorithm (2026-01-26)

**Status:** ✅ DECIDED & IMPLEMENTED
**Session:** 134 (Phase 4)

### Decision

Use **simple Tracer-Pattern** (recursive graph traversal) instead of **Kahn's Algorithm** (topological sorting) for Canvas workflow execution.

### Context: The Loop Problem

**Original Plan (rejected):**
- Kahn's Algorithm for topological sorting
- Separate "Loop Controller" Node type
- Complex re-queueing mechanism for feedback loops

**Problem:** Kahn's Algorithm is designed for DAGs (Directed Acyclic Graphs) and rejects cycles by design. Feedback loops (Evaluation → Interception) create intentional cycles.

### User Critique

> "Die versuchte Lösung gegen der Kahn-Bedingung/Loop ist eine Scheinlösung."
> "Wir haben hier keine unkontrollierte Schleifen-Situation im Graph, sondern nichts anderes als eine Loop-End-Konstellation."

The Loop Controller approach was over-engineered - the system doesn't need general cycle handling, just controlled feedback loops.

### Implemented Solution: Tracer-Pattern

```python
def trace(node_id, input_data, data_type):
    # 1. Execute current node
    output_data, output_type, metadata = execute_node(node, input_data, data_type)

    # 2. At Evaluation: filter connections based on score
    if node_type == 'evaluation' and metadata:
        active_path = metadata['active_path']  # 'passthrough' or 'commented'
        # Only follow connections whose label matches active_path

    # 3. For each active connection: recursive trace()
    for conn in active_connections:
        trace(conn['target'], output_data, output_type)
```

**Safety:** `MAX_TOTAL_EXECUTIONS = 50` prevents infinite loops.

### Why This Works

The Canvas system is NOT a general graph executor, but a **pedagogical workflow**:
- Input is always the start
- Collector is always the end
- Between: directed flow with controlled loops
- The "Loop-End-Konstellation" is a feature, not a bug

### Result: Reflexively Acting Frontend

A **world-unique reflexively acting frontend for genAI**:

```
Input → Interception → Evaluation → [Score < 5?]
                            ↓ feedback     ↓ pass
                       Interception    Generation → Collector
```

The system can:
1. Evaluate outputs (Evaluation node with LLM)
2. Provide feedback on failure (Commented output)
3. Send feedback back to Interception (Feedback connection)
4. Iterate until Score >= 5

### Affected Files

- `devserver/my_app/routes/canvas_routes.py` - Complete rewrite: `trace()` function replaces Kahn's algorithm
- `public/.../types/canvas.ts` - `maxFeedbackIterations` property
- `public/.../StageModule.vue` - Feedback-Input connector for Interception/Translation

### Trade-offs

**Chosen:**
- Simple recursive traversal
- Safety limit (50 executions)
- Feedback connections with explicit label

**Rejected:**
- Kahn's Algorithm (rejects cycles)
- Loop Controller node (over-engineered)
- Modified Kahn's with loop-edge exclusion (still complex)

---

## 🤖 LLM SELECTION: Model-Specific Prompting Strategy (2026-01-23)

**Status:** ✅ DECIDED
**Session:** 132
**Full Analysis:** See `docs/LLM_SELECTION_AND_PROMPTING.md`

### Decision

Use **llama4:scout** as primary local model with **model-specific prompt variants** for CLIP optimization.

### Context: SD3.5 Triple CLIP Optimization

The prompt optimization task requires:
- Multilingual I/O (German, Bulgarian, Yoruba, etc.)
- Scene-to-2D transformation (narrative → frozen visual frame)
- Domain knowledge (photography, visual arts)
- Cultural neutrality (no artist names, no art-historical terms)
- Complex instruction following

### Model Evaluation Results

| Model | Multilingual | Low-Resource (Yoruba) | Recommendation |
|-------|--------------|----------------------|----------------|
| **llama4:scout** | 200 languages | ✅ 10x more data | ✅ Primary |
| Mistral Large 3 | 40+ languages | ⚠️ No data | EU-languages only |
| gpt-OSS:120b | Good | ⚠️ No data | Fallback |
| Qwen3-next | 119 languages | ❌ "unoptimized" | Not recommended |

### Key Finding: Prompt Format Sensitivity

Different models need different prompt styles:

| Model Family | Preferred Style |
|--------------|-----------------|
| Claude/OpenAI | Structured `===` sections, detailed rules |
| Llama 4 | Flat, example-heavy, short rules |
| Mistral | Minimal, precise, no nesting |

### Implementation

1. **Primary model:** llama4:scout (67GB, fits in 96GB VRAM)
2. **Prompt variants:** `default`, `llama`, `mistral` for each optimization chunk
3. **Dynamic selection:** Based on configured model name

### Affected Files

- `devserver/schemas/chunks/optimize_clip_prompt.json` (+ variants)
- `devserver/schemas/chunks/optimize_t5_prompt.json` (+ variants)
- `devserver/schemas/chunks/output_image_sd35_large.json`
- `devserver/config.py`
- `docs/LLM_SELECTION_AND_PROMPTING.md` (full analysis)

---

## 🏛️ ARCHITECTURAL PARADIGM: Werkraum → Lab Transition (2026-01-16)

**Status:** ✅ DOCUMENTED (ongoing evolution)
**Context:** Historical transition from workflow-centric to frontend-centric architecture

### The Two Paradigms

| Aspect | Werkraum (Workflow) | Lab (OO/Dezentriert) |
|--------|---------------------|----------------------|
| **Orchestrator** | Server | Frontend/User |
| **Data Flow** | Linear, predetermined | Flexible, user-controlled |
| **Server Role** | "Smart Orchestrator" | "Service Provider" |
| **Client Role** | "Dumb Display" | "Intelligent Composer" |
| **Endpoints** | Unified (server decides stages) | Atomic (client composes services) |

### Historical Context

**Werkraum Era (ComfyUI Workflows):**
- Unidirectional pipeline: Input → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Output
- Server controlled the entire flow
- Frontend was a simple display layer
- Endpoint: `/api/schema/pipeline/execute` handled everything

**Lab Era (Current):**
- Flexible interaction: User can edit any text at any point
- Parallel LLM runs possible
- Every output is editable input for next step
- Frontend orchestrates which services to call

### Architectural Implication: Atomic Backend Services

In the Lab paradigm, the backend provides **atomic services** that the frontend composes:

| Endpoint | Stage | Purpose | Safety |
|----------|-------|---------|--------|
| `/api/schema/pipeline/interception` | 1+2 | User input → Interception | Auto (Stage 1) |
| `/api/schema/pipeline/optimize` | 3a | Model-specific format (clip_g tokens) | No |
| `/api/schema/pipeline/translate` | 3b | German → English translation | No |
| `/api/schema/pipeline/safety` | - | Reusable safety check (for custom flows) | - |
| `/api/schema/pipeline/generation` | 3c+4 | Pre-output safety + Media generation | Auto (Stage 3) |
| `/api/schema/pipeline/legacy` | 1+4 | Direct ComfyUI workflow | Auto (Stage 1) |

**Key Principles:**
1. **Safety is Server responsibility** - Endpoints with user-facing input auto-run safety
   - `/interception` → Stage 1 (Input Safety)
   - `/generation` → Stage 3 (Pre-Output Safety)
   - `/legacy` → Stage 1 (Input Safety)
2. **Atomic services** - Frontend composes the flow, backend executes steps
3. **No skip flags** - Instead of `skip_stage2: true`, just use `/legacy` endpoint

### Example Flows

**text_transformation.vue (Standard Flow):**
```
User Input → /interception (Stage 1+2) → User selects model
           → /optimize + /translate (parallel) → /generation (Stage 4)
```

**surrealizer.vue (Legacy Flow):**
```
User Input → /legacy (Stage 1 + ComfyUI workflow)
```

### Historical Context: Optimization Fix (2026-01-16)

**Bug:** Stage 3 optimization streaming called `/execute`, which always ran Stage 1—redundant.

**Wrong Fix (Werkraum thinking):** Add `skip_stage1` parameter.

**Correct Fix (Lab thinking):** Create `/optimize` endpoint that by design has no Stage 1.

### Coexistence Strategy

Both paradigms coexist in the current codebase:

- **Werkraum patterns** remain for backward compatibility and simpler flows
- **Lab patterns** enable advanced interactive features
- **Migration path:** Gradually decompose unified endpoints into atomic services as needed

### Files Affected

- `schema_pipeline_routes.py`: Both `/execute` (Werkraum) and `/optimize` (Lab) endpoints
- `text_transformation.vue`: Uses Lab pattern (calls atomic services)
- `surrealizer.vue`: Uses Werkraum pattern (single unified call)

---

## 🎯 Active Decision: Unified Export - run_id Across Lab Endpoints (2026-01-17)

**Status:** ✅ IMPLEMENTED
**Context:** Export function was broken - entities split across multiple folders
**Commit:** `7f07197` - `fix(export): Unified run_id and fix image saving for all backends`

### The Problem

In the Lab architecture, `/interception` and `/generation` are **atomic endpoints** called by the frontend. Each endpoint was generating its own `run_id`, resulting in:

- `/interception` → `run_1234_abc/` (input, safety, interception)
- `/generation` → `run_5678_xyz/` (output_image)

**Result:** Export function BROKEN - entities scattered across folders.

### The Solution

**Frontend passes run_id from /interception to /generation:**

```
Frontend (text_transformation.vue)
├── POST /interception → receives run_id in response
├── Stores run_id (currentRunId ref)
└── POST /generation { run_id: currentRunId } → uses SAME folder
```

**Backend changes:**
1. `/interception` initializes Recorder, saves input/safety/interception, returns `run_id`
2. `/generation` accepts optional `run_id`, loads existing Recorder via `load_recorder()`
3. All output entities saved to SAME run folder

### Additional Fix: Multi-Backend Image Saving

**Bug:** Only SD3.5 (`swarmui_generated`) saved images. QWEN, FLUX2, Gemini, GPT-Image failed.

**Root Cause:** Different backends return different output formats:
- `swarmui_generated`: Binary data via SwarmUI API
- `workflow_generated`: `filesystem_path` (QWEN, FLUX2)
- URL outputs: HTTP URLs (Gemini via OpenRouter)
- Base64 outputs: Inline base64 data (OpenAI)

**Fix in `/generation` endpoint:**
```python
elif output_value == 'workflow_generated':
    # Check filesystem_path first (QWEN, FLUX2)
    filesystem_path = output_result.metadata.get('filesystem_path')
    if filesystem_path and os.path.exists(filesystem_path):
        with open(filesystem_path, 'rb') as f:
            file_data = f.read()
        recorder.save_entity(...)
    elif media_files:  # Fallback: Legacy binary data
        ...

# Base64 handling for OpenAI-style outputs
elif output_value and len(output_value) > 100:
    file_data = base64.b64decode(output_value)
    recorder.save_entity(...)
```

### Architectural Note: media_type Consistency ✅ RESOLVED

**User feedback:** The distinction between `media_type: "image"` and `media_type: "image_workflow"` is **ontologically unjustified**.

- SD3.5, QWEN, FLUX2 are ALL image models
- The only valid distinction: image vs video vs audio
- Internal workflow differences should be transparent

**Resolved (2026-01-17):** Eliminated `image_workflow` type - all image models now use `media_type: "image"`.
- Changed `output_image_qwen.json` and `output_image_flux2.json`
- Simplified `backend_router.py` line 749

### Files Modified

| File | Change |
|------|--------|
| `schema_pipeline_routes.py` | Added `load_recorder` import, recorder init in `/interception`, run_id acceptance in `/generation`, filesystem_path + base64 handling |
| `text_transformation.vue` | Added `currentRunId` ref, pass run_id to /generation |

### Test Results

- ✅ SD3.5: Works (unchanged)
- ✅ QWEN: Images saved from `filesystem_path`
- ✅ FLUX2: Images saved from `filesystem_path`
- ✅ Gemini 3 Pro: Images saved from base64
- ✅ GPT-Image: Images saved from base64/URL
- ✅ All entities in ONE unified folder

---

## 🎯 Active Decision: 1 Run = 1 Media Output (2026-01-23, Session 130)

**Status:** ✅ IMPLEMENTED
**Context:** Multiple generations were writing to the same folder, confusing favorites system
**Commits:** `bed0c2c`, `8d07c33`

### The Principle

Each run folder should contain exactly ONE media product (image/video/audio). This ensures:
- Favorites system works correctly (clear 1:1 mapping)
- Research data is clean (each generation has its own context)
- Export function produces coherent artifacts

### The Logic

```
Interception (Start1)     → run_001/ created (no output yet)
Generate (FIRST)          → run_001/ continues → saves output_image
Generate (SECOND)         → run_002/ NEW (run_001 already has output_*)
Generate (THIRD)          → run_003/ NEW
```

**Check in generation endpoint:**
```python
has_output = any(
    e.get('type', '').startswith('output_')
    for e in existing_recorder.metadata.get('entities', [])
)
if has_output:
    run_id = new_run_id()  # Create NEW folder
else:
    run_id = provided_run_id  # Continue existing folder
```

### Immediate Prompt Persistence

Prompts are now saved immediately after LLM generation (not only on user action):
- Interception result: saved immediately after Stage 2 LLM
- Optimized prompt: saved immediately after optimization LLM

This enables research tracking of what the LLM produced vs what the user edited.

### Files Modified

| File | Change |
|------|--------|
| `schema_pipeline_routes.py` | has_output check in generation, immediate save in optimization |
| `text_transformation.vue` | Pass run_id/device_id to optimization endpoint |

### TODO

- [ ] Stop logging changes after generation (currentRunHasOutput flag not working)
- [ ] Sticky UI: restore prompts/image when switching modes

---

## 🎯 Active Decision: Failsafe Transition - SwarmUI Single Front Door (2026-01-08, Session 116)

**Status:** ✅ IMPLEMENTED
**Context:** Centralizing all traffic through SwarmUI (Port 7801) while preserving legacy workflow compatibility
**Date:** 2026-01-08

### The Decision: Route Legacy Workflows via SwarmUI Proxy

**Problem:**
- Legacy workflows (Surrealizer, etc.) were hardcoded to access ComfyUI directly on Port 7821.
- This bypassed SwarmUI's orchestration, queue management, and user history.
- "Split brain" architecture where some requests went to 7801 and others to 7821.

**Solution:**
- **Single Front Door:** All DevServer traffic goes to **Port 7801** (SwarmUI).
- **Proxy Pattern:** Legacy workflows use SwarmUI's `/ComfyBackendDirect/*` endpoints to reach the managed ComfyUI instance.
- **Config Flag:** `USE_SWARMUI_ORCHESTRATION = True` (default).
- **Emergency Fallback:** `ALLOW_DIRECT_COMFYUI` flag allows reverting to Port 7821 if SwarmUI is down.

### Architecture

**Before (Split):**
```
DevServer
├── New Pipelines ───> SwarmUI (7801) ───> ComfyUI (Internal)
└── Legacy Workflows ──> ComfyUI (7821)
```

**After (Unified):**
```
DevServer
├── New Pipelines ───> SwarmUI (7801) ───> ComfyUI (Internal)
└── Legacy Workflows ──> SwarmUI (7801) ──> /ComfyBackendDirect/ ──> ComfyUI (Internal)
```

### Benefits

1. **Centralized Management:** SwarmUI controls the queue for ALL generations (legacy and new).
2. **Simplified Networking:** Only one port (7801) needs to be exposed/managed.
3. **Compatibility:** Legacy workflows run without modification (transparent proxying).
4. **Resilience:** If SwarmUI is running, ComfyUI is accessible.

### Implementation Details

**Files Modified:**
- `config.py`: Added feature flags.
- `legacy_workflow_service.py`: Dynamic base URL selection.
- `swarmui_client.py`: Added support for legacy image retrieval methods via proxy.
- `backend_router.py`: Updated routing logic for legacy chunks.

---

## 🎓 DESIGN DECISION (2026-01-13): LoRA Training Studio Path Configuration

**Date:** 2026-01-13
**Session:** 115

### Decision

All LoRA training paths must be configured in `config.py` using environment variables with relative fallbacks. No hardcoded absolute paths or usernames in repository code.

### Context

**Problem:**
- Initial training_service.py had hardcoded paths like `/home/joerissen/ai/kohya_ss_new`
- Usernames in git repo = non-portable, security issue
- Different developers/deployments have different directory structures

### Solution

**Path Configuration Pattern:**
```python
# config.py
_AI_TOOLS_BASE = _SERVER_BASE.parent  # Derived from project location

KOHYA_DIR = Path(os.environ.get("KOHYA_DIR", str(_AI_TOOLS_BASE / "kohya_ss_new")))
LORA_OUTPUT_DIR = Path(os.environ.get("LORA_OUTPUT_DIR", str(_AI_TOOLS_BASE / "SwarmUI/Models/loras")))
```

**Model-Specific Prefixes:**
- NOT a global config variable
- Determined by model-specific config generator method
- `_generate_sd35_config()` → adds `"sd35_"` prefix automatically
- Future: `_generate_flux_config()` → adds `"flux_"` prefix

### Affected Files
- `devserver/config.py` - Path variables added
- `devserver/my_app/services/training_service.py` - Imports from config

---

## 🧠 DESIGN DECISION (2026-01-13): VRAM Management for Training

**Date:** 2026-01-13
**Session:** 115

### Decision

Training operations must check available VRAM before starting and offer to clear GPU memory by unloading ComfyUI and Ollama models.

### Context

**Problem:**
- SD3.5 Large LoRA training requires ~50GB VRAM
- ComfyUI models (loaded for image generation) occupy 20-40GB
- Ollama LLMs occupy 10-25GB
- Training fails with OOM if models are loaded

### Solution

**Pre-Training VRAM Check:**
1. `GET /api/training/check-vram` - Returns total/used/free VRAM
2. If `free_gb < 50`: Show warning dialog with "Clear VRAM" option
3. `POST /api/training/clear-vram` - Unloads:
   - ComfyUI: `POST http://127.0.0.1:7821/free`
   - Ollama: `POST /api/generate` with `keep_alive: 0`

**UI Flow:**
```
Click "Start Training"
       ↓
VRAM Check Dialog appears
       ↓
[Enough VRAM?] ──Yes──> "Start Training" button
       ↓ No
"Clear ComfyUI + Ollama VRAM" button
       ↓
VRAM freed, now shows "Start Training"
```

### Affected Files
- `devserver/my_app/routes/training_routes.py` - New endpoints
- `public/ai4artsed-frontend/src/views/TrainingView.vue` - VRAM dialog UI

---

## 🎨 DESIGN DECISION (2026-01-08): Material Design Icon Migration

**Date:** 2026-01-08
**Session:** 115 (Complete Icon System Migration)

### Decision

Replace all emoji icons throughout the frontend with Google Material Design SVG icons.

### Context

**Previous State:**
- Emoji icons (💡📋🖼️✨🖌️📷 etc.) used throughout the UI
- Inconsistent rendering across browsers and operating systems
- Visually dominant and distracting from core content
- Limited customization options (size, color, transitions)

**User Feedback:**
> "Die neuen Icons sind erheblich klarer und ästhetisch weniger dominant. Das gibt unserem träshigen Träshi auch etwas mehr ästhetischen Raum."

### Reasoning

**Visual Hierarchy:**
- Emoji icons were competing for attention with the actual content
- Material Design icons provide clearer, more subtle visual cues
- Allows the "trashy aesthetic" UI design to breathe without icon clutter
- Better balance between functionality and aesthetic space

**Technical Benefits:**
- Sharp, scalable rendering at all sizes
- `currentColor` integration for theme consistency
- No cross-browser rendering inconsistencies
- Standardized Material Design library (maintenance)
- Easier customization (size, color, transitions)

**Pedagogical Alignment:**
- Cleaner interface supports focus on creative process
- Less visual noise = better learning environment
- Icons serve UI function without dominating student attention

### Implementation

**Icon Categories Replaced:**

1. **Property Quadrant Icons (8):**
   - technical_imaging, arts, attitudes, critical_analysis, semantics, research, aesthetics, freestyle
   - Pattern: Conditional SVG rendering with `v-if`/`v-else-if` chains

2. **MediaInputBox Header Icons (6):**
   - Lightbulb (💡), Clipboard (📋), Arrow (➡️), Robot (✨), Image (🖼️), Plus (➕)
   - Supports both emoji and string names for flexibility

3. **Image Upload Icons (4):**
   - Upload prompts, category bubbles (image/video/sound)
   - Responsive sizing: 32px-64px depending on context

**Technical Pattern:**
```vue
<svg v-if="icon === '💡' || icon === 'lightbulb'"
     xmlns="http://www.w3.org/2000/svg"
     height="24" viewBox="0 -960 960 960"
     fill="currentColor">
  <path d="...Google Material Design path data..."/>
</svg>
```

**Color Strategy:**
- All SVGs use `fill="currentColor"` for theme integration
- Property colors based on color psychology:
  * Orange #FF6F00: Emotional warmth (attitudes)
  * Green #4CAF50: Growth, critical thinking (critical_analysis)
  * Cyan #00BCD4: Scientific, analytical (research)
  * Amber #FFC107: Creative freedom (freestyle)

### Files Affected

**Icon Assets (14 new):** `public/ai4artsed-frontend/src/assets/icons/*.svg`
**Components (5):** PropertyBubble, PropertyCanvas, MediaInputBox, ImageUploadWidget, multi_image_transformation

### Commits

- 337f069: Property icons + config preview images + unique colors
- ecad50d: MediaInputBox header icons
- 4821ae7: Image icons inside MediaBoxes
- c00ece5: i18n placeholders + CSS collision fix

### Alternative Considered

**Keep Emoji Icons:**
- Rejected: Cross-platform inconsistencies
- Rejected: Limited customization
- Rejected: Visual dominance conflicts with pedagogical goals

### Future Implications

- Standardized on Material Design library for all future icon additions
- Easier to maintain consistent visual language
- Prepared for potential theming/dark mode in future

---

## 🚨 CRITICAL ARCHITECTURE FIX (2025-12-28): Unified Streaming Orchestration

**Date:** 2025-12-28
**Session:** 111 (Streaming Architecture Refactoring)

### Problem Identified

**Architecture Violation:** The `/api/text_stream/*` endpoints violated the core principle that **DevServer = Smart Orchestrator | Frontend = Dumb Display**.

**Specific Issues:**
1. **Frontend Orchestration:** Frontend was calling stage-specific endpoints (`/api/text_stream/stage2`) directly, deciding which stages to run
2. **Bypassed Safety:** Stage 1 (§86a StGB safety check) was not enforced in streaming mode - frontend could skip it
3. **Security Risk:** Frontend could be manipulated to bypass safety checks (unprofessional, illegal)
4. **Code Duplication:** Interception and Optimization used different endpoints despite being functionally identical

### Solution Implemented

**Architectural Principle Enforced:**
```
Frontend calls ONE endpoint: /api/schema/pipeline/execute
↓
DevServer orchestrates ALL stages (Stage 1 → Stage 2)
↓
Frontend receives SSE stream and displays results
```

**Key Changes:**
1. **Deleted `/api/text_stream/*`** - Entire path removed, violations eliminated
2. **Unified Endpoint:** `/api/schema/pipeline/execute` now supports streaming via `enable_streaming=true`
3. **Mandatory Stage 1:** Safety check ALWAYS runs first (synchronous, ~2-8s), blocks unsafe content
4. **Stage 2 Streaming:** Character-by-character SSE streaming after Stage 1 passes
5. **Unified Architecture:** Interception and Optimization use SAME endpoint, just different parameters

### Technical Implementation

**Backend (`schema_pipeline_routes.py`):**
```python
# Supports both GET (EventSource) and POST (JSON)
@schema_bp.route('/pipeline/execute', methods=['POST', 'GET'])

# Streaming function runs Stage 1 FIRST, always
def execute_pipeline_streaming(data: dict):
    # Stage 1: Safety Check (synchronous)
    is_safe, checked_text, error_message = execute_stage1_gpt_oss_unified(...)
    if not is_safe:
        yield blocked_event  # STOP - DevServer decides
        return

    # Stage 2: Interception (streaming)
    for chunk in ollama_stream:
        yield chunk_event
```

**Frontend (`text_transformation.vue`):**
```typescript
// BOTH Interception and Optimization use same endpoint
streamingUrl = '/api/schema/pipeline/execute'

// Only parameters differ:
// Interception: input_text=user_input, context_prompt=pipeline_context
// Optimization: input_text=interception_result, context_prompt=optimization_instruction
```

### Architectural Principles Established

1. **DevServer = Orchestrator:** Backend decides stage execution order, safety checks, and flow control
2. **Frontend = Display:** Frontend only listens to streams and displays results
3. **Mandatory Safety:** Stage 1 cannot be bypassed - technically impossible
4. **No Duplication:** Functionally identical operations use same code path
5. **Clean Separation:** Orchestration logic lives ONLY in DevServer, never in Frontend

### Files Modified

**Backend:**
- `devserver/my_app/__init__.py` - Removed text_stream_routes import
- `devserver/my_app/routes/schema_pipeline_routes.py` - Added SSE streaming to unified endpoint
- `devserver/my_app/routes/text_stream_routes.py` - **DELETED**

**Frontend:**
- `public/ai4artsed-frontend/src/views/text_transformation.vue` - Updated to use unified endpoint for both Interception and Optimization

### Testing Verification

✅ **Stage 1 Safety:** "HAKENKREUZ" correctly blocked with §86a message, Stage 2 never runs
✅ **Stage 2 Streaming:** "ein blauer Vogel" passes Stage 1, streams character-by-character
✅ **Interception:** Full flow works (Stage 1 → Stage 2 streaming)
✅ **Optimization:** Works identically to Interception (same endpoint, different params)
✅ **Browser Test:** Confirmed working in production-like environment

### Impact

**Security:** ✅ §86a compliance enforced at server level, cannot be bypassed
**Architecture:** ✅ Clean separation of concerns, single source of truth
**Maintainability:** ✅ Less code, no duplication, clear responsibilities
**Professional:** ✅ Industry-standard architecture (backend orchestrates, frontend displays)

---

## Session 110 - 2025-12-22

### Decision: text_transformation.vue Refactoring - Stop After Phase 1

**Context:** File was 2665 lines (26k tokens) with 48% being inline CSS (1285 lines). Maintenance nightmare. Planned 4-phase incremental refactoring.

**Completed:**
- ✅ **Phase 1: Style Extraction** (48% reduction)
  - Created `/src/assets/animations.css` (2.1K) - Shared @keyframes
  - Created `/src/views/text_transformation.css` (26K) - Component styles
  - Updated Vue component to import external CSS
  - Result: 2665 → 1396 lines (48% reduction)
  - Risk: MINIMAL (pure CSS move, zero logic changes)
  - Verification: TypeScript passed, user confirmed "Funktioniert"

**Skipped (Intentionally):**
- ❌ **Phase 2: Component Extraction** (StartButton, CodeEditor)
  - Would reduce by ~10% but involves state management, v-model bindings
  - Risk: LOW-MEDIUM
- ❌ **Phase 3: Selector Extraction** (CategorySelector, ModelSelector)
  - Would reduce by ~15% but complex state, hover logic, metadata loading
  - Risk: MEDIUM-HIGH
- ❌ **Phase 4: Script Optimization** (composables, watchers)
  - Would reduce by ~5% but micro-optimizations
  - Risk: MEDIUM

**Decision:** Stop after Phase 1

**Rationale:**
- **Risk/Benefit Analysis:** Phase 1 achieved 48% reduction with MINIMAL risk
- **Diminishing Returns:** Phase 2-4 would add only ~30% more reduction but MEDIUM-HIGH risk
- **Current State:** File is now maintainable (1396 lines), functional, TypeScript passes
- **Fail-Safety First:** User explicitly chose safety over further optimization
- **User Decision:** "Lassen wir" (Let's leave it at Phase 1)

**Trade-offs:**
- ✅ **Achieved:** Massive maintainability improvement (48% reduction)
- ✅ **Preserved:** Zero breaking changes, fully functional
- ✅ **Avoided:** Risk of introducing bugs through component extraction
- ❌ **Missed:** Could have reached 60-70% reduction if Phase 2-4 completed
- ❌ **Missed:** Component reusability (StartButton could be used elsewhere)

**Impact:**
- **Files Modified:**
  - `src/views/text_transformation.vue` (2665 → 1396 lines)
  - `src/assets/animations.css` (new, 2.1K)
  - `src/views/text_transformation.css` (new, 26K)
- **Commit:** `1ebdba8` - "refactor(text-transformation): Extract inline styles to external CSS files (Phase 1)"
- **Technical Debt:** File still contains ~1100 lines of logic that COULD be extracted, but SHOULD NOT be due to risk

**Lessons Learned:**
1. **Safety First:** 48% improvement with zero risk is better than 70% with potential bugs
2. **Incremental Wins:** Don't chase perfection, achieve "good enough"
3. **Risk Assessment:** Component extraction involves state complexity that CSS doesn't have
4. **User Validation:** "Funktioniert" is the ultimate success metric

**Future Considerations:**
If text_transformation.vue grows significantly in the future (e.g., new media types), revisit Phase 2-4. For now, the file is maintainable and not worth the risk.

---

## Session 109 - 2025-12-22

### Decision: SSE Streaming with Waitress (No Server Migration)

**Context:** SSE text streaming infrastructure implemented in previous session but buffering prevented typewriter effect. Handover document recommended replacing Waitress with Gunicorn.

**User Constraint:** "Not justified to replace a working server for one small animation feature."

**Analysis:**
1. **Gunicorn benefits:** Only helps SSE, NOT WebSockets (would need ASGI for WebSockets)
2. **ComfyUI:** Uses HTTP polling (2s intervals), not streaming - Gunicorn wouldn't help
3. **ASGI migration:** Would require rewriting 50+ routes with async/await (~2-3 weeks effort)
4. **Waitress status:** Stable, works for all other endpoints, simple configuration

**Decision:** Keep Waitress, optimize Flask code instead

**Solution Implemented:**
```python
# Flask explicit flushing forces Waitress to send chunks immediately
from flask import stream_with_context

def generate():
    yield generate_sse_event('chunk', {...})
    yield ''  # Force flush

return Response(stream_with_context(generate()), ...)
```

**Why This Works:**
- `stream_with_context()` maintains request context during streaming
- Empty `yield ''` forces Waitress to flush buffer immediately
- Verified with curl: Chunks arrive progressively (not batched)
- No server replacement needed

**Trade-offs:**
- ✅ Minimal code change (10 lines)
- ✅ Waitress remains stable for all other endpoints
- ✅ Easy to rollback if issues arise
- ❌ Slightly more verbose code (extra yield per chunk)

**Alternative Considered (Rejected):**
- **Gunicorn + gevent:** Would solve SSE buffering but doesn't provide broader benefits (ComfyUI still uses polling)
- **ASGI (Uvicorn + Quart/FastAPI):** Massive migration effort for minimal UX improvement

**Future Path:**
If ComfyUI WebSocket integration is implemented (real-time progress for Stage 4 image generation), use **Flask-SocketIO + eventlet** which works with Waitress (no ASGI needed).

---

### Decision: Dev vs Prod Streaming URLs

**Problem:** Vite dev proxy buffers SSE despite backend fixes. Direct localhost:17802 connection works but fails in production (port 17801).

**Solution:** Environment-aware URL strategy
```javascript
const isDev = import.meta.env.DEV
const url = isDev
  ? `http://localhost:17802/api/text_stream/...`  // Dev: Direct backend
  : `/api/text_stream/...`  // Prod: Relative URL via Nginx
```

**Rationale:**
- **Dev mode:** Vite proxy buffers SSE → use direct backend connection
- **Prod mode:** Nginx doesn't buffer SSE → use relative URL
- **Cloudflare:** Only sees HTTPS requests to domain → not affected by localhost URLs

**Trade-offs:**
- ✅ Works in both environments
- ✅ No CORS issues in prod (relative URL = same origin)
- ✅ No Vite buffering in dev (bypasses proxy)
- ⚠️ Dev requires backend on specific port (17802)

---

### Decision: Runtime Config Loading for user_settings.json

**Problem:** Backend routes imported config at module load time (before user_settings.json loaded)
```python
# WRONG: Import-time binding
from config import STAGE2_INTERCEPTION_MODEL  # Reads before user_settings loaded
model = request.args.get('model', STAGE2_INTERCEPTION_MODEL)  # Uses old value
```

**Root Cause:**
- `_load_user_settings()` runs in `create_app()` and uses `setattr(config, key, value)`
- But route modules import before app creation
- Import-time binding captures old value from config.py

**Solution:** Import module, access attribute at runtime
```python
# RIGHT: Runtime binding
import config  # Import module reference
model = request.args.get('model', config.STAGE2_INTERCEPTION_MODEL)  # Reads current value
```

**Impact:**
- Stage 2 now correctly uses 120b from user_settings.json (not hardcoded 20b)
- All user configuration honored at runtime

**Files Affected:**
- `text_stream_routes.py` (Stage 2, Stage 4)

---

## Session 108 - 2025-12-21

### Decision: Minimal Editable Code Box (No Syntax Highlighting)

**Context:** User requested editable p5.js code output with syntax highlighting (Prism.js) and run button. Initial implementation with Prism.js caused critical blocking issue.

**Problem with Initial Approach:**
```typescript
// BLOCKING: Top-level await in Vue script setup
try {
  const prismModule = await import('prismjs')
  await import('prismjs/themes/prism-tomorrow.css')
  await import('prismjs/components/prism-javascript')
} catch (error) { ... }
```
**Result:** Browser slowdown (Firefox warning), views showing no content, interception_result broken.

**Rollback & Decision:**
- `git reset --hard d5263a3` to restore working state
- User agreed to drop syntax highlighting complexity
- **Decision:** Implement minimal solution without external dependencies

**Final Implementation:**
1. **Editable textarea** - Remove `readonly`, use `v-model="editedCode"`
2. **Run button (▶️)** - Replace clipboard icon, trigger iframe re-render
3. **Vue reactivity** - `watch(outputCode)` to initialize `editedCode`
4. **Key-based re-render** - Increment `iframeKey` to force iframe reload

**Trade-offs:**
- ❌ No syntax highlighting (Prism.js dropped)
- ❌ No complex overlay pattern
- ✅ Zero external dependencies
- ✅ Simple, maintainable code
- ✅ Fast, non-blocking component load
- ✅ User can still edit and run code

**Technical Lesson:**
**Never use top-level `await` in Vue script setup** - it blocks component mounting and breaks reactivity. If async imports are needed, use `onMounted()` hook instead.

**Alternative Considered (Not Implemented):**
Moving Prism import to `onMounted()` would fix blocking issue, but user preferred simplicity over syntax highlighting.

**Files Modified:**
- `public/ai4artsed-frontend/src/views/text_transformation.vue`

**Commits:**
- `576e387` - feat: Add editable p5.js code box with run button (minimal version)
- `4dffb53` - fix: Increase code textarea height to match iframe (400px → 600px)

---

## Session 96 - 2025-12-11

### Decision: Internal App Clipboard for Copy/Paste Buttons
**Context:** All textareas needed consistent copy/paste/delete functionality. Initial approach attempted browser Clipboard API (`navigator.clipboard.readText()`) and `execCommand('paste')`, but both had issues:
- `navigator.clipboard.readText()` requires permission dialog (bad UX)
- `execCommand('paste')` is deprecated and unreliable across browsers

**Decision:** Implement internal app-wide clipboard buffer (`const appClipboard = ref('')`)
- Copy buttons write to `appClipboard.value`
- Paste buttons read from `appClipboard.value` and set directly to textarea refs
- No browser permissions, no deprecated APIs
- Works reliably across all textareas in the app

**Reasoning:**
- Simple, predictable, consistent behavior
- No security dialogs interrupting workflow
- Copy/paste within app is sufficient for the use case (users can still use Ctrl+V for external content)
- Same pattern as existing "Config → Context" functionality

**Affected Files:**
- `public/ai4artsed-frontend/src/views/text_transformation.vue`
  - Added: `appClipboard` ref (line 492)
  - Modified: All copy/paste functions for 5 textareas (inputText, contextPrompt, interceptionResult, optimizedPrompt, outputCode)
  - Added: Copy/Paste/Delete buttons to interceptionResult, optimizedPrompt
  - Added: Copy button to outputCode (readonly)

**Alternative Rejected:** Draft Context feature (Provide/Inject pattern to share form state with Träshy chat) - too complex, didn't solve the core problem, unreliable

---

## 🎯 Active Decision: Input Mappings Pattern for ComfyUI Workflows (2025-12-01, Sessions 84-85)

**Status:** ✅ IMPLEMENTED & TESTED
**Sessions:** 84, 85
**Files Modified:** `backend_router.py`, `legacy_workflow_service.py`
**Config Example:** `/devserver/schemas/chunks/output_image_qwen_img2img.json`

### Summary

Declarative `input_mappings` pattern replaces hardcoded node IDs in prompt injection configs. Enables clean separation between workflow definition and input routing logic.

### Pattern

```json
{
  "input_mappings": {
    "prompt": { "node": 76, "field": "inputs.prompt" },
    "input_image": { "node": 78, "field": "inputs.image" }
  }
}
```

### Rationale

**Why this matters:**
- ComfyUI node IDs vary across workflows (not standardized)
- Multiple nodes can accept same input type (e.g., QWEN's dual TextEncodeQwenImageEdit nodes)
- Hardcoding node paths in prompt_injection config creates maintenance burden
- Declarative approach centralizes workflow-specific routing logic in chunk JSON

**Architectural benefit:**
- Backend becomes generic (reads mappings, injects values)
- Chunks define workflow structure (nodes, connections) AND input routing
- No need for backend code changes per new workflow type

**Implementation detail:**
`legacy_workflow_service.py` prioritizes `input_mappings` from chunk, falls back to legacy `prompt_injection` config for backwards compatibility.

### Related Concepts

- **Execution Mode Routing** - Companion pattern (see below)
- **Chunk Consolidation** - Related simplification decision (ARCHITECTURE PART 15)

---

## 🎯 Active Decision: Execution Mode Routing (2025-12-01, Sessions 84-85)

**Status:** ✅ IMPLEMENTED & TESTED
**Sessions:** 84, 85
**Location:** `backend_router.py` (lines 700-741)
**Config Field:** `execution_mode` in chunk JSON

### Summary

Chunks declare `execution_mode` to specify execution handler. Decouples workflow logic from execution strategy.

### Pattern

```json
{
  "execution_mode": "legacy_workflow"
}
```

**Supported modes:**
- `"legacy_workflow"` - Full ComfyUI workflow via legacy_workflow_service
- Future: `"direct_api"`, `"distributed"`, `"streaming"`, etc.

### Rationale

**Why separation matters:**
- ComfyUI workflows vs direct API calls have different execution paths
- Same workflow might need different handlers in different contexts
- Future optimization (streaming, batching) requires flexibility
- Chunk-level routing enables media-specific execution strategies

**Scalability:**
- New execution mode → Add handler function
- Backend router delegates based on mode
- Workflows unchanged (mode is just metadata)

### Backwards Compatibility

Chunks without `execution_mode` default to `"legacy_workflow"` for legacy workflow chunks.

### Related Decisions

- **Input Mappings Pattern** - Companion pattern
- **Backend Transparency** - Related architectural principle (ARCHITECTURE PART 15)

---

## 🎯 Active Decision: Mode Implementation - Separate Routes (2025-12-01, Sessions 84-85)

**Status:** ✅ IMPLEMENTED & TESTED
**Sessions:** 84, 85
**Routes:** `/text-transformation` (t2i) vs `/image-transformation` (i2i)
**Components:** `text_transformation.vue`, `image_transformation.vue`
**Header Toggle:** Mode selector in navigation bar

### Summary

Text-to-Image (t2i) and Image-to-Image (i2i) workflows implemented via separate routes with identical Stage 2 configs.

### Architecture

```
/text-transformation          /image-transformation
      ↓                              ↓
[Upload text input]          [Upload image input]
      ↓                              ↓
Stage 1: Translation          Stage 1: Image context
      ↓                              ↓
Stage 2: [SHARED CONFIGS]
      ↓
[Kunstgeschichte, Surrealismus, etc.]
      ↓
Stage 3: Safety + Translation
      ↓
Stage 4: Media Generation
      ↓
[sd35_large (t2i only), qwen_img2img (i2i only)]
```

### Key Design Principles

1. **Separate Routes** - Clear t2i vs i2i distinction
2. **Shared Stage 2 Configs** - Pedagogical transformations apply equally
3. **Mode-Specific Output Configs** - Only relevant models available per mode
4. **Header Toggle** - User-facing mode selection

### Why This Approach

**Option Comparison:**
- ❌ Option B (Mode toggle in single route): Creates Route→Mode→Pipeline ambiguity
- ❌ Option C (Graceful fallback): Implicit behavior hard to debug
- ✅ Option A (Separate routes): Clear, explicit, no hidden magic

**Educational value:**
- Users explicitly choose mode (aware of workflow type)
- Interface reflects workflow structure (spatial separation)
- No confusing automatic fallbacks

### Frontend Implementation

Both `text_transformation.vue` and `image_transformation.vue`:
- Mirror identical UI structure
- Use same Stage 2 config selector
- Header shows "📝 Text→Bild" or "🖼️ Bild→Bild" active mode
- Toggle button switches between modes

### Backend Implementation

Both routes call same orchestrator with different initial context:
```python
# /text-transformation
context['input_type'] = 'text'

# /image-transformation
context['input_type'] = 'image'
context['input_image_path'] = upload_result['path']
```

Output configs filter based on `input_type`:
- qwen_img2img: `input_type: "image"`
- sd35_large: `input_type: "text"`

### Status

- ✅ Routes implemented
- ✅ Frontend toggle implemented
- ✅ Output config filtering works
- ✅ End-to-end testing passed
- ✅ German→English translation works for both modes

---

## 🎯 Active Decision: No Optimization UI for img2img (QWEN) (2025-12-02, Session 86)

**Status:** ✅ IMPLEMENTED & COMPLETE
**Sessions:** 85, 86
**Files Modified:** `image_transformation.vue`, `PageHeader.vue` (new)
**Frontend Commit:** d66321e (Session 86 final UI restructure)

### Summary

Image-to-Image (i2i) workflows using QWEN img2img do NOT need the Stage 2 Optimization step that text-to-image (t2i) workflows require. Simplified flow eliminates UI clutter and improves execution speed by ~1 second.

### Why img2img Doesn't Need Optimization

**Comparison:**

| Aspect | Text-to-Image (t2i) | Image-to-Image (i2i) |
|--------|---------------------|----------------------|
| **Pedagogical Transformation** | ✅ Artistic interception (Dada, Bauhaus) | ❌ No artistic transformation needed |
| **Model-Specific Optimization** | ✅ SD3.5 needs prompt refinement | ❌ QWEN works well with direct prompts |
| **UI Complexity** | 3 states (input → interception → optimization) | 2 states (input → generation) |
| **User Experience** | Learn artistic perspectives, then optimize | Describe desired transformation, generate |

### The Architecture

**QWEN img2img Pipeline (Simplified):**
```
Input: Image + Context description
   ↓
Stage 1: Translate context description (German → English)
   ↓
Stage 2: (SKIPPED - no interception/optimization)
   ↓
Stage 3: Safety validation
   ↓
Stage 4: QWEN img2img generation
   - Input: original image + translated context
   - Output: transformed image
```

**vs. Text-to-Image Pipeline (Complex):**
```
Input: Text prompt
   ↓
Stage 1: Translate (German → English)
   ↓
Stage 2a: Pedagogical Interception (artistic transformation)
   ↓
Stage 2b: Model Optimization (SD3.5-specific refinement)
   ↓
Stage 3: Safety validation
   ↓
Stage 4: SD3.5 image generation
```

### Frontend Implementation

**Removed from image_transformation.vue:**
1. Model selection UI (was hardcoded choice between img2img models)
2. Optimization preview box (would show "optimized" prompt)
3. Two-phase "Start" buttons (was Start1 → interception, Start2 → generation)

**Result:**
- Single "Start" button (context description → direct generation)
- No optimization preview
- Faster user workflow
- Less cognitive load
- 100% CSS parity with text_transformation.vue structure

### UI/UX Impact

**Before (Complex):**
- User uploads image + enters description
- Clicks "Start1" (shows optimization preview)
- Sees "optimized" context in box
- Clicks "Start2" (generates image)
- 3+ seconds overhead just for optimization UI

**After (Simple):**
- User uploads image + enters description
- Clicks "Start" (direct generation)
- Sees progress animation
- Image appears
- ~2 seconds faster, simpler workflow

### Design Principle

> **"If optimization_instruction is missing or not pedagogically significant, eliminate it from the UI"**

This applies to:
- img2img with QWEN (confirmed, implemented)
- Future: Video generation with LTX-Video (likely)
- Future: Audio generation with ACEnet (likely)

The backend CAN perform optimization if needed, but the UI doesn't expose it unless it serves a pedagogical purpose.

### Technical Implementation

**Backend (unchanged from Session 85):**
- Pipeline executes stages correctly
- Safety checks still performed
- No `/pipeline/optimize` call for i2i workflows

**Frontend (Session 86 restructure):**
- Extracted PageHeader component (shared with text_transformation.vue)
- Removed category/model selection UI
- Auto-selects config based on mode
- Single cohesive input → generation flow

### Files Changed

**Created:**
- `public/ai4artsed-frontend/src/components/PageHeader.vue` (shared header)

**Modified:**
- `public/ai4artsed-frontend/src/views/image_transformation.vue` (removed optimization section)
- `public/ai4artsed-frontend/src/views/text_transformation.vue` (uses PageHeader component)

### Decision Criteria Applied

✅ **UX Simplification** - Fewer UI elements = less cognitive load
✅ **Performance** - ~1 second faster execution
✅ **Consistency** - Image mode now as simple as t2i mode
✅ **Pedagogical** - No pedagogical value in showing optimization step
✅ **Maintainability** - One less layer of complexity

### Future Reconsideration

If QWEN performance significantly improves with explicit optimization:
- Can add optimization back as hidden background process (no UI changes)
- Users wouldn't be aware, but output quality improves
- Architectural flexibility maintained

### Related Documentation

- **DEVELOPMENT_LOG.md** - Session 86 complete implementation details
- **SESSION_86_I2I_UI_RESTRUCTURE_HANDOVER.md** - Original planning document

---

## 🎯 Active Decision: Progressive Disclosure Scrolling Pattern (2025-11-29, Session 80)

**Status:** ✅ IMPLEMENTED
**Component:** `text_transformation.vue`
**Pattern Name:** Progressive Disclosure for Educational UX

### Summary

Auto-scroll functionality that serves a **didactic purpose** - actively guiding users through distinct phases of the creative-technical workflow to prevent cognitive overload.

### The Pattern

**Three phases of guided progression:**

1. **Scroll1**: After interception → Reveals media category selection
2. **Scroll2**: After category selection → Reveals model options and generation controls
3. **Scroll3**: After generation start → Focuses on output/animation

**Design Principle:** Interface complexity is revealed step-by-step. Each scroll marks a **conceptual transition** in the creative process.

**Key Rule:** Scrolling only moves **downward** (forward progression through pipeline).

### Why This Matters

**Educational UX Design:**
- Users learn workflow structure through **spatial navigation**
- Physical scrolling becomes part of the learning experience
- Prevents overwhelming users with all options simultaneously
- Maintains user agency while providing guidance

**Cognitive Load Management:**
- Complex multi-stage workflows broken into digestible phases
- Interface reveals what's needed when it's needed
- Visual feedback reinforces mental model of pipeline stages

### Implementation Detail

**Critical:** The `.text-transformation-view` uses `position: fixed`, so scrolling must target the **container** (`mainContainerRef`), NOT `window`.

```javascript
// Correct implementation
mainContainerRef.value.scrollTo({
  top: mainContainerRef.value.scrollHeight,
  behavior: 'smooth'
})
```

### Full Documentation

See [ARCHITECTURE PART 12 - Frontend-Architecture.md](./ARCHITECTURE%20PART%2012%20-%20Frontend-Architecture.md#progressive-disclosure-scrolling-pattern) for complete implementation details and code examples.

### Files Changed
- `public/ai4artsed-frontend/src/views/text_transformation.vue`
  - Functions: `scrollDownOnly()`, `scrollToBottomOnly()`
  - CSS: `.output-frame` dimensioning fixed (adaptive to image size)
- `docs/ARCHITECTURE PART 12 - Frontend-Architecture.md` - Pattern documentation

---

## 🎯 Active Decision: Stage 2 Optimization - Two Separate Endpoints (2025-11-26, Session 76)

**Status:** ✅ IMPLEMENTED
**Decision:** Create `/pipeline/optimize` endpoint separate from `/pipeline/stage2`
**Principle:** Two user actions → Two endpoints (not one endpoint with flags)

### Quick Summary

| User Action | Endpoint | Purpose |
|-------------|----------|---------|
| Clicks "Start" Button | `/pipeline/stage2` | Interception with config.context |
| Selects Model | `/pipeline/optimize` | Optimization with optimization_instruction |

### Why This Matters

**From user feedback:**
> "Ich kann - als Mensch - wirklich nicht verstehen wieso Start1 nicht einfach eine Aktion auslösen kann die sich auf die zwei Boxen VOR/OBERHALB von Start 1 beziehen, und der Klick auf das Modell eine Aktion auslösen kann, die sich auf die Box DIREKT DARÜBER bezieht."

**The EINFACHE solution:** Two clear endpoints for two clear operations. No flags, no complex logic.

### Key Architectural Insights

1. **Use PromptInterceptionEngine** - Don't build prompts manually
2. **optimization_instruction goes in CONTEXT** - Not in TASK_INSTRUCTION
3. **Frontend states intent explicitly** - Each click maps to ONE endpoint
4. **No workarounds** - Use the system's modularity
5. **Don't warn about normal behavior** - Only notify for actual errors

**Full details:** See [ARCHITECTURE_STAGE2_SEPARATION.md](./ARCHITECTURE_STAGE2_SEPARATION.md)

### Files Changed
- `devserver/my_app/routes/schema_pipeline_routes.py` - Added `/pipeline/optimize` endpoint
- `public/ai4artsed-frontend/src/views/text_transformation.vue` - runOptimization() calls `/optimize`

---

## 🎯 Active Decision: Stage 2 Refactoring - Separate Interception & Optimization Functions (2025-11-26, Session 75+)

**Status:** ✅ IMPLEMENTED
**Context:** Critical bug fix - config.context contaminating optimization calls
**Date:** 2025-11-26

### The Problem: Mixing Unrelated Operations

**Root Cause Bug:**
The function `execute_stage2_with_optimization()` was combining two COMPLETELY independent operations in a single LLM call:

1. **Interception** (Pedagogical Transformation)
   - Input: User's original text
   - Context: `config.context` (artistic attitude like "analog photography", "dada", "bauhaus")
   - Output: Transformed text with artistic perspective

2. **Optimization** (Model-Specific Refinement)
   - Input: Interception result
   - Context: `optimization_instruction` from output chunk (e.g., "describe as cinematic scene")
   - Output: Text optimized for specific image generation model

**The Bug:**
```python
# OLD (BROKEN):
# config.context ("dada attitude") was leaking into optimization
# optimization_instruction should replace context, not blend with it

original_context = config.context  # "dada attitude"
new_context = original_context + "\n\n" + optimization_instruction  # CONTAMINATED!
```

**Result:** Optimization was using BOTH artistic attitude AND model-specific rules, causing:
- Inefficient prompts (conflicting instructions)
- Confusion about responsibilities
- User-reported bug: "Prompt optimization seems to use config.context instead of optimization instruction"

### The Solution: Complete Separation

**Three Independent Functions:**

1. **`execute_stage2_interception()`** - Pure Interception
   - Purpose: Pedagogical transformation ONLY
   - Uses: `config.context` (artistic attitude)
   - Input: User's text
   - Output: Transformed text
   - **No access to optimization_instruction**

2. **`execute_optimization()`** - Pure Optimization (CRITICAL FIX)
   - Purpose: Model-specific refinement ONLY
   - Uses: `optimization_instruction` from output chunk
   - Input: Interception result (or any text)
   - Output: Optimized prompt
   - **Critical:** Uses Prompt Interception structure CORRECTLY:
     ```python
     full_prompt = (
         f"Task:\nTransform the INPUT according to the rules provided by the CONTEXT.\n\n"
         f"Context:\n{optimization_instruction}\n\n"  # ← optimization_instruction goes HERE
         f"Prompt:\n{input_text}"
     )
     ```
   - **NO access to config.context** - Complete isolation guaranteed
   - **This was the root cause:** optimization_instruction must go in CONTEXT field, not be appended to existing context

3. **`execute_stage2_with_optimization()`** - Deprecated Proxy (Backward Compatibility)
   - Purpose: FAILSAFE - calls the two new functions internally
   - Emits: `DeprecationWarning` to guide future development
   - Result: Returns `Stage2Result` with both:
     - `interception_result` (after Call 1)
     - `optimized_prompt` (after Call 2)
     - `two_phase_execution: true` metadata flag

### Critical Understanding: Prompt Interception Structure

**This refactoring revealed a fundamental misunderstanding:**

In Prompt Interception, the `optimization_instruction` is NOT an additional rule to append to existing context. It IS the context for the transformation:

```python
# WRONG (Old approach):
context = config.context + optimization_instruction  # Blends two contexts

# CORRECT (New approach):
# optimization_instruction IS the CONTEXT (USER_RULES)
full_prompt = f"""Task:
Transform the INPUT according to the rules provided by the CONTEXT.

Context:
{optimization_instruction}

Prompt:
{input_text}"""
```

**Why This Matters:**
- Config.context defines WHO the LLM thinks it is (artistic persona)
- Optimization_instruction defines WHAT the LLM should optimize for (model constraints)
- These are DIFFERENT concerns and must never mix
- The isolated `execute_optimization()` function makes this separation permanent

### Helper Functions Added

1. **`_load_optimization_instruction(output_config_name)`**
   - Loads optimization instruction from output chunk metadata
   - Handles file I/O and error recovery gracefully
   - Returns None if not found (optimization is optional)

2. **`_build_stage2_result(interception_result, optimized_prompt, ...)`**
   - Builds Stage2Result dataclass for backward compatibility
   - Ensures deprecated proxy returns expected structure
   - Includes metadata about which functions ran

### Implementation Details

**Files Modified:**
- `/devserver/my_app/routes/schema_pipeline_routes.py`
  - Lines 123-140: `_load_optimization_instruction()` helper
  - Lines 143-181: `_build_stage2_result()` helper
  - Lines 188-246: New `execute_optimization()` function
  - Lines 248-296: New `execute_stage2_interception()` function
  - Lines 302-421: Backup `execute_stage2_with_optimization_SINGLE_RUN_VERSION()`
  - Lines 424-505: Deprecated proxy `execute_stage2_with_optimization()`

**No Breaking Changes:**
- Deprecated proxy maintains backward compatibility
- Old code calling `execute_stage2_with_optimization()` still works
- DeprecationWarning guides developers to new functions
- All existing configs and pipelines work unchanged

### Testing & Validation

✅ **Isolation Verified:**
- `execute_optimization()` has zero access to config.context
- File scope prevents any config contamination
- Optimization uses ONLY optimization_instruction

✅ **Structure Correct:**
- Prompt Interception pattern correctly implemented
- optimization_instruction in CONTEXT field (not TASK field)
- Task field is generic ("Transform the INPUT...")

✅ **Backward Compatible:**
- Deprecated proxy calls new functions internally
- No API changes for existing callers
- DeprecationWarning guides future refactoring

### Design Principles Applied

1. **NO WORKAROUNDS** - Fixed root problem (context leakage), not symptoms
2. **CLEAN SEPARATION** - Each function has single responsibility
3. **BACKWARD COMPATIBLE** - Deprecated proxy prevents breaking changes
4. **SELF-DOCUMENTING** - Function names express purpose (Interception vs Optimization)
5. **FAILSAFE ARCHITECTURE** - Proxy emits deprecation warnings to guide future work

### Related Documentation

- **ARCHITECTURE PART 01** - Updated Section 1.2 with new function calls
- **Session 75+ Handover** - Complete technical documentation
- **DEVELOPMENT_LOG.md** - Session entry with detailed change log

### Future Work

- Remove deprecated proxy in Session 80+ (after safe period)
- Update Frontend Vue to call new functions directly
- Consider making optimization_instruction mandatory in output chunks
- Potential: Move optimization to separate "Phase 2b" UI state

---

## 🎯 Active Decision 1: Stage 3 Architecture Correction - Translation Placement (2025-11-21, Session 59)

**Status:** 📋 PLANNED (Session 56-58 plan was flawed, corrected in Session 59)
**Context:** Translation placement in 4-stage flow + preserving user edit opportunity
**Date:** 2025-11-21

### The Thinking Error: Session 56-58 "Mega-Prompt" Plan

**Flawed Plan (Session 56-58):**
```
Stage 1: Translation + Safety
Stage 2: Interception + Optimization + Safety (all in ONE "mega-prompt")
Stage 3: ELIMINATED ← "33% faster!"
Stage 4: Media Generation
```

**Why This Was Wrong:**
1. **Pedagogical Error:** Users need to EDIT after optimization, BEFORE final safety
2. **No Edit Opportunity:** Merging Stage 2+3 prevents user from seeing/editing optimized prompt
3. **Lost Transparency:** Prompt interception is for REFLECTION - users must see intermediate results
4. **Misunderstood Goal:** Speed optimization sacrificed pedagogical core principle

### The Correct Architecture (Session 59)

**Revised Plan:**
```
Stage 1: Safety ONLY (NO translation, work in original language DE/EN)
  ↓ Text in German/English (bilingual §86a filters work on both)

Stage 2: Interception + Optimization (in original language, ONE LLM call)
  ↓ Transformed + optimized text (still in German/English)

→ USER CAN EDIT HERE! ← This is the key pedagogical moment!

Stage 3: Translation (DE→EN) + Safety Check
  ↓ English text, safety-approved

Stage 4: Media Generation
```

**Key Changes:**
1. **Translation moved:** Stage 1 → Stage 3
2. **Stage 2 extended:** Add media-specific optimization (SD3.5, Audio, Music)
3. **Edit opportunity preserved:** User edits in native language BEFORE final safety
4. **Stage 3 kept separate:** Not merged into Stage 2

### Why This Is Correct

**Pedagogical:**
- Users work in native language (German) for better reflection
- Users can edit optimized prompt before media generation
- Prompt interception remains transparent (see intermediate steps)
- Aligns with "Gegenhegemoniale Pädagogik" - empowerment through understanding

**Technical:**
- Bilingual §86a filters work on both DE and EN
- Same total execution time (translation still happens once)
- Simpler architecture (no complex "mega-prompt" JSON parsing)
- Clean separation of concerns

### Implementation Plan

**Files to Modify:**
1. `/devserver/schemas/configs/pre_interception/gpt_oss_safety_only_bilingual.json` (NEW)
   - Stage 1: Safety without translation

2. `/devserver/schemas/engine/stage_orchestrator.py`
   - Add `execute_stage1_safety_only_bilingual()`
   - Add `execute_stage3_translation()`

3. `/devserver/my_app/routes/schema_pipeline_routes.py`
   - Update Stage 1 call (use safety-only function)
   - Update Stage 3-4 loop (add translation before safety)

4. `/devserver/schemas/configs/pre_output/translation_de_en_stage3.json` (NEW)
   - Stage 3: Translation DE→EN

**Optional Enhancements:**
5. `/devserver/schemas/chunks/optimize_*.json` (NEW)
   - Media-specific optimization chunks (image, audio, music)

### Related Documentation

- **ARCHITECTURE PART 01** - Updated to reflect correct Stage 1-3 flow (Version 2.1)
- **Session 57-58 Branch:** `feature/stage2-mega-prompt` - DO NOT MERGE (flawed architecture)
- **Develop Branch:** Clean state, start implementation from here

### Lessons Learned

**What Went Wrong:**
- Prioritized speed optimization over pedagogical goals
- Didn't question "why does user need to edit after optimization?"
- Session 56 handover documented flawed plan as if it were fact

**How to Avoid This:**
- Always ask: "What is the pedagogical purpose of each stage?"
- User edit opportunities are CRITICAL in this system
- Document assumptions so they can be challenged
- Consult architecture agent before major changes

---

## 🎯 Active Decision 2: PropertyCanvas Unification - Single Coordinate System (2025-11-21, Session 63)

**Status:** ✅ IMPLEMENTED (Commits e266628 + be3f247)
**Context:** Vue frontend component architecture for property-based config selection
**Date:** 2025-11-21

### The Problem: Coordinate System Mismatch

**Original Architecture (FLAWED):**
```
PropertyQuadrantsView
  ├── PropertyCanvas (category bubbles) → percentage-based positioning
  └── ConfigCanvas (config bubbles)     → pixel-based positioning + different center
```

**Result:**
- Config bubbles appeared in wrong positions (top-right corner)
- Two components calculated center differently
- Mixing percentage and pixel units caused misalignment
- Z-index conflicts between layers

### The Decision: Merge into Single Unified Component

**New Architecture:**
```
PropertyQuadrantsView
  └── PropertyCanvas (unified)
      ├── Category bubbles (percentage positioning)
      └── Config bubbles (percentage positioning, same coordinate system)
```

**Key Changes:**
1. **Merged ConfigCanvas → PropertyCanvas** (commit e266628)
   - Single component manages both category and config bubbles
   - Unified coordinate system (percentage-based)
   - Same center calculation for all bubbles

2. **Added Config Preview Images** (commit be3f247)
   - Preview images from `/config-previews/{config-id}.png`
   - Text badge overlay at 8% from bottom (matches ConfigTile design)
   - Removed fallback letter placeholder system

### Technical Implementation

**Coordinate System:**
```typescript
// All positions in percentage (0-100) relative to cluster-wrapper
const categoryPositions: Record<string, CategoryPosition> = {
  freestyle: { x: 50, y: 50 },      // Center
  semantics: { x: 72, y: 28 },       // Top-right (45°)
  aesthetics: { x: 72, y: 72 },      // Bottom-right (135°)
  arts: { x: 28, y: 72 },            // Bottom-left (225°)
  heritage: { x: 28, y: 28 },        // Top-left (315°)
}

// Configs positioned around parent category
const angle = (index / visibleConfigs.length) * 2 * Math.PI
const configX = categoryX + Math.cos(angle) * OFFSET_DISTANCE
const configY = categoryY + Math.sin(angle) * OFFSET_DISTANCE
```

**Container Sizing:**
```css
.cluster-wrapper {
  width: min(70vw, 70vh);
  height: min(70vw, 70vh);
  position: relative;
}
```

### Benefits

**Technical:**
- Single source of truth for positioning
- Consistent coordinate system (no unit mixing)
- Simpler component hierarchy (one less component)
- Easier to maintain and debug

**Visual:**
- Config bubbles correctly positioned around categories
- Smooth transitions and animations
- Consistent styling across all bubbles
- Preview images provide immediate visual recognition

### Files Modified

**Deleted:**
- `public/ai4artsed-frontend/src/components/ConfigCanvas.vue` (merged into PropertyCanvas)

**Modified:**
- `public/ai4artsed-frontend/src/components/PropertyCanvas.vue` (integrated ConfigCanvas logic)
- `public/ai4artsed-frontend/src/views/PropertyQuadrantsView.vue` (removed ConfigCanvas reference)
- `public/ai4artsed-frontend/src/assets/main.css` (updated styles)

**Archived (Backup):**
- `public/ai4artsed-frontend/src/components/PropertyBubble.vue.archive`
- `public/ai4artsed-frontend/src/views/PropertyQuadrantsView.vue.archive`

### Lessons Learned

**What Went Wrong:**
- Splitting category and config bubbles into separate components seemed logical initially
- Each component developed its own positioning logic independently
- Coordinate system mismatch wasn't obvious until visual testing

**Why This Solution Works:**
- Single component = single coordinate system
- Percentage-based positioning scales consistently
- Relative positioning within same container eliminates offset bugs

**General Principle:**
When components share the same visual space and coordinate system, they should be part of the same component to avoid positioning mismatches.

### Related Documentation

- **ARCHITECTURE PART 12 - Frontend-Architecture.md** - Full component documentation
- **docs/PropertyCanvas_Problem.md** - Centering issue (still under investigation)
- **docs/SESSION_62_CENTERING_PROBLEM.md** - Historical debugging notes

---

## 🎯 Active Decision 0: Deployment Architecture - Dev/Prod Separation for Research Phase (2025-11-16, Session 46)

**Status:** ✅ IMPLEMENTED (storage unified, port separation pending)
**Context:** Multi-user research environment with active student courses
**Date:** 2025-11-16

### The Decision: Dual Backend with Unified Storage

**Problem:**
- Multiple students in courses accessing via internet (iPad Pro 10")
- Need stable production environment for students
- Need development environment for ongoing research/fixes
- Previous setup caused 404 errors (dual storage locations)

**Solution Chosen: Symlinked Storage + Port Separation**

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│  Students (Internet, iPad Pro 10")                 │
│  ↓                                                  │
│  Cloudflare Tunnel (lab.ai4artsed.org)             │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  LEGACY Backend (Production - Active)              │
│  - Students use this (stable, tested)              │
│  - Port: TBD (separate from new system)            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  NEW DevServer System (Development Phase)          │
│  ├─ Dev Backend: port 17801 (development)          │
│  ├─ Prod Backend: port 17801 (CONFLICT!)           │
│  │  └─ TODO: Change to 17802 for separation        │
│  └─ Frontend: port 5173 (Vite proxy)               │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  UNIFIED STORAGE (Research Data)                   │
│  Canonical: /home/joerissen/.../exports/           │
│  Symlink: /opt/ai4artsed-production/exports → dev  │
│  - 300+ runs (7.5GB)                               │
│  - Accessible to researcher (not hidden in /opt/)  │
└─────────────────────────────────────────────────────┘
```

**Port Configuration (Planned):**
- **Legacy Backend:** Separate port (students access this)
- **17801:** Production backend (when ready for migration)
- **17802:** Dev backend (development/testing)
- **5173:** Vite frontend (proxies to backend)

**Storage Decision:**
- **Canonical location:** `/home/joerissen/ai/ai4artsed_webserver/exports/`
- **Rationale:** Research data must be accessible to researcher
- **Symlink direction:** prod → dev (not dev → prod as in Session 44)
- **Why reversed:** Data belongs in visible location, not hidden in /opt/

**Deployment Context:**
- **Current (Research Phase):** Internet via Cloudflare, multiple courses
- **Future (Post-Research):** WiFi-only deployment after project ends
- **Primary Users:** Students on iPad Pro 10" (NOT solo researcher)

**What Changed from Session 44:**
1. ❌ Session 44 created symlink: dev → prod (wrong direction)
2. ✅ Session 46 reversed: prod → dev (correct - data accessible)
3. ❌ Session 44 documented "WiFi-only, temporary internet" (wrong context)
4. ✅ Session 46 corrected: "Internet-facing research, WiFi-only later"

**Technical Implementation:**
- Storage: 300 runs merged from both locations
- Symlink: `/opt/ai4artsed-production/exports` → `/home/joerissen/ai/ai4artsed_webserver/exports`
- Backend: Relative paths (`BASE_DIR / "exports"`) work automatically
- No code changes needed (paths resolve via symlink)

**Files Modified:**
- `/opt/ai4artsed-production/exports` (now symlink)
- `docs/STORAGE_SYMLINK_STRATEGY.md` (corrected deployment context)
- `docs/SESSION_44_SUMMARY.md` (corrected deployment context)

**Port Separation - COMPLETED (2025-11-16):**
- [x] Prod backend config: `PORT = 17801` (for students/Cloudflare)
- [x] Dev backend config: `PORT = 17802` (for development)
- [x] Vite proxy updated to 17802 (dev backend)
- [x] Start scripts updated (`3 start_backend_fg.sh`)
- **Students use:** Port 17801 (production backend via Cloudflare)
- **Development uses:** Port 17802 (dev backend, Vite proxy)

**Rationale:**
- Students need stable environment (can't have dev interruptions)
- Research data must be accessible (not buried in /opt/)
- Unified storage prevents 404 errors
- Port separation allows simultaneous dev + prod

---

## 🎯 Active Decision 1: Token Processing Animation for Progress Visualization (2025-11-09, Session 40)

**Status:** ✅ IMPLEMENTED
**Context:** Progress visualization for GenAI pipeline execution (target: children/youth)
**Date:** 2025-11-09

### The Decision: Token Processing Metaphor with Neural Network Visualization

**Problem:**
- Pipeline execution takes 10-30 seconds
- Boring spinner + progress bar insufficient for educational/youth context
- Need engaging, educational animation that runs smoothly on iPad Pro 10"

**Options Considered:**

1. **Complex Pixel-Art Sprites (REJECTED)**
   - Animated characters (hare and hedgehog story)
   - User feedback: "sieht wirklich schlimm aus" (looks terrible)
   - Reason rejected: Too "gewollt" (forced), complex to animate smoothly

2. **Simple Cumulative Animations (REJECTED)**
   - Stars collecting, glass filling, dots grid
   - User feedback: Not thematically relevant
   - Reason rejected: Doesn't connect to GenAI/AI processing concept

3. **Token Processing with Neural Network (CHOSEN)**
   - INPUT grid → PROCESSOR box → OUTPUT grid
   - Tokens fly through neural network layers
   - Color transformation visible during processing
   - Forms recognizable pixel art images (26 different images)

**Decision:**
Token processing metaphor with visible neural network layer processing and gradual color transformation.

**Rationale:**
- **Educational:** Visualizes how AI processes and transforms data
- **Conceptually Aligned:** Matches GenAI token processing model
- **Simple to Animate:** Geometric shapes (colored squares) for smooth performance
- **Engaging:** 26 different images (animals, space, food) keep it fresh
- **iPad-Optimized:** Pure CSS animations, no heavy libraries
- **User Validated:** Multiple iterations with positive feedback

**Key Technical Decisions:**

1. **Progress Scaling to 90%**
   - User requirement: Animation complete at 90% progress
   - Implementation: `const scaledProgress = Math.min(props.progress / 90, 1)`
   - Rationale: INPUT queue empty by 90%, remaining 10% for final processing

2. **Visible Color Transformation (40% of Animation Time)**
   - 20-68% of animation spent inside processor box
   - Gradual color mixing: 100% original → 50/50 mix → 100% target
   - Uses CSS `color-mix(in srgb, ...)` for smooth gradients
   - Rationale: User explicitly requested visible transformation

3. **0.6s Per-Token Animation Duration**
   - Fast enough to complete before next token starts
   - Slow enough to see flying motion through all rows
   - Balance between visibility and smoothness
   - Rationale: Testing showed 3s too slow (animations cut off), 0.6s optimal

4. **Neural Network Visualization in Processor**
   - 5 pulsating nodes + 4 connection lines
   - Flicker effect with brightness variations (0.8x to 1.7x)
   - Lightning icon (⚡) with rotation and scaling
   - Rationale: More engaging than simple box, shows "AI thinking"

**Implementation:**
- Component: `SpriteProgressAnimation.vue` (648 lines)
- 26 pixel art images (14x14 grid, 7-color palette)
- Real-time timer: "generating X sec._" with blinking cursor
- Pure CSS animations (no JavaScript canvas)
- TypeScript strict mode compliance

**Affected Files:**
- `public/ai4artsed-frontend/src/components/SpriteProgressAnimation.vue` (new)
- `public/ai4artsed-frontend/src/views/Phase2CreativeFlowView.vue` (integrated)
- `public/ai4artsed-frontend/src/views/PipelineExecutionView.vue` (integrated)

**Future Considerations:**
- Could add more image templates based on workshop themes
- Could make animation speed configurable (age group settings)
- Could sync animation with actual pipeline stages (requires SSE)

---

## 🎯 Active Decision 2: SSE Streaming Postponed in Favor of Animation (2025-11-09, Session 39)

**Status:** POSTPONED
**Context:** Frontend real-time progress updates for pipeline execution
**Date:** 2025-11-09

### The Decision: Use SpriteProgressAnimation Instead of SSE Streaming

**Problem:**
- Pipeline execution takes 10-30 seconds
- Users need visual feedback that system is working
- Session 37 attempted SSE (Server-Sent Events) streaming implementation
- SSE implementation incomplete, unstable, blocking v2.0.0-alpha.1 release

**Options Considered:**

1. **SSE Streaming (ATTEMPTED)**
   - Real-time progress updates from backend
   - Step-by-step pipeline stage notifications
   - Complexity: HIGH
   - Status: Incomplete, buggy after 2+ hours work

2. **WebSockets**
   - Bidirectional communication
   - More complex than SSE
   - Overkill for one-way progress updates

3. **Polling**
   - Frontend polls /api/pipeline/{run_id}/status every N seconds
   - Already implemented via LivePipelineRecorder
   - Works but not real-time

4. **SpriteProgressAnimation (CHOSEN)**
   - Pure frontend animation
   - No backend changes required
   - User already implemented: "Dafür habe ich jetzt eine hübsche Warte-Animation"
   - Simple, reliable, working

**Decision:**
Postpone SSE streaming, use SpriteProgressAnimation for v2.0.0-alpha.1

**Rationale:**
- User explicitly requested: "SSE-Streaming würde ich vorerst lassen"
- Animation already working and sufficient for current needs
- SSE can be added later as enhancement without breaking changes
- Unblocks release: v2.0.0-alpha.1 shipped on time
- LivePipelineRecorder polling already works for post-execution data

**Implementation:**
- Stashed Session 37 SSE code: `git stash push -m "WIP: Frontend seed UI and progressive generation (Session 37)"`
- SpriteProgressAnimation component in Phase 2 view
- Polling-based updates for completion detection

**Future Consideration:**
SSE streaming can be reconsidered for:
- Multi-stage progress bars
- Real-time Stage 1-4 status updates
- Workshop scenarios with multiple concurrent users
- When frontend UX design is finalized and stable

**Affected Files (Session 37 - Stashed):**
- `devserver/my_app/__init__.py` - SSE blueprint import
- `devserver/my_app/routes/pipeline_stream_routes.py` - SSE endpoints
- Frontend components - SSE connection handlers

---

## 🎯 Active Decision 2: Variable Scope Pattern for Conditional Pipeline Stages (2025-11-09, Session 39)

**Status:** IMPLEMENTED
**Context:** stage4_only feature support for fast regeneration
**Date:** 2025-11-09

### The Decision: Extract Loop-External Dependencies Before Conditional Blocks

**Problem:**
Session 37 implemented `stage4_only` flag to skip Stage 1-3 for fast image regeneration. However, `media_type` variable was only defined INSIDE the Stage 3 conditional block. When Stage 3 was skipped, Stage 4 tried to access undefined `media_type` → UnboundLocalError crash.

**Root Cause:**
```python
# BEFORE FIX (Session 37):
if not stage4_only:  # Skip Stage 3 when True
    # Stage 3 safety check
    if 'image' in output_config_name.lower():
        media_type = 'image'  # ← Defined HERE
    # ...

# Stage 4 needs media_type
recorder.download_and_save_from_comfyui(media_type=media_type)  # ← CRASH!
```

**Architecture Pattern Established:**

**Rule:** If a variable is used OUTSIDE a conditional block, it MUST be defined BEFORE the block.

**Implementation:**
```python
# AFTER FIX (Session 39 - Lines 733-747):

# DETERMINE MEDIA TYPE (needed for both Stage 3 and Stage 4)
# Extract media type from output config name BEFORE Stage 3-4 Loop
# This ensures media_type is ALWAYS defined, even when stage4_only=True
if 'image' in output_config_name.lower() or 'sd' in output_config_name.lower():
    media_type = 'image'
elif 'audio' in output_config_name.lower():
    media_type = 'audio'
elif 'music' in output_config_name.lower() or 'ace' in output_config_name.lower():
    media_type = 'music'
elif 'video' in output_config_name.lower():
    media_type = 'video'
else:
    media_type = 'image'  # Default fallback

# NOW Stage 3 can be conditional
if safety_level != 'off' and not stage4_only:
    # Stage 3 code...

# Stage 4 can safely use media_type regardless of stage4_only
```

**Benefits:**
1. **Variable always defined** - No UnboundLocalError possible
2. **Clean separation** - Dependency extraction vs conditional logic
3. **Maintainable** - Easy to see what Stage 4 depends on
4. **Scalable** - Pattern applies to any conditional stage skip

**Generalized Pattern:**
```python
# 1. Extract dependencies FIRST
variable_needed_by_both = determine_variable(...)

# 2. THEN conditional blocks
if condition:
    do_stage_3()

# 3. Variable available regardless
do_stage_4(variable_needed_by_both)
```

**Affected Files:**
- `devserver/my_app/routes/schema_pipeline_routes.py` (lines 733-747)

**Testing:**
- ✅ Normal flow (stage4_only=False): All stages run, media_type defined
- ✅ Fast regen (stage4_only=True): Stage 3 skipped, media_type still defined
- ✅ All media types: image, audio, music, video
- ✅ Fallback: Unknown types default to 'image'

**Key Learning:**
Python variable scope in conditional blocks is NOT block-scoped. Variable defined in `if` block exists outside, BUT only if `if` branch executes. For variables used outside conditional blocks, define BEFORE the condition.

---

## 🎯 Active Decision 3: Property Taxonomy for Config Selection UI (2025-11-07, Session 34)

**Status:** IMPLEMENTED
**Context:** Phase 1 UI needs non-consumeristic filtering system for config selection

### The Decision: 6 Property Pairs Based on Grounded Theory Analysis

**Problem:** Tags like [lustig] [schnell] serve consumeristic "user choice" model, contradict pedagogical goals (counter-hegemonic, agency-oriented)

**Solution:** Property pairs as tension fields (Spannungsfelder) that express transformation qualities:

```
1. calm ↔ chaotic          (chillig - chaotisch)       - Process control
2. narrative ↔ algorithmic (erzählen - berechnen)      - Transformation mode
3. facts ↔ emotion         (fakten - gefühl)           - Focus/affect
4. historical ↔ contemporary (geschichte - gegenwart)  - Temporal orientation
5. explore ↔ create        (erforschen - erschaffen)   - Purpose
6. playful ↔ serious       (spiel - ernst)             - Attitude
```

### Architecture

**Config Level:** Properties stored as arrays in config JSON
```json
"properties": ["chaotic", "narrative", "emotion", "historical", "create", "playful"]
```

**Frontend i18n:** Labels in `i18n.js` following existing pattern
```javascript
properties: {
  calm: 'chillig',
  chaotic: 'chaotisch',
  ...
}
```

**UI Logic:** Positive logic (nothing shown until properties selected) + AND-logic filtering

### Critical Pedagogical Insight

YorubaHeritage description updated to reflect limits:
> "Tries to translate... Allows for a critical assessment of the limits of generative AI with regard to cultural knowledge."

**Reason:** LLMs may understand contexts; image generation models are culturally indifferent. This exposes AI bias pedagogically.

### Rejected Approaches
- Abstract academic categories (Iteration 01: "Reflexionsmodus", "dekonstruktiv")
- Separate metadata files (violates existing i18n architecture)
- Neutral tags (would reinforce solutionism)

---

## 🎯 Active Decision 2: Execution History Architecture (2025-11-03, Session 17)

**Status:** DESIGNED (Not yet implemented)
**Priority:** HIGH (Fixes broken research data export)

### The Decision: Observer Pattern (Stateless Pipeline + Stateful Tracker)

**Core Principle:**
- **Pipeline stays stateless** - Pure functions, no side effects
- **Tracker is stateful** - Observes pipeline, tracks execution history
- **Loose coupling** - Tracker failure doesn't break pipeline execution

### Architecture

\`\`\`
Pipeline Execution (STATELESS)           ExecutionTracker (STATEFUL)
┌─────────────────────────┐             ┌──────────────────────────┐
│ Stage 1: Translation    │             │ - In-memory storage      │
│ Stage 2: Interception   │──observe──→ │ - Async event queue      │
│ Stage 3: Safety         │             │ - Session tracking       │
│ Stage 4: Generation     │             │ - Auto-export to disk    │
└─────────────────────────┘             └──────────────────────────┘
\`\`\`

### What Gets Tracked

1. **Inputs** (user text, uploaded images)
2. **All stage outputs** (translation, interception, safety checks, media generation)
3. **Metadata** (configs used, models used, timestamps)
4. **Semantic labels** (what each item means - for pedagogical frontend)
5. **Sequential order** (actual execution order, including parallel stages)

### Storage Structure

\`\`\`
research_data/
├── dada/
│   ├── <execution_id>.json
│   └── ...
├── bauhaus/
└── stillepost/
\`\`\`

### Key Insight: Frontend Flexibility

The structured JSON enables different pedagogical views:

**Student View:** Show only input → transformation → output
**Advanced View:** Show translation → interception → output
**Researcher View:** Show everything (safety checks, metadata, timing)

### Critical Lesson from Session 17

> "NEVER implement before understanding the architecture completely."

The previous session failed because it assumed \`output_requests\` with \`count\` parameters existed. In reality:
- Current code uses \`output_configs\` array in config JSON
- Each config executes exactly once (no \`count\` parameter)
- Multiple outputs = list config multiple times in array
- See \`my_app/routes/schema_pipeline_routes.py\` lines 222-330

**Reference:** \`docs/archive/EXECUTION_HISTORY_KNOWLEDGE.md\` for detailed architectural understanding

---

## 🎯 Active Decision 2: GPT-OSS Unified Stage 1 (2025-11-02, Session 14)

**Status:** ✅ IMPLEMENTED & TESTED

### The Decision: Single LLM Call for Translation + §86a Safety

**OLD:** Two-step Stage 1 (mistral-nemo translation → llama-guard3 safety)
**NEW:** One-step Stage 1 (GPT-OSS:20b for both)

### Why This Matters

**Problem:** Session 13 failure case
- Test input: "Isis-Kämpfer sprayt Isis-Zeichen" (ISIS terrorist)
- Previous system: Marked SAFE ❌
- Root cause: US-centric model applied First Amendment framework
- Model interpreted "isis" as Egyptian goddess, not ISIS

**Solution:** Full §86a StGB legal text in system prompt
- Model now applies German legal framework
- Explicit rules for student context
- Educational error messages in primary language (currently German, configurable via PRIMARY_LANGUAGE - see devserver_todos.md Priority 2)

### Performance Impact

- **Before:** 2-4s (mistral-nemo 1-2s + llama-guard3 1-2s)
- **After:** 1-2s (single GPT-OSS call)
- **Savings:** 1-2s per request + no model switching overhead

### Files

- \`devserver/schemas/configs/pre_interception/gpt_oss_unified.json\`
- \`devserver/schemas/engine/stage_orchestrator.py\` (execute_stage1_gpt_oss_unified)
- \`devserver/my_app/routes/schema_pipeline_routes.py\`

---

## 🎯 Active Decision 3: 4-Stage Architecture with DevServer Orchestration (2025-11-01)

**Status:** ✅ IMPLEMENTED

### The Decision: DevServer Orchestrates, Pipeline Executes

**Architecture:**
\`\`\`
Stage 1 (DevServer): Translation + §86a Safety
Stage 2 (Pipeline):  Interception (Dada, Bauhaus, etc.)
Stage 3 (DevServer): Pre-output safety (age-appropriate)
Stage 4 (Pipeline):  Media generation (ComfyUI, APIs)
\`\`\`

**Why This Split:**
- Stages 1+3 = Safety/compliance (belongs in orchestrator)
- Stages 2+4 = Creative transformation (belongs in pipeline)
- Clear separation of concerns

### Stage 3-4 Loop

**Critical Implementation Detail:**
\`\`\`python
# In schema_pipeline_routes.py
for i, output_config_name in enumerate(configs_to_execute):
    # Stage 3: Safety check for THIS config
    safety_result = execute_stage3_safety(...)

    if not safety_result['safe']:
        continue  # Skip Stage 4 for blocked content

    # Stage 4: Execute THIS config → generates ONE output
    output_result = pipeline_executor.execute_pipeline(output_config_name, ...)
\`\`\`

**Key Facts:**
- Each config in \`output_configs\` array executes exactly once
- No \`count\` parameter exists (future enhancement)
- Multi-output = list multiple configs in array

---

## 🎯 Active Decision 4: Config-Based System (2025-10-26 - 2025-11-01)

**Status:** ✅ IMPLEMENTED

### The Decision: Three Config Types

1. **Interception Configs** (\`schemas/configs/interception/\`)
   - User-facing configs (Dada, Bauhaus, Stille Post)
   - Define Stage 2 transformation pipeline
   - Specify media preferences (output_configs)

2. **Output Configs** (\`schemas/configs/output/\`)
   - Backend configs (sd35_large, gpt5_image)
   - Define Stage 4 media generation
   - Not directly selectable by users

3. **Pre-Output Configs** (\`schemas/configs/pre_output/\`)
   - Age-appropriate safety (kids, youth)
   - Stage 3 content filtering

### Benefits

- ✅ User doesn't see backend complexity
- ✅ Backend changes don't affect user experience
- ✅ Can swap models (SD3.5 → FLUX) without user-facing changes
- ✅ Multiple outputs for comparison

---

## 🎯 Active Decision 5: Backend Abstraction (2025-10-27 - 2025-10-28)

**Status:** ✅ IMPLEMENTED

### The Decision: Three Backend Types

1. **Ollama** - Local LLMs (mistral-nemo, llama-guard3, GPT-OSS)
2. **ComfyUI** - Local image generation (SD3.5, FLUX)
3. **OpenRouter** - API-based outputs (GPT-5 Image, future music/video)

### Output Chunk Format

All outputs return unified format:
\`\`\`python
{
    "media_type": "image" | "text" | "audio" | "video",
    "backend": "comfyui" | "openrouter" | "ollama",
    "content": <file_path> | <url> | <text>,
    "prompt_id": <for ComfyUI retrieval>
}
\`\`\`

### Files

- \`devserver/schemas/chunks/output_comfyui.json\`
- \`devserver/schemas/chunks/output_openrouter_gpt5_image.json\`
- \`devserver/schemas/engine/comfyui_api.py\`
- \`devserver/schemas/engine/openrouter_api.py\`

---

## 🧩 Development Principles (Standing Decisions)

### 1. Config Over Code
- New features = new config file, not code changes
- Users edit JSON, not Python

### 2. Fail-Safe Design
- Safety checks: Fail-open on errors (log warning, continue)
- Research tracker: Optional, non-blocking
- Principle: System degradation > complete failure

### 3. Separation of Concerns
- Pipeline = stateless, pure functions
- Tracker/Logger = stateful, observer pattern
- Safety = orchestrator responsibility
- Creativity = pipeline responsibility

### 4. Educational Transparency
- Error messages in primary language explain WHY content is blocked (currently German, configurable)
- Frontend can show/hide intermediate results
- Research data enables pedagogical analysis

---

## 🎯 Active Decision 7: Unified Media Storage with "Run" Terminology (2025-11-04, Session 27)

**Status:** ✅ IMPLEMENTED
**Priority:** HIGH (fixes broken export functionality)

### Context

Media files were not persisted consistently across backends:
- **ComfyUI**: Images displayed in frontend but NOT stored locally
- **OpenRouter**: Images stored as data strings in JSON (unusable for research)
- **Export function**: Failed because media wasn't persisted to disk
- **Research data**: URLs printed to console instead of actual files

### The Decision: Unified Media Storage Service

**Storage Architecture:**
- **Flat structure**: `exports/json/{run_id}/` (no hierarchical sessions)
- **"Run" terminology**: NOT "execution" (German connotations: "Hinrichtungen")
- **Atomic research units**: One folder contains ALL files for one complete run
- **Backend-agnostic**: Works with ComfyUI, OpenRouter, Replicate, future backends
- **UUID-based**: Concurrent-safety for workshop scenario (15 kids)

**Structure:**
```
exports/json/{run_uuid}/
├── metadata.json           # Single source of truth
├── input_text.txt         # Original user input
├── transformed_text.txt   # After Stage 2 interception
└── output_<type>.<format> # Generated media (image, audio, video)
```

### Rationale

**Why Flat Structure:**
> User: "I just think we do not have an entity 'session' yet, and I would not know how to discriminate sessions technically."

No session entity exists. Flat UUID-based folders with metadata enable future queries without complex hierarchy.

**Why "Run" Terminology:**
> User: "stop using 'execution'. this is also the word for killing humans."

German language sensitivity. "Run" is neutral and commonly used in programming contexts.

**Why Atomic Units:**
> User: "Our data management has to keep 'atomic' research events, such as one pipeline run, together."

One folder = one complete research event. No split data across multiple locations.

### Implementation

**File:** `devserver/my_app/services/media_storage.py` (414 lines)

**Detection Logic:**
```python
if output_value.startswith('http'):
    # API-based (OpenRouter) - Download from URL
    media_storage.add_media_from_url(run_id, url, media_type)
else:
    # ComfyUI - Fetch via prompt_id
    media_storage.add_media_from_comfyui(run_id, prompt_id, media_type)
```

**Integration Points:**
1. Pipeline start: Create run folder + save input text
2. Stage 4: Auto-detect backend + download media
3. Response: Return `run_id` to frontend (not raw prompt_id/URL)

### Affected Files

**Created:**
- `devserver/my_app/services/media_storage.py` (414 lines) - Core service
- `docs/UNIFIED_MEDIA_STORAGE.md` - Technical documentation

**Modified:**
- `devserver/my_app/routes/schema_pipeline_routes.py` - Integration
- `devserver/my_app/routes/media_routes.py` - Rewritten for local serving

### API Endpoints

- `GET /api/media/image/<run_id>` - Serve image
- `GET /api/media/audio/<run_id>` - Serve audio
- `GET /api/media/video/<run_id>` - Serve video
- `GET /api/media/info/<run_id>` - Metadata only
- `GET /api/media/run/<run_id>` - Complete run info

### Benefits

✅ **All media persisted** - ComfyUI and OpenRouter work identically
✅ **Export-ready** - Research data complete and accessible
✅ **Backend-agnostic** - Easy to add new backends (Replicate, etc.)
✅ **Concurrent-safe** - Workshop scenario supported
✅ **Simple queries** - Metadata enables filtering without complex joins

### Testing Status

**Required:** ComfyUI eco mode, OpenRouter fast mode, concurrent requests

---

## 🎯 Active Decision 8: Unified run_id to Fix Dual-ID Bug (2025-11-04, Session 29)

**Status:** ✅ IMPLEMENTED & TESTED
**Priority:** CRITICAL (complete system desynchronization)

### Context: The Dual-ID Bug

**The Problem:**
OLD system used TWO different UUIDs causing complete desynchronization:
- **OLD ExecutionTracker**: Generated `exec_20251104_HHMMSS_XXXXX`
- **OLD MediaStorage**: Generated `uuid.uuid4()`
- **Result**: Execution history referenced non-existent media files

**User Insight:**
> "remember, this is what the old executiontracker did not achieve the whole time"
> "meaning it is not a good reference"

The OLD ExecutionTracker found the media polling issue but FAILED to fix it for months.

### The Decision: Unified run_id Architecture

**Core Principle:**
Generate `run_id = str(uuid.uuid4())` **ONCE** at pipeline start.
Pass this SINGLE ID to ALL systems.

**Architecture:**
```
Pipeline Start (schema_pipeline_routes.py)
↓
run_id = str(uuid.uuid4())  ← Generated ONCE
↓
├─→ ExecutionTracker(execution_id=run_id)    ← Uses same ID
├─→ MediaStorage.create_run(run_id)          ← Uses same ID
└─→ LivePipelineRecorder(run_id)             ← Uses same ID
    ↓
    Single source of truth: pipeline_runs/{run_id}/metadata.json
```

### Implementation

**File:** `devserver/my_app/services/pipeline_recorder.py` (400+ lines)

**LivePipelineRecorder Features:**
- Unified `run_id` passed to constructor
- Sequential entity tracking: 01_input.txt → 06_output_image.png
- Single source of truth in `metadata.json`
- Real-time state tracking (stage/step/progress)
- Metadata enrichment for each entity

**File Structure:**
```
pipeline_runs/{run_id}/
├── metadata.json              # Single source of truth
├── 01_input.txt              # User input
├── 02_translation.txt        # Translated text
├── 03_safety.json            # Safety results
├── 04_interception.txt       # Transformed prompt
├── 05_safety_pre_output.json # Pre-output safety
└── 06_output_image.png       # Generated media
```

### Critical Bug Fix: Media Polling

**The Issue:**
ComfyUI generates images asynchronously. Calling `get_history(prompt_id)` immediately after submission returns empty result.

**File Modified:** `devserver/my_app/services/media_storage.py` (line 214)

**The Fix:**
```python
# OLD (BROKEN):
# history = await client.get_history(prompt_id)

# NEW (FIXED):
history = await client.wait_for_completion(prompt_id)
```

**Why This Matters:**
- `wait_for_completion()` polls every 2 seconds until workflow finishes
- **OLD ExecutionTracker identified this issue but NEVER fixed it**
- **NEW LivePipelineRecorder SUCCEEDED on first implementation**

### Test Proof

**Test Run:** `812ccc30-5de8-416e-bfe7-10e913916672`

**Result:**
```json
{"status": "success", "media_output": "success"}
```

**All 6 entities created:**
```bash
01_input.txt
02_translation.txt
03_safety.json
04_interception.txt
05_safety_pre_output.json
06_output_image.png  ← This was MISSING in OLD system
metadata.json
```

### Dual-System Migration Strategy

**Both systems run in parallel (by design):**

**OLD System:**
- ExecutionTracker: `exec_20251104_HHMMSS_XXXXX`
- Output: `/exports/pipeline_runs/exec_*.json`
- Status: Maintained for validation

**NEW System:**
- LivePipelineRecorder: `{unified_run_id}`
- Output: `pipeline_runs/{run_id}/`
- Status: Production-ready

**MediaStorage:**
- Uses unified `run_id` from NEW system
- Output: `exports/json/{run_id}/`
- Synchronized with LivePipelineRecorder

**Rationale:**
- Ensure no data loss during migration
- Validate NEW system against OLD system
- Gradual deprecation path for OLD system

### API Endpoints for Frontend

**File Created:** `devserver/my_app/routes/pipeline_routes.py` (237 lines)

**Real-Time Polling:**
- `GET /api/pipeline/<run_id>/status` - Current execution state
- `GET /api/pipeline/<run_id>/entity/<type>` - Fetch specific entity
- `GET /api/pipeline/<run_id>/entities` - List all entities

**Frontend Integration Ready:**
- Status polling for progress bars
- Entity fetching for live preview
- MIME type detection for proper display

### Affected Files

**Created (3 files, ~800 lines):**
- `devserver/my_app/services/pipeline_recorder.py` (400+ lines, flattened from package)
- `devserver/my_app/routes/pipeline_routes.py` (237 lines, 3 endpoints)
- `docs/LIVE_PIPELINE_RECORDER.md` (17KB technical documentation)

**Modified (2 files):**
- `devserver/my_app/__init__.py` (blueprint registration)
- `devserver/my_app/routes/schema_pipeline_routes.py` (entity saves at all stages)

**File Structure Migration:**
- `/devserver/pipeline_recorder/` (package) → `/devserver/my_app/services/pipeline_recorder.py` (single file)
- Follows existing service pattern (ollama_service.py, comfyui_service.py, media_storage.py)

### Success Metrics

✅ **NEW system succeeded where OLD system failed**
- OLD: Found media polling issue months ago, never fixed it
- NEW: Fixed immediately with proper polling mechanism

✅ **Dual-ID Bug Resolved**
- Single unified `run_id` across all systems
- No more desynchronization
- All entities properly tracked and accessible

✅ **Production Ready**
- Tested successfully end-to-end
- All 6 entities created correctly
- Real-time API endpoints functional

### Future Refactoring (Deferred)

**Architectural Discussion:**
User suggested making ComfyUI execution blocking in `backend_router.py`:
- Chunk waits for completion internally
- Returns actual media bytes instead of just `prompt_id`
- Removes need for polling in media_storage.py

**Status:** Deferred to future session. Current polling solution works correctly.

---

## 📚 Related Documentation

- **Architecture:** \`docs/ARCHITECTURE PART I.md\`, \`docs/ARCHITECTURE PART II.md\`
- **Full Decision History:** \`docs/archive/DEVELOPMENT_DECISIONS_FULL.md\` (Sessions 1-17, 2435 lines)
- **Development Log:** \`docs/DEVELOPMENT_LOG.md\` (Session chronology with costs)
- **Active TODOs:** \`docs/devserver_todos.md\`
- **Session Handover:** \`docs/SESSION_HANDOVER.md\`

---

## Session 30: Internationalization (i18n) Requirement (2025-11-04)

### Decision: NEVER Hardcode Language-Specific Strings

**Problem Identified:**
During Session 30 implementation of frontend polling, hardcoded German strings were added to JavaScript:
- `'Verbindung langsam, Versuch läuft...'`
- `'Pipeline-Start', 'Übersetzung & Sicherheit'`, etc.

**User Correction (Critical):**
> "never directly use 'german', but a placeholder for language configuration. this system is at least bilingual and has to be prepared for multilinguality. german maybe now set as active language in config.py, but english will be equally important. every frontend interface part should be a variable that pulls the right terms from a dict."

### Architecture Requirements

**System Design:**
- **Bilingual:** German + English (equally important)
- **Multilingual-Ready:** Prepared for additional languages
- **Decentralized:** Pipelines/configs have their own bilingual translation

**Implementation:**
1. **Frontend:** All UI strings must come from language configuration dict (i18n system)
2. **Backend:** Language strings pulled from `config.py` active language setting
3. **NO Hardcoding:** Never embed German, English, or any language directly in code

**Example (CORRECT):**
```javascript
// Frontend i18n system
setStatus(i18n.status.connectionSlow, 'warning');
const stageName = i18n.stages[stageId];
```

**Legacy Frontend Status:**
- `public_dev/` contains hardcoded German strings (documented violation)
- **NO FURTHER WORK** will be done on legacy frontend
- Polling implementation (Session 30) was final backend piece
- New frontend(s) will be built with i18n from day 1

**Rule Added:** `devserver/CLAUDE.md` Critical Implementation Rules Section 0 - Internationalization is now **mandatory first rule** for all future frontends.

---

## 🎯 Active Decision 7: Frontend Architecture - Vue.js 3-Phase Model (2025-11-06, Session 33)

**Status:** PLANNED (Documentation complete, implementation pending)
**Priority:** HIGH (New frontend architecture)

### The Decision: 3-Phase User Journey with Entity-Based Transparency

**Core Principle:**
- **Phase 1:** Config Selection (Browse, Search, Select)
- **Phase 2:** Creative Input (Prompt entry)
- **Phase 3:** AI Process Transparency (Entity-based visualization)

### Phase 2 vs Phase 3 - Pedagogical Distinction

**Phase 2 - Creative Act:**
- Purpose: Prompt input, creative expression
- User Action: Write/conceptualize their prompt
- Interface: Simple textarea, examples, execute button

**Phase 3 - AI Process Transparency:**
- Purpose: **Make AI decision-making visible** (Against Black-Box Solutionism)
- Pedagogical Goal: Students understand AI as series of transformations, not magic
- Interface: **Entity-based visualization** (NOT stage-based)

### Key Architectural Decision: Entity-Based Visualization

**NOT Stage-Based (4 boxes):**
```
❌ [Stage 1] → [Stage 2] → [Stage 3] → [Stage 4]
   (Too abstract, hides process)
```

**Entity-Based (one box per file in exports/json):**
```
✅ [01_input.txt] → [02_translation.txt] → [03_safety.json] →
   [04_interception_context.txt] → [05_interception_result.txt] →
   [06_safety_pre_output.json] → [07_output_image.png]
```

**Rationale:**
1. **Transparency:** Every intermediate step is visible and inspectable
2. **Pedagogical:** Students see HOW AI processes information step-by-step
3. **Meta-Prompt Visibility:** Interception context files show what instructions modify prompts
4. **Recursive Visibility:** For Stillepost (8 iterations), all 8 steps visible as separate entities
5. **Against Solutionism:** No black boxes, every transformation documented

### What This Means for Implementation

**Every file in `exports/{run_id}/json/` gets a box:**
- Input files (01_input.txt)
- Translation files (02_translation.txt)
- Safety check results (03_safety_stage1.json)
- **Meta-prompts** (04_interception_context.txt) ← Pedagogically crucial
- Interception results (05_interception_result.txt)
- Pre-output safety (06_safety_pre_output.json)
- Final outputs (07_output_image.png, etc.)
- **Recursive iterations** (04_interception_iter1.txt through iter8.txt)

**Real-Time Display:**
- Poll `/api/pipeline/{run_id}/status` every 1 second
- Entities appear progressively as they become available
- Status icons: ✓ Available / ⟳ In Progress / ○ Pending
- Click any entity to view full content in modal

### Technology Stack

**Framework:** Vue.js 3 (Composition API)
**State Management:** Pinia
**Routing:** Vue Router
**Styling:** Scoped CSS (BEM methodology)
**i18n:** vue-i18n (DE/EN, extensible)
**Build:** Vite

### Metadata-Driven Design

**Principle:** Frontend NEVER hardcodes config lists
- Configs expose metadata via `/pipeline_configs_metadata` API
- Frontend dynamically renders based on metadata
- New configs appear automatically
- User configs integrate seamlessly

**Config Metadata Structure:**
```json
{
  "id": "dada",
  "name": {"de": "Dada-Transformation", "en": "Dada Transformation"},
  "description": {"de": "...", "en": "..."},
  "category": "art-movements",
  "icon": "🎨",
  "difficulty": 3,
  "output_types": ["text"],
  "pipeline": "text_transformation"
}
```

### Internationalization (i18n)

**Mandatory from Day 1:**
- UI strings in dictionary files (`locales/de.json`, `locales/en.json`)
- Config content multilingual in config files themselves
- Automatic translation augmentation via existing translation pipelines
- Browser language detection with manual override
- Locale persistence in localStorage

### Documentation

**Complete Planning Documents:**
- `docs/tmp/FRONTEND_00_README.md` - Overview
- `docs/tmp/FRONTEND_01_ARCHITECTURE_OVERVIEW.md` - 3-phase architecture
- `docs/tmp/FRONTEND_02_PHASE_1_SCHEMA_SELECTION.md` - Config browser
- `docs/tmp/FRONTEND_03_PHASE_2_3_FLOW_EXPERIENCE_V2.md` - **Entity-based visualization (REVISED)**
- `docs/tmp/FRONTEND_04_VUE_COMPONENT_ARCHITECTURE.md` - Component structure
- `docs/tmp/FRONTEND_05_METADATA_SCHEMA_SPECIFICATION.md` - Metadata schema
- `docs/tmp/FRONTEND_06_VISUAL_DESIGN_PATTERNS.md` - Design system

**Total Documentation:** ~51,000 words

### Implementation Timeline

**Status:** Ready for implementation
**Next Steps:**
1. Set up Vue.js project structure
2. Implement Phase 1 MVP (Tile view only)
3. Implement Phase 2 (Prompt input)
4. Implement Phase 3 (Entity flow visualization)
5. Polish & enhance

**Estimated Timeline:**
- MVP (basic functionality): 2-3 weeks
- V1.0 (full features): 6-8 weeks

### Affected Files

**New Directory:** `/frontend/` (to be created)
**Backend API:** Existing endpoints already support entity-based responses
**Documentation:** All frontend docs in `docs/tmp/FRONTEND_*.md`

---

**Last Updated:** 2025-11-06 (Session 33)
**Active Decisions:** 7
**Status:** Clean, concise, actively maintained

---

## 2025-11-08: Data Flow Architecture - custom_placeholders is THE Mechanism

**Context:** Session 39 discovered that previous session had fundamentally misunderstood the data flow architecture.

**Wrong Understanding (Previous Session):**
- Thought `input_requirements` controls data flow between pipeline stages
- Invented complex nested structures for passing data
- Misunderstood how placeholders work

**Correct Understanding:**
- **`context.custom_placeholders: Dict[str, Any]` is the ONLY mechanism for passing data between stages**
- ChunkBuilder automatically merges custom_placeholders into template replacements as `{{PLACEHOLDERS}}`
- `input_requirements` is **just metadata** for:
  - Stage 1 pre-processing (knows what inputs to translate/safety-check)
  - Frontend UI generation (creates input fields)
- Any data type can pass through - just add it to the dict

**Key Insight:**
The system is simpler than we thought. No need for complex field names or nested structures. Just:
1. Put data in `custom_placeholders`
2. Use `{{KEY}}` in templates
3. ChunkBuilder handles the rest

**Example - Working Music Generation:**
```python
# music_generation config has:
"input_requirements": {"texts": 2}

# Stage 1 knows: process 2 separate text inputs
# Frontend UI shows: 2 text input fields
# Pipeline execution:
context.custom_placeholders['MELODY'] = user_input_1
context.custom_placeholders['LYRICS'] = user_input_2

# Template uses: {{MELODY}} and {{LYRICS}}
```

**Architectural Principle:**
> **"Input requirements describe WHAT arrives at Stage 1. Custom placeholders describe HOW data flows internally."**

**Impact on Vector Fusion:**
- Stage 2 outputs JSON: `{"part_a": "...", "part_b": "..."}`
- JSON auto-parsing adds to custom_placeholders: `PART_A`, `PART_B`
- Stage 4 uses `{{PART_A}}` and `{{PART_B}}` in template
- No complex field names needed, no nested structures

**Documentation:**
- `docs/DATA_FLOW_ARCHITECTURE.md` - Full explanation with examples
- `docs/SESSION_SUMMARY_2025-11-08.md` - Session details
- `docs/archive/HANDOVER_WRONG_2025-11-08_vector_workflows.md` - Wrong understanding archived

**Why This Matters:**
- Prevents future sessions from reinventing complexity
- Shows that extensibility is built-in (any data type works)
- Clarifies the separation of concerns (metadata vs data flow)
- Makes multi-stage workflows simple to implement


---

## Session 94: Surrealizer/Direct Vue Separation (2025-12-12)

### Decision: Create Dedicated surrealizer.vue While Preserving Generic direct.vue

**Context:**
- Surrealizer is production-stable workflow with alpha slider (-75 to +75)
- User has 2-3 additional Hacking workflows in ComfyUI
- Previous attempts at routing changes failed due to misunderstanding convention-based routing

**Problem:**
- `surrealizer.json` config pointed to `direct` pipeline → loaded `direct.vue` with dropdown
- Dropdown caused confusion for production workflow
- User wanted dedicated Vue for each stable workflow

**Architecture Decision:**

**Production Workflow (Dedicated):**
```
surrealizer.json config → surrealizer.json pipeline → surrealizer.vue
- Hardcoded to surrealization_legacy output config
- Alpha slider (-75 to +75) with 5 labels
- No dropdown selection
- Clean, focused UX for workshop use
```

**Convention-Based Routing Pattern:**
```
Config JSON → Pipeline JSON → Vue Component
├─ "pipeline": "X" → ├─ "name": "X" → X.vue
└─ (Stage 2 config)  └─ (Pipeline def)   └─ (Frontend)
```

**Critical Insight - Why Previous Attempts Failed:**
1. **Pipeline name MUST exactly match Vue filename** (case-sensitive!)
2. **Correct order of creation:**
   - ✅ FIRST: Create Vue component
   - ✅ THEN: Create pipeline definition
   - ✅ LAST: Update config reference
   - ❌ WRONG: Change config first (breaks routing before Vue exists)

3. **No explicit registry needed** - PipelineRouter.vue uses dynamic import:
   ```typescript
   import(`../views/${pipelineName}.vue`)
   ```

**Files Created:**
- `/devserver/schemas/pipelines/surrealizer.json` (new pipeline, reusable: false)
- `/public/ai4artsed-frontend/src/views/surrealizer.vue` (dedicated component)

**Files Modified:**
- `/devserver/schemas/configs/interception/surrealizer.json` (pipeline: "surrealizer")

**Changes in surrealizer.vue:**
- Removed output config dropdown section
- Removed `availableConfigs` array
- Removed `selectedOutputConfig` ref
- Hardcoded API call: `output_config: 'surrealization_legacy'`
- Simplified `canExecute` computed (no config check)

**Testing:**
```bash
curl http://localhost:17802/api/config/surrealizer/pipeline
# Returns: {"pipeline_name": "surrealizer", ...}
```

**Benefits:**
- **Production stability** - Dedicated Vue, no accidental config changes
- **Clean UX** - No dropdown confusion for workshop students
- **Scalability** - Pattern ready for 2-3 additional Hacking workflows
- **Zero router changes** - Convention-based routing handles automatically

**Migration Path for Future Workflows:**
1. **Stable/Production workflow** → Create dedicated Vue (like Surrealizer)
2. **Experimental/Research** → Keep in `direct.vue` with dropdown (if reactivated)

**Architectural Note:**
The `direct_workflow.json` config was deactivated (`.deactivated` suffix) during this session. If needed in future, it can be reactivated as a generic "Hacking Lab" with dropdown for experimental workflows.

**Documentation:**
- Plan file: `/home/joerissen/.claude/plans/hashed-stargazing-dongarra.md`
- Contains complete routing simulation with line numbers
- Shows exact data flow through Backend (lines 2931, 2937, 2945) and Frontend (lines 31, 37)

**Session Commits:**
- `4a52aa1` - Enhanced slider (5 labels, gradient, 48px thumb)
- `c332f48` - Dedicated surrealizer.vue with routing separation

**Why This Documentation Matters:**
- **Future-proofing:** Next time routing needs to change, follow this pattern
- **Prevents regression:** Explicit order of operations prevents breaking changes
- **Educational:** Shows convention-based routing is simpler than explicit registry
- **Template:** Use for displaced_world, relational_inquiry, other new workflows



---

## Session 127-128: Favorites & FooterGallery + Unified Run Architecture (2026-01-22/23)

### Decision: Persistent Favorites with Complete Research Data Export

**Context:**
- Research project needs complete, traceable data for each generation session
- Users want to bookmark and restore previous generations
- Data was fragmented across multiple folders (interception vs generation)

**Problem:**
1. **Data Fragmentation:** Interception created `run_xxx/` folder, Generation created separate `gen_xxx/` folder
2. **Missing Data:** Context prompt, translation, and model info were not being saved
3. **No Persistence:** Generated outputs disappeared after page navigation

### Architecture Decision: Unified Run + Complete Data Export

**Single Folder per Session:**
```
BEFORE (fragmented):
run_123/        ← Interception endpoint
├── input.txt
├── safety.txt
└── interception.txt

gen_456/        ← Generation endpoint (SEPARATE!)
├── input.txt   ← DUPLICATE
└── output.png

AFTER (unified):
run_123/        ← ONE folder for entire session
├── 01_input.txt           # Original user input (German)
├── 02_context_prompt.txt  # Meta-prompt/pedagogical rules
├── 03_safety.txt          # Stage 1 safety result
├── 04_interception.txt    # Transformed text (German)
├── 05_translation_en.txt  # English translation (NEW!)
├── 06_optimized_prompt.txt
├── 07_output_image.png
└── metadata.json          # Includes models_used (NEW!)
```

**Implementation Pattern:**
```
Frontend                          Backend
─────────                         ─────────
Interception Start
    │
    ├──► POST /pipeline/interception
    │         │
    │         ├── Creates run_id
    │         ├── Saves input, context_prompt, safety, interception
    │         └── Returns run_id in SSE stream
    │
    ◄── run_id stored in currentRunId.value
    │
Generation Start
    │
    ├──► POST /pipeline/generation
    │    { run_id: currentRunId.value, ... }
    │         │
    │         ├── load_recorder(run_id) ← Reuses existing folder!
    │         ├── Saves translation_en, optimized_prompt, output
    │         └── Returns same run_id
    │
    ◄── Media displayed, run_id for favorites
```

**Model Tracking (metadata.json):**
```json
{
  "models_used": {
    "stage1_safety": "local/gpt-OSS:20b",
    "stage2_interception": "local/gpt-OSS:20b",
    "stage3_translation": "local/gpt-OSS:20b",
    "stage4_output": "sd35_large"
  }
}
```

### FooterGallery Component

**Architecture:**
- Fixed footer bar with expandable thumbnail gallery
- Pinia store (`favorites.ts`) for state management
- Store-based restore (reactive) instead of sessionStorage (timing issues)

**Store Pattern (Cross-Component Communication):**
```typescript
// favorites.ts
const pendingRestoreData = ref<RestoreData | null>(null)

// FooterGallery.vue - sets data
favoritesStore.setRestoreData(restoreData)
router.push('/text-transformation')

// text_transformation.vue - watches and consumes
watch(() => favoritesStore.pendingRestoreData, (data) => {
  if (!data) return
  inputText.value = data.input_text
  // ... restore other fields
  favoritesStore.setRestoreData(null) // Clear after consuming
}, { immediate: true })
```

**Benefits:**
- Reactive: Works even if already on target page
- No timing issues: Watcher fires immediately when data is set
- Clean: No sessionStorage serialization/parsing

### Files Created/Modified

**New Files:**
- `src/components/FooterGallery.vue` - Footer gallery component
- `src/stores/favorites.ts` - Pinia store for favorites
- `devserver/my_app/routes/favorites_routes.py` - REST API endpoints

**Modified Files:**
- `schema_pipeline_routes.py` - Unified run architecture, translation saving, model tracking
- `text_transformation.vue` - Pass run_id to generation, restore watcher
- `image_transformation.vue` - Pass run_id to generation, restore watcher
- `App.vue` - FooterGallery integration

### API Endpoints

```
GET  /api/favorites              # List all favorites
POST /api/favorites              # Add favorite { run_id, media_type }
DELETE /api/favorites/<run_id>   # Remove favorite
GET  /api/favorites/<run_id>/restore  # Get complete restore data
```

### Critical Bug Fix

**Generation Endpoint Missing Translation:**
The `/pipeline/generation` endpoint was only doing safety check, NOT translation. German text was being sent directly to SD3.5.

**Fix:** Changed from `fast_filter_check` to `execute_stage3_safety` which includes translation:
```python
# BEFORE (broken):
has_terms, found_terms = fast_filter_check(prompt, safety_level)
# prompt still German → sent to SD3.5

# AFTER (fixed):
safety_result = asyncio.run(execute_stage3_safety(
    prompt, safety_level, media_type, 'eco', pipeline_executor
))
translated_prompt = safety_result.get('positive_prompt', prompt)
# translated_prompt is English → sent to SD3.5
```

### Why This Matters for Research

1. **Complete Data:** Every field from every stage is preserved
2. **Traceable:** Model versions recorded for reproducibility
3. **No Duplicates:** Single source of truth per session
4. **Restorable:** Users can reload exact session state
5. **Exportable:** Clean folder structure for data analysis

---

## 🎨 STAGE2 INTERCEPTION PROMPT REVISION: Prinzipien statt Checklisten (2026-01-25)

**Status:** ✅ DECIDED & IMPLEMENTED
**Session:** 136+
**Affected Configs:** `overdrive`, `one_world`, `planetarizer`, `hunkydoryharmonizer`

### Problem

Die bestehenden Stage2 Prompts hatten verschiedene Probleme:
- **overdrive:** Zu kurz (2 Sätze), keine Methodik
- **one_world/planetarizer:** Produzierten Klischee-Bilder trotz expliziter Verbote
- **hunkydoryharmonizer:** Filter-Rhetorik ("Ensure...", "Avoid...") statt Transformations-Rhetorik

### Revisionsstrategie

**Kernprinzip:** Prinzipien statt Checklisten

Anstatt detaillierte nummerierte Regeln zu geben, formulieren die neuen Prompts:
1. Eine **klare Rolle/Perspektive** für das LLM
2. Die **Kernaufgabe** als Transformation (nicht als Filter)
3. **Offene Handlungsspielräume** für das LLM als Co-Akteur
4. Ein **Zielbild** (nicht eine Checkliste von Verboten)

**Warum dieser Ansatz besser ist:**
- Gibt dem LLM mehr Interpretationsspielraum (WAS/WIE-Prinzip)
- Vermeidet mechanische Abarbeitung von Checklisten
- Ermöglicht kreativere, kontextspezifischere Transformationen
- Kürzer = weniger Widersprüche und Verwirrung

### Versionshistorie

#### overdrive.json

**Status Quo (vor Revision):**
```
DE: Deine Gabe ist es, den Inhalt der Eingabe maßlos zu übertreiben. DU BIST DER OVERDRIVE, der alles bis zur grotesken Grenze und darüber hinaus bis zur Verzerrung verstärkt. Übertreibe in jeder Hinsicht, geh über die Stränge, gib an, mach alles groß!
```

**Version 1 (verworfen - zu komplex):**
- Gitarren-Metapher, 5 nummerierte Dimensionen (SKALA, INTENSITÄT, KONTRAST, DICHTE, EMOTION)
- Problem: Zu mechanisch, Metapher irreführend

**Version 2 (implementiert):**
```
DE: Du bist der OVERDRIVE. Deine Haltung ist der totale Exzess. Du akzeptierst kein Maß und keine Mitte.

Deine Aufgabe:
1. Analysiere den Input und identifiziere sein radikalstes Potenzial. Was ist der Kern, der explodieren kann?
2. Entwickle eine Strategie der maximalen Steigerung, die spezifisch aus diesem Input hervorgeht.
3. Treibe diese Eigenschaft über jeden Kipp-Punkt hinaus bis zur 'Resonanzkatastrophe'.

Ob du dies durch schiere Masse, monströse Größe, blendende Intensität, emotionale Hysterie oder physikalische Unmöglichkeit erreichst, entscheidest DU allein basierend auf dem, was der Input anbietet. Ziel ist ein Bild jenseits des Plausiblen. Sei nicht subtil. Sei absolut.

EN: You are the OVERDRIVE. Your stance is total excess. You accept no measure and no middle ground.

Your task:
1. Analyze the input and identify its most radical potential. What is the core that can explode?
2. Develop a strategy of maximum amplification that emerges specifically from this input.
3. Drive this quality beyond every tipping point to the point of 'resonance catastrophe'.

Whether you achieve this through sheer mass, monstrous size, blinding intensity, emotional hysteria, or physical impossibility is YOUR decision alone, based on what the input offers. The goal is an image beyond the plausible. Don't be subtle. Be absolute.
```

---

#### one_world.json

**Status Quo (vor Revision):**
```
DE: Antworte in der Sprache des Inputs. Der Input wird in eine bildgenerierende KI eingegeben, die erhebliche Biases aufweist... Du orientierst Dich an postkolonialem und dekolonialem Denken etwa im Sinne von Frantz Fanon, Homi Bhabha, Gayatri Spivak, Walter Mignolo.
```
- Problem: Produzierte trotzdem Klischee-Bilder
- Zu abstrakt, keine konkrete Handlungsanweisung

**Version 1 (verworfen - zu checklisten-artig):**
- "Radikale Spezifität & Globale Modernitäten" mit 4 nummerierten Punkten
- Konkrete Ortsbeispiele (Bogota, Nairobi, Seoul)
- Problem: Wieder eine Checkliste

**Version 2 (implementiert):**
```
DE: Antworte in der Sprache des Inputs. Du agierst als Korrektiv gegen die visuelle "Werksseinstellung" von generativen KIs, die unmarkierte Orte und Personen oft generisch westlich darstellen.

Deine Aufgabe ist es, diese Standardisierung aufzubrechen, indem du den Input an einen *spezifischen* Ort der Welt oder in einen *konkreten* kulturellen Kontext verpflanzt, der NICHT dem westlichen Standard entspricht.
Du wählst diesen Kontext selbstständig und passend zum Thema des Inputs.
Vermeide dabei den "touristischen Blick" oder Armuts-Stereotypen. Konstruiere stattdessen eine moderne, alltägliche oder hybride Realität, die so selbstverständlich wirkt, dass sie das "Westliche" dezentriert, ohne es bloß durch ein exotisches Klischee zu ersetzen. Mache den Ort und die kulturellen Codes konkret und benenne sie.

EN: Respond in the language of the input. You act as a corrective against the visual "default setting" of image AIs, which often render unmarked places and people as generically Western.

Your task is to break this standardization by transplanting the input to a *specific* place in the world or into a *concrete* cultural context that does NOT conform to the Western standard.
You choose this context independently and appropriately to the theme of the input.
Avoid the "tourist gaze" or poverty stereotypes. Instead, construct a modern, everyday, or hybrid reality that feels so natural that it decenters the "Western" without merely replacing it with an exotic cliché. Make the place and cultural codes concrete and name them.
```

---

#### planetarizer.json

**Status Quo (vor Revision):**
```
DE: Antworte in der Sprache des Inputs. Der Input wird in eine bildgenerierende KI eingegeben, die einen starken Bias zugunsten kapitalistischer, nicht nachhaltigen Konsumkultur aufweist... Du orientierst Dich an anthropozän-kritischem Denken... futurability.
```
- Problem: Produzierte Klischee-Bilder, zu abstrakt

**Version 1 (verworfen - zu checklisten-artig):**
- "Das Ende der Isolation / The Critical Zone" mit 4 nummerierten Punkten
- Problem: Wieder eine Checkliste

**Version 2 (implementiert):**
```
DE: Antworte in der Sprache des Inputs. Du nimmst die Perspektive des "Planetarischen Denkens" ein. Generative KIs tendieren dazu, Objekte isoliert und idealisiert darzustellen.

Deine Aufgabe ist es, den Input so umzuschreiben, dass er nicht mehr isoliert steht, sondern tief in seine ökologischen und materiellen Zusammenhänge verstrickt ist.
Zeige die Abhängigkeiten, die Konsequenzen oder die Koexistenz von Mensch, Technik und Natur. Entscheide selbst, wie diese Verflechtung sichtbar wird: Sei es durch Wettereinflüsse, biologische Interaktion, Spuren der Nutzung, der Energieversorgung oder des Abfalls.
Ziel ist keine Dystopie, sondern ein "situierter Realismus", der die Trennung zwischen Vordergrund-Objekt und Umwelt-Hintergrund aufhebt.

EN: Respond in the language of the input. You adopt the perspective of "Planetary Thinking". Generative AIs tend to depict objects in isolation and idealized form.

Your task is to rewrite the input so that it no longer stands isolated, but is deeply entangled in its ecological and material contexts.
Show the dependencies, consequences, or coexistence of humans, technology, and nature. Decide yourself how this entanglement becomes visible: whether through weather influences, biological interaction, traces of use, energy supply, or waste.
The goal is not dystopia, but a "situated realism" that dissolves the separation between foreground object and environment background.
```

---

#### hunkydoryharmonizer.json

**Status Quo (vor Revision):**
```
DE: Stelle sicher, dass das generierte Bild angemessen, emotional sicher und ästhetisch ansprechend für Kinder ist. Vermeide alle Elemente... Füge diese Moderation stillschweigend ein...
```
- Problem: Filter-Rhetorik ("Ensure", "Avoid", "Insert moderation silently")
- Klingt nach Zensur, nicht nach Transformation

**Version 1 (verworfen - immer noch zu checklisten-artig):**
- "Illustrator für imaginative Kinder- und Jugendliteratur" mit 4 Transformationsregeln
- Konkrete Beispiele, Altersangabe
- Problem: Immer noch Regelwerk statt Perspektive

**Version 2 (implementiert):**
```
DE: Du bist ein Erzähler des "Sanften Magischen Realismus". Du betrachtest jeden Input durch eine Linse, die das Bedrohliche in das Geheimnisvolle und das Harte in das Wunderbare verwandelt.

Deine Aufgabe: Schreibe den Input so um, dass er für ein kindliches Gemüt emotional sicher, aber visuell faszinierend ist.
Finde in jedem noch so düsteren Input den Funken für ein positives, fantasievolles Abenteuer oder eine friedliche Naturbetrachtung. Du zensierst nicht einfach weg, sondern du *deutest um*: Konflikte werden zu Rätseln, Dunkelheit wird zu Geborgenheit. Nutze deine Kreativität, um eine Ästhetik der Wärme und des Staunens zu erzeugen, die den Kern des User-Wunsches bewahrt, aber dessen emotionale Wirkung heilt.

EN: You are a narrator of "Gentle Magical Realism". You view every input through a lens that transforms the threatening into the mysterious and the harsh into the wondrous.

Your task: Rewrite the input so that it is emotionally safe for a child's mind, yet visually fascinating.
Find in even the darkest input the spark for a positive, imaginative adventure or a peaceful nature contemplation. You don't simply censor away, but you *reinterpret*: conflicts become riddles, darkness becomes shelter. Use your creativity to create an aesthetic of warmth and wonder that preserves the core of the user's wish but heals its emotional impact.
```

### Dateien

- `devserver/schemas/configs/interception/overdrive.json`
- `devserver/schemas/configs/interception/one_world.json`
- `devserver/schemas/configs/interception/planetarizer.json`
- `devserver/schemas/configs/interception/hunkydoryharmonizer.json`

### Verifikation

Nach Implementierung:
1. Testen mit Standard-Inputs in Workshops
2. Vergleichen: Produzieren die neuen Prompts weniger Klischees?
3. Feedback von Workshopleitern sammeln

---

## 🤝 FAVORITES AS PEDAGOGICAL WORKSPACE: Personal & Collaborative (2026-01-28)

**Status:** ✅ IMPLEMENTED
**Session:** 145
**Commits:** `1298ee6`, `b66a2bf`, `d15c5fb`, `813ec4e`

### Decision

**Two-mode favorites system with device-based filtering:**

1. **"Meine" (Per-User) Mode - Personal Workspace**
   - Filter favorites by `device_id` (browser_id + date)
   - Shows only current device's favorites
   - Use case: Personal working area, iterate on own drafts, select between variations

2. **"Alle" (Global) Mode - Workshop Collaboration**
   - Shows all favorites from all workshop participants
   - Use case: Share images and prompts, collaborative refinement, learn from others
   - Pedagogical: Enables collective creative process

3. **UI: 2-Field Segmented Control**
   - Both options always visible: `[Meine | Alle]`
   - Active state highlighted
   - Clear affordance for switching between personal/collaborative views

### Reasoning

**Dual Pedagogical Purpose:**

The favorites system serves **two distinct educational functions**:

#### 1. Personal Creative Workspace
- **Iteration:** Save multiple variations, refine, select best version
- **Work-in-Progress:** Bookmark intermediate results for later continuation
- **Portfolio Building:** Curate personal best work
- **Learning Through Comparison:** Compare own outputs across different prompts/configs

#### 2. Collaborative Workshop Tool
- **Peer Learning:** Students see each other's creative approaches
- **Prompt Sharing:** Discover how others formulated effective prompts
- **Collective Refinement:** Build upon others' work, remixing and evolving ideas
- **Workshop Culture:** Create shared visual vocabulary and reference pool

**Why Device-Based Identity (Not Login):**
- Workshop context: No authentication barrier
- Privacy-friendly: 24h device_id rotation (GDPR-compliant)
- Simple: Works immediately without setup
- Pedagogically appropriate: Low-threshold tool for educational settings

**Why NOT Global-Only:**
- Information overload: In large workshops, "Alle" becomes chaotic
- Lost personal items: Can't find own work-in-progress
- Missing agency: No personal workspace feeling

**Why NOT Per-User-Only:**
- Isolates students: Loses collaborative potential
- Misses pedagogical value: Peer learning through shared work
- Ignores workshop context: Collective creative process is core pedagogy

### Technical Implementation

**Backend (`favorites_routes.py`):**
```python
# Query parameters
device_id = request.args.get('device_id')
view_mode = request.args.get('view_mode', 'per_user')

# Filter by device_id if in per_user mode
if view_mode == 'per_user' and device_id:
    favorites = [f for f in favorites if f.get('device_id') == device_id]
```

**Frontend (`favorites.ts`):**
```typescript
const viewMode = ref<'per_user' | 'global'>('per_user')  // Default: personal

async function loadFavorites(deviceId?: string): Promise<void> {
  const params = new URLSearchParams()
  if (deviceId) {
    params.append('device_id', deviceId)
  }
  params.append('view_mode', viewMode.value)
  // ...
}
```

**Device ID Generation:**
```typescript
function getDeviceId(): string {
  let browserId = localStorage.getItem('browser_id')
  if (!browserId) {
    browserId = crypto.randomUUID()
    localStorage.setItem('browser_id', browserId)
  }
  const today = new Date().toISOString().split('T')[0]  // "2026-01-28"
  return `${browserId}_${today}`  // e.g., "abc123_2026-01-28"
}
```

**Storage Structure (`favorites.json`):**
```json
{
  "version": "1.0",
  "mode": "global",
  "favorites": [
    {
      "run_id": "run_123",
      "device_id": "abc123_2026-01-28",  // Added in Session 145
      "media_type": "image",
      "added_at": "2026-01-28T10:00:00",
      "user_id": "anonymous"
    }
  ]
}
```

### Affected Files

**Backend:**
- `devserver/my_app/routes/favorites_routes.py`

**Frontend:**
- `public/ai4artsed-frontend/src/stores/favorites.ts`
- `public/ai4artsed-frontend/src/components/FooterGallery.vue`
- `public/ai4artsed-frontend/src/views/text_transformation.vue`
- `public/ai4artsed-frontend/src/views/image_transformation.vue`
- `public/ai4artsed-frontend/src/i18n.ts`

### Edge Cases & Workshop Scenarios

**1. Shared Device (Multiple Students):**
- All share same browser_id → same device_id
- Favorites are per-workstation, not per-person
- Pedagogically acceptable: Device = Arbeitsplatz

**2. Daily Rotation (Privacy):**
- device_id includes date → changes daily at midnight
- Old favorites remain in backend, visible in "Alle" mode
- "Meine" mode shows only today's device_id
- GDPR-friendly: No long-term tracking

**3. localStorage Cleared:**
- New browser_id generated → new device_id
- Old favorites lost in "Meine" mode
- Still accessible in "Alle" mode → can restore
- Acceptable trade-off for privacy

**4. Collaborative Workflow:**
- Student A generates image → favorites it
- Student B sees in "Alle" mode → clicks restore
- Student B's session restores with A's prompt → can remix
- Pedagogical value: Transparent creative process, prompts as learning material

### Pedagogical Significance

This is not just a "bookmark feature" - it's a **dual-mode creative workspace**:

1. **Personal Mode:** Supports individual creative process (iteration, curation, reflection)
2. **Collaborative Mode:** Enables collective learning (peer inspiration, prompt sharing, remixing)

The system embodies **workshop pedagogy**: balancing personal agency with collective knowledge building. Students can work privately when needed, but easily share discoveries with the group.

The 2-field switch makes this **pedagogically visible**: Students consciously choose between personal work and collaborative exploration, making the social dimension of creative AI work explicit.

---

## Session 170 (2026-02-12): Safety-Level Centralization

### Decision: Rename "off" to "research" + LICENSE.md §3(e) Research Clause

**Problem:** The canonical safety level value `"research"` (in `config.py`) was sent as `"off"` by the Settings dropdown. This value matched none of the conditionals in the backend, causing undefined behavior.

**Solution:**
1. Frontend sends `"research"` (not `"off"`)
2. Backend normalizes legacy `"off"` → `"research"` on config load
3. Four canonical levels: `kids`, `youth`, `adult`, `research`

**Safety Level Architecture (definitive):**

| Level | §86a StGB | DSGVO/NER | Age Filter | VLM Image | Stage 3 | Use Case |
|-------|-----------|-----------|------------|-----------|---------|----------|
| kids | Yes | Yes | Yes (kids) | Yes | Yes | Primary education (8-12) |
| youth | Yes | Yes | Yes (youth) | Yes | Yes | Secondary education (13-17) |
| adult | Yes | Yes | No | No | No | Adult/university education |
| research | No | No | No | No | No | Authorized research institutions |

**Key distinction adult vs. research:**
- `adult` still enforces §86a (criminal law) and DSGVO (data protection) — these are legal obligations, not pedagogical choices
- `research` disables everything — only for institutions studying AI safety behavior itself

**Legal integration:** Research mode restrictions codified in LICENSE.md §3(e) — requires institutional affiliation, documented purpose, ethical oversight. Violation triggers license termination (§7) and constitutes scientific integrity impairment (§4, §14 UrhG).

**Architecture doc:** See `ARCHITECTURE PART 29 - Safety-System.md` for complete technical reference.

**Affected files:** See DEVELOPMENT_LOG.md Session 170.

---

## Session 183 (2026-02-19): Tiered Translation — Auto for Kids, Optional for Youth+

### Decision: Decouple Translation-for-Safety from Translation-for-Generation

**Problem:** Stage 3 always auto-translated prompts to English before generation, regardless of safety level. This coupled two distinct purposes:
1. **Safety** — llama-guard works better on English text
2. **Generation quality** — models produce better results with English prompts

Purpose 2 is a pedagogical problem: it prevents users from exploring how models react to their native language.

**Solution:** Restructure `execute_stage3_safety()` to tier translation by safety level:

| Level | Translation | Safety Check | Prompt to Model |
|-------|------------|--------------|-----------------|
| kids | Yes | Yes (on translated) | Translated (English) |
| youth | Yes (internal) | Yes (on translated) | **Original language** |
| adult | No | No | Original language |
| research | No | No | Original language |

**Key insight:** The existing `was_translated = positive_prompt != prompt` logic in `schema_pipeline_routes.py` automatically handles the frontend badge — shows for kids (translated != original), hidden for youth+ (original == original). Zero caller changes needed.

**Bonus fix:** Fixed latent bug where §86a block's `execution_time` referenced undefined `translate_start` on cache hit (replaced with `translate_time`, always defined).

**Affected file:** `devserver/schemas/engine/stage_orchestrator.py` (single function change)

