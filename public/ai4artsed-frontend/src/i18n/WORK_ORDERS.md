# i18n Translation Work Orders

Instructions for the `i18n-translator` agent: process all entries under **Pending**, translate
the listed keys from `en.ts` into `de.ts`, `tr.ts`, `ko.ts`, `uk.ts`, `fr.ts`, `es.ts`, `he.ts`, `ar.ts`, then move
each processed work order to **Completed** with a date stamp.

## Pending

### WO-2026-02-27-expert-energy-fact-reword
- **Session**: 220
- **Scope**: en.ts
- **Changed keys** (new or modified):
  - `edutainment.energy.expert_1` (MODIFIED): emoji changed from 📊 to ⚡, text reworded to "Current draw: {watts} W | GPU load: {util}% | Accumulated: {kwh} kWh (integrated over time)"
- **Context**: The old text used "=" which falsely implied kWh is derived from a single W reading. kWh is actually integrated from sampled power readings over time. Reworded to separate the three independent readings with pipes. Also replaced 📊 emoji.

### WO-2026-02-27-replace-chart-emoji
- **Session**: 220
- **Scope**: en.ts
- **Changed keys** (new or modified):
  - `edutainment.model.expert_2` (MODIFIED): emoji 📊 -> 🔧 (only emoji changed, text unchanged)
  - `edutainment.environment.expert_1` (MODIFIED): emoji 📊 -> 🌍 (only emoji changed, text unchanged)
- **Context**: 📊 (bar chart emoji) was flagged as forbidden. Replaced with contextually fitting alternatives: 🔧 for quantization, 🌍 for CO₂ calculation. Only the leading emoji changed, NOT the text.

### WO-2026-02-27-denoising-progress-view
- **Session**: 220
- **Scope**: en.ts
- **Changed keys** (new or modified):
  - `edutainment.denoising.modelLoading` (NEW)
  - `edutainment.denoising.modelCard` (NEW)
  - `edutainment.denoising.publisher` (NEW)
  - `edutainment.denoising.architecture` (NEW)
  - `edutainment.denoising.parameters` (NEW)
  - `edutainment.denoising.textEncoders` (NEW)
  - `edutainment.denoising.quantization` (NEW)
  - `edutainment.denoising.vramRequired` (NEW)
  - `edutainment.denoising.resolution` (NEW)
  - `edutainment.denoising.license` (NEW)
  - `edutainment.denoising.fairCulture` (NEW)
  - `edutainment.denoising.safetyByDesign` (NEW)
  - `edutainment.denoising.denoisingActive` (NEW)
- **Context**: Labels for expert-mode denoising progress view. Shows model "Steckbrief" (identity card) during VRAM loading and live denoising stats during generation. Technical/educational terms, keep short.

<!-- Add new work orders here. Format:

### WO-YYYY-MM-DD-short-description
- **Session**: <number>
- **Scope**: en.ts (or additional files like canvas.ts, interception configs)
- **Changed keys** (new or modified):
  - `section.subsection.key` (NEW)
  - `section.subsection.key` (MODIFIED): "old value" -> "new value"
- **Context**: Brief semantic description to guide translation accuracy.

Tags:
  - (NEW) = key did not exist before, translate from English
  - (MODIFIED) = English text changed, all 5 translations are stale and must be re-done
-->

## Completed

### WO-2026-02-23-hebrew-arabic-interception-configs
- **Completed**: 2026-02-26
- **Scope**: 33 interception config JSONs (32 with LocalizedString + hunkydoryharmonizer) + llama_guard_explanations.json
- **Result**: Added he/ar to name, description, category in all configs. llama_guard_explanations: he/ar added to base_message, hint_message, all 13 codes, fallback. 4 configs skipped (flat string descriptions: lyrics_from_theme, lyrics_refinement, tags_generation, tag_suggestion_from_lyrics).

### WO-2026-02-26-surrealizer-fusion-strategy
- **Completed**: 2026-02-26
- **Scope**: en.ts → de/tr/ko/uk/fr/es/he/ar
- **Result**: 13 keys (4 MODIFIED + 9 NEW) translated into all 8 target languages.

### WO-2026-02-25-random-prompt-token-limit
- **Completed**: 2026-02-26
- **Scope**: en.ts → de/tr/ko/uk/fr/es/he/ar
- **Result**: 1 NEW key translated into all 8 target languages.

### WO-2026-02-25-sketch-canvas
- **Completed**: 2026-02-26
- **Scope**: en.ts → de/tr/ko/uk/fr/es/he/ar
- **Result**: 11 NEW keys translated into all 8 target languages.

### WO-2026-02-25-backend-status-dashboard
- **Completed**: 2026-02-26
- **Scope**: en.ts → de/tr/ko/uk/fr/es/he/ar
- **Result**: 31 NEW keys translated into all 8 target languages.

### WO-2026-02-23-hebrew-arabic-language-labels
- **Completed**: 2026-02-26
- **Scope**: en.ts → de/tr/ko/uk/fr/es/he/ar
- **Result**: 2 NEW keys (hebrewHe, arabicAr) translated with native language names.

### WO-2026-02-23-spanish-language-label
- **Completed**: 2026-02-26
- **Scope**: en.ts → de/tr/ko/uk/fr/es/he/ar
- **Result**: 1 NEW key (spanishEs) translated into all 8 target languages.

### WO-2026-02-24-trans-aktion-poetry-configs
- **Completed**: 2026-02-26
- **Scope**: 6 JSON files in `devserver/schemas/configs/interception/`: trans_aktion_basho, trans_aktion_hoelderlin, trans_aktion_mirabai, trans_aktion_nahuatl, trans_aktion_sappho, trans_aktion_yoruba_oriki
- **Result**: Added tr/ko/uk/fr/es/he/ar to name+description+category in all 6 trans_aktion configs. Note: original WO listed rilke/dickinson/whitman but actual files were sappho/mirabai/nahuatl/yoruba_oriki (renamed in a session that didn't update the WO).

### WO-2026-02-23-hebrew-full-translation (HE portion of hebrew-arabic-full-translation)
- **Completed**: 2026-02-24
- **Scope**: he.ts — full translation of all ~1370 keys from en.ts
- **Result**: All 30 top-level sections translated. vue-tsc and build pass.

### WO-2026-02-23-arabic-full-translation
- **Completed**: 2026-02-24
- **Scope**: ar.ts — full translation of all ~1370 keys from en.ts
- **Result**: All sections translated (3-way parallel split + assembly). vue-tsc and build pass.
