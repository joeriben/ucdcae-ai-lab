# i18n Translation Work Orders

Instructions for the `i18n-translator` agent: process all entries under **Pending**, translate
the listed keys from `en.ts` into `de.ts`, `tr.ts`, `ko.ts`, `uk.ts`, `fr.ts`, `es.ts`, `he.ts`, `ar.ts`, then move
each processed work order to **Completed** with a date stamp.

## Pending

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

### WO-2026-02-25-backend-status-dashboard
- **Session**: N/A
- **Scope**: en.ts
- **Changed keys** (new):
  - `settings.tabs.status` (NEW): "Backend Status"
  - `settings.backendStatus.loading` (NEW): "Checking backend status..."
  - `settings.backendStatus.refresh` (NEW): "Refresh"
  - `settings.backendStatus.refreshing` (NEW): "Refreshing..."
  - `settings.backendStatus.localInfrastructure` (NEW): "Local Infrastructure"
  - `settings.backendStatus.cloudApis` (NEW): "Cloud APIs"
  - `settings.backendStatus.outputConfigs` (NEW): "Output Configs by Backend"
  - `settings.backendStatus.reachable` (NEW): "Reachable"
  - `settings.backendStatus.unreachable` (NEW): "Unreachable"
  - `settings.backendStatus.available` (NEW): "Available"
  - `settings.backendStatus.unavailable` (NEW): "Unavailable"
  - `settings.backendStatus.configured` (NEW): "Configured"
  - `settings.backendStatus.notConfigured` (NEW): "Not Configured"
  - `settings.backendStatus.gpuService` (NEW): "GPU Service"
  - `settings.backendStatus.subBackend` (NEW): "Sub-Backend"
  - `settings.backendStatus.status` (NEW): "Status"
  - `settings.backendStatus.comfyui` (NEW): "ComfyUI / SwarmUI"
  - `settings.backendStatus.ollama` (NEW): "Ollama"
  - `settings.backendStatus.gpuHardware` (NEW): "GPU Hardware"
  - `settings.backendStatus.notDetected` (NEW): "Not detected"
  - `settings.backendStatus.showModels` (NEW): "Show models"
  - `settings.backendStatus.hideModels` (NEW): "Hide models"
  - `settings.backendStatus.provider` (NEW): "Provider"
  - `settings.backendStatus.keyStatus` (NEW): "API Key"
  - `settings.backendStatus.dsgvoLabel` (NEW): "DSGVO"
  - `settings.backendStatus.region` (NEW): "Region"
  - `settings.backendStatus.dsgvoCompliant` (NEW): "Compliant"
  - `settings.backendStatus.dsgvoNotCompliant` (NEW): "Not Compliant"
  - `settings.backendStatus.configsAvailable` (NEW): "{available} of {total} configs available"
  - `settings.backendStatus.hidden` (NEW): "hidden"
- **Context**: New "Backend Status" tab in Settings page showing infrastructure health dashboard. All keys are technical/admin-facing. Translate status labels naturally (e.g. "Erreichbar"/"Nicht erreichbar" in German, etc.). Keep technical terms like "GPU Service", "ComfyUI", "Ollama", "DSGVO", "API Key" untranslated. The `{available}` and `{total}` are interpolation placeholders — keep them as-is.

### WO-2026-02-24-trans-aktion-poetry-configs
- **Session**: 208
- **Scope**: 5 JSON files in `devserver/schemas/configs/interception/`: `trans_aktion_rilke.json`, `trans_aktion_hoelderlin.json`, `trans_aktion_basho.json`, `trans_aktion_dickinson.json`, `trans_aktion_whitman.json`
- **Changed keys**: Add `tr`, `ko`, `uk`, `fr`, `es`, `he`, `ar` entries to `name`, `description`, `category` LocalizedString objects
- **Context**: 5 new Trans-Aktion interception configs using real poetry as collision material. Each config has `en` + `de` already. Translate ONLY `name`, `description`, and `category` — the `context` field contains meta-prompts (LLM instructions) and poems and must NOT be translated. The poet names (Rilke, Hoelderlin, Basho, Dickinson, Whitman) should remain unchanged in all languages.

### WO-2026-02-23-hebrew-arabic-language-labels
- **Session**: 201
- **Scope**: en.ts + all language files (de, tr, ko, uk, fr, es, he, ar)
- **Changed keys** (new):
  - `settings.general.hebrewHe` (NEW): "Hebrew (he)"
  - `settings.general.arabicAr` (NEW): "Arabic (ar)"
- **Context**: New language option labels for Hebrew and Arabic RTL language support. Translate the language name into each target language. Examples: German "Hebräisch (he)" / "Arabisch (ar)", Turkish "İbranice (he)" / "Arapça (ar)", Korean "히브리어 (he)" / "아랍어 (ar)", Ukrainian "Іврит (he)" / "Арабська (ar)", French "Hébreu (he)" / "Arabe (ar)", Spanish "Hebreo (he)" / "Árabe (ar)". Note: he.ts should have hebrewHe = "עברית (he)" and ar.ts should have arabicAr = "العربية (ar)".

### WO-2026-02-23-hebrew-arabic-interception-configs
- **Session**: 201
- **Scope**: All 36 JSON files in `devserver/schemas/configs/interception/` + `devserver/schemas/llama_guard_explanations.json`
- **Changed keys**: Add `he` and `ar` entries to all LocalizedString objects (name, description, hint_message, base_message, category, codes)
- **Context**: Interception config names/descriptions and safety filter explanations need Hebrew and Arabic translations. These are user-facing strings shown in the UI. The `context` field in interception configs is a meta-prompt (LLM instruction) and should NOT be translated — only `name`, `description`, and `category` fields.

### WO-2026-02-23-spanish-language-label
- **Session**: 200
- **Scope**: en.ts
- **Changed keys** (new):
  - `settings.general.spanishEs` (NEW): "Spanish (es)"
- **Context**: New language option label for Spanish, added as part of i18n Spanish (es) language support. Translate the language name into each target language (e.g. German: "Spanisch (es)", Turkish: "İspanyolca (es)", Korean: "스페인어 (es)", Ukrainian: "Іспанська (es)", French: "Espagnol (es)"). Note: es.ts already has this key set to "Español (es)".

## Completed

### WO-2026-02-23-hebrew-full-translation (HE portion of hebrew-arabic-full-translation)
- **Completed**: 2026-02-24
- **Scope**: he.ts — full translation of all ~1370 keys from en.ts
- **Result**: All 30 top-level sections translated. vue-tsc and build pass.

### WO-2026-02-23-arabic-full-translation
- **Completed**: 2026-02-24
- **Scope**: ar.ts — full translation of all ~1370 keys from en.ts
- **Result**: All sections translated (3-way parallel split + assembly). vue-tsc and build pass.
