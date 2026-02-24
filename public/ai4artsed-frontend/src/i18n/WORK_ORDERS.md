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
