# i18n Translation Work Orders

Instructions for the `i18n-translator` agent: process all entries under **Pending**, translate
the listed keys from `en.ts` into `de.ts`, `tr.ts`, `ko.ts`, `uk.ts`, `fr.ts`, `es.ts`, then move
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

### WO-2026-02-23-spanish-language-label
- **Session**: 200
- **Scope**: en.ts
- **Changed keys** (new):
  - `settings.general.spanishEs` (NEW): "Spanish (es)"
- **Context**: New language option label for Spanish, added as part of i18n Spanish (es) language support. Translate the language name into each target language (e.g. German: "Spanisch (es)", Turkish: "İspanyolca (es)", Korean: "스페인어 (es)", Ukrainian: "Іспанська (es)", French: "Espagnol (es)"). Note: es.ts already has this key set to "Español (es)".

## Completed

<!-- Processed work orders are moved here with date stamp -->
