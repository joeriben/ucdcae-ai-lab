# TODO: Trans-Aktion — Von Prompt-Interception zu Prompt-Abduktion

## Vision (aus DEVELOPMENT_DECISIONS.md, 2026-02-24)

**Kernproblem**: Die ursprüngliche Interception-Idee war ABDUKTION — "Entführung" des User-Prompts durch das System. Im Entwicklungsprozess wurde das transgressiv-künstlerische vom pragmatisch-pädagogischen absorbiert. User wählt Preset, LLM transformiert brav. Das ist das GEGENTEIL der Vision.

**Zweites verlorenes Prinzip**: Multimodale Impulse (nicht nur Text-Regeln) — anti-logozentrische Position, im text-zentrierten Design verloren.

## Aufgabe: Vue-Komponente für multimodale abduktive Interception

Ein Vue das eine **multimodale Eingabe abduktiv intercepted**:
- Akzeptiert Text, Bild (Kamera), Audio (Mikrofon)
- Abduktion durch genuíne Kontingenz, NICHT durch LLM-Prompting
- User REAGIERT auf Systemwiderständigkeit statt sie zu STEUERN

## Drei Quellen genuiner Kontingenz (implementierbar)

1. **Modell-Insuffizienz**: 0.5B-2B Modelle scheitern strukturell. Brüche/Drift/Fragmente sind unvermeidlich, nicht simuliert. Das Modell IST materiell unfähig zur Compliance.

2. **Domänen-Mismatch**: Spezialisierte Modelle (Code, Math, Guard, Emotion, Surveillance) sehen die Welt NUR durch ihre Trainingslinse. Bias ist real, nicht simuliert. Pädagogisch explosiv.

3. **Vektor-Operationen**: Mathematische Operationen im Embedding-Raum, am LLM komplett vorbei. Surrealizer demonstriert das Prinzip bereits (CLIP-L-only). Erweiterung: Impuls-Text-Eigenschaften (Varianz, Normen, Entropie) als Verformungsanweisung auf WAS-Embedding.

## Die Traumkette: Gestapelte Depravation

```
Insuffizientes LLM (0.5B) → schreibt gebrochenen Prompt
  → "Depraved" Encoder (korrupter CLIP/T5) → encodiert systematisch falsch
    → Mächtiges Diffusionsmodell (FLUX.2) → rendert hochwertig
```

Paradox: Bildqualität steht, aber WAS das Bild zeigt ist durch genuine Kontingenz unvorhersehbar.

## Dreischichtiger Kollisions-Ansatz (Forschungssynthese)

1. **Schicht 1 — Mechanisch (kein Modell)**: N+7 / Satz-Interleaving / Cut-Up. SpaCy als Parser. Genuine Kollision.
2. **Schicht 2 — Insuffizientes Modell**: Bekommt mechanisch kollidiertes Material, scheitert partiell. Das Scheitern IST die Kunst.
3. **Schicht 3 — Embedding-SLERP**: T5-Mittelpunkt zwischen Gedicht und Prompt. Verbindet sich mit T5 SAE-Forschung (Session 192).

## Forschungsstand (Quellen in DEVELOPMENT_DECISIONS.md)

- **LLM-Prompted Fusion ist fundamental der falsche Ansatz** — LLMs simulieren Inkoheärenz statt sie zu produzieren (EBR Mai 2024, CHI 2024 "Art or Artifice?")
- **Repeat Curse / Induction Head Toxicity** (ACL 2025) — Attention-Köpfe kopieren Input
- **CS4 Benchmark** — Constraint-Satisfaction degradiert nichtlinear bei kleinen Modellen
- **Prompt-als-Instruktions-Konflikt** — Gedicht als konkurrierende System-Instruktion = genuine Confusion

## Kunsthistorischer Kontext

- Oulipo (formale Constraints als kreative Motoren)
- Menkman Glitch Studies Manifesto (Glitch als positiver Disruptor)
- Parrish *Articulations* (Embedding-Raum als Geographie)
- Bowman VAE-Interpolation (geometrische Notwendigkeit)
- Goodwin *1 the Road* (Kollision Trainingskorpus × Echtzeit-Input × insuffizientes Netzwerk)

## Referenz-Dateien

- `docs/DEVELOPMENT_DECISIONS.md` ab Zeile 4428 — vollständige Analyse
- `docs/DEVELOPMENT_DECISIONS.md` ab Zeile 4480 — Forschungsstand + Taxonomie
- Bestehende Vue: `src/views/text_transformation.vue`
- Surrealizer-Prinzip: `/SwarmUI/dlbackend/ComfyUI/custom_nodes/ai4artsed_comfyui/ai4artsed_t5_clip_fusion.py`
- T5 SAE Plan: `docs/plans/t5_interpretability_research.md`

## Status

- [x] Forschungssynthese geschrieben (DEVELOPMENT_DECISIONS.md)
- [x] "steal"/"stehlen" aus allen Configs entfernt
- [ ] Vue-Komponente für multimodale abduktive Interception
- [ ] Schicht 1: Mechanische Kollision (SpaCy N+7, Interleaving)
- [ ] Schicht 2: Insuffizientes Modell (enable_thinking fix, repeat_penalty)
- [ ] Schicht 3: Embedding-SLERP (T5 SAE Forschung)
- [ ] Domänen-Mismatch-Configs (Guard, Code, Surveillance als Interception-Modelle)
- [ ] Depraved Encoder (T5-Small, Extremquantisierung, Attention-Nulling)
