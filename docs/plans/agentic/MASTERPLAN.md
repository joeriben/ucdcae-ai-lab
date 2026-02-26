# Agentic AI4ArtsEd: MASTER PLAN

## Context

AI4ArtsEd begann als orchestrierte Pipeline-Plattform (2025). Mit der Reifung agentischer AI-Muster (ICLR 2026 RSI Workshop, UNO Framework, Claude Opus 4.6) kann die Plattform nun selbst agentisch handeln: Erfahrungen sammeln, sich selbst monitoren, Skills autonom ausführen, und — als Forschungsbeitrag — ein **diversitätssensibles Prompt-Transformationsmodell** via Knowledge Distillation trainieren.

**Bestehende agentische Infrastruktur**:
- Wikipedia Research Loop = ReAct-Agent (`pipeline_executor.py:496-720`)
- VRAMCoordinator = Ressourcen-Negotiation mit LRU-Eviction
- SSE `user_activity` = Idle-Detektor | ConfigLoader = Plugin-Architektur
- Canvas Executor = DAG-Workflow-Engine
- 3.752 Runs / 7.1 GB Experience-Daten

---

## Architecture Overview

```
┌──────────────────────────────────────┐
│          USER / WORKSHOP             │
└──────────────┬───────────────────────┘
               │ HTTP/SSE
┌──────────────▼───────────────────────┐
│        DEVSERVER (17802)             │
│  ┌────────────────────────────────┐  │
│  │  4-Stage Orchestrator (heilig) │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │  AGENTIC DAEMON (Daemon-Thread)│  │
│  │  Monitor │ Scheduler │ Skills  │  │
│  │  Experience Engine │ Prompt AI │  │
│  └────────────────────────────────┘  │
└──────────────┬───────────────────────┘
               │ HTTP
┌──────────────▼───────────────────────┐
│      GPU SERVICE (17803)             │
│  Diffusers │ HeartMuLa │ LLM │ SAE  │
│       RTX 6000 Blackwell 96GB       │
└──────────────────────────────────────┘
```

Daemon lebt IM DevServer-Prozess (kein separater Prozess).

---

## 5 Pillars (Kurzübersicht)

| # | Pillar | Kern | Phase-Plan |
|---|--------|------|------------|
| 1 | **Experience Engine** | JSON-Aggregation + Session-Aware Narrative Summaries (Platform-Sensibilitaet) + Embeddings | `phase-1-experience-engine.md` |
| 2 | **Self-Monitoring Daemon** | 30s-Tick, Health-Checks, Idle-Detection, Pre-Emption | `phase-0-daemon-foundation.md` |
| 3 | **Skill System** | AUTO/CONFIRM/MANUAL Bounded Autonomy | `phase-2-skill-system.md` |
| 4 | **Prompt Intelligence** | UNO-inspired Rule Extraction, Experience Hints, Meta-Prompting | `phase-3-prompt-intelligence.md` |
| 5 | **Pipeline Laboratory** | Sandboxed `*_lab/` Config-Experimente | `phase-4-pipeline-lab.md` |

**Querschnitt**: Knowledge Distillation (Claude Sonnet → qwen3:4b LoRA, diversitätssensibel) → `phase-5-knowledge-distillation.md`

---

## Phase-Abhängigkeiten

```
Phase 0: Daemon Foundation ──────────────────────┐
    │                                            │
    ▼                                            ▼
Phase 1: Experience Engine              Phase 2: Skill System + CONFIRM-Flow
    │                                            │
    ├────────────────────┐                       │
    ▼                    ▼                       │
Phase 3: Prompt      Phase 5: Knowledge          │
  Intelligence        Distillation (LoRA)        │
    │                    │                       │
    └────────┬───────────┘                       │
             ▼                                   │
Phase 4: Pipeline Lab ◄─────────────────────────┘
```

---

## Key Design Decisions (gelten für ALLE Phasen)

1. **Daemon-Thread, kein separater Prozess** — stirbt mit dem Server
2. **Pre-Emption**: Jeder User-Request stoppt Idle-Skills sofort (`threading.Event`)
3. **Permission-Tiers**: AUTO (read-only) / CONFIRM (SSE-Notification) / MANUAL (User-Befehl)
4. **Safety-Code ist TABU**: `stage_orchestrator.py`, Safety-Configs → NIEMALS agentisch modifizierbar
5. **English-Channel für PI**: Translation pro MediaBox (Schalter existiert), LoRA-Modell nur Englisch
6. **Diversitätssensibles Training**: Kern-Innovation, KEINE kulturelle Hierarchie, WIE-Regeln statt "im Stil von"
7. **Production-Daten only**: Workshop-Runs (`{uuid}_{date}` Device-IDs), KEINE Dev-Testing-Prompts
8. **Platform-Sensibilitaet**: Das System entwickelt narratives Verstaendnis fuer User-Intent (nicht nur Statistik). Session-Analyse identifiziert Modi kreativer Engagement: Identitaetsausdruck, aesthetische Suche, Dekonstruktion, Exploration, Selbst-Artikulation, Grenz-Austestung. Dieses Wissen fliesst in Trashy und den Daemon.

## VRAM-Budget (96 GB RTX 6000 Blackwell)

| Operation | VRAM | Koexistenz mit Workshop (~22 GB) |
|-----------|------|----------------------------------|
| Experience Aggregation | 0 GB | Immer |
| T5-Base Embedding | ~2 GB | Ja |
| QLoRA Training (qwen3:4b) | ~4 GB | Ja |
| LoRA Inference | +0.1 GB | Vernachlässigbar |
| **Peak** | **~28 GB** | **von 96 GB** |

---

## Phase-Pläne (nach Approval zu erstellen)

Jeder Phase-Plan enthält:
- Detaillierte Architektur + Code-Beispiele
- Betroffene Dateien + existierende Patterns die wiederverwendet werden
- Abhängigkeiten + Pre-Conditions
- Verification/Testing-Strategie
- Risiken + Mitigationen

| Datei | Inhalt | Umfang |
|-------|--------|--------|
| `phase-0-daemon-foundation.md` | AgenticDaemon, SystemMonitor, Idle-Detection, 3 AUTO-Skills | ~2 Wochen |
| `phase-1-experience-engine.md` | ExperienceAggregator, Tier A JSON, Session-Aware Narrative Summaries (Platform-Sensibilitaet), Feedback-Signale | ~2-3 Wochen |
| `phase-2-skill-system.md` | Skill-Interface, CONFIRM-Flow via SSE, i18n-Batch, Model-Preload | ~2 Wochen |
| `phase-3-prompt-intelligence.md` | UNO-Lite (Rule Extraction, Clustering, Cognitive Gap), Experience Hints, A/B-Framework | ~3-4 Wochen |
| `phase-4-pipeline-lab.md` | `*_lab/` Ordner, Config-Proposal-Skill, Frontend Lab-Modus | ~4 Wochen |
| `phase-5-knowledge-distillation.md` | Corpus-Generation (Random Node + Claude Sonnet), QLoRA Training (Unsloth), Diversitätssensibles Training, Deployment, A/B-Testing | Forschungs-Phase |

---

## Quellen

### Kernreferenzen
- [UNO: Improve LLM Systems with User Logs (2026)](https://arxiv.org/abs/2602.06470) — Rule Extraction, Cognitive Gap, PEM/REM
- [ICLR 2026 RSI Workshop](https://recursive-workshop.github.io/) — Recursive Self-Improvement Taxonomy
- [Better Self-Improving AI Agents (Nakajima)](https://yoheinakajima.com/better-ways-to-build-self-improving-ai-agents/) — 6 Mechanismen, Skill Libraries

### LoRA/QLoRA
- [Unsloth: Fine-tuning & RL for LLMs](https://github.com/unslothai/unsloth) — 2x faster, 70% less VRAM
- [Qwen3 Fine-Tuning (Unsloth)](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

### Agentic Patterns
- [Agentic AI Infrastructure 2025-2026](https://medium.com/@vinniesmandava/the-agentic-ai-infrastructure-landscape-in-2025-2026-a-strategic-analysis-for-tool-builders-b0da8368aee2)
- [Self-Healing Infrastructure (2026)](https://earezki.com/ai-news/2026-02-23-self-healing-infrastructure-with-agentic-ai-from-monitoring-to-autonomous-resolution/)
- [Meta-Prompting (IntuitionLabs)](https://intuitionlabs.ai/articles/meta-prompting-llm-self-optimization)

---

## Aktionsplan nach Approval

1. Masterplan → `docs/plans/agentic/MASTERPLAN.md` (dieses Dokument)
2. Bisheriges Detail-Dokument → `docs/plans/agentic/RESEARCH_NOTES.md` (UNO + QLoRA Deep Dives)
3. Phase-Pläne → `docs/plans/agentic/phase-{0-5}-*.md` (je ~1-2 Seiten, implementierungsbereit)
4. Memory-Update → Agentic-Architektur-Referenz in MEMORY.md
