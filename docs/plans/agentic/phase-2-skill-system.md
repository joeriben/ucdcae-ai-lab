# Phase 2: Skill System + CONFIRM-Flow

**Abhaengigkeiten**: Phase 0 (Daemon)
**Geschaetzter Umfang**: ~2 Wochen

---

## Ziel

Vollstaendiges Skill-Interface mit AUTO/CONFIRM/MANUAL Tiers und SSE-basiertem CONFIRM-Flow fuer User-Approval.

## Permission-Tiers

```
AUTO     = Read-only, keine Seiteneffekte. Laeuft ohne Rueckfrage.
CONFIRM  = Hat Seiteneffekte. Daemon loggt Absicht, wartet auf User-Approval via SSE.
MANUAL   = Nur durch expliziten User-Befehl (REST-Endpoint oder Frontend-Button).
```

## Skill-Interface

```python
class Skill(ABC):
    name: str
    permission: Literal["AUTO", "CONFIRM", "MANUAL"]
    description: str  # Fuer Frontend-Anzeige

    @abstractmethod
    def should_run(self, ctx: DaemonContext) -> bool: ...

    @abstractmethod
    def execute(self, ctx: DaemonContext, stop_event: threading.Event) -> SkillResult: ...

    def estimate_resources(self) -> dict:
        return {"vram_gb": 0, "disk_gb": 0, "duration_min": 0}
```

## CONFIRM-Flow

### Via SSE (Browser verbunden)

```
1. Daemon: skill.should_run() -> True
2. Daemon: Speichert PendingSkill in pending_skills.json
3. Daemon: Sendet SSE Event: {"type": "skill_confirm", "skill": "run_i18n_batch", "reason": "3 Work Orders pending"}
4. Frontend: Zeigt Notification mit Approve/Deny
5. User: Klickt Approve
6. Frontend: POST /api/daemon/skills/{skill_name}/approve
7. Daemon: Fuehrt Skill aus
```

### Headless (kein Browser)

```
1. Daemon: Speichert in pending_skills.json
2. User: Beim naechsten Login -> Frontend liest pending_skills.json
3. User: Approved/Denied batch
```

## Neue CONFIRM-Skills

### `preload_models`
- **Trigger**: Konfigurierte Workshop-Zeit (z.B. "Mo-Fr 08:00" in daemon_config.json)
- **Aktion**: `POST /api/diffusers/preload` + Ollama model pull
- **Ressourcen**: VRAM-intensiv, deshalb CONFIRM

### `run_i18n_batch`
- **Trigger**: `src/i18n/WORK_ORDERS.md` hat Eintraege + Idle
- **Aktion**: i18n-Translator-Agent starten (`.claude/agents/i18n-translator.md`)
- **Implementierung**: Subprocess `6_run_i18n_translator.sh --unattended`
- **Ressourcen**: LLM-Zeit, kein VRAM (nutzt Ollama)

## Daemon-Config (`exports/experience/daemon_config.json`)

```json
{
  "workshop_schedule": {
    "preload_time": "08:00",
    "preload_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
    "preload_models": ["sd35_large", "qwen3:4b"]
  },
  "skill_overrides": {
    "run_i18n_batch": "AUTO"
  }
}
```

## REST-Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/daemon/status` | GET | Health + Skills + Idle |
| `/api/daemon/skills` | GET | Alle registrierten Skills |
| `/api/daemon/skills/{name}/trigger` | POST | Manuell ausloesen |
| `/api/daemon/skills/{name}/approve` | POST | CONFIRM-Skill freigeben |
| `/api/daemon/skills/{name}/deny` | POST | CONFIRM-Skill ablehnen |
| `/api/daemon/pending` | GET | Pending CONFIRM-Skills |

## Betroffene Dateien

| Datei | Aktion |
|-------|--------|
| `devserver/my_app/skills/base.py` | EDIT: SkillResult, DaemonContext erweitern |
| `devserver/my_app/skills/preload_models.py` | NEU |
| `devserver/my_app/skills/run_i18n_batch.py` | NEU |
| `devserver/my_app/routes/daemon_routes.py` | NEU: REST-Endpoints |
| `devserver/my_app/routes/sse_routes.py` | EDIT: skill_confirm Events |
| Frontend: Notification-Overlay | NEU: Vue-Komponente |

## Verification

1. WORK_ORDERS.md mit Eintraegen -> Daemon erkennt, sendet SSE
2. Frontend zeigt "i18n batch bereit (3 Work Orders)" Notification
3. User klickt Approve -> Agent startet
4. `curl localhost:17802/api/daemon/pending` -> Liste der wartenden Skills
5. `curl -X POST localhost:17802/api/daemon/skills/preload_models/trigger` -> Manueller Trigger
