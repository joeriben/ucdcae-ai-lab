# Phase 0: Daemon Foundation

**Abhaengigkeiten**: Keine (Startpunkt)
**Geschaetzter Umfang**: ~2 Wochen
**Blocked by**: Safety Regression Fix (muss vorher abgeschlossen sein)

---

## Ziel

AgenticDaemon als Daemon-Thread im DevServer: 30s-Tick, Health-Checks, Idle-Detection, Pre-Emption. Grundgeruest fuer alle weiteren Phasen.

## Komponenten

### 1. AgenticDaemon (`devserver/my_app/services/agentic_daemon.py`)

```python
class AgenticDaemon:
    def __init__(self, app):
        self.monitor = SystemMonitor()
        self.skills = SkillRegistry()
        self._idle = False
        self._stop_event = threading.Event()
        self._tick_interval = 30  # seconds

    def start(self):
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                self._log("error", f"Tick failed: {e}")
            self._stop_event.wait(self._tick_interval)

    def _tick(self):
        status = self.monitor.check_all()
        self._idle = self._detect_idle(status)
        self.skills.run_due_skills(status, self._idle, self._stop_event)
```

### 2. SystemMonitor (`devserver/my_app/services/system_monitor.py`)

Checks (alle 30s):
| Check | Quelle | Implementierung |
|-------|--------|-----------------|
| GPU Service | `GET localhost:17803/api/health` | requests.get, 5s timeout |
| Ollama | `GET localhost:11434/api/tags` | requests.get, 3s timeout |
| VRAM | GPU Service `/api/health/vram` | Parse VRAMCoordinator status |
| Disk | `os.statvfs(exports_path)` | Warnung <10GB |
| Active Users | Import `user_activity` aus `sse_routes.py` | 5min Timeout |

### 3. Idle Detection

```python
def _detect_idle(self, status):
    return (
        status.active_users == 0
        and status.ollama_semaphore_free  # Keine laufenden LLM-Calls
        and status.gpu_models_in_use == 0  # VRAMCoordinator refcount
        and status.seconds_since_last_activity > 120
    )
```

### 4. Pre-Emption

Jeder HTTP-Request setzt `_idle = False`. Skills pruefen `stop_event.is_set()` in ihren Loops.

Implementierung: Flask `before_request` Hook:
```python
@app.before_request
def notify_daemon():
    if hasattr(app, 'agentic_daemon'):
        app.agentic_daemon.on_user_activity()
```

### 5. Drei AUTO-Skills (Phase 0)

| Skill | Trigger | Aktion |
|-------|---------|--------|
| `health_check` | Alle 30s | Loggt Status, Warnung bei Problemen |
| `validate_configs` | Idle + taeglich | ConfigLoader.validate() ausfuehren |
| `cleanup_vram` | VRAM >80% | `POST /api/{backend}/unload` fuer aelteste Modelle |

### 6. Logging

```
exports/experience/daemon_log.jsonl
```

Format: `{"timestamp": "...", "level": "info|warn|error", "skill": "health_check", "message": "...", "data": {...}}`

### 7. REST-Endpoint

`GET /api/daemon/status` -> Aktueller Status, letzte Skill-Ausfuehrungen, Health-History.

## Betroffene Dateien

| Datei | Aktion |
|-------|--------|
| `devserver/my_app/services/agentic_daemon.py` | NEU |
| `devserver/my_app/services/system_monitor.py` | NEU |
| `devserver/my_app/skills/` | NEU (Ordner + base.py + 3 Skills) |
| `devserver/my_app/__init__.py` | Daemon starten bei App-Init |
| `devserver/my_app/routes/` | Neuer Blueprint fuer `/api/daemon/` |

## Bestehende Patterns die wiederverwendet werden

- `user_activity` Dict aus `sse_routes.py:43-68`
- `ModelAvailabilityService` Health-Checks aus `model_availability_service.py`
- `threading.Thread(daemon=True)` Pattern aus `training_service.py:659`
- JSONL-Logging Pattern (neu, aber simpel)

## Verification

1. Server starten -> `daemon_log.jsonl` erscheint mit Health-Checks
2. `curl localhost:17802/api/daemon/status` -> JSON mit Health + Idle-Status
3. GPU Service stoppen -> Daemon loggt Warnung
4. Generation ausfuehren -> Daemon idle=False waehrend Ausfuehrung
5. 2 min warten -> Daemon idle=True, validate_configs laeuft
