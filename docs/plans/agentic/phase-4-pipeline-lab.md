# Phase 4: Pipeline Laboratory

**Abhaengigkeiten**: Phase 1 + Phase 3 (Erfahrungsdaten + Regeln)
**Geschaetzter Umfang**: ~4 Wochen

---

## Ziel

Sandboxed Ordner fuer Agent-generierte Interception-Config-Experimente. CONFIRM-basierter Workflow: Agent schlaegt vor, User reviewed, approved Configs wandern in Produktion.

## Ordner-Struktur

```
devserver/schemas/
  configs/interception/          # Produktion (42+, unveraendert)
  configs/interception_lab/      # NEU: Agent-generierte Experimente
```

`interception_lab/` wird vom ConfigLoader SEPARAT geladen. Configs erhalten `"experimental": true` Flag.

## Komponenten

### 1. ConfigLoader-Erweiterung

```python
# config_loader.py: Separates Laden von Lab-Configs
def _load_lab_configs(self):
    lab_path = Path("schemas/configs/interception_lab")
    for config_file in lab_path.glob("*.json"):
        config = json.load(config_file)
        config["experimental"] = True
        self._configs[config["id"]] = config
```

### 2. Config-Generator (`devserver/my_app/services/config_generator.py`)

```python
class ConfigGenerator:
    def propose_config(self, experience_data, stop_event):
        """LLM generiert Config-Vorschlag basierend auf Erfahrungsdaten."""
        # 1. Identify opportunity (Muster in Erfahrungsdaten)
        opportunity = self._find_opportunity(experience_data)
        # 2. Load 3 aehnliche existierende Configs als Template
        templates = self._load_similar_configs(opportunity)
        # 3. LLM generiert neuen Config-Entwurf
        config = self._generate_config(opportunity, templates)
        # 4. Validierung (Schema, Pflichtfelder, keine Safety-Konflikte)
        self._validate(config)
        # 5. Speichern in interception_lab/
        self._save_to_lab(config)
        return config
```

### 3. `propose_config` Skill (CONFIRM)

```python
class ProposeConfigSkill(Skill):
    name = "propose_config"
    permission = "CONFIRM"

    def should_run(self, ctx):
        return (ctx.is_idle
                and self._has_enough_experience(min_runs=500)
                and self._days_since_last_proposal() >= 7)
```

### 4. Lab-Modus im Frontend

- Experimentelle Configs erscheinen in separater Sektion mit "Lab"-Badge
- User kann Lab-Config testen (normaler Generation-Flow)
- "Promote to Production" Button -> Move von `interception_lab/` zu `interception/`

### 5. Opportunity Detection

Aus `config_performance.json` + `prompt_patterns.json`:
- "Viele Prompts zu Thema X, aber keine spezifische Config"
- "Config Y hat <5% Favoriten-Rate, Kontext koennte verbessert werden"
- "Cluster Z in Embeddings hat keine zugeordnete Config"

## Betroffene Dateien

| Datei | Aktion |
|-------|--------|
| `devserver/schemas/configs/interception_lab/` | NEU (Ordner) |
| `devserver/my_app/services/config_generator.py` | NEU |
| `devserver/my_app/skills/propose_config.py` | NEU |
| `devserver/schemas/engine/config_loader.py` | EDIT: Lab-Configs laden |
| Frontend: Config-Selektor | EDIT: Lab-Sektion + Badge |
| Frontend: Promote-Button | NEU |

## Verification

1. `propose_config` Skill manuell triggern -> JSON in `interception_lab/`
2. Config validiert (Schema-konform, Pflichtfelder vorhanden)
3. Frontend zeigt Lab-Config mit Badge
4. Generation mit Lab-Config funktioniert
5. "Promote" -> Config in `interception/`, verschwindet aus Lab
