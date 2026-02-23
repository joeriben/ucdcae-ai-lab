#!/usr/bin/env python3
"""
Phase 1b: Structured Probing Corpus

Generates ~6250 template-based prompts from probing_specs.py:
- Pillar 1: 15 traditions × 250 prompts = 3750
- Pillar 2: ~2000 material-physical prompts
- Controls: ~500 (baselines + absurd + compositional + minimal pairs)

Appends to existing corpus.json (created by build_corpus.py).

Usage: venv/bin/python research/t5_interpretability/build_probing_corpus.py
"""

import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CORPUS_PATH, DATA_DIR,
    PROMPTS_PER_TEMPLATE, PROMPTS_PER_TRADITION,
    PILLAR1_TOTAL, PILLAR2_TARGET, CONTROLS_TARGET,
)
from probing_specs import (
    TRADITIONS, TraditionSpec,
    EXCITATION_TYPES, MATERIALS, ENVIRONMENTS,
    TIME_PATTERNS, SPECTRAL_AXES, DYNAMICS,
    BASELINE_PROMPTS, ABSURD_PROMPTS,
    MINIMAL_PAIR_SEEDS, COMPOSITIONAL_TEMPLATES,
)


def _sample_n(items: list, n: int) -> list:
    """Sample n items from list, cycling if n > len(items)."""
    if not items:
        return []
    result = []
    while len(result) < n:
        remaining = n - len(result)
        if remaining >= len(items):
            result.extend(items)
        else:
            result.extend(random.sample(items, remaining))
    return result[:n]


def generate_tradition_prompts(spec: TraditionSpec) -> list[dict]:
    """Generate 250 prompts for one tradition (5 templates × 50 each)."""
    entries = []
    label = random.choice(spec.region_labels)

    # Template 1: Instruments (50)
    instruments = _sample_n(spec.instruments, PROMPTS_PER_TEMPLATE)
    acoustic_descriptors = [
        "a slow melody", "rapid passages", "a sustained drone",
        "rhythmic ostinato", "gentle arpeggios", "powerful crescendo",
        "a soft lullaby", "virtuosic improvisation", "a ceremonial theme",
        "syncopated patterns", "delicate ornaments", "deep resonant tones",
        "high shimmering harmonics", "a melancholic phrase", "bright staccato notes",
        "a meditative drone", "rapid trills", "sweeping glissando",
        "a dance rhythm", "expressive vibrato", "percussive accents",
        "flowing legato lines", "a call to prayer melody", "whispering overtones",
        "thunderous bass notes", "playful melodic figures", "a solemn processional",
        "intricate filigree", "bold fanfare", "quiet contemplation",
        "cascading runs", "sustained chords", "rhythmic drive",
        "mournful lament", "celebratory flourish", "mysterious murmur",
        "pulsing rhythm", "floating melody", "grounded bass",
        "sharp attack", "smooth decay", "building intensity",
        "fading echo", "layered texture", "sparse simplicity",
        "complex counterpoint", "unison statement", "solo cadenza",
        "gentle tremolo", "powerful climax",
    ]
    for i in range(PROMPTS_PER_TEMPLATE):
        entries.append({
            "text": f"{label} {instruments[i]} playing {acoustic_descriptors[i % len(acoustic_descriptors)]}",
            "source": "probing",
            "category": "pillar1_instrument",
            "subcategory": spec.name,
        })

    # Template 2: Ensemble (50)
    ensembles = _sample_n(spec.ensemble_contexts, PROMPTS_PER_TEMPLATE)
    performance_contexts = [
        "a festive celebration", "a solemn ceremony", "a dance performance",
        "a meditation session", "a storytelling gathering", "a wedding procession",
        "a harvest festival", "a funeral rite", "a seasonal ritual",
        "a court entertainment", "a street performance", "a religious service",
        "an intimate gathering", "a competition", "a communal work session",
        "a children's game", "a healing ceremony", "a coming-of-age ritual",
        "a dawn greeting", "an evening farewell", "a birth celebration",
        "a peace ceremony", "a remembrance gathering", "a initiation rite",
        "a marketplace performance", "a family reunion", "a pilgrimage song",
        "a battle preparation", "a victory celebration", "a prayer service",
        "a nature worship", "a moon ceremony", "a fire ritual",
        "a water blessing", "a mountain offering", "a ancestral communion",
        "a spring welcoming", "a winter solstice", "a rain invocation",
        "a thanksgiving feast", "a naming ceremony", "a oath-taking",
        "a reconciliation", "a farewell journey", "a homecoming",
        "an improvised session", "a rehearsal", "a teaching demonstration",
        "a spontaneous jam", "a formal recital",
    ]
    for i in range(PROMPTS_PER_TEMPLATE):
        entries.append({
            "text": f"{ensembles[i % len(ensembles)]} performing {performance_contexts[i % len(performance_contexts)]}",
            "source": "probing",
            "category": "pillar1_ensemble",
            "subcategory": spec.name,
        })

    # Template 3: Vocal (50)
    vocals = _sample_n(spec.vocal_styles, PROMPTS_PER_TEMPLATE)
    vocal_descriptors = [
        "with intense emotional expression", "in a quiet intimate setting",
        "building from whisper to full voice", "with elaborate ornamentation",
        "in slow contemplative tempo", "with rhythmic precision",
        "expressing deep sorrow", "radiating joyful energy",
        "in call-and-response form", "with haunting beauty",
        "projecting across open space", "in hushed reverential tone",
        "with virtuosic technique", "in natural untrained voice",
        "with powerful dramatic projection", "in gentle soothing quality",
        "with rapid syllabic delivery", "in sustained lyrical phrases",
        "with microtonal inflections", "expressing longing and desire",
        "in ceremonial solemnity", "with playful humor",
        "in narrative storytelling mode", "with ecstatic devotion",
        "expressing defiance and strength", "in lullaby gentleness",
        "with raspy textured timbre", "in clear bell-like tone",
        "with chest voice resonance", "in head voice clarity",
        "expressing gratitude", "with sorrowful mourning",
        "in triumphant celebration", "with meditative calm",
        "expressing ancestral wisdom", "in youthful exuberance",
        "with aged gravelly texture", "in crystalline purity",
        "expressing communal unity", "with individual distinctiveness",
        "in harmonic blend with ensemble", "with solo prominence",
        "expressing spiritual transcendence", "in earthly groundedness",
        "with dynamic contrast", "in steady unwavering tone",
        "expressing urgency", "in relaxed unhurried pace",
        "with dramatic crescendo", "in gentle diminuendo",
    ]
    for i in range(PROMPTS_PER_TEMPLATE):
        entries.append({
            "text": f"{label} {vocals[i % len(vocals)]} {vocal_descriptors[i % len(vocal_descriptors)]}",
            "source": "probing",
            "category": "pillar1_vocal",
            "subcategory": spec.name,
        })

    # Template 4: Context (50)
    spaces = _sample_n(spec.spatial_contexts, PROMPTS_PER_TEMPLATE)
    for i in range(PROMPTS_PER_TEMPLATE):
        entries.append({
            "text": f"{label} music performed in {spaces[i % len(spaces)]}",
            "source": "probing",
            "category": "pillar1_context",
            "subcategory": spec.name,
        })

    # Template 5: Blind descriptors (50)
    blinds = _sample_n(spec.blind_descriptors, PROMPTS_PER_TEMPLATE)
    for i in range(PROMPTS_PER_TEMPLATE):
        entries.append({
            "text": blinds[i],
            "source": "probing",
            "category": "pillar1_blind",
            "subcategory": spec.name,
        })

    assert len(entries) == PROMPTS_PER_TRADITION, f"{spec.name}: expected {PROMPTS_PER_TRADITION}, got {len(entries)}"
    return entries


def generate_pillar2_prompts() -> list[dict]:
    """Generate ~2000 material-physical prompts from Pillar 2 vocabulary."""
    entries = []

    # A. Excitation types: 6 categories, target ~400 total (~67 each)
    for exc_type, examples in EXCITATION_TYPES.items():
        per_type = 67 if exc_type != "vocal" else 65  # adjust for total
        sampled = _sample_n(examples, per_type)
        for text in sampled:
            entries.append({
                "text": text,
                "source": "probing",
                "category": "pillar2_excitation",
                "subcategory": exc_type,
            })

    # B. Materials: 10 categories × 30 each = 300
    for mat_name, examples in MATERIALS.items():
        sampled = _sample_n(examples, 30)
        for text in sampled:
            entries.append({
                "text": text,
                "source": "probing",
                "category": "pillar2_material",
                "subcategory": mat_name,
            })

    # C. Environments: 8 categories × ~38 each = ~304
    for env_name, examples in ENVIRONMENTS.items():
        sampled = _sample_n(examples, 38)
        for text in sampled:
            entries.append({
                "text": text,
                "source": "probing",
                "category": "pillar2_environment",
                "subcategory": env_name,
            })

    # D. Time patterns: 6 categories × 50 each = 300
    for pattern_name, examples in TIME_PATTERNS.items():
        sampled = _sample_n(examples, 50)
        for text in sampled:
            entries.append({
                "text": text,
                "source": "probing",
                "category": "pillar2_time",
                "subcategory": pattern_name,
            })

    # E. Spectral: 4 axes × 75 each = 300
    for axis_name, gradient_steps in SPECTRAL_AXES.items():
        sampled = _sample_n(gradient_steps, 75)
        for text in sampled:
            entries.append({
                "text": text,
                "source": "probing",
                "category": "pillar2_spectral",
                "subcategory": axis_name,
            })

    # F. Dynamics: ~300 from 15 descriptors
    sampled = _sample_n(DYNAMICS, 300)
    for text in sampled:
        entries.append({
            "text": text,
            "source": "probing",
            "category": "pillar2_dynamics",
            "subcategory": "dynamics",
        })

    logger.info(f"Pillar 2: {len(entries)} prompts (target ~{PILLAR2_TARGET})")
    return entries


def generate_control_prompts() -> list[dict]:
    """Generate ~500 structural control prompts."""
    entries = []

    # Baselines (~100)
    for text in BASELINE_PROMPTS:
        entries.append({
            "text": text,
            "source": "probing",
            "category": "control_baseline",
            "subcategory": "baseline",
        })

    # Absurd/impossible (~100, reuse 50 × 2 if needed)
    absurd_sampled = _sample_n(ABSURD_PROMPTS, 100)
    for text in absurd_sampled:
        entries.append({
            "text": text,
            "source": "probing",
            "category": "control_absurd",
            "subcategory": "absurd",
        })

    # Minimal pairs (~100 = 50 pairs × 2)
    for pair_a, pair_b in MINIMAL_PAIR_SEEDS:
        entries.append({
            "text": pair_a,
            "source": "probing",
            "category": "control_minimal_pair",
            "subcategory": "minimal_pair_a",
        })
        entries.append({
            "text": pair_b,
            "source": "probing",
            "category": "control_minimal_pair",
            "subcategory": "minimal_pair_b",
        })

    # Compositional (~200): fill from templates with random vocab
    all_excitation = [ex for exs in EXCITATION_TYPES.values() for ex in exs]
    all_materials_flat = [m for ms in MATERIALS.values() for m in ms]
    all_environments_flat = [e for envs in ENVIRONMENTS.values() for e in envs]
    all_spectral = [s for ss in SPECTRAL_AXES.values() for s in ss]
    all_time = [t for ts in TIME_PATTERNS.values() for t in ts]
    material_names = list(MATERIALS.keys())
    tradition_names = [t.name for t in TRADITIONS]

    for _ in range(200):
        template = random.choice(COMPOSITIONAL_TEMPLATES)
        text = template.format(
            excitation=random.choice(all_excitation),
            tradition=random.choice(tradition_names),
            material=random.choice(all_materials_flat),
            environment=random.choice(all_environments_flat),
            spectral=random.choice(all_spectral),
            dynamic=random.choice(DYNAMICS),
            time_pattern=random.choice(all_time),
            material_name=random.choice(material_names),
        )
        entries.append({
            "text": text,
            "source": "probing",
            "category": "control_compositional",
            "subcategory": "compositional",
        })

    logger.info(f"Controls: {len(entries)} prompts (target ~{CONTROLS_TARGET})")
    return entries


def main():
    random.seed(42)  # reproducible corpus
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing corpus (from build_corpus.py)
    if CORPUS_PATH.exists():
        with open(CORPUS_PATH) as f:
            corpus = json.load(f)
        logger.info(f"Loaded existing corpus: {len(corpus)} entries")
    else:
        corpus = []
        logger.info("No existing corpus found, starting fresh")

    # Count existing probing entries to avoid double-appending
    existing_probing = sum(1 for e in corpus if e.get("source") == "probing")
    if existing_probing > 0:
        logger.warning(f"Found {existing_probing} existing probing entries — removing before regenerating")
        corpus = [e for e in corpus if e.get("source") != "probing"]

    # Generate Pillar 1: 15 traditions × 250
    pillar1 = []
    for spec in TRADITIONS:
        tradition_prompts = generate_tradition_prompts(spec)
        pillar1.extend(tradition_prompts)
        logger.info(f"  {spec.name}: {len(tradition_prompts)} prompts")

    assert len(pillar1) == PILLAR1_TOTAL, f"Pillar 1: expected {PILLAR1_TOTAL}, got {len(pillar1)}"
    logger.info(f"Pillar 1 total: {len(pillar1)}")

    # Generate Pillar 2
    pillar2 = generate_pillar2_prompts()

    # Generate Controls
    controls = generate_control_prompts()

    # Combine
    probing = pillar1 + pillar2 + controls
    logger.info(f"Total probing prompts: {len(probing)}")

    corpus.extend(probing)

    # Deduplicate probing entries (text-level, within probing only)
    seen_texts = set()
    deduped = []
    for entry in corpus:
        key = entry["text"].strip().lower()
        if key not in seen_texts:
            seen_texts.add(key)
            deduped.append(entry)
    removed = len(corpus) - len(deduped)
    if removed:
        logger.info(f"Removed {removed} duplicate probing entries")
    corpus = deduped

    # Save
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=None)

    # Print distribution
    categories = {}
    for entry in corpus:
        cat = entry["category"]
        categories[cat] = categories.get(cat, 0) + 1

    logger.info(f"\nFinal corpus: {len(corpus)} entries")
    logger.info("Distribution:")
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
