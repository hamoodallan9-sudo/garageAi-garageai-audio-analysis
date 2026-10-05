import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "data" / "car_faults_glossary.md"
SEVERITY_GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "data" / "severity_glossary.md"

# Maps a keyword found in a "## N) ..." heading to the audio ensemble's class label
# (see app/core/config.py MODELS_DIR / label_encoder.pkl -> 01_sway_bar_bushing,
# 02_serpentine_belt, 03_brake_squeal).
CATEGORY_KEY_BY_KEYWORD = [
    ("دينمو", "02_serpentine_belt"),
    ("دايمو", "02_serpentine_belt"),
    ("بريك", "03_brake_squeal"),
    ("فرامل", "03_brake_squeal"),
    ("ميزان", "01_sway_bar_bushing"),
    ("توازن", "01_sway_bar_bushing"),
]


@dataclass(frozen=True)
class Chunk:
    category_key: str
    category_label: str
    section: str
    text: str


def _match_category_key(heading: str) -> str:
    for keyword, key in CATEGORY_KEY_BY_KEYWORD:
        if keyword in heading:
            return key
    raise ValueError(f"Could not map category heading to a known fault class: {heading!r}")


def parse_glossary(path: Path = GLOSSARY_PATH) -> List[Chunk]:
    text = path.read_text(encoding="utf-8")
    chunks: List[Chunk] = []

    category_label = None
    category_key = None
    section = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        h2 = re.match(r"^##\s+\d+\)\s*(.+)$", line)
        if h2:
            category_label = h2.group(1).strip()
            category_key = _match_category_key(category_label)
            section = None
            continue

        h3 = re.match(r"^###\s+(.+)$", line)
        if h3:
            section = h3.group(1).strip()
            continue

        bullet = re.match(r"^-\s+(.+)$", line)
        if bullet and category_key and category_label and section:
            chunks.append(Chunk(category_key, category_label, section, bullet.group(1).strip()))

    return chunks


def category_context(category_key: str, path: Path = GLOSSARY_PATH) -> str:
    """Full cause/sound/symptom text for one category, for grounding the LLM's answer."""
    lines = [f"{c.section}: {c.text}" for c in parse_glossary(path) if c.category_key == category_key]
    return "\n".join(lines)


def severity_context(category_key: str, path: Path = SEVERITY_GLOSSARY_PATH) -> str:
    """Data-mined severity signals + follow-up questions for one category (see
    app/data/severity_glossary.md), for grounding the LLM's severity assessment."""
    lines = [f"{c.section}: {c.text}" for c in parse_glossary(path) if c.category_key == category_key]
    return "\n".join(lines)


def category_labels(path: Path = GLOSSARY_PATH) -> dict:
    """category_key -> human-readable heading text, e.g. for grounding on a category_key
    that came from elsewhere (like the audio classifier) rather than from parsing itself."""
    labels: dict = {}
    for chunk in parse_glossary(path):
        labels.setdefault(chunk.category_key, chunk.category_label)
    return labels
