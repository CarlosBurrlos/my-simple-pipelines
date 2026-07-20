"""Config and paths for the catalog pipeline.

This is the module allowed to know domain names — "etsy", "thumb", palette
names, label vocabularies. It reads config/*.yaml and hands plain values to the
packages/. The rule (CLAUDE.md §3) runs one way: pipelines may import packages
and know names; packages may never import pipelines or learn a name.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

# repo root: .../pipelines/catalog/config.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"

DB_PATH = DATA_DIR / "catalog.db"
LOGS_DIR = DATA_DIR / "logs"
INCOMING_DIR = DATA_DIR / "incoming"      # raw uploads land here
ORIGINALS_DIR = DATA_DIR / "originals"    # normalized masters
DERIVATIVES_DIR = DATA_DIR / "derivatives"

# Quarantine anything whose short side is under this — never upscale silently.
MIN_SHORT_SIDE = 2000


def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    with path.open() as f:
        return yaml.safe_load(f) or {}


@functools.lru_cache(maxsize=1)
def profiles() -> dict[str, dict]:
    """Derivative profiles keyed by name (etsy/web/preview/thumb)."""
    return _load_yaml("profiles.yaml")


@functools.lru_cache(maxsize=1)
def palette_pairs() -> dict[str, tuple[int, int, int]]:
    """name -> (r, g, b) from palette.yaml, ready for tagging.Palette.from_pairs."""
    raw = _load_yaml("palette.yaml").get("colors", {})
    return {name: tuple(rgb) for name, rgb in raw.items()}


@dataclass(frozen=True)
class LabelGroup:
    key: str
    template: str
    threshold: float
    values: list[str]

    def prompt(self, value: str) -> str:
        return self.template.format(value)


@functools.lru_cache(maxsize=1)
def label_groups() -> list[LabelGroup]:
    """CLIP zero-shot candidate groups for attribute tagging (layer 6)."""
    raw = _load_yaml("labels.yaml").get("groups", {})
    out = []
    for key, spec in raw.items():
        out.append(
            LabelGroup(
                key=key,
                template=spec.get("template", "a photo of {}"),
                threshold=float(spec.get("threshold", 0.2)),
                values=list(spec.get("values", [])),
            )
        )
    return out


@functools.lru_cache(maxsize=1)
def themes() -> list[dict]:
    """Themes for scoring — empty in v1 by design (CLAUDE.md §5)."""
    return _load_yaml("themes.yaml").get("themes", []) or []


def ensure_data_dirs() -> None:
    for d in (DATA_DIR, LOGS_DIR, INCOMING_DIR, ORIGINALS_DIR, DERIVATIVES_DIR):
        d.mkdir(parents=True, exist_ok=True)
