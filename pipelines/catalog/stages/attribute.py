"""Layer 6 — attribute tagging via CLIP zero-shot.

Each label group (from config) becomes a matrix of text-prompt embeddings. The
product's image vector is scored against them; the argmax per group wins if it
clears the group's threshold. Tags are 'clip'-sourced. Scores are NOT calibrated
(CLAUDE.md §10) — the thresholds are coarse guards, not probabilities.

The scoring itself (tag_attributes) is pure numpy and takes precomputed label
matrices, so it tests without loading CLIP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tagging import cosine


@dataclass(frozen=True)
class AttrTag:
    key: str
    value: str
    source: str         # always 'clip'
    confidence: float


@dataclass(frozen=True)
class LabelIndex:
    key: str
    values: list[str]
    matrix: np.ndarray  # (n_values, dim) text embeddings, row-aligned to values
    threshold: float


def build_label_index(encoder, groups) -> list[LabelIndex]:
    """Embed every group's prompts once. groups are config.LabelGroup instances."""
    out = []
    for g in groups:
        prompts = [g.prompt(v) for v in g.values]
        matrix = encoder.embed_text(prompts)
        out.append(LabelIndex(key=g.key, values=list(g.values), matrix=matrix, threshold=g.threshold))
    return out


def tag_attributes(image_vec: np.ndarray, indexes: list[LabelIndex]) -> list[AttrTag]:
    """Best label per group that clears threshold, as 'clip'-sourced tags."""
    tags = []
    for idx in indexes:
        if idx.matrix.size == 0:
            continue
        sims = cosine.similarity(image_vec, idx.matrix)
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score >= idx.threshold:
            tags.append(
                AttrTag(key=idx.key, value=idx.values[best], source="clip", confidence=score)
            )
    return tags
