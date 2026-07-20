"""Cosine similarity and top-K retrieval over vectors. Pure numpy.

This is the one operation layers 5/6/7/9 all share: image-to-image,
text-to-image and theme-to-image retrieval are the same cosine because CLIP's
vision and text encoders land in the same space.
"""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or a stack of row vectors. Zero vectors stay zero."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n else v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against each row of matrix."""
    q = normalize(query)
    m = normalize(matrix)
    return m @ q


def top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return the k highest-similarity (row_index, score) pairs, best first."""
    if matrix.size == 0 or k <= 0:
        return []
    sims = similarity(query, matrix)
    k = min(k, sims.shape[0])
    # argpartition for the k best, then sort just those.
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]
