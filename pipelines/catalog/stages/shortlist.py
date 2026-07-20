"""Layer 9 — shortlist.

Catalog-wide packing can't ship every image to the model. Retrieve a candidate
set cheaply, then spend the model only on that. The candidate set is a union of:
  - top-K by image vector  (visual/tonal neighbours)
  - top-K by text vector    (functional neighbours: the 'desk setup' case)
  - tag overlap             (plain SQL over tags)
  - pack centroids          (skipped until packs exist)

Tune for recall, generously — anything dropped here the model never sees. The
pure retrieval math is in tagging.cosine; this module joins it to the ledger.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from tagging import cosine
from pipelines.catalog.stages import write


def _neighbours(target_id: str, ids: list[str], matrix: np.ndarray, k: int) -> list[str]:
    """Top-k product ids by cosine to target's own row, excluding target."""
    if target_id not in ids or matrix.size == 0:
        return []
    ti = ids.index(target_id)
    ranked = cosine.top_k(matrix[ti], matrix, k + 1)  # +1 because target ranks itself first
    return [ids[i] for i, _ in ranked if ids[i] != target_id][:k]


def tag_overlap(conn: sqlite3.Connection, target_id: str) -> set[str]:
    """Products sharing at least one (key, value) tag with the target."""
    rows = conn.execute(
        """
        SELECT DISTINCT t2.product_id
        FROM tags t1
        JOIN tags t2 ON t1.key = t2.key AND t1.value = t2.value
        WHERE t1.product_id = ? AND t2.product_id != ?
        """,
        (target_id, target_id),
    ).fetchall()
    return {r["product_id"] for r in rows}


def shortlist_for(
    conn: sqlite3.Connection,
    target_id: str,
    model: str,
    k: int = 20,
) -> set[str]:
    """The union candidate set for one target product (excludes the target)."""
    img_ids, img_mat = write.load_embeddings(conn, "image", model)
    txt_ids, txt_mat = write.load_embeddings(conn, "text", model)

    candidates: set[str] = set()
    candidates.update(_neighbours(target_id, img_ids, img_mat, k))
    candidates.update(_neighbours(target_id, txt_ids, txt_mat, k))
    candidates.update(tag_overlap(conn, target_id))
    candidates.discard(target_id)
    return candidates


def shortlist_catalog(conn: sqlite3.Connection, model: str, k: int = 20) -> dict[str, set[str]]:
    """Shortlist every product against the rest. Packs are proposed catalog-wide."""
    ids = [r["id"] for r in conn.execute("SELECT id FROM products ORDER BY id")]
    return {pid: shortlist_for(conn, pid, model, k) for pid in ids}
