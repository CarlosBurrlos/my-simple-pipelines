"""Layer 11 — ledger writes.

The single place the pipeline turns stage results into rows. Vectors are stored
as raw numpy bytes with their dtype/dim/model recorded, because vectors from
different checkpoints are not comparable and must never be silently mixed.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from ledger import new_id
from pipelines.catalog.stages.attribute import AttrTag
from pipelines.catalog.stages.color import ColorTag
from pipelines.catalog.stages.derive import DerivativeResult
from pipelines.catalog.stages.intake import IntakeResult


def insert_product(
    conn: sqlite3.Connection, display_name: str | None = None, external_ref: str | None = None
) -> str:
    pid = new_id()
    conn.execute(
        "INSERT INTO products (id, external_ref, display_name) VALUES (?, ?, ?)",
        (pid, external_ref, display_name),
    )
    return pid


def insert_asset(conn: sqlite3.Connection, r: IntakeResult) -> str:
    conn.execute(
        "INSERT INTO assets (id, product_id, source_path, width, height, short_side, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (r.asset_id, r.product_id, r.source_path, r.width, r.height, r.short_side, r.status),
    )
    return r.asset_id


def insert_derivatives(conn: sqlite3.Connection, asset_id: str, rows: list[DerivativeResult]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO derivatives "
        "(id, asset_id, profile, path, format, width, height, bytes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(new_id(), asset_id, d.profile, d.path, d.format, d.width, d.height, d.bytes) for d in rows],
    )


def insert_tag(
    conn: sqlite3.Connection,
    product_id: str,
    key: str,
    value: str,
    source: str,
    confidence: float | None = None,
) -> str:
    tid = new_id()
    conn.execute(
        "INSERT INTO tags (id, product_id, key, value, source, confidence) VALUES (?, ?, ?, ?, ?, ?)",
        (tid, product_id, key, value, source, confidence),
    )
    return tid


def insert_color_tag(conn: sqlite3.Connection, product_id: str, tag: ColorTag) -> str:
    return insert_tag(conn, product_id, tag.key, tag.value, tag.source, tag.confidence)


def insert_attr_tags(conn: sqlite3.Connection, product_id: str, tags: list[AttrTag]) -> None:
    for t in tags:
        insert_tag(conn, product_id, t.key, t.value, t.source, t.confidence)


def vector_to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim).copy()


def insert_embedding(
    conn: sqlite3.Connection,
    product_id: str,
    kind: str,
    model: str,
    vector: np.ndarray,
    asset_id: str | None = None,
) -> str:
    eid = new_id()
    vec = np.asarray(vector, dtype=np.float32).ravel()
    conn.execute(
        "INSERT INTO embeddings (id, product_id, asset_id, kind, model, dim, dtype, vector) "
        "VALUES (?, ?, ?, ?, ?, ?, 'float32', ?)",
        (eid, product_id, asset_id, kind, model, vec.shape[0], vector_to_blob(vec)),
    )
    return eid


def set_hero(conn: sqlite3.Connection, product_id: str, asset_id: str) -> None:
    conn.execute("UPDATE products SET hero_asset_id = ? WHERE id = ?", (asset_id, product_id))


def insert_pack(
    conn: sqlite3.Connection, title: str, rationale: str, member_ids: list[str], status: str = "proposed"
) -> str:
    pack_id = new_id()
    conn.execute(
        "INSERT INTO packs (id, title, rationale, status) VALUES (?, ?, ?, ?)",
        (pack_id, title, rationale, status),
    )
    conn.executemany(
        "INSERT OR IGNORE INTO pack_members (pack_id, product_id) VALUES (?, ?)",
        [(pack_id, m) for m in member_ids],
    )
    return pack_id


def load_embeddings(conn: sqlite3.Connection, kind: str, model: str) -> tuple[list[str], np.ndarray]:
    """Return (product_ids, matrix) for all embeddings of one kind and model.

    Only one model at a time — mixing checkpoints would compare incomparable
    vectors. Returns an empty (list, (0,0) array) when there are none.
    """
    rows = conn.execute(
        "SELECT product_id, dim, vector FROM embeddings WHERE kind = ? AND model = ? ORDER BY product_id",
        (kind, model),
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    ids = [r["product_id"] for r in rows]
    mat = np.stack([blob_to_vector(r["vector"], r["dim"]) for r in rows])
    return ids, mat
