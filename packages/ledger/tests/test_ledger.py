"""Tests for the ledger: migrations, pragmas, constraints, IDs."""

from __future__ import annotations

import pytest

from ledger import connect, migrate, new_id


def test_migrate_creates_baseline_once(tmp_path):
    db = tmp_path / "catalog.db"
    first = migrate(db)
    assert "0000_baseline" in first
    # Second run is a no-op — nothing newly applied.
    assert migrate(db) == []


def test_wal_and_foreign_keys_enabled(tmp_path):
    db = tmp_path / "catalog.db"
    migrate(db)
    conn = connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_product_and_asset_roundtrip(tmp_path):
    db = tmp_path / "catalog.db"
    migrate(db)
    conn = connect(db)
    try:
        pid = new_id()
        conn.execute(
            "INSERT INTO products (id, external_ref, display_name) VALUES (?, ?, ?)",
            (pid, None, "Blue Mug"),
        )
        aid = new_id()
        conn.execute(
            "INSERT INTO assets (id, product_id, source_path, width, height, short_side) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (aid, pid, "/data/x.jpg", 3000, 4000, 3000),
        )
        row = conn.execute("SELECT display_name FROM products WHERE id=?", (pid,)).fetchone()
        assert row["display_name"] == "Blue Mug"
    finally:
        conn.close()


def test_tag_source_is_constrained(tmp_path):
    db = tmp_path / "catalog.db"
    migrate(db)
    conn = connect(db)
    try:
        pid = new_id()
        conn.execute("INSERT INTO products (id) VALUES (?)", (pid,))
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO tags (id, product_id, key, value, source) VALUES (?, ?, ?, ?, ?)",
                (new_id(), pid, "color", "green", "guess"),  # not in allowed set
            )
    finally:
        conn.close()


def test_foreign_key_is_enforced(tmp_path):
    db = tmp_path / "catalog.db"
    migrate(db)
    conn = connect(db)
    try:
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO assets (id, product_id, source_path, width, height, short_side) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (new_id(), "nonexistent", "/x.jpg", 10, 10, 10),
            )
    finally:
        conn.close()


def test_new_id_is_unique_and_sortable():
    ids = [new_id() for _ in range(100)]
    assert len(set(ids)) == 100
    assert ids == sorted(ids)  # ULIDs are monotonic within a process/time
