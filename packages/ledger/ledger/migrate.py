"""Migration runner.

The baseline is schema.sql (at the package root, the path CLAUDE.md documents),
recorded as version '0000_baseline'. Anything after it is a numbered file in
migrations/ (e.g. '0001_add_theme_scores.sql'), applied in filename order, each
recorded in schema_migrations so it runs at most once.

Re-running against an up-to-date database is a no-op. This package is used
editable from the workspace, never wheel-installed, so resolving data files by
path relative to this module is correct.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ledger.db import connect

BASELINE_VERSION = "0000_baseline"

_PKG_ROOT = Path(__file__).resolve().parent.parent  # packages/ledger/
SCHEMA_PATH = _PKG_ROOT / "schema.sql"
MIGRATIONS_DIR = _PKG_ROOT / "migrations"


def _applied(conn: sqlite3.Connection) -> set[str]:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if row is None:
        return set()
    return {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}


def _record(conn: sqlite3.Connection, version: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations (version) VALUES (?)", (version,)
    )


def _migration_files() -> list[tuple[str, Path]]:
    """Return (version, path) pairs from migrations/, sorted by filename."""
    if not MIGRATIONS_DIR.is_dir():
        return []
    files = sorted(MIGRATIONS_DIR.glob("*.sql"), key=lambda p: p.name)
    return [(p.stem, p) for p in files]


def migrate(db_path: str | Path) -> list[str]:
    """Bring the database at db_path up to date. Return versions newly applied."""
    conn = connect(db_path)
    try:
        applied = _applied(conn)
        newly: list[str] = []

        if BASELINE_VERSION not in applied:
            conn.executescript(SCHEMA_PATH.read_text())
            _record(conn, BASELINE_VERSION)
            newly.append(BASELINE_VERSION)

        for version, path in _migration_files():
            if version in applied:
                continue
            conn.executescript(path.read_text())
            _record(conn, version)
            newly.append(version)

        return newly
    finally:
        conn.close()
