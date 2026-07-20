"""Opaque primary keys. ULIDs are lexicographically sortable by creation time,
which is why they work as PKs without leaking a SKU or an autoincrement count."""

from __future__ import annotations

from ulid import ULID


def new_id() -> str:
    """A fresh opaque ULID as a string, ready to use as a primary key."""
    return str(ULID())
