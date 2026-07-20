"""Layer 13 — artifacts.

Render an HTML sheet per approved pack from the ledger, via Jinja2. Only
approved packs produce artifacts; proposed/rejected never do.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from jinja2 import Environment

from ledger import connect, migrate
from pipelines.catalog import config

_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>{{ pack.title }}</title>
<style>
  body { font: 15px/1.5 system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
  h1 { margin-bottom: .2rem; }
  .rationale { color: #555; margin-bottom: 1.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; }
  .card { border: 1px solid #eee; border-radius: 8px; overflow: hidden; }
  .card img { width: 100%; display: block; background: #fafafa; }
  .card .name { padding: .5rem .75rem; font-weight: 600; }
  .card .tags { padding: 0 .75rem .75rem; color: #777; font-size: 13px; }
</style>
<h1>{{ pack.title }}</h1>
<p class="rationale">{{ pack.rationale }}</p>
<div class="grid">
{% for item in items %}
  <div class="card">
    {% if item.hero %}<img src="{{ item.hero }}" alt="{{ item.name }}">{% endif %}
    <div class="name">{{ item.name or item.product_id }}</div>
    <div class="tags">{{ item.tags | join(' · ') }}</div>
  </div>
{% endfor %}
</div>
"""


def _pack_items(conn: sqlite3.Connection, pack_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT pr.id, pr.display_name, pr.hero_asset_id
        FROM pack_members m JOIN products pr ON pr.id = m.product_id
        WHERE m.pack_id = ? ORDER BY pr.id
        """,
        (pack_id,),
    ).fetchall()
    items = []
    for r in rows:
        hero = None
        if r["hero_asset_id"]:
            d = conn.execute(
                "SELECT path FROM derivatives WHERE asset_id = ? ORDER BY (profile='web') DESC LIMIT 1",
                (r["hero_asset_id"],),
            ).fetchone()
            hero = d["path"] if d else None
        tags = [
            f"{t['key']}: {t['value']}"
            for t in conn.execute(
                "SELECT key, value FROM tags WHERE product_id = ? ORDER BY key", (r["id"],)
            )
        ]
        items.append({"product_id": r["id"], "name": r["display_name"], "hero": hero, "tags": tags})
    return items


def render_pack(conn: sqlite3.Connection, pack_id: str) -> str:
    pack = conn.execute("SELECT id, title, rationale, status FROM packs WHERE id = ?", (pack_id,)).fetchone()
    if pack is None:
        raise KeyError(pack_id)
    env = Environment(autoescape=True)
    return env.from_string(_TEMPLATE).render(pack=dict(pack), items=_pack_items(conn, pack_id))


def build_approved(db_path: str | None = None, out_dir: Path | None = None) -> list[Path]:
    """Render every approved pack to out_dir/<pack_id>.html. Return paths written."""
    db_path = db_path or str(config.DB_PATH)
    out_dir = out_dir or (config.DATA_DIR / "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    migrate(db_path)
    conn = connect(db_path)
    try:
        ids = [r["id"] for r in conn.execute("SELECT id FROM packs WHERE status = 'approved'")]
        written = []
        for pid in ids:
            path = out_dir / f"{pid}.html"
            path.write_text(render_pack(conn, pid), encoding="utf-8")
            written.append(path)
        return written
    finally:
        conn.close()


if __name__ == "__main__":
    for p in build_approved():
        print(p)
