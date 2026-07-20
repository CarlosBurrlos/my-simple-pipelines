"""Batch orchestration: turn grouped uploads into ledger rows, then propose.

Layers 3-8 run unattended and need no model, so ingest works with encoder=None
(intake, colour, derivatives, ledger writes). Pass a warm tagging.clip.Encoder
to also run embedding (5/7) and attribute tagging (6). Proposals (9/10/13) are a
separate catalog-wide step.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from tagging import Palette

from pipelines.catalog import config
from pipelines.catalog.stages import attribute, color, derive, embed, intake, judge, shortlist, write


@dataclass
class Group:
    display_name: str
    image_paths: list[str]
    external_ref: str | None = None


@dataclass
class IngestOutcome:
    product_id: str
    asset_ids: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    color: str | None = None
    attributes: list[tuple[str, str]] = field(default_factory=list)


def ingest_group(
    conn: sqlite3.Connection,
    group: Group,
    *,
    palette: Palette,
    encoder=None,
    label_indexes=None,
    logger=None,
) -> IngestOutcome:
    """Run layers 3-8 (+5/6/7 if an encoder is given) for one product."""
    config.ensure_data_dirs()
    pid = write.insert_product(conn, display_name=group.display_name, external_ref=group.external_ref)
    outcome = IngestOutcome(product_id=pid)

    hero_candidate = None
    for src in group.image_paths:
        res = intake.normalize_to_master(src, pid)
        write.insert_asset(conn, res)
        outcome.asset_ids.append(res.asset_id)
        if res.status == "quarantine":
            outcome.quarantined.append(res.asset_id)
            if logger:
                logger.log("intake", "quarantined (short side < min)", product_id=pid,
                           level="warn", asset_id=res.asset_id, short_side=res.short_side)
            continue
        hero_candidate = hero_candidate or res

        derivs = derive.render_all(res.source_path, config.profiles(), config.DERIVATIVES_DIR, res.asset_id)
        write.insert_derivatives(conn, res.asset_id, derivs)

    if hero_candidate is None:
        if logger:
            logger.log("intake", "all shots quarantined", product_id=pid, level="warn")
        return outcome

    # Colour tag from the first usable master.
    from PIL import Image

    with Image.open(hero_candidate.source_path) as im:
        ctag = color.extract_color(im, palette)
    write.insert_color_tag(conn, pid, ctag)
    outcome.color = ctag.value

    if encoder is not None:
        img_vec = embed.embed_image(encoder, hero_candidate.source_path)
        write.insert_embedding(conn, pid, "image", encoder.id, img_vec, asset_id=hero_candidate.asset_id)

        if label_indexes is None:
            label_indexes = attribute.build_label_index(encoder, config.label_groups())
        attrs = attribute.tag_attributes(img_vec, label_indexes)
        write.insert_attr_tags(conn, pid, attrs)
        outcome.attributes = [(a.key, a.value) for a in attrs]

        # Text vector from the tag summary (layer 7) — functional neighbours.
        summary = embed.tag_summary([(ctag.key, ctag.value), *outcome.attributes])
        txt_vec = embed.embed_text(encoder, summary)
        write.insert_embedding(conn, pid, "text", encoder.id, txt_vec)

    return outcome


def ingest_batch(conn: sqlite3.Connection, groups: list[Group], *, encoder=None, logger=None) -> list[IngestOutcome]:
    palette = Palette.from_pairs(config.palette_pairs())
    label_indexes = (
        attribute.build_label_index(encoder, config.label_groups()) if encoder is not None else None
    )
    out = []
    for g in groups:
        with (logger.timed("ingest", "product ingested") if logger else _null()):
            out.append(ingest_group(conn, g, palette=palette, encoder=encoder,
                                    label_indexes=label_indexes, logger=logger))
    return out


def propose_packs(conn: sqlite3.Connection, model_id: str, *, sheets_dir: Path | None = None,
                  client=None, model: str = "claude-opus-4-8", k: int = 20, logger=None) -> dict:
    """Layers 9-10: shortlist catalog-wide, build sheets, one model call, persist proposals."""
    sheets_dir = sheets_dir or (config.DATA_DIR / "sheets")
    lists = shortlist.shortlist_catalog(conn, model_id, k=k)

    sheets = []
    for pid in lists:
        rows = conn.execute(
            "SELECT id, source_path FROM assets WHERE product_id = ? AND status = 'ok' ORDER BY id", (pid,)
        ).fetchall()
        if not rows:
            continue
        shots = [(f"{r['id']}.jpg", r["source_path"]) for r in rows]
        tag_rows = conn.execute("SELECT key, value FROM tags WHERE product_id = ?", (pid,)).fetchall()
        sheet = judge.build_contact_sheet(pid, shots, sheets_dir / f"{pid}.jpg")
        sheet.tags = [(t["key"], t["value"]) for t in tag_rows]
        sheets.append(sheet)

    proposal = judge.propose(sheets, lists, client=client, model=model, logger=logger)
    _persist_proposal(conn, proposal, sheets)
    return proposal


def _persist_proposal(conn: sqlite3.Connection, proposal: dict, sheets: list[judge.ProductSheet]) -> None:
    # Heroes: filename '<asset_id>.jpg' -> hero_asset_id.
    for pid, filename in (proposal.get("heroes") or {}).items():
        asset_id = str(filename).rsplit(".", 1)[0]
        write.set_hero(conn, pid, asset_id)
    for pack in proposal.get("packs") or []:
        write.insert_pack(conn, pack.get("title", "untitled"), pack.get("rationale", ""),
                          list(pack.get("members", [])), status="proposed")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
