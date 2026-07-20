"""Tests for ledger writes, shortlist, judge helpers, approval, artifacts."""

from __future__ import annotations

import numpy as np

from pipelines.catalog import approve, artifacts, config
from pipelines.catalog.stages import judge, shortlist, write
from pipelines.catalog.tests.conftest import make_image


MODEL = "ViT-B-32/openai"


def _product_with_vector(conn, name, vec, tags=()):
    pid = write.insert_product(conn, display_name=name)
    write.insert_embedding(conn, pid, "image", MODEL, np.asarray(vec, dtype=np.float32))
    for k, v in tags:
        write.insert_tag(conn, pid, k, v, "clip")
    return pid


def test_embedding_blob_roundtrip(db):
    vec = np.arange(6, dtype=np.float32)
    pid = write.insert_product(db, "x")
    write.insert_embedding(db, pid, "image", MODEL, vec)
    ids, mat = write.load_embeddings(db, "image", MODEL)
    assert ids == [pid]
    assert np.allclose(mat[0], vec)


def test_shortlist_unions_neighbours_and_tags(db):
    a = _product_with_vector(db, "a", [1, 0, 0], tags=[("category", "mug")])
    b = _product_with_vector(db, "b", [0.99, 0.01, 0], tags=[("category", "bowl")])   # visual neighbour
    _c = _product_with_vector(db, "c", [0, 0, 1], tags=[("category", "mug")])         # tag neighbour only
    got = shortlist.shortlist_for(db, a, MODEL, k=1)
    assert b in got          # nearest image vector
    assert _c in got         # shares a 'mug' tag
    assert a not in got      # never includes itself


def test_judge_contact_sheet_and_parse(db, tmp_path):
    src = make_image(tmp_path / "s.png", size=(500, 500))
    sheet = judge.build_contact_sheet("prodX", [("a.jpg", str(src)), ("b.jpg", str(src))],
                                      tmp_path / "sheet.jpg")
    assert sheet.shots == ["a.jpg", "b.jpg"]
    from PIL import Image
    with Image.open(sheet.image_path) as im:
        assert im.size[0] > 0

    parsed = judge.parse_proposal('```json\n{"heroes": {"p": "a.jpg"}, "packs": []}\n```')
    assert parsed["heroes"]["p"] == "a.jpg"


def test_judge_build_messages_shape(tmp_path):
    src = make_image(tmp_path / "s.png", size=(300, 300))
    sheet = judge.build_contact_sheet("p1", [("a.jpg", str(src))], tmp_path / "sh.jpg")
    sheet.tags = [("color", "green")]
    msgs = judge.build_messages([sheet], {"p1": {"p2"}})
    assert msgs[0]["role"] == "user"
    kinds = [b["type"] for b in msgs[0]["content"]]
    assert "text" in kinds and "image" in kinds


def test_propose_persistence_via_fake_client(db, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    # Two products so there's something to hero/pack.
    p1 = write.insert_product(db, "one")
    p2 = write.insert_product(db, "two")
    for p in (p1, p2):
        from pipelines.catalog.stages.intake import IntakeResult
        src = make_image(tmp_path / f"{p}.png", size=(400, 400))
        # register an ok asset row pointing at a real file
        write.insert_asset(db, IntakeResult(
            asset_id=f"asset-{p}", product_id=p, source_path=str(src),
            width=400, height=400, short_side=400, status="ok", original_format="PNG"))
        write.insert_embedding(db, p, "image", MODEL, np.array([1.0, 0.0], dtype=np.float32))

    class FakeResp:
        content = [type("B", (), {"type": "text", "text":
                    '{"heroes": {"%s": "asset-%s.jpg"}, "packs": '
                    '[{"title":"Set","rationale":"go together","members":["%s","%s"]}]}'
                    % (p1, p1, p1, p2)})()]
        usage = type("U", (), {"input_tokens": 1, "output_tokens": 2})()

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                return FakeResp()

    proposal = __import__("pipelines.catalog.run", fromlist=["propose_packs"]).propose_packs(
        db, MODEL, sheets_dir=tmp_path / "sheets", client=FakeClient())
    assert proposal["packs"][0]["title"] == "Set"
    # hero + pack persisted
    hero = db.execute("SELECT hero_asset_id FROM products WHERE id=?", (p1,)).fetchone()[0]
    assert hero == f"asset-{p1}"
    assert db.execute("SELECT COUNT(*) FROM packs WHERE status='proposed'").fetchone()[0] == 1


def test_approve_flow_and_artifacts(db):
    p = write.insert_product(db, "solo")
    pack = write.insert_pack(db, "Cozy Set", "warm and matching", [p], status="proposed")
    db.commit()

    assert len(approve.list_proposed(db)) == 1
    assert approve.approve_all(db) == 1
    assert len(approve.list_proposed(db)) == 0

    html = artifacts.render_pack(db, pack)
    assert "Cozy Set" in html and "solo" in html
