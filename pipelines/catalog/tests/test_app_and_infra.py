"""Tests for config loaders, run logging, and the FastAPI app end to end."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from pipelines.catalog import config
from pipelines.catalog.logging import RunLogger, prune_old_logs


def test_config_loaders():
    assert set(config.profiles()) >= {"etsy", "web", "preview", "thumb"}
    pairs = config.palette_pairs()
    assert pairs["forest"] == (30, 80, 55)
    groups = config.label_groups()
    assert any(g.key == "category" for g in groups)
    assert config.themes() == []  # off in v1


def test_run_logger_writes_jsonl(catalog_env):
    import json

    with RunLogger("run-abc", mirror=False) as log:
        with log.timed("intake", "did a thing", product_id="p1"):
            pass
        log.log("color", "tagged", product_id="p1", level="info")
    lines = (config.LOGS_DIR / "run-run-abc.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["run_id"] == "run-abc" and rec["stage"] == "intake"
    assert rec["duration_ms"] is not None


def test_prune_old_logs_removes_stale(catalog_env):
    import os
    import time

    old = config.LOGS_DIR / "run-old.jsonl"
    old.write_text("{}\n")
    stale = time.time() - 20 * 86400
    os.utime(old, (stale, stale))
    fresh = config.LOGS_DIR / "run-new.jsonl"
    fresh.write_text("{}\n")
    removed = prune_old_logs()
    assert old in removed and fresh.exists()


def _png_upload(color=(60, 140, 80), size=(2200, 2200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_app_upload_group_and_runs(catalog_env):
    from pipelines.catalog.ui import app as app_module

    # `with` fires the startup event (migrate + ensure_data_dirs).
    with TestClient(app_module.app) as client:
        up = client.post("/upload", files=[("files", ("a.png", _png_upload(), "image/png"))])
        assert up.status_code == 200
        name = up.json()["files"][0]["name"]

        # thumbnail is servable
        assert client.get(f"/incoming/{name}").status_code == 200

        grp = client.post("/group", json={"groups": [{"display_name": "Mug", "files": [name]}]})
        assert grp.status_code == 200
        body = grp.json()
        assert len(body["products"]) == 1
        assert body["products"][0]["assets"] == 1
        assert body["products"][0]["color"] is not None  # 2200px -> ok, colour tagged

        runs = client.get("/runs").json()["runs"]
        assert body["run_id"] in runs
        detail = client.get(f"/runs/{body['run_id']}").json()
        assert detail["records"]
