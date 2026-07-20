"""FastAPI app — the v1 entry point.

Three jobs: accept dropped photos, let the user group them into products on
screen (identity is folder-per-product, not filenames — there is no bulk rename),
and view run logs at /runs. Ingest of a grouping runs layers 3-8 synchronously;
CLIP embedding and proposals are opt-in (they need the clip extra / an API key).

    uv run uvicorn pipelines.catalog.ui.app:app --reload
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ledger import connect, migrate, new_id
from pipelines.catalog import config
from pipelines.catalog.logging import RunLogger
from pipelines.catalog.run import Group, ingest_batch

_STATIC = Path(__file__).parent / "static"
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_data_dirs()
    migrate(config.DB_PATH)
    yield


app = FastAPI(title="catalog", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/upload")
async def upload(files: list[UploadFile]) -> JSONResponse:
    """Save dropped files to the incoming dir under collision-proof names."""
    saved = []
    for f in files:
        safe = _SAFE.sub("_", Path(f.filename or "photo").name)
        name = f"{new_id()}__{safe}"
        dest = config.INCOMING_DIR / name
        dest.write_bytes(await f.read())
        saved.append({"name": name, "original": f.filename, "url": f"/incoming/{name}"})
    return JSONResponse({"files": saved})


@app.post("/group")
def group(payload: dict) -> JSONResponse:
    """Ingest grouped photos. payload = {groups: [{display_name, files: [name, ...]}]}."""
    groups_in = payload.get("groups") or []
    if not groups_in:
        raise HTTPException(400, "no groups provided")

    groups = []
    for g in groups_in:
        paths = []
        for name in g.get("files", []):
            p = config.INCOMING_DIR / Path(name).name
            if not p.exists():
                raise HTTPException(400, f"unknown upload: {name}")
            paths.append(str(p))
        if paths:
            groups.append(Group(display_name=g.get("display_name") or "untitled", image_paths=paths))

    run_id = new_id()
    conn = connect(config.DB_PATH)
    try:
        with RunLogger(run_id) as logger:
            outcomes = ingest_batch(conn, groups, encoder=None, logger=logger)
    finally:
        conn.close()

    return JSONResponse(
        {
            "run_id": run_id,
            "products": [
                {
                    "product_id": o.product_id,
                    "assets": len(o.asset_ids),
                    "quarantined": len(o.quarantined),
                    "color": o.color,
                }
                for o in outcomes
            ],
        }
    )


@app.get("/runs")
def runs() -> JSONResponse:
    files = sorted(config.LOGS_DIR.glob("run-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return JSONResponse({"runs": [p.stem.removeprefix("run-") for p in files]})


@app.get("/runs/{run_id}")
def run_detail(run_id: str) -> JSONResponse:
    path = config.LOGS_DIR / f"run-{_SAFE.sub('_', run_id)}.jsonl"
    if not path.exists():
        raise HTTPException(404, "no such run")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return JSONResponse({"run_id": run_id, "records": records})


@app.get("/incoming/{name}")
def incoming(name: str) -> FileResponse:
    """Serve an uploaded original so the grouping UI can show thumbnails."""
    path = config.INCOMING_DIR / Path(name).name  # Path(...).name blocks traversal
    if not path.exists():
        raise HTTPException(404, "no such upload")
    return FileResponse(path)
