"""Structured run logging (layer 7).

One JSONL file per run at data/logs/run-{run_id}.jsonl, one object per line,
mirrored to stdout. Every record carries the same stable keys — ts, run_id,
stage, product_id, level, msg, duration_ms — because these become OTel span
attributes later. Lines are flushed individually so a crashed worker keeps its
last lines. Runs older than 14 days are pruned.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipelines.catalog import config

RETENTION_DAYS = 14


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class RunLogger:
    """Append-only JSONL logger scoped to a single run_id."""

    def __init__(self, run_id: str, logs_dir: Path | None = None, mirror: bool = True):
        self.run_id = run_id
        self.mirror = mirror
        self.logs_dir = logs_dir or config.LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.logs_dir / f"run-{run_id}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def log(
        self,
        stage: str,
        msg: str,
        *,
        product_id: str | None = None,
        level: str = "info",
        duration_ms: float | None = None,
        **extra,
    ) -> None:
        record = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "stage": stage,
            "product_id": product_id,
            "level": level,
            "msg": msg,
            "duration_ms": duration_ms,
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()  # a crashed worker must not lose its last lines
        if self.mirror:
            print(line, file=sys.stdout, flush=True)

    @contextmanager
    def timed(self, stage: str, msg: str, *, product_id: str | None = None, **extra):
        """Log msg with the wall-clock duration of the enclosed block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            dur = (time.perf_counter() - start) * 1000.0
            self.log(stage, msg, product_id=product_id, duration_ms=round(dur, 2), **extra)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def prune_old_logs(logs_dir: Path | None = None, retention_days: int = RETENTION_DAYS) -> list[Path]:
    """Delete run logs older than retention_days. Return the paths removed."""
    logs_dir = logs_dir or config.LOGS_DIR
    if not logs_dir.is_dir():
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).timestamp()
    removed = []
    for p in logs_dir.glob("run-*.jsonl"):
        if p.stat().st_mtime < cutoff:
            p.unlink()
            removed.append(p)
    return removed
