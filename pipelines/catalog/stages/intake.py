"""Layer 3 — normalize & probe.

HEIC (or anything Pillow reads) in; an upright, opaque, white-flattened RGB
master on disk out. Anything whose short side is under MIN_SHORT_SIDE is marked
'quarantine' rather than upscaled silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import imageops
from ledger import new_id
from pipelines.catalog import config


@dataclass(frozen=True)
class IntakeResult:
    asset_id: str
    product_id: str
    source_path: str        # the normalized master on disk
    width: int
    height: int
    short_side: int
    status: str             # 'ok' | 'quarantine'
    original_format: str | None


def normalize_to_master(
    src: str | Path,
    product_id: str,
    originals_dir: Path | None = None,
    min_short_side: int = config.MIN_SHORT_SIDE,
) -> IntakeResult:
    """Normalize src and write a master under originals_dir/<product_id>/<asset_id>.jpg."""
    src = Path(src)
    originals_dir = originals_dir or config.ORIGINALS_DIR
    probed = imageops.probe(src)

    with Image.open(src) as im:
        master = imageops.normalize(im)

    asset_id = new_id()
    dest = originals_dir / product_id / f"{asset_id}.jpg"
    # Master is high-quality JPEG: derivatives are rendered from it, and product
    # photos don't need lossless masters for a v1 marketplace pipeline.
    imageops.encode(master, dest, fmt="JPEG", quality=95)

    w, h = master.size
    short = min(w, h)
    status = "ok" if short >= min_short_side else "quarantine"
    return IntakeResult(
        asset_id=asset_id,
        product_id=product_id,
        source_path=str(dest),
        width=w,
        height=h,
        short_side=short,
        status=status,
        original_format=probed.format,
    )
