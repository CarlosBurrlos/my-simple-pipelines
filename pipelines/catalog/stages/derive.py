"""Layer 8 — derivatives.

For each named profile, render a marketplace image with imageops and record what
was written. Runs pre-gate (before judgment), so it wastes a little work on
losing shots in exchange for layers 3-8 staying fully unattended.

This stage is the one that knows what "etsy" means; imageops only ever sees the
numbers and format string pulled out of the profile here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import imageops


@dataclass(frozen=True)
class DerivativeResult:
    profile: str
    path: str
    format: str
    width: int
    height: int
    bytes: int


def render_derivative(
    master: Image.Image,
    profile_name: str,
    spec: dict,
    out_path: Path,
) -> DerivativeResult:
    """Render one derivative from an already-normalized master image."""
    im = master
    if spec.get("square"):
        im = imageops.square_crop(im)
    im = imageops.resize(im, int(spec["long_edge"]))
    fmt = str(spec["format"])
    n = imageops.encode(
        im,
        out_path,
        fmt=fmt,
        quality=int(spec.get("quality", 85)),
        max_bytes=spec.get("max_bytes"),
    )
    w, h = im.size
    return DerivativeResult(
        profile=profile_name,
        path=str(out_path),
        format=fmt.upper(),
        width=w,
        height=h,
        bytes=n,
    )


def render_all(
    master_path: str | Path,
    profiles: dict[str, dict],
    out_dir: Path,
    asset_id: str,
) -> list[DerivativeResult]:
    """Render every profile for one asset into out_dir/<asset_id>/<profile>.<ext>."""
    master_path = Path(master_path)
    results = []
    with Image.open(master_path) as master:
        master.load()
        for name, spec in profiles.items():
            ext = "jpg" if str(spec["format"]).upper() in ("JPEG", "JPG") else str(spec["format"]).lower()
            out_path = out_dir / asset_id / f"{name}.{ext}"
            results.append(render_derivative(master, name, spec, out_path))
    return results
