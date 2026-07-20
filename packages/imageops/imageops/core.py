"""Self-contained image primitives.

The one rule this module lives by: it never learns a domain name. It takes a
path, an image, and numbers, and returns an image or plain metadata. Sizes,
quality integers and byte ceilings are arguments; strings like "etsy" or
"thumb" belong to the caller, never here.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# iPhone ships HEIC; Pillow cannot open it until this runs. Registering at
# import means every entry point into this module gets HEIC support for free.
register_heif_opener()

WHITE = (255, 255, 255)


@dataclass(frozen=True)
class Probe:
    """Plain facts about an image file. No judgment, no domain."""

    width: int
    height: int
    format: str | None
    mode: str
    short_side: int


def probe(path: str | Path) -> Probe:
    """Read dimensions and format without decoding pixels where avoidable."""
    with Image.open(path) as im:
        fmt = im.format  # exif_transpose returns a fresh image with no .format
        mode = im.mode
        # EXIF orientation can swap the logical width/height; report what a
        # viewer would actually see, matching what normalize() will produce.
        upright = ImageOps.exif_transpose(im)
        w, h = upright.size
        return Probe(
            width=w,
            height=h,
            format=fmt,
            mode=mode,
            short_side=min(w, h),
        )


def flatten(im: Image.Image, background: tuple[int, int, int] = WHITE) -> Image.Image:
    """Composite any alpha/palette transparency onto a solid background.

    Transparent pixels render black on some marketplaces; flattening to white
    is the safe default, but the colour is the caller's to choose.
    """
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, background)
        canvas.paste(rgba, mask=rgba.split()[-1])
        return canvas
    return im.convert("RGB")


def normalize(im: Image.Image, background: tuple[int, int, int] = WHITE) -> Image.Image:
    """Apply EXIF rotation, drop transparency, land in RGB.

    The canonical form everything downstream assumes: upright, opaque, RGB.
    """
    return flatten(ImageOps.exif_transpose(im), background)


def square_crop(im: Image.Image) -> Image.Image:
    """Centre-crop to a 1:1 square using the shorter side."""
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side))


def resize(im: Image.Image, long_edge: int) -> Image.Image:
    """Scale so the longest side equals long_edge, preserving aspect ratio.

    Never upscales: an image already smaller than the target is returned as-is.
    """
    w, h = im.size
    if max(w, h) <= long_edge:
        return im
    if w >= h:
        new = (long_edge, round(h * long_edge / w))
    else:
        new = (round(w * long_edge / h), long_edge)
    return im.resize(new, Image.Resampling.LANCZOS)


def encode(
    im: Image.Image,
    out: str | Path,
    fmt: str,
    quality: int,
    max_bytes: int | None = None,
) -> int:
    """Write im to out in the given format; return bytes written.

    When max_bytes is set and the format is lossy, quality is stepped down
    until the encoded size fits (or a floor is hit). Returns the final size.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.upper()
    if fmt == "JPG":
        fmt = "JPEG"

    def render(q: int) -> bytes:
        buf = io.BytesIO()
        params: dict[str, object] = {"format": fmt}
        if fmt in ("JPEG", "WEBP"):
            params["quality"] = q
        if fmt == "JPEG":
            params["optimize"] = True
            params["progressive"] = True
        im.save(buf, **params)
        return buf.getvalue()

    data = render(quality)
    if max_bytes is not None and fmt in ("JPEG", "WEBP"):
        q = quality
        while len(data) > max_bytes and q > 20:
            q -= 5
            data = render(q)

    out.write_bytes(data)
    return len(data)
