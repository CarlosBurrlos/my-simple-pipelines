"""Tests for imageops. Synthetic images only — no network, no fixtures on disk."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from imageops import encode, flatten, normalize, probe, resize, square_crop


def _png_bytes(size, color, mode="RGB"):
    im = Image.new(mode, size, color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_probe_reports_dimensions(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (640, 480), "red").save(p)
    got = probe(p)
    assert (got.width, got.height) == (640, 480)
    assert got.short_side == 480
    assert got.format == "PNG"


def test_flatten_composites_alpha_onto_white():
    # Fully transparent RGBA -> should become the background, not black.
    im = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    out = flatten(im)
    assert out.mode == "RGB"
    assert out.getpixel((5, 5)) == (255, 255, 255)


def test_flatten_respects_background_argument():
    im = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    out = flatten(im, background=(10, 20, 30))
    assert out.getpixel((0, 0)) == (10, 20, 30)


def test_normalize_lands_in_rgb():
    im = Image.new("RGBA", (8, 8), (5, 5, 5, 255))
    out = normalize(im)
    assert out.mode == "RGB"


def test_square_crop_is_centered_square():
    im = Image.new("RGB", (100, 40), "blue")
    out = square_crop(im)
    assert out.size == (40, 40)


def test_resize_scales_long_edge_and_keeps_aspect():
    im = Image.new("RGB", (800, 400), "green")
    out = resize(im, 200)
    assert out.size == (200, 100)


def test_resize_never_upscales():
    im = Image.new("RGB", (100, 50), "green")
    out = resize(im, 500)
    assert out.size == (100, 50)


def test_encode_writes_jpeg_and_returns_size(tmp_path):
    im = Image.new("RGB", (256, 256), "white")
    out = tmp_path / "sub" / "o.jpg"
    n = encode(im, out, fmt="jpg", quality=85)
    assert out.exists()
    assert n == out.stat().st_size
    with Image.open(out) as reread:
        assert reread.format == "JPEG"


def test_encode_honors_max_bytes(tmp_path):
    # A smooth gradient compresses well, so quality-stepping can actually reach
    # a tight ceiling. (Pure random noise is near-incompressible and would hit
    # the quality floor without meeting any small ceiling — a bad test target.)
    im = Image.new("RGB", (1024, 1024))
    px = [(x % 256, (x + y) % 256, y % 256) for y in range(1024) for x in range(1024)]
    im.putdata(px)
    out = tmp_path / "big.jpg"

    full = encode(im, tmp_path / "full.jpg", fmt="jpg", quality=95)
    ceiling = full // 2
    n = encode(im, out, fmt="jpg", quality=95, max_bytes=ceiling)
    assert n <= ceiling
    assert n < full  # stepping down actually shrank it


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP", "PNG"])
def test_encode_supports_multiple_formats(tmp_path, fmt):
    im = Image.new("RGB", (32, 32), "white")
    out = tmp_path / f"o.{fmt.lower()}"
    encode(im, out, fmt=fmt, quality=80)
    with Image.open(out) as reread:
        assert reread.format == fmt
