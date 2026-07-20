"""Layer 4 — colour tagging.

KMeans over a downsampled copy finds the dominant colour; the palette (loaded
from config, handed to tagging) names it. The tag's source is 'palette'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from tagging import Palette

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class ColorTag:
    key: str            # always 'color'
    value: str          # palette name, e.g. 'military green'
    rgb: RGB
    source: str         # always 'palette'
    confidence: float   # fraction of pixels in the dominant cluster


def dominant_rgb(im: Image.Image, k: int = 5, sample: int = 200) -> tuple[RGB, float]:
    """Return the most populous KMeans cluster centre and its pixel fraction.

    The image is thumbnailed to at most `sample` px first — colour is a global
    property and clustering full-res pixels is wasteful.
    """
    im = im.convert("RGB").copy()
    im.thumbnail((sample, sample))
    pixels = np.asarray(im, dtype=np.float64).reshape(-1, 3)
    k = min(k, len(np.unique(pixels, axis=0)))
    k = max(k, 1)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pixels)
    labels, counts = np.unique(km.labels_, return_counts=True)
    winner = labels[int(np.argmax(counts))]
    frac = float(counts.max() / counts.sum())
    centre = km.cluster_centers_[winner]
    rgb = tuple(int(round(c)) for c in centre)
    return rgb, frac


def extract_color(im: Image.Image, palette: Palette, k: int = 5) -> ColorTag:
    """Dominant colour named against the palette, as a 'palette'-sourced tag."""
    rgb, frac = dominant_rgb(im, k=k)
    match = palette.nearest(rgb)
    return ColorTag(key="color", value=match.name, rgb=rgb, source="palette", confidence=frac)
