"""Palette matching. Hand it named reference colours and a query RGB; it returns
the nearest name. It never reads palette.yaml or any config — the caller loads
the palette and passes it in, which is what keeps this module domain-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Swatch:
    name: str
    rgb: RGB


@dataclass(frozen=True)
class Match:
    name: str
    rgb: RGB
    distance: float


class Palette:
    """A fixed set of named colours, queryable for the nearest name.

    Distance is Euclidean in a luma-weighted RGB space — cheap, and closer to
    perceived difference than raw RGB without dragging in a colour library.
    """

    # Rough perceptual weights (ITU-R BT.601 luma coefficients).
    _WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)

    def __init__(self, swatches: list[Swatch]):
        if not swatches:
            raise ValueError("Palette needs at least one swatch")
        self._swatches = swatches
        self._points = np.array([s.rgb for s in swatches], dtype=np.float64)

    @classmethod
    def from_pairs(cls, pairs: dict[str, RGB] | list[tuple[str, RGB]]) -> "Palette":
        items = pairs.items() if isinstance(pairs, dict) else pairs
        return cls([Swatch(name, tuple(rgb)) for name, rgb in items])

    def nearest(self, rgb: RGB) -> Match:
        q = np.asarray(rgb, dtype=np.float64)
        diff = (self._points - q) * self._WEIGHTS
        d2 = np.einsum("ij,ij->i", diff, diff)
        i = int(np.argmin(d2))
        s = self._swatches[i]
        return Match(name=s.name, rgb=s.rgb, distance=float(np.sqrt(d2[i])))

    def __len__(self) -> int:
        return len(self._swatches)
