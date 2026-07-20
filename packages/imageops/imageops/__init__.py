"""imageops — self-contained image normalization and derivative encoding.

Lift-and-shift friendly: pass a path and some numbers, get an image back. It
knows nothing about the catalog pipeline, profiles, or any marketplace.
"""

from imageops.core import (
    Probe,
    encode,
    flatten,
    normalize,
    probe,
    resize,
    square_crop,
)

__all__ = [
    "Probe",
    "encode",
    "flatten",
    "normalize",
    "probe",
    "resize",
    "square_crop",
]
