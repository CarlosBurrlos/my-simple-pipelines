"""tagging — generic tagging primitives, handed their data, never reading config.

Palette matching and cosine retrieval are pure numpy and always available. CLIP
(clip.Encoder) pulls torch lazily, so importing this package is cheap and does
not require the `clip` extra unless you actually embed.
"""

from tagging.cosine import normalize, similarity, top_k
from tagging.palette import Match, Palette, Swatch

__all__ = ["Match", "Palette", "Swatch", "normalize", "similarity", "top_k"]
