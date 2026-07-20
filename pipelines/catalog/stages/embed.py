"""Layer 5 — image embedding (and the text-vector half of layer 7).

Thin wrappers over tagging.clip.Encoder. The heavy model lives in the tagging
package; this stage only decides *what* to embed for a product: its hero-or-first
master image, and a short text summary of its tags.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def embed_image(encoder, master_path) -> np.ndarray:
    """CLIP image vector for one master, L2-normalized float32."""
    with Image.open(master_path) as im:
        im = im.convert("RGB")
        return encoder.embed_image(im)


def embed_text(encoder, text: str) -> np.ndarray:
    """CLIP text vector for a product's tag summary (layer 7).

    Text neighbours catch functional groupings image vectors miss — a mug +
    notebook + lamp 'desk setup' that looks nothing alike.
    """
    return encoder.embed_text(text)


def tag_summary(tags: list[tuple[str, str]]) -> str:
    """Flatten (key, value) tags into a short phrase CLIP can embed."""
    return ", ".join(value for _key, value in tags)
