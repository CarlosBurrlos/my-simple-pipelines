"""CLIP vision + text encoders — one model, three jobs (embed image, embed text,
score labels). torch and open_clip are imported lazily inside the loader so that
importing `tagging` costs nothing until you actually embed something.

Embeddings from different checkpoints are not comparable, so every vector this
produces is tagged with model_id(); store it alongside the vector and never mix.
"""

from __future__ import annotations

import functools

import numpy as np

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"


def model_id(model: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED) -> str:
    """Stable identifier to store beside every vector: 'ViT-B-32/openai'."""
    return f"{model}/{pretrained}"


@functools.lru_cache(maxsize=2)
def _load(model: str, pretrained: str):
    # Imported here, not at module top, so the pure-numpy users of `tagging`
    # never pay the torch import cost or need it installed at all.
    import open_clip
    import torch

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    net, _, preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
    net = net.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model)
    return net, preprocess, tokenizer, device


class Encoder:
    """A warm, reusable CLIP encoder. Construct once; reuse across a batch.

    Model load dominates cold start (~600MB weights on first ever run), so hold
    one of these for the life of a worker rather than rebuilding per image.
    """

    def __init__(self, model: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED):
        self.model = model
        self.pretrained = pretrained
        self._net, self._preprocess, self._tokenizer, self._device = _load(model, pretrained)

    @property
    def id(self) -> str:
        return model_id(self.model, self.pretrained)

    def embed_image(self, images) -> np.ndarray:
        """Embed one PIL image or a list of them. Returns (n, dim) float32, L2-normalized."""
        import torch

        single = not isinstance(images, (list, tuple))
        batch = [images] if single else list(images)
        tensor = torch.stack([self._preprocess(im) for im in batch]).to(self._device)
        with torch.no_grad():
            feats = self._net.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().numpy().astype("float32")
        return arr[0] if single else arr

    def embed_text(self, texts) -> np.ndarray:
        """Embed one string or a list. Returns (n, dim) float32, L2-normalized."""
        import torch

        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        tokens = self._tokenizer(batch).to(self._device)
        with torch.no_grad():
            feats = self._net.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().numpy().astype("float32")
        return arr[0] if single else arr
