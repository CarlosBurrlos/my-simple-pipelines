"""Tests for the pure-numpy tagging primitives. CLIP is not exercised here (it
needs the clip extra + weights); its lazy import is asserted instead."""

from __future__ import annotations

import numpy as np

from tagging import Palette, normalize, similarity, top_k


def test_palette_nearest_exact():
    pal = Palette.from_pairs({"black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0)})
    assert pal.nearest((250, 5, 5)).name == "red"
    assert pal.nearest((10, 10, 10)).name == "black"


def test_palette_from_list_and_len():
    pal = Palette.from_pairs([("navy", (0, 0, 128)), ("cream", (255, 253, 240))])
    assert len(pal) == 2
    assert pal.nearest((0, 0, 120)).name == "navy"


def test_palette_rejects_empty():
    import pytest

    with pytest.raises(ValueError):
        Palette.from_pairs([])


def test_normalize_unit_length():
    v = normalize(np.array([3.0, 4.0]))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_normalize_handles_zero():
    v = normalize(np.zeros(4))
    assert np.allclose(v, 0.0)


def test_similarity_identical_is_one():
    q = np.array([1.0, 2.0, 3.0])
    m = np.stack([q, -q, np.array([3.0, 2.0, 1.0])])
    sims = similarity(q, m)
    assert abs(sims[0] - 1.0) < 1e-6
    assert abs(sims[1] + 1.0) < 1e-6


def test_top_k_orders_by_similarity():
    q = np.array([1.0, 0.0])
    m = np.stack([
        np.array([1.0, 0.0]),    # sim 1.0
        np.array([0.7, 0.7]),    # sim ~0.707
        np.array([0.0, 1.0]),    # sim 0.0
        np.array([-1.0, 0.0]),   # sim -1.0
    ])
    got = top_k(q, m, k=2)
    assert [i for i, _ in got] == [0, 1]
    assert got[0][1] > got[1][1]


def test_top_k_clamps_and_empties():
    assert top_k(np.array([1.0]), np.empty((0, 1)), 3) == []
    m = np.ones((2, 1))
    assert len(top_k(np.array([1.0]), m, 10)) == 2


def test_clip_import_is_lazy():
    # Importing tagging must not require torch. model_id works without weights.
    from tagging import clip

    assert clip.model_id() == "ViT-B-32/openai"
