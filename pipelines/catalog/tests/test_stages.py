"""Tests for the mechanical stages: intake, colour, derive, attribute scoring."""

from __future__ import annotations

import numpy as np
from PIL import Image

from tagging import Palette
from pipelines.catalog import config
from pipelines.catalog.stages import attribute, color, derive, intake
from pipelines.catalog.tests.conftest import make_image


def test_intake_produces_ok_master(catalog_env, tmp_path):
    src = make_image(tmp_path / "in.png", size=(2200, 2400))
    res = intake.normalize_to_master(src, "prod1")
    assert res.status == "ok"
    assert res.short_side == 2200
    assert res.original_format == "PNG"
    with Image.open(res.source_path) as m:
        assert m.format == "JPEG" and m.mode == "RGB"


def test_intake_quarantines_small(catalog_env, tmp_path):
    src = make_image(tmp_path / "small.png", size=(800, 600))
    res = intake.normalize_to_master(src, "prod2")
    assert res.status == "quarantine"


def test_color_names_dominant(catalog_env):
    im = Image.new("RGB", (300, 300), (30, 80, 55))  # ~forest
    pal = Palette.from_pairs(config.palette_pairs())
    tag = color.extract_color(im, pal)
    assert tag.key == "color" and tag.source == "palette"
    assert tag.value in {"forest", "green", "military green"}
    assert 0.9 < tag.confidence <= 1.0  # solid colour -> one dominant cluster


def test_derive_renders_all_profiles(catalog_env, tmp_path):
    src = make_image(tmp_path / "m.png", size=(3000, 2000), color=(200, 35, 40))
    res = intake.normalize_to_master(src, "prod3")
    outs = derive.render_all(res.source_path, config.profiles(), config.DERIVATIVES_DIR, res.asset_id)
    by = {d.profile: d for d in outs}
    assert set(by) == set(config.profiles())
    # etsy is square + JPEG + under its byte ceiling.
    assert by["etsy"].width == by["etsy"].height
    assert by["etsy"].format == "JPEG"
    assert by["etsy"].bytes <= config.profiles()["etsy"]["max_bytes"]
    # never upscales beyond long_edge.
    assert max(by["web"].width, by["web"].height) <= config.profiles()["web"]["long_edge"]


def test_attribute_scoring_picks_best_over_threshold():
    # Two orthogonal label rows; craft an image vector aligned to row 1.
    row0 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    row1 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    idx = attribute.LabelIndex(key="material", values=["wood", "glass"],
                               matrix=np.stack([row0, row1]), threshold=0.5)
    img = row1.copy()
    tags = attribute.tag_attributes(img, [idx])
    assert len(tags) == 1
    assert tags[0].value == "glass" and tags[0].source == "clip"
    assert tags[0].confidence > 0.99


def test_attribute_scoring_respects_threshold():
    row = np.array([1, 0, 0, 0], dtype=np.float32)
    idx = attribute.LabelIndex(key="style", values=["x"], matrix=row[None, :], threshold=0.9)
    orthogonal = np.array([0, 1, 0, 0], dtype=np.float32)
    assert attribute.tag_attributes(orthogonal, [idx]) == []
