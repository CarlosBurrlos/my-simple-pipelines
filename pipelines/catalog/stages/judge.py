"""Layer 10 — judgment. The only stage that requires a model.

It receives contact sheets (one composite per product: thumbnails of every
candidate shot with filenames printed underneath) plus the layer-9 shortlist and
tags. It returns the hero shot per product and pack proposals with written
rationale. One API call per batch. Nothing it produces is final — everything
lands status='proposed'.

Everything behind propose() is swappable: point base_url/client at Ollama's
OpenAI-compatible endpoint and nothing else changes. The Anthropic path is the
default because aesthetic ranking and theme naming are where small local VLMs
are weakest, and this is the cheapest call in the pipeline to get right.
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

CELL = 256          # thumbnail cell size in the contact sheet
PAD = 8
CAPTION_H = 18


@dataclass
class ProductSheet:
    product_id: str
    image_path: str          # the rendered contact sheet on disk
    shots: list[str]         # filenames printed on the sheet, in grid order
    tags: list[tuple[str, str]] = field(default_factory=list)


def build_contact_sheet(product_id: str, shots: list[tuple[str, str]], out_path: Path) -> ProductSheet:
    """Compose one sheet from (filename, image_path) pairs.

    Filenames are drawn under each thumbnail so the model can name a hero by the
    exact filename, which is how its choice maps back to an asset.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = max(len(shots), 1)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = CELL + PAD
    cell_h = CELL + CAPTION_H + PAD
    sheet = Image.new("RGB", (cols * cell_w + PAD, rows * cell_h + PAD), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for i, (label, path) in enumerate(shots):
        r, c = divmod(i, cols)
        x = PAD + c * cell_w
        y = PAD + r * cell_h
        with Image.open(path) as im:
            im = im.convert("RGB").copy()
            im.thumbnail((CELL, CELL))
            sheet.paste(im, (x + (CELL - im.width) // 2, y + (CELL - im.height) // 2))
        draw.text((x, y + CELL + 2), label, fill=(0, 0, 0))

    sheet.save(out_path, format="JPEG", quality=88)
    return ProductSheet(
        product_id=product_id,
        image_path=str(out_path),
        shots=[label for label, _ in shots],
    )


SYSTEM_PROMPT = (
    "You are a product-catalog merchandiser. You are given one contact sheet per "
    "product (thumbnails with filenames printed beneath each shot), each product's "
    "tags, and a shortlist of which products are candidate neighbours of each other. "
    "Do two things and return STRICT JSON only, no prose:\n"
    "1. For each product, pick the single best hero shot by its exact filename.\n"
    "2. Propose sellable packs: coherent bundles of products that belong together, "
    "each with a short title and a one-sentence rationale.\n"
    'Return: {"heroes": {"<product_id>": "<filename>"}, '
    '"packs": [{"title": str, "rationale": str, "members": ["<product_id>", ...]}]}'
)


def _encode_image(path: str) -> dict:
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def build_messages(sheets: list[ProductSheet], shortlist: dict[str, set[str]]) -> list[dict]:
    """Assemble the user message content: sheets interleaved with their context.

    Pure and testable — no network. Returns Anthropic message content blocks.
    """
    content: list[dict] = []
    for s in sheets:
        neighbours = sorted(shortlist.get(s.product_id, set()))
        tagline = "; ".join(f"{k}:{v}" for k, v in s.tags) or "(none)"
        content.append(
            {
                "type": "text",
                "text": (
                    f"product_id={s.product_id}\ntags={tagline}\n"
                    f"shortlist_neighbours={neighbours}\nshots={s.shots}"
                ),
            }
        )
        content.append(_encode_image(s.image_path))
    return [{"role": "user", "content": content}]


def parse_proposal(text: str) -> dict:
    """Extract the JSON object from a model response, tolerant of code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def propose(
    sheets: list[ProductSheet],
    shortlist: dict[str, set[str]],
    *,
    client=None,
    model: str = "claude-opus-4-8",
    max_tokens: int = 4096,
    logger=None,
) -> dict:
    """One model call for the whole batch. Returns the parsed proposal dict.

    The only thing that knows an LLM is involved. Swap `client` for an
    OpenAI-compatible one (Ollama) and nothing upstream changes.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    messages = build_messages(sheets, shortlist)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT, messages=messages
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    proposal = parse_proposal(text)

    if logger is not None:
        usage = getattr(resp, "usage", None)
        logger.log(
            "judge",
            "proposal received",
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            proposal=proposal,  # log the raw proposal — when a pack looks wrong you want the exact output
        )
    return proposal
