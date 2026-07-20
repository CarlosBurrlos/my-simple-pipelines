# my-simple-pipelines

Monorepo of small pipelines. First and only one today: `catalog`.

Takes product photos from an iPhone, normalizes them, tags them
mechanically, produces marketplace derivatives, and proposes sellable
"packs" via an LLM. Everything lands in a SQLite ledger. A human
approves proposals in bulk; that is the only manual step.

Python throughout. One toolchain, one `pyproject.toml`. The only
non-Python pieces are a static HTML page for the upload UI and two
vendored JSON data files.

---

## 1. Setup

macOS. **Python 3.12.0**, pinned in `.python-version`.

Dependencies live in `pyproject.toml` — no `requirements.txt`, and don't
add one. Any package manager works; **uv preferred**.

```bash
uv sync          # or: pip install -e ".[dev]"

# initialise the ledger
uv run python -c "import sqlite3,pathlib; \
  sqlite3.connect('data/catalog.db').executescript( \
  pathlib.Path('packages/ledger/schema.sql').read_text())"
```

The workspace wires `packages/*` as local path deps, which is what makes
`from imageops import ...` work from anywhere in the repo without
installing or publishing anything.

### Dependencies

| Package | Why | Layer |
|---|---|---|
| `pillow` | all image work | 3, 8 |
| `pillow-heif` | **iPhone HEIC — Pillow cannot open it otherwise** | 3 |
| `numpy` | cosine similarity, vector storage | 5, 9 |
| `scikit-learn` | KMeans for dominant colour | 4 |
| `open_clip_torch` + `torch` | CLIP vision + text encoders | 5, 6, 7 |
| `fastapi`, `uvicorn`, `python-multipart` | upload UI + SSE | 6 |
| `pyyaml` | config files | all |
| `python-ulid` | opaque primary keys | 11 |
| `anthropic` | judgment call | 10 |
| `jinja2` | pack artifacts | 13 |
| `watchdog` | filesystem observer | 1 (later) |

### Vendored data — do this before layer 4

```bash
mkdir -p config/vendor

# colour naming: hex -> "military green"
curl -L -o config/vendor/color-names.json \
  https://raw.githubusercontent.com/meodai/color-names/master/dist/colornames.json

# controlled vocabulary + CLIP candidate labels
git clone --depth 1 https://github.com/Shopify/product-taxonomy /tmp/taxonomy
cp -r /tmp/taxonomy/dist/en/*.json config/vendor/
```

Then prune `color-names.json` into `config/palette.yaml` — the full list
is huge and you only want names you'd actually put on a listing.

### Apple Silicon note

`torch` uses the MPS backend on M-series. First CLIP load pulls weights
(~600MB for ViT-B/32) and takes a minute; subsequent loads are cached in
`~/.cache/huggingface`. Model load dominates cold-start, so keep a warm
process during development rather than re-running a script per image.

### Running it

```bash
uv run uvicorn pipelines.catalog.ui.app:app --reload   # upload UI + /runs
```

### Local judgment (optional)

Ollama is installed. To use it instead of the API for layer 10:

```bash
ollama serve
ollama pull qwen2.5vl:7b        # vision model — must be multimodal
```

`stages/judge.py` exposes `propose(shortlist, sheets) -> dict` and
nothing else knows what is behind it. Ollama serves an OpenAI-compatible
endpoint at `http://localhost:11434/v1`, so switching is a base-URL and
client change inside that one file.

Default to the Anthropic API. Aesthetic ranking and theme naming are
where small local VLMs are weakest, and this is one call per batch — the
cheapest part of the pipeline to optimise.

### Telemetry (later, not v1)

```bash
docker run --rm -it -p 18888:18888 -p 4317:18889 -p 4318:18890 \
  --name aspire-dashboard mcr.microsoft.com/dotnet/aspire-dashboard:latest
```

Accepts OTLP from any language; no .NET involved. In-memory only, so
restarts lose data. Worth adding once the UI, worker, and CLIP inference
are separate processes — not before.

---

## 2. Structure

```
my-simple-pipelines/
├── packages/
│   ├── imageops/       # normalize, square_crop, resize, encode, probe
│   ├── ledger/         # SQLite schema, migrations, queries
│   └── tagging/        # palette match, CLIP embed, theme scoring
├── pipelines/
│   └── catalog/
│       ├── watcher.py  # observer -> queue                  (LATER)
│       ├── worker.py   # quiescence -> verify -> dispatch   (LATER)
│       ├── ui/         # FastAPI upload + grouping + /runs  (v1 entry)
│       └── stages/     # intake, color, derive, judge, write
├── config/
│   ├── profiles.yaml   # etsy / web / preview / thumb
│   ├── palette.yaml    # pruned from vendor/color-names.json
│   ├── themes.yaml     # empty in v1
│   └── vendor/         # unmodified upstream data
├── data/
│   ├── catalog.db
│   └── logs/
└── pyproject.toml      # workspace, packages as local path deps
```

---

## 3. The one architectural rule

**`packages/` never imports from `pipelines/`.**

This is load-bearing. `imageops` exists to be lifted into unrelated
projects, and it can only do that while self-contained: pass it a path
and some numbers, get an image back.

It never happens deliberately. It happens because it reads tidier:

```python
# packages/imageops/core.py
from pipelines.catalog.config import PROFILES   # <- never

def encode(im, out, profile_name):
    spec = PROFILES[profile_name]
```

Fewer arguments — and now `imageops` knows what `"etsy"` means. Your
other project has no etsy. That one import turns a copy-paste into a
rewrite, and drags the ledger and schema along behind it.

**The signal:** `packages/` learning a *name* from the domain. Sizes,
quality integers, byte ceilings are fine as arguments. A recognised
string like `"etsy"`, `"pack"`, or `"hero"` is the failure.

Applies to `ledger` and `tagging` too. `tagging` is the one to watch:
CLIP embedding and palette matching are generic, but the moment it reads
`themes.yaml` itself rather than being handed themes, it stops
travelling.

---

## 4. Layers

All stay in the design; some are deliberately unimplemented in v1.

| # | Layer | Deps | v1? |
|---|-------|------|-----|
| 1 | Watch | `watchdog`, `queue.Queue` | later |
| 2 | Quiesce & verify | `os.walk`, Pillow `.verify()` | later |
| 3 | Normalize & probe | Pillow, `pillow-heif` | yes |
| 4 | Colour tagging | KMeans, `color-names` | yes |
| 5 | Image embedding | CLIP vision encoder | yes |
| 6 | Attribute tagging | CLIP text encoder, Shopify taxonomy | yes |
| 7 | Text embedding | CLIP text encoder | yes |
| 8 | Derivatives | `imageops` + `profiles.yaml` | yes |
| 9 | Shortlist | numpy cosine + SQL over tags | yes |
| 10 | Judgment | `anthropic` SDK | yes |
| 11 | Ledger write | sqlite3 (WAL) | yes |
| 12 | Approval | CLI | yes |
| 13 | Artifacts | sqlite3 + Jinja2 | yes |

CLIP appears in 5, 6, and 7 — one model, three jobs. Its vision and text
encoders output into the **same** vector space, which is why
image-to-image, text-to-image, and theme-to-image retrieval are all the
same cosine operation.

### Layer 10 in one paragraph

The only stage requiring a model. It receives contact sheets (one
composite image per product, thumbnails of every candidate shot with
filenames printed underneath) plus the layer-9 shortlist and its tags.
It returns two things: the hero shot per product, and pack proposals
with written rationale. One API call per batch. Nothing it produces is
final — everything lands as `status='proposed'`.

---

## 5. Decisions already made — do not relitigate

- **Monorepo**, not split repos. Split when two pipelines need different
  versions of a package, not before.
- **Opaque ULID primary keys.** `products.external_ref` holds a SKU and
  is NULL for now. Never key on SKU even once they exist.
- **Identity comes from folder-per-product**, not filenames. The UI lets
  the user group photos on screen; no bulk-rename step exists.
- **`display_name` is cosmetic only.** Never embedded, never a tag, never
  influences retrieval. All real metadata is machine-derived so human
  fatigue cannot corrupt it.
- **Derivatives run pre-gate**, before judgment. Wastes a little work on
  losing shots; keeps layers 3–8 fully unattended.
- **Manual grouping in v1.** CLIP pre-clustering is a v2 upgrade.
- **Theme scoring off in v1.** `themes.yaml` stays empty. The model
  proposes packs from tag sets. Populate themes later, from concepts you
  notice missing.
- **Packs are proposed catalog-wide**, not per-batch.
- **Full-auto proposals**, but always written `status='proposed'`.
  Approval is bulk (`approve --batch`), never per-item.
- **Quarantine under 2000px short side.** Never upscale silently.
- **Tag `source`** (`palette` | `clip` | `manual` | `model`) is required
  on every tag and weights shortlist ranking.
- **Judgment is one API call**, not multiple agents. If it outgrows one
  call, split by product — not by "agent".

---

## 6. Shortlisting (layer 9)

Catalog-wide packing cannot ship 5,000 images to a model. Retrieve a
candidate set cheaply, then spend the model only on that.

Union of:
- **top-K by image vector** — visual/tonal neighbours
- **top-K by text vector** — functional neighbours (catches a mug +
  notebook + lamp "desk setup"; image vectors never will)
- **tag overlap** — plain SQL over `tags`
- **pack centroids** — mean of member vectors; skip until packs exist

Tune for **recall**, generously. Anything dropped here the model never
sees.

---

## 7. Logging

- JSONL to `data/logs/run-{run_id}.jsonl`, one object per line. Mirror
  to stdout.
- Every record carries `ts`, `run_id`, `stage`, `product_id`, `level`,
  `msg`, `duration_ms`. These become OTel span attributes later — keep
  them stable.
- Layer 10 additionally logs token counts and the raw proposal JSON.
  When a pack looks wrong you want the exact output, not a summary.
- Flush per line. A crashed worker must not lose its last lines.
- Prune runs older than 14 days.
- Viewed at `/runs` in the FastAPI app. No separate container, no log
  service.

---

## 8. Build order

1. `packages/imageops` + tests
2. `packages/ledger` schema + migration runner
3. `config/profiles.yaml`, then vendor and prune `palette.yaml`
4. `stages/intake.py` — layer 3, HEIC in, quarantine out
5. `stages/color.py` — layer 4
6. `pipelines/catalog/ui/` — FastAPI drop zone, grouping, `/runs`
7. `packages/tagging` — CLIP load, embed, taxonomy scoring (5/6/7)
8. `stages/derive.py` — layer 8
9. `stages/judge.py` + shortlist — 9/10
10. approval CLI + artifacts — 12/13

---

## 9. Conventions

### Commits

Conventional commits. First match wins — evaluate top to bottom, stop.

| Question | Type |
|---|---|
| Fixes incorrect API/UI behavior? | `fix` |
| New or changed feature in API/UI? | `feat` |
| Measurably improves performance? | `perf` |
| Restructures code, no behavior change? | `refactor` |
| Formatting/whitespace only? | `style` |
| Adds or corrects tests? | `test` |
| Documentation only? | `docs` |
| Build tools, dependencies, versions? | `build` |
| Infra, CI/CD, deploy, backups? | `ops` |
| Anything else | `chore` |

`chore` is the fallback, not the shortcut. If you reach for it, re-read
the table.

`fix` is scoped to API/UI bugs. A broken CI step is `ops`. A wrong
dependency pin is `build`. Neither is `fix` — otherwise `fix` becomes
the same lazy catch-all `chore` is guarded against.

Schema changes resist the table: `feat(ledger)` when a column enables a
feature, `refactor(ledger)` when structural, `fix(ledger)` when the old
schema was wrong. Pick one and move on.

**Scopes** (optional): `imageops` · `ledger` · `tagging` · `catalog` ·
`ui` · `config`. Never use issue identifiers as scopes.

**Rules:**
- Imperative present tense: "add", not "added" or "adds"
- No capital first letter, no trailing period
- Breaking changes: `!` before the colon — `feat(ledger)!: drop batch_id`
- Breaking changes get a footer starting `BREAKING CHANGE:`
- Initial commit is `chore: init`
- Merge and revert commits keep git's default message format

```
feat(ui): add drag-and-drop grouping for uploaded photos
fix(imageops): honour EXIF rotation on HEIC input
refactor(tagging): extract cosine shortlist into its own module
build: add pillow-heif for iPhone photo support
docs: record why display_name is cosmetic only
```

### When to stop and ask

Ask before proceeding if:

- A schema change would drop a column or lose existing rows
- The work requires contradicting a decision in section 5
- A new runtime dependency is needed
- A change would make `packages/` depend on `pipelines/`
- Anything under `data/` would be deleted or overwritten

Otherwise: choose, state the assumption in one line, and continue.

**Ambiguous commit types, naming, and file layout are not worth a
prompt.** Cheap questions devalue expensive ones — if every trivial
ambiguity triggers a prompt, the prompts get rubber-stamped and the one
that mattered goes through with the rest. A mis-typed commit costs an
`--amend`; a dropped column costs data. Ask about the second kind only.

---

## 10. Gotchas

- iPhone shoots **HEIC**. Pillow cannot open it without `pillow-heif`.
- CLIP resizes to **224×224** internally. Fine texture and small logos
  are invisible to it; composition, colour, and subject survive.
- Embeddings from different checkpoints are **not comparable**. Store
  the model name; changing models means recomputing everything.
- CLIP confidence scores are **not calibrated**. Thresholds need tuning
  against real images — which is why v1 skips theme scoring entirely.
- Transparent pixels render **black** on Etsy. Always flatten to white.
- WAL mode on SQLite, since the worker writes while you read.
- Etsy does not list WebP among supported formats. WebP is for Shopify
  and your own site; Etsy gets JPG.
