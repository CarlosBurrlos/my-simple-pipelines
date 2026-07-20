# my-simple-pipelines

Monorepo of small pipelines. First and only one today: `catalog`.

Takes product photos from an iPhone, normalizes them, tags them
mechanically, produces marketplace derivatives, and proposes sellable
"packs" via an LLM. Everything lands in a SQLite ledger. A human
approves proposals in bulk; that is the only manual step.

---

## Structure

```
my-simple-pipelines/
├── packages/
│   ├── imageops/       # normalize, square_crop, resize, encode, probe
│   ├── ledger/         # SQLite schema, migrations, queries
│   └── tagging/        # palette match, CLIP embed, theme scoring
├── pipelines/
│   └── catalog/
│       ├── watcher.py  # observer -> queue        (LATER, not v1)
│       ├── worker.py   # quiescence -> verify -> dispatch  (LATER)
│       ├── ui/         # FastAPI upload + grouping  (v1 entry point)
│       └── stages/     # intake, derive, tag, judge, write
├── config/
│   ├── profiles.yaml   # etsy / web / preview / thumb
│   ├── palette.yaml    # forked from meodai/color-names
│   └── themes.yaml     # empty in v1
├── data/
│   └── catalog.db
└── pyproject.toml      # workspace, packages as local path deps
```

### The one architectural rule

**`packages/` never imports from `pipelines/`.**

`imageops` in particular reads no config, knows nothing about products
or packs, and takes every parameter explicitly. This is what lets it be
lifted into an unrelated project later. If you find yourself passing a
profile name into `imageops`, stop.

---

## Layers

Numbered for reference. All of them stay in the design; some are
deliberately unimplemented in v1.

| # | Layer | Deps | v1? |
|---|-------|------|-----|
| 1 | Watch | `watchdog`, `queue.Queue` | later |
| 2 | Quiesce & verify | `os.walk`, Pillow `.verify()` | later |
| 3 | Normalize & probe | Pillow, `pillow-heif` | yes |
| 4 | Color tagging | KMeans, `color-names` | yes |
| 5 | Image embedding | CLIP vision encoder | yes |
| 6 | Attribute tagging | CLIP text encoder, Shopify taxonomy | yes |
| 7 | Text embedding | CLIP text encoder | yes |
| 8 | Derivatives | `imageops` + `profiles.yaml` | yes |
| 9 | Shortlist | numpy cosine + SQL over tags | yes |
| 10 | Judgment | `anthropic` SDK | yes |
| 11 | Ledger write | sqlite3 (WAL) | yes |
| 12 | Approval | CLI | yes |
| 13 | Artifacts | sqlite3 + Jinja2 | yes |

CLIP appears in layers 5, 6, and 7. One model, three jobs. Its vision
and text encoders output into the **same** vector space, which is why
image-to-image, text-to-image, and theme-to-image retrieval are all the
same cosine operation.

---

## Decisions already made — do not relitigate

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
  losing shots; keeps layers 3-8 fully unattended.
- **Manual grouping in v1.** CLIP pre-clustering is a v2 upgrade.
- **Theme scoring off in v1.** `themes.yaml` stays empty. The model
  proposes packs by reading tag sets from the ledger. Populate themes
  later, from concepts you notice missing.
- **Packs are proposed catalog-wide**, not per-batch.
- **Full-auto proposals**, but always written `status='proposed'`.
  Approval is bulk (`approve --batch`), never per-item.
- **Quarantine under 2000px short side.** Never upscale silently.
- **Tag `source`** (`palette` | `clip` | `manual` | `model`) is required
  on every tag and weights shortlist ranking.

---

## Vendored data

- **[meodai/color-names](https://github.com/meodai/color-names)** (MIT) —
  hex to human-readable name. Fork and prune into `config/palette.yaml`.
  Solves "military green" without hand-building a palette.
- **[Shopify/product-taxonomy](https://github.com/Shopify/product-taxonomy)** —
  use `dist/` JSON. Provides the controlled vocabulary for attribute
  tags AND the candidate label list for CLIP zero-shot scoring.

---

## Shortlisting (layer 9)

Catalog-wide packing cannot ship 5,000 images to a model. So retrieve a
candidate set cheaply, then spend the model only on that.

Union of:
- **top-K by image vector** — visual/tonal neighbors
- **top-K by text vector** — functional neighbors (this is what catches
  a mug + notebook + lamp "desk setup"; image vectors never will)
- **tag overlap** — plain SQL over `tags`
- **pack centroids** — mean of member vectors; skip until packs exist

Tune for **recall**, generously. Anything dropped here the model never
sees.

---

## Build order

1. `packages/imageops` + tests (done — port from handoff)
2. `packages/ledger` schema + migration runner (done — port from handoff)
3. `config/profiles.yaml` (done), then vendor `palette.yaml`
4. `stages/intake.py` — layer 3, HEIC in, quarantine out
5. `stages/color.py` — layer 4
6. `pipelines/catalog/ui/` — FastAPI drop zone + grouping
7. `packages/tagging` — CLIP load, embed, taxonomy scoring (5/6/7)
8. `stages/derive.py` — layer 8
9. `stages/judge.py` + shortlist — 9/10
10. approval CLI + artifacts — 12/13

---

## Gotchas

- iPhone shoots **HEIC**. Pillow cannot open it without `pillow-heif`.
- CLIP resizes to **224x224** internally. Fine texture and small logos
  are invisible to it; composition, color, and subject survive.
- Embeddings from different checkpoints are **not comparable**. Store
  the model name; changing models means recomputing everything.
- CLIP confidence scores are **not calibrated**. Thresholds need tuning
  against real images, which is why v1 skips theme scoring entirely.
- Transparent pixels render **black** on Etsy. Always flatten to white.
- WAL mode on SQLite, since the worker writes while you read.

---

## Current State

This repository is newly initialized. As of the initial commit it contains only a `README.md` with the project title (`my-simple-pipelines`) — there is no source code, build tooling, dependency manifest, or tests yet.

There are consequently no build, lint, or test commands to document at this time.

However - leverage the above content for projdct/code conventions.

---

# CLAUDE.md — Addendum

Append to the existing `CLAUDE.md`. Conventions and escalation rules only;
architecture and decisions live in the main file.

---

## The boundary rule

**`packages/` never imports from `pipelines/`.**

This is the load-bearing rule of the whole repo. `imageops` exists to be
lifted into unrelated projects. It can only do that while it is
self-contained: pass it a path and some numbers, get an image back.

It never happens deliberately. It happens because it reads tidier:

```python
# packages/imageops/core.py
from pipelines.catalog.config import PROFILES   # <- never

def encode(im, out, profile_name):
    spec = PROFILES[profile_name]
```

Fewer arguments, and now `imageops` knows what `"etsy"` means. Your other
project has no etsy. That one import turns a copy-paste into a rewrite,
and drags the ledger and schema along behind it.

**The signal:** `packages/` learning a *name* from the domain. Sizes,
quality integers, and byte ceilings are fine as arguments. A recognized
string like `"etsy"`, `"pack"`, or `"hero"` is the failure.

Applies to `ledger` and `tagging` too. `tagging` is the one to watch:
CLIP embedding and palette matching are generic, but the moment it reads
`themes.yaml` itself rather than being handed themes, it stops
travelling.

---

## Commits

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

### Scopes

Optional. Use these when applicable:

`imageops` · `ledger` · `tagging` · `catalog` · `ui` · `config`

Never use issue identifiers as scopes.

### Rules

- Imperative present tense: "add", not "added" or "adds"
- No capital first letter, no trailing period
- Breaking changes: `!` before the colon — `feat(ledger)!: drop batch_id`
- Breaking changes get a footer starting `BREAKING CHANGE:`
- Initial commit is `chore: init`
- Merge and revert commits keep git's default message format

### Examples

```
feat(ui): add drag-and-drop grouping for uploaded photos
fix(imageops): honour EXIF rotation on HEIC input
refactor(tagging): extract cosine shortlist into its own module
build: add pillow-heif for iPhone photo support
ops: add boundary-check subagent for packages/ import rule
docs: record why display_name is cosmetic only
chore: init
```

Schema changes resist the table: `feat(ledger)` when a column enables a
feature, `refactor(ledger)` when structural, `fix(ledger)` when the old
schema was wrong. Pick one and move on — not worth a rule.

---

## When to stop and ask

Ask before proceeding if:

- A schema change would drop a column or lose existing rows
- The work requires contradicting a decision in `CLAUDE.md`
- A new runtime dependency is needed
- A change would make `packages/` depend on `pipelines/`
- Anything under `data/` would be deleted or overwritten

Otherwise: choose, state the assumption in one line, and continue.

**Ambiguous commit types, naming, and file layout are not worth a
prompt.** Cheap questions devalue expensive ones — if every trivial
ambiguity triggers a prompt, the prompts get rubber-stamped, and the one
that mattered goes through with the rest.

The test is cost asymmetry: a mis-typed commit costs an `--amend`, a
dropped column costs data. Ask about the second kind only.
