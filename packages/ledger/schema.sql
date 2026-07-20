-- Catalog ledger — base schema.
--
-- Idempotent: every statement is IF NOT EXISTS so this file can be replayed
-- as the baseline by the migration runner or the bootstrap snippet in
-- CLAUDE.md without harm. Structural changes after v1 go in migrations/,
-- never by editing this file in place.
--
-- Keys are opaque ULIDs (TEXT). external_ref carries a SKU and is NULL until
-- SKUs exist; nothing ever keys on it. display_name is cosmetic — never
-- embedded, tagged, or used in retrieval.

PRAGMA foreign_keys = ON;

-- Applied-migration bookkeeping. The baseline (this file) is recorded as
-- version '0000_baseline' by the runner.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- A sellable product: one folder of photos the user grouped on screen.
CREATE TABLE IF NOT EXISTS products (
    id           TEXT PRIMARY KEY,
    external_ref TEXT,                       -- SKU, NULL for now; never a key
    display_name TEXT,                       -- cosmetic only
    hero_asset_id TEXT REFERENCES assets(id),-- proposed hero shot (layer 10)
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- A source image belonging to a product, after intake/normalization.
CREATE TABLE IF NOT EXISTS assets (
    id          TEXT PRIMARY KEY,
    product_id  TEXT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,               -- normalized master on disk
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    short_side  INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ok'
                CHECK (status IN ('ok', 'quarantine')),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_assets_product ON assets(product_id);

-- A marketplace derivative rendered from an asset by a named profile.
-- profile is stored as a plain string; the ledger records it but imageops
-- never learns what it means.
CREATE TABLE IF NOT EXISTS derivatives (
    id         TEXT PRIMARY KEY,
    asset_id   TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    profile    TEXT NOT NULL,                -- 'etsy' | 'web' | 'preview' | 'thumb'
    path       TEXT NOT NULL,
    format     TEXT NOT NULL,                -- 'JPEG' | 'WEBP' | ...
    width      INTEGER NOT NULL,
    height     INTEGER NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (asset_id, profile)
);

-- Machine-derived metadata. source is required and weights shortlist ranking.
CREATE TABLE IF NOT EXISTS tags (
    id         TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,                -- 'color' | 'category' | 'material' | ...
    value      TEXT NOT NULL,
    source     TEXT NOT NULL
               CHECK (source IN ('palette', 'clip', 'manual', 'model')),
    confidence REAL,                         -- NULL for non-scored sources
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_tags_product ON tags(product_id);
CREATE INDEX IF NOT EXISTS idx_tags_kv ON tags(key, value);

-- CLIP vectors. Image and text vectors share one space, so kind distinguishes
-- them. model is required: vectors from different checkpoints are not
-- comparable, and changing models means recomputing everything.
CREATE TABLE IF NOT EXISTS embeddings (
    id         TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    asset_id   TEXT REFERENCES assets(id) ON DELETE CASCADE,  -- NULL for text
    kind       TEXT NOT NULL CHECK (kind IN ('image', 'text')),
    model      TEXT NOT NULL,               -- e.g. 'ViT-B-32/openai'
    dim        INTEGER NOT NULL,
    dtype      TEXT NOT NULL DEFAULT 'float32',
    vector     BLOB NOT NULL,               -- numpy .tobytes(), row-major
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_embeddings_product ON embeddings(product_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_kind_model ON embeddings(kind, model);

-- A proposed pack of products. Nothing the model produces is final: packs land
-- as 'proposed' and a human approves in bulk.
CREATE TABLE IF NOT EXISTS packs (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    rationale  TEXT,                         -- model's written justification
    status     TEXT NOT NULL DEFAULT 'proposed'
               CHECK (status IN ('proposed', 'approved', 'rejected')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS pack_members (
    pack_id    TEXT NOT NULL REFERENCES packs(id) ON DELETE CASCADE,
    product_id TEXT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    PRIMARY KEY (pack_id, product_id)
);
