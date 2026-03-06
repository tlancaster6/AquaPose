"""SQL DDL constants for the SampleStore SQLite database."""

from __future__ import annotations

SCHEMA_VERSION = 1

SOURCE_PRIORITY: dict[str, int] = {
    "pseudo": 0,
    "corrected": 1,
    "manual": 2,
}
"""Source priority for deduplication upsert logic.

Higher numeric value wins when the same image (by content hash) is imported
from multiple sources.
"""

# Expected metadata keys (conventional, not enforced):
#   "confidence" (float, 0-1): per-image mean confidence score (pseudo-labels)
#   "run_id" (str): source pipeline run ID
#   "gap_reason" (str): gap classification reason (e.g., "occlusion", "low_visibility")
#   "curvature_bin" (int): diversity sampling bin index
#   "camera_id" (str): source camera identifier

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS samples (
    id            TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    source        TEXT NOT NULL,
    image_path    TEXT NOT NULL,
    label_path    TEXT NOT NULL,
    parent_id     TEXT,
    import_batch_id TEXT,
    tags          TEXT NOT NULL DEFAULT '[]',
    provenance    TEXT NOT NULL DEFAULT '[]',
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES samples(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS datasets (
    name          TEXT PRIMARY KEY,
    query_recipe  TEXT NOT NULL,
    sample_ids    TEXT NOT NULL,
    split_seed    INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    run_id        TEXT PRIMARY KEY,
    dataset_name  TEXT,
    weights_path  TEXT NOT NULL,
    model_type    TEXT NOT NULL,
    metrics       TEXT NOT NULL DEFAULT '{}',
    tag           TEXT,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (dataset_name) REFERENCES datasets(name)
);

CREATE INDEX IF NOT EXISTS idx_content_hash ON samples(content_hash);
CREATE INDEX IF NOT EXISTS idx_source ON samples(source);
CREATE INDEX IF NOT EXISTS idx_parent_id ON samples(parent_id);
CREATE INDEX IF NOT EXISTS idx_import_batch_id ON samples(import_batch_id);
"""
