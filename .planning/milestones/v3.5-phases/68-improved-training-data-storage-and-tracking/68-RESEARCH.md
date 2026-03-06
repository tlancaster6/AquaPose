# Phase 68: Improved Training Data Storage and Tracking - Research

**Researched:** 2026-03-06
**Domain:** SQLite sample store, CLI data management, dataset assembly with symlinks
**Confidence:** HIGH

## Summary

This phase replaces ad-hoc training data storage (scattered directories, freeform paths, no metadata) with a centralized SQLite-backed sample store per model type (OBB, Pose). The store tracks provenance, supports tag-based querying, content-hash deduplication, augmentation lineage, and reproducible dataset assembly via symlinks. The existing `scripts/build_yolo_training_data.py` is folded into a CLI command, and model lineage is tracked in the store database.

The technical domain is well-understood: Python's `sqlite3` module (stdlib, no dependencies), Click CLI commands (already used throughout), and pathlib-based file management. No new dependencies are needed. The main complexity is in the schema design, the import/dedup/upsert logic, and wiring the new store into existing training and assembly workflows.

**Primary recommendation:** Use Python's stdlib `sqlite3` with WAL mode, content-addressed UUIDs for filenames, and symlink-based dataset assembly. Keep the store module self-contained under `training/store.py` with a thin API surface.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Custom lightweight SQLite sample store per project
- Schema designed to map onto FiftyOne sample fields for future migration
- No premature abstraction layer
- Two separate databases: OBB store and Pose store
- Centralized copy model with UUID filenames
- Content hash deduplication with source priority hierarchy (manual > corrected > pseudo)
- Provenance history as JSON array per sample
- Augmented variants as new samples with parent_id, cascade delete on parent upsert
- Augmentation at import time via --augment flag
- Dataset versioning with query recipe + resolved UUID list
- Symlink-based assembly (no file duplication)
- Models table linking trained models to dataset manifests
- Auto-update config.yaml after training with clear message
- No separate `model use` command
- COCO conversion as separate CLI command from import
- Soft delete (exclude tag) and hard delete (--purge) with cascade

### Claude's Discretion
- Internal module structure and function signatures
- SQLite schema column types and index design
- CLI command naming and option naming within the `data` subgroup
- Error handling strategy and logging patterns
- Test structure and coverage approach

### Deferred Ideas (OUT OF SCOPE)
- FiftyOne integration: evaluate after using custom store for one retrain cycle
- `aquapose model use <run_id>` command
- Visual sample browsing
- CVAT integration via FiftyOne's annotate() API
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sqlite3 | stdlib (3.45.1) | Sample store database | Zero-dependency, ACID, WAL mode for concurrent reads, already available |
| click | existing dep | CLI commands for data management | Already used for all AquaPose CLI commands |
| pathlib | stdlib | File path management, symlinks | Already used throughout codebase |
| hashlib | stdlib | Content hashing for dedup (SHA-256) | Fast, collision-resistant, stdlib |
| uuid | stdlib | UUID4 filenames for imported samples | Guaranteed unique, decoupled from source |
| json | stdlib | Provenance history, metadata storage | Already used for sidecars throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| shutil | stdlib | File copy during import | Copy images+labels into managed directory |
| yaml (PyYAML) | existing dep | config.yaml auto-update, dataset.yaml | Already used via yaml.safe_load/yaml.dump |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sqlite3 | TinyDB | Simpler API but no SQL queries, no ACID, worse for complex filtering |
| sqlite3 | FiftyOne | Full-featured but heavy dependency, deferred per user decision |
| Content hash | Filename-based dedup | Fragile, doesn't handle renamed files |

**Installation:**
No new dependencies needed. All tools are Python stdlib or existing project deps.

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/training/
    store.py             # SampleStore class (SQLite operations, import, query, dedup)
    store_schema.py      # SQL DDL constants, migration helpers
    data_cli.py          # Click commands: import, assemble, list, exclude, remove, convert
    dataset_assembly.py  # EXISTING — refactor to query store instead of scanning dirs
    run_manager.py       # EXISTING — extend with models table write-back
    coco_interchange.py  # EXISTING — reused by convert command
    elastic_deform.py    # EXISTING — called during --augment import
```

### Pattern 1: SampleStore Class
**What:** A single class wrapping SQLite operations for one store (OBB or Pose).
**When to use:** All data management operations.
**Example:**
```python
class SampleStore:
    """SQLite-backed sample store for training data management."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.root = db_path.parent  # training_data/obb/ or training_data/pose/
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn
```

### Pattern 2: Content-Hash Dedup with Priority Upsert
**What:** SHA-256 hash of image bytes as dedup key. On conflict, higher-priority source replaces lower.
**When to use:** Every import operation.
**Example:**
```python
SOURCE_PRIORITY = {"pseudo": 0, "corrected": 1, "manual": 2}

def _compute_content_hash(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()

def import_sample(self, image_path: Path, label_path: Path,
                  source: str, metadata: dict) -> str:
    content_hash = self._compute_content_hash(image_path)
    existing = self._find_by_hash(content_hash)
    if existing:
        if SOURCE_PRIORITY[source] >= SOURCE_PRIORITY[existing["source"]]:
            self._upsert(existing["id"], label_path, source, metadata)
            return existing["id"]
        else:
            logger.info("Skipping lower-priority import for %s", image_path)
            return existing["id"]
    # New sample: copy files, insert row
    sample_id = str(uuid.uuid4())
    # ... copy + insert
    return sample_id
```

### Pattern 3: Symlink-Based Assembly
**What:** Assembled datasets use symlinks to store files, no duplication.
**When to use:** `aquapose data assemble` command.
**Example:**
```python
def assemble_dataset(self, name: str, query: dict, seed: int) -> Path:
    dataset_dir = self.root / "datasets" / name
    # Create YOLO structure with symlinks
    for sample in self.query(query):
        img_link = dataset_dir / "images" / "train" / f"{sample['id']}.jpg"
        img_link.symlink_to(self.images_dir / f"{sample['id']}.jpg")
    # Store manifest: query recipe + resolved UUIDs
```

### Pattern 4: Cascade Delete for Augmented Children
**What:** When a parent sample is upserted or deleted, all augmented children are removed (DB rows + files).
**When to use:** Import with upsert, `aquapose data remove`.
**Example:**
```python
def _cascade_delete_children(self, parent_id: str) -> int:
    """Delete augmented children of a sample. Returns count deleted."""
    children = self._conn.execute(
        "SELECT id FROM samples WHERE parent_id = ?", (parent_id,)
    ).fetchall()
    for child in children:
        self._delete_files(child["id"])
    self._conn.execute(
        "DELETE FROM samples WHERE parent_id = ?", (parent_id,)
    )
    return len(children)
```

### Anti-Patterns to Avoid
- **Storing absolute paths in the database:** Use relative paths from the store root. Absolute paths break when projects are moved.
- **Lazy schema creation scattered across methods:** Initialize schema in `__init__` or a dedicated `_ensure_schema()` called once.
- **Mixing store logic with CLI logic:** Keep `store.py` purely data-oriented; CLI formatting and Click decorators in `data_cli.py`.
- **Using ORM (SQLAlchemy):** Overkill for a single-table schema. Raw sqlite3 is simpler and has zero dependencies.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Content hashing | Custom hash function | hashlib.sha256 | Proven, fast, collision-resistant |
| UUID generation | Custom unique ID scheme | uuid.uuid4() | Globally unique, no coordination needed |
| COCO conversion | New converter | Existing coco_interchange.py | Already tested, handles edge cases |
| Elastic augmentation | New deformation code | Existing elastic_deform.py | Already debugged through A/B experiment |
| YAML config updates | Manual file editing | yaml.safe_load + yaml.dump | Round-trip safe (comments stripped, accepted per project decisions) |
| Symlink creation | os.symlink | pathlib.Path.symlink_to | Cross-platform, cleaner API |

**Key insight:** The "new" code in this phase is primarily the SQLite store, import logic, and CLI commands. All heavy processing (COCO conversion, augmentation, geometry) already exists and should be called, not reimplemented.

## Common Pitfalls

### Pitfall 1: SQLite Concurrent Write Locks
**What goes wrong:** Multiple processes try to write to the same SQLite database simultaneously, causing "database is locked" errors.
**Why it happens:** SQLite allows only one writer at a time (even with WAL mode).
**How to avoid:** This is a single-user CLI tool, not a server. WAL mode is sufficient. Use `with conn:` context manager for transactions. Set a reasonable `busy_timeout` (e.g., 5 seconds) as safety.
**Warning signs:** "database is locked" errors during import.

### Pitfall 2: Symlink Target Paths
**What goes wrong:** Symlinks break when using absolute paths and the project is moved, or when using relative paths calculated incorrectly.
**Why it happens:** `Path.symlink_to()` creates the link with whatever target you give it.
**How to avoid:** Use `os.path.relpath()` to compute the relative path from the symlink location to the target file. This keeps assembled datasets portable.
**Warning signs:** `FileNotFoundError` when training on an assembled dataset.

### Pitfall 3: Cascade Delete Ordering
**What goes wrong:** Deleting a parent before its children leaves orphan files on disk (even if DB rows cascade via foreign key).
**Why it happens:** SQLite's ON DELETE CASCADE removes DB rows but not filesystem files.
**How to avoid:** Always delete children's files first, then delete parent's files, then let the DB cascade handle rows. Or: query children explicitly, delete their files, then delete parent row with CASCADE.
**Warning signs:** Orphan image/label files accumulating in the store directory.

### Pitfall 4: Content Hash Collisions Between OBB and Pose
**What goes wrong:** Same source image produces identical content hash in both OBB and Pose stores.
**Why it happens:** OBB stores full-frame images; Pose stores crops. Same frame could appear in OBB store, but crops from that frame are different images with different hashes.
**How to avoid:** This is actually fine by design -- separate databases mean hashes are scoped per store. No cross-store dedup needed. The `import_batch_id` metadata field tracks cross-type provenance.

### Pitfall 5: UUID Filenames and Ultralytics
**What goes wrong:** Ultralytics expects matching stems between images/ and labels/ directories.
**Why it happens:** YOLO format matches `images/train/foo.jpg` to `labels/train/foo.txt` by stem.
**How to avoid:** Use the same UUID for both image and label filenames. The store already guarantees this.

### Pitfall 6: Augmented Sample Provenance on Re-import
**What goes wrong:** User re-imports a sample (upsert), augmented children are cascade-deleted, but user forgets to re-run augmentation.
**Why it happens:** Cascade delete is silent about augmentation loss.
**How to avoid:** Print a clear warning message: "Deleted N augmented variants of upserted sample. Re-run with --augment to regenerate."

## Code Examples

### SQLite Schema (verified pattern from stdlib docs)
```python
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,           -- UUID4
    content_hash TEXT NOT NULL,    -- SHA-256 of image bytes
    source TEXT NOT NULL,          -- 'manual', 'corrected', 'pseudo'
    image_path TEXT NOT NULL,      -- relative to store root
    label_path TEXT NOT NULL,      -- relative to store root
    parent_id TEXT,                -- UUID of parent (for augmented variants)
    import_batch_id TEXT,          -- shared across OBB+Pose from same COCO import
    tags TEXT DEFAULT '[]',        -- JSON array of tag strings
    provenance TEXT DEFAULT '[]',  -- JSON array of provenance entries
    metadata TEXT DEFAULT '{}',    -- JSON dict of additional metadata
    created_at TEXT NOT NULL,      -- ISO 8601 timestamp
    updated_at TEXT NOT NULL,      -- ISO 8601 timestamp
    FOREIGN KEY (parent_id) REFERENCES samples(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_content_hash ON samples(content_hash);
CREATE INDEX IF NOT EXISTS idx_source ON samples(source);
CREATE INDEX IF NOT EXISTS idx_parent_id ON samples(parent_id);
CREATE INDEX IF NOT EXISTS idx_import_batch_id ON samples(import_batch_id);

CREATE TABLE IF NOT EXISTS datasets (
    name TEXT PRIMARY KEY,
    query_recipe TEXT NOT NULL,    -- JSON dict of filter params
    sample_ids TEXT NOT NULL,      -- JSON array of resolved UUIDs
    split_seed INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    run_id TEXT PRIMARY KEY,
    dataset_name TEXT,
    weights_path TEXT NOT NULL,
    model_type TEXT NOT NULL,      -- 'obb' or 'pose'
    metrics TEXT DEFAULT '{}',     -- JSON dict from summary.json
    tag TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (dataset_name) REFERENCES datasets(name)
);
"""
```

### CLI Command Structure
```python
# data_cli.py
@click.group("data")
def data_group() -> None:
    """Manage training data stores."""

@data_group.command("import")
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--store", required=True, type=click.Choice(["obb", "pose"]))
@click.option("--source", required=True, type=click.Choice(["manual", "corrected", "pseudo"]))
@click.option("--input-dir", required=True, type=click.Path(exists=True))
@click.option("--augment", is_flag=True, help="Generate elastic augmentation variants")
@click.option("--batch-id", default=None, help="Shared import batch ID for cross-type tracking")
def import_cmd(...): ...

@data_group.command("assemble")
# ... query filters, name, seed

@data_group.command("list")
# ... summary of store contents

@data_group.command("exclude")
# ... soft delete by ID or filter

@data_group.command("remove")
@click.option("--purge", is_flag=True)
# ... hard delete

@data_group.command("convert")
# ... COCO -> YOLO conversion (replaces build_yolo_training_data.py)
```

### Content Hash Computation
```python
import hashlib

def compute_content_hash(path: Path) -> str:
    """Compute SHA-256 content hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

### Provenance History Update
```python
def _append_provenance(existing: list[dict], action: str, source: str,
                       batch_id: str | None) -> list[dict]:
    entry = {
        "action": action,
        "source": source,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if batch_id:
        entry["import_batch_id"] = batch_id
    return existing + [entry]
```

### Config Auto-Update After Training
```python
def update_config_weights(config_path: Path, model_type: str,
                          weights_path: Path) -> None:
    """Update project config.yaml with new model weights path."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    key_map = {"obb": "detection", "pose": "midline"}
    section = key_map.get(model_type)
    if section and section in config:
        config[section]["weights_path"] = str(weights_path)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo(click.style(
        f"Updated {config_path}: {section}.weights_path = {weights_path}",
        bold=True, fg="green"
    ))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Scattered directories with path-encoded provenance | SQLite store with metadata | This phase | Queryable, reproducible, navigable |
| `scripts/build_yolo_training_data.py` (argparse) | `aquapose data convert` (Click) | This phase | Consistent CLI, discoverable |
| File copy for dataset assembly | Symlinks to store | This phase | No disk duplication |
| Manual model path management | Models table + config auto-update | This phase | Lineage tracking, less friction |

**Deprecated/outdated:**
- `scripts/build_yolo_training_data.py`: Replaced by `aquapose data convert` command. Script to be deleted.
- Direct file-scan assembly in `dataset_assembly.py`: Replaced by store query + symlink assembly.

## Open Questions

1. **Tag storage format**
   - What we know: Tags stored as JSON array in TEXT column. FiftyOne uses a dedicated tags collection.
   - What's unclear: Whether JSON array queries (`json_each`) perform well enough or if a junction table is needed.
   - Recommendation: Start with JSON array in TEXT column. Sample counts are small (hundreds to low thousands). Migrate to junction table only if query performance becomes an issue.

2. **Existing data migration**
   - What we know: User has manual annotations and pseudo-labels in the current directory structure.
   - What's unclear: Whether a one-time migration script is needed or users will re-import from COCO sources.
   - Recommendation: Provide an import command that can ingest existing YOLO-format directories (auto-detect source type). No complex migration needed since the data volume is small.

3. **Store initialization timing**
   - What we know: Store database and directories need to exist before any data operation.
   - What's unclear: Whether to auto-create on first use or require explicit `aquapose data init`.
   - Recommendation: Auto-create store directories and database on first import. No explicit init command needed.

## Sources

### Primary (HIGH confidence)
- Python sqlite3 stdlib docs -- schema patterns, WAL mode, Row factory
- Existing codebase analysis -- `training/cli.py`, `training/dataset_assembly.py`, `training/run_manager.py`, `training/coco_interchange.py`, `training/elastic_deform.py`, `scripts/build_yolo_training_data.py`
- CONTEXT.md user decisions -- all architectural choices locked

### Secondary (MEDIUM confidence)
- FiftyOne sample field schema -- informed column naming for future migration compatibility

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All stdlib, no new dependencies, well-understood
- Architecture: HIGH - Schema design is straightforward, all patterns are proven
- Pitfalls: HIGH - SQLite, symlinks, and filesystem operations are well-characterized domains

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable domain, no fast-moving dependencies)
