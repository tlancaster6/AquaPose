# Phase 68: Improved Training Data Storage and Tracking - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the current ad-hoc training data storage (scattered directories, freeform paths, no metadata) with a centralized SQLite-backed sample store that tracks provenance, supports tag-based querying, and enables reproducible dataset assembly. Fold the last remaining script (`scripts/build_yolo_training_data.py`) into the CLI. This phase covers storage infrastructure and data management commands only — CLI reorganization is Phase 69.

</domain>

<decisions>
## Implementation Decisions

### Architecture
- Custom lightweight SQLite sample store per project
- Designed so the metadata schema maps onto FiftyOne sample fields if future migration is desired
- No premature abstraction layer — clean functions behind a simple API, rewrite the backend if migrating later

### Storage layout
- Two separate databases: one for OBB (full-frame), one for Pose (crop-space)
  - `training_data/obb/store.db` + `training_data/obb/images/` + `training_data/obb/labels/`
  - `training_data/pose/store.db` + `training_data/pose/images/` + `training_data/pose/labels/`
- Centralized copy model: import copies images+labels into the managed directory
- UUID filenames for imported samples (guaranteed unique, fully decoupled from source)
- Cross-type provenance tracked via shared `import_batch_id` in metadata (e.g., when a full-frame COCO annotation produces both OBB and Pose entries)

### Deduplication
- Dedup key: content hash of the image (within each store)
- Source priority hierarchy: `manual > corrected > pseudo`
- On conflict: incoming source with higher or equal priority triggers upsert (replace label, update metadata, append to provenance history); lower priority source is skipped with info message
- Augmented variants have different pixel content (elastic deformation) so different content hashes — no special dedup logic needed
- Provenance history stored as JSON array per sample: `[{"action": "imported", "source": "pseudo", ...}, {"action": "replaced", "source": "manual", ...}]`

### Augmentation integration
- Augmented variants are new samples in the store with a `parent_id` pointing to the original
- Augmentation triggered at import time via `--augment` flag
- Tag-based inclusion in assembly (augmented samples get an `augmented` tag, assembly includes/excludes by tag like any other filter)
- Cascade delete: when a parent is upserted (re-imported), all augmented children are deleted (DB rows + files). User re-runs `--augment` if they want new variants. This applies even if re-import is done without `--augment`.

### Dataset versioning
- Each assembled dataset stores both the query recipe (filter parameters) AND the resolved list of sample UUIDs
- Assembled datasets live in named subdirectories: `training_data/datasets/{name}/`
- Assembly uses symlinks from the assembled directory to the store (no file duplication)
- Models table in the same SQLite database links trained models to their dataset manifests

### Model lineage
- Trained models tracked in a `models` table within the store database
- After training completes, `config.yaml` is automatically updated to point to the new best model weights
- A clear message is printed at the end of training showing what was updated
- No separate `model use` command — users edit config.yaml manually to switch models

### COCO conversion
- `scripts/build_yolo_training_data.py` functionality folded into a new CLI command (separate from import)
- Conversion command produces standard YOLO-format output; user then chooses what to import from that output
- This allows selective import (e.g., import OBB only from COCO annotations with an outdated skeleton schema)

### Data removal
- Soft delete (exclude tag) available via `aquapose data exclude` — samples stay in store but filtered out of assembly queries by default. Reversible.
- Hard delete via `aquapose data remove --purge` — deletes DB rows + files on disk. Permanent.
- Both cascade to augmented children

</decisions>

<specifics>
## Specific Ideas

### Motivation
- Two equal pain points: navigability (hard to understand what exists after a break) and retrain friction (too many manual path-stitching steps)
- The current system encodes provenance via file location rather than metadata, forcing every data flow to maintain its own directory structure

### Full retrain cycle to support
1. Initial training: manual COCO annotations → convert → import → (optionally augment) → assemble → train first model
2. Pipeline run → pseudo-label generation → import pseudo-labels to store
3. Optional CVAT correction loop: export pseudo-labels → correct in CVAT → re-import as corrected (upserts to manual priority)
4. Assemble new training set from store (query by source, confidence, tags) → train improved model
5. Config auto-updates to new model weights → repeat from step 2

### Key sample types
- Full-frame OBB: camera frames with oriented bounding boxes (OBB store)
- Crop-space pose: single-fish crops with 6 keypoints (Pose store)
- These are never mixed in a single training set, hence separate databases
- Pseudo-labels may only cover a subset of fish in a frame (confidence-filtered), which is why a pure full-frame canonical format doesn't work for Pose

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `training/dataset_assembly.py`: Current assembly logic (collect, filter, split, copy). Core filtering logic reusable; file-copy logic replaced by store queries + symlinks
- `training/run_manager.py`: Training run management, summary.json, provenance extraction. Models table extends this
- `training/coco_interchange.py`: COCO-to-YOLO and YOLO-to-COCO conversion. Used by the new convert command
- `training/pseudo_label_cli.py`: Pseudo-label generation. Output format stays the same; a new import step brings results into the store
- `training/elastic_deform.py`: Elastic augmentation. Called during `--augment` import
- `training/frame_selection.py`: Temporal subsampling and diversity sampling. Usable as assembly filters
- `training/geometry.py`: OBB computation, affine crops, keypoint transforms. Used by COCO conversion
- `scripts/build_yolo_training_data.py`: Full-frame COCO → YOLO-OBB + crop-space YOLO-Pose. To be folded into CLI

### Established Patterns
- Click-based CLI with subgroups attached via `cli.add_command()`
- Config-centric: most commands read `config.yaml` for path resolution
- YOLO directory convention: `images/{train,val}/` + `labels/{train,val}/` + `dataset.yaml`
- Run directories are timestamped: `run_{YYYYMMDD_HHMMSS}/`
- Confidence sidecars: `confidence.json` per pseudo-label batch

### Integration Points
- `training/cli.py`: New `data` subgroup for store management commands
- `config.yaml`: Auto-update `detection.weights_path` / `midline.weights_path` after training
- `training/pseudo_label_cli.py`: After generation, add import-to-store step (or separate command)
- `scripts/build_yolo_training_data.py`: Logic moves to new CLI command, script deleted

</code_context>

<deferred>
## Deferred Ideas

- FiftyOne integration: SQLite schema designed to be migratable to FiftyOne sample fields. Evaluate after using custom store for one retrain cycle.
- `aquapose model use <run_id>` command for switching active models without editing config — not needed now, add if manual config editing becomes painful
- Visual sample browsing (FiftyOne App equivalent) — custom store would need a separate viewer tool
- CVAT integration via FiftyOne's `annotate()` API — currently handled via manual COCO export/import cycle which works fine

</deferred>

---

*Phase: 68-improved-training-data-storage-and-tracking*
*Context gathered: 2026-03-06*
