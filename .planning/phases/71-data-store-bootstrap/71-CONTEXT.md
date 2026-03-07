# Phase 71: Data Store Bootstrap - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Import all existing manual annotations into the data store with baseline OBB and pose models trained, registered, and sanity-checked. Most infrastructure (SampleStore, CLI commands, training wrappers) already exists -- this phase exercises the workflow end-to-end and fills two gaps (temporal split, exclusion reasons).

</domain>

<decisions>
## Implementation Decisions

### Temporal split strategy
- Contiguous block holdout: reserve the last 20% of unique frame indices as val
- Split at the frame-index level, not individual image level -- all cameras from a given frame stay in the same split
- Frame index parsed from filename: last section between final underscore and `.png` extension
- Temporal split logic added to `data convert` via `--split-mode temporal|random` flag (default: `random` for backward compat)
- Current data: 10 frame indices (612000-693000, step ~9000), 49 images across 4-6 cameras each

### Convert-import workflow
- `data convert` and `data import` remain separate steps (inspect converted data before committing to store)
- `data import` tags samples from the `val/` subdirectory with a `"val"` tag
- `store.assemble()` gets a new `--split-mode` option:
  - `random` (current default): random split from val-eligible candidates
  - `tagged`: val set = samples with `"val"` tag, everything else is train
- For random mode, a `--val-candidates` tag filter so only matching samples are candidates for val (rest go straight to train)
- This keeps val set consistent across store re-assemblies when using the `tagged` mode

### Exclusion reason tagging
- `data exclude --reason TAG` adds both `"excluded"` and the reason string as separate tags in the JSON tags array
- Free-text reason (no predefined enum) -- e.g. `bad_crop`, `occluded`, `duplicate`
- `SampleStore.exclude()` accepts optional `reason: str | None` parameter -- no schema changes needed
- `data status` shows breakdown by reason tags: e.g. "Excluded: 5 (3 bad_crop, 2 occluded)"

### Baseline training config
- OBB: 100 epochs, yolo26n-obb, mosaic=0.3 (update CLI default from 1.0), patience=100, imgsz=640
- Pose: 100 epochs, yolo26n-pose, mosaic=0.1 (update CLI default from 1.0), imgsz=128 (update from 640), rect=True (add as new flag, default True), patience=100
- Elastic augmentation applied during pose import (`data import --augment`): 4 variants per sample, 5-15 degree range
- No elastic augmentation for OBB (already handled by `data import`)
- Both models tagged as `"baseline"` in the store

### Claude's Discretion
- Exact implementation of frame index parsing from filenames
- How to surface rect training mode in the YOLO training wrappers
- Error handling for edge cases in temporal split (single frame index, etc.)

</decisions>

<specifics>
## Specific Ideas

- Mosaic at 1.0 is wasteful for small targets (fish in full frames for OBB, tiny crops for pose) -- reduced to 0.3/0.1 respectively
- Pose crops are 128x64 but were training at imgsz=640 (upscaling 5x with padding) -- switching to imgsz=128 + rect=True to train at native resolution
- The augmentation experiment showed elastic augmentation reduces curvature bias (OKS-vs-curvature slope: -0.71 to -0.30), so baseline pose model includes augmented variants

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `SampleStore` (`training/store.py`): Full CRUD, content-hash dedup, provenance, import/export, assemble with symlinks
- `data_cli.py`: All CLI commands exist (import, convert, assemble, status, list, exclude, include, remove)
- `coco_convert.py`: COCO-to-YOLO conversion for both OBB and pose formats
- `run_manager.py`: Training run directory management, config snapshot, metrics parsing, model registration
- `training/cli.py`: `train obb`, `train pose`, `train seg`, `train compare` commands with auto-registration
- `elastic_deform.py`: Elastic augmentation integrated into `data import --augment`

### Established Patterns
- Store DB lives at `{project_dir}/training_data/{obb|pose}/store.db`
- Training runs at `{project_dir}/training/{model_type}/run_{timestamp}/`
- Model registration: `run_manager.register_trained_model()` writes to store and updates project config
- Import uses content-hash dedup with source-priority upsert (manual > corrected > pseudo)
- Tags stored as JSON array in samples table; no schema changes needed for reason tags

### Integration Points
- `data convert` reads from `~/aquapose/projects/YH/training_data/raw/` (COCO JSON + images)
- `data import` reads YOLO-format output from convert
- `train obb/pose` reads from store-assembled datasets or direct YOLO dirs
- Project config (`config.yaml`) updated with new weights paths after training

</code_context>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 71-data-store-bootstrap*
*Context gathered: 2026-03-07*
