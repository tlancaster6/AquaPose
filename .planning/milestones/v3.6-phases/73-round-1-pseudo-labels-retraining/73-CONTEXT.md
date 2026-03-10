# Phase 73: Round 1 Pseudo-Labels & Retraining - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate pseudo-labels from the Phase 72 baseline pipeline run, select a high-impact subset via diversity-maximizing criteria, exhaustively review and curate that subset, import into the data store, train round 1 OBB and pose models, and run an A/B comparison quantifying the value of human curation. Covers ITER-02, ITER-03, ITER-06.

</domain>

<decisions>
## Implementation Decisions

### Pseudo-label subset selection
- Target ~50 full-frame OBB images and ~320 pose crop images (mirroring manual annotation set sizes)
- OBB selection uses 3-axis diversity: camera coverage (~4 per camera with 2 flex slots), temporal spread within each camera (divide into 4 temporal bins, pick one per bin), fish count tiebreaker (prefer underrepresented fish counts)
- Pose selection uses 2-axis diversity: camera coverage and 3D curvature stratification (from confidence.json curvature_3d field)
- All selection metadata is queryable from existing confidence.json sidecars and filenames

### Manual curation process (CVAT-based)
- Load diversity-selected subset into CVAT for full manual correction (not just pass/fail exclusion)
- User corrects OBB bounding boxes and pose keypoint positions directly in CVAT
- User removes samples that are unsalvageable (wrong fish, background-only, etc.)
- CVAT export produces corrected YOLO-format labels for re-import

### Curation A/B comparison
- Both arms start from the same diversity-selected pseudo-label subset
- Arm A (curated): manually corrected labels imported as source=manual (upserts over pseudo via priority), plus exclusions for removed samples
- Arm B (uncurated): original pseudo-labels with no corrections or exclusions
- **Sequencing**: uncurated dataset must be assembled BEFORE importing CVAT corrections, since store source priority (manual > pseudo) would otherwise replace originals in both arms
- Correction quantification: compare original pseudo-labels vs CVAT exports to measure correction magnitude (OBB IoU delta, pose keypoint displacement, samples added/removed)
- Primary comparison metric: training val metrics via `train compare`
- Winner selection: user decides at checkpoint based on numbers and training curves

### Dataset composition
- Elastic augmentation on manual annotations only (pseudo-labels have natural curvature diversity)
- Consensus-only pseudo-labels for pose (no gap-fill labels in round 1)
- OBB pseudo-labels include both consensus and gap sources (merged by the generate CLI)
- Import with source=pseudo, round=1, with confidence scores stored as metadata on each sample

### Validation sets
- Primary val set: Phase 71 manual val set (unchanged, used during training for early stopping and metrics)
- Secondary val set (pose only): ~20% holdout from curated pseudo-label pose crops, split temporally (later frames to val). OBB budget (~50) is too small for a meaningful holdout.
- Secondary set used for post-training evaluation only (not during training)
- Baseline model also evaluated on secondary set to characterize improvement

### Training configuration
- Train from scratch (no transfer learning from baseline weights)
- Same hyperparameters as Phase 71 baseline: epochs=100, patience=100, mosaic defaults (0.3 OBB, 0.1 pose)
- Run tags: `round1-curated` / `round1-uncurated`
- 4 training runs total: OBB curated, OBB uncurated, pose curated, pose uncurated

### Claude's Discretion
- Exact implementation of the diversity selection script (stratification bins, tiebreaker logic)
- How confidence scores are stored as metadata during store import (may need minor store schema extension)
- Order of operations for the secondary val set evaluation

</decisions>

<specifics>
## Specific Ideas

- User wants full manual correction in CVAT (not just exclusion marking) to maximize curated data quality
- Selection strategy should maximize impact of the limited pseudo-label budget rather than using everything
- Secondary pseudo-label val set enables characterizing generalization to new scenarios, not just held-out manual annotations
- Baseline model should also be evaluated on pseudo-label holdout for full improvement characterization

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pseudo_label_cli.py`: Full pseudo-label generate + inspect CLI with confidence.json sidecars, temporal stepping, visualization
- `data_cli.py`: `import`, `convert`, `assemble`, `exclude --reason TAG`, `status` commands
- `training/cli.py`: `train obb`, `train pose` with `--weights`, `--tag`, and auto-registration; `train compare` for side-by-side
- `elastic_deform.py`: `generate_deformed_labels()` for elastic augmentation
- `store.py`: `SampleStore` with `import_sample()`, `assemble()`, `exclude()` supporting provenance tracking
- `compare.py`: `discover_runs()`, `load_run_summaries()`, `format_comparison_table()`

### Established Patterns
- Confidence.json sidecars contain per-label metadata: confidence, curvature_3d, source, raw_metrics, tracked_fish_count
- Filenames encode frame_idx and cam_id: `{frame_idx:06d}_{cam_id}` for OBB, `{frame_idx:06d}_{cam_id}_{fish_idx:03d}` for pose crops
- Temporal split convention from Phase 71 (BOOT-04): later temporal indices go to val
- Store import supports source and provenance fields
- Training runs auto-register models and write summary.json

### Integration Points
- Pseudo-label generation reads from Phase 72's run directory diagnostic caches
- Store import feeds into `store.assemble()` which produces YOLO-format datasets
- Assembled datasets are consumed by `train obb` / `train pose` CLI commands
- `train compare` reads summary.json from training run directories
- Known bug: `store.assemble()` doesn't write `kpt_shape`/`flip_idx` in pose dataset.yaml (manual fix needed, same as Phase 71)

</code_context>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 73-round-1-pseudo-labels-retraining*
*Context gathered: 2026-03-07*
