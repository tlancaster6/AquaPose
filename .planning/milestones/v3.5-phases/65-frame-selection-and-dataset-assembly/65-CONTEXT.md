# Phase 65: Frame Selection and Dataset Assembly - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a training dataset from manual annotations plus filtered pseudo-labels (Source A consensus + Source B gap-fill) with controlled temporal subsampling, pose-diversity sampling, and validation splits. Users can assemble datasets pooling from multiple runs with independent confidence thresholds per source.

</domain>

<decisions>
## Implementation Decisions

### Pose-Diversity Sampling
- Diversity metric: curvature only (mean/max curvature of the 3D spline control points). Captures the main behavioral variation (straight swimming vs turning) without overcomplicating.
- Pipeline order: sequential -- temporal subsampling (every Kth frame) first to reduce volume, then diversity sampling from the subsampled pool
- Selection method: K-means binning on curvature values, sample equally from each bin. K is configurable (e.g., 5 bins). Guarantees coverage across straight/mild/sharp curves.
- Diversity is per-fish: each fish has its own curvature distribution, sampled independently. The union of selected frames across all fish forms the final frame set.
- Frames with zero reconstructions are filtered before diversity sampling

### Dataset Pooling & Thresholds
- Confidence filtering applied at assembly time -- the assembled dataset is ready to train on as-is, no runtime filtering needed
- Separate default confidence thresholds for Source A (consensus) and Source B (gap-fill) via independent CLI flags. Gap-fill labels are inherently less reliable (camera didn't contribute to reconstruction).
- Gap-reason filtering: include/exclude flags by gap reason (e.g., `--exclude-gap-reason no-tracklet`). Users can skip specific gap types that produce unreliable labels.
- Manual annotations always included in full -- bypass all confidence/diversity filtering. They are ground truth.

### Validation Split Strategy
- Fixed fraction of manual data (configurable, default 20%) held out as the manual validation set, per-camera stratified so each camera is represented
- Manual val set is the official `val` split in dataset.yaml (the one Ultralytics uses for model selection during training)
- Separate pseudo-label validation set: a fraction of pseudo-labels held out with a JSON metadata sidecar recording source (A/B), gap_reason, and confidence per image
- Pseudo-label val is evaluation-only (post-training diagnostics), not used for model selection
- Pseudo-label val breakdown via metadata sidecar, not subdirectories

### CLI and Output Format
- Single command: `aquapose pseudo-label assemble`
- Multi-run support: accepts multiple `--run-dir` flags to pool pseudo-labels from different pipeline runs
- Manual data input: `--manual-dir` pointing to a YOLO-format directory containing `dataset.yaml` + `images/` + `labels/` (matching existing `--data-dir` pattern from training CLI)
- Output to project-level datasets directory (user-specified or project config default), not tied to a single run_dir
- Assembled dataset produces a YOLO-standard directory with `dataset.yaml`, `images/{train,val}/`, `labels/{train,val}/` ready for `aquapose train`

### Claude's Discretion
- Exact K-means implementation (sklearn vs scipy vs manual)
- Curvature computation method from spline control points
- How to handle multi-run image name collisions (prefix with run ID, etc.)
- Internal iteration strategy for processing multiple runs
- Default confidence threshold values for Source A and Source B
- Pseudo-label val fraction default

</decisions>

<specifics>
## Specific Ideas

- Phase 63 needs a prerequisite fix: pose pseudo-labels must output OBB-cropped stretch-fitted images with crop-space keypoints (matching `build_yolo_training_data.py` and the internal crop/transform logic during pose inference). Currently Phase 63 outputs pose labels in full-frame coordinates. This fix should be dispatched separately before Phase 65 execution.
- OBB pseudo-labels remain in full-frame coordinates (no change needed)
- The assembled dataset's `val` key in `dataset.yaml` must point to the manual val images for correct Ultralytics usage

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `generate_fish_labels()` (`training/pseudo_labels.py`): Per-fish per-camera label generation with confidence scoring
- `compute_confidence_score()` (`training/pseudo_labels.py`): Composite 0-1 confidence from mean_residual, n_cameras, per_camera_variance
- `stratified_split()` (`training/datasets.py`): Per-camera stratified train/val splitting -- can be adapted for manual val set construction
- `affine_warp_crop()` and `transform_keypoints()` (`training/geometry.py`): OBB-to-crop transform utilities (used by build_yolo_training_data.py)
- `pca_obb()` (`training/geometry.py`): PCA-based OBB computation
- `pseudo_label_group` (`training/pseudo_label_cli.py`): Existing Click CLI group for pseudo-label commands
- `confidence.json` sidecar: Per-image confidence metadata from Phase 63/64 generation

### Established Patterns
- CLI groups via Click with `--data-dir` pattern for YOLO dataset directories
- YOLO dataset structure: `{images,labels}/{train,val}/` with `dataset.yaml`
- Training wrappers expect `--data-dir` pointing to directory containing `dataset.yaml`
- Confidence sidecar as JSON alongside labels
- Dynamic import of engine modules in training CLI to avoid import boundary violation

### Integration Points
- Phase 63 output: `run_dir/pseudo_labels/consensus/{obb,pose}/` with `confidence.json`
- Phase 64 output: `run_dir/pseudo_labels/gap/{obb,pose}/` with gap-reason metadata in `confidence.json`
- Manual training data: YOLO-format directory with `dataset.yaml`
- Training CLI (`training/cli.py`): consumes assembled dataset via `--data-dir`
- Midline3D spline control points: source for curvature computation in diversity sampling
- Diagnostic caches: contain `midlines_3d` needed for curvature computation

</code_context>

<deferred>
## Deferred Ideas

- Phase 63 pose crop fix: dispatched as a separate quick task (pose pseudo-labels must output OBB-cropped images with crop-space keypoints)

</deferred>

---

*Phase: 65-frame-selection-and-dataset-assembly*
*Context gathered: 2026-03-05*
