# Phase 63: Pseudo-Label Generation (Source A) - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate high-confidence OBB and pose training labels by reprojecting consensus 3D reconstructions from diagnostic caches into contributing camera views. Output in standard YOLO txt+yaml format with confidence metadata sidecar. Source A labels come only from cameras that contributed to the reconstruction; gap cameras are Phase 64 (Source B).

</domain>

<decisions>
## Implementation Decisions

### Confidence Scoring
- Composite 0-1 score from weighted combination of reconstruction quality metrics (mean_residual, n_cameras, per_camera_residual_variance)
- Sidecar includes both composite score AND raw metrics (mean_residual, n_cameras, per_camera_residuals) for post-hoc analysis and threshold tuning
- No confidence floor at generation time -- output all labels regardless of score; Phase 65 handles filtering during dataset assembly

### OBB Derivation
- Reuse the PCA-based OBB approach from `scripts/build_yolo_training_data.py::pca_obb()` -- PCA on reprojected spline keypoints with configurable `lateral_pad`
- Promote `pca_obb` from scripts/ to a shared module (e.g., `training/geometry.py`) so both scripts/ and pseudo-label code import from there
- `lateral_pad` is a config parameter (not derived from half-widths, since pose backend does not produce them)

### CLI and I/O
- New top-level CLI group: `aquapose pseudo-label generate --config path/to/run/config.yaml`
- The `--config` flag expects the frozen config.yaml produced by each pipeline run (stored in run_dir/), NOT the minimal user config. The frozen config has all resolved paths including output_dir which locates the run directory.
- Output goes to `run_dir/pseudo_labels/{obb,pose}/` each with `images/`, `labels/`, `dataset.yaml` -- ready to pass directly to `train_yolo_obb()` or `train_yolo_pose()`
- Confidence metadata sidecar stored alongside labels

### Per-Camera Filtering
- Per-camera residual filtering: if a camera's residual exceeds a configurable threshold, skip the label for that camera even if the fish-level reconstruction is good
- Separate config field `pseudo_label.max_camera_residual_px` (independent from composite confidence score)
- Excluded cameras are NOT logged for Phase 64 -- Phase 64 re-derives gaps independently via inverse LUT visibility cross-referencing

### Claude's Discretion
- Composite confidence formula design (specific weights/normalization)
- Confidence sidecar file format (JSON, CSV, etc.)
- How to evaluate the 3D spline at keypoint t-values for pose labels
- Internal chunk iteration strategy for processing diagnostic caches
- Image extraction approach (symlinks vs copies from video frames)

</decisions>

<specifics>
## Specific Ideas

- OBB generation should match the approach used for manual training data (`pca_obb` from `scripts/build_yolo_training_data.py`) to keep pseudo-labels consistent with manual labels
- The CLI hint should guide users to point at the run's frozen config.yaml, not the minimal project config

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `RefractiveProjectionModel.project()` (`calibration/projection.py`): 3D-to-2D refractive projection, torch-based with Snell's law Newton-Raphson
- `pca_obb()` (`scripts/build_yolo_training_data.py:166`): PCA-based OBB from keypoints + lateral_pad -- to be promoted to shared module
- `Midline3D` (`core/types/reconstruction.py`): has `control_points`, `knots`, `degree`, `half_widths`, `mean_residual`, `n_cameras`, `per_camera_residuals`, `plane_normal`, `plane_centroid`
- `MidlineConfig.keypoint_t_values`: per-keypoint arc fractions for spline evaluation
- `DiagnosticObserver` (`engine/diagnostic_observer.py`): writes per-chunk `cache.pkl` with full PipelineContext
- `train_yolo_obb()` and `train_yolo_pose()`: training wrappers consuming `dataset.yaml` + `images/` + `labels/` directories
- `load_config()` (`engine/config.py`): loads frozen config from YAML
- `build_yolo_training_data.py`: full reference for YOLO txt format writing (OBB, pose, seg)

### Established Patterns
- CLI groups via Click (`training/prep.py` has `prep_group`, `training/cli.py` has `train_group`)
- Frozen dataclass config hierarchy with YAML loading (`engine/config.py`)
- Diagnostic cache envelope format: `{"run_id", "timestamp", "version_fingerprint", "context"}` in pickle

### Integration Points
- Diagnostic caches at `run_dir/diagnostics/chunk_*/cache.pkl` contain full PipelineContext per chunk
- PipelineContext.midlines_3d: `list[dict[int, Midline3D]]` indexed by frame
- CalibrationData loaded from `config.calibration_path` provides RefractiveProjectionModel per camera
- New `pseudo-label` CLI group needs to be registered in the main CLI entrypoint (`cli.py`)

</code_context>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 63-pseudo-label-generation-source-a*
*Context gathered: 2026-03-05*
