# Phase 61: Z-Denoising - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Clean z-noise from 3D reconstructions via per-frame plane projection (Component A) and temporal smoothing of plane orientation (Component B). Produces splines suitable for accurate pseudo-label reprojection. Two independent, additive components with separate config toggles. Does not modify triangulation or spline fitting internals.

</domain>

<decisions>
## Implementation Decisions

### IRLS Weighting Strategy
- Camera count is the quality signal for SVD plane-fit weights (more cameras = higher weight)
- No explicit endpoint down-weighting taper -- let camera count handle it naturally
- Zero IRLS iterations -- single-pass weighted SVD only
- Degenerate (straight fish) detection via singular value ratio; use horizontal plane (normal = [0,0,1]) as default; let Component B smooth through degenerate frames

### Midline3D Storage
- Extend existing Midline3D dataclass with new optional fields: `plane_normal` (3,), `plane_centroid` (3,), `off_plane_residuals` (n_sample_points,), `is_degenerate_plane` (bool)
- All new fields are Optional, None when `plane_projection.enabled=False`
- Off-plane residuals stored at sample resolution (n_sample_points=15), matching half_widths shape
- HDF5 writer adds corresponding datasets with NaN/False fill values

### Component B Location and Behavior
- Separate CLI command (e.g. `aquapose smooth-planes`), not integrated into the pipeline orchestrator
- In-place HDF5 modification: stores `smoothed_plane_normal` alongside original `plane_normal`, updates `control_points` to smoothed orientation
- Both raw and smoothed normals available in one file for comparison
- Gaussian filter with configurable `sigma_frames` (default ~3-5 frames per design doc)

### Validation Approach
- Eval harness metrics for both gates -- extend existing reconstruction eval module
- Component A gate: residual-delta metric (before/after plane projection)
- Component B gate: z-range, frame-to-frame z-profile RMS, SNR metrics
- Soft metrics with warnings when thresholds exceeded, not hard gates that fail the eval
- Approximate thresholds: residual delta < ~0.5 px, median z-range < ~1 cm, z-profile RMS < 0.1 cm, SNR > 1 for most fish

### Claude's Discretion
- Exact singular value ratio threshold for degenerate plane detection
- Gaussian sigma_frames default value (within 3-5 range)
- Eval metric implementation details
- Normal sign consistency algorithm in Component B

</decisions>

<specifics>
## Specific Ideas

- Component A is purely per-frame; Component B is purely temporal. Clean separation -- A never peeks at neighboring frames.
- Degenerate frames get a horizontal default normal from A, which B naturally smooths through using neighboring non-degenerate frames.
- The smooth-planes CLI preserves both raw and smoothed normals so pre-vs-post comparison is possible from a single file.
- Pre-smoothing control points are recoverable from raw normal + centroid (undo the rotation) even though control_points are overwritten.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DltBackend._reconstruct_fish()` (`core/reconstruction/backends/dlt.py:225`): Insertion point for Component A -- between triangulation result and `fit_spline()` call
- `_TriangulationResult` (`dlt.py:81`): Already provides `inlier_cam_ids` (list of camera IDs per body point) for camera count weighting
- `fit_spline()` (`core/reconstruction/utils.py:106`): Unchanged -- receives projected points instead of raw points
- `Midline3D` (`core/types/reconstruction.py:16`): Dataclass to extend with plane metadata
- `Midline3DWriter` (`io/midline_writer.py:27`): HDF5 writer to extend with new datasets
- `read_midline3d_results()` (`io/midline_writer.py:216`): Reader to extend for new fields

### Established Patterns
- Frozen dataclass config hierarchy (`engine/config.py`): `ReconstructionConfig` needs `plane_projection` sub-config
- HDF5 chunked-append pattern in `Midline3DWriter`: new datasets follow same `_make()` + buffer pattern
- Eval harness in `evaluation/` module: extend with z-denoising metrics

### Integration Points
- `ReconstructionConfig` (`engine/config.py:252`): Add `plane_projection.enabled` toggle
- CLI (`cli.py`): Add `aquapose smooth-planes` command for Component B
- Chunk orchestrator (`engine/orchestrator.py`): No changes needed -- Component B runs separately

</code_context>

<deferred>
## Deferred Ideas

- Leave-one-out validation using manual multi-view annotations: hold out one camera, reconstruct from remaining, reproject into held-out camera, compare to ground truth. Validates pseudo-label quality -- belongs in Phase 63 (Pseudo-Label Generation).
- Dorsoventral arch recovery (Component C): temporal averaging of off-plane residuals. Not needed for pseudo-label quality. Future work per design doc.

</deferred>

---

*Phase: 61-z-denoising*
*Context gathered: 2026-03-05*
