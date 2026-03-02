# Phase 43: Triangulation Rebuild - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

A new reconstruction backend (`"dlt"`) that uses confidence-weighted DLT triangulation with outlier rejection and B-spline fitting. Lives alongside the existing `"triangulation"` and `"curve_optimizer"` backends until Phase 45 cleanup. Does not modify ReconstructionStage — purely a new backend module + registry entry.

</domain>

<decisions>
## Implementation Decisions

### Orientation & Correspondence
- Backend assumes correspondences are already solved by upstream (Stage 4)
- No head-tail alignment — point i is the same anatomy across all cameras
- No epipolar refinement — the backend triangulates whatever points it receives
- No group-level flip hints — pure geometry module with no awareness of tracklet groups
- NaN points (missing keypoints) are silently skipped from that body point's camera set

### Outlier Rejection
- Fixed configurable pixel threshold (single value, e.g., 50px default)
- Single pass: triangulate all cameras → compute per-camera reprojection residual → reject cameras above threshold → re-triangulate with inliers
- When rejection leaves fewer than 2 inlier cameras for a body point, drop the point entirely (no fallback)
- Keep the ray-angle filter (5-degree minimum) from existing code to prevent ill-conditioned DLT from near-parallel camera pairs
- Confidence weighting uses sqrt(confidence), matching existing convention

### Backend Organization
- Registry key: `"dlt"`
- Single file: `backends/dlt.py` with `DltBackend` class
- Shared helpers (`_weighted_triangulate_rays`, `_fit_spline`, `_build_spline_knots`, `_pixel_half_width_to_metres`) extracted to `reconstruction/utils.py`, imported by both old and new backends
- Backend registry (`backends/__init__.py`) updated with `"dlt"` entry
- ReconstructionStage unchanged — delegates via `reconstruct_frame()` as before

### Midline2D Input Handling
- Primarily targets direct-pose keypoint output (6 keypoints interpolated to 15 points by Stage 4)
- Segment-then-extract output also works (15 arc-length-sampled points)
- Read point count dynamically from `len(midline.points)` — no hardcoded 15
- When `point_confidence` is None (segment-then-extract), treat as uniform 1.0 (unweighted DLT)
- Half-widths converted to world metres via pinhole approximation (same as existing)
- Expect all-zero half-widths from current pose estimation setup (keypoint models don't produce widths)

### Claude's Discretion
- Default value for the outlier rejection threshold (empirically tuned in Phase 44)
- Internal structure of DltBackend (method decomposition, helper organization)
- Which helpers to extract vs keep backend-specific
- Test structure and fixture design

</decisions>

<specifics>
## Specific Ideas

- The algorithm is intentionally stripped-down: triangulate → reject → re-triangulate → fit spline. No multi-strategy branching, no epipolar geometry, no orientation logic.
- The existing `_weighted_triangulate_rays` and `_fit_spline` are pure functions suitable for extraction without modification.
- Both upstream midline backends already produce 15 points (well over the 9-point minimum for 7-control-point B-spline fitting), so no special handling for underdetermined systems.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_weighted_triangulate_rays()` in `triangulation.py`: Confidence-weighted DLT via normal equations — extract to `utils.py`
- `_fit_spline()` in `triangulation.py`: `make_lsq_spline` with configurable knot vector — extract to `utils.py`
- `_build_spline_knots()` in `triangulation.py`: Clamped cubic B-spline knot vector builder — extract to `utils.py`
- `_pixel_half_width_to_metres()` in `triangulation.py`: Pinhole half-width conversion — extract to `utils.py`
- `TriangulationBackend._load_models()`: Calibration loading pattern to replicate in DltBackend
- `triangulate_rays()` in `calibration/projection.py`: Unweighted DLT (used when all weights are 1.0)
- `RefractiveProjectionModel.cast_ray()` and `.project()`: Ray casting and reprojection for residual computation

### Established Patterns
- Backend class with `reconstruct_frame(frame_idx, midline_set) -> dict[int, Midline3D]` interface
- Calibration loaded eagerly at construction (fail-fast)
- Backend resolved by string key via `get_backend()` factory
- Constants as module-level `UPPER_SNAKE_CASE` with sensible defaults

### Integration Points
- `backends/__init__.py`: Add `"dlt"` case to `get_backend()` factory
- `reconstruction/utils.py`: New shared module, imported by both `backends/triangulation.py` and `backends/dlt.py`
- `Midline3D` dataclass: Output type unchanged — `control_points`, `knots`, `degree`, `arc_length`, `half_widths`, `n_cameras`, `mean_residual`, `max_residual`, `is_low_confidence`, `per_camera_residuals`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 43-triangulation-rebuild*
*Context gathered: 2026-03-02*
