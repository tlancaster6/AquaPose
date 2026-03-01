---
phase: 37-pipeline-integration
plan: 02
subsystem: core
tags: [midline, backends, yolo-seg, yolo-pose, skeletonization, spline, affine-crop]

# Dependency graph
requires:
  - phase: 37-pipeline-integration
    provides: SegmentationBackend and PoseEstimationBackend stubs (Plan 01) with stored kwargs
  - phase: 35-codebase-cleanup
    provides: extract_affine_crop, invert_affine_points in segmentation/crop.py
  - phase: reconstruction
    provides: _adaptive_smooth, _skeleton_and_widths, _longest_path_bfs, _resample_arc_length helpers
provides:
  - SegmentationBackend with real YOLO-seg inference, mask skeletonization, affine back-projection
  - PoseEstimationBackend with real YOLO-pose inference, confidence filtering, spline interpolation
  - _keypoints_to_midline module-level function (scipy.interpolate.interp1d-based)
  - 21 unit tests covering both backends (9 seg + 12 pose)
affects: [37-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - OBB-aligned affine crop via extract_affine_crop(fit_obb=True, mask_background=True)
    - YOLO mask/keypoint extraction uses .cpu().numpy() for CUDA tensor safety
    - Failure-safe detection processing: every exception path returns AnnotatedDetection(midline=None)
    - Half-width scale-back via sqrt(M[0,0]^2 + M[0,1]^2) from affine matrix

key-files:
  created:
    - src/aquapose/core/midline/backends/segmentation.py
    - src/aquapose/core/midline/backends/pose_estimation.py
    - tests/unit/core/midline/test_segmentation_backend.py
    - tests/unit/core/midline/test_pose_estimation_backend.py
  modified: []

key-decisions:
  - "SegmentationBackend uses _adaptive_smooth/_skeleton_and_widths/_longest_path_bfs/_resample_arc_length from reconstruction/midline.py (no hand-rolling)"
  - "PoseEstimationBackend uses scipy.interpolate.interp1d with kind=linear and fill_value=extrapolate for spatial coords; bounds_error=False fill_value=(conf[0],conf[-1]) for confidence"
  - "half_widths in PoseEstimationBackend are zeros — pose backend has no distance transform for width estimation"
  - "OBB dims computed from obb_points side lengths when available; falls back to bbox w/h"
  - "angle=None detection falls back to 0.0 in both backends (axis-aligned crop)"

patterns-established:
  - "Backend _process_detection method: single detection in, AnnotatedDetection out, all exceptions caught"
  - "_extract_crop and _extract_keypoints as private helpers separating concerns in each backend"

requirements-completed: [PIPE-01, PIPE-02]

# Metrics
duration: 17min
completed: 2026-03-01
---

# Phase 37 Plan 02: Pipeline Integration Summary

**YOLO-seg skeletonization and YOLO-pose spline interpolation wired into SegmentationBackend and PoseEstimationBackend via OBB-aligned affine crops with full-frame coordinate back-projection**

## Performance

- **Duration:** 17 min
- **Started:** 2026-03-01T22:40:35Z
- **Completed:** 2026-03-01T22:57:00Z
- **Tasks:** 2
- **Files modified:** 4 (2 implementation files replaced, 2 test files created)

## Accomplishments

- `SegmentationBackend` now loads a YOLO-seg model, runs inference on OBB-aligned affine crops, extracts binary masks, skeletonizes via existing reconstruction/midline.py helpers, and back-projects to full-frame coordinates as `Midline2D`
- `PoseEstimationBackend` now loads a YOLO-pose model, extracts keypoints with confidence filtering, fits a linear spline via `_keypoints_to_midline`, and back-projects to full-frame as `Midline2D` with `point_confidence`
- Both backends handle all failure cases without exceptions: no model, no results, below-threshold mask/keypoints all return `AnnotatedDetection(midline=None)`
- 21 unit tests added covering instantiation, no-model path, mocked inference, area thresholds, confidence filtering, angle=None fallback, import boundary

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement SegmentationBackend with YOLO-seg inference and skeletonization** - `a54f6fd` (feat)
2. **Task 2: Implement PoseEstimationBackend with YOLO-pose inference and spline fitting** - `34b6d37` (feat)

## Files Created/Modified

- `src/aquapose/core/midline/backends/segmentation.py` - Full YOLO-seg inference pipeline replacing no-op stub
- `src/aquapose/core/midline/backends/pose_estimation.py` - Full YOLO-pose + spline interpolation replacing no-op stub
- `tests/unit/core/midline/test_segmentation_backend.py` - 9 unit tests for SegmentationBackend
- `tests/unit/core/midline/test_pose_estimation_backend.py` - 12 unit tests for PoseEstimationBackend

## Decisions Made

- **Skeletonization helpers reused** from `reconstruction/midline.py` (imported directly, not copied) — `_adaptive_smooth`, `_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length`
- **half_widths are zeros** in PoseEstimationBackend because pose-only pipeline has no distance transform; downstream triangulation uses `point_confidence` instead
- **OBB dims** computed from `obb_points` side lengths when available, falling back to `bbox[2], bbox[3]` when `obb_points` is None
- **`_keypoints_to_midline`** exposed as a module-level function (not private) to enable direct unit testing

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test assertion for angle=None fallback required mock model to reach crop extraction**
- **Found during:** Task 1 (test_none_angle_falls_back_to_zero)
- **Issue:** Original test checked `mock_extract.call_args is not None` but with `model=None`, the backend short-circuits before calling `extract_affine_crop`. The assertion was incorrectly written.
- **Fix:** Updated test to inject a mock model so crop extraction is exercised, then assert `angle_math_rad=0.0` in call kwargs
- **Files modified:** `tests/unit/core/midline/test_segmentation_backend.py`, `tests/unit/core/midline/test_pose_estimation_backend.py`
- **Verification:** Tests pass after fix
- **Committed in:** a54f6fd, 34b6d37 (part of task commits)

**2. [Rule 1 - Bug] Lint errors in test files (unused vars, import order)**
- **Found during:** Task 2 (hatch run lint)
- **Issue:** F841 (kpts_batch/conf_batch unused), RUF059 (conf unused), I001 (import order), SIM102 (nested if), F401 (unused SimpleNamespace)
- **Fix:** Removed unused variables, ran `ruff check --fix` for auto-fixable issues, manually collapsed nested if
- **Files modified:** `tests/unit/core/midline/test_pose_estimation_backend.py`, `tests/unit/core/midline/test_segmentation_backend.py`
- **Verification:** `hatch run lint` passes with "All checks passed!"
- **Committed in:** 34b6d37 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 - Bugs)
**Impact on plan:** Both fixes necessary for correctness and CI. No scope creep.

## Issues Encountered

- Pre-existing test failures discovered: `test_pipeline_writes_config_artifact`, `test_config_artifact_written_before_stages` (from Phase 37 Plan 01), and 18+ failures in `test_build_yolo_training_data.py` (caused by quick-13/14 build script refactoring). All pre-existing — verified via git stash. Not fixed per scope boundary rule.

## Next Phase Readiness

- Both backends are fully wired; `MidlineStage` can now produce real `Midline2D` objects when model weights are provided
- Plan 03 can connect these backends to the full `PosePipeline` end-to-end with real video and calibration
- Coordinate back-projection is handled by `invert_affine_points` — confirmed working in unit tests with identity affine matrix

---
*Phase: 37-pipeline-integration*
*Completed: 2026-03-01*

## Self-Check: PASSED

- FOUND: src/aquapose/core/midline/backends/segmentation.py
- FOUND: src/aquapose/core/midline/backends/pose_estimation.py
- FOUND: tests/unit/core/midline/test_segmentation_backend.py
- FOUND: tests/unit/core/midline/test_pose_estimation_backend.py
- FOUND commit: a54f6fd (Task 1 - SegmentationBackend)
- FOUND commit: 34b6d37 (Task 2 - PoseEstimationBackend)
