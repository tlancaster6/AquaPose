---
phase: 33-keypoint-midline-backend
plan: 01
subsystem: core/midline
tags: [backend, keypoint-regression, spline-fitting, nan-padding, cli, config]
dependency_graph:
  requires: [32-02]
  provides: [DirectPoseBackend-implementation, MidlineConfig-direct-pose-fields, prep-CLI]
  affects: [engine/pipeline, core/midline/stage, training/cli]
tech_stack:
  added: [scipy.interpolate.CubicSpline, scipy.interpolate.interp1d]
  patterns: [affine-crop-inference, confidence-heuristic, nan-padding, arc-fraction-calibration]
key_files:
  created:
    - src/aquapose/core/midline/backends/direct_pose.py
    - src/aquapose/training/prep.py
    - tests/unit/core/midline/test_direct_pose_backend.py
  modified:
    - src/aquapose/core/midline/backends/__init__.py
    - src/aquapose/core/midline/stage.py
    - src/aquapose/engine/config.py
    - src/aquapose/training/__init__.py
    - src/aquapose/cli.py
    - tests/unit/core/midline/test_midline_stage.py
decisions:
  - "Detection.centroid missing from dataclass: backend derives centroid from bbox (x + w/2, y + h/2) via hasattr guard, matching pattern in association/stage.py"
  - "CubicSpline falls back to linear interpolation when fewer than 4 unique t-values are visible, preventing scipy ValueError"
  - "torch.load and _PoseModel patched at source modules (aquapose.training.pose._PoseModel, torch.load) since direct_pose.py lazy-imports both inside __init__"
  - "prep CLI docstring references Args: section from function docstring — kept for click --help clarity"
metrics:
  duration: 23 min
  completed_date: "2026-03-01"
  tasks_completed: 2
  files_modified: 9
---

# Phase 33 Plan 01: DirectPoseBackend Implementation Summary

Full DirectPoseBackend implementation — keypoint regression midline backend with CubicSpline fitting, NaN-padding, and confidence heuristic, plus MidlineConfig extensions and calibrate-keypoints CLI.

## What Was Built

### Task 1: DirectPoseBackend Implementation

**`src/aquapose/core/midline/backends/direct_pose.py`** — complete rewrite of the stub:

- `DirectPoseBackend.__init__`: loads `_PoseModel` from `aquapose.training.pose` with lazy torch import, validates weights path (fail-fast), stores config params
- `process_frame`: for each camera/detection, extracts OBB-aligned affine crop, runs inference, computes confidence heuristic, applies floor, fits CubicSpline through visible keypoints, resamples to exactly `n_points`, NaN-pads outside `[t_min_obs, t_max_obs]`
- Confidence heuristic: `1 - 2 * max(|x-0.5|, |y-0.5|)` — crops pushing keypoints to edges give conf=0
- Centroid derived from bbox when `Detection.centroid` attribute not present
- Spline fallback: CubicSpline (>=4 visible) or linear interp (<4 visible)
- Returns `Midline2D` with `is_head_to_tail=True`, zero `half_widths`, and `point_confidence`

**`src/aquapose/core/midline/backends/__init__.py`** — updated docstring to reflect `direct_pose` is fully implemented with kwargs documentation.

**`tests/unit/core/midline/test_direct_pose_backend.py`** — 9 unit tests:
- `test_constructor_validates_weights_path`
- `test_process_frame_returns_annotated_detections`
- `test_output_always_n_sample_points`
- `test_partial_visibility_nan_padding`
- `test_below_min_observed_returns_none_midline`
- `test_axis_aligned_detection_angle_none`
- `test_confidence_heuristic`
- `test_empty_camera_returns_empty_list`
- `test_both_backends_same_shape`

**`tests/unit/core/midline/test_midline_stage.py`** — removed 3 `NotImplementedError` tests, added `test_backend_registry_direct_pose_constructs`.

### Task 2: Config Extension, Build Wiring, prep CLI

**`src/aquapose/engine/config.py`** — `MidlineConfig` extended with 4 new fields:
- `keypoint_weights_path: str | None = None`
- `keypoint_t_values: list[float] | None = None`
- `keypoint_confidence_floor: float = 0.1`
- `min_observed_keypoints: int = 3`

All fields have defaults, so existing YAML configs load without error.

**`src/aquapose/core/midline/stage.py`** — `MidlineStage.__init__` branches on `backend == "direct_pose"`: extracts keypoint-specific fields from `midline_config` and passes them to `get_backend("direct_pose", ...)`. The `segment_then_extract` path is unchanged.

**`src/aquapose/training/prep.py`** — new file with `prep_group` click group and `calibrate-keypoints` subcommand:
- Reads COCO-format annotation JSON
- Computes per-keypoint arc-length fractions averaged across all annotated instances
- Writes YAML snippet with `keypoint_t_values` list
- Reports number of annotations processed and computed values

**`src/aquapose/training/__init__.py`** — added `prep_group` to imports and `__all__`.

**`src/aquapose/cli.py`** — registered `prep_group` with `cli.add_command(prep_group)`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Detection has no `centroid` field**

- **Found during:** Task 1 implementation
- **Issue:** `DirectPoseBackend` references `det.centroid` but `Detection` dataclass only has `bbox`, `mask`, `area`, `confidence`, `angle`, `obb_points` — no `centroid` attribute
- **Fix:** Added `hasattr(det, "centroid")` guard; when missing, derives centroid as `(bx + bw/2, by + bh/2)` from bbox. Same pattern used in `association/stage.py`
- **Files modified:** `src/aquapose/core/midline/backends/direct_pose.py`
- **Commit:** 4926f4d

**2. [Rule 2 - Missing critical functionality] CubicSpline requires >= 4 points**

- **Found during:** Task 1 implementation
- **Issue:** `scipy.interpolate.CubicSpline` raises `ValueError` when fewer than 4 unique t-values are provided. With only 2-3 visible keypoints the spline fails
- **Fix:** Added branch: CubicSpline when `len(t_unique) >= 4`, else fall back to `interp1d` with `kind="linear"` — ensures graceful degradation for partial visibility
- **Files modified:** `src/aquapose/core/midline/backends/direct_pose.py`
- **Commit:** 4926f4d

**3. [Rule 1 - Bug] Test patch targets for lazy imports**

- **Found during:** Task 1 test authoring
- **Issue:** Tests tried to patch `aquapose.core.midline.backends.direct_pose._PoseModel` and `aquapose.core.midline.backends.direct_pose.torch.load`, but these don't exist at module level (both are lazy-imported inside `__init__`)
- **Fix:** Patch `aquapose.training.pose._PoseModel` (module where it lives) and `torch.load` (global module), then replace `backend._model` directly after construction
- **Files modified:** `tests/unit/core/midline/test_direct_pose_backend.py`
- **Commit:** 4926f4d

## Verification Results

- `hatch run test tests/unit/core/midline/ -x`: 662 passed, 0 failed
- `hatch run test -x`: 662 passed, 0 failed
- `aquapose prep --help`: shows prep group with calibrate-keypoints command
- `aquapose prep calibrate-keypoints --help`: shows --annotations, --output, --n-keypoints flags
- Import boundary check: 3 pre-existing warnings, 0 new violations
- Typecheck: 38 pre-existing errors (in detector.py, midline.py, etc.), 0 new errors in modified files

## Self-Check: PASSED

- src/aquapose/core/midline/backends/direct_pose.py: FOUND
- src/aquapose/training/prep.py: FOUND
- tests/unit/core/midline/test_direct_pose_backend.py: FOUND
- .planning/phases/33-keypoint-midline-backend/33-01-SUMMARY.md: FOUND
- Commit 4926f4d: FOUND
- Commit cf1b42d: FOUND
