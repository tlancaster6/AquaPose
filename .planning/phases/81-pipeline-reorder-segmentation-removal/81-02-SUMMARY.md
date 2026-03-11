---
phase: 81-pipeline-reorder-segmentation-removal
plan: "02"
subsystem: core/reconstruction, core/pose, engine, evaluation
tags: [segmentation-removal, reconstruction, keypoint-interpolation, dead-code, pipeline-v37]
dependency_graph:
  requires: ["81-01"]
  provides: ["clean-codebase-no-segmentation", "reconstruction-keypoint-interpolation"]
  affects: ["evaluation-tuning", "diagnostic-observer", "overlay-viz"]
tech_stack:
  added: []
  patterns:
    - "6→15 keypoint interpolation via scipy.interpolate.interp1d in ReconstructionStage"
    - "Detection.keypoints (6,2) as source for midline construction"
    - "Backward-compat dual-path in tuning.py/runner.py for legacy diagnostic runs"
key_files:
  created: []
  modified:
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/engine/diagnostic_observer.py
    - src/aquapose/evaluation/tuning.py
    - src/aquapose/evaluation/runner.py
    - src/aquapose/evaluation/viz/overlay.py
    - src/aquapose/core/synthetic.py
    - src/aquapose/core/types/frame_source.py
    - src/aquapose/io/discovery.py
    - src/aquapose/engine/orchestrator.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py
    - tests/unit/engine/test_diagnostic_observer.py
    - tests/unit/evaluation/test_tuning.py
    - tests/unit/core/pose/test_pose_stage.py
  deleted:
    - src/aquapose/core/pose/backends/segmentation.py
    - src/aquapose/core/pose/orientation.py
    - src/aquapose/core/pose/midline.py
    - tests/unit/core/midline/test_orientation.py
    - tests/unit/core/midline/test_segmentation_backend.py
    - tests/unit/test_midline.py
  moved:
    - "tests/unit/core/midline/test_pose_estimation_backend.py → tests/unit/core/pose/test_pose_estimation_backend.py"
    - "tests/unit/core/midline/test_direct_pose_backend.py → tests/unit/core/pose/test_direct_pose_backend.py"
decisions:
  - "keypoint_t_values=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0] defined as module-level constant in ReconstructionStage — avoids cross-module coupling with PoseConfig"
  - "tuning.py and runner.py use dual-path (detections v3.7 + annotated_detections legacy fallback) to maintain backward compat with old diagnostic runs"
  - "_find_matching_detection uses bbox centroid distance matching (same tolerance=10px as legacy)"
metrics:
  duration_minutes: 60
  tasks_completed: 2
  files_changed: 18
  completed_date: "2026-03-11"
requirements_completed: [PIPE-02, PIPE-03]
---

# Phase 81 Plan 02: Delete Segmentation Dead Code and Update ReconstructionStage Summary

**One-liner:** Deleted segmentation/orientation dead code (3 files), added `_keypoints_to_midline()` interpolation (6→15 points via interp1d) to ReconstructionStage reading from `Detection.keypoints`, and updated all consumers and tests for the v3.7 pipeline data flow.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Delete dead code, update ReconstructionStage + consumers | a851c22 | stage.py, segmentation.py (deleted), tuning.py, runner.py, overlay.py |
| 1a | Fix Detection return type in _find_matching_detection | a923253 | stage.py |
| 1b | Remove dead AnnotatedDetection centroid branch | 57e5b74 | stage.py |
| 2 | Fix all tests for new pipeline order and module layout | f4c09fc | test_reconstruction_stage.py, test_diagnostic_observer.py, test_tuning.py, test_pose_stage.py |

## What Was Built

### Dead Code Deleted
- `src/aquapose/core/pose/backends/segmentation.py` — AnnotatedDetection-producing segmentation backend
- `src/aquapose/core/pose/orientation.py` — orientation estimation (removed per Phase 81 decision)
- `src/aquapose/core/pose/midline.py` — midline utilities only used by segmentation.py
- Corresponding tests: `test_orientation.py`, `test_segmentation_backend.py`, `test_midline.py`

### ReconstructionStage Updated
- Added `_KEYPOINT_T_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]` module-level constant
- Added `_keypoints_to_midline(kpts_xy, t_values, confidences, n_points)` — interpolates visible keypoints to dense midline via `scipy.interpolate.interp1d`
- Renamed `_find_matching_annotated()` → `_find_matching_detection()` typed as `list[Detection] -> Detection | None`
- `_run_with_tracklet_groups()` now reads `context.detections`, finds Detection by bbox centroid, reads `det.keypoints`/`det.keypoint_conf`, calls `_keypoints_to_midline(..., n_points=15)`

### Consumer Updates
- **DiagnosticObserver**: Removed `annotated_detections` from `StageSnapshot` and `_PER_FRAME_FIELDS`
- **tuning.py**: `_build_midline_sets()` has v3.7 path (reads `ctx.detections`, calls `_keypoints_to_midline`) with legacy fallback for old diagnostic runs
- **runner.py**: Same dual-path pattern with `_match_detection_by_centroid()` helper
- **overlay.py**: Replaced `all_annotated_detections` → reads `ctx.detections`, draws raw `det.keypoints`
- **Docstrings**: Updated 4 stale "MidlineStage" references to "PoseStage" in discovery.py, frame_source.py, synthetic.py, orchestrator.py

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing Detection import and weak return type in _find_matching_detection**
- **Found during:** Post-task typecheck (hatch run check)
- **Issue:** `_find_matching_detection` returned `object | None` instead of `Detection | None`, causing 5 basedpyright errors for `.keypoints`/`.keypoint_conf` attribute access
- **Fix:** Added `from aquapose.core.types.detection import Detection` import; tightened return type annotation
- **Files modified:** `src/aquapose/core/reconstruction/stage.py`
- **Commit:** a923253

**2. [Rule 1 - Bug] Dead `elif hasattr(det, "centroid")` branch after type tightening**
- **Found during:** Typecheck after fix 1
- **Issue:** Legacy branch for `AnnotatedDetection.centroid` attribute unreachable with `list[Detection]` parameter; caused reportAttributeAccessIssue
- **Fix:** Removed the `elif` branch and the surrounding `if hasattr(det, "bbox")` guard (Detection always has bbox)
- **Files modified:** `src/aquapose/core/reconstruction/stage.py`
- **Commit:** 57e5b74

**3. [Rule 1 - Bug] Unused variable `detections` in test_reconstruction_stage.py**
- **Found during:** Task 2 commit (ruff pre-commit hook, F841)
- **Issue:** `detections` variable assigned but `detections_2` used for the ctx; ruff F841 error
- **Fix:** Removed the unused `detections` assignment
- **Files modified:** `tests/unit/core/reconstruction/test_reconstruction_stage.py`
- **Commit:** included in f4c09fc

**4. [Rule 3 - Blocking] tuning.py ruff format failure on first commit attempt**
- **Found during:** Task 1 commit (pre-commit hook reformatted tuning.py)
- **Fix:** Re-staged tuning.py with reformatted content
- **Commit:** included in a851c22

## Verification Results

```
hatch run test: 1105 passed, 3 skipped, 14 deselected
hatch run check: 26 errors (all pre-existing; 4 new errors resolved by this plan)
```

Pre-existing typecheck errors not introduced by this plan include: `get_backend()` returning `object` (pipeline.py, reconstruction/stage.py reconstruct_frame call), `VideoWriter_fourcc` cv2 attribute, LutConfig protocol issue.

## Self-Check

- [x] `src/aquapose/core/reconstruction/stage.py` exists with `_keypoints_to_midline`
- [x] `src/aquapose/core/pose/backends/segmentation.py` deleted
- [x] `src/aquapose/core/pose/orientation.py` deleted
- [x] Commits a851c22, f4c09fc, a923253, 57e5b74 exist in git log
- [x] 1105 tests pass

## Self-Check: PASSED
