---
phase: 95-spline-refactoring
plan: "01"
subsystem: reconstruction
tags: [refactor, types, config, dlt-backend, spline]
dependency_graph:
  requires: [94-01]
  provides: [Midline3D raw-keypoint mode, spline_enabled config toggle, conditional DltBackend path]
  affects: [reconstruction pipeline, evaluation stages, io writer, pseudo-label training]
tech_stack:
  added: []
  patterns: [optional-field dataclass, conditional algorithm path, None-guard type narrowing]
key_files:
  created: []
  modified:
    - src/aquapose/core/types/reconstruction.py
    - src/aquapose/engine/config.py
    - src/aquapose/core/reconstruction/backends/dlt.py
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/engine/pipeline.py
    - tests/unit/core/reconstruction/test_dlt_backend.py
    - src/aquapose/evaluation/stages/reconstruction.py
    - src/aquapose/io/midline_writer.py
    - src/aquapose/training/pseudo_labels.py
decisions:
  - "Raw keypoints as primary reconstruction output: spline_enabled defaults to False"
  - "Midline3D field reorder: required fields first, all optional fields with defaults"
  - "dlt_backend fixture updated to spline_enabled=True to preserve existing test behavior"
  - "midline_writer skips raw-keypoint midlines (control_points=None) with continue guard"
  - "reproject_spline_keypoints raises ValueError for raw-mode midlines rather than silent failure"
metrics:
  duration: "~8 min"
  completed: "2026-03-13"
  tasks_completed: 2
  files_modified: 9
---

# Phase 95 Plan 01: Spline Refactoring — Midline3D and DltBackend Summary

**One-liner:** Midline3D extended with optional points/control_points fields and DltBackend split into raw-keypoint (default) and spline-fitted paths via spline_enabled toggle.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Extend Midline3D type and add spline_enabled config | 22515e6 | reconstruction.py, config.py |
| 2 | Refactor DltBackend to conditionally skip spline fitting | f3ef0c7 | dlt.py, stage.py, pipeline.py, test_dlt_backend.py, + 3 callers |

## What Was Built

### Task 1: Midline3D and ReconstructionConfig

**Midline3D** was restructured to support two representations:

- **Raw-keypoint mode** (`spline_enabled=False`): `points` field holds triangulated 3D keypoints `(N, 3)`. `control_points`, `knots`, `degree`, `arc_length` are all `None`.
- **Spline-fitted mode** (`spline_enabled=True`): `control_points` populated. `points` also populated for consistency.

Field ordering changed: required fields (`fish_id`, `frame_index`, `half_widths`, `n_cameras`, `mean_residual`, `max_residual`) come first, all optional fields follow with `= None` or `= False` defaults.

**ReconstructionConfig** gained `spline_enabled: bool = False` after `n_sample_points`.

### Task 2: DltBackend conditional spline path

`DltBackend.__init__` and `from_models` gained `spline_enabled: bool = False`:

- When `False`: spline knots are not precomputed, `_min_body_points = _MIN_RAW_BODY_POINTS = 2`. `_reconstruct_fish` skips `fit_spline()` and the `_MAX_ENDPOINT_GAP` endpoint check. Returns `Midline3D` with `points=pts_3d_arr_full`, `control_points=None`.
- When `True`: backward-compatible. Precomputes spline knots, enforces endpoint gap check, calls `fit_spline()`, returns `Midline3D` with `control_points` populated (and `points` also set).

`ReconstructionStage` gained `spline_enabled` parameter, forwarded to the backend. `_interpolate_gaps` updated to branch on `control_points is not None` for spline vs raw mode.

`pipeline.py` wired `spline_enabled=config.reconstruction.spline_enabled` to `ReconstructionStage`.

### Tests

Added `dlt_backend_raw` fixture (`spline_enabled=False`) alongside updated `dlt_backend` fixture (`spline_enabled=True` to preserve existing behavior).

Added `TestSplineDisabled` (3 tests) and `TestSplineEnabled` (2 tests).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Type errors from optional Midline3D fields in downstream callers**
- **Found during:** Task 2 typecheck
- **Issue:** Making `control_points`, `knots`, `degree`, `arc_length` optional broke 9 downstream access sites that assumed these fields were always non-None
- **Fix:** Added None guards in all 4 affected files: evaluation stages (2 call sites), midline_writer (skip raw midlines), pseudo_labels (ValueError for spline-specific utility + 2 centroid access sites)
- **Files modified:** evaluation/stages/reconstruction.py, io/midline_writer.py, training/pseudo_labels.py, core/reconstruction/backends/dlt.py (assert)
- **Commits:** f3ef0c7

**2. [Rule 1 - Bug] Existing test_fish_skipped_when_too_few_points used default backend**
- **Found during:** Task 2 test run
- **Issue:** Test created `DltBackend(...)` without `spline_enabled=True`, so now uses raw mode which accepts 5 body points (above `_MIN_RAW_BODY_POINTS=2`)
- **Fix:** Updated test to explicitly pass `spline_enabled=True`
- **Files modified:** tests/unit/core/reconstruction/test_dlt_backend.py
- **Commits:** f3ef0c7

## Verification

- `hatch run test -x`: 1203 passed, 3 skipped, 14 deselected
- `hatch run check`: 0 errors, 0 warnings, 0 notes
- Manual: `Midline3D.__dataclass_fields__` contains both `points` and `control_points`

## Self-Check: PASSED

All key files exist. Both commits found. 1203 tests pass. 0 typecheck errors.
