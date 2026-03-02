---
phase: 43-triangulation-rebuild
plan: "02"
subsystem: reconstruction
tags: [feature, dlt, triangulation, backend, tdd]
dependency_graph:
  requires: [43-01 utils.py shared helpers]
  provides: [reconstruction/backends/dlt.py DltBackend]
  affects: [reconstruction/backends/__init__.py, future Phase 44 eval harness]
tech_stack:
  added: []
  patterns: [confidence-weighted DLT, single-strategy outlier rejection, B-spline fitting, TDD]
key_files:
  created:
    - src/aquapose/core/reconstruction/backends/dlt.py
    - tests/unit/core/reconstruction/test_dlt_backend.py
  modified:
    - src/aquapose/core/reconstruction/backends/__init__.py
decisions:
  - DltBackend replicates _load_models static method from TriangulationBackend (no shared coupling to old backend)
  - Ray-angle filter applied only to 2-camera case; DLT handles multi-camera near-parallel naturally
  - Water surface rejection applied twice (pre- and post-outlier-rejection) for defense in depth
  - Mock models in tests use proper pinhole geometry (camera above water, fish at positive Z) to produce valid underwater triangulations
metrics:
  duration: "9 minutes"
  completed: "2026-03-02"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 1
---

# Phase 43 Plan 02: Implement DltBackend Summary

Implemented DltBackend — a stripped-down reconstruction backend using confidence-weighted DLT triangulation with single-pass outlier rejection and B-spline fitting, replacing the complex multi-strategy triangulation logic.

## What Was Built

`src/aquapose/core/reconstruction/backends/dlt.py` — new backend implementing:
- `DltBackend.__init__`: loads calibration eagerly (fail-fast), same _load_models pattern as TriangulationBackend
- `DltBackend.reconstruct_frame`: single-strategy algorithm — triangulate all cameras → reject outliers → re-triangulate inliers → fit spline
- `DltBackend._triangulate_body_point`: core per-body-point logic with ray-angle filter, water surface rejection, outlier rejection, re-triangulation
- `DltBackend._tri_rays`: weighted/unweighted DLT dispatch (weighted when any confidence != 1.0)
- `DltBackend._convert_half_widths`: pinhole approximation + interp1d for full body profile
- Module constants: `DEFAULT_OUTLIER_THRESHOLD=50.0`, `DEFAULT_N_CONTROL_POINTS=7`, `DEFAULT_LOW_CONFIDENCE_FRACTION=0.2`, `_MIN_RAY_ANGLE_DEG=5.0`

Key simplifications over TriangulationBackend:
- NO orientation alignment — upstream pose backend provides ordered keypoints
- NO epipolar refinement — ordered keypoints eliminate correspondence ambiguity
- NO camera-count branching — single path regardless of camera count

`src/aquapose/core/reconstruction/backends/__init__.py` — updated to:
- Register "dlt" case in get_backend() factory
- Update ValueError message and module docstring to include "dlt"

`tests/unit/core/reconstruction/test_dlt_backend.py` — 197-line test file covering:
- Basic reconstruction: returns Midline3D with correct field types (control_points shape (7,3), knots (11,), etc.)
- Multiple fish in one frame
- NaN handling: NaN in one camera skips that camera for that body point only; other cameras continue
- Outlier rejection: extra cameras with models but not in midline_set don't interfere
- Water surface rejection: horizontal rays producing Z <= water_z are dropped, fish skipped
- Low-confidence flagging: 2-camera reconstruction flags is_low_confidence=True
- Insufficient body points: fish with < MIN_BODY_POINTS valid triangulations skipped
- Half-widths passthrough: output has non-negative float32 half_widths array
- Registry: get_backend("dlt", calibration_path=...) returns DltBackend instance
- Module constants: all five constants exist with correct values
- Confidence weighting: both None and non-None point_confidence produce valid results

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Implement DltBackend (TDD) | 210c0e7 | dlt.py, test_dlt_backend.py |
| 2 | Register DltBackend in backends/__init__.py | 1c54c11 | backends/__init__.py |

## Verification Results

- `hatch run test tests/unit/core/reconstruction/ -x`: 748 passed, 3 skipped, 0 failed
- `hatch run check`: ruff lint passes cleanly; basedpyright 40 errors all pre-existing in unrelated engine files
- `get_backend("dlt", calibration_path="...")` returns DltBackend instance
- DltBackend.reconstruct_frame produces Midline3D with all required fields
- No camera-count branching in DltBackend (confirmed: single triangulate → reject → re-triangulate path)
- No orientation alignment or epipolar refinement in DltBackend

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mock model geometry produced all triangulations at water surface**
- Found during: Task 1 TDD GREEN phase
- Issue: Initial mock model cast rays straight down (+Z) from water surface (Z=0). All 3 camera rays were parallel (same direction), so DLT triangulated to Z=0 exactly = water_z → rejected by water surface filter. Result: empty reconstruction.
- Fix: Redesigned mock model to use proper pinhole geometry — camera at negative Z (above water), looking in +Z direction. Rays computed as normalized direction from camera center through pixel, intersecting water surface to get origin. Fish placed at Z=0.5 (positive = underwater). Triangulation correctly produces Z=0.5 > water_z=0.0.
- Files modified: tests/unit/core/reconstruction/test_dlt_backend.py (cast_ray and project functions in _make_mock_model)
- Commit: 210c0e7 (included in main implementation commit)

**2. [Rule 1 - Bug] _make_midline_set used np.stack with scalar v causing shape mismatch**
- Found during: Task 1 TDD GREEN phase
- Issue: v = scalar float was stacked with u = array via np.stack([u, v]), producing "all arrays must have the same shape" error
- Fix: Changed to v = np.full(n_body_points, v_val) then np.column_stack([u, v])
- Files modified: tests/unit/core/reconstruction/test_dlt_backend.py
- Commit: 210c0e7 (included in main implementation commit)

**3. [Rule 2 - Missing] MagicMock(spec=mock_object) raises InvalidSpecError**
- Found during: Task 1 TDD GREEN phase
- Issue: Python 3.12 unittest.mock forbids speccing a Mock with another Mock object
- Fix: Replaced `MagicMock(spec=model)` with plain `MagicMock()` in TestWaterSurfaceRejection
- Files modified: tests/unit/core/reconstruction/test_dlt_backend.py
- Commit: 210c0e7 (included in main implementation commit)

## Self-Check: PASSED

- [x] src/aquapose/core/reconstruction/backends/dlt.py exists: FOUND
- [x] tests/unit/core/reconstruction/test_dlt_backend.py exists: FOUND
- [x] Commit 210c0e7 exists: FOUND
- [x] Commit 1c54c11 exists: FOUND
- [x] get_backend("dlt") returns DltBackend: VERIFIED (test passes)
- [x] 748 tests pass: VERIFIED
