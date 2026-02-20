---
phase: 01-calibration-and-refractive-geometry
plan: 01
subsystem: calibration
tags: [pytorch, aquacal, refractive-geometry, snells-law, newton-raphson, autograd, opencv]

# Dependency graph
requires: []
provides:
  - "load_calibration_data(): loads AquaCal JSON into PyTorch CameraData/CalibrationData tensors"
  - "RefractiveProjectionModel.project(): differentiable 3D->2D refractive projection via Newton-Raphson"
  - "RefractiveProjectionModel.cast_ray(): 2D->3D ray casting via Snell's law at air-water interface"
  - "triangulate_rays(): SVD least-squares ray intersection (moved from uncertainty.py during post-phase cleanup)"
  - "compute_undistortion_maps(): OpenCV fisheye/pinhole undistortion remap tables"
  - "Cross-validated against AquaCal NumPy at atol=1e-5"
affects:
  - phase-02-segmentation
  - phase-03-fish-mesh
  - phase-04-analysis-by-synthesis
  - phase-05-multi-fish-tracking
  - phase-06-evaluation

# Tech tracking
tech-stack:
  added:
    - "aquacal (PyPI) — AquaCal calibration JSON loading"
    - "cv2.fisheye.* / cv2.initUndistortRectifyMap — OpenCV undistortion"
  patterns:
    - "Newton-Raphson with fixed 10 iterations for autograd-safe refractive solve"
    - "torch.clamp + torch.minimum (not in-place) to preserve autograd graph"
    - "Epsilon 1e-12 in sqrt and denominators to avoid NaN gradients"
    - "float32 for K/R/t/tensors, float64 preserved for dist_coeffs (OpenCV requirement)"

key-files:
  created:
    - "src/aquapose/calibration/loader.py"
    - "src/aquapose/calibration/projection.py"
    - "tests/unit/calibration/__init__.py"
    - "tests/unit/calibration/test_loader.py"
    - "tests/unit/calibration/test_projection.py"
    - "tests/integration/test_calibration_cross_validation.py"
  modified:
    - "src/aquapose/calibration/__init__.py"
    - "pyproject.toml"

key-decisions:
  - "Cross-validation compares against AquaCal NumPy (refractive_project, trace_ray_air_to_water) rather than AquaMVS PyTorch — AquaMVS cannot be imported in hatch env due to missing open3d/lightglue deps; AquaCal NumPy is the ground truth both use anyway"
  - "aquacal installed via pip install -e (local editable) in hatch env; added as pyproject.toml dependency for reproducibility"
  - "K_inv float32 inversion tolerance set to atol=1e-4 (not 1e-5) — float32 inversion with fx=1400 produces ~6e-5 error, which is expected float32 precision"

patterns-established:
  - "Calibration fixture pattern: mock aquacal_load_calibration with synthetic CalibrationResult for unit tests (no file I/O)"
  - "Cross-validation pattern: build parallel AquaCal Camera+Interface and AquaPose RefractiveProjectionModel from same params dict, compare numerically"
  - "Refractive projection test geometry: camera at (0.635, 0, 0), R=eye, t=(-0.635,0,0), fx=fy=1400, water_z=0.978"

# Metrics
duration: 45min
completed: 2026-02-19
---

# Phase 1 Plan 1: Calibration Loader and Refractive Projection Model Summary

**Differentiable refractive projection via 10-iteration Newton-Raphson and Snell's law ray casting, cross-validated against AquaCal NumPy at atol=1e-5 across 60 tests**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-02-19T00:00:00Z
- **Completed:** 2026-02-19T00:45:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Ported `RefractiveProjectionModel` with differentiable `project()` (Newton-Raphson) and `cast_ray()` (Snell's law) from AquaMVS into `aquapose.calibration`
- Ported `load_calibration_data()` and `compute_undistortion_maps()` from AquaMVS, wiring AquaCal JSON into PyTorch float32 tensors
- 60 tests pass: 48 unit tests covering shapes/dtypes/autograd/gradcheck/roundtrip, 12 cross-validation tests against AquaCal NumPy reference at atol=1e-5

## Task Commits

1. **Task 1: Port calibration loader and refractive projection model** - `6b46cb4` (feat)
2. **Task 2: Write unit and cross-validation tests** - `07f4416` (feat)

## Files Created/Modified

- `src/aquapose/calibration/loader.py` — CameraData, CalibrationData, UndistortionMaps, load_calibration_data(), compute_undistortion_maps(), undistort_image() (UndistortionData alias removed during post-phase cleanup)
- `src/aquapose/calibration/projection.py` — RefractiveProjectionModel with project() and cast_ray(); triangulate_rays() (moved here from uncertainty.py during post-phase cleanup)
- `src/aquapose/calibration/__init__.py` — Public API with sorted __all__
- `pyproject.toml` — Added aquacal dependency
- `tests/unit/calibration/test_loader.py` — 22 unit tests for loader module
- `tests/unit/calibration/test_projection.py` — 25 unit tests for projection model
- `tests/integration/test_calibration_cross_validation.py` — 12 cross-validation tests (slow)

## Decisions Made

- **Cross-validation reference**: Compared AquaPose against AquaCal NumPy (`refractive_project`, `trace_ray_air_to_water`) rather than AquaMVS PyTorch. AquaMVS cannot be imported in the hatch environment due to missing `open3d`/`lightglue` dependencies. AquaCal NumPy is the true ground truth both AquaMVS and AquaPose implement against.
- **aquacal installation**: Installed via `pip install -e` in hatch env; added to `pyproject.toml` dependencies for reproducibility.
- **K_inv tolerance**: Unit test uses `atol=1e-4` for `K @ K_inv ≈ I` — float32 inversion of a matrix with focal length 1400 produces ~6e-5 error, which is expected float32 precision (not a bug in the implementation).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed overly-tight K_inv tolerance in test**
- **Found during:** Task 2 (unit test execution)
- **Issue:** `test_precomputes_K_inv` used `atol=1e-5` but float32 inversion of K with fx=1400 produces ~6e-5 error — a normal float32 precision artifact, not a code bug
- **Fix:** Changed test tolerance from `atol=1e-5` to `atol=1e-4` with explanatory docstring
- **Files modified:** `tests/unit/calibration/test_projection.py`
- **Verification:** Test passes; documented in docstring why 1e-4 is appropriate
- **Committed in:** `07f4416` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 test tolerance bug)
**Impact on plan:** Minor test fix only; no production code changes. Implementation matches specification exactly.

## Issues Encountered

- **AquaMVS not importable**: `aquamvs` package requires `open3d` and `lightglue` which are not available in the hatch environment. Resolution: cross-validation against AquaCal NumPy directly (the actual ground truth), matching the pattern AquaMVS itself uses in its own test suite.
- **aquacal not installed**: Neither aquacal nor aquamvs were pre-installed in hatch env. Installed aquacal via `pip install -e` from local checkout; aquamvs installed with `--no-deps`.

## Next Phase Readiness

- `aquapose.calibration` module is complete and cross-validated — all downstream phases can depend on it
- `RefractiveProjectionModel.project()` and `cast_ray()` are differentiable — ready for Phase 4 analysis-by-synthesis gradient descent
- Phase 2 (segmentation) and Phase 3 (fish mesh) can proceed independently; both depend only on Phase 1 outputs which are now available

---
*Phase: 01-calibration-and-refractive-geometry*
*Completed: 2026-02-19*
