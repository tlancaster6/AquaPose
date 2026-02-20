---
phase: 01-calibration-and-refractive-geometry
verified: 2026-02-19T23:07:22Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 1: Calibration and Refractive Geometry Verification Report

**Phase Goal:** A validated, differentiable refractive projection layer is available that downstream phases can import with confidence -- errors here would silently corrupt every gradient in the system
**Verified:** 2026-02-19T23:07:22Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | load_calibration_data() returns CalibrationData with cameras dict, water_z, interface_normal, n_air, n_water from AquaCal JSON | VERIFIED | loader.py lines 108-162; 7 loader unit tests cover shapes, dtypes, squeezing, water_z extraction; all pass |
| 2 | RefractiveProjectionModel.project() maps 3D underwater points to 2D pixel coordinates with Newton-Raphson solver | VERIFIED | projection.py lines 125-213; 10-iteration fixed Newton-Raphson; nadir test confirms correct projection; 8 project tests pass |
| 3 | RefractiveProjectionModel.cast_ray() maps 2D pixel coordinates to 3D underwater ray origins and directions via Snell law | VERIFIED | projection.py lines 65-123; Snell law vectorized; origins-on-surface, unit-direction, and angle tests all pass |
| 4 | Autograd backward pass completes without error through project() and cast_ray() | VERIFIED | TestProject::test_autograd_backward_completes and TestCastRay::test_autograd_backward_completes; both pass with finite non-zero grads |
| 5 | torch.autograd.gradcheck passes for project() with float64 inputs | VERIFIED | TestProject::test_gradcheck_float64; passes with atol=1e-4, rtol=1e-3 |
| 6 | Cross-validation against reference implementation passes within atol=1e-5 for both project and cast_ray | VERIFIED | 12 slow integration tests in test_calibration_cross_validation.py; all pass at atol=1e-5 against AquaCal refractive_project and trace_ray_air_to_water |
| 7 | Z-uncertainty report quantifies X/Y/Z reconstruction error as a function of tank depth for 13-camera geometry | VERIFIED | docs/reports/z_uncertainty_report.md contains 10-row depth table (0.800-1.250m), summary stats, interpretation paragraph |
| 8 | Report shows error curves at uniform depth intervals across the full tank range | VERIFIED | 5cm intervals, 10 depths from 0.800m to 1.250m; three PNG plots committed alongside report |
| 9 | X and Y errors are substantially smaller than Z errors for top-down cameras | VERIFIED | Z/XY anisotropy 30x-577x (mean 132x); test_z_errors_larger_than_xy_errors_for_top_down_geometry passes |
| 10 | Report includes embedded matplotlib plots showing error vs. depth for X, Y, Z separately | VERIFIED | z_uncertainty_xyz_error.png, z_uncertainty_ratio.png, z_uncertainty_cameras.png all exist; embedded as relative image links in report |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/calibration/loader.py` | AquaCal bridge -- load JSON into PyTorch tensors | VERIFIED | 216 lines; load_calibration_data, CameraData, CalibrationData, UndistortionMaps, compute_undistortion_maps, undistort_image all present |
| `src/aquapose/calibration/projection.py` | Differentiable refractive projection, ray casting, and triangulation | VERIFIED | RefractiveProjectionModel with project(), cast_ray(), to(); triangulate_rays() moved here from uncertainty.py during post-phase cleanup; non-in-place ops and epsilon guards throughout |
| `src/aquapose/calibration/__init__.py` | Public API for calibration module | VERIFIED | Exports 12 public symbols via __all__; covers both plan 01-01 and 01-02 symbols |
| `tests/integration/test_calibration_cross_validation.py` | Cross-validation tests against AquaCal reference | VERIFIED | 12 tests in 2 classes (TestCastRayCrossValidation, TestProjectCrossValidation); all @pytest.mark.slow; all pass at atol=1e-5 |
| `src/aquapose/calibration/uncertainty.py` | Z-uncertainty analytical characterization | VERIFIED | compute_triangulation_uncertainty, build_synthetic_rig, build_rig_from_calibration, generate_uncertainty_report, UncertaintyResult all present; triangulate_rays moved to projection.py |
| `tests/unit/calibration/test_uncertainty.py` | Unit tests for uncertainty calculations | VERIFIED | 16 tests across 3 classes; includes test_compute_triangulation_uncertainty; all pass |
| `docs/reports/z_uncertainty_report.md` | Z-uncertainty characterization report with numeric data | VERIFIED | Numeric Z/XY ratios in depth table; anisotropy interpretation paragraph present; embedded plot links |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `loader.py` | `aquacal.io.serialization.load_calibration` | import and call | WIRED | Line 9: `from aquacal.io.serialization import load_calibration as aquacal_load_calibration`; called on line 121 |
| `projection.py` | `loader.py` CameraData tensors (K, R, t) | RefractiveProjectionModel constructor | WIRED | Constructor accepts K, R, t directly; cross-validation tests build both from same params dict confirming protocol compatibility |
| `tests/integration/test_calibration_cross_validation.py` | `aquacal.core.refractive_geometry` | NumPy reference comparison | WIRED | Lines 11-12: imports refractive_project, trace_ray_air_to_water; called for every test point in both project and cast_ray test classes |
| `uncertainty.py` | `projection.py` RefractiveProjectionModel + triangulate_rays | uses project(), cast_ray(), and triangulate_rays() | WIRED | `from .projection import RefractiveProjectionModel, triangulate_rays` |
| `uncertainty.py` | `aquacal.datasets.synthetic` | build_synthetic_rig() | WIRED | Lazy import of `generate_camera_intrinsics` inside build_synthetic_rig() |

### Requirements Coverage

No REQUIREMENTS.md phase mapping exists for phase 1. All success criteria from both PLANs (01-01, 01-02) verified directly above.

### Anti-Patterns Found

None. All calibration source files scanned for TODO/FIXME/XXX/placeholder patterns -- zero matches. No empty return stubs detected. No console.log-only handlers.

### Human Verification Required

None required. All automated checks pass. The key correctness properties (Snell law angles, Newton-Raphson convergence, gradcheck, Z/XY anisotropy) are verified programmatically. No visual or real-time behavior to assess.

### Gaps Summary

No gaps. All 10 observable truths verified, all 7 required artifacts exist at full implementation depth (not stubs), all 5 key links wired and confirmed active.

**Notable plan deviation (not a gap):** The cross-validation tests compare against AquaCal NumPy (refractive_project, trace_ray_air_to_water) rather than AquaMVS PyTorch as originally specified in plan 01-01. AquaMVS cannot be imported in the hatch environment due to missing open3d/lightglue dependencies. The SUMMARY documents this as an intentional decision -- AquaCal NumPy is the shared ground truth that both AquaMVS and AquaPose implement against, making it a stronger validation target. Tests pass at atol=1e-5.

**Test totals (confirmed by running):**
- 62 unit tests pass: 22 loader, 25 projection, 16 uncertainty, 1 smoke
- 12 integration/cross-validation tests pass (slow, require aquacal)
- 74 total tests, 0 failures, 0 errors

---

_Verified: 2026-02-19T23:07:22Z_
_Verifier: Claude (gsd-verifier)_
