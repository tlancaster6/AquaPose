---
phase: 09-curve-based-optimization-as-a-replacement-for-triangulation
plan: 01
subsystem: reconstruction
tags: [optimization, b-spline, chamfer, lbfgs, cuda, coarse-to-fine, warm-start]
dependency_graph:
  requires:
    - src/aquapose/calibration/projection.py (RefractiveProjectionModel.project)
    - src/aquapose/reconstruction/triangulation.py (SPLINE_KNOTS, Midline3D, MidlineSet)
  provides:
    - src/aquapose/reconstruction/curve_optimizer.py (CurveOptimizerConfig, CurveOptimizer, optimize_midlines)
  affects:
    - src/aquapose/reconstruction/__init__.py (updated __all__)
tech_stack:
  added: []
  patterns:
    - Batched L-BFGS optimization with (N_fish, K, 3) tensors on GPU
    - B-spline basis matrix cache for O(1) lookup after first build
    - Chamfer distance via torch.cdist (correspondence-free matching)
    - Huber loss for robust per-camera aggregation
    - Coarse-to-fine: K=4 -> K=7 control points via basis matrix upsample
    - Adaptive early stopping with per-fish convergence masking
key_files:
  created:
    - src/aquapose/reconstruction/curve_optimizer.py
    - tests/unit/test_curve_optimizer.py
  modified:
    - src/aquapose/reconstruction/__init__.py
decisions:
  - Clamped cos_bend to [-1+1e-6, 1-1e-6] before acos to prevent NaN gradients at collinear control points (acos gradient is -1/sqrt(1-x^2), undefined at x=+-1)
  - Bend angle computed as acos(v1.v2) not acos(-(v1.v2)) - 0 for straight spine, pi for U-turn
  - Curvature penalty is only active when bend_angle > max_bend_angle_deg; at init (straight spine) penalty and gradient are both 0
  - Huber delta fixed at 17.5px (midpoint of 15-20px range from RESEARCH.md Pitfall 7)
  - warm_start_loss_ratio check done per-fish before fine stage: reverts to cold start if loss > 2x prev_loss
  - test_optimize_synthetic_fish marked @pytest.mark.slow (takes 5-10s on CPU)
metrics:
  duration_minutes: 13
  tasks_completed: 2
  files_created: 2
  files_modified: 1
  completed_date: "2026-02-23"
requirements_satisfied: [RECON-03, RECON-04, RECON-05]
---

# Phase 09 Plan 01: CurveOptimizer Implementation Summary

One-liner: Correspondence-free B-spline optimizer via batched L-BFGS + chamfer distance + refractive reprojection with coarse-to-fine, warm-start, and adaptive early stopping.

## Objective

Implement `src/aquapose/reconstruction/curve_optimizer.py` — the core module for Phase 9's curve-based 3D midline reconstruction. This replaces point-correspondence triangulation with direct optimization of 3D B-spline control points against 2D skeleton observations.

## What Was Built

### Task 1: `curve_optimizer.py`

- **`CurveOptimizerConfig` dataclass**: All regularization weights exposed (lambda_length, lambda_curvature, lambda_smoothness, max_bend_angle_deg, n_coarse/fine_ctrl, lbfgs params, convergence, warm_start_loss_ratio).
- **B-spline basis cache**: `_build_basis_matrix(n_eval, n_ctrl)` builds clamped uniform cubic B-spline basis matrices. `get_basis()` provides module-level `_BASIS_CACHE` keyed by `(n_eval, n_ctrl)`.
- **Chamfer distance**: `_chamfer_distance_2d(proj, obs)` via `torch.cdist`. Handles empty tensors. Returns mean of both directed distances.
- **Loss functions**: `_data_loss` (chamfer + Huber per camera), `_length_penalty` (±30% tolerance band), `_curvature_penalty` (per-joint bend angle limit), `_smoothness_penalty` (second-difference).
- **Initialization**: `_cold_start` (straight line centered at centroid), `_init_ctrl_pts` (warm-start with K mismatch handling via basis upsample, else cold start). `_upsample_ctrl_pts` for coarse→fine transition.
- **`CurveOptimizer` class**: Coarse stage (K=4, L-BFGS), upsample, warm-start fallback check, fine stage (K=7) with manual per-fish adaptive convergence masking. Outputs `dict[int, Midline3D]` matching triangulation.py contract. Stores `_warm_starts` and `_prev_losses` for next frame.
- **`optimize_midlines()` convenience wrapper**: Single-frame stateless interface.

### Task 2: `test_curve_optimizer.py` (21 tests)

- B-spline basis: shape, partition-of-unity, endpoint interpolation, cache hit
- Chamfer: identical points, known distance, empty inputs
- Length penalty: zero at nominal, positive outside tolerance
- Curvature penalty: zero for straight spine, positive for 90° bend
- Smoothness penalty: zero for evenly-spaced collinear points
- Cold start: shape, centering, length span, orientation
- Integration: `test_optimize_synthetic_fish` (3-camera rig, known GT spline, arc length within 30%, residual < 20px)
- Warm-start storage, empty input handling, config defaults

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed NaN gradient from `acos` at collinear control points**
- **Found during:** Task 2 — `test_optimize_synthetic_fish` produced NaN control points in optimization output
- **Issue:** `_curvature_penalty` used `cos_bend = -(v1·v2)` (supplement angle) instead of `cos_bend = v1·v2` (bend angle). For a straight spine, `-(v1·v2) = -1`, so `acos(-1) = π`. The gradient of `acos(-1)` is `-1/sqrt(1-1)` = NaN. Penalty was 0 (correct) but backward still flowed NaN gradient into control points, causing L-BFGS to diverge.
- **Fix 1:** Changed `cos_bend` to use `(v1_unit * v2_unit).sum(dim=2)` — 0° for straight, 90° for right-angle bend.
- **Fix 2:** Clamped `cos_bend` to `[-1+1e-6, 1-1e-6]` before `acos` so gradient stays finite even when control points are nearly collinear.
- **Files modified:** `src/aquapose/reconstruction/curve_optimizer.py`
- **Commit:** 642eca4

## Self-Check

Files created:
- `src/aquapose/reconstruction/curve_optimizer.py` — exists
- `tests/unit/test_curve_optimizer.py` — exists
- `src/aquapose/reconstruction/__init__.py` — modified

Commits:
- 4bd57eb: feat(09-01): implement CurveOptimizer — verified
- 642eca4: feat(09-01): add unit tests — verified

Tests: 21/21 passing
Lint: All checks passed
Typecheck: 0 errors in curve_optimizer.py (7 pre-existing errors in other files)
