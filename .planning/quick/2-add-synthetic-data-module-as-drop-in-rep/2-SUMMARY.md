---
phase: quick-2
plan: 01
subsystem: synthetic
tags: [synthetic-data, testing, triangulation, curve-optimizer, diagnostics]
dependency_graph:
  requires:
    - aquapose.calibration.projection.RefractiveProjectionModel
    - aquapose.reconstruction.triangulation (MidlineSet, Midline3D, SPLINE_KNOTS, SPLINE_K)
    - aquapose.reconstruction.midline.Midline2D
  provides:
    - aquapose.synthetic (FishConfig, generate_synthetic_midline_sets, build_fabricated_rig)
    - scripts/diagnose_pipeline.py --synthetic flag
  affects:
    - scripts/diagnose_pipeline.py
tech_stack:
  added:
    - src/aquapose/synthetic/ (new package: fish.py, rig.py, stubs.py, __init__.py)
    - tests/unit/synthetic/ (new test package)
  patterns:
    - Pinhole half-width approximation: hw_px = hw_m * fx / depth
    - B-spline GT via make_lsq_spline with canonical SPLINE_KNOTS
    - NaN masking for occluded body points in Midline2D
    - Ground truth comparison: mean Euclidean distance between control points in mm
key_files:
  created:
    - src/aquapose/synthetic/__init__.py
    - src/aquapose/synthetic/fish.py
    - src/aquapose/synthetic/rig.py
    - src/aquapose/synthetic/stubs.py
    - tests/unit/synthetic/__init__.py
    - tests/unit/synthetic/test_synthetic.py
  modified:
    - scripts/diagnose_pipeline.py
decisions:
  - Circular arc curvature sign convention: positive curvature bends left (Y direction), centre of curvature at +sign*r perpendicular to heading
  - Half-width world-to-pixel conversion uses same pinhole approximation as triangulation.py (_pixel_half_width_to_metres)
  - diagnose_pipeline.py refactored into main() + _run_synthetic() + _run_real() for clean path separation
  - Fabricated rig fallback: --synthetic uses build_fabricated_rig when --calibration path does not exist
  - Ground truth comparison skips frames where fish_id not in reconstruction output (not an error)
metrics:
  duration: 9 minutes
  completed: 2026-02-23
  tasks_completed: 2
  files_created: 6
  files_modified: 1
---

# Quick Task 2: Synthetic Data Module Summary

**One-liner:** Synthetic fish shape generation (straight line + circular arc) projected through refractive cameras, producing MidlineSet/Midline3D compatible with both triangulation and CurveOptimizer, with --synthetic flag in diagnose_pipeline.py enabling quantitative ground truth comparison.

## What Was Built

### src/aquapose/synthetic/ (new package)

**rig.py** — `build_fabricated_rig(n_cameras_x=3, n_cameras_y=3, ...)` creates rectangular grids of synthetic downward-looking cameras with identity rotation, placed symmetrically around origin. Camera IDs: `"syn_00"`, `"syn_01"`, etc.

**fish.py** — Core synthetic data generation:
- `FishConfig` dataclass: position, heading_rad, curvature, scale, n_points
- `generate_fish_3d(config)`: straight line (curvature=0) or circular arc (curvature!=0) fish shapes in 3D
- `generate_fish_half_widths(n_points, max_ratio, scale)`: elliptical body profile peaking at 40% from head
- `make_ground_truth_midline3d(...)`: B-spline GT via `make_lsq_spline` with SPLINE_KNOTS; n_cameras=99, residuals=0
- `project_fish_to_midline2d(...)`: refractive projection with NaN masking for invalid points, returns None if <3 visible
- `generate_synthetic_midline_sets(models, fish_configs, n_frames)`: full pipeline producing (list[MidlineSet], list[dict[int, Midline3D]])

**stubs.py** — `generate_synthetic_detections` and `generate_synthetic_tracks` stubs (NotImplementedError with descriptive docstrings)

### tests/unit/synthetic/test_synthetic.py (11 tests)

All 11 pass in <2s (no `@pytest.mark.slow`):
- `test_build_fabricated_rig_default` — 9 cameras, syn_ prefix, RefractiveProjectionModel instances
- `test_build_fabricated_rig_custom` — 2x4 = 8 cameras
- `test_generate_fish_3d_straight` — correct point count, arc length ~= scale
- `test_generate_fish_3d_arc` — curvature=10 produces correct arc length, non-trivial curve
- `test_generate_fish_3d_heading` — heading=pi/2 spreads along Y axis
- `test_project_fish_returns_midline2d` — correct shape, fish_id, camera_id, frame_index
- `test_project_fish_returns_none_when_not_visible` — fish above water_z returns None
- `test_generate_synthetic_midline_sets_structure` — 1 frame, 1 fish, multi-camera
- `test_generate_synthetic_midline_sets_multi_fish` — 3 fish configs, all 3 fish_ids present
- `test_ground_truth_midline3d_valid` — control_points shape (7,3), knots (11,), arc_length > 0
- `test_round_trip_accuracy` — projected pixels within 1px of direct model.project() call

### scripts/diagnose_pipeline.py (updated)

Refactored into three functions:
- `main()`: routes to `_run_synthetic()` or `_run_real()` based on `--synthetic` flag
- `_run_synthetic()`: generates fish configs, calls `generate_synthetic_midline_sets`, runs Stage 5, prints GT comparison
- `_run_real()`: original pipeline logic (unchanged)

New CLI flags:
- `--synthetic`: bypass stages 1-4
- `--n-fish N`: number of synthetic fish (alternating straight/arc, positions spread along X)
- `--n-synthetic-cameras N`: NxN fabricated rig when --calibration not available

Ground truth comparison output example:
```
=== Synthetic Ground Truth Comparison ===
  Fish 0: mean control-point error = 7.78 mm over 3 frame(s)

  Overall mean control-point error: 7.78 mm
```

## Deviations from Plan

None — plan executed exactly as written.

## Verification Results

- `hatch run check` passes (ruff + basedpyright on src/)
- `hatch run test tests/unit/synthetic/` — 11/11 pass
- `python scripts/diagnose_pipeline.py --synthetic --stop-frame 3 --method triangulation` completes successfully with 3/3 frames with 3D midlines, 7.78mm GT error
- `python scripts/diagnose_pipeline.py --synthetic --stop-frame 3 --method curve` completes successfully with 6.08mm GT error
- Pre-existing test failures (4) in test_triangulation, test_tracker, test_overlay are unchanged

## Self-Check: PASSED

Files created/modified verified:
- src/aquapose/synthetic/__init__.py: EXISTS
- src/aquapose/synthetic/fish.py: EXISTS
- src/aquapose/synthetic/rig.py: EXISTS
- src/aquapose/synthetic/stubs.py: EXISTS
- tests/unit/synthetic/__init__.py: EXISTS
- tests/unit/synthetic/test_synthetic.py: EXISTS
- scripts/diagnose_pipeline.py: MODIFIED

Commits:
- 1bfee86: feat(quick-2-01): add synthetic data module for controlled pipeline testing
- b9dd734: feat(quick-2-01): add synthetic unit tests and --synthetic flag for diagnose_pipeline.py
