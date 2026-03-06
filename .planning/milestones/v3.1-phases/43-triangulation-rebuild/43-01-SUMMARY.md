---
phase: 43-triangulation-rebuild
plan: "01"
subsystem: reconstruction
tags: [refactor, extraction, utils, spline, triangulation]
dependency_graph:
  requires: []
  provides: [reconstruction/utils.py shared helper functions]
  affects: [reconstruction/triangulation.py, future DltBackend]
tech_stack:
  added: []
  patterns: [public helper extraction, backward-compatible aliases, noqa re-export]
key_files:
  created:
    - src/aquapose/core/reconstruction/utils.py
    - tests/unit/core/reconstruction/test_reconstruction_utils.py
  modified:
    - src/aquapose/core/reconstruction/triangulation.py
decisions:
  - MIN_BODY_POINTS re-exported from triangulation.py via noqa F401 to preserve existing test imports without changing test files
  - Backward-compat aliases (_fit_spline, _pixel_half_width_to_metres, etc.) kept in triangulation.py for zero-change backward compatibility
  - utils.py fit_spline uses build_spline_knots(7) as default (matching old SPLINE_KNOTS constant) rather than referencing SPLINE_KNOTS
metrics:
  duration: "8 minutes"
  completed: "2026-03-02"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 1
---

# Phase 43 Plan 01: Extract Shared Reconstruction Helpers to utils.py Summary

Pure refactor extracting four shared helper functions from triangulation.py into a new utils.py module to enable code reuse by the upcoming DltBackend without duplication.

## What Was Built

`src/aquapose/core/reconstruction/utils.py` — new shared helper module providing:
- `build_spline_knots(n_control_points)` — builds clamped cubic B-spline knot vectors
- `weighted_triangulate_rays(origins, directions, weights)` — weighted DLT triangulation
- `fit_spline(u_param, pts_3d, knots, min_body_points)` — fits cubic B-spline to 3D points
- `pixel_half_width_to_metres(hw_px, depth_m, focal_px)` — pinhole approximation conversion
- Constants: `SPLINE_K = 3`, `MIN_BODY_POINTS = 9`

`src/aquapose/core/reconstruction/triangulation.py` — updated to:
- Import all four functions and both constants from utils.py
- Remove the four private function definitions (saved ~130 lines)
- Keep backward-compatible aliases (`_fit_spline`, `_pixel_half_width_to_metres`, etc.)
- Re-export `MIN_BODY_POINTS` with `# noqa: F401` to preserve existing test imports

`tests/unit/core/reconstruction/test_reconstruction_utils.py` — 279-line test file covering:
- `build_spline_knots` for n=7 (11 knots), n=4 (8 knots), length formula, dtype, monotonicity
- `weighted_triangulate_rays` with 4-camera uniform weight recovery, output shape, weight bias
- `fit_spline` None for too-few points, correct shapes on valid input, arc length positivity
- `pixel_half_width_to_metres` exact arithmetic (10, 0.5, 500 = 0.01), proportionality checks
- Module constant type assertions

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create reconstruction/utils.py with extracted helpers (TDD) | 4777375 | utils.py, triangulation.py, test_reconstruction_utils.py |
| 2 | Verify extraction is behavior-preserving | (no new files) | verified 728 tests pass |

## Verification Results

- `hatch run test tests/unit/core/reconstruction/ tests/unit/test_triangulation.py`: 728 passed, 0 failed
- `hatch run typecheck`: 0 errors in modified files (40 pre-existing errors in unrelated engine files)
- `hatch run lint`: passes (ruff auto-fixed formatting on first commit)
- utils.py: 4 public functions, 2 constants, `__all__` defined
- triangulation.py: no longer defines `_build_spline_knots`, `_weighted_triangulate_rays`, `_fit_spline`, or `_pixel_half_width_to_metres` locally

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] TDD test for weighted_triangulate_rays weight-bias used 2-camera geometry where weights have no effect**
- Found during: Task 1 TDD GREEN phase
- Issue: With 2 cameras, weighted DLT least-squares always produces the same point regardless of weights because there is only one geometric solution
- Fix: Replaced with 3-camera test where 2 cameras agree on target_a and 1 camera points at target_b; high weight on the minority camera biases result correctly
- Files modified: tests/unit/core/reconstruction/test_reconstruction_utils.py
- Commit: 4777375 (included in same TDD commit)

**2. [Rule 2 - Missing] MIN_BODY_POINTS import removed by ruff as "unused"**
- Found during: pre-commit hook after first commit attempt
- Issue: ruff F401 removed MIN_BODY_POINTS from triangulation.py imports because it's only re-exported (used in docstring and by test_triangulation.py), causing ImportError in existing test
- Fix: Added `# noqa: F401` comment to the import line to preserve re-export intentionally
- Files modified: src/aquapose/core/reconstruction/triangulation.py
- Commit: 4777375 (fixed before successful commit)

## Self-Check: PASSED

- [x] src/aquapose/core/reconstruction/utils.py exists: FOUND
- [x] tests/unit/core/reconstruction/test_reconstruction_utils.py exists: FOUND
- [x] Commit 4777375 exists: FOUND
- [x] triangulation.py imports from utils.py: VERIFIED (grep confirmed)
- [x] No old private function definitions remain in triangulation.py: VERIFIED
