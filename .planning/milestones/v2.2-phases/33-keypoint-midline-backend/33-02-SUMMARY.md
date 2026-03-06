---
phase: 33-keypoint-midline-backend
plan: 02
subsystem: reconstruction
tags: [triangulation, curve-optimizer, confidence-weighting, chamfer, DLT, weighted-least-squares]

# Dependency graph
requires:
  - phase: 33-01
    provides: DirectPoseBackend with Midline2D.point_confidence field populated
  - phase: 30-config-and-contracts
    provides: Midline2D dataclass with point_confidence field defined

provides:
  - Confidence-weighted DLT triangulation via _weighted_triangulate_rays in triangulation.py
  - Confidence-weighted chamfer distance via _weighted_chamfer_distance_2d in curve_optimizer.py
  - Both backends respect point_confidence when present; uniform weights when None (backward compat)

affects:
  - 34-stabilization
  - any phase consuming Midline3D quality metrics

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sqrt(confidence) weighting: per-point confidence from keypoint backend transformed to sqrt(conf) before use as weight"
    - "Backward compat guard: all weighted paths check weights != 1.0 before switching code paths"
    - "NaN exclusion: NaN coordinate body points excluded entirely from DLT, not passed as zero-weight rows"
    - "confidence_per_fish parallel structure: list of dicts mirroring midlines_per_fish, None entries use unweighted path"

key-files:
  created:
    - tests/unit/core/reconstruction/test_confidence_weighting.py
  modified:
    - src/aquapose/reconstruction/triangulation.py
    - src/aquapose/reconstruction/curve_optimizer.py

key-decisions:
  - "Weighted triangulation uses normal equations (A=sum w_i*M_i, b=sum w_i*M_i@o_i) not DLT A-matrix — mirrors existing triangulate_rays() approach for consistency"
  - "_tri_rays() local helper in _triangulate_body_point dispatches to weighted/unweighted based on _use_weights flag — avoids repeated weight checks in 3 code paths"
  - "confidence_per_fish is a parallel list structure alongside midlines_per_fish — passed to all _data_loss() calls including seed checks, snapshots, convergence monitors, and closures"
  - "obs->proj direction weighted in _weighted_chamfer_distance_2d; proj->obs direction remains unweighted (projected spline points have no per-point confidence)"

patterns-established:
  - "Parallel confidence structure: when adding per-point metadata to observations, mirror the midlines_per_fish list structure"
  - "None = backward compat: all weighted functions accept None as 'use unweighted path' for segment_then_extract compatibility"

requirements-completed:
  - RECON-01
  - RECON-02

# Metrics
duration: 35min
completed: 2026-03-01
---

# Phase 33 Plan 02: Confidence-Weighted Reconstruction Summary

**sqrt(confidence) weighting added to both triangulation (DLT normal equations) and curve optimizer (chamfer obs->proj direction), with full backward compatibility when point_confidence is None**

## Performance

- **Duration:** 35 min
- **Started:** 2026-03-01T01:06:53Z
- **Completed:** 2026-03-01T01:41:53Z
- **Tasks:** 2
- **Files modified:** 3 (2 src + 1 test)

## Accomplishments

- `_weighted_triangulate_rays()` added to triangulation.py using normal-equations approach (consistent with existing `triangulate_rays`), applied in 2-cam, 3-7 cam inlier retri, and 8+ cam paths
- `_weighted_chamfer_distance_2d()` added to curve_optimizer.py with per-point weighted obs->proj direction
- `triangulate_midlines()` extracts `sqrt(point_confidence)` per body point per camera and passes to `_triangulate_body_point()` via `weights` dict
- `optimize_midlines()` builds `confidence_per_fish` parallel structure and propagates to all `_data_loss()` calls (9 call sites updated)
- 12 unit tests covering uniform-weight backward compat, high-weight bias, NaN exclusion, outlier downweighting, and data-loss equivalence

## Task Commits

1. **Task 1: Confidence-weighted triangulation** - `8de6a04` (feat)
2. **Task 2: Confidence-weighted chamfer in curve_optimizer** - `ec31191` (feat)

## Files Created/Modified

- `src/aquapose/reconstruction/triangulation.py` - Added `_weighted_triangulate_rays()`, `weights` kwarg to `_triangulate_body_point()`, `_tri_rays()` local helper, and confidence extraction in `triangulate_midlines()`
- `src/aquapose/reconstruction/curve_optimizer.py` - Added `_weighted_chamfer_distance_2d()`, `confidence_per_fish` kwarg to `_data_loss()`, and `confidence_per_fish` parallel structure built in `optimize_midlines()`
- `tests/unit/core/reconstruction/test_confidence_weighting.py` - 12 unit tests across 4 test classes

## Decisions Made

- Normal equations approach for weighted DLT (`A = sum w_i * M_i`) mirrors existing `triangulate_rays()` exactly — unweighted case with all w=1.0 produces identical output
- `_tri_rays()` local helper dispatches to weighted or unweighted based on a `_use_weights` flag computed once per call — avoids repeated weight checks across 3 triangulation branches
- Seed pair triangulation in the 3-7 camera exhaustive search remains unweighted (pairwise scoring is for candidate selection, not accuracy) — only the final inlier re-triangulation is weighted
- `obs->proj` weighted, `proj->obs` unweighted in chamfer — projected spline points have no per-point confidence

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Test `test_high_weight_biases_result_toward_that_ray` initially failed because the 2-camera geometry was degenerate (rays happened to intersect close to the same point regardless of weights). Redesigned to use 3 cameras: 2 near-parallel noisy cameras vs 1 orthogonal accurate camera, making the weight effect clearly measurable.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Both reconstruction backends now consume `Midline2D.point_confidence` from DirectPoseBackend
- RECON-01 and RECON-02 complete
- Phase 33 complete: DirectPoseBackend (33-01) + confidence-weighted reconstruction (33-02)
- Ready for Phase 34 (Stabilization)

## Self-Check: PASSED

- `src/aquapose/reconstruction/triangulation.py` — FOUND
- `src/aquapose/reconstruction/curve_optimizer.py` — FOUND
- `tests/unit/core/reconstruction/test_confidence_weighting.py` — FOUND
- Commit `8de6a04` — FOUND (feat(33-02): add confidence-weighted DLT triangulation)
- Commit `ec31191` — FOUND (feat(33-02): add confidence-weighted chamfer)
- 676 tests pass, 0 failures

---
*Phase: 33-keypoint-midline-backend*
*Completed: 2026-03-01*
