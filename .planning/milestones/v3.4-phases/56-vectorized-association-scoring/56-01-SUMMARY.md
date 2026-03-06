---
phase: 56-vectorized-association-scoring
plan: 01
subsystem: core
tags: [numpy, vectorization, ray-geometry, association]

requires:
  - phase: none
    provides: n/a
provides:
  - ray_ray_closest_point_batch() pure-NumPy vectorized function
  - Parametric identity tests validating batch vs scalar
affects: [56-02]

tech-stack:
  added: []
  patterns: [scalar-to-batch broadcasting with np.where for near-parallel masking]

key-files:
  created: []
  modified:
    - src/aquapose/core/association/scoring.py
    - src/aquapose/core/association/__init__.py
    - tests/unit/core/association/test_scoring.py

key-decisions:
  - "Cast all inputs to float64 inside batch function to match scalar path precision"
  - "Use np.where with safe_denom pattern to avoid branching on parallel mask"
  - "Return distances only (no midpoints) since score_tracklet_pair discards them"

patterns-established:
  - "Scalar-to-batch: element-wise sum(axis=1) replaces np.dot for batched dot products"

requirements-completed: [ASSOC-01, ASSOC-02]

duration: 5min
completed: 2026-03-04
---

# Plan 56-01: Vectorized ray_ray_closest_point_batch Summary

**Pure-NumPy vectorized ray-ray distance for N pairs via element-wise broadcasting with near-parallel fallback**

## Performance

- **Duration:** 5 min
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Implemented ray_ray_closest_point_batch() matching scalar output within atol=1e-6 across random, parallel, mixed, empty, single, and intersecting ray configurations
- Updated exports in scoring.__all__ and association.__init__.__all__
- All 817 existing tests continue to pass

## Task Commits

1. **Task 1+2: Implement batch function, tests, and exports** - `331df95` (feat)

## Files Created/Modified
- `src/aquapose/core/association/scoring.py` - Added ray_ray_closest_point_batch() after scalar version
- `src/aquapose/core/association/__init__.py` - Added batch function to imports and __all__
- `tests/unit/core/association/test_scoring.py` - Added TestRayRayClosestPointBatch with 10 test cases

## Decisions Made
None - followed plan as specified

## Deviations from Plan
None - plan executed exactly as written

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ray_ray_closest_point_batch() ready for Plan 56-02 to use in score_tracklet_pair() rewrite
- All existing tests pass, providing regression safety for Plan 56-02

---
*Phase: 56-vectorized-association-scoring*
*Completed: 2026-03-04*
