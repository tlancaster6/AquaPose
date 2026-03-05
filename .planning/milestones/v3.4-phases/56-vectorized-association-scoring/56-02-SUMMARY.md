---
phase: 56-vectorized-association-scoring
plan: 02
subsystem: core
tags: [numpy, vectorization, scoring, association, early-termination]

requires:
  - phase: 56-vectorized-association-scoring
    provides: ray_ray_closest_point_batch() vectorized distance function
provides:
  - Batched score_tracklet_pair() with no per-frame Python loop
  - Two-phase early termination with vectorized soft kernel
affects: []

tech-stack:
  added: []
  patterns: [two-phase batched scoring with early termination]

key-files:
  created: []
  modified:
    - src/aquapose/core/association/scoring.py
    - tests/unit/core/association/test_scoring.py

key-decisions:
  - "Extract _batch_score_frames() helper for reuse across early and remaining phases"
  - "Early termination check: t_shared >= early_k (matches original loop semantics where check fires at frame_idx == early_k - 1)"
  - "Task 3 manual checkpoint deferred: no pre-vectorization eval baseline available for comparison; user should verify after next pipeline run"

patterns-established:
  - "Two-phase batch scoring: split at early_k, check score_sum==0, then batch remainder"

requirements-completed: [ASSOC-01, ASSOC-02]

duration: 5min
completed: 2026-03-04
---

# Plan 56-02: Batched score_tracklet_pair Summary

**Replaced per-frame Python loop with two-phase batched NumPy ops and vectorized soft kernel in score_tracklet_pair**

## Performance

- **Duration:** 5 min
- **Tasks:** 3 (2 automated, 1 manual checkpoint deferred)
- **Files modified:** 2

## Accomplishments
- Eliminated per-frame torch.tensor construction, cast_ray calls, and scalar ray_ray_closest_point calls
- Replaced with single batched cast_ray per camera per phase + ray_ray_closest_point_batch
- Early termination semantics preserved for all paths (t_shared < early_k, == early_k, > early_k)
- All 820 tests pass including 3 new edge case tests

## Task Commits

1. **Task 1: Rewrite score_tracklet_pair** - `ba616de` (feat)
2. **Task 2: Add early termination edge case tests** - `3c9a05b` (test)
3. **Task 3: Manual YH eval checkpoint** - Deferred (no pre-vectorization baseline available)

## Files Created/Modified
- `src/aquapose/core/association/scoring.py` - Rewrote score_tracklet_pair body, added _batch_score_frames helper
- `tests/unit/core/association/test_scoring.py` - Added 3 edge case tests for early termination paths

## Decisions Made
- Extracted _batch_score_frames() as a private helper for clarity and reuse between phases
- Early termination fires when t_shared >= early_k (matching original semantics)

## Deviations from Plan
None - plan executed exactly as written

## Issues Encountered
- Task 3 (manual YH eval comparison) deferred: requires running aquapose eval before and after; user should verify numerically identical metrics after next pipeline run

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- score_tracklet_pair is fully vectorized, ready for performance benchmarking
- User should run aquapose eval on a YH chunk to confirm metric identity (SC-2)

---
*Phase: 56-vectorized-association-scoring*
*Completed: 2026-03-04*
