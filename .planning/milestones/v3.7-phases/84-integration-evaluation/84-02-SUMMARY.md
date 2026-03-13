---
phase: 84-integration-evaluation
plan: 02
subsystem: tracking
tags: [keypoint-tracker, ocsort, evaluation, bidi-removal, oc-sort]

# Dependency graph
requires:
  - phase: 84-01
    provides: bidi-merge bug analysis, evaluation script, 84-EVALUATION.md
provides:
  - Single-pass KeypointTracker (no bidi dead code)
  - Updated 84-EVALUATION.md with single-pass metrics
  - Fixed evaluate_custom_tracker.py (OOM + exclusive assignment)
affects: [85-cross-view-association, future-tracker-tuning]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - evaluate_custom_tracker.py uses two-pass architecture (cache detections, then render from video)
    - Exclusive linear_sum_assignment for video annotation track-ID assignment

key-files:
  created: []
  modified:
    - src/aquapose/core/tracking/keypoint_tracker.py
    - src/aquapose/core/tracking/__init__.py
    - scripts/evaluate_custom_tracker.py
    - .planning/phases/84-integration-evaluation/84-EVALUATION.md
    - eval_tracker_output/run_log.txt

key-decisions:
  - "Bidi merge was already stripped before this plan ran; only stale comments remained to clean"
  - "Single-pass keypoint_bidi: 42 tracks, 0.936 coverage (vs bidi 44 tracks, 0.898)"
  - "BYTE trigger no longer fires after bidi removal (coverage 0.936 >= 0.90)"
  - "Root cause of remaining fragmentation: occlusion reacquisition identity breaks, not temporal gaps"

patterns-established:
  - "evaluate_custom_tracker.py two-pass: cache Detection objects, then re-read frames from disk for rendering"

requirements-completed: [INTEG-01]

# Metrics
duration: 25min
completed: 2026-03-11
---

# Phase 84 Plan 02: Strip Bidi Merge & Re-evaluate Summary

**Removed bidirectional merge dead code from KeypointTracker; single-pass re-eval shows 42 tracks at 93.6% coverage (vs buggy bidi 44 tracks at 89.8%), BYTE trigger no longer fires**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-11T09:20:00Z
- **Completed:** 2026-03-11T09:50:00Z
- **Tasks:** 3 complete (Task 4 is checkpoint:human-verify)
- **Files modified:** 4

## Accomplishments

- Cleaned up stale bidi comments from `keypoint_tracker.py` and `__init__.py` (bidi merge was already stripped in Phase 83)
- Fixed `evaluate_custom_tracker.py`: removed OOM-causing `all_raw_frames` list; re-reads frames from disk in render pass
- Fixed renderer: greedy nearest-neighbor → exclusive `linear_sum_assignment` to eliminate duplicate track IDs in video
- Re-ran full evaluation (1200 frames, GPU): single-pass 42 tracks at 93.6% coverage vs prior bidi 44 tracks at 89.8%
- Updated 84-EVALUATION.md with new single-pass column and updated interpretation

## Task Commits

1. **Tasks 1+2: Clean up stale bidi comments, verify tests** - `f66d0d8` (refactor)
2. **Task 3a: Fix evaluate_custom_tracker.py** - `43872d2` (fix)
3. **Task 3b: Re-run evaluation, update metrics document** - `8219c1f` (feat)

## Files Created/Modified

- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/tracking/keypoint_tracker.py` - Removed stale bidi section header and docstring references
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/tracking/__init__.py` - Updated docstring noting bidi removal
- `/home/tlancaster6/Projects/AquaPose/scripts/evaluate_custom_tracker.py` - OOM fix (no all_raw_frames), exclusive assignment renderer
- `/home/tlancaster6/Projects/AquaPose/.planning/phases/84-integration-evaluation/84-EVALUATION.md` - New single-pass metrics column, updated interpretation

## Decisions Made

- Bidi merge was already fully stripped before this plan ran (done during Phase 83 implementation); this plan cleaned up the remaining stale comments
- Single-pass tracker is strictly better than bidi on all metrics: fewer tracks (42 vs 44), higher coverage (93.6% vs 89.8%), lower coast frequency (0.064 vs 0.102)
- Root cause of remaining over-fragmentation (42 vs target 9) is occlusion reacquisition identity breaks, not temporal gaps — same as OC-SORT (30 tracks)
- BYTE-style secondary pass (TRACK-10) no longer triggered after bidi removal

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Lint error: zip() missing strict= parameter**
- **Found during:** Task 3 (evaluaton script commit)
- **Issue:** Ruff B905: `zip(row_idx, col_idx)` missing `strict=` parameter
- **Fix:** Added `strict=False` to the zip call
- **Files modified:** scripts/evaluate_custom_tracker.py
- **Verification:** Pre-commit hooks passed
- **Committed in:** 43872d2

---

**Total deviations:** 1 auto-fixed (1 blocking lint error)
**Impact on plan:** Minor lint fix required by pre-commit hooks. No scope creep.

## Issues Encountered

- The bidi merge was already stripped before this plan ran (by the Phase 83 implementation), so Tasks 1 and 2 only needed comment cleanup rather than code removal. The evaluation script changes (OOM fix + exclusive assignment) were the main substantive work.

## Next Phase Readiness

- KeypointTracker is a clean single-pass tracker — ready for Phase 85 cross-view association work
- 84-EVALUATION.md has accurate single-pass numbers for comparison
- Task 4 (human-verify checkpoint) pending: user review of updated evaluation document and annotated video

---
*Phase: 84-integration-evaluation*
*Completed: 2026-03-11*
