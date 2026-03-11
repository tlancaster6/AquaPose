---
phase: 86-cleanup-conditional
plan: 01
subsystem: tracking
tags: [keypoint-tracker, serialization, dead-code]

requires:
  - phase: 85-code-quality-audit-cli-smoke-test
    provides: audit findings identifying handoff bug and dead code
provides:
  - Bug-free cross-chunk handoff serialization (no duplicate frames)
  - Dead OC-SORT types removed (FishTrack, TrackState, TrackHealth)
  - Dead _reproject_3d_midline function removed
affects: []

tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - src/aquapose/core/tracking/keypoint_tracker.py
    - src/aquapose/core/tracking/types.py
    - src/aquapose/core/tracking/__init__.py
    - src/aquapose/evaluation/viz/overlay.py
    - src/aquapose/synthetic/stubs.py
    - tests/unit/core/tracking/test_keypoint_tracker.py

key-decisions:
  - "Empty builders created on restore rather than skipping builder creation entirely — ensures new detections in next chunk are accumulated"

patterns-established: []

requirements-completed: []

duration: 5min
completed: 2026-03-11
---

# Plan 86-01 Summary

**Cross-chunk handoff bug fixed (builders stripped from serialization) and 305 lines of dead OC-SORT code removed**

## Performance

- **Duration:** 5 min
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Fixed cross-chunk handoff: get_state() no longer serializes builder frame history, preventing duplicate frame indices across chunks
- from_state() creates fresh empty builders for active tracks so new detections accumulate correctly
- Removed FishTrack, TrackState, TrackHealth classes (legacy v1.0 types with no consumers)
- Removed unused _reproject_3d_midline from overlay.py
- Added 4 regression tests for handoff correctness

## Task Commits

1. **Task 1: Fix cross-chunk handoff** - `d23e5aa` (fix + test, TDD)
2. **Task 2: Remove dead code** - `522e3dd` (refactor)

## Files Created/Modified
- `src/aquapose/core/tracking/keypoint_tracker.py` - Stripped builders from get_state/from_state
- `src/aquapose/core/tracking/types.py` - Removed FishTrack, TrackState, TrackHealth (305 lines)
- `src/aquapose/core/tracking/__init__.py` - Updated exports
- `src/aquapose/evaluation/viz/overlay.py` - Removed _reproject_3d_midline
- `src/aquapose/synthetic/stubs.py` - Updated docstring reference
- `tests/unit/core/tracking/test_keypoint_tracker.py` - Added TestCrossChunkHandoff class

## Decisions Made
- Empty builders created on restore (not skipped entirely) to ensure new detections in next chunk are accumulated correctly

## Deviations from Plan
None - plan executed exactly as written

## Issues Encountered
None

## Next Phase Readiness
- Tracking module is clean: only Tracklet2D remains as the domain type
- Cross-chunk handoff is regression-tested

---
*Phase: 86-cleanup-conditional*
*Completed: 2026-03-11*
