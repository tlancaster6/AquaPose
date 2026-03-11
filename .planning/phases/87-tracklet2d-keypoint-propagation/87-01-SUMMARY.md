---
phase: 87-tracklet2d-keypoint-propagation
plan: 01
subsystem: tracking
tags: [tracklet2d, keypoints, numpy, dataclass]

requires:
  - phase: 85-code-quality-audit-cli-smoke-test
    provides: Clean keypoint_tracker.py with _KptTrackletBuilder accumulating keypoints
provides:
  - Tracklet2D with keypoints (T, K, 2) and keypoint_conf (T, K) fields
  - to_tracklet2d() stacks builder keypoint lists into numpy arrays
  - 4 round-trip tests covering shape correctness and coasted confidence semantics
affects: [88-multi-keypoint-association-scoring, association]

tech-stack:
  added: []
  patterns: [optional-numpy-fields-with-none-default]

key-files:
  created:
    - tests/unit/core/tracking/test_keypoint_tracker.py (TestTracklet2DKeypointRoundtrip class)
  modified:
    - src/aquapose/core/tracking/types.py
    - src/aquapose/core/tracking/keypoint_tracker.py

key-decisions:
  - "None defaults for backward compat — existing Tracklet2D constructors (e.g. clustering.py _merge_fragments) unaffected"
  - "np.stack over np.array — clearer error on shape mismatch"

patterns-established:
  - "Optional numpy array fields on frozen dataclasses use None default, not empty array"

requirements-completed: [DATA-01, DATA-02]

duration: 5min
completed: 2026-03-11
---

# Phase 87, Plan 01: Tracklet2D Keypoint Propagation Summary

**Tracklet2D now carries per-frame keypoint positions (T,K,2) and confidences (T,K) from tracker to association stage**

## Performance

- **Duration:** ~5 min
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `keypoints` and `keypoint_conf` optional fields to `Tracklet2D` dataclass with None defaults
- Wired `_KptTrackletBuilder.to_tracklet2d()` to stack accumulated keypoint lists via `np.stack()`
- Added 4 round-trip tests in `TestTracklet2DKeypointRoundtrip` covering shape, dtype, coasted zeros, and detected nonzero confidence

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend Tracklet2D and wire to_tracklet2d()** - `1ae1708` (feat)
2. **Task 2: Round-trip keypoint tests** - `535ced3` (test)

## Files Created/Modified
- `src/aquapose/core/tracking/types.py` - Added keypoints and keypoint_conf fields to Tracklet2D
- `src/aquapose/core/tracking/keypoint_tracker.py` - Updated to_tracklet2d() to stack keypoint arrays
- `tests/unit/core/tracking/test_keypoint_tracker.py` - Added TestTracklet2DKeypointRoundtrip test class

## Decisions Made
None - followed plan as specified.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Tracklet2D now carries keypoint data, ready for Phase 88 (multi-keypoint association scoring)
- Both fields default to None so existing code paths are unaffected

---
*Phase: 87-tracklet2d-keypoint-propagation*
*Completed: 2026-03-11*
