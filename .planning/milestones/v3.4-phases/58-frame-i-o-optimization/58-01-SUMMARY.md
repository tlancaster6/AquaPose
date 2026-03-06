---
phase: 58-frame-i-o-optimization
plan: 01
subsystem: io
tags: [threading, prefetch, queue, cv2, frame-io]

# Dependency graph
requires: []
provides:
  - "ChunkFrameSource with background prefetch via daemon thread + bounded queue"
  - "DetectionStage missing-camera guard"
affects: [59-batched-model-inference, 60-end-to-end-performance-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: [background-thread-prefetch, sentinel-queue-termination, stop-event-coordination]

key-files:
  created: []
  modified:
    - src/aquapose/core/types/frame_source.py
    - src/aquapose/core/detection/stage.py
    - tests/unit/core/types/test_frame_source.py
    - tests/unit/engine/test_chunk_handoff.py

key-decisions:
  - "Queue maxsize=2 balances memory (2 frames x 12 cameras) vs prefetch benefit"
  - "Undistortion runs in background thread so main thread receives ready-to-use frames"
  - "Decode failure skips camera (warning) rather than killing the frame or raising"

patterns-established:
  - "Prefetch pattern: daemon thread + bounded queue + sentinel + stop_event for cooperative shutdown"
  - "Missing-camera guard: stages check cam_id presence before accessing frames dict"

requirements-completed: [FIO-01, FIO-02]

# Metrics
duration: 6min
completed: 2026-03-05
---

# Phase 58 Plan 01: Frame I/O Optimization Summary

**Background-thread prefetch in ChunkFrameSource with sequential cap.read() and bounded queue, plus missing-camera guard in DetectionStage**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-05T02:36:29Z
- **Completed:** 2026-03-05T02:42:46Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Replaced seek-based ChunkFrameSource.__iter__ with background daemon thread using queue.Queue(maxsize=2)
- Sequential cap.read() in worker thread eliminates per-frame seeking overhead (~12% of pipeline time)
- Undistortion happens in background thread, main thread receives ready-to-use frames
- Single-camera decode failure logs warning and skips camera without killing the frame
- DetectionStage handles missing cameras gracefully with empty detection list
- 6 new unit tests covering prefetch iteration, cleanup, concurrency guard, missing camera handling, exception propagation, and protocol conformance

## Task Commits

Each task was committed atomically:

1. **Task 1: Add prefetch tests for ChunkFrameSource** - `acbecd3` (test, TDD RED)
2. **Task 2: Implement background prefetch and fix DetectionStage** - `95eeb2f` (feat, TDD GREEN)

## Files Created/Modified
- `src/aquapose/core/types/frame_source.py` - ChunkFrameSource with _prefetch_worker, _ensure_captures_positioned, queue-based __iter__, cleanup __exit__
- `src/aquapose/core/detection/stage.py` - Missing-camera guard in detection loop
- `tests/unit/core/types/test_frame_source.py` - 6 new prefetch tests
- `tests/unit/engine/test_chunk_handoff.py` - Updated _MockVideoFrameSource with _captures and _undist_maps for prefetch compatibility

## Decisions Made
- Queue maxsize=2: balances memory (2 full multi-camera frames buffered) vs prefetch lookahead
- Undistortion in background thread: main thread receives ready-to-use frames, maximizing GPU overlap
- Decode failure = skip camera: a single camera dropout should not kill the entire frame

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated _MockVideoFrameSource in test_chunk_handoff.py**
- **Found during:** Task 2 (implementation)
- **Issue:** Existing _MockVideoFrameSource lacked _captures and _undist_maps attributes, causing test_chunk_frame_source_iteration to fail with AttributeError
- **Fix:** Added _MockCapture class and populated _captures/_undist_maps on the mock
- **Files modified:** tests/unit/engine/test_chunk_handoff.py
- **Verification:** All 8 chunk handoff tests pass
- **Committed in:** 95eeb2f (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Mock update was necessary for prefetch compatibility. No scope creep.

## Issues Encountered
- Pre-commit hook caught PT012 (multi-statement pytest.raises block) -- extracted helper function _start_concurrent_iter to satisfy single-statement rule

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ChunkFrameSource prefetch is production-ready for pipeline use
- DetectionStage hardened against missing cameras
- Ready for Phase 59 (batched model inference) which will further improve GPU utilization

---
*Phase: 58-frame-i-o-optimization*
*Completed: 2026-03-05*
