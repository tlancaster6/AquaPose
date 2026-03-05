---
phase: 59-batched-yolo-inference
plan: 02
subsystem: core
tags: [yolo, batched-inference, detection, oom-retry, gpu]

# Dependency graph
requires:
  - phase: 59-batched-yolo-inference/01
    provides: predict_with_oom_retry utility, BatchState, detection_batch_frames config field
provides:
  - detect_batch() method on YOLOOBBBackend and YOLOBackend
  - Batched DetectionStage.run() with OOM retry
  - Shared parsing helpers (_parse_results, _parse_box_results)
affects: [59-batched-yolo-inference/03, 60-end-to-end-performance-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: [batched-backend-method, shared-result-parsing, predict_with_oom_retry-wrapping]

key-files:
  created: []
  modified:
    - src/aquapose/core/detection/backends/yolo_obb.py
    - src/aquapose/core/detection/backends/yolo.py
    - src/aquapose/core/detection/stage.py
    - src/aquapose/engine/pipeline.py
    - tests/unit/core/detection/test_detection_stage.py
    - tests/unit/segmentation/test_detector.py

key-decisions:
  - "Extract OBB parsing into _parse_results() and box parsing into _parse_box_results() to share between detect() and detect_batch()"
  - "Use r.orig_shape for frame dimensions in batch parsing since frame is not available in shared helper"

patterns-established:
  - "detect_batch(frames) -> list[list[Detection]]: batch detection method pattern for all backends"
  - "predict_with_oom_retry wrapping detect_batch in stage.run(): OOM-resilient batch call pattern"

requirements-completed: [BATCH-01]

# Metrics
duration: 4min
completed: 2026-03-05
---

# Phase 59 Plan 02: Batched Detection Backends Summary

**detect_batch() on both YOLO backends with DetectionStage.run() refactored to single batched call per timestep via predict_with_oom_retry**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-05T02:57:09Z
- **Completed:** 2026-03-05T03:01:31Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Added `detect_batch()` to both `YOLOOBBBackend` and `YOLOBackend` for batched GPU inference
- Refactored `DetectionStage.run()` to collect all camera frames per timestep and call `detect_batch()` once (replacing 12 serial `detect()` calls)
- Wrapped batch calls with `predict_with_oom_retry` for automatic batch-size halving on CUDA OOM
- Extracted shared parsing logic to eliminate code duplication between single and batch paths

## Task Commits

Each task was committed atomically:

1. **Task 1: Add detect_batch() to both detection backends** - `5c40d1e` (feat)
2. **Task 2: Refactor DetectionStage.run() for batched inference** - `c9c1db1` (feat)

## Files Created/Modified
- `src/aquapose/core/detection/backends/yolo_obb.py` - Added `detect_batch()` and `_parse_results()` shared parsing
- `src/aquapose/core/detection/backends/yolo.py` - Added `detect_batch()` on YOLOBackend and `_parse_box_results()` module helper
- `src/aquapose/core/detection/stage.py` - Refactored `run()` for batched inference with OOM retry
- `src/aquapose/engine/pipeline.py` - Wired `detection_batch_frames` config through `build_stages()`
- `tests/unit/core/detection/test_detection_stage.py` - Added batch call and missing camera tests
- `tests/unit/segmentation/test_detector.py` - Fixed mock to include `orig_shape` for refactored parsing

## Decisions Made
- Extracted OBB parsing into `_parse_results()` instance method and box parsing into `_parse_box_results()` module-level function to avoid duplicating parsing logic between `detect()` and `detect_batch()`
- Used `r.orig_shape` (from ultralytics Result object) for frame dimensions in the shared box parser, since the frame array is not available when parsing batch results

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test mock missing orig_shape attribute**
- **Found during:** Task 1 (detect_batch backend addition)
- **Issue:** Refactoring `YOLODetector.detect()` to use shared `_parse_box_results()` broke an existing test in `tests/unit/segmentation/test_detector.py` because the mock Result object lacked `orig_shape`
- **Fix:** Added `mock_result.orig_shape = (480, 640)` to the mock factory
- **Files modified:** `tests/unit/segmentation/test_detector.py`
- **Verification:** All 850 tests pass
- **Committed in:** 5c40d1e (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary fix for test compatibility after refactoring. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Detection backends and stage are fully batched with OOM resilience
- Ready for Plan 03 (batched midline inference) which follows the same pattern
- `predict_with_oom_retry` + `BatchState` pattern proven and ready for reuse

---
*Phase: 59-batched-yolo-inference*
*Completed: 2026-03-05*
