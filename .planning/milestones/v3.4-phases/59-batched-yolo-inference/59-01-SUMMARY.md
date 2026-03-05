---
phase: 59-batched-yolo-inference
plan: 01
subsystem: core
tags: [cuda, oom, batch-inference, gpu, retry]

# Dependency graph
requires: []
provides:
  - "BatchState dataclass for adaptive batch size tracking"
  - "predict_with_oom_retry utility for chunked GPU inference with OOM recovery"
  - "DetectionConfig.detection_batch_frames config field (default 0)"
  - "MidlineConfig.midline_batch_crops config field (default 0)"
affects: [59-02-PLAN, 59-03-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns: ["OOM retry with batch halving for GPU inference"]

key-files:
  created:
    - src/aquapose/core/inference.py
    - tests/unit/core/test_inference.py
  modified:
    - src/aquapose/core/__init__.py
    - src/aquapose/engine/config.py

key-decisions:
  - "Mutable BatchState (not frozen) to persist batch size reductions across calls"
  - "batch_size=0 means no limit (send all inputs in one call) for both config fields"

patterns-established:
  - "OOM retry pattern: catch CUDA OOM, halve batch, retry from scratch, persist in BatchState"

requirements-completed: [BATCH-03, BATCH-04]

# Metrics
duration: 2min
completed: 2026-03-05
---

# Phase 59 Plan 01: OOM Retry Utility Summary

**BatchState + predict_with_oom_retry for chunked GPU inference with automatic OOM halving, plus batch-size config fields on DetectionConfig and MidlineConfig**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-05T02:52:46Z
- **Completed:** 2026-03-05T02:54:37Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created `predict_with_oom_retry` utility that chunks inputs, catches CUDA OOM, halves batch size, and retries from scratch
- Added `BatchState` dataclass that persists adaptive batch size across calls for sticky OOM recovery
- Added `detection_batch_frames` and `midline_batch_crops` config fields with default 0 (no limit)
- 9 unit tests covering normal batching, OOM halving, state persistence, error propagation, and edge cases

## Task Commits

Each task was committed atomically:

1. **Task 1: Create OOM retry utility with tests** - `b11300d` (feat)
2. **Task 2: Add batch size config fields** - `501f632` (feat)

## Files Created/Modified
- `src/aquapose/core/inference.py` - BatchState dataclass and predict_with_oom_retry function
- `tests/unit/core/test_inference.py` - 9 unit tests for OOM retry logic
- `src/aquapose/core/__init__.py` - Export BatchState and predict_with_oom_retry
- `src/aquapose/engine/config.py` - Added detection_batch_frames and midline_batch_crops fields

## Decisions Made
- Used mutable dataclass for BatchState (not frozen) since it must track state mutations across calls
- batch_size=0 semantics: send all inputs in a single call (consistent with "no limit" meaning)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- BatchState and predict_with_oom_retry ready for integration into DetectionStage (59-02) and MidlineStage (59-03)
- Config fields ready for YAML and CLI override usage

---
*Phase: 59-batched-yolo-inference*
*Completed: 2026-03-05*
