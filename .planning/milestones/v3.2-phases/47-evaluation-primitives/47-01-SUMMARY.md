---
phase: 47-evaluation-primitives
plan: "01"
subsystem: evaluation
tags: [dataclass, pure-function, metrics, detection, tracking, tdd]

# Dependency graph
requires: []
provides:
  - "evaluation/stages/ subpackage with __init__.py placeholder"
  - "DetectionMetrics frozen dataclass with total_detections, mean/std_confidence, mean_jitter, per_camera_counts, and to_dict()"
  - "evaluate_detection() pure function — no engine imports"
  - "TrackingMetrics frozen dataclass with track_count, length stats, coast_frequency, detection_coverage, and to_dict()"
  - "evaluate_tracking() pure function — no engine imports"
affects: [47-02, 47-03, 48-evaluation-runner, 49-tuning-orchestrator]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Frozen dataclass + pure function evaluator pattern for each pipeline stage"
    - "to_dict() with explicit Python-native casts (float(), int()) for JSON serialization"
    - "AST-based test assertion for zero engine imports in evaluator modules"
    - "TDD flow: failing import test → implement module → all tests green → lint fix → commit"

key-files:
  created:
    - src/aquapose/evaluation/stages/__init__.py
    - src/aquapose/evaluation/stages/detection.py
    - src/aquapose/evaluation/stages/tracking.py
    - tests/unit/evaluation/test_stage_detection.py
    - tests/unit/evaluation/test_stage_tracking.py
  modified: []

key-decisions:
  - "mean_jitter defined as mean(abs(diff(counts))) per camera averaged over all cameras — stable cameras contribute 0.0, flickering cameras raise the mean"
  - "detection_coverage = 1.0 - coast_frequency (not recomputed independently) since every frame is either detected or coasted"
  - "Zero-length tracklets guarded by checking total_frames before division — length_min=0 is a valid result"
  - "stages/__init__.py left as placeholder; exports will be added in Plan 03 once all evaluators exist"

patterns-established:
  - "Evaluator pattern: frozen dataclass + to_dict() + pure evaluate_*() function with no engine imports"
  - "Jitter computation via _compute_jitter() helper using np.mean(np.abs(np.diff(counts_arr))) per camera"

requirements-completed: [EVAL-01, EVAL-02]

# Metrics
duration: 4min
completed: 2026-03-03
---

# Phase 47 Plan 01: Evaluation Stages Detection and Tracking Summary

**Frozen dataclass evaluators for detection (jitter, confidence, per-camera balance) and tracking (length stats, coast frequency) using TDD with zero engine imports**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-03T18:44:47Z
- **Completed:** 2026-03-03T18:48:37Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created `evaluation/stages/` subpackage establishing the evaluator pattern for all subsequent stage evaluators
- Implemented `DetectionMetrics` with jitter, confidence stats, and per-camera balance via pure `evaluate_detection()` function
- Implemented `TrackingMetrics` with track length distribution, coast frequency, and detection coverage via pure `evaluate_tracking()` function
- 20 unit tests across both evaluators covering empty input, known values, edge cases, to_dict serialization, and AST no-engine-import assertions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create stages/ subpackage with detection evaluator** - `08e3791` (feat)
2. **Task 2: Create tracking evaluator** - `b104d16` (feat)

**Plan metadata:** (see docs commit below)

_Note: TDD tasks committed as single feat commits (test + implementation together)_

## Files Created/Modified
- `src/aquapose/evaluation/stages/__init__.py` - Placeholder package docstring; exports deferred to Plan 03
- `src/aquapose/evaluation/stages/detection.py` - DetectionMetrics frozen dataclass and evaluate_detection() pure function
- `src/aquapose/evaluation/stages/tracking.py` - TrackingMetrics frozen dataclass and evaluate_tracking() pure function
- `tests/unit/evaluation/test_stage_detection.py` - 10 unit tests for detection evaluator
- `tests/unit/evaluation/test_stage_tracking.py` - 10 unit tests for tracking evaluator

## Decisions Made
- `mean_jitter` is defined as mean of per-camera mean(abs(diff(counts))) — this means a stable camera contributes 0.0 to the average, matching the plan's example (stable [5,5,5] + flickering [5,1,5] = jitter 2.0)
- `detection_coverage = 1.0 - coast_frequency` since Tracklet2D.frame_status only contains "detected" or "coasted" — no third state
- Zero-length tracklets (empty `frames` tuple) are included in `track_count` but contribute 0 to frame totals, giving `length_min=0` — guarded div-by-zero on `total_frames`
- `stages/__init__.py` is a placeholder; full public API exports deferred to Plan 03 when all five evaluators are available

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing lint errors in `test_stage_association.py` and `association.py` (out of scope, not caused by this plan's changes) — ignored
- Pre-existing type errors in unrelated modules (yolo backends, midline stage) — not caused by our changes, ignored

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `evaluation/stages/detection.py` and `tracking.py` are ready for use by Plan 02 (midline/association/reconstruction evaluators)
- Pattern is established: frozen dataclass + evaluate_*() function + to_dict() + no engine imports
- Plan 03 can wire all five evaluators into the `stages/__init__.py` public API

---
*Phase: 47-evaluation-primitives*
*Completed: 2026-03-03*

## Self-Check: PASSED
