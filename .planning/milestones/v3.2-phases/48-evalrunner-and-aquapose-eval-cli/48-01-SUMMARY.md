---
phase: 48-evalrunner-and-aquapose-eval-cli
plan: 01
subsystem: evaluation
tags: [eval-runner, stage-cache, pickle, evaluation, orchestration]

# Dependency graph
requires:
  - phase: 47-evaluation-primitives
    provides: "Five stage evaluators (evaluate_detection, evaluate_tracking, evaluate_association, evaluate_midline, evaluate_reconstruction) and their metric dataclasses"
  - phase: 46-diagnostic-observer-and-cache-infrastructure
    provides: "DiagnosticObserver cache envelope format (dict with run_id, timestamp, stage_name, version_fingerprint, context), load_stage_cache, StaleCacheError"
provides:
  - "EvalRunner class: discovers per-stage pickle caches, loads them, calls stage evaluators, returns EvalRunnerResult"
  - "EvalRunnerResult frozen dataclass: optional per-stage metrics, run_id, stages_present, frames_evaluated, frames_available, to_dict()"
  - "evaluation/__init__.py: EvalRunner and EvalRunnerResult exported"
affects: [48-02, 48-03, phase-49-tuning-orchestrator]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Inline engine imports in runner.py to avoid top-level engine coupling"
    - "TDD Red-Green cycle with synthetic pickle cache fixtures in tests"
    - "MidlineSet derivation from tracklet_groups + annotated_detections using centroid proximity matching (mirrors DiagnosticObserver logic)"

key-files:
  created:
    - src/aquapose/evaluation/runner.py
    - tests/unit/evaluation/test_runner.py
  modified:
    - src/aquapose/evaluation/__init__.py

key-decisions:
  - "MidlineSet construction for evaluate_association uses tracklet_groups + annotated_detections centroid proximity matching (same as DiagnosticObserver.export_midline_fixtures)"
  - "evaluate_midline receives list[dict[int, Midline2D]] by taking first camera's Midline2D per fish per frame from MidlineSet (single-camera representative midline)"
  - "EvalRunner._read_n_animals() uses inline import of load_config to avoid top-level engine coupling"
  - "No top-level from aquapose.engine imports in runner.py — all engine imports are inline"
  - "StaleCacheError propagates upward; only missing cache files are silently skipped (FileNotFoundError from cache probe)"

patterns-established:
  - "Synthetic pickle cache fixtures: build PipelineContext objects, wrap in envelope dict, pickle.dumps to tmp_path/diagnostics/<stage>_cache.pkl"
  - "EvalRunner returns empty EvalRunnerResult (all None) when no caches present — no exception raised"

requirements-completed: [EVAL-06, EVAL-07]

# Metrics
duration: 7min
completed: 2026-03-03
---

# Phase 48 Plan 01: EvalRunner and EvalRunnerResult Summary

**EvalRunner orchestration class that discovers per-stage pickle caches, calls Phase 47 evaluators with unpacked PipelineContext data, and aggregates results into a frozen EvalRunnerResult with to_dict() serialization**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-03T19:10:54Z
- **Completed:** 2026-03-03T19:18:00Z
- **Tasks:** 1 (TDD: RED + GREEN + REFACTOR)
- **Files modified:** 3

## Accomplishments

- EvalRunner class discovers 5 stage caches, silently skips missing ones, calls correct evaluators
- EvalRunnerResult frozen dataclass with optional per-stage metrics and JSON-serializable to_dict()
- MidlineSet construction from tracklet_groups + annotated_detections via centroid proximity matching
- n_frames sampling via select_frames() with proper frames_evaluated tracking
- n_animals read from config.yaml (inline import, no top-level engine coupling)
- 10 comprehensive unit tests with synthetic pickle cache fixtures — all pass without GPU or real pipeline

## Task Commits

Each task was committed atomically (TDD pattern):

1. **Task 1 RED: Add failing tests** - `e547e9b` (test)
2. **Task 1 GREEN: Implement EvalRunner** - `bbae5a6` (feat)

_TDD tasks have RED + GREEN commits; no REFACTOR needed._

## Files Created/Modified

- `src/aquapose/evaluation/runner.py` - EvalRunner class and EvalRunnerResult dataclass
- `tests/unit/evaluation/test_runner.py` - 10 unit tests with synthetic cache fixtures
- `src/aquapose/evaluation/__init__.py` - Added EvalRunner and EvalRunnerResult exports

## Decisions Made

- MidlineSet construction mirrors DiagnosticObserver.export_midline_fixtures (centroid proximity matching from tracklet centroids to AnnotatedDetection bboxes)
- evaluate_midline receives `list[dict[int, Midline2D]]` by taking first camera's Midline2D per fish per frame from MidlineSet — covers the majority of fish-frame pairs without needing arbitrary camera selection logic
- No top-level `from aquapose.engine` imports in runner.py — config loaded inline in `_read_n_animals()`
- StaleCacheError propagates upward; FileNotFoundError (missing cache files) is the only silently skipped case

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Detection constructor call in tests**
- **Found during:** Task 1 GREEN (running tests)
- **Issue:** Tests used `Detection(bbox=..., confidence=..., area=..., camera_id=..., frame_index=...)` but Detection does not have `camera_id` or `frame_index` fields
- **Fix:** Updated `_make_detection()` in tests to use correct signature `Detection(bbox=..., mask=None, area=..., confidence=...)`
- **Files modified:** tests/unit/evaluation/test_runner.py
- **Committed in:** bbae5a6

**2. [Rule 1 - Bug] Fixed Tracklet2D constructor call in tests**
- **Found during:** Task 1 GREEN (running tests)
- **Issue:** Tests used `Tracklet2D(track_id=..., camera_id=..., frames=..., centroids=..., frame_status=...)` but Tracklet2D also requires `bboxes` field
- **Fix:** Added `bboxes` field to `_make_tracklet2d()` helper
- **Files modified:** tests/unit/evaluation/test_runner.py
- **Committed in:** bbae5a6

**3. [Rule 1 - Bug] Removed unused variable `sampled_set` in runner.py**
- **Found during:** Task 1 GREEN (ruff lint check)
- **Issue:** `sampled_set = set(sampled_indices)` was computed but never used
- **Fix:** Removed the variable assignment
- **Files modified:** src/aquapose/evaluation/runner.py
- **Committed in:** bbae5a6

---

**Total deviations:** 3 auto-fixed (all Rule 1 - bugs discovered during test execution and lint)
**Impact on plan:** All fixes necessary for correctness. No scope creep.

## Issues Encountered

- hatch CLI was broken in the bash environment (VIRTUAL_ENV env var missing). Used the venv Python binary directly (`~/.local/share/hatch/env/virtual/aquapose/U-e77gu2/aquapose/bin/python`) to run tests. Lint and typecheck still used the venv's ruff and basedpyright binaries.

## Next Phase Readiness

- EvalRunner is ready for 48-02 (aquapose eval CLI integration)
- EvalRunner is ready for Phase 49 (TuningOrchestrator) which will use it for per-stage sweeping
- One pending concern: the centroid matching tolerance (5px) is hardcoded — may need tuning if real pipeline data shows poor association recall

---
*Phase: 48-evalrunner-and-aquapose-eval-cli*
*Completed: 2026-03-03*
