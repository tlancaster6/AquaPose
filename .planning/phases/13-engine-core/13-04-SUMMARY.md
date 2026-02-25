---
phase: 13-engine-core
plan: "04"
subsystem: engine
tags: [pipeline, orchestrator, eventbus, lifecycle-events, config-artifact, structural-typing]

# Dependency graph
requires:
  - phase: 13-engine-core plan 01
    provides: Stage Protocol, PipelineContext dataclass
  - phase: 13-engine-core plan 02
    provides: PipelineConfig frozen dataclass, load_config(), serialize_config()
  - phase: 13-engine-core plan 03
    provides: Event dataclasses (6 types), Observer Protocol, EventBus with MRO-aware dispatch
provides:
  - PosePipeline orchestrator class — single canonical entrypoint (run())
  - Config YAML written as first artifact before any stage executes (ENG-08)
  - Lifecycle event emission: PipelineStart, StageStart, StageComplete, PipelineComplete, PipelineFailed
  - Per-stage wall-clock timing recorded in context.stage_timing
  - add_observer() / remove_observer() convenience API
  - 8 unit tests covering orchestration, events, config artifact, timing, error handling, observer-less mode
affects:
  - 13-05 and all subsequent engine plans that run the pipeline
  - Phase 15 stage migrations (plug real stages into PosePipeline)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PosePipeline owns EventBus — observers subscribe to Event base type to receive all events"
    - "Config artifact written before stage loop — guarantees reproducibility even on stage failure"
    - "Stage timing keyed by type(stage).__name__ — class name as natural identifier"
    - "try/except wraps entire stage loop — PipelineFailed always emitted before exception re-raises"

key-files:
  created:
    - src/aquapose/engine/pipeline.py
    - tests/unit/engine/test_pipeline.py
  modified:
    - src/aquapose/engine/__init__.py

key-decisions:
  - "PosePipeline writes config.yaml before emitting PipelineStart — config artifact is truly first, not just first-stage-first"
  - "Stage timing keyed by class name (type(stage).__name__) — simpler than instance name, consistent with Stage Protocol's class-oriented identity"
  - "Observers subscribe to Event base type in constructor — simplest API for observers that want all events"

patterns-established:
  - "Config-first artifact pattern: serialize_config() written to disk before any stage runs (ENG-08)"
  - "Fault isolation: PipelineFailed emitted then exception re-raised — observers always notified regardless of outcome"

requirements-completed: [ENG-06, ENG-08]

# Metrics
duration: 4min
completed: "2026-02-25"
---

# Phase 13 Plan 04: PosePipeline Orchestrator Summary

**PosePipeline orchestrator with config-first artifact guarantee, synchronous lifecycle event emission via EventBus, and per-stage timing — completing the engine core skeleton**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-25T21:34:00Z
- **Completed:** 2026-02-25T21:38:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Implemented PosePipeline with `run()` as the single canonical entrypoint — creates output dir, writes `config.yaml` first, emits lifecycle events, executes stages in order, records timing
- Config artifact (ENG-08) is written before PipelineStart is emitted, guaranteeing reproducibility even if the first stage fails
- PipelineFailed event always emitted before exception re-raises, so observers (timing, logging) receive complete run telemetry
- Wrote 8 unit tests (all pass) covering: stage order, lifecycle events, config artifact existence, pre-stage config guarantee, error/PipelineFailed, timing, observer-less mode, cross-stage context flow

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement PosePipeline orchestrator skeleton** - `9ffe5fe` (feat)
2. **Task 2: Write tests for pipeline orchestration, event emission, and config artifact** - `9736226` (test)

## Files Created/Modified

- `src/aquapose/engine/pipeline.py` - PosePipeline class with run(), add_observer(), remove_observer(); 160 lines
- `src/aquapose/engine/__init__.py` - Updated to export PosePipeline (added import + __all__ entry)
- `tests/unit/engine/test_pipeline.py` - 8 unit tests using MockStage, FailingStage, ConfigCheckStage, RecordingObserver helpers; 207 lines

## Decisions Made

- Config artifact written to disk before `PipelineStart` event emitted — ensures the config is on disk even if an observer crashes on PipelineStart; stronger guarantee than "before first stage"
- Stage timing uses `type(stage).__name__` as key — aligns with Stage Protocol's class-oriented identity model; the `_order` list (written by MockStage in tests) is a test-only artifact
- Observers passed to constructor subscribe to `Event` base type — receive all events without needing to enumerate event types; `add_observer()` / `remove_observer()` allow post-construction adjustment

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_pipeline_records_stage_timing to use distinct stage classes**
- **Found during:** Task 2 (test writing)
- **Issue:** Test expected timing keys "Alpha" and "Beta" (instance names), but pipeline keys by `type(stage).__name__`. Two `MockStage("Alpha")` and `MockStage("Beta")` both have class name "MockStage" — second overwrites first.
- **Fix:** Replaced with inline `FirstStage` and `SecondStage` classes so timing keys are "FirstStage" and "SecondStage"
- **Files modified:** tests/unit/engine/test_pipeline.py
- **Verification:** All 8 tests pass
- **Committed in:** 9736226 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - test correctness bug)
**Impact on plan:** Necessary fix — test was testing incorrect behavior. The pipeline's use of class names for timing is correct. No scope creep.

## Issues Encountered

- Pre-commit ruff fixed a minor unused import in test file (auto-resolved on re-stage, standard pattern from prior plans)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Engine core is complete: Stage Protocol, PipelineContext, Config hierarchy, Event system, Observer protocol, EventBus, PosePipeline orchestrator
- Phase 13 is fully done — all 4 plans complete, 25 engine tests passing
- Phase 15 (stage migrations) can plug real computation stages into PosePipeline with no engine changes needed
- No blockers

## Self-Check: PASSED

- FOUND: src/aquapose/engine/pipeline.py
- FOUND: tests/unit/engine/test_pipeline.py
- FOUND: .planning/phases/13-engine-core/13-04-SUMMARY.md
- FOUND commit: 9ffe5fe (PosePipeline implementation)
- FOUND commit: 9736226 (8 pipeline tests)
- All 456 tests pass including 8 new pipeline tests

---
*Phase: 13-engine-core*
*Completed: 2026-02-25*
