---
phase: 46-engine-primitives
plan: "02"
subsystem: engine
tags: [pickle, cache, pipeline, stage-skip, diagnostic-observer, carry-forward]

# Dependency graph
requires:
  - phase: 46-engine-primitives/01
    provides: StaleCacheError, load_stage_cache, context_fingerprint, carry_forward on PipelineContext

provides:
  - DiagnosticObserver writes per-stage pickle caches on StageComplete when output_dir is set
  - PosePipeline.run() accepts initial_context to skip already-populated stages
  - Skipped stages emit StageComplete(summary={'skipped': True}, elapsed_seconds=0.0)
  - CarryForward extracted from initial_context at pipeline start
  - CarryForward injected into context.carry_forward after TrackingStage runs
  - Unit tests for all new behavior (9 tests)

affects:
  - 47-evaluation-primitives
  - 48-context-loader
  - 49-tuning-orchestrator

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stage-skip via class name lookup in _STAGE_OUTPUT_FIELDS dict"
    - "Cache envelope: {run_id, timestamp, stage_name, version_fingerprint, context}"
    - "run_id captured from PipelineStart event into DiagnosticObserver._run_id (no __init__ change)"

key-files:
  created:
    - tests/unit/engine/test_stage_cache_write.py
    - tests/unit/engine/test_stage_skip.py
  modified:
    - src/aquapose/engine/diagnostic_observer.py
    - src/aquapose/engine/pipeline.py

key-decisions:
  - "Stage name lookup uses type(stage).__name__ matched against _STAGE_OUTPUT_FIELDS dict — pure string matching, no isinstance"
  - "Skip detection fires only when initial_context is provided and ALL output fields are non-None"
  - "DiagnosticObserver captures run_id from PipelineStart event (not __init__) to keep signature stable"
  - "stage_key derived by removesuffix('Stage').lower() — DetectionStage -> detection_cache.pkl"

patterns-established:
  - "Cache envelope format: dict with run_id, timestamp, stage_name, version_fingerprint, context keys"
  - "Test stubs for stage-skip tests use class names matching _STAGE_OUTPUT_FIELDS keys for correct skip detection"

requirements-completed:
  - INFRA-01
  - INFRA-02

# Metrics
duration: 6min
completed: 2026-03-03
---

# Phase 46 Plan 02: Per-Stage Pickle Cache Writing and Stage-Skip Logic Summary

**DiagnosticObserver writes per-stage pickle envelopes on StageComplete; PosePipeline.run() auto-skips stages whose output fields are pre-populated in initial_context**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-03T18:25:19Z
- **Completed:** 2026-03-03T18:31:00Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- DiagnosticObserver captures run_id from PipelineStart and writes `diagnostics/<stage>_cache.pkl` (envelope format) after each StageComplete when output_dir is set
- PosePipeline.run() accepts `initial_context: PipelineContext | None = None`; stages with all output fields populated are auto-skipped with proper events emitted
- TrackingStage carry_forward is injected into context.carry_forward after the stage runs; initial carry is extracted from initial_context.carry_forward at pipeline start
- 9 unit tests covering cache write, envelope contents, round-trip loading, run_id capture, skip detection, skip events, no-skip without initial_context, carry extraction, and carry injection

## Task Commits

All tasks completed in a single atomic commit (tasks executed together):

1. **Task 46.2.1: Add pickle cache writing to DiagnosticObserver** - `3d732b6` (feat)
2. **Task 46.2.2: Add initial_context and stage-skip logic to PosePipeline.run()** - `3d732b6` (feat)
3. **Task 46.2.3: Write unit tests for cache writing and stage-skip logic** - `3d732b6` (feat)

## Files Created/Modified

- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/diagnostic_observer.py` - Added `_run_id`, PipelineStart handling, `_write_stage_cache()` method
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/pipeline.py` - Added `_STAGE_OUTPUT_FIELDS` constant, `initial_context` param, skip detection loop, carry injection after TrackingStage
- `/home/tlancaster6/Projects/AquaPose/tests/unit/engine/test_stage_cache_write.py` - 4 tests for DiagnosticObserver cache writing
- `/home/tlancaster6/Projects/AquaPose/tests/unit/engine/test_stage_skip.py` - 5 tests for PosePipeline stage-skip logic

## Decisions Made

- Stage class name lookup via `type(stage).__name__` matched to `_STAGE_OUTPUT_FIELDS` dict (no isinstance check beyond TrackingStage special-casing)
- Skip detection requires ALL output fields to be non-None (not any) — conservative: only skip when truly complete
- Test stubs use class names matching `_STAGE_OUTPUT_FIELDS` keys (e.g., class `DetectionStage`) to ensure skip logic fires correctly in tests without importing GPU-dependent real stages
- `stage_key = stage_name.removesuffix("Stage").lower()` normalizes filename: `DetectionStage` -> `detection_cache.pkl`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Initial test stubs used `StubDetection` class name (not `DetectionStage`), causing skip logic to not fire. Fixed by naming stub classes to match `_STAGE_OUTPUT_FIELDS` keys.
- Unused `pytest` import in test_stage_skip.py flagged by ruff. Removed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- DiagnosticObserver now writes per-stage pickle caches ready for consumption by phase 47 (evaluation primitives) and phase 48 (ContextLoader)
- PosePipeline stage-skip is the foundation for phase 49 TuningOrchestrator selective re-execution
- All 703 unit tests pass, no regressions

---
*Phase: 46-engine-primitives*
*Completed: 2026-03-03*
