---
phase: 13-engine-core
plan: 01
subsystem: engine
tags: [typing.Protocol, dataclass, structural-typing, pipeline, engine]

# Dependency graph
requires: []
provides:
  - Stage runtime-checkable Protocol with structural typing (no inheritance required)
  - PipelineContext typed dataclass with Optional fields for all 5 pipeline stages
  - PipelineContext.get() convenience method with clear error on missing upstream stage
  - engine/ package with strict import boundary (stdlib-only, no computation modules)
  - 7 unit tests covering protocol conformance, accumulation, and import boundary
affects:
  - 13-02-config
  - 13-03-orchestrator
  - 13-04-stage-migrations
  - all subsequent engine plans

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stage as typing.Protocol with @runtime_checkable for structural isinstance() checks"
    - "PipelineContext as @dataclass with None defaults and .get() guard method"
    - "Import boundary: engine/ uses only stdlib types — no domain types at runtime or TYPE_CHECKING"

key-files:
  created:
    - src/aquapose/engine/__init__.py
    - src/aquapose/engine/stages.py
    - tests/unit/engine/__init__.py
    - tests/unit/engine/test_stages.py
  modified: []

key-decisions:
  - "Stage Protocol uses @runtime_checkable so isinstance(obj, Stage) works for pipeline validation"
  - "PipelineContext fields use generic list/dict stdlib types only to enforce ENG-07 import boundary"
  - "get() raises ValueError (not AttributeError or KeyError) to distinguish missing-stage from missing-field errors"

patterns-established:
  - "Structural typing: stages conform to Stage protocol with no inheritance — duck typing enforced via Protocol"
  - "Import boundary: engine package may not import from computation modules even under TYPE_CHECKING"

requirements-completed: [ENG-01, ENG-02, ENG-07]

# Metrics
duration: 6min
completed: 2026-02-25
---

# Phase 13 Plan 01: Engine Stage Protocol and PipelineContext Summary

**runtime_checkable Stage Protocol and typed PipelineContext dataclass establishing the engine package contract with strict stdlib-only import boundary (ENG-07)**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-25T21:24:27Z
- **Completed:** 2026-02-25T21:30:35Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Stage Protocol defined via `typing.Protocol` with `@runtime_checkable` — any class with `run(context) -> PipelineContext` is automatically a Stage without inheritance
- PipelineContext dataclass with 7 Optional fields (frame_count, camera_ids, detections, masks, tracks, midline_sets, midlines_3d) and a `stage_timing` dict
- PipelineContext.get() raises ValueError with a clear "stage hasn't run yet" message when a field is None
- Import boundary strictly enforced: stages.py uses only `dataclasses` and `typing` from stdlib — no domain types
- 7 unit tests all passing: structural typing, non-conformance rejection, accumulation, defaults, get() error, get() value, import boundary

## Task Commits

Each task was committed atomically:

1. **Task 1: Create engine package with Stage Protocol and PipelineContext** - `5047a49` (feat)
2. **Task 2: Write tests for Stage protocol conformance and PipelineContext behavior** - `3137e91` (test)

## Files Created/Modified

- `src/aquapose/engine/__init__.py` - Engine package public API, exports Stage and PipelineContext via __all__
- `src/aquapose/engine/stages.py` - Stage Protocol with @runtime_checkable, PipelineContext dataclass with .get() method
- `tests/unit/engine/__init__.py` - Engine test package init
- `tests/unit/engine/test_stages.py` - 7 unit tests for protocol conformance, accumulation, and import boundary

## Decisions Made

- Used `@runtime_checkable` on Stage Protocol so `isinstance(obj, Stage)` works at runtime — needed for pipeline validation and future health checks
- PipelineContext fields declared as generic `list[dict[str, list]]` etc. rather than domain types, to keep the import boundary clean without TYPE_CHECKING exceptions
- `get()` raises `ValueError` (not `KeyError`) so callers can distinguish "stage hasn't run yet" from "invalid field name" (which raises `AttributeError`)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Pre-commit (ruff) auto-fixed minor import ordering and parentheses style issues on first commit attempt. Re-staged and committed cleanly on second attempt.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Engine package foundation is complete — Stage Protocol and PipelineContext are the contracts all subsequent engine plans build on
- Plans 13-02 (config), 13-03 (orchestrator), and 13-04 (stage migrations) all depend on these types
- No blockers

---
*Phase: 13-engine-core*
*Completed: 2026-02-25*
