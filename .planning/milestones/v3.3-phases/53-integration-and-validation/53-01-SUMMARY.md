---
phase: 53-integration-and-validation
plan: 01
subsystem: engine
tags: [cli, orchestrator, chunk-mode, hdf5, observers]

# Dependency graph
requires:
  - phase: 52-chunk-orchestrator-and-handoff
    provides: ChunkOrchestrator with chunk loop, Midline3DWriter HDF5 output, ChunkHandoff stitching
provides:
  - CLI run command delegates entirely to ChunkOrchestrator (config-only handoff)
  - ChunkOrchestrator accepts max_chunks, stop_after, extra_observers, frame_source params
  - Mode conflict validation (diagnostic + multi-chunk raises ValueError)
  - HDF5ExportObserver deleted from codebase; orchestrator owns HDF5 via Midline3DWriter
affects:
  - 53-02 (remaining integration/validation plans)
  - evaluation tooling that may reference observer types

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "config-only handoff: CLI creates PipelineConfig, passes to ChunkOrchestrator"
    - "contextlib.ExitStack for conditional lifecycle management"

key-files:
  created: []
  modified:
    - src/aquapose/cli.py
    - src/aquapose/engine/orchestrator.py
    - src/aquapose/engine/observer_factory.py
    - src/aquapose/engine/__init__.py
    - tests/unit/engine/test_cli.py
    - tests/unit/engine/test_resume_cli.py
    - tests/unit/engine/test_chunk_orchestrator.py
  deleted:
    - src/aquapose/engine/hdf5_observer.py
    - tests/unit/engine/test_hdf5_observer.py

key-decisions:
  - "CLI delegates entirely to ChunkOrchestrator — no direct PosePipeline construction in cli.py"
  - "--resume-from CLI flag removed; load_stage_cache remains in core/context.py for programmatic use"
  - "--max-chunks added to CLI and ChunkOrchestrator constructor for single-chunk dry runs"
  - "HDF5ExportObserver deleted — orchestrator owns HDF5 output via Midline3DWriter"
  - "contextlib.ExitStack used so caller-owned frame_source is not closed by orchestrator"
  - "Mode conflict validation: diagnostic + chunk_size>0 + max_chunks!=1 raises ValueError"

patterns-established:
  - "config-only handoff: CLI builds PipelineConfig, hands to orchestrator; orchestrator builds stages/observers"
  - "frame_source lifecycle: CLI owns VideoFrameSource; passes to orchestrator; orchestrator does not close it"

requirements-completed: [OUT-02, INTEG-01, INTEG-02]

# Metrics
duration: 6min
completed: 2026-03-04
---

# Phase 53 Plan 01: Integration and Validation Summary

**ChunkOrchestrator wired as universal production path: HDF5ExportObserver deleted, CLI delegates via config-only handoff, --max-chunks added, diagnostic/chunk mode conflict validated**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-04T00:07:44Z
- **Completed:** 2026-03-04T00:13:44Z
- **Tasks:** 4
- **Files modified:** 9 (2 deleted, 7 modified)

## Accomplishments

- Deleted HDF5ExportObserver entirely — module, tests, factory registration, __init__ export, and orchestrator filter all removed
- ChunkOrchestrator now accepts max_chunks, stop_after, extra_observers, and frame_source parameters with mode conflict validation
- CLI `run` command refactored to config-only handoff pattern: builds PipelineConfig, passes to ChunkOrchestrator
- --resume-from CLI flag removed; --max-chunks added; --add-observer hdf5 choice removed
- All tests updated: mock ChunkOrchestrator instead of PosePipeline, mode conflict tests added

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete HDF5ExportObserver and remove all references** - `96187c9` (feat)
2. **Task 2: Add max_chunks, stop_after, extra_observers to ChunkOrchestrator** - `db5ebef` (feat)
3. **Task 3: Refactor CLI to delegate to ChunkOrchestrator** - `9fc6fef` (feat)
4. **Task 4: Update all tests for new CLI dispatch and mode conflict** - `bf6aa49` (feat)

## Files Created/Modified

- `src/aquapose/engine/hdf5_observer.py` - DELETED
- `tests/unit/engine/test_hdf5_observer.py` - DELETED
- `src/aquapose/engine/__init__.py` - Removed HDF5ExportObserver import and __all__ entry
- `src/aquapose/engine/observer_factory.py` - Removed hdf5 from _OBSERVER_MAP and all HDF5ExportObserver construction
- `src/aquapose/engine/orchestrator.py` - Added max_chunks, stop_after, extra_observers, frame_source; mode conflict validation; contextlib.ExitStack
- `src/aquapose/cli.py` - Config-only handoff to ChunkOrchestrator; --resume-from removed; --max-chunks added
- `tests/unit/engine/test_cli.py` - Rewritten to mock ChunkOrchestrator; HDF5 tests removed; max_chunks test added
- `tests/unit/engine/test_resume_cli.py` - --resume-from CLI tests removed; kept programmatic load_stage_cache tests
- `tests/unit/engine/test_chunk_orchestrator.py` - Mode conflict validation tests added

## Decisions Made

- CLI delegates entirely to ChunkOrchestrator — no direct PosePipeline or observer construction in cli.py
- --resume-from CLI flag removed; load_stage_cache stays in core/context.py for programmatic use (evaluation sweeps)
- HDF5ExportObserver deleted permanently — orchestrator owns HDF5 output via Midline3DWriter
- contextlib.ExitStack used to handle conditional frame_source lifecycle (caller-owned vs orchestrator-owned)
- Mode conflict validation: diagnostic + multi-chunk raises ValueError at construction time, not at run time

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Ruff format auto-fixed test file indentation on two commits (pre-commit hook). No logic changes — purely formatting.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- OUT-02, INTEG-01, INTEG-02 requirements met
- ChunkOrchestrator is the universal production path
- All tests passing (784 passed), lint clean
- Ready for Phase 53 Plan 02

---
*Phase: 53-integration-and-validation*
*Completed: 2026-03-04*
