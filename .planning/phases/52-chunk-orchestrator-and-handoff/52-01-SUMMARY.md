---
phase: 52-chunk-orchestrator-and-handoff
plan: "01"
subsystem: engine
tags: [chunk-mode, frame-source, orchestrator, dataclass, pickle, serialization]

# Dependency graph
requires:
  - phase: 51-frame-source-refactor
    provides: VideoFrameSource protocol and concrete implementation used by ChunkFrameSource

provides:
  - ChunkHandoff frozen dataclass for cross-chunk state (tracks_2d_state, identity_map, next_global_id)
  - ChunkFrameSource windowed view into VideoFrameSource with local 0-based indexing
  - write_handoff atomic serializer using temp-file + os.replace pattern
  - PipelineConfig.chunk_size field (int | None, default None) with sub-100 warning
  - engine/orchestrator.py module as home for chunk infrastructure

affects:
  - 52-02-chunk-orchestrator (uses ChunkHandoff, ChunkFrameSource, chunk_size)
  - 52-03-carry-forward-migration (ChunkHandoff is the new type alongside CarryForward until plan 03)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Frozen dataclass for cross-boundary state (immutable, replaced wholesale each chunk)
    - Atomic file write via tempfile.NamedTemporaryFile + os.replace
    - Windowed frame source with no-op context manager (orchestrator opens VideoFrameSource once)

key-files:
  created:
    - src/aquapose/engine/orchestrator.py
    - tests/unit/engine/test_chunk_handoff.py
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/engine/__init__.py
    - src/aquapose/core/types/frame_source.py
    - src/aquapose/core/types/__init__.py

key-decisions:
  - "ChunkHandoff placed in engine/orchestrator.py (not core/context.py) — plan instruction; circular import concern noted in STATE.md for plan 52-02 to watch"
  - "ChunkFrameSource context manager is no-op — orchestrator opens VideoFrameSource once for the entire run"
  - "chunk_size=0 treated same as None by callers using `config.chunk_size or None` — no special coercion in load_config"

patterns-established:
  - "Atomic write pattern: tempfile.NamedTemporaryFile + os.replace for safe disk handoff files"
  - "Windowed frame source: ChunkFrameSource wraps VideoFrameSource with local 0-based index space and global_frame_offset property"

requirements-completed: [CHUNK-02, CHUNK-03, CHUNK-04, CHUNK-05]

# Metrics
duration: 5min
completed: 2026-03-03
---

# Phase 52 Plan 01: Chunk Orchestrator Foundation Summary

**ChunkHandoff frozen dataclass, ChunkFrameSource windowed view, write_handoff atomic serializer, and PipelineConfig.chunk_size field establishing the chunk-mode data model**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-03T23:22:15Z
- **Completed:** 2026-03-03T23:25:48Z
- **Tasks:** 4
- **Files modified:** 6

## Accomplishments

- Added `chunk_size: int | None = None` to `PipelineConfig` with sub-100 warning via `logger.warning`
- Implemented `ChunkFrameSource` in `core/types/frame_source.py` with no-op context manager, local index iteration, `global_frame_offset`, and `read_frame`
- Created `engine/orchestrator.py` with `ChunkHandoff` frozen dataclass and `write_handoff` atomic serializer
- 8 new unit tests covering all above components; all 795 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Add chunk_size to PipelineConfig** - `ed304f6` (feat)
2. **Task 2: Implement ChunkFrameSource** - `d1093af` (feat)
3. **Task 3: Create engine/orchestrator.py** - `777d4e8` (feat)
4. **Task 4: Write unit tests** - `82e2ca1` (test)

**Plan metadata:** TBD (docs commit)

## Files Created/Modified

- `src/aquapose/engine/config.py` - Added `chunk_size: int | None = None` field, module-level logger, and sub-100 warning in `load_config()`
- `src/aquapose/engine/orchestrator.py` - New module with `ChunkHandoff` frozen dataclass and `write_handoff` atomic serializer
- `src/aquapose/engine/__init__.py` - Added `ChunkHandoff` and `write_handoff` exports
- `src/aquapose/core/types/frame_source.py` - Added `ChunkFrameSource` class with all protocol methods
- `src/aquapose/core/types/__init__.py` - Added `ChunkFrameSource` export
- `tests/unit/engine/test_chunk_handoff.py` - 8 unit tests for all new components

## Decisions Made

- `ChunkHandoff` placed in `engine/orchestrator.py` per plan instructions (plan 52-02 should verify no circular imports with `core/tracking/stage.py`)
- `chunk_size=0` treated as None by convention (`config.chunk_size or None`) rather than coerced in `load_config()`; keeps config layer simple
- `ChunkFrameSource` no-op context manager: orchestrator owns VideoFrameSource lifecycle, chunks are just views

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Ruff pre-commit hooks auto-fixed import ordering and formatting on each commit (3 hook auto-fix cycles). Not a deviation — standard project workflow.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `ChunkHandoff`, `ChunkFrameSource`, `write_handoff`, and `chunk_size` all ready for plan 52-02 (ChunkOrchestrator)
- Plan 52-02 should verify no circular import: `engine/orchestrator.py` importing from `core/tracking/stage.py` was flagged as a concern in STATE.md

---
*Phase: 52-chunk-orchestrator-and-handoff*
*Completed: 2026-03-03*
