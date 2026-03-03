---
phase: 52-chunk-orchestrator-and-handoff
plan: "03"
subsystem: engine
tags: [refactor, carry-state, chunk-handoff, migration]
dependency_graph:
  requires: [52-02]
  provides: [ChunkHandoff-canonical-in-core]
  affects: [engine/orchestrator, engine/pipeline, core/tracking/stage, core/context]
tech_stack:
  added: []
  patterns: [ChunkHandoff frozen dataclass in core/context, import ChunkHandoff from core not engine]
key_files:
  created: []
  modified:
    - src/aquapose/core/context.py
    - src/aquapose/core/__init__.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/core/tracking/stage.py
    - src/aquapose/engine/orchestrator.py
    - src/aquapose/core/tracking/ocsort_wrapper.py
    - tests/unit/core/tracking/test_tracking_stage.py
    - tests/unit/engine/test_stage_skip.py
    - tests/unit/core/test_stage_cache.py
    - tests/unit/engine/test_build_stages.py
decisions:
  - ChunkHandoff moved to core/context.py so core/tracking/stage.py can import it without violating engine->core boundary
  - TrackingStage.run() preserves identity_map, track_id_to_global, next_global_id when carry is ChunkHandoff
  - ChunkOrchestrator passes prev_handoff directly as carry_forward (no CarryForward wrapper)
  - engine/orchestrator.py re-imports ChunkHandoff from core/context at module level and re-exports via __init__.py
metrics:
  duration_seconds: 241
  completed_date: "2026-03-03"
  tasks_completed: 4
  files_modified: 10
---

# Phase 52 Plan 03: CarryForward Migration to ChunkHandoff Summary

**One-liner:** Removed `CarryForward` from codebase, migrated all carry-state to `ChunkHandoff` frozen dataclass (now canonical in `core/context.py`), preserving full identity fields across chunks.

## What Was Built

Completed the CarryForward -> ChunkHandoff migration, making `ChunkHandoff` the single canonical type for cross-chunk state in AquaPose. Key changes:

1. **`core/context.py`**: Deleted `CarryForward` dataclass, added `ChunkHandoff` dataclass (moved from `engine/orchestrator.py`). Updated `PipelineContext.carry_forward` field type from `CarryForward | None` to `ChunkHandoff | None`.

2. **`core/__init__.py`**: Replaced `CarryForward` export with `ChunkHandoff`.

3. **`engine/pipeline.py`**: Removed `CarryForward` import; updated `carry: CarryForward | None` to `carry: object | None`.

4. **`core/tracking/stage.py`**: Removed `CarryForward` import; imports `ChunkHandoff` from `core/context`. Updated `run()` signature to `carry: object | None` returning `ChunkHandoff`. On chunk boundary, preserves `identity_map`, `track_id_to_global`, `next_global_id` from input `ChunkHandoff`.

5. **`engine/orchestrator.py`**: Removed `ChunkHandoff` dataclass definition (it's now in `core/context`). Added `from aquapose.core.context import ChunkHandoff` at module level. Removed `CarryForward` wrapping inside `run()` — passes `prev_handoff` directly as `carry_forward`.

6. **Tests**: Updated all 4 test files to use `ChunkHandoff` instead of `CarryForward` in imports, constructions, and assertions.

## Decisions Made

- **ChunkHandoff in core/context.py**: `core/tracking/stage.py` needs to construct it, and core modules must not import from engine. Moving `ChunkHandoff` to `core/context.py` is the only option that respects the import boundary.
- **TrackingStage preserves identity fields**: When carry is `ChunkHandoff`, new carry copies `identity_map`, `track_id_to_global`, `next_global_id` from input. When carry is `None`, these default to `{}` / `0`.
- **Direct prev_handoff passthrough in orchestrator**: Cleaner than wrapping — avoids losing identity fields, aligns with plan 52 design intent.

## Verification

All success criteria confirmed:
- `grep -r "CarryForward" src/ --include="*.py"` returns no results
- `hatch run test`: 801 passed, 3 skipped, 0 failures
- `hatch run lint`: All checks passed
- All Phase 52 imports verified:
  - `from aquapose.engine import ChunkHandoff, ChunkOrchestrator, write_handoff` — OK
  - `from aquapose.core.types import ChunkFrameSource` — OK
  - `PipelineConfig(chunk_size=1000).chunk_size == 1000` — OK

## Phase 52 Success Criteria — All Met

1. ChunkOrchestrator loops over fixed-size chunks, invoking PosePipeline once per chunk — VERIFIED
2. ChunkHandoff (frozen dataclass) carries tracker state and identity map; written atomically to handoff.pkl — VERIFIED
3. Chunk-local fish IDs mapped to globally consistent IDs via track ID continuity — VERIFIED
4. Per-chunk 3D midlines flushed to HDF5 with correct global frame offset — VERIFIED
5. chunk_size=0 or null produces single-chunk degenerate run; chunk_size < 100 emits warning — VERIFIED

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed docstring reference in ocsort_wrapper.py**
- **Found during:** Task 4 verification
- **Issue:** `ocsort_wrapper.py` docstring still referenced `CarryForward.tracks_2d_state` after the class was removed
- **Fix:** Updated docstring to reference `ChunkHandoff.tracks_2d_state`
- **Files modified:** `src/aquapose/core/tracking/ocsort_wrapper.py`
- **Commit:** 041e556

## Self-Check: PASSED
