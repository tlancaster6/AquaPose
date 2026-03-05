---
phase: 52-chunk-orchestrator-and-handoff
plan: 02
subsystem: engine
tags: [chunk-mode, orchestration, identity-stitching, hdf5, ocsort]

# Dependency graph
requires:
  - phase: 52-01
    provides: ChunkHandoff, ChunkFrameSource, write_handoff, PipelineConfig.chunk_size foundation

provides:
  - ChunkOrchestrator class in engine/orchestrator.py with full chunk loop
  - _stitch_identities() private helper with majority-vote conflict resolution
  - track_id_to_global field on ChunkHandoff for OC-SORT continuity stitching
  - _build_stages_for_chunk(), _format_eta(), _write_skipped_metadata() helpers
  - 6 unit tests covering identity stitching and chunk boundary computation

affects:
  - 52-03-PLAN.md (CarryForward migration — uses ChunkOrchestrator as entrypoint)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Chunk orchestration pattern: VideoFrameSource opened once, ChunkFrameSource views per chunk
    - Identity stitching via OC-SORT track ID continuity with majority-vote conflict resolution
    - HDF5 managed by orchestrator (not observer) to apply global frame offsets per chunk
    - Failed chunks produce ID gap in next_global_id to prevent accidental reuse

key-files:
  created:
    - tests/unit/engine/test_chunk_orchestrator.py
  modified:
    - src/aquapose/engine/orchestrator.py
    - src/aquapose/engine/__init__.py
    - tests/unit/engine/test_chunk_handoff.py

key-decisions:
  - "ChunkOrchestrator calls build_stages(config, frame_source=chunk_source) rather than duplicating stage wiring — build_stages already supports frame_source injection"
  - "PipelineContext is a mutable dataclass so initial_context.carry_forward = carry is safe (no FrozenInstanceError)"
  - "HDF5ExportObserver stripped from observer list per chunk — orchestrator owns Midline3DWriter and applies global frame offset"
  - "ConsoleObserver stripped unless verbose=True — chunk progress line printed to stdout instead"

patterns-established:
  - "Identity stitching: (camera_id, track_id) -> global_fish_id lookup from prev_handoff.track_id_to_global, majority-vote resolves conflicts"
  - "Failed chunk recovery: prev_handoff=None (fresh trackers next chunk) + next_global_id += 1 (ID gap for isolation)"

requirements-completed: [CHUNK-01, CHUNK-04, CHUNK-05, IDENT-01, IDENT-02, OUT-01]

# Metrics
duration: 8min
completed: 2026-03-03
---

# Phase 52 Plan 02: ChunkOrchestrator Summary

**ChunkOrchestrator with majority-vote identity stitching, per-chunk HDF5 flush with global frame offset, and failed-chunk skip with ID gap isolation**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-03T23:25:48Z
- **Completed:** 2026-03-03T23:33:36Z
- **Tasks:** 3
- **Files modified:** 4 (orchestrator.py, __init__.py, test_chunk_orchestrator.py, test_chunk_handoff.py)

## Accomplishments

- Added `track_id_to_global` field to `ChunkHandoff` and implemented `_stitch_identities()` with majority-vote conflict resolution for track continuity across chunk boundaries
- Implemented `ChunkOrchestrator.run()` with the full chunk loop: ChunkFrameSource windowing, PosePipeline per chunk, identity stitching, HDF5 flush with `chunk_start + local_idx` global offset, atomic ChunkHandoff write, progress reporting, and failed-chunk skip
- Exported `ChunkOrchestrator` from `aquapose.engine` and created 6 unit tests covering all identity stitching cases and boundary computation

## Task Commits

1. **Task 1: ChunkHandoff track_id_to_global and _stitch_identities** - `24e8b4a` (feat)
2. **Task 2: ChunkOrchestrator.run() and helpers** - `bf91a9d` (feat)
3. **Task 3: Export and unit tests** - `a696e9b` (feat)
4. **Typecheck fixes** - `1e239a8` (fix)

## Files Created/Modified

- `src/aquapose/engine/orchestrator.py` - Added track_id_to_global to ChunkHandoff, _stitch_identities(), _build_stages_for_chunk(), _format_eta(), _write_skipped_metadata(), and ChunkOrchestrator class
- `src/aquapose/engine/__init__.py` - Added ChunkOrchestrator to imports and __all__
- `tests/unit/engine/test_chunk_orchestrator.py` - 6 unit tests for identity stitching and chunk boundary computation
- `tests/unit/engine/test_chunk_handoff.py` - Updated to include track_id_to_global in ChunkHandoff construction

## Decisions Made

- Used `build_stages(config, frame_source=chunk_source)` instead of duplicating stage wiring — `build_stages` already supports `frame_source` injection parameter
- `PipelineContext` is mutable (`@dataclass` not `@dataclass(frozen=True)`) so direct `initial_context.carry_forward = carry` assignment is valid
- `HDF5ExportObserver` stripped from observer list per chunk — orchestrator owns `Midline3DWriter` lifecycle and applies `chunk_start + local_idx` global frame offset
- `ConsoleObserver` suppressed unless `verbose=True` — replaced by per-chunk progress line to stdout

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_chunk_handoff.py tests broken by ChunkHandoff schema change**
- **Found during:** Task 3 (unit test run)
- **Issue:** `test_chunk_handoff.py` constructed `ChunkHandoff` without the new `track_id_to_global` field added in Task 1, causing `TypeError: missing 1 required positional argument`
- **Fix:** Updated all 3 affected test constructions in `test_chunk_handoff.py` to include `track_id_to_global={...}` parameter
- **Files modified:** `tests/unit/engine/test_chunk_handoff.py`
- **Verification:** All 801 tests pass after fix
- **Committed in:** `a696e9b` (Task 3 commit)

**2. [Rule 1 - Bug] Fixed typecheck errors from `object` type annotation on config**
- **Found during:** Post-task verification (`hatch run typecheck`)
- **Issue:** `ChunkOrchestrator.__init__` typed `config: object`, causing attribute access errors (config.output_dir, config.n_animals etc.) and incompatible arg type when passing to PosePipeline
- **Fix:** Added `TYPE_CHECKING` import guard for `PipelineConfig`, changed param type to `PipelineConfig`, removed `type: ignore` comments, fixed `int | None` dict key type with explicit `int()` cast
- **Files modified:** `src/aquapose/engine/orchestrator.py`
- **Verification:** `hatch run typecheck` reports no errors for orchestrator.py
- **Committed in:** `1e239a8`

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered

- Ruff pre-commit hook reformatted the file on first commit attempt (Task 2) and flagged an unused variable `next_id` in tests (Task 3) — both resolved before final commit.

## Next Phase Readiness

- `ChunkOrchestrator` is the production chunk-mode entrypoint, ready for use by 52-03
- Plan 52-03 handles CarryForward migration to use ChunkHandoff for cross-chunk state

---
*Phase: 52-chunk-orchestrator-and-handoff*
*Completed: 2026-03-03*
