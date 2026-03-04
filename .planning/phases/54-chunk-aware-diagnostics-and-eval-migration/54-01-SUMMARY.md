---
phase: 54-chunk-aware-diagnostics-and-eval-migration
plan: 01
subsystem: engine
tags: [diagnostic-observer, chunk-orchestrator, observer-factory, pickle-cache, manifest-json]

# Dependency graph
requires:
  - phase: 53-integration-and-validation
    provides: ChunkOrchestrator, build_observers, DiagnosticObserver with per-stage cache
provides:
  - DiagnosticObserver with per-chunk single-cache layout (diagnostics/chunk_NNN/cache.pkl)
  - manifest.json written/appended at diagnostics/manifest.json per chunk
  - load_chunk_cache() in core/context.py with fingerprint-mismatch warning
  - ChunkOrchestrator allows diagnostic mode with multi-chunk runs (mutual exclusion removed)
affects: 54-02, 54-03, 54-04, eval tooling, Jupyter diagnostic workflows

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Per-chunk single-cache layout: diagnostics/chunk_NNN/cache.pkl written on PipelineComplete
    - Manifest append pattern: each chunk appends to diagnostics/manifest.json atomically
    - Fingerprint mismatch as warning not error (soft compat check)

key-files:
  created:
    - tests/unit/engine/test_orchestrator.py
  modified:
    - src/aquapose/engine/diagnostic_observer.py
    - src/aquapose/core/context.py
    - src/aquapose/engine/observer_factory.py
    - src/aquapose/engine/orchestrator.py
    - src/aquapose/engine/__init__.py
    - src/aquapose/core/__init__.py
    - tests/unit/engine/test_diagnostic_observer.py
    - tests/unit/engine/test_resume_cli.py
    - tests/unit/engine/test_stage_cache_write.py
    - tests/unit/engine/test_chunk_orchestrator.py

key-decisions:
  - "Per-chunk cache written on PipelineComplete, not per-stage — single cache.pkl contains full final context"
  - "Manifest JSON appended per chunk (read-modify-write) so each DiagnosticObserver instance handles its own chunk entry"
  - "Fingerprint mismatch logs warning but does not raise StaleCacheError — allows loading caches from minor code evolution"
  - "Mutual exclusion between diagnostic and chunk mode removed — DiagnosticObserver now supports multi-chunk runs"

patterns-established:
  - "chunk_idx parameter: DiagnosticObserver accepts chunk_idx=N, writes to diagnostics/chunk_NNN/"
  - "build_observers chunk_idx forwarding: orchestrator passes chunk_idx per iteration so observers write to correct subdirectory"
  - "Atomic manifest write: tempfile + os.replace prevents partial writes on manifest.json"

requirements-completed: []

# Metrics
duration: 10min
completed: 2026-03-04
---

# Phase 54 Plan 01: Chunk-Aware Diagnostic Cache Layout Summary

**DiagnosticObserver restructured to write one cache.pkl per chunk into diagnostics/chunk_NNN/ with manifest.json, removing the diagnostic+chunk mutual exclusion**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-04T19:56:37Z
- **Completed:** 2026-03-04T20:07:01Z
- **Tasks:** 2 (Task 1 TDD with 3 commits, Task 2 with 1 commit)
- **Files modified:** 10

## Accomplishments

- Rewrote DiagnosticObserver to write `diagnostics/chunk_{idx:03d}/cache.pkl` on PipelineComplete (not per-stage), containing the full final PipelineContext
- Added `manifest.json` written/appended per chunk with run_id, total_frames, chunk_size, version_fingerprint, and per-chunk entry list
- Added `load_chunk_cache()` to `core/context.py` — loads new format, warns on fingerprint mismatch (no StaleCacheError for fingerprint issues)
- Removed ValueError for diagnostic+multi-chunk from ChunkOrchestrator; diagnostic mode now works with any chunk_size
- Wired `chunk_idx` through `build_observers()` to DiagnosticObserver so each chunk writes to the correct subdirectory
- Updated all affected tests including test_resume_cli.py and test_stage_cache_write.py for new layout

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Add failing tests** - `7ceb4f4` (test)
2. **Task 1 GREEN: Restructure DiagnosticObserver for per-chunk layout** - `313119f` (feat)
3. **Task 2: Remove mutual exclusion, wire chunk_idx** - `01eff05` (feat)

## Files Created/Modified

- `src/aquapose/engine/diagnostic_observer.py` - Added chunk_idx param, _write_chunk_cache(), _write_manifest(); removed _write_stage_cache()
- `src/aquapose/core/context.py` - Added load_chunk_cache() with fingerprint-mismatch warning
- `src/aquapose/engine/observer_factory.py` - Added chunk_idx param, forwarded to DiagnosticObserver
- `src/aquapose/engine/orchestrator.py` - Removed ValueError, pass chunk_idx to build_observers, add _update_manifest_total_frames()
- `src/aquapose/engine/__init__.py` - Export load_chunk_cache
- `src/aquapose/core/__init__.py` - Export load_chunk_cache
- `tests/unit/engine/test_diagnostic_observer.py` - Added 14 new tests for new layout
- `tests/unit/engine/test_orchestrator.py` - New file: diagnostic+chunk and chunk_idx wiring tests
- `tests/unit/engine/test_resume_cli.py` - Updated end-to-end test for new cache layout
- `tests/unit/engine/test_stage_cache_write.py` - Rewritten for PipelineComplete-triggered cache write

## Decisions Made

- **Per-chunk single cache on PipelineComplete**: The old design wrote one file per stage (detection_cache.pkl etc.) on each StageComplete. The new design writes one cache.pkl per chunk on PipelineComplete containing the full final PipelineContext. Simpler for multi-chunk workflows.
- **Manifest append per observer instance**: Each DiagnosticObserver reads the existing manifest.json, adds its chunk's entry, and writes back. This allows incremental appending without orchestrator coordination per chunk.
- **Fingerprint mismatch = warning, not error**: Previously StaleCacheError was raised on any mismatch. Now it's a warning — allows loading caches from minor field additions without blocking workflows.
- **Mutual exclusion removal**: Per plan specification, the diagnostic+chunk ValueError from Phase 53 is removed. DiagnosticObserver's new layout supports multi-chunk runs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_resume_cli.py and test_stage_cache_write.py for new cache layout**
- **Found during:** Task 1 GREEN (implementing DiagnosticObserver rewrite)
- **Issue:** test_resume_cli.py and test_stage_cache_write.py tested the old per-stage cache behavior (detection_cache.pkl written on StageComplete). After removing _write_stage_cache(), these tests failed.
- **Fix:** Updated test_resume_cli.py to fire PipelineComplete and load from chunk_000/cache.pkl. Rewrote test_stage_cache_write.py to use new PipelineComplete-triggered write flow.
- **Files modified:** tests/unit/engine/test_resume_cli.py, tests/unit/engine/test_stage_cache_write.py
- **Verification:** All tests pass (797 pass, only 5 pre-existing failures unrelated to this work)
- **Committed in:** 313119f (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug: updated tests for behavior change)
**Impact on plan:** Required fix — these tests were testing behavior explicitly removed by this plan. No scope creep.

## Issues Encountered

- Pre-existing test failures in `tests/unit/evaluation/test_stage_association.py` (5 tests asserting stale DEFAULT_GRID values) — confirmed pre-existing before this work, logged to deferred-items.md.

## Next Phase Readiness

- DiagnosticObserver chunk-aware layout ready for use by eval tooling in plans 54-02 through 54-04
- load_chunk_cache() available for eval sweeps to load per-chunk caches
- Diagnostic mode can now be used on full multi-chunk runs
- manifest.json provides chunk inventory for eval tooling to discover available caches

---
*Phase: 54-chunk-aware-diagnostics-and-eval-migration*
*Completed: 2026-03-04*
