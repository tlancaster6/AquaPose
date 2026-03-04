---
phase: 54-chunk-aware-diagnostics-and-eval-migration
plan: 02
subsystem: evaluation
tags: [evaluation, diagnostics, chunk-cache, multi-chunk, pipeline-context]

# Dependency graph
requires:
  - phase: 54-01
    provides: per-chunk single-cache layout (chunk_NNN/cache.pkl + manifest.json), load_chunk_cache()
provides:
  - EvalRunner loads from chunk_NNN/cache.pkl layout (manifest-based and fallback glob discovery)
  - _merge_chunk_contexts() merges multiple PipelineContext objects with correct frame offsets
  - load_run_context() shared utility returns (merged_ctx, manifest_meta) for any run directory
  - TuningOrchestrator uses load_run_context() instead of per-stage flat file loading
affects: [evaluation/viz, CLI eval subcommand, future tuning workflows]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - load_run_context() as shared entry point for all evaluation tooling consuming diagnostic caches
    - stages_present inferred from merged PipelineContext field presence (not file existence)
    - _offset_tracklet_frames/_offset_group_frames for cross-chunk frame index consistency

key-files:
  created: []
  modified:
    - src/aquapose/evaluation/runner.py
    - src/aquapose/evaluation/tuning.py
    - src/aquapose/evaluation/__init__.py
    - tests/unit/evaluation/test_runner.py
    - tests/unit/evaluation/test_tuning.py

key-decisions:
  - "load_run_context() placed in runner.py and exported — single shared utility for both EvalRunner and TuningOrchestrator"
  - "stages_present inferred from merged context fields (not per-stage cache files) — aligns with single-cache-per-chunk model"
  - "Tracklet frame indices offset by chunk start_frame during merge — critical for cross-chunk eval correctness"
  - "Manifest used when present; fallback glob discovery for runs without manifest (robustness)"
  - "Single-chunk runs bypass merge entirely — original PipelineContext returned as-is for zero overhead"

patterns-established:
  - "load_run_context(run_dir) -> (PipelineContext | None, dict): canonical entry for any eval tool loading diagnostic data"
  - "Stage presence detection via ctx.field is not None — works for both single and merged contexts"

requirements-completed: []

# Metrics
duration: 9min
completed: 2026-03-04
---

# Phase 54 Plan 02: Eval Migration to Chunk Cache Layout Summary

**EvalRunner and TuningOrchestrator migrated to per-chunk cache layout via shared load_run_context() utility with multi-chunk merging and correct frame index offsets**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-03-04T20:09:54Z
- **Completed:** 2026-03-04T20:19:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Rewrote EvalRunner to use new chunk_NNN/cache.pkl layout with manifest-based and fallback glob discovery
- Added load_run_context() shared utility that merges multiple PipelineContext objects with correct frame offsets
- Updated TuningOrchestrator to use load_run_context() instead of per-stage flat file loading
- Added 17 new tests for EvalRunner (chunk layout, multi-chunk merging, fallback discovery, load_run_context utility)
- Added 4 new tests for TuningOrchestrator chunk loading
- Exported load_run_context from evaluation package __init__.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Update EvalRunner for chunk cache layout** - `36dccb2` (feat)
2. **Task 2: Update TuningOrchestrator for chunk cache layout** - `89a5182` (feat)

**Plan metadata:** (docs commit follows)

_Note: Task 1 used TDD (RED-GREEN cycle). Task 2 added new tests for chunk loading._

## Files Created/Modified

- `src/aquapose/evaluation/runner.py` - Rewrote _discover_caches() as load_run_context(), added _merge_chunk_contexts(), _offset_tracklet_frames(), _offset_group_frames(); EvalRunner now uses merged context; added load_run_context to __all__
- `src/aquapose/evaluation/tuning.py` - Replaced load_stage_cache() loop with load_run_context(); builds _caches dict from merged context field presence
- `src/aquapose/evaluation/__init__.py` - Added load_run_context to imports and __all__
- `tests/unit/evaluation/test_runner.py` - Replaced old per-stage fixture helpers with chunk layout fixtures; 17 tests covering single-chunk, multi-chunk, manifest/fallback discovery, load_run_context
- `tests/unit/evaluation/test_tuning.py` - Added TestTuningOrchestratorChunkLoading with 4 tests

## Decisions Made

- load_run_context() placed in runner.py and exported — single shared utility for both EvalRunner and TuningOrchestrator, avoids duplication as specified in plan
- stages_present inferred from merged PipelineContext field presence (not per-stage cache files) — naturally aligns with single-cache-per-chunk model where all stage data lives in one context
- Tracklet frame indices offset by chunk start_frame during merge — critical for cross-chunk eval correctness (tracklet.frames must be globally consistent)
- Manifest used when present, fallback glob discovery for runs without manifest — robust to both cases
- Single-chunk runs return original PipelineContext as-is — zero overhead, identical behavior to pre-change

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Pre-commit hooks fixed minor ruff formatting issues (loop variable renaming, unused variable prefixing, import sort order in __all__) — all resolved in same commit cycle.

## Next Phase Readiness

- EvalRunner and TuningOrchestrator both use new chunk cache layout
- load_run_context() available as a shared utility for Phase 54 Plans 03-04 (visualization migration)
- aquapose eval CLI works with new layout (EvalRunner is the CLI backend)

## Self-Check: PASSED

- FOUND: src/aquapose/evaluation/runner.py
- FOUND: src/aquapose/evaluation/tuning.py
- FOUND: src/aquapose/evaluation/__init__.py
- FOUND: tests/unit/evaluation/test_runner.py
- FOUND: tests/unit/evaluation/test_tuning.py
- FOUND: commit 36dccb2 (Task 1)
- FOUND: commit 89a5182 (Task 2)

---
*Phase: 54-chunk-aware-diagnostics-and-eval-migration*
*Completed: 2026-03-04*
