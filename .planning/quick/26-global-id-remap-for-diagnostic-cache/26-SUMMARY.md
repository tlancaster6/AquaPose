---
phase: 26-global-id-remap-for-diagnostic-cache
plan: 01
subsystem: engine
tags: [diagnostic, orchestrator, identity-stitching, chunk-mode]
dependency_graph:
  requires: []
  provides: [globally-consistent fish IDs in diagnostic cache]
  affects: [diagnostic_observer.py, orchestrator.py]
tech_stack:
  added: []
  patterns: [deferred-write, in-place-remap, dataclasses.replace]
key_files:
  created: []
  modified:
    - src/aquapose/engine/diagnostic_observer.py
    - src/aquapose/engine/orchestrator.py
    - tests/unit/engine/test_diagnostic_observer.py
    - tests/unit/engine/test_chunk_orchestrator.py
    - tests/unit/engine/test_resume_cli.py
    - tests/unit/engine/test_stage_cache_write.py
decisions:
  - flush_cache() is the sole disk-write path; PipelineComplete is now a no-op for I/O
  - midlines_3d remapped in-place (Midline3D is mutable) to avoid copy overhead
  - tracklet_groups rebuilt via dataclasses.replace (TrackletGroup is frozen=True)
  - HDF5 write loop simplified: uses already-remapped midlines, no double remap
  - DiagnosticObserver import inside the observers loop (not module-level) to avoid circular import risk
metrics:
  duration: 9min
  completed: "2026-03-11"
  tasks: 2
  files: 6
---

# Quick Task 26: Global ID Remap for Diagnostic Cache Summary

**One-liner:** Deferred diagnostic cache write to post-stitching so cache.pkl always contains globally-consistent fish IDs, not chunk-local IDs.

## What Was Built

### Task 1: Deferred DiagnosticObserver Cache Write

`diagnostic_observer.py` — `_on_pipeline_complete()` no longer writes cache.pkl or manifest.json. It now only updates `self._last_context` (the event context reference). A new public `flush_cache()` method is the sole path to disk: it writes `_write_chunk_cache()` and `_write_manifest()` using the stored context.

Updated 4 test files (`test_diagnostic_observer.py`, `test_chunk_orchestrator.py`, `test_resume_cli.py`, `test_stage_cache_write.py`) to add `observer.flush_cache()` before disk assertions. Added 2 new tests:
- `test_pipeline_complete_does_not_write_cache` — validates the deferral contract
- `test_flush_cache_writes_cache_and_manifest` — validates flush_cache() writes both files

### Task 2: In-Place Context Remap + Observer Flush in Orchestrator

`orchestrator.py` — After `_stitch_identities()` returns `identity_map`:

1. **Remap `context.midlines_3d` in-place**: iterates each frame dict, updates `Midline3D.fish_id` directly (mutable dataclass), rebuilds the dict with global keys.
2. **Remap `context.tracklet_groups` in-place**: rebuilds list using `dc_replace(group, fish_id=global_id)` since `TrackletGroup` is frozen.
3. **Simplified HDF5 write**: `writer.write_frame(global_frame_idx, frame_midlines)` directly — no redundant remap loop.
4. **Flush observer**: `for obs in observers: if isinstance(obs, DiagnosticObserver): obs.flush_cache()` — called after context is fully remapped.
5. **Simplified handoff building**: `gid = group.fish_id` (already global) instead of `identity_map.get(group.fish_id, group.fish_id)`.

Added `test_diagnostic_observer_flushed_after_remap` integration test: runs orchestrator with a real `DiagnosticObserver` and `TrackletGroup` (local fish_id=5), verifies cache.pkl contains global fish_id=0 in both `midlines_3d` and `tracklet_groups`.

Fixed two pre-existing test issues discovered by the new code path (Rule 1 auto-fix):
- `test_degenerate_single_chunk_output`: fake midline was `object()` (no `fish_id`), changed to `SimpleNamespace(fish_id=0)`
- `test_multi_chunk_mechanical_correctness`: groups were `SimpleNamespace` (incompatible with `dc_replace`), changed to real `TrackletGroup`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_resume_cli.py and test_stage_cache_write.py also expected old PipelineComplete write behavior**
- **Found during:** Task 1 test run
- **Issue:** Plan only listed `test_diagnostic_observer.py` and `test_chunk_orchestrator.py`, but two additional test files (`test_resume_cli.py`, `test_stage_cache_write.py`) called `observer.on_event(PipelineComplete(...))` and then immediately asserted cache files existed
- **Fix:** Added `observer.flush_cache()` after PipelineComplete in both files
- **Files modified:** `tests/unit/engine/test_resume_cli.py`, `tests/unit/engine/test_stage_cache_write.py`
- **Commits:** 0dc0866

**2. [Rule 1 - Bug] Existing integration tests used plain `object()` for fake midlines**
- **Found during:** Task 2 — new orchestrator code mutates `ml.fish_id = global_id` but `object()` has no `fish_id`
- **Fix:** Changed to `types.SimpleNamespace(fish_id=...)` in both degenerate and multi-chunk tests
- **Files modified:** `tests/unit/engine/test_chunk_orchestrator.py`
- **Commit:** 8bc3b20

**3. [Rule 1 - Bug] Existing multi-chunk test used `SimpleNamespace` for groups**
- **Found during:** Task 2 — `dataclasses.replace` requires actual dataclasses, not SimpleNamespace
- **Fix:** Changed to real `TrackletGroup(fish_id=0, tracklets=(...))` instances
- **Files modified:** `tests/unit/engine/test_chunk_orchestrator.py`
- **Commit:** 8bc3b20

## Self-Check: PASSED

Files verified:
- `src/aquapose/engine/diagnostic_observer.py` — exists, has `flush_cache()` method
- `src/aquapose/engine/orchestrator.py` — exists, has in-place remap + flush call
- `tests/unit/engine/test_chunk_orchestrator.py` — exists, has `test_diagnostic_observer_flushed_after_remap`

Commits verified:
- `0dc0866` — Task 1: defer DiagnosticObserver cache write
- `8bc3b20` — Task 2: remap context in-place and flush observer

All 1196 tests pass, 0 new failures.
