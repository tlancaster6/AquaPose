---
phase: 54-chunk-aware-diagnostics-and-eval-migration
plan: "04"
subsystem: engine
tags: [cleanup, observers, refactor]
dependency_graph:
  requires: [54-03]
  provides: []
  affects: [engine, cli]
tech_stack:
  added: []
  patterns: [observer-cleanup, separation-of-concerns]
key_files:
  created: []
  modified:
    - src/aquapose/engine/observer_factory.py
    - src/aquapose/engine/__init__.py
    - src/aquapose/engine/orchestrator.py
    - src/aquapose/cli.py
    - src/aquapose/evaluation/viz/trails.py
  deleted:
    - src/aquapose/engine/overlay_observer.py
    - src/aquapose/engine/animation_observer.py
    - src/aquapose/engine/tracklet_trail_observer.py
    - tests/unit/engine/test_overlay_observer.py
    - tests/unit/engine/test_animation_observer.py
    - tests/unit/engine/test_tracklet_trail_observer.py
key_decisions:
  - Removed frame_source parameter from build_observers entirely since no remaining observers need it
  - diagnostic mode now produces [Console, Timing, Diagnostic] only — no viz observers
  - --add-observer CLI choices trimmed to [timing, diagnostic, console]
metrics:
  duration: "~5 minutes"
  completed: "2026-03-04T20:27:39Z"
  tasks_completed: 1
  tasks_total: 1
  files_modified: 5
  files_deleted: 6
---

# Phase 54 Plan 04: Remove Visualization Observers from Engine Summary

Deleted three in-pipeline viz observers and cleaned all references, leaving engine diagnostic mode with Console + Timing + DiagnosticObserver only.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Delete visualization observers and clean observer_factory | 83ef301 | Deleted 3 observers + 3 test files; updated factory, init, orchestrator, cli, trails.py |

## What Was Built

Removed `overlay_observer.py`, `animation_observer.py`, and `tracklet_trail_observer.py` from `engine/` together with their unit tests. These observers depended on a live `frame_source` and wrote video files mid-pipeline. With visualization migrated to `aquapose viz` post-run CLI commands (Plan 03), keeping them in the engine was redundant and violated the separation: pipeline runs produce caches, viz commands consume them.

**Changes made:**

- `observer_factory.py`: Removed imports of three deleted observers, trimmed `_OBSERVER_MAP` to `{timing, diagnostic, console}`, rewrote `diagnostic` mode to emit only `[Console, Timing, DiagnosticObserver]`, removed `synthetic` mode special-casing (it now just gets `Console + Timing` like `production`), removed `frame_source` parameter entirely since no remaining observer needs it.
- `engine/__init__.py`: Removed `Animation3DObserver`, `Overlay2DObserver`, `TrackletTrailObserver` imports and `__all__` entries.
- `orchestrator.py`: Dropped `frame_source=chunk_source` kwarg from `build_observers` call and updated comment.
- `cli.py`: Removed `"overlay2d"` and `"animation3d"` from `--add-observer` choices.
- `evaluation/viz/trails.py`: Cleaned stale comment referencing `TrackletTrailObserver`.

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `hatch run test` (excluding pre-existing failure in `test_stage_association.py`): 790 passed
- `hatch run lint`: All checks passed
- `grep -r "overlay_observer|animation_observer|tracklet_trail_observer" src/`: No results
- `grep -r "Overlay2DObserver|Animation3DObserver|TrackletTrailObserver" src/`: No results

Note: `test_stage_association.py::test_default_grid_ray_distance_threshold_values` is a pre-existing failure (wrong expected values for `DEFAULT_GRID` ray_distance_threshold) confirmed to exist before this plan's changes.

## Self-Check: PASSED

- `83ef301` exists in git log
- All 6 deleted files confirmed absent from filesystem
- Updated files verified in place
