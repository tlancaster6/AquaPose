---
phase: 54-chunk-aware-diagnostics-and-eval-migration
plan: 03
subsystem: visualization
tags: [opencv, plotly, cli, click, evaluation, chunk-cache, diagnostics]

# Dependency graph
requires:
  - phase: 54-chunk-aware-diagnostics-and-eval-migration
    provides: per-chunk cache layout (chunk_NNN/cache.pkl + manifest.json) from Plan 01
provides:
  - aquapose viz CLI group with overlay/animation/trails/all subcommands
  - evaluation/viz/ package with generate_overlay, generate_animation, generate_trails, generate_all
  - Shared chunk cache loader utility (load_all_chunk_caches)
  - Deterministic fish color assignment from global fish_id
  - Multi-chunk continuous output (no per-chunk segmentation in output)
affects: [phase-55, pipeline-users, downstream-visualization-consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Chunk cache loader reads manifest.json and loads chunk_NNN/cache.pkl in order
    - Rebase tracklet frame indices to global space when merging across chunks
    - Deterministic color: palette[fish_id % palette_length]
    - generate_all() catches per-visualization failures, never propagates
    - CLI subcommands: run_dir argument + optional --output-dir / -o flag

key-files:
  created:
    - src/aquapose/evaluation/viz/__init__.py
    - src/aquapose/evaluation/viz/_loader.py
    - src/aquapose/evaluation/viz/overlay.py
    - src/aquapose/evaluation/viz/animation.py
    - src/aquapose/evaluation/viz/trails.py
    - tests/unit/evaluation/test_viz.py
  modified:
    - src/aquapose/cli.py
    - src/aquapose/evaluation/__init__.py

key-decisions:
  - "Rebase tracklet frame indices to global frame space when merging chunks (frame_offset per chunk)"
  - "VideoFrameSource requires both video_dir and calibration_path — raise FileNotFoundError before construction if calib missing"
  - "ctx_mgr typed as AbstractContextManager to satisfy basedpyright for with-statement usage"
  - "spline.knots and spline.degree accessed via getattr() since spline parameter typed as object"

patterns-established:
  - "load_all_chunk_caches() is the shared entry point for all viz modules to load multi-chunk data"
  - "generate_all() returns dict[str, Path | Exception] — callers check isinstance(v, Exception) for failures"
  - "Fish color assignment: PALETTE[fish_id % len(PALETTE)] — consistent across all viz modules"

requirements-completed: []

# Metrics
duration: 12min
completed: 2026-03-04
---

# Phase 54 Plan 03: Viz CLI Subcommands Summary

**Post-run visualization CLI (aquapose viz overlay/animation/trails/all) loading chunk caches and producing continuous multi-chunk output via evaluation/viz/ package**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-04T20:10:20Z
- **Completed:** 2026-03-04T20:22:26Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Created `evaluation/viz/` package with three independent visualization generators and a shared chunk cache loader
- All three generators merge data across chunks into continuous output (no per-chunk segmentation for the viewer)
- Wired `aquapose viz overlay|animation|trails|all` CLI group with `--output-dir` flag and graceful failure reporting
- 19 unit tests covering loader, generate_all graceful failure, and output path contracts

## Task Commits

Each task was committed atomically:

1. **Task 1: Create evaluation/viz/ modules** - `54ed0c7` (feat)
2. **Task 2: Wire aquapose viz CLI group** - `d389a57` (feat)
3. **Fix: Typecheck errors in viz modules** - `f3926f3` (fix) [Rule 1 - Bug]

## Files Created/Modified
- `src/aquapose/evaluation/viz/__init__.py` - Package exports + generate_all() with graceful failure
- `src/aquapose/evaluation/viz/_loader.py` - Shared chunk cache loader: reads manifest.json, loads all chunk_NNN/cache.pkl
- `src/aquapose/evaluation/viz/overlay.py` - generate_overlay(): reprojected 3D midlines on mosaic video across all chunks
- `src/aquapose/evaluation/viz/animation.py` - generate_animation(): merged midlines_3d as Plotly 3D HTML with unified scrubber
- `src/aquapose/evaluation/viz/trails.py` - generate_trails(): rebased tracklet trails + association mosaic across chunks
- `src/aquapose/cli.py` - Added viz CLI group with 4 subcommands
- `src/aquapose/evaluation/__init__.py` - Added viz re-exports
- `tests/unit/evaluation/test_viz.py` - 19 unit tests

## Decisions Made
- Tracklet frame indices rebased to global frame space when merging chunks (add chunk's frame offset)
- `VideoFrameSource` requires calibration_path (not optional) — raise `FileNotFoundError` before construction when calibration absent, fallback to black frames in outer except
- `ctx_mgr: AbstractContextManager` type annotation satisfies basedpyright for `with ctx_mgr as frame_iter` usage
- `spline.knots`/`spline.degree` accessed via `getattr()` since spline parameter typed as `object` throughout the codebase

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed typecheck errors in viz modules**
- **Found during:** Task 2 (after running hatch run check)
- **Issue:** Direct attribute access `spline.knots`/`spline.degree` on `object`-typed parameter; invalid `camera_ids` param in VideoFrameSource; untyped `ctx_mgr` for with-statement
- **Fix:** Used `getattr()` for spline attributes, removed invalid param, typed `ctx_mgr` as `AbstractContextManager`
- **Files modified:** overlay.py, animation.py, trails.py
- **Verification:** `hatch run typecheck` shows only pre-existing `VideoWriter_fourcc` errors (same pattern as overlay_observer.py, tracklet_trail_observer.py)
- **Committed in:** f3926f3

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Typecheck fixes necessary for correctness. No scope creep.

## Issues Encountered
- Pre-existing `test_stage_association.py::test_default_grid_ray_distance_threshold_values` failure (DEFAULT_GRID values mismatch) — pre-dates this plan, out of scope. Logged to deferred items.
- Pre-existing `VideoWriter_fourcc` basedpyright errors — same issue exists in `engine/overlay_observer.py` and `engine/tracklet_trail_observer.py`, not introduced by this plan.

## Next Phase Readiness
- `aquapose viz` CLI commands are ready for use on diagnostic runs
- The `generate_all()` function is the primary user entry point
- Phase 54-04 can proceed independently

## Self-Check: PASSED

All files exist and all commits verified:
- Files: 8/8 found
- Commits: 3/3 found (54ed0c7, d389a57, f3926f3)

---
*Phase: 54-chunk-aware-diagnostics-and-eval-migration*
*Completed: 2026-03-04*
