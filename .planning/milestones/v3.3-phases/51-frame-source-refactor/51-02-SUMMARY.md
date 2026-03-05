---
phase: 51-frame-source-refactor
plan: 02
subsystem: engine
tags: [frame-source, observer, video-set, config, refactor]

# Dependency graph
requires:
  - phase: 51-frame-source-refactor plan 01
    provides: FrameSource protocol and VideoFrameSource concrete implementation
provides:
  - Overlay2DObserver and TrackletTrailObserver accept injected FrameSource instead of video_dir
  - observer_factory.build_observers() accepts optional frame_source parameter
  - VideoSet deleted — no code imports from io/video.py
  - stop_frame removed from PipelineConfig with migration hint in _RENAME_HINTS
  - CLI creates VideoFrameSource and shares it with both build_stages and build_observers
affects: [52-chunk-orchestrator, any code constructing observers or building pipeline stages]

# Tech tracking
tech-stack:
  added: []
  patterns: [injected-frame-source, shared-frame-source-between-stages-and-observers]

key-files:
  created: []
  modified:
    - src/aquapose/engine/overlay_observer.py
    - src/aquapose/engine/tracklet_trail_observer.py
    - src/aquapose/engine/observer_factory.py
    - src/aquapose/engine/config.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/cli.py
    - tests/unit/engine/test_overlay_observer.py
    - tests/unit/engine/test_tracklet_trail_observer.py
    - tests/unit/engine/test_config.py
    - tests/unit/engine/test_cli.py
    - tests/unit/engine/test_resume_cli.py
    - tests/e2e/test_smoke.py
  deleted:
    - src/aquapose/io/video.py
    - tests/unit/io/test_video.py

key-decisions:
  - "VideoFrameSource is created in cli.py's run command and shared with both build_stages and build_observers — avoids double construction"
  - "VideoFrameSource imported at module level in cli.py (not locally) so tests can patch aquapose.cli.VideoFrameSource"
  - "Observers fall back to synthetic black frames when frame_source is None — preserves synthetic mode compatibility"
  - "stop_frame in YAML now raises ValueError with _RENAME_HINTS pointing to max_frames on frame source"

patterns-established:
  - "Shared-frame-source pattern: create VideoFrameSource once in CLI, pass to both build_stages and build_observers"
  - "Optional injection pattern: observer accepts frame_source=None, falls back to synthetic frames when absent"

requirements-completed: [FRAME-01, FRAME-03]

# Metrics
duration: 15min
completed: 2026-03-03
---

# Phase 51 Plan 02: Observer Migration and VideoSet Deletion Summary

**Observers migrated to injected FrameSource, VideoSet deleted, stop_frame removed from PipelineConfig with migration hint — frame windowing is now purely a frame-source concern**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-03T23:00:35Z
- **Completed:** 2026-03-03T23:15:00Z
- **Tasks:** 2
- **Files modified:** 12 modified, 2 deleted

## Accomplishments
- Migrated `Overlay2DObserver` and `TrackletTrailObserver` to accept an injected `FrameSource` instead of constructing a `VideoSet` internally from `video_dir`
- Removed `stop_frame` from `PipelineConfig` and added a `_RENAME_HINTS` migration hint directing users to `max_frames on frame source`
- Deleted `src/aquapose/io/video.py` (VideoSet class) and its test file — fully replaced by `VideoFrameSource` from Plan 01
- Updated CLI to create `VideoFrameSource` once and share it with both `build_stages` and `build_observers`

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate observers to FrameSource and update observer_factory** - `c23abf8` (feat)
2. **Task 2: Remove stop_frame from config, delete VideoSet, update tests** - `64ba5cc` (feat)

**Plan metadata:** (pending final docs commit)

## Files Created/Modified
- `src/aquapose/engine/overlay_observer.py` — constructor changed to accept `frame_source: FrameSource | None = None` instead of `video_dir`; falls back to synthetic black frames when None
- `src/aquapose/engine/tracklet_trail_observer.py` — constructor changed to accept `frame_source: FrameSource | None = None`; removed `stop_frame` parameter; uses `assert self._frame_source is not None` before context manager use
- `src/aquapose/engine/observer_factory.py` — `build_observers()` accepts `frame_source: FrameSource | None = None`; passes it to both observer constructors
- `src/aquapose/engine/config.py` — removed `stop_frame: int | None = None` field; updated `_RENAME_HINTS` with migration hint
- `src/aquapose/engine/pipeline.py` — `build_stages()` accepts `frame_source: VideoFrameSource | None = None` via TYPE_CHECKING import; removed `max_frames=config.stop_frame`
- `src/aquapose/cli.py` — module-level import of `VideoFrameSource`; creates it in `run` command for non-synthetic modes; passes to both `build_stages` and `build_observers`
- `tests/unit/engine/test_overlay_observer.py` — removed `video_dir=tmp_path` from constructor
- `tests/unit/engine/test_tracklet_trail_observer.py` — removed `video_dir` from `_make_observer` helper; removed `config.stop_frame` mock
- `tests/unit/engine/test_config.py` — updated test to assert `stop_frame` in YAML raises `ValueError` with "max_frames" in message
- `tests/unit/engine/test_cli.py` — added `patch("aquapose.cli.VideoFrameSource")` to `mock_pipeline` fixture
- `tests/unit/engine/test_resume_cli.py` — added `VideoFrameSource` patch to `_mock_pipeline_context` helper; updated return value unpacking
- `tests/e2e/test_smoke.py` — renamed `stop_frame` to `max_frames` parameter in `_build_real_config`
- `src/aquapose/io/video.py` — **deleted** (VideoSet replaced by VideoFrameSource)
- `tests/unit/io/test_video.py` — **deleted** (VideoSet tests no longer needed)

## Decisions Made
- **Shared VideoFrameSource in CLI**: Create once in `cli.py`'s `run` command and pass to both `build_stages` and `build_observers`, avoiding double construction and ensuring frame windowing is consistent.
- **Module-level import for testability**: `VideoFrameSource` imported at module level in `cli.py` so tests can patch `aquapose.cli.VideoFrameSource` before `VideoFrameSource.__init__` runs.
- **Optional injection with synthetic fallback**: Observers accept `frame_source=None` and fall back to synthetic black frames — preserves synthetic mode compatibility without special-casing.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed TrackletTrailObserver type error with assert before context manager**
- **Found during:** Task 1 (Migrate observers to FrameSource)
- **Issue:** `ctx_mgr = self._frame_source` where type is `FrameSource | None` caused basedpyright to report `reportOptionalContextManager` — `None` cannot be used as context manager
- **Fix:** Added `assert self._frame_source is not None` before context manager assignment in both `_generate_per_camera_trails` and `_generate_association_mosaic`
- **Files modified:** `src/aquapose/engine/tracklet_trail_observer.py`
- **Verification:** `hatch run typecheck` passes; tests pass
- **Committed in:** `c23abf8` (Task 1 commit)

**2. [Rule 3 - Blocking] Fixed CLI test failures from VideoFrameSource construction running before mock patches**
- **Found during:** Task 1 (Migrate observers to FrameSource)
- **Issue:** `VideoFrameSource` imported locally inside `run()` function body; `patch("aquapose.cli.VideoFrameSource")` could not intercept it, causing `FileNotFoundError` when tests ran with real filesystem
- **Fix:** Moved `VideoFrameSource` import to module level in `cli.py`; added `patch("aquapose.cli.VideoFrameSource")` to `test_cli.py` `mock_pipeline` fixture and `test_resume_cli.py` `_mock_pipeline_context` helper; updated return value unpacking in `test_resume_cli.py`
- **Files modified:** `src/aquapose/cli.py`, `tests/unit/engine/test_cli.py`, `tests/unit/engine/test_resume_cli.py`
- **Verification:** All 787 unit tests pass
- **Committed in:** `c23abf8` (Task 1 commit)

**3. [Rule 1 - Bug] Fixed pipeline.py type annotation too loose for frame_source**
- **Found during:** Task 1 (Migrate observers to FrameSource)
- **Issue:** `build_stages(config, frame_source: object = None)` — `object` type is not assignable to `VideoFrameSource | None`
- **Fix:** Used proper `VideoFrameSource | None` annotation with `TYPE_CHECKING` import block
- **Files modified:** `src/aquapose/engine/pipeline.py`
- **Verification:** `hatch run typecheck` passes
- **Committed in:** `c23abf8` (Task 1 commit)

**4. [Rule 3 - Blocking] Fixed pre-commit ruff formatter re-staging requirement**
- **Found during:** Both tasks (encountered twice)
- **Issue:** Pre-commit hook reformatted staged files after staging, causing commit to include reformatted diffs that weren't re-staged
- **Fix:** Re-staged all modified files and recommitted after formatter ran
- **Files modified:** Various (auto-reformatted)
- **Verification:** Pre-commit hooks pass cleanly on recommit
- **Committed in:** Both task commits

---

**Total deviations:** 4 auto-fixed (2 bugs, 2 blocking)
**Impact on plan:** All auto-fixes necessary for type correctness and test infrastructure. No scope creep.

## Issues Encountered
- Pre-commit ruff formatter fired twice (once per task), requiring re-stage/recommit each time — expected behavior for this project's pre-commit setup.
- `git add src/aquapose/io/video.py` failed on deleted file — used `git add -u` to stage deletions, then added remaining modified files.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 51 frame source refactor is complete — FrameSource protocol established, all consumers migrated, VideoSet deleted
- Phase 52 (Chunk Orchestrator) can proceed — it depends on the FrameSource/VideoFrameSource from Phase 51 for ChunkFrameSource views
- The shared-frame-source pattern established here (create once, pass to multiple consumers) is the model for the chunk orchestrator's VideoFrameSource lifecycle management

## Self-Check: PASSED

- overlay_observer.py: FOUND
- tracklet_trail_observer.py: FOUND
- observer_factory.py: FOUND
- config.py: FOUND
- io/video.py: CONFIRMED DELETED
- test_video.py: CONFIRMED DELETED
- Commit c23abf8: FOUND
- Commit 64ba5cc: FOUND

---
*Phase: 51-frame-source-refactor*
*Completed: 2026-03-03*
