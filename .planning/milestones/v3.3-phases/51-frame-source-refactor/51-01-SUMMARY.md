---
phase: 51-frame-source-refactor
plan: 01
subsystem: core
tags: [frame-source, protocol, video-io, refactor, tdd]

requires:
  - phase: none
    provides: existing VideoSet in io/video.py plus stage constructors with direct video I/O

provides:
  - FrameSource runtime-checkable Protocol in core/types/frame_source.py
  - VideoFrameSource concrete implementation (video discovery, calibration, undistortion, max_frames)
  - DetectionStage and MidlineStage accepting FrameSource via constructor injection
  - build_stages() creating one VideoFrameSource shared by both stages

affects:
  - 51-02-PLAN (observers that still import VideoSet)
  - 52-chunk-mode (will add ChunkFrameSource implementing FrameSource)

tech-stack:
  added: []
  patterns:
    - "FrameSource protocol: injectable multi-camera frame provider (runtime_checkable)"
    - "VideoFrameSource: concrete protocol implementation absorbing video discovery + calibration loading"
    - "Stage constructor injection: stages receive frame_source, not video_dir"

key-files:
  created:
    - src/aquapose/core/types/frame_source.py
    - tests/unit/core/types/test_frame_source.py
    - tests/unit/core/types/__init__.py
  modified:
    - src/aquapose/core/types/__init__.py
    - src/aquapose/core/detection/stage.py
    - src/aquapose/core/midline/stage.py
    - src/aquapose/engine/pipeline.py
    - tests/unit/core/detection/test_detection_stage.py
    - tests/unit/core/midline/test_midline_stage.py
    - tests/unit/engine/test_build_stages.py
    - tests/unit/engine/test_pipeline.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py

key-decisions:
  - "VideoSet retained in io/video.py — observers still import it; Plan 02 handles removal"
  - "MidlineStage keeps calibration_path param — needed for ForwardLUT loading in orientation resolution"
  - "VideoFrameSource shared by DetectionStage and MidlineStage (single construction in build_stages)"
  - "FrameSource is runtime_checkable Protocol — enables isinstance checks without inheritance"

patterns-established:
  - "FrameSource protocol: any multi-camera frame provider implements camera_ids, __len__, __iter__, __enter__/__exit__, read_frame"
  - "Stage injection: stages receive pre-built frame_source, never open files internally"
  - "build_stages() owns VideoFrameSource construction and injects it to relevant stages"

requirements-completed:
  - FRAME-01
  - FRAME-02

duration: 10min
completed: 2026-03-03
---

# Phase 51 Plan 01: Frame Source Refactor Summary

**FrameSource protocol + VideoFrameSource implementation that decouples DetectionStage and MidlineStage from video I/O, injected via build_stages()**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-03T22:47:34Z
- **Completed:** 2026-03-03T22:57:17Z
- **Tasks:** 2
- **Files modified:** 9 modified + 3 created

## Accomplishments

- Defined `FrameSource` as a `runtime_checkable Protocol` in `core/types/frame_source.py` with the full interface (camera_ids, __len__, __iter__, __enter__/__exit__, read_frame, k_new)
- Implemented `VideoFrameSource` absorbing video discovery, calibration loading, undistortion, and max_frames windowing that were previously scattered across stage constructors
- Migrated `DetectionStage` and `MidlineStage` constructors to accept `frame_source: FrameSource` — neither stage opens video files internally
- Updated `build_stages()` to construct one `VideoFrameSource` and share it with both stages; `stop_frame` config becomes `max_frames`

## Task Commits

Each task was committed atomically:

1. **Task 1: RED - FrameSource protocol tests** - `7899062` (test)
2. **Task 1: GREEN - FrameSource protocol implementation** - `b89eb4e` (feat)
3. **Task 2: Migrate stages + build_stages** - `1357b68` (feat)

## Files Created/Modified

- `src/aquapose/core/types/frame_source.py` - FrameSource protocol + VideoFrameSource concrete class
- `src/aquapose/core/types/__init__.py` - exports FrameSource and VideoFrameSource
- `src/aquapose/core/detection/stage.py` - constructor now accepts `frame_source: FrameSource`
- `src/aquapose/core/midline/stage.py` - constructor now accepts `frame_source: FrameSource`, keeps calibration_path
- `src/aquapose/engine/pipeline.py` - build_stages() creates VideoFrameSource and injects it
- `tests/unit/core/types/test_frame_source.py` - protocol conformance, max_frames, camera_ids, error handling
- `tests/unit/core/types/__init__.py` - package init for test directory
- `tests/unit/core/detection/test_detection_stage.py` - updated _build_stage helper to use mock FrameSource
- `tests/unit/core/midline/test_midline_stage.py` - updated _build_stage helper to use mock FrameSource
- `tests/unit/engine/test_build_stages.py` - added VideoFrameSource patch to all production-mode tests
- `tests/unit/engine/test_pipeline.py` - added VideoFrameSource patch to build_stages tests
- `tests/unit/core/reconstruction/test_reconstruction_stage.py` - added VideoFrameSource patch

## Decisions Made

- **VideoSet retained**: Observers (overlay, animation, HDF5) still import VideoSet. Plan 02 handles observer migration and VideoSet removal.
- **MidlineStage keeps calibration_path**: The `_apply_orientation` method calls `load_forward_luts(calibration_path, ...)`. Removing it would require a separate ForwardLUT injection, which is beyond this plan's scope.
- **Shared VideoFrameSource**: Both DetectionStage and MidlineStage receive the same VideoFrameSource instance from build_stages(). This matches the prior behavior (same video files, same calibration) and enables future chunk-mode coordination.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test patch targets corrected for module-level imports**
- **Found during:** Task 1 (FrameSource tests, GREEN phase)
- **Issue:** Tests initially patched `aquapose.calibration.loader.load_calibration_data` but `load_calibration_data` is imported at module level in `frame_source.py`, requiring patch at `aquapose.core.types.frame_source.load_calibration_data`
- **Fix:** Corrected patch targets; also moved `load_calibration_data` to module-level import in `frame_source.py` for consistency with other calibration imports
- **Files modified:** tests/unit/core/types/test_frame_source.py, src/aquapose/core/types/frame_source.py
- **Committed in:** b89eb4e (Task 1 commit)

**2. [Rule 1 - Bug] Test mock missing numpy arrays for UndistortionMaps**
- **Found during:** Task 1 (FrameSource tests, GREEN phase)
- **Issue:** `undistort_image` calls `cv2.remap(image, undistortion.map_x, undistortion.map_y, ...)` — mock with `MagicMock()` attributes caused OpenCV error
- **Fix:** Added proper numpy float32 arrays (zeros) to mock undistortion maps in `_build_vfs` helper
- **Files modified:** tests/unit/core/types/test_frame_source.py
- **Committed in:** b89eb4e (Task 1 commit)

**3. [Rule 1 - Bug] Multiple tests across codebase needed VideoFrameSource patched**
- **Found during:** Task 2 (after migrating build_stages)
- **Issue:** Several existing tests call `build_stages()` with fake paths — they patched stage constructors but not `VideoFrameSource.__init__`, which now runs before stage construction
- **Fix:** Added `VideoFrameSource.__init__` patches to test_build_stages.py, test_pipeline.py, and test_reconstruction_stage.py
- **Files modified:** tests/unit/engine/test_build_stages.py, tests/unit/engine/test_pipeline.py, tests/unit/core/reconstruction/test_reconstruction_stage.py
- **Committed in:** 1357b68 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1 bugs)
**Impact on plan:** All auto-fixes were necessary for tests to pass. No scope creep — corrections to test infrastructure only.

## Issues Encountered

None — all issues handled via deviation rules above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- FrameSource protocol established; Phase 52 can implement `ChunkFrameSource` without touching stages
- VideoSet still in `io/video.py` for observers — Plan 02 (observer migration) must run before VideoSet can be removed
- All tests pass (799 passed, 3 skipped) and lint is clean

---
*Phase: 51-frame-source-refactor*
*Completed: 2026-03-03*

## Self-Check: PASSED

All artifacts verified:
- FOUND: src/aquapose/core/types/frame_source.py
- FOUND: src/aquapose/core/detection/stage.py (updated)
- FOUND: src/aquapose/core/midline/stage.py (updated)
- FOUND: src/aquapose/engine/pipeline.py (updated)
- FOUND: .planning/phases/51-frame-source-refactor/51-01-SUMMARY.md
- FOUND commit: 7899062 (test: FrameSource failing tests)
- FOUND commit: b89eb4e (feat: FrameSource implementation)
- FOUND commit: 1357b68 (feat: migrate stages + build_stages)
