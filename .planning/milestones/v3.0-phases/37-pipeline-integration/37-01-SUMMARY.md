---
phase: 37-pipeline-integration
plan: 01
subsystem: core
tags: [midline, backends, config, pipeline, yolo-seg, yolo-pose, refactor]

# Dependency graph
requires:
  - phase: 35-codebase-cleanup
    provides: Midline backend stubs (segment_then_extract, direct_pose) as insertion points
  - phase: 36-training-wrappers
    provides: YOLO-seg and YOLO-pose training wrappers for future backend integration
provides:
  - Renamed midline backends: "segmentation" and "pose_estimation"
  - Updated backend registry resolving new names
  - MidlineConfig.backend defaults to "segmentation"; old names raise ValueError
  - MidlineConfig.keypoint_confidence_floor updated to 0.3
  - SegmentationBackend and PoseEstimationBackend with explicit constructor kwargs
affects: [37-02, 37-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Backend stubs store constructor kwargs as instance attributes for Plan 02 wiring
    - Backend kind validation at MidlineConfig construction time (not at runtime)

key-files:
  created:
    - src/aquapose/core/midline/backends/segmentation.py
    - src/aquapose/core/midline/backends/pose_estimation.py
  modified:
    - src/aquapose/core/midline/backends/__init__.py
    - src/aquapose/core/midline/stage.py
    - src/aquapose/core/midline/__init__.py
    - src/aquapose/engine/config.py
    - tests/unit/core/midline/test_midline_stage.py
    - tests/unit/core/midline/test_direct_pose_backend.py
    - tests/unit/engine/test_config.py
    - tests/e2e/test_smoke.py
  deleted:
    - src/aquapose/core/midline/backends/segment_then_extract.py
    - src/aquapose/core/midline/backends/direct_pose.py

key-decisions:
  - "Backend names changed from segment_then_extract/direct_pose to segmentation/pose_estimation"
  - "MidlineConfig.keypoint_confidence_floor default raised from 0.1 to 0.3 per CONTEXT.md"
  - "New backend stubs accept explicit kwargs (not **kwargs) and store them as instance attributes for Plan 02"

patterns-established:
  - "Backend registry: get_backend(kind, **kwargs) returns configured instance; unknown kind raises ValueError"
  - "MidlineStage branching: if backend == 'pose_estimation' gets special kwargs from midline_config"

requirements-completed: [PIPE-03]

# Metrics
duration: 8min
completed: 2026-03-01
---

# Phase 37 Plan 01: Pipeline Integration Summary

**Renamed midline backends from segment_then_extract/direct_pose to segmentation/pose_estimation across config, registry, stage, and all tests**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-01T22:29:54Z
- **Completed:** 2026-03-01T22:37:55Z
- **Tasks:** 2
- **Files modified:** 10 (2 created, 2 deleted, 8 updated)

## Accomplishments
- Renamed backend files and classes: `SegmentThenExtractBackend` -> `SegmentationBackend`, `DirectPoseBackend` -> `PoseEstimationBackend`
- Updated `MidlineConfig._valid_backends` to `{"segmentation", "pose_estimation"}` — old names now raise `ValueError`
- Updated `MidlineConfig.backend` default from `"segment_then_extract"` to `"segmentation"`
- Updated `MidlineConfig.keypoint_confidence_floor` default from `0.1` to `0.3`
- New stubs accept explicit constructor kwargs and store as instance attributes for Plan 02
- All tests updated; 638 pass (3 pre-existing failures unrelated to this plan)

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename backend files, update registry, config, stage, and pipeline** - `cf8a958` (refactor)
2. **Task 2: Update all tests referencing old backend names** - `d6c26fe` (test)

## Files Created/Modified

- `src/aquapose/core/midline/backends/segmentation.py` - SegmentationBackend (no-op stub, stores kwargs for Plan 02)
- `src/aquapose/core/midline/backends/pose_estimation.py` - PoseEstimationBackend (no-op stub, stores kwargs for Plan 02)
- `src/aquapose/core/midline/backends/__init__.py` - Registry updated to new names
- `src/aquapose/core/midline/stage.py` - Default backend and branch condition updated
- `src/aquapose/core/midline/__init__.py` - Now exports SegmentationBackend and PoseEstimationBackend
- `src/aquapose/engine/config.py` - MidlineConfig validation and defaults updated
- `tests/unit/core/midline/test_midline_stage.py` - Module list and import updated
- `tests/unit/core/midline/test_direct_pose_backend.py` - Rewritten for PoseEstimationBackend
- `tests/unit/engine/test_config.py` - 6 new tests for backend validation and defaults
- `tests/e2e/test_smoke.py` - backend="segment_then_extract" fixed to "segmentation"

## Decisions Made
- New backend stubs accept explicit kwargs (not `**kwargs`) and store them as instance attributes so Plan 02 can use them directly without reconstructing the object
- `keypoint_confidence_floor` default raised to 0.3 per CONTEXT.md decision (higher confidence threshold for cleaner pose keypoints)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_direct_pose_backend.py (missing from plan's file list)**
- **Found during:** Task 2 (test suite run)
- **Issue:** `test_direct_pose_backend.py` imported from `direct_pose` module (deleted) — caused collection error
- **Fix:** Rewrote file to use `PoseEstimationBackend` with updated test names
- **Files modified:** `tests/unit/core/midline/test_direct_pose_backend.py`
- **Verification:** Tests pass after update
- **Committed in:** d6c26fe (Task 2 commit)

**2. [Rule 1 - Bug] Fixed e2e smoke test using old backend name**
- **Found during:** Task 2 (grep for remaining old names)
- **Issue:** `tests/e2e/test_smoke.py` line 219 specified `backend="segment_then_extract"` which now raises ValueError
- **Fix:** Updated to `backend="segmentation"`
- **Files modified:** `tests/e2e/test_smoke.py`
- **Verification:** No ValueError from config construction in smoke test
- **Committed in:** d6c26fe (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 - Bugs)
**Impact on plan:** Both fixes necessary for correctness — plan's file list missed test_direct_pose_backend.py, and e2e smoke test had stale backend name. No scope creep.

## Issues Encountered
- 3 pre-existing test failures found (unrelated to this plan): `test_pipeline_writes_config_artifact`, `test_config_artifact_written_before_stages`, and `TestIntegrationPipeline::test_obb_dataset_structure`. Verified pre-existing via git stash. Not fixed per scope boundary rule.

## Next Phase Readiness
- Backend naming is consistent across the entire codebase
- Plan 02 can now implement real YOLO-seg inference into `SegmentationBackend` using the stored kwargs
- Plan 02 can implement real YOLO-pose inference into `PoseEstimationBackend` using the stored kwargs

---
*Phase: 37-pipeline-integration*
*Completed: 2026-03-01*
