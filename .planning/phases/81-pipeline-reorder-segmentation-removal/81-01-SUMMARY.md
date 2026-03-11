---
phase: 81-pipeline-reorder-segmentation-removal
plan: 01
subsystem: pipeline
tags: [pose-estimation, detection, pipeline, refactor, keypoints]

# Dependency graph
requires:
  - phase: 78.1-obb-pose-production-retrain
    provides: pose model that outputs 6 anatomical keypoints
  - phase: 80-baseline-metrics
    provides: OC-SORT baseline, established pipeline structure to reorder
provides:
  - PoseStage class in core/pose/ (formerly core/midline/)
  - Detection.keypoints and Detection.keypoint_conf fields
  - v3.7 pipeline order: Detection -> Pose -> Tracking -> Association -> Reconstruction
  - PipelineContext without annotated_detections
  - PoseConfig (renamed from MidlineConfig), config.pose key
affects:
  - 82-keypoint-cost-association
  - 83-keypoint-state-kalman
  - reconstruction-stage (reads tracklet_groups, no longer annotated_detections)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - PoseStage enriches Detection objects in-place (no wrapper type, no new context field)
    - process_batch returns (kpts_xy, kpts_conf) tuples instead of AnnotatedDetection objects
    - Stage ordering: pose runs before tracking so keypoints are available for OKS cost

key-files:
  created:
    - src/aquapose/core/pose/stage.py
    - src/aquapose/core/pose/__init__.py
    - src/aquapose/core/pose/backends/pose_estimation.py
    - src/aquapose/core/pose/backends/segmentation.py
    - src/aquapose/core/pose/backends/__init__.py
    - src/aquapose/core/pose/types.py
    - src/aquapose/core/pose/crop.py
    - src/aquapose/core/pose/midline.py
    - src/aquapose/core/pose/orientation.py
    - tests/unit/core/pose/__init__.py
    - tests/unit/core/pose/test_pose_stage.py
  modified:
    - src/aquapose/core/types/detection.py
    - src/aquapose/core/context.py
    - src/aquapose/core/__init__.py
    - src/aquapose/engine/config.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/cli.py
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/evaluation/runner.py
    - src/aquapose/evaluation/tuning.py
    - src/aquapose/training/pseudo_label_cli.py
    - 20+ test files updated for new imports/APIs

key-decisions:
  - "PoseStage writes raw 6-keypoint data in-place on Detection objects — no AnnotatedDetection wrapper, no annotated_detections context field"
  - "process_batch() returns (kpts_xy, kpts_conf) tuples — PoseStage handles back-projection to full-frame coordinates"
  - "Segmentation backend kept in core/pose/backends/ but removed from backend registry (only pose_estimation registered)"
  - "YAML 'midline' key accepted as deprecated alias for 'pose' for backward compatibility"
  - "ReconstructionStage fallback path removed — now raises ValueError if tracklet_groups is None"

patterns-established:
  - "In-place enrichment pattern: stages write data onto existing objects rather than creating new context fields where possible"
  - "process_batch API: backend returns list of (kpts_xy, kpts_conf) | (None, None) tuples; stage handles projection"

requirements-completed: [PIPE-01, PIPE-03]

# Metrics
duration: ~90min
completed: 2026-03-11
---

# Phase 81 Plan 01: Pipeline Reorder — Segmentation Removal Summary

**core/midline renamed to core/pose with PoseStage rewritten to enrich Detection objects in-place with raw 6-keypoint data, pipeline reordered to Detection->Pose->Tracking->Association->Reconstruction, and annotated_detections removed from PipelineContext**

## Performance

- **Duration:** ~90 min
- **Started:** 2026-03-10T~23:30:00Z
- **Completed:** 2026-03-11T~01:15:00Z
- **Tasks:** 2
- **Files modified:** ~35

## Accomplishments
- Renamed `core/midline/` to `core/pose/` with full git history preservation
- Rewrote `PoseStage.run()` to write `det.keypoints` and `det.keypoint_conf` in-place on Detection objects (no AnnotatedDetection wrapper, no 6→15 upsampling)
- Removed `annotated_detections` from `PipelineContext` and cascaded that removal through reconstruction/stage, evaluation/runner, evaluation/tuning, and 20+ test files
- Reordered `build_stages()` to Detection->Pose->Tracking->Association->Reconstruction (PoseStage now at index 1)
- Renamed `MidlineConfig` to `PoseConfig`, `config.midline` to `config.pose` throughout engine and CLI

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename core/midline to core/pose, add Detection keypoints fields, rewrite PoseStage** - `f87f395` (feat)
2. **Task 2: Pipeline/context/config/CLI update for v3.7 stage order** - `3e899b9` (feat)

## Files Created/Modified
- `src/aquapose/core/pose/stage.py` - PoseStage: in-place keypoint enrichment on Detection objects
- `src/aquapose/core/pose/backends/pose_estimation.py` - process_batch returns (kpts_xy, kpts_conf) tuples
- `src/aquapose/core/types/detection.py` - Added keypoints (K,2) and keypoint_conf (K,) optional fields
- `src/aquapose/core/context.py` - Removed annotated_detections field; updated docstring to v3.7 data flow
- `src/aquapose/engine/pipeline.py` - build_stages() reordered; _STAGE_OUTPUT_FIELDS updated; PoseStage at index 1
- `src/aquapose/engine/config.py` - MidlineConfig→PoseConfig; midline→pose; removed segmentation-only fields
- `src/aquapose/cli.py` - stop_after choices: "pose" replaces "midline"
- `src/aquapose/core/reconstruction/stage.py` - Removed annotated_detections fallback; now raises ValueError if tracklet_groups missing
- `src/aquapose/evaluation/runner.py` - Removed annotated_detections merge; midline stage presence detection removed
- `src/aquapose/evaluation/tuning.py` - getattr guard for annotated_detections (backward compat with old cached contexts)
- `src/aquapose/training/pseudo_label_cli.py` - pipeline_config.midline → pipeline_config.pose
- `tests/unit/core/pose/test_pose_stage.py` - New test: protocol conformance, in-place keypoint enrichment, batched inference path
- `tests/unit/core/midline/test_pose_estimation_backend.py` - Rewritten for process_batch API
- `tests/unit/core/midline/test_direct_pose_backend.py` - Rewritten for process_batch API
- `tests/unit/core/midline/test_segmentation_backend.py` - Import paths updated to core.pose.*
- 15+ additional test files updated for import path changes and API removals

## Decisions Made
- PoseStage enriches Detection objects in-place rather than creating AnnotatedDetection wrappers — simplifies downstream stages that now read `det.keypoints` directly
- `process_batch()` return type changed to `list[tuple[ndarray|None, ndarray|None]]` — PoseStage handles full-frame back-projection via `invert_affine_points`
- Segmentation backend retained in codebase but removed from backend registry (not user-facing)
- YAML backward compat: `"midline"` config key maps to `"pose"` (no breaking change for existing project configs)
- ReconstructionStage now hard-fails if `tracklet_groups` is None rather than silently falling back to `annotated_detections`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Updated 20+ test files for import path changes and API removals**
- **Found during:** Task 2 (test suite run after source changes)
- **Issue:** 7 test collection errors from `ModuleNotFoundError: No module named 'aquapose.core.midline'`; additional test failures from removed `annotated_detections` field, removed `process_frame` API, removed `MidlineConfig`
- **Fix:** Updated all imports from `aquapose.core.midline.*` to `aquapose.core.pose.*`; rewrote `test_pose_estimation_backend.py` and `test_direct_pose_backend.py` for `process_batch` API; removed `annotated_detections` assertions from context tests; updated `MidlineConfig`→`PoseConfig` in config tests; updated stage ordering indices in pipeline/build_stages tests
- **Files modified:** 20+ test files across tests/unit/core/, tests/unit/engine/, tests/unit/evaluation/, tests/unit/
- **Verification:** `hatch run test` passes: 1141 passed, 3 skipped, 14 deselected, 20 warnings
- **Committed in:** 3e899b9 (Task 2 commit)

**2. [Rule 1 - Bug] Removed stale annotated_detections fallback in ReconstructionStage**
- **Found during:** Task 2 (source file audit during test fixes)
- **Issue:** `reconstruction/stage.py` line 130 fell back to `context.annotated_detections` when `tracklet_groups` was None — this would crash with AttributeError since `annotated_detections` was removed from PipelineContext
- **Fix:** Replaced fallback with explicit ValueError("ReconstructionStage requires context.tracklet_groups...")
- **Files modified:** src/aquapose/core/reconstruction/stage.py
- **Verification:** `test_run_raises_if_tracklet_groups_missing` passes
- **Committed in:** 3e899b9 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 missing critical test updates, 1 bug fix)
**Impact on plan:** Both auto-fixes required for correctness. No scope creep — all fixes directly caused by the plan's changes.

## Issues Encountered
- Pre-commit hook (ruff format/lint) modified staged files on first commit attempt for Task 2; re-staged and recommitted successfully.

## Next Phase Readiness
- `det.keypoints` (shape K,2) and `det.keypoint_conf` (shape K,) available on every Detection after PoseStage runs
- Phase 82 (keypoint-cost-association) can use `det.keypoints` for OKS-based cost in association
- Phase 83 (keypoint-state-kalman) can extend KF state with keypoint positions read from detections
- No blockers

---
*Phase: 81-pipeline-reorder-segmentation-removal*
*Completed: 2026-03-11*
