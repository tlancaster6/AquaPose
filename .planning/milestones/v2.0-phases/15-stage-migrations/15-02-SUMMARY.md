---
phase: 15-stage-migrations
plan: 02
subsystem: core
tags: [stage-protocol, u-net, skeletonization, bfs, midline, pipeline]

# Dependency graph
requires:
  - phase: 15-01
    provides: DetectionStage in core/detection/ and PipelineContext.detections (Stage 1)
  - phase: 13-engine-core
    provides: Stage Protocol, PipelineContext, engine contracts (ENG-07)
provides:
  - MidlineStage in core/midline/stage.py satisfying Stage Protocol
  - segment_then_extract backend combining U-Net segmentation + skeleton+BFS midline extraction
  - direct_pose stub backend proving registry pattern (raises NotImplementedError)
  - AnnotatedDetection type bundling Detection with mask, crop_region, and Midline2D
  - Backend registry get_backend() factory in core/midline/backends/__init__.py
  - MidlineConfig extended with backend, n_points, min_area fields
affects: [15-03-association, 15-04-tracking, 15-05-reconstruction, phase-16-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Per-detection midline extraction (not per-track as in v1.0) — all detections annotated before tracking
    - fish_id placeholder -1 at midline stage — tracking has not yet assigned IDs
    - Eager model loading in backend constructor — fail-fast on missing weights path
    - TYPE_CHECKING guard for PipelineContext annotation — engine/ never imported at runtime (ENG-07)

key-files:
  created:
    - src/aquapose/core/midline/__init__.py
    - src/aquapose/core/midline/types.py
    - src/aquapose/core/midline/stage.py
    - src/aquapose/core/midline/backends/__init__.py
    - src/aquapose/core/midline/backends/segment_then_extract.py
    - src/aquapose/core/midline/backends/direct_pose.py
    - tests/unit/core/midline/__init__.py
    - tests/unit/core/midline/test_midline_stage.py
  modified:
    - src/aquapose/engine/config.py (MidlineConfig: added backend, n_points, min_area fields)

key-decisions:
  - "MidlineStage extracts midlines for ALL detections (not just tracked fish) — tracking has not run yet; fish_id placeholder -1 used"
  - "segment_then_extract backend directly calls private helpers from reconstruction.midline (_adaptive_smooth, _check_skip_mask, etc.) — avoids reimplementing the midline pipeline"
  - "AnnotatedDetection wraps Detection + mask + CropRegion + Midline2D — clean wrapper type rather than mutating Detection in place"
  - "MidlineConfig gains backend, n_points, min_area — keeps all midline stage parameters in one frozen config"

patterns-established:
  - "Backend pattern: get_backend() factory with kind string, supporting both implemented and stub backends"
  - "Stage stub pattern: DirectPoseBackend raises NotImplementedError at construction to prove registry extensibility"

requirements-completed: [STG-02]

# Metrics
duration: 9min
completed: 2026-02-26
---

# Phase 15 Plan 02: Midline Stage Migration Summary

**MidlineStage (Stage 2) migrated to core/midline/ with segment-then-extract backend: U-Net crops, adaptive smoothing, skeletonization + BFS, arc-length resample — annotates all detections with 2D midlines before tracking**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-26T06:23:29Z
- **Completed:** 2026-02-26T06:32:58Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- MidlineStage class satisfies Stage Protocol via structural typing — `isinstance(stage, Stage)` returns True
- Segment-then-extract backend fully implements v1.0 midline pipeline: crop regions, U-Net segmentation, adaptive smooth, skeletonize, BFS longest path, arc-length resample, crop-to-frame transform
- Direct pose backend stub proves registry pattern (raises NotImplementedError with clear message)
- All 10 interface tests pass covering: protocol conformance, context wiring, backend registry, import boundary, fail-fast on missing weights

## Task Commits

1. **Task 1: Create core/midline/ module with segment-then-extract backend and stage** - `07e9150` (feat)
2. **Task 2: Interface tests for MidlineStage** - `a3b74b8` (test)

**Plan metadata:** (created with final docs commit)

## Files Created/Modified
- `src/aquapose/core/midline/__init__.py` - Package exports: MidlineStage, Midline2D, AnnotatedDetection
- `src/aquapose/core/midline/types.py` - AnnotatedDetection dataclass wrapping Detection + mask + crop_region + Midline2D
- `src/aquapose/core/midline/stage.py` - MidlineStage: reads context.detections, opens VideoSet, calls backend.process_frame, writes context.annotated_detections
- `src/aquapose/core/midline/backends/__init__.py` - Backend registry: get_backend() factory for "segment_then_extract" and "direct_pose"
- `src/aquapose/core/midline/backends/segment_then_extract.py` - Full U-Net + midline pipeline; annotates all detections (not just tracked fish)
- `src/aquapose/core/midline/backends/direct_pose.py` - Stub: constructor and process_frame raise NotImplementedError
- `src/aquapose/engine/config.py` - MidlineConfig gains backend (str), n_points (int=15), min_area (int=300) fields
- `tests/unit/core/midline/__init__.py` - Test package init
- `tests/unit/core/midline/test_midline_stage.py` - 10 interface tests with mock-based approach

## Decisions Made
- **Per-detection midline extraction**: Unlike v1.0 which extracted midlines only for tracked fish, MidlineStage annotates ALL detections. Tracking hasn't assigned fish IDs yet at Stage 2; `fish_id=-1` is used as placeholder in Midline2D.
- **Segment-then-extract backend reuses reconstruction.midline private helpers**: `_adaptive_smooth`, `_check_skip_mask`, `_crop_to_frame`, `_longest_path_bfs`, `_resample_arc_length`, `_skeleton_and_widths` are imported directly from `aquapose.reconstruction.midline` to avoid reimplementing the midline pipeline.
- **AnnotatedDetection wrapper type**: Rather than mutating Detection objects in place, AnnotatedDetection wraps the original Detection with optional mask, crop_region, and midline fields. Midline/mask may be None when extraction fails.
- **MidlineConfig extended with backend, n_points, min_area**: All MidlineStage parameters captured in config for serialization and CLI override support.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Ruff RUF043 lint error: regex pattern string `"nonexistent_weights.pth"` needed to be raw (`r"nonexistent_weights\.pth"`) for the `.` metacharacter. Fixed immediately, second commit succeeded.
- Pre-existing basedpyright type errors (7) in `reconstruction/midline.py`, `segmentation/detector.py`, and `visualization/plot3d.py` — all pre-existing, out-of-scope, not caused by this plan's changes.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MidlineStage satisfies Stage Protocol — ready for integration into PosePipeline orchestrator
- context.annotated_detections populated for Stage 3 (Association)
- AnnotatedDetection carries midline points for cross-camera clustering in the Association stage

---
*Phase: 15-stage-migrations*
*Completed: 2026-02-26*
