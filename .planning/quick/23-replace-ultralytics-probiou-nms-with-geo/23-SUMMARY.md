---
phase: quick-23
plan: 01
subsystem: detection
tags: [shapely, nms, polygon-iou, yolo-obb]

requires:
  - phase: none
    provides: n/a
provides:
  - Geometric polygon NMS replacing probiou for OBB detection deduplication
  - Configurable iou_threshold on DetectionConfig (YAML: detection.iou_threshold)
affects: [detection, pipeline-config]

tech-stack:
  added: [shapely>=2.0]
  patterns: [geometric-polygon-nms]

key-files:
  created: []
  modified:
    - src/aquapose/core/detection/backends/yolo_obb.py
    - src/aquapose/engine/config.py
    - src/aquapose/engine/pipeline.py
    - tests/unit/core/detection/test_detection_stage.py
    - pyproject.toml

key-decisions:
  - "Hardcode iou=0.95 in YOLO predict to disable internal probiou NMS rather than removing the parameter entirely"
  - "polygon_nms is a module-level function (not a method) for testability and reuse"

patterns-established:
  - "Geometric NMS: use Shapely polygon intersection for exact IoU on oriented bounding boxes"

requirements-completed: [QUICK-23]

duration: 3min
completed: 2026-03-07
---

# Quick Task 23: Replace Ultralytics Probiou NMS with Geometric Polygon NMS Summary

**Shapely-based geometric polygon NMS replacing probiou Gaussian approximation for accurate OBB deduplication on elongated fish**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T18:33:30Z
- **Completed:** 2026-03-07T18:36:58Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Implemented `polygon_nms()` using Shapely exact polygon intersection/union for IoU
- Disabled YOLO's internal probiou NMS (hardcoded iou=0.95) so all filtering uses geometric NMS
- Added `iou_threshold` field to `DetectionConfig` (default 0.45), wired through `build_stages`
- Added 5 test cases covering empty, single, overlapping, non-overlapping, and chain suppression

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Add failing tests for polygon_nms** - `3191c41` (test)
2. **Task 1 (GREEN): Implement geometric polygon NMS** - `316cb95` (feat)
3. **Task 2: Add iou_threshold to DetectionConfig and wire through build_stages** - `8b2ce6e` (feat)

## Files Created/Modified
- `pyproject.toml` - Added shapely>=2.0 dependency
- `src/aquapose/core/detection/backends/yolo_obb.py` - Added `polygon_nms()`, hardcoded iou=0.95, applied NMS in `_parse_results`
- `src/aquapose/engine/config.py` - Added `iou_threshold: float = 0.45` to DetectionConfig
- `src/aquapose/engine/pipeline.py` - Passes `iou_threshold` to DetectionStage in `build_stages`
- `tests/unit/core/detection/test_detection_stage.py` - 5 new tests for `polygon_nms`

## Decisions Made
- Hardcoded `iou=0.95` in YOLO predict calls rather than removing the parameter, so YOLO's internal NMS still prevents exact duplicates while deferring real suppression to geometric NMS
- Implemented `polygon_nms` as a module-level function for testability and potential reuse outside the backend

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

---
*Quick Task: 23-replace-ultralytics-probiou-nms-with-geo*
*Completed: 2026-03-07*
