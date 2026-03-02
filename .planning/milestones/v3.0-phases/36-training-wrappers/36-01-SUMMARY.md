---
phase: 36-training-wrappers
plan: 01
subsystem: training
tags: [yolo, coco, segmentation, ndjson, data-pipeline]

# Dependency graph
requires:
  - phase: 35-codebase-cleanup
    provides: clean codebase with stripped old backends
provides:
  - "--mode seg in build_yolo_training_data.py converts COCO segmentation polygons to per-crop NDJSON"
  - "TestFormatSegAnnotation and TestSegConverter test suite covering polygon transform and multi-fish labeling"
affects: [36-training-wrappers plan 02 (yolo_seg training wrapper consumes seg NDJSON)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-detection-crop NDJSON format: one JSON record per crop, annotations list all visible fish"
    - "Multi-ring COCO polygon: keep largest ring by flat-list length"
    - "Polygon affine transform via homogeneous coords: affine_mat @ [x,y,1].T"
    - "Seg polygon in-bounds check: at least 3 vertices within [0,crop_w) x [0,crop_h)"

key-files:
  created: []
  modified:
    - scripts/build_yolo_training_data.py
    - tests/unit/test_build_yolo_training_data.py

key-decisions:
  - "format_seg_annotation normalizes polygon vertices to [0,1] by dividing by crop dimensions and clipping"
  - "generate_seg_dataset uses same affine crop pipeline as generate_pose_dataset; OBB defined from keypoints, polygons from segmentation field"
  - "Annotations without segmentation field are silently skipped (no error); their keypoints can still define an OBB"
  - "val_split=0.0 edge case: max(1, int(n*0.0)) = 1 means at least 1 crop goes to val when n > 1"

patterns-established:
  - "Seg mode: target annotation keypoints -> OBB crop -> transform ALL fish polygons in image into crop space"
  - "Polygon filtering: skip if fewer than 3 vertices land in crop bounds after affine transform"

requirements-completed: [DATA-01]

# Metrics
duration: 25min
completed: 2026-03-01
---

# Phase 36 Plan 01: Training Wrappers — Seg Data Converter Summary

**COCO segmentation polygon-to-per-crop NDJSON converter (`--mode seg`) added to `build_yolo_training_data.py` with full `TestSegConverter` and `TestFormatSegAnnotation` test suites**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-01T20:50:00Z
- **Completed:** 2026-03-01T21:15:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Task 1 (`b0fd4fc`): `--mode seg` CLI flag, `format_seg_annotation()`, and `generate_seg_dataset()` added to `build_yolo_training_data.py` (was already committed from a prior session)
- Task 2 (`8d0d782`): `TestFormatSegAnnotation` (4 tests) and `TestSegConverter` (5 integration tests) added covering polygon normalization, multi-ring selection, affine transform correctness, multi-fish labeling, and missing-segmentation skipping

## Task Commits

Each task was committed atomically:

1. **Task 1: Add --mode flag and generate_seg_dataset function** - `b0fd4fc` (feat)
2. **Task 2: Add TestSegConverter tests** - `8d0d782` (feat)

## Files Created/Modified

- `scripts/build_yolo_training_data.py` - Added `--mode seg`, `format_seg_annotation()`, `generate_seg_dataset()`; produces `seg/images/{train,val}/`, `seg/{train,val}.ndjson`, `seg/data.yaml`
- `tests/unit/test_build_yolo_training_data.py` - Added `format_seg_annotation`, `generate_seg_dataset` imports; `TestFormatSegAnnotation` class (4 tests); `TestSegConverter` class (5 integration tests)

## Decisions Made

- `val_split=0.0` edge case acknowledged: the code correctly uses `max(1, int(n * val_split))` which forces at least 1 crop to val when there are multiple crops. Tests check both train+val NDJSON files for total crop counts.
- Polygon clipping uses `crop_w - 1e-6` as upper bound (matching existing pose crop clipping pattern) to avoid floating-point boundary issues.

## Deviations from Plan

None — plan executed exactly as written. Task 1 was already committed from a prior session (`b0fd4fc`); Task 2 was the only new work needed.

## Issues Encountered

- `test_all_fish_in_crop_labeled` and `test_missing_segmentation_skipped` initially used `val_split=0.0` and only checked `train.ndjson`, but the `generate_seg_dataset` split logic always sends at least 1 crop to val when there are multiple crops. Fixed by checking both train and val NDJSON files for total crop count assertions.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- DATA-01 satisfied: `build_yolo_training_data.py --mode seg` produces NDJSON-format seg datasets matching the pose/OBB pattern
- Plan 36-02 (yolo_seg training wrapper) can consume `seg/data.yaml` and `seg/{train,val}.ndjson` directly
- Pre-existing `test_pipeline.py` failures (`test_pipeline_writes_config_artifact`, `test_config_artifact_written_before_stages`) are out of scope for this plan

---
*Phase: 36-training-wrappers*
*Completed: 2026-03-01*
