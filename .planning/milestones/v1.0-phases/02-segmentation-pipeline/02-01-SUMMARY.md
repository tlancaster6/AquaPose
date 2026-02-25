---
phase: 02-segmentation-pipeline
plan: 01
subsystem: segmentation
tags: [cleanup, label-studio, coco, pseudo-labels, pycocotools]

# Dependency graph
requires: []
provides:
  - Clean segmentation module with no Label Studio references
  - to_coco_dataset() function in pseudo_labeler.py for COCO JSON output
  - Lean scripts/ directory with only production-relevant scripts
affects: [02-segmentation-pipeline, 03-fish-mesh-model-and-3d-initialization]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "to_coco_dataset moved to pseudo_labeler.py — COCO conversion co-located with pseudo-label generation"
    - "Integration test updated to test YOLO->SAM2->COCO pipeline instead of YOLO->SAM2->LabelStudio"

key-files:
  created: []
  modified:
    - src/aquapose/segmentation/__init__.py
    - src/aquapose/segmentation/pseudo_labeler.py
    - pyproject.toml
    - tests/unit/segmentation/test_pseudo_labeler.py
    - tests/integration/segmentation/test_yolo_sam_integration.py

key-decisions:
  - "to_coco_dataset lives in pseudo_labeler.py — COCO conversion is a natural output of the pseudo-labeling stage, not a Label Studio concern"
  - "Integration test now tests YOLO->SAM2->COCO export pipeline end-to-end"
  - "test_bbox_conversion_to_sam2_format updated to use crop-relative coordinates (pre-existing test was written for non-crop predict() version)"

patterns-established:
  - "No Label Studio dependencies anywhere in the segmentation module"
  - "Scripts directory contains only production-relevant scripts (no debug/test/exploration scripts)"

requirements-completed: [SEG-03]

# Metrics
duration: 20min
completed: 2026-02-20
---

# Phase 02 Plan 01: Codebase Cleanup Summary

**Removed Label Studio module and 10 debug scripts; preserved to_coco_dataset in pseudo_labeler.py with 182 tests passing**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-02-20T23:05:00Z
- **Completed:** 2026-02-20T23:25:57Z
- **Tasks:** 2
- **Files modified:** 7 (5 modified, 3 deleted, 10 filesystem-only deletions)

## Accomplishments

- Deleted `label_studio.py` and `test_label_studio.py` — Label Studio is fully removed from the segmentation module
- Moved `to_coco_dataset()` into `pseudo_labeler.py` so COCO output is co-located with pseudo-label generation
- Removed `label-studio-converter` from `pyproject.toml` dependencies
- Deleted 10 debug/exploration scripts from `scripts/` — only production scripts remain
- All 182 unit and integration tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete Label Studio module and clean dependencies** - `86529f5` (feat)
2. **Task 2: Delete debug and exploration scripts** - `d366610` (chore)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/aquapose/segmentation/__init__.py` - Updated to import `to_coco_dataset` from `.pseudo_labeler` (removed all Label Studio imports)
- `src/aquapose/segmentation/pseudo_labeler.py` - Added `to_coco_dataset()` function (moved from label_studio.py), added `json` import
- `pyproject.toml` - Removed `label-studio-converter` from dependencies
- `tests/unit/segmentation/test_pseudo_labeler.py` - Added `TestToCOCODataset` class (5 tests), `sample_mask` fixture, fixed `test_bbox_conversion_to_sam2_format` to use crop-relative coordinates
- `tests/integration/segmentation/test_yolo_sam_integration.py` - Updated to test YOLO->SAM2->COCO pipeline (replaced Label Studio export test)
- **Deleted:** `src/aquapose/segmentation/label_studio.py`
- **Deleted:** `tests/unit/segmentation/test_label_studio.py`
- **Deleted (filesystem):** `scripts/_debug_mask.py`, `scripts/_test_single.py`, `scripts/diagnose_mog2.py`, `scripts/verify_mog2_recall.py`, `scripts/verify_pseudo_labels.py`, `scripts/visualize_pseudo_labels.py`, `scripts/rerun_sam2_on_images.py`, `scripts/run_pseudo_labels.py`, `scripts/test_mog2.py`, `scripts/test_sam2.py`

## Decisions Made

- `to_coco_dataset` belongs in `pseudo_labeler.py` — it converts AnnotatedFrame lists to COCO JSON, which is the direct output format of the pseudo-labeling stage
- Integration test now validates the end-to-end YOLO->SAM2->COCO export pipeline rather than the Label Studio export pipeline

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed broken integration test importing removed export_to_label_studio**
- **Found during:** Task 1 (Delete Label Studio module)
- **Issue:** `tests/integration/segmentation/test_yolo_sam_integration.py` imported `export_to_label_studio` and tested the Label Studio export pipeline, which is now gone
- **Fix:** Rewrote the third test class (`TestFullPipelineDetectorToExport`) to test the COCO export pipeline instead, updated imports to use `to_coco_dataset` and `AnnotatedFrame`
- **Files modified:** `tests/integration/segmentation/test_yolo_sam_integration.py`
- **Verification:** `hatch run test` — 182 passed
- **Committed in:** `86529f5` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed test_bbox_conversion_to_sam2_format asserting stale full-frame coordinates**
- **Found during:** Task 1 (running tests after Label Studio deletion)
- **Issue:** Test expected `box=[100, 200, 150, 260]` (full-frame absolute coords) but current `pseudo_labeler.py` subtracts the crop origin before passing to SAM2, yielding `[12., 15., 62., 75.]` (crop-relative). The test was written for an older non-crop implementation.
- **Fix:** Updated test to compute the expected crop-relative box using `compute_crop_region()` and assert against that
- **Files modified:** `tests/unit/segmentation/test_pseudo_labeler.py`
- **Verification:** `hatch run test` — 182 passed
- **Committed in:** `86529f5` (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bug fixes in tests)
**Impact on plan:** Both fixes necessary to restore test correctness after removing Label Studio. No scope creep.

## Issues Encountered

- Pre-existing typecheck errors in `detector.py` (4 errors: cv2 normalize signature, YOLO import, optional iterable) — confirmed pre-existing, out of scope per deviation rules

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Codebase is clean: no Label Studio references, no debug scripts
- `to_coco_dataset` importable from `aquapose.segmentation`
- Ready for Plan 02: build_training_data.py consolidation and dataset assembly pipeline
- 182 tests passing, lint clean

---
*Phase: 02-segmentation-pipeline*
*Completed: 2026-02-20*
