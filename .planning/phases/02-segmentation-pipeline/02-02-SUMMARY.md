---
phase: 02-segmentation-pipeline
plan: 02
subsystem: segmentation
tags: [sam2, label-studio, pseudo-labels, coco, rle, annotation]

requires:
  - phase: 02-segmentation-pipeline
    plan: 01
    provides: MOG2Detector and Detection dataclass for fish bounding boxes
provides:
  - SAMPseudoLabeler for refining MOG2 masks via SAM2
  - Label Studio export with brush RLE format for human correction
  - Label Studio import for reading corrected annotations
  - COCO JSON conversion with pycocotools RLE encoding
  - FrameAnnotation and AnnotatedFrame dataclasses for pipeline data flow
affects: [02-03, segmentation]

tech-stack:
  added: [sam2, label-studio-converter, pycocotools, torchvision]
  patterns: [lazy model loading, brush RLE format, COCO JSON with RLE masks]

key-files:
  created:
    - src/aquapose/segmentation/pseudo_labeler.py
    - src/aquapose/segmentation/label_studio.py
    - tests/unit/segmentation/test_pseudo_labeler.py
    - tests/unit/segmentation/test_label_studio.py
  modified:
    - src/aquapose/segmentation/__init__.py
    - pyproject.toml

key-decisions:
  - "SAM2 predictor lazily loaded on first predict() call to avoid GPU allocation on import"
  - "Label Studio uses its own RLE variant (mask2rle) not pycocotools RLE"
  - "COCO JSON stores RLE counts as UTF-8 string for JSON serialization"
  - "decode_rle returns RGBA flat array, reshape to (H, W, 4) and take first channel"

patterns-established:
  - "FrameAnnotation/AnnotatedFrame dataclasses for pipeline data flow"
  - "Lazy model loading pattern for GPU-heavy models"
  - "Two RLE formats: LS brush (mask2rle) for annotation, pycocotools for training"

duration: 12min
completed: 2026-02-19
---

# Plan 02-02: SAM2 Pseudo-Labeler and Label Studio IO Summary

**SAM2 mask refinement from MOG2 detections with Label Studio brush RLE export/import and COCO JSON conversion**

## Performance

- **Duration:** 12 min
- **Completed:** 2026-02-19
- **Tasks:** 1 (auto) + 1 (checkpoint -- human verification)
- **Files modified:** 6

## Accomplishments
- SAMPseudoLabeler refines rough MOG2 masks into precise segmentation masks via SAM2 box+mask prompts
- Label Studio export creates task JSON with brush RLE masks for human annotation review
- Label Studio import reads corrected annotations back to binary numpy masks
- COCO JSON conversion encodes masks as pycocotools RLE for training
- 21 unit tests covering pseudo-labeler (mock SAM2), Label Studio export/import, COCO conversion

## Task Commits

1. **Task 1: SAM2 pseudo-labeler and Label Studio IO** - `2b13849` (feat)

**Note:** Task 2 is a human verification checkpoint -- user should verify Label Studio round-trip works in practice.

## Files Created/Modified
- `src/aquapose/segmentation/pseudo_labeler.py` - SAMPseudoLabeler class, FrameAnnotation/AnnotatedFrame dataclasses
- `src/aquapose/segmentation/label_studio.py` - export_to_label_studio, import_from_label_studio, to_coco_dataset
- `tests/unit/segmentation/test_pseudo_labeler.py` - 8 tests for pseudo-labeler
- `tests/unit/segmentation/test_label_studio.py` - 13 tests for Label Studio IO
- `src/aquapose/segmentation/__init__.py` - Updated exports
- `pyproject.toml` - Added pycocotools, label-studio-converter, torchvision deps

## Decisions Made
- SAM2 model lazily loaded on first use (avoids GPU allocation on import)
- Label Studio brush RLE used for annotation (mask2rle), pycocotools RLE for COCO output
- decode_rle returns RGBA flat array that reshapes to (H, W, 4); first channel taken for binary mask
- sam2 import suppressed with pyright: ignore since it requires separate installation

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
None

## User Setup Required
- SAM2 must be installed separately: `git clone https://github.com/facebookresearch/sam2.git && pip install -e sam2/`
- Label Studio: `pip install label-studio` for the annotation UI (already installed via label-studio-converter dep)

## Next Phase Readiness
- Pseudo-labels and COCO JSON output ready for Mask R-CNN training (02-03)
- Human verification of Label Studio round-trip deferred as informational checkpoint

---
*Phase: 02-segmentation-pipeline*
*Completed: 2026-02-19*
