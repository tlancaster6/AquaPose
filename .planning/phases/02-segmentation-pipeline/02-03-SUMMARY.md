---
phase: 02-segmentation-pipeline
plan: 03
subsystem: segmentation
tags: [mask-rcnn, inference-pipeline, stratified-split, variable-crops, crop-space]

# Dependency graph
requires:
  - phase: 02-02
    provides: "CropDataset native-resolution, stratified_split, CropRegion/paste_mask utilities"
provides:
  - MaskRCNNSegmentor.segment() returning crop-space masks with CropRegion metadata
  - Backward-compatible predict() wrapping segment() with trivial CropRegion
  - training.py using stratified_split instead of random_split
  - training.py accepting train_json/val_json for pre-split COCO files
affects: [02-segmentation-pipeline, optimization-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "segment(crops, crop_regions) is the primary inference entry point — detect -> crop -> segment pipeline"
    - "SegmentationResult.mask is crop-space ndarray (uint8, 0/255) — callers paste_mask() for full-frame"
    - "predict() is a backward-compat wrapper that creates trivial CropRegion covering full image"
    - "training.py uses stratified_split when coco_json given; uses train_json/val_json when both provided"

key-files:
  created: []
  modified:
    - src/aquapose/segmentation/model.py
    - src/aquapose/segmentation/training.py
    - tests/unit/segmentation/test_model.py
    - tests/unit/segmentation/test_training.py

key-decisions:
  - "segment() accepts crops+crop_regions, returns crop-space masks — callers reconstruct full-frame via paste_mask(result.mask, result.crop_region)"
  - "predict() kept for backward compatibility; internally calls segment() with trivial CropRegion — no behavior change for existing callers"
  - "SegmentationResult.mask_rle removed — callers who need RLE encode it themselves; raw ndarray is more useful"
  - "train() uses stratified_split by default; accepts train_json/val_json to consume build_training_data.py output directly"

patterns-established:
  - "Crop-space mask output: all segment() results are in crop coordinates, not frame coordinates"
  - "Full-frame reconstruction: paste_mask(result.mask, result.crop_region) is the canonical pattern"

requirements-completed: [SEG-04, SEG-05]

# Metrics
duration: 5min
completed: 2026-02-20
---

# Phase 02 Plan 03: Mask R-CNN Model and Training Update Summary

**Separate detect/crop/segment pipeline with crop-space mask output + CropRegion metadata; stratified per-camera train/val split**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-20T23:44:11Z
- **Completed:** 2026-02-20T23:48:59Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added `segment(crops, crop_regions)` as the primary inference entry point on `MaskRCNNSegmentor`
- `SegmentationResult` now carries `mask` (crop-space ndarray, 0/255) and `crop_region` instead of `mask_rle`
- `predict()` retained as backward-compatible wrapper (calls `segment()` with trivial `CropRegion`)
- Batch inference processes all crops in a single forward pass (FPN handles variable sizes natively)
- Replaced `random_split` with `stratified_split` in `training.py` for per-camera val representation
- Added `train_json`/`val_json` parameters to `train()` for consuming pre-split COCO files from `build_training_data.py`
- All 204 unit tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor MaskRCNNSegmentor for separate stages and crop-space output** - `5c4cbde` (feat)
2. **Task 2: Update training pipeline for stratified split and variable crops** - `25ce169` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/aquapose/segmentation/model.py` - Added `segment()` method; `SegmentationResult` now has `mask` (ndarray) and `crop_region` (CropRegion); `predict()` wraps `segment()` with trivial CropRegion; removed `mask_rle`; added `CropRegion` import
- `src/aquapose/segmentation/training.py` - Replaced `random_split` with `stratified_split`; added `train_json`/`val_json` parameters; removed unused `random_split` import; updated docstrings
- `tests/unit/segmentation/test_model.py` - Rewrote tests: `segment()` with single/batch/variable-size crops, crop_region attachment verification, mismatch error, backward-compat `predict()` tests; removed stale `mask_rle` assertions
- `tests/unit/segmentation/test_training.py` - Added tests: stratified split (6 images, 2 cameras), pre-split JSON file loading, variable-size crop training and evaluation; added `_split_coco_json` helper and `_create_training_fixture` parameters

## Decisions Made

- `SegmentationResult.mask_rle` removed in favor of `mask: np.ndarray` (crop-space, uint8, 0/255) — callers who need RLE can encode with `pycocotools.mask.encode()` themselves; raw ndarray is universally more useful downstream
- `predict()` kept for backward compatibility using trivial CropRegion (x1=0, y1=0, x2=W, y2=H, frame_h=H, frame_w=W) — no API break for existing callers
- `train()` uses `stratified_split` by default when `train_json`/`val_json` are absent; accepts both to load datasets directly from `build_training_data.py` output

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Pre-existing typecheck errors in `detector.py` (4 errors: cv2 normalize signature, YOLO import, optional iterable) — confirmed pre-existing, out of scope per deviation rules

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Full detect -> crop -> segment pipeline is callable as separate stages
- `MaskRCNNSegmentor.segment()` accepts pre-cropped images with CropRegion metadata
- Callers reconstruct full-frame masks via `paste_mask(result.mask, result.crop_region)`
- Training uses per-camera stratified split and accepts pre-split COCO JSON
- 204 tests passing, lint clean
- Ready for actual training run on generated pseudo-label dataset

## Self-Check: PASSED

All created/modified files found. All task commits (5c4cbde, 25ce169) verified in git log.

---
*Phase: 02-segmentation-pipeline*
*Completed: 2026-02-20*
