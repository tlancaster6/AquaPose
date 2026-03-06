---
phase: 02-segmentation-pipeline
plan: 02
subsystem: segmentation
tags: [sam2, box-only, pseudo-labels, quality-filtering, coco, stratified-split, negative-examples]

# Dependency graph
requires: ["02-01"]
provides:
  - SAMPseudoLabeler with box-only prompting (no mask prompt)
  - filter_mask() reusable quality filter in aquapose.segmentation
  - CropDataset loading native-resolution crops (no forced 256x256)
  - stratified_split() for per-camera 80/20 train/val splits
  - build_training_data.py with negative examples and train/val JSON output
affects: [02-segmentation-pipeline, training-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SAM2 always box-only — no mask_input prompt, _mask_to_logits removed entirely"
    - "filter_mask() co-located in pseudo_labeler.py — quality filtering is a pseudo-label concern"
    - "CropDataset variable-size — Mask R-CNN FPN+RoI handles variable inputs natively"
    - "stratified_split() groups by camera_id field in COCO JSON for proportional splits"
    - "Negative examples sampled from random bbox-sized regions; saved as empty-annotation COCO entries"

key-files:
  created: []
  modified:
    - src/aquapose/segmentation/pseudo_labeler.py
    - src/aquapose/segmentation/dataset.py
    - src/aquapose/segmentation/__init__.py
    - src/aquapose/segmentation/training.py
    - tests/unit/segmentation/test_pseudo_labeler.py
    - tests/unit/segmentation/test_dataset.py
    - scripts/build_training_data.py

key-decisions:
  - "filter_mask() lives in pseudo_labeler.py — quality filtering is part of the pseudo-labeling stage, not a separate concern"
  - "CropDataset drops crop_size parameter entirely — Mask R-CNN handles variable inputs via FPN, no resize needed"
  - "stratified_split uses camera_id COCO field to group images — each camera gets proportional val representation"
  - "Negative crops use random bbox-sized patches from video margins — size range 100-300px matches plausible fish scale"
  - "build_training_data.py writes train.json + val.json + coco_annotations.json — three files, one source of truth"

requirements-completed: [SEG-01, SEG-02, SEG-04]

# Metrics
duration: 11min
completed: 2026-02-20
---

# Phase 02 Plan 02: Pseudo-Labeling Pipeline Update Summary

**Box-only SAM2 with quality filtering, variable-size CropDataset, per-camera stratified split, and ~10% negative examples**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-02-20T23:28:58Z
- **Completed:** 2026-02-20T23:39:58Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Removed `_mask_to_logits` and `use_mask_prompt` parameter from `SAMPseudoLabeler.predict()` — box-only is now the only mode
- Added `_select_largest_mask()` helper for multi-mask SAM2 output
- Added `filter_mask()` function with confidence/area/fill-ratio checks — replaces local `_filter_mask` in build script
- Added `draw_pseudolabels` constructor parameter for debug visualization (saves annotated crops to `debug/`)
- Updated `CropDataset` to load crops at native resolution — removed forced `crop_size=256` resize
- Added `stratified_split()` for per-camera 80/20 train/val splits based on `camera_id` COCO field
- Updated `build_training_data.py` to use module-level `filter_mask` and `stratified_split`
- Added `--neg-fraction`, `--val-fraction`, `--seed` arguments to `generate` subcommand
- Added negative crop generation (~10% background crops per camera)
- Script now writes `train.json` and `val.json` alongside combined `coco_annotations.json`
- All 203 unit tests pass (was 182, added 21 new tests)

## Task Commits

1. **Task 1: Box-only SAM2 with quality filtering** - `003e8de` (feat)
2. **Task 2: Variable-size CropDataset, stratified split, negative examples** - `2766e7d` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/aquapose/segmentation/pseudo_labeler.py` - Removed `_mask_to_logits` and `use_mask_prompt`; added `_select_largest_mask`, `filter_mask`, `draw_pseudolabels`
- `src/aquapose/segmentation/dataset.py` - Removed `crop_size` resize; added `stratified_split()`
- `src/aquapose/segmentation/__init__.py` - Exported `filter_mask` and `stratified_split`
- `src/aquapose/segmentation/training.py` - Removed `crop_size` parameter from `CropDataset` calls (Rule 1 auto-fix)
- `tests/unit/segmentation/test_pseudo_labeler.py` - Removed `TestMaskToLogits`; added tests for `filter_mask`, `_select_largest_mask`, box-only mode
- `tests/unit/segmentation/test_dataset.py` - Replaced 256x256 shape assertions with native-size tests; added `TestStratifiedSplit` (7 tests) and `TestCropDatasetVariableSize` (4 tests)
- `scripts/build_training_data.py` - Uses module `filter_mask`/`stratified_split`, adds negative example generation and train/val JSON output

## Decisions Made

- `filter_mask()` belongs in `pseudo_labeler.py` — quality filtering is part of pseudo-labeling, not a build-script concern
- `CropDataset` drops `crop_size` entirely — Mask R-CNN's FPN + RoI pooling handles variable-size inputs natively, so forced resize adds no value and discards spatial information
- `stratified_split` groups by `camera_id` COCO field — ensures every camera is proportionally represented in both splits regardless of how many frames were sampled per camera
- Negative crops use random bbox-sized patches (100-300px range) from video frames — matches plausible fish crop dimensions at inference time

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed training.py CropDataset calls after removing crop_size parameter**
- **Found during:** Task 2 (typecheck after removing crop_size from CropDataset)
- **Issue:** `training.py` called `CropDataset(coco_json, image_root, crop_size, augment=True)` with `crop_size` as a positional argument. After removing `crop_size` from `CropDataset.__init__`, this would pass `augment` as a positional arg mapping to nothing — TypeError at runtime
- **Fix:** Removed `crop_size` from all three `CropDataset()` calls in `training.py`; removed `crop_size` parameter from `train()` and `evaluate()` signatures; updated docstrings
- **Files modified:** `src/aquapose/segmentation/training.py`
- **Verification:** `hatch run test` — 203 passed

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug fix in training.py due to removed parameter)
**Impact on plan:** Necessary correctness fix. `training.py` would have raised `TypeError` on first use. No scope creep.

## Issues Encountered

- Pre-existing typecheck errors in `detector.py` (4 errors: cv2 normalize signature, YOLO import, optional iterable) — confirmed pre-existing, out of scope

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SAM2 pseudo-labeler uses box-only prompting exclusively
- Quality filtering via `filter_mask()` is a reusable module-level function
- `CropDataset` supports variable-size crops (not fixed 256x256)
- Per-camera stratified 80/20 split implemented via `stratified_split()`
- `build_training_data.py` generates ~10% negative examples and writes `train.json` / `val.json`
- 203 tests passing, lint clean
- Ready for Plan 03: Mask R-CNN training on generated dataset

## Self-Check: PASSED

All created/modified files found. All task commits (003e8de, 2766e7d) verified in git log.

---
*Phase: 02-segmentation-pipeline*
*Completed: 2026-02-20*
