---
phase: 107-unfrozen-backbone-fine-tuning
plan: 01
subsystem: training
tags: [pytorch, swin, metric-learning, fine-tuning, reid]

requires:
  - phase: 106-cli-integration
    provides: ReID CLI infrastructure and projection head training
provides:
  - unfreeze_last_n_blocks for selective Swin backbone unfreezing
  - ImageCropDataset with FishEmbedder-matching normalization
  - train_reid_end_to_end with differential LR and combined checkpoints
  - ReidTrainingConfig extended with unfreeze_blocks, lr_backbone_factor, gradient_clip_val
affects: [107-02, reid-cli, swap-detection]

tech-stack:
  added: []
  patterns: [differential-lr-param-groups, selective-backbone-unfreezing, combined-checkpoint-format]

key-files:
  created: []
  modified:
    - src/aquapose/training/reid_training.py
    - src/aquapose/training/__init__.py
    - tests/unit/training/test_reid_training.py

key-decisions:
  - "Single combined checkpoint with backbone_state_dict + head_state_dict + config dict"
  - "Gradient clipping at max_norm=1.0 by default, configurable via gradient_clip_val"
  - "No data augmentation in end-to-end path (start simple, add later if needed)"
  - "Null check on cv2.imread in ImageCropDataset for type safety"

patterns-established:
  - "Combined checkpoint format: {backbone_state_dict, head_state_dict, config} detected by presence of backbone_state_dict key"
  - "Differential LR: backbone at lr_head * lr_backbone_factor, head at lr_head, single Adam optimizer"

requirements-completed: [UNFREEZE-01, UNFREEZE-02]

duration: 12min
completed: 2026-03-25
---

# Plan 107-01: Backbone Unfreezing + End-to-End Training Summary

**Selective Swin block unfreezing with differential LR training and combined backbone+head checkpoints for ReID fine-tuning**

## Performance

- **Duration:** 12 min
- **Tasks:** 1 (TDD-style)
- **Files modified:** 3

## Accomplishments
- Added `unfreeze_last_n_blocks()` for freezing all backbone params then selectively enabling last N Swin blocks + final LayerNorm
- Added `ImageCropDataset` with identical normalization to `FishEmbedder.embed_batch`: BGR->RGB, resize, (x/255-0.5)/0.5
- Added `train_reid_end_to_end()` with differential learning rates, gradient clipping, and combined checkpoint saving
- Extended `ReidTrainingConfig` with `unfreeze_blocks`, `lr_backbone_factor`, `gradient_clip_val` fields
- All existing frozen-path code untouched (backward compatible)

## Task Commits

1. **Task 1: Add unfreezing, ImageCropDataset, and end-to-end training** - `1500139` (feat)

## Files Created/Modified
- `src/aquapose/training/reid_training.py` - Added unfreeze_last_n_blocks, ImageCropDataset, _collect_crop_paths, train_reid_end_to_end; extended ReidTrainingConfig
- `src/aquapose/training/__init__.py` - Exported new public symbols
- `tests/unit/training/test_reid_training.py` - Added tests for unfreezing, ImageCropDataset normalization, crop path collection

## Decisions Made
- Single combined checkpoint file (detectable by `backbone_state_dict` key presence) rather than separate files
- Gradient clipping enabled by default (1.0 max norm) as a safety measure for the larger parameter count
- No image augmentation in end-to-end path (keeps things simple; can add later)

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- train_reid_end_to_end and combined checkpoint format ready for Plan 107-02 CLI integration
- _collect_crop_paths reuses manifest discovery from build_feature_cache

---
*Phase: 107-unfrozen-backbone-fine-tuning*
*Completed: 2026-03-25*
