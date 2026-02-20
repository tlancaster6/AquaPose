---
phase: 02-segmentation-pipeline
plan: 03
subsystem: segmentation
tags: [mask-rcnn, torchvision, training, coco, dataset, augmentation]

requires:
  - phase: 02-segmentation-pipeline
    plan: 02
    provides: COCO JSON annotations from Label Studio workflow
provides:
  - CropDataset for COCO-format training data with augmentation
  - MaskRCNNSegmentor wrapping torchvision maskrcnn_resnet50_fpn_v2
  - Training script with SGD, StepLR, validation IoU tracking
  - Evaluate function computing per-image mask IoU
  - Complete segmentation module public API
affects: [phase-4, reconstruction, segmentation]

tech-stack:
  added: [torchvision maskrcnn_resnet50_fpn_v2, FastRCNNPredictor, MaskRCNNPredictor]
  patterns: [custom collate_fn for detection, augmentation with mask consistency]

key-files:
  created:
    - src/aquapose/segmentation/dataset.py
    - src/aquapose/segmentation/model.py
    - src/aquapose/segmentation/training.py
    - tests/unit/segmentation/test_dataset.py
    - tests/unit/segmentation/test_model.py
    - tests/unit/segmentation/test_training.py
  modified:
    - src/aquapose/segmentation/__init__.py

key-decisions:
  - "torchvision maskrcnn_resnet50_fpn_v2 instead of Detectron2 (unmaintained, Windows-incompatible)"
  - "Custom collate_fn returning tuple(zip(*batch)) for Mask R-CNN list-of-dicts format"
  - "Augmentation applied consistently to image, masks, and boxes (flip, rotate, jitter)"
  - "SGD with momentum=0.9, weight_decay=0.0005, StepLR decay at 80% of epochs"

patterns-established:
  - "CropDataset __getitem__ returns (image_tensor, target_dict) for torchvision"
  - "MaskRCNNSegmentor.predict accepts list[np.ndarray] batch"
  - "SegmentationResult with RLE-encoded masks"

duration: 15min
completed: 2026-02-19
---

# Plan 02-03: Mask R-CNN Dataset, Model, Training Summary

**Mask R-CNN training pipeline with torchvision using CropDataset, SGD optimizer, StepLR scheduler, and per-image IoU evaluation**

## Performance

- **Duration:** 15 min
- **Completed:** 2026-02-19
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- CropDataset loads COCO JSON annotations and produces 256x256 crops with consistent augmentation
- MaskRCNNSegmentor wraps torchvision maskrcnn_resnet50_fpn_v2 with proper head replacement
- Training script handles SGD optimization, LR scheduling, model checkpointing
- Evaluate function computes mean mask IoU per image
- Complete segmentation module API: 12 public symbols exported

## Task Commits

1. **Task 1: CropDataset and MaskRCNNSegmentor** - `69cef1f` (feat)
2. **Task 2: Training script, evaluate, pipeline API** - `b87fc08` (feat)

## Files Created/Modified
- `src/aquapose/segmentation/dataset.py` - CropDataset with COCO loading and augmentation
- `src/aquapose/segmentation/model.py` - MaskRCNNSegmentor and SegmentationResult
- `src/aquapose/segmentation/training.py` - train() and evaluate() entry points
- `tests/unit/segmentation/test_dataset.py` - 9 dataset tests
- `tests/unit/segmentation/test_model.py` - 5 model tests (4 @slow)
- `tests/unit/segmentation/test_training.py` - 2 training tests (@slow)
- `src/aquapose/segmentation/__init__.py` - Complete public API exports

## Decisions Made
- Used torchvision maskrcnn_resnet50_fpn_v2 (not Detectron2) per research recommendation
- Custom collate_fn with tuple(zip(*batch)) for Mask R-CNN's list-of-dicts format
- Augmentation includes flips, 90-degree rotations, brightness/contrast jitter
- Validation every 5 epochs; best model saved by IoU

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
None

## User Setup Required
None - all dependencies already added in 02-02.

## Next Phase Readiness
- Complete segmentation pipeline available for Phase 4 reconstruction
- MOG2Detector -> SAMPseudoLabeler -> Label Studio -> CropDataset -> MaskRCNNSegmentor chain complete
- Real data training requires annotated COCO JSON from the Label Studio workflow

---
*Phase: 02-segmentation-pipeline*
*Completed: 2026-02-19*
