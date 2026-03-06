---
phase: 02-segmentation-pipeline
plan: 04
subsystem: segmentation
tags: [unet, pipeline-integration, training, model-replacement]

# Dependency graph
requires:
  - phase: 02-03
    provides: "MaskRCNNSegmentor.segment() crop-space output, stratified split training"
  - phase: 02.1.1
    provides: "YOLODetector, SAM2 pseudo-label pipeline"
provides:
  - UNetSegmentor replacing MaskRCNNSegmentor as default segmentation model
  - BinaryMaskDataset for U-Net training at fixed 128x128
  - Differential LR training (encoder 1/10th of decoder)
  - Full pipeline CLI working end-to-end (generate -> train -> evaluate)
affects: [02-segmentation-pipeline, optimization-pipeline]

# Tech tracking
tech-stack:
  added:
    - "MobileNetV3-Small encoder (torchvision pretrained, 2.5M params)"
  removed:
    - "Mask R-CNN as default segmentor (retained for backward compat)"
  patterns:
    - "UNetSegmentor(weights_path, confidence_threshold) — same segment()/predict() interface as MaskRCNNSegmentor"
    - "BinaryMaskDataset returns (image, mask) at 128x128 — standard batched DataLoader, no custom collate"
    - "Differential LR: encoder_lr = lr * 0.1, decoder_lr = lr (AdamW)"
    - "BCE + Dice loss for binary segmentation"

key-files:
  created:
    - scripts/visualize_val_predictions.py
  modified:
    - src/aquapose/segmentation/model.py
    - src/aquapose/segmentation/dataset.py
    - src/aquapose/segmentation/training.py
    - src/aquapose/segmentation/__init__.py
    - scripts/build_training_data.py
    - tests/unit/segmentation/test_model.py
    - tests/unit/segmentation/test_dataset.py
    - tests/unit/segmentation/test_training.py

key-decisions:
  - "Replaced Mask R-CNN with lightweight U-Net — Mask R-CNN re-does detection (RPN + box head) on crops that already contain a single centered fish; U-Net performs binary segmentation directly"
  - "MobileNetV3-Small encoder (2.5M params) vs ResNet-50 (25M) — sufficient for binary segmentation on small crops, ~10x less VRAM"
  - "Fixed 128x128 input — crops are small (median ~56px), 128 provides headroom; enables standard batched DataLoader"
  - "Bilinear upsample + conv decoder (not transposed conv) — avoids checkerboard artifacts"
  - "BCE + Dice loss equally weighted — BCE for per-pixel gradients, Dice for class imbalance"
  - "Differential LR: pretrained encoder at 1/10th decoder LR — prevents overwriting encoder features before decoder learns"
  - "AdamW with lr=1e-4 (not SGD 0.005) — AdamW standard range, SGD default was causing training instability"
  - "Best val IoU 0.623 accepted despite 0.90 target — sufficient to unblock Phase 4; can improve with more data later"
  - "MaskRCNNSegmentor retained in codebase for backward compatibility but no longer default"

patterns-established:
  - "apply_augmentation() extracted as standalone function shared by CropDataset and BinaryMaskDataset"
  - "stratified_split() accepts both CropDataset and BinaryMaskDataset"

requirements-completed: [SEG-01, SEG-02, SEG-04, SEG-05]
requirements-partial: [SEG-03]

# Metrics
duration: multi-session
completed: 2026-02-20
---

# Phase 02 Plan 04: U-Net Model Replacement and Pipeline Verification Summary

**Replaced Mask R-CNN with lightweight U-Net (MobileNetV3-Small); trained on SAM2 pseudo-labels; best val IoU 0.623**

## Performance

- **Duration:** Multi-session (model replacement + training tuning + evaluation)
- **Completed:** 2026-02-20
- **Files modified:** 8
- **Files created:** 1

## Accomplishments

### Model Architecture
- Added `_UNet` module: MobileNetV3-Small encoder + 4-level decoder with skip connections
- Added `UNetSegmentor` with identical `segment()`/`predict()`/`get_model()` interface to MaskRCNNSegmentor
- Fixed 128x128 input/output with resize back to original crop dimensions
- Single-channel sigmoid output → threshold at 0.5 → binary mask (0/255 uint8)
- Confidence = mean foreground probability of predicted pixels

### Dataset
- Added `BinaryMaskDataset`: returns `(image, mask)` tensors at 128x128
- Merges all per-instance masks into single binary mask (one fish per crop)
- Extracted `apply_augmentation()` to standalone function shared by both dataset classes
- `stratified_split()` updated to accept both dataset types

### Training Pipeline
- BCE + Dice loss (equally weighted) replaces Mask R-CNN multi-task loss
- Standard batched DataLoader (no custom collate_fn needed)
- AdamW optimizer with differential LR: encoder 1e-5, decoder 1e-4
- CosineAnnealingLR scheduler, gradient clipping at 5.0
- Defaults: 100 epochs, batch_size=8, patience=20, weight_decay=1e-3
- Validation every epoch (not every 5)

### Training Results
- Dataset: 194 train / 48 val images from 12 cameras
- Best val IoU: 0.623 (early stopping at epoch 85)
- Predictions capture fish shapes but boundaries are blobby — expected with 194 training images
- Accepted as sufficient to unblock Phase 4

### Testing
- 233 unit tests passing (up from 204)
- Removed `@pytest.mark.slow` from U-Net tests (no pretrained download needed)
- Tests cover: construction, forward pass, segment(), predict(), batch/variable sizes, confidence filtering, BinaryMaskDataset, training, evaluation

## Training Parameter Evolution

| Parameter | Initial | After tuning | Rationale |
|-----------|---------|-------------|-----------|
| Optimizer | SGD (from Mask R-CNN) | AdamW | Standard for fine-tuning |
| LR | 0.005 | 1e-4 | 0.005 caused training collapse |
| Encoder LR | same | 1/10th decoder | Protect pretrained features |
| Weight decay | 0.01 | 1e-3 | Less regularization for small dataset |
| Epochs | 40 | 100 | Small dataset needs more passes |
| Batch size | 4 | 8 | More stable gradients |
| Patience | 10 (every 5 epochs) | 20 (every epoch) | More runway, finer monitoring |

## Decisions Made

- U-Net replaces Mask R-CNN as default — Mask R-CNN's RPN + box head is redundant when crops already contain a single centered fish from YOLO detection
- MobileNetV3-Small (2.5M params) is sufficient encoder — 10x smaller than ResNet-50, adequate for binary segmentation on small crops
- 0.623 val IoU accepted below 0.90 target — limited by 194 training images; more data (--max-frames 20+) is the highest-ROI improvement path
- MaskRCNNSegmentor kept in codebase but deprecated — backward compatibility preserved

## Paths to Improve IoU

1. **More training data** — regenerate with `--max-frames 20` or `30` (currently 10 per camera)
2. **Higher input resolution** — 192 or 256 instead of 128
3. **Heavier augmentation** — elastic deform, Gaussian blur, cutout
4. **Larger encoder** — MobileNetV3-Large

## Files Created/Modified

- `src/aquapose/segmentation/model.py` — Added `_UNet`, `_DecoderBlock`, `UNetSegmentor`; kept `MaskRCNNSegmentor`
- `src/aquapose/segmentation/dataset.py` — Added `BinaryMaskDataset`; extracted `apply_augmentation()`
- `src/aquapose/segmentation/training.py` — Replaced Mask R-CNN loop with U-Net: BCE+Dice loss, batched DataLoader, differential LR, per-epoch validation
- `src/aquapose/segmentation/__init__.py` — Added `UNetSegmentor`, `BinaryMaskDataset` to exports
- `scripts/build_training_data.py` — Updated strings from "Mask R-CNN" to "U-Net", default output dir
- `scripts/visualize_val_predictions.py` — New: generates side-by-side GT vs prediction mosaic
- `tests/unit/segmentation/test_model.py` — Rewrote for UNetSegmentor
- `tests/unit/segmentation/test_dataset.py` — Added BinaryMaskDataset tests
- `tests/unit/segmentation/test_training.py` — Updated for U-Net training

## Next Phase Readiness

- `UNetSegmentor.segment(crops, crop_regions)` produces crop-space binary masks
- Full pipeline: YOLO detect → crop → U-Net segment → paste_mask for full-frame
- 233 tests passing, lint clean
- Ready for Phase 4 (Single-Fish Reconstruction) which consumes segmentation masks

## Self-Check: PASSED

All modified files verified. 233 tests passing, ruff lint clean.

---
*Phase: 02-segmentation-pipeline*
*Completed: 2026-02-20*
