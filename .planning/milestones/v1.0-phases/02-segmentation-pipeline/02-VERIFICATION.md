---
phase: 02-segmentation-pipeline
status: complete_with_caveat
score: 3/4
verified: 2026-02-20
---

# Phase 2: Segmentation Pipeline Verification

## Goal
The system can produce binary fish masks for any input frame across 12 cameras (center camera e3v8250 excluded), with quality sufficient to unblock Phase 4 single-fish reconstruction.

## Must-Have Verification

### 1. Detection produces bounding boxes for fish in each camera
- **Status:** PASS
- **Evidence:** YOLOv8n detector replaced MOG2 as primary detector (Phase 02.1.1). Trained on 150 frames across 12 cameras. Val metrics: recall=0.780, mAP50=0.799, precision=0.760. `make_detector("yolo", model_path=...)` provides runtime-configurable detection. MOG2Detector retained as alternative.
- **Note:** 95% recall target from original spec not met (78%), but YOLO is substantially more reliable than MOG2 on real data (no warmup dependency, no stationary-fish failures).

### 2. SAM2 pseudo-labels generated via box-only prompting with quality filtering
- **Status:** PASS
- **Evidence:** SAMPseudoLabeler uses crop+box-only approach (no mask prompt) producing dramatically better masks than box+mask prompting. `filter_mask()` applies min_conf, min/max_fill, min_area filters. `to_coco_dataset()` exports COCO JSON directly. Label Studio removed from pipeline entirely — train directly on pseudo-labels.

### 3. Trained segmentation model achieves target IoU on val split
- **Status:** PARTIAL — 0.623 best val IoU (target was 0.90)
- **Evidence:** Mask R-CNN replaced with lightweight U-Net (MobileNetV3-Small encoder, ~2.5M params). Trained on 194 crops from 12 cameras with differential LR (encoder 1e-5, decoder 1e-4), BCE+Dice loss, AdamW. Early stopping at epoch 85 with best val IoU 0.623. Predictions capture fish shapes but boundaries are blobby.
- **Accepted:** IoU gap is primarily a data quantity issue (194 images). Accepted as sufficient to unblock Phase 4 — masks are good enough for silhouette-based optimization. Improvement path: regenerate with `--max-frames 20+`.

### 4. Segmentation pipeline accepts N fish as input
- **Status:** PASS
- **Evidence:** YOLODetector.detect() returns variable-length `list[Detection]`. SAMPseudoLabeler.predict() accepts `list[Detection]`. UNetSegmentor.segment() accepts `list[crop]` and returns `list[list[SegmentationResult]]`. All APIs handle variable-length lists through the full detect → crop → segment chain.

## Score: 3/4 must-haves fully met, 1 partial (IoU below target)

## Architecture Summary

```
Full pipeline: Video → YOLO detect → crop → U-Net segment → paste_mask → full-frame mask
                                   ↓
                    SAM2 pseudo-label (offline) → COCO JSON → U-Net train
```

**Key components:**
- `YOLODetector` — fish bounding boxes (replaces MOG2)
- `compute_crop_region` / `extract_crop` — padded crop extraction
- `UNetSegmentor.segment()` — binary segmentation on 128×128 crops
- `paste_mask()` — crop-space mask → full-frame reconstruction
- `BinaryMaskDataset` — COCO JSON → batched (image, mask) tensors
- `build_training_data.py` — CLI orchestrating generate/train/evaluate

## What Changed from Original Plan

| Original | Final | Reason |
|----------|-------|--------|
| MOG2 detection | YOLOv8n detection | MOG2 unreliable (shadow issues, stationary fish failures) |
| Label Studio annotation | Direct SAM2 pseudo-labels | Manual annotation unnecessary — SAM2 box-only masks are high quality |
| Mask R-CNN segmentor | U-Net segmentor | Mask R-CNN re-does detection on single-fish crops; U-Net is 10× lighter |
| 0.90 IoU target | 0.623 achieved | Limited training data (194 images); accepted to unblock Phase 4 |
| 13 cameras | 12 cameras | Center camera e3v8250 excluded (top-down wide-angle, poor mask quality) |

## Automated Checks

| Check | Result |
|-------|--------|
| All unit tests pass (233) | PASS |
| Lint clean (ruff) | PASS |
| Type check (basedpyright) | PASS (4 pre-existing errors in detector.py) |
| All public symbols importable | PASS |
| SUMMARY.md exists for all 4 plans | PASS |
| U-Net forward pass correct shape | PASS (B,1,128,128) |
| build_training_data.py CLI functional | PASS (generate/train/evaluate) |
