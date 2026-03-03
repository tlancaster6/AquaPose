---
created: 2026-03-01T16:35:08.480Z
title: Replace custom U-Net with Ultralytics-only model stack
area: segmentation
files:
  - src/aquapose/segmentation/unet.py
  - src/aquapose/training/
  - src/aquapose/segmentation/
---

## Problem

The project currently maintains a custom U-Net implementation (MobileNetV3-Small encoder, ~2.5M params) alongside Ultralytics YOLO models for detection, OBB, and pose. This creates two separate codepaths for training, inference, weight loading, and configuration. The custom U-Net:

- Requires its own training loop, data loading, augmentation, and checkpoint logic
- Has a separate inference API from the Ultralytics models
- Achieved only 0.623 val IoU (below the 0.90 target), suggesting the architecture may not be optimal
- Adds maintenance burden for a one-off model that doesn't benefit from Ultralytics ecosystem updates

Ultralytics already provides segmentation models (YOLO-Seg) that could replace the custom U-Net, unifying the entire model stack under one framework.

## Solution

- Replace custom U-Net segmentation with Ultralytics YOLO-Seg (instance segmentation)
- Remove `src/aquapose/segmentation/unet.py`, custom `BinaryMaskDataset`, and custom training loop
- Unify all training under Ultralytics API: detect, obb, seg, pose all use the same `model.train()` interface
- Update `UNetSegmentor` to a YOLO-Seg based segmentor (or generalize the segmentor interface)
- Benefits: single training API, pretrained backbones, built-in augmentation, active community, easy model swapping (nano/small/medium)
- Convert existing U-Net training data to YOLO segmentation format (polygon annotations)
- Consider whether instance segmentation (per-fish masks) is better than the current binary mask approach
