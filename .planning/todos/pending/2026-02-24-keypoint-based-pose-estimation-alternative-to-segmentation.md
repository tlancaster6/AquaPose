---
created: 2026-02-24T05:27:01.386Z
title: Keypoint-based pose estimation alternative to segmentation
area: initialization
files:
  - src/aquapose/segmentation/
  - src/aquapose/initialization/
---

## Problem

The current pipeline relies on segmentation → medial axis extraction → arc-length sampling to obtain 2D midline keypoints for triangulation. This chain is fragile:

1. **U-Net IoU is only 0.623** — noisy masks produce inconsistent skeletons across views
2. **Medial axis extraction** is sensitive to mask boundary noise, especially at fin tips and tail
3. **Multiple processing steps** compound errors before any 3D reconstruction begins
4. **Cross-view consistency** depends on clean, repeatable midline extraction from imperfect masks

A keypoint-based approach would predict midline control points directly from image crops, bypassing segmentation and skeletonization entirely.

## Solution

- Train a keypoint regression model (e.g., YOLO-Pose, HRNet, or a lightweight heatmap head on MobileNet) to predict N midline keypoints directly from fish crops
- Keypoints would be ordered along the body axis (head → tail), providing natural correspondence for triangulation
- Label keypoints using existing 3D GT splines projected back to 2D as training supervision (self-supervised from current pipeline outputs on good frames)
- Could coexist with segmentation pathway — use keypoints for 3D reconstruction while keeping segmentation for silhouette-based refinement
- Evaluate whether direct keypoint detection improves cross-view triangulation consistency vs. the current segmentation+skeleton pipeline
- Consider YOLO-Pose since we already have YOLO infrastructure for detection
