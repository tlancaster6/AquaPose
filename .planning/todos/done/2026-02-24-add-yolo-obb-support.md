---
created: 2026-02-24T05:23:17.112Z
title: Add YOLO-OBB support
area: segmentation
files:
  - src/aquapose/segmentation/
---

## Problem

The current YOLO detection pipeline uses axis-aligned bounding boxes. For elongated fish at arbitrary orientations, axis-aligned boxes waste significant area and can overlap heavily when fish are close together. YOLO-OBB (Oriented Bounding Box) provides tighter, rotation-aware bounding boxes that better match fish body orientation, improving:

1. **Crop quality** — tighter crops mean less background, better downstream segmentation
2. **Fish separation** — oriented boxes reduce overlap between adjacent fish
3. **Detection association** — OBB angle provides a coarse heading prior for cross-view matching

## Solution

- Train a YOLOv8-OBB model on the existing fish dataset (requires converting labels to OBB format with rotation angles)
- Update `make_detector("yolo", ...)` to support OBB model variants
- Extract orientation angle from OBB predictions as an additional detection attribute
- Update crop utilities in `src/aquapose/segmentation/crop.py` to use oriented crops when OBB detections are available
- Consider using OBB angle as a heading prior for initialization
