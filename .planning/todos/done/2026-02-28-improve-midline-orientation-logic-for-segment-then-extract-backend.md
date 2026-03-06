---
created: 2026-02-28T12:55:30.476Z
title: Improve midline orientation logic for segment-then-extract backend
area: reconstruction
files:
  - src/aquapose/reconstruction/
---

## Problem

The current segment-then-extract backend has unreliable midline orientation logic — it fails a significant portion of the time, producing head-tail ambiguities that complicate downstream 3D reconstruction. When midlines are flipped inconsistently across views, triangulation and spline fitting degrade because corresponding points are mismatched.

## Solution

Short-term: Improve the heuristic orientation logic in the segment-then-extract path (e.g., curvature-based or area-based head detection, cross-view orientation consensus).

Long-term: The planned keypoint-based swappable backend will solve this for free — pose estimation inherently resolves head vs. tail orientation at detection time. Once that backend is available, the segment-then-extract orientation issue becomes moot for users who switch backends. However, fixing orientation in the current backend is still valuable for users who prefer the segmentation approach.
