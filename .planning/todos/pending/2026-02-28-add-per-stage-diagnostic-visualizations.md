---
created: 2026-02-28T13:05:42.546Z
title: Add per-stage diagnostic visualizations
area: visualization
files:
  - src/aquapose/visualization/
  - src/aquapose/engine/pipeline.py
---

## Problem

When the pipeline produces poor results, it's hard to pinpoint which stage is failing. Each stage (Detection, 2D Tracking, Association, Midline, Reconstruction) can fail in different ways, but there are no targeted visualizations that make failure modes obvious at a glance. Debugging currently requires manual inspection of intermediate data structures.

## Solution

Examine each pipeline stage and develop one or more targeted visualizations per stage:

- **Detection**: Overlay bounding boxes + confidence scores on frames; highlight missed detections vs false positives
- **2D Tracking**: Draw track trajectories with ID labels over time; color-code track switches/fragmentations
- **Association**: Visualize cross-view identity clusters; show which detections were grouped and which were orphaned; epipolar line overlays
- **Midline**: Overlay extracted midlines on masks; show arc-length sample points; highlight orientation flips
- **Reconstruction**: 3D scatter of triangulated points with reprojection error heatmap; show which cameras contributed to each point

These could be implemented as observer plugins or standalone diagnostic functions callable from the CLI.
