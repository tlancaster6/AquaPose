---
created: 2026-02-22T23:05:19.472Z
title: Adaptive depth range for epipolar refinement
area: reconstruction
files:
  - src/aquapose/reconstruction/triangulation.py:564
---

## Problem

`_refine_correspondences_epipolar` uses a hardcoded `torch.linspace(0.5, 3.0, 50)` for depth sampling along rays when tracing epipolar curves. This parametric distance depends on camera height above water and tank depth, which vary per setup. Currently changed from the original `(0.05, 1.5, 25)` to `(0.5, 3.0, 50)` based on a specific rig (~1m above water, ~1m deep tank), but this won't generalize.

The `n_depth_samples` parameter is accepted by the function but the actual range is not configurable.

## Solution

Either:
1. **Adaptive range**: Compute from calibration data — use `water_z` and camera positions to estimate min/max parametric ray distances that intersect the tank volume, per camera pair.
2. **Expose as parameter**: Add `depth_range: tuple[float, float]` to `_refine_correspondences_epipolar` and propagate up to `triangulate_midlines` with a sensible default. Document units as parametric ray distance in metres.

Option 1 is more robust but more complex. Option 2 is a quick win.
