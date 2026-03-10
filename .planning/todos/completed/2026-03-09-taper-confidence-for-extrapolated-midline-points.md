---
created: 2026-03-09T18:17:09.468Z
title: NaN-out extrapolated midline points instead of clamping confidence
area: core
files:
  - src/aquapose/core/midline/backends/pose_estimation.py:31-78
---

## Problem

When endpoint keypoints (nose at t=0.0 or tail at t=1.0) are dropped by the confidence floor (0.3), `_keypoints_to_midline` extrapolates a straight line beyond the last visible keypoint. The confidence for these extrapolated points is **clamped** to the last visible keypoint's confidence value (e.g. 0.6), making them appear trustworthy.

Downstream in DLT triangulation, these extrapolated points get sqrt(confidence) weighting (~0.77), so they actively pull the 3D spline toward a rigid straight tail/nose. The result is fish that appear somewhat rigid — the extrapolated geometry is a straight line, and nothing in the pipeline knows to discount it.

**Measured impact** (phase 72 baseline run, 275k midlines across 30 chunks):
- 10.6% of midlines have a clamped (dropped) tail endpoint
- 1.1% have a clamped nose endpoint
- Tail confidence systematically lower than nose (median 0.86 vs 0.94)
- 33% of tails below 0.8 confidence, 18.5% below 0.7

## Solution

**NaN-out extrapolated points** rather than extrapolating with clamped confidence. In `_keypoints_to_midline`, after interpolation, set points outside `[t_first_visible, t_last_visible]` to NaN with confidence 0. Keep the 15-point array shape.

The entire downstream pipeline already handles NaN midline points natively:
- DLT triangulation (`dlt.py:488-502`): NaN → `valid_nc=False`, weight=0 — camera contributes nothing for that body point
- 3D spline fit (`dlt.py:289`): requires 9+ valid triangulated points (out of 15)
- Endpoint gap check (`dlt.py:306`): rejects if valid points don't reach within 20% of endpoints

With 12 cameras and only ~10% tail-drop rate, there will almost always be enough cameras with real tail observations to triangulate those body points. The cameras that didn't see the tail simply abstain rather than voting for a straight line.
