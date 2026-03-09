---
created: 2026-03-09T18:17:09.468Z
title: Taper confidence for extrapolated midline points
area: core
files:
  - src/aquapose/core/midline/backends/pose_estimation.py:65-71
  - src/aquapose/core/reconstruction/utils.py:138
---

## Problem

When endpoint keypoints (nose at t=0.0 or tail at t=1.0) are dropped by the confidence floor (0.3), `_keypoints_to_midline` extrapolates a straight line beyond the last visible keypoint. The confidence for these extrapolated points is **clamped** to the last visible keypoint's confidence value (e.g. 0.6), making them appear trustworthy.

Downstream in DLT triangulation, these extrapolated points get sqrt(confidence) weighting (~0.77), so they actively pull the 3D spline toward a rigid straight tail/nose. The result is fish that appear somewhat rigid — the extrapolated geometry is a straight line, and nothing in the pipeline knows to discount it.

The confidence interpolation in `_keypoints_to_midline` (pose_estimation.py:65-71) uses `fill_value=(confidences[0], confidences[-1])` which clamps rather than decaying.

## Solution

1. **Primary fix**: In `_keypoints_to_midline`, replace clamped `fill_value` with a linear decay to zero beyond the observed keypoint range. Extrapolated midline points get confidence ramping from the last visible keypoint's value down to 0.0 at the body endpoint (t=0.0 or t=1.0). This causes DLT triangulation to naturally downweight extrapolated regions in favor of cameras that actually observed the endpoint.

2. **Optional enhancement**: Pass confidence-derived weights to `make_lsq_spline` via its `w=` parameter (utils.py:138) so the 3D spline fit itself also downweights low-confidence body points. Currently the spline fit is unweighted least-squares.
