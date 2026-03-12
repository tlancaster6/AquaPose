---
created: 2026-03-12T14:02:55.846Z
title: Triangulate keypoints directly instead of 6-to-15 upsampling
area: reconstruction
files:
  - src/aquapose/core/reconstruction/stage.py:47-108
  - src/aquapose/core/reconstruction/stage.py:284-288
  - src/aquapose/core/reconstruction/backends/dlt.py:157
  - src/aquapose/evaluation/runner.py:618-622
---

## Problem

The reconstruction pipeline upsamples 6 anatomical keypoints to 15 "body points" via linear spline interpolation in 2D pixel space (`_keypoints_to_midline`), then triangulates each of the 15 points independently across cameras. The 9 interpolated points add no independent information — they are linear combinations of the 6 real observations. Worse, they introduce correlated error: if one keypoint is off by a few pixels in one camera, multiple interpolated body points between it and its neighbours are all biased in the same direction, creating systematic triangulation distortion.

This design was inherited from an earlier segment-then-extract pipeline that produced many noisy medial axis points and needed upsampling for spline fitting. With discrete keypoints from the pose model, it is an unnecessary indirection.

The DLT backend requires `min_body_points = n_control_points + 2 = 9` valid 3D points for spline fitting (7 control points, cubic B-spline). With only 6 keypoints, some of which may fail triangulation, this threshold cannot be met without the upsampling step.

## Solution

- Triangulate the 6 keypoints directly (they have known anatomical correspondence across cameras)
- Reduce `n_control_points` (e.g. 4) so that `min_body_points` drops to 6, or use a lower-degree spline
- Remove `_keypoints_to_midline` from the reconstruction path (may still be useful in eval runner)
- Update DLT backend to work with variable body point counts rather than assuming 15
- Propagate the change to the eval runner's `_build_midline_sets`
