---
quick_task: 25
subsystem: midline
tags: [nan-masking, triangulation, pose-estimation]
dependency_graph:
  requires: []
  provides: [NaN-masked extrapolated midline points]
  affects: [DLT triangulation, Midline2D point_confidence]
tech_stack:
  added: []
  patterns: [NaN sentinel for invalid measurements]
key_files:
  created: []
  modified:
    - src/aquapose/core/midline/backends/pose_estimation.py
    - tests/unit/core/midline/test_pose_estimation_backend.py
decisions:
  - "Mask after extrapolation rather than clamp fill_value — simpler, avoids scipy behavior differences"
  - "Use NaN sentinel (not special confidence value) so downstream code can filter with np.isfinite"
metrics:
  duration: ~8 minutes
  completed: 2026-03-10
  tasks_completed: 2
  files_modified: 2
---

# Quick Task 25: NaN-out Extrapolated Midline Points Summary

**One-liner:** `_keypoints_to_midline` now NaN-masks xy coordinates and zeroes confidence for points outside the visible keypoint t-range, so cameras with dropped endpoints abstain from DLT triangulation rather than voting for straight-line extrapolation.

## What Was Done

Modified `_keypoints_to_midline` in `pose_estimation.py` to post-mask extrapolated points. After the existing `interp1d` calls produce `x_out`, `y_out`, `conf_out`, a masking step sets positions outside `[t_values[0], t_values[-1]]` to `np.nan` and confidence to `0.0`.

Added four new unit tests covering:
- Nose and tail both dropped (t in [0.2, 0.8]) — boundary indices NaN/conf=0, interior finite
- Full range [0.0, 1.0] — no NaN anywhere
- Tail only dropped (t in [0.0, 0.6]) — beyond t=0.6 is NaN/conf=0
- Shape preserved (n_points, 2) / (n_points,) regardless of NaN presence

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 (RED) | ce8da38 | test(25-01): add failing tests for NaN-out extrapolation behavior |
| 2 (GREEN) | e7b29a7 | feat(25-01): NaN-out extrapolated midline points in _keypoints_to_midline |

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `hatch run test tests/unit/core/midline/test_pose_estimation_backend.py` — all tests pass (1146 passed)
- `hatch run lint` — all checks passed
- Pre-existing typecheck errors in pose_estimation.py (lines 63, 69, 76, 171, 535, 537) are unchanged; no new errors introduced

## Self-Check: PASSED

- [x] `src/aquapose/core/midline/backends/pose_estimation.py` — exists, contains `np.nan`
- [x] `tests/unit/core/midline/test_pose_estimation_backend.py` — exists, contains `test_keypoints_to_midline_nan`
- [x] Commit ce8da38 — exists (RED tests)
- [x] Commit e7b29a7 — exists (GREEN implementation)
