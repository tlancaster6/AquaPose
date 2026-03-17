---
phase: quick-28
plan: "01"
subsystem: core/tracking
tags: [interpolation, keypoints, tracking, spline]
dependency_graph:
  requires: [interpolate_gaps]
  provides: [interpolate_low_confidence_keypoints]
  affects: [KeypointTracker.get_tracklets, Tracklet2D.keypoints, Tracklet2D.keypoint_conf]
tech_stack:
  added: []
  patterns: [cubic-spline per-keypoint interpolation, TDD red-green]
key_files:
  created: []
  modified:
    - src/aquapose/core/tracking/keypoint_tracker.py
    - tests/unit/core/tracking/test_keypoint_tracker.py
decisions:
  - "Mutate builder in-place and return it (same pattern as interpolate_gaps, avoids extra allocation)"
  - "Persist kpt_conf_threshold through get_state/from_state for cross-chunk handoff consistency"
  - "Use _T convention for unused dimension variable T to satisfy ruff RUF059"
metrics:
  duration: "~10 minutes"
  completed: "2026-03-17"
  tasks_completed: 1
  files_changed: 2
---

# Quick Task 28: Add Per-Keypoint Interpolation to KeypointTracker Summary

**One-liner:** Per-keypoint cubic spline interpolation fills low-confidence keypoints within detected frames using confident frames as knots, wired into KeypointTracker.get_tracklets after whole-frame gap fill.

## What Was Built

New `interpolate_low_confidence_keypoints` function in `keypoint_tracker.py` that:

- Iterates over each keypoint index (0..K-1) independently
- Builds a CubicSpline from frames where that keypoint's confidence >= `conf_threshold`
- Replaces keypoint positions at low-confidence frames with spline interpolation
- Sets interpolated keypoint conf to 0.0
- Skips keypoints with fewer than 2 confident frames (insufficient for CubicSpline)
- Coasted frames (conf=0.0 from `interpolate_gaps`) are targets, not knot sources

Wired into `KeypointTracker.get_tracklets` immediately after `interpolate_gaps`:

```python
gap_filled.append(
    interpolate_low_confidence_keypoints(
        interpolate_gaps(builder, max_gap_frames=self._max_gap_frames),
        conf_threshold=self._kpt_conf_threshold,
    )
)
```

New `kpt_conf_threshold` parameter added to `KeypointTracker.__init__` (default 0.3), persisted through `get_state`/`from_state` for cross-chunk handoff.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Failing tests for interpolate_low_confidence_keypoints | 03aa762 | tests/unit/core/tracking/test_keypoint_tracker.py |
| GREEN | Implement interpolate_low_confidence_keypoints | a7f072d | src/aquapose/core/tracking/keypoint_tracker.py |

## Tests Added

`TestInterpolateLowConfidenceKeypoints` class with 6 tests:

1. `test_all_confident_keypoints_unchanged` — no modification when all conf > threshold
2. `test_single_low_conf_keypoint_interpolated` — corrupted low-conf kpt is replaced
3. `test_interpolated_keypoint_conf_set_to_zero` — conf=0.0 after interpolation
4. `test_fewer_than_two_confident_frames_left_unchanged` — <2 knots = no interpolation
5. `test_coasted_frames_not_used_as_knot_points` — coasted frames are targets, not sources
6. `test_multiple_keypoints_interpolated_independently` — multiple kpts interpolated per frame

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `src/aquapose/core/tracking/keypoint_tracker.py` — modified, contains `interpolate_low_confidence_keypoints`
- `tests/unit/core/tracking/test_keypoint_tracker.py` — modified, contains `TestInterpolateLowConfidenceKeypoints`
- Commits 03aa762 and a7f072d exist in git log
- All 1220 tests pass, no new lint errors in modified files
