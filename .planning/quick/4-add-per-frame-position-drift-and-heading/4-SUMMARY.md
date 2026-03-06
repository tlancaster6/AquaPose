---
phase: quick-4
plan: "01"
subsystem: synthetic
tags: [synthetic, drift, testing, fish-config]
dependency_graph:
  requires: []
  provides: [per-frame-drift, velocity-field, angular-velocity-field]
  affects: [generate_synthetic_midline_sets, diagnose_pipeline]
tech_stack:
  added: []
  patterns: [dataclass-defaults-for-backward-compat, linear-drift-per-frame]
key_files:
  created: []
  modified:
    - src/aquapose/synthetic/fish.py
    - scripts/diagnose_pipeline.py
    - tests/unit/synthetic/test_synthetic.py
decisions:
  - "Zero-default drift fields ensure full backward compatibility — all existing tests pass unchanged"
  - "Drift applied as linear function of frame_offset (not frame_index) so frame_start has no effect on within-sequence drift"
  - "drifted_cfg preserves all original fields (curvature, scale, n_points, velocity, angular_velocity) so drift is stable across frames"
metrics:
  duration: ~8 min
  completed: "2026-02-23T16:56:40Z"
  tasks_completed: 2
  files_modified: 3
---

# Quick Task 4: Add Per-Frame Position Drift and Heading Summary

Per-frame velocity and angular_velocity fields added to FishConfig, enabling synthetic fish to swim through the scene across multi-frame sequences.

## What Was Built

Added `velocity: tuple[float, float, float]` and `angular_velocity: float` fields to `FishConfig` with zero defaults (backward compatible). In `generate_synthetic_midline_sets`, each frame now builds a `drifted_cfg` that applies linear displacement: `position_t = position + frame_offset * velocity` and `heading_t = heading_rad + frame_offset * angular_velocity`. The `diagnose_pipeline.py` `_run_synthetic()` function now gives odd-indexed fish non-zero drift (velocity=(0.002, 0.001, 0.0), angular_velocity=0.05) while even-indexed fish remain stationary.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add drift fields to FishConfig and apply per-frame drift | f4f9ae9 | src/aquapose/synthetic/fish.py |
| 2 | Update diagnose_pipeline defaults and add drift unit tests | 48b12b5 | scripts/diagnose_pipeline.py, tests/unit/synthetic/test_synthetic.py |

## Tests

14 synthetic unit tests pass (11 pre-existing + 3 new):
- `test_drift_position_changes_across_frames`: X centroid increases monotonically; frame 4 matches initial_x + 4*0.01 within 1mm
- `test_drift_heading_changes_across_frames`: rotating fish spine direction changes across frames; static fish control points identical
- `test_zero_drift_matches_static`: zero-drift FishConfig produces identical control points across all frames

Pre-existing test failures (4 unrelated tests in triangulation, tracker, and overlay) were present before this change and are out of scope.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

- [x] src/aquapose/synthetic/fish.py has `velocity` and `angular_velocity` fields
- [x] generate_synthetic_midline_sets creates `drifted_cfg` per frame
- [x] tests/unit/synthetic/test_synthetic.py contains `test_drift_*` tests
- [x] diagnose_pipeline.py uses `velocity` and `angular_velocity` for odd-indexed fish
- [x] Commits f4f9ae9 and 48b12b5 exist

## Self-Check: PASSED
