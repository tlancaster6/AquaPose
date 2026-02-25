---
phase: quick-8
plan: 01
subsystem: tracking
tags: [velocity-smoothing, ring-buffer, fish-tracker, noise-reduction]
dependency_graph:
  requires: []
  provides: [windowed-velocity-smoothing]
  affects: [tracking/tracker.py]
tech_stack:
  added: []
  patterns: [ring-buffer-averaging, post-init-dataclass-fields]
key_files:
  created: []
  modified:
    - src/aquapose/tracking/tracker.py
    - tests/unit/tracking/test_tracker.py
decisions:
  - "__post_init__ used to initialize velocity_history deque with maxlen=velocity_window, since deque maxlen must depend on velocity_window field value"
  - "velocity_history field declared with field(init=False, repr=False) to exclude from repr and constructor"
  - "velocity_window=1 is the exact backward-compatible mode — produces identical output to pre-change single-delta behaviour"
metrics:
  duration_seconds: 763
  completed_date: "2026-02-24"
  tasks_completed: 2
  files_modified: 2
---

# Quick Task 8: Windowed Velocity Smoothing for Tracking Summary

Windowed velocity smoothing in FishTrack via a deque ring buffer averaging the last N frame-to-frame position deltas (default N=5), replacing the single noisy frame delta previously used.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add velocity_history ring buffer and windowed smoothing to FishTrack | b73b123 | src/aquapose/tracking/tracker.py |
| 2 | Add 5 unit tests for windowed velocity smoothing | 0cf5295 | tests/unit/tracking/test_tracker.py |

## What Was Built

### Task 1: velocity_history ring buffer

Added to `src/aquapose/tracking/tracker.py`:

- `DEFAULT_VELOCITY_WINDOW: int = 5` module constant
- `velocity_window: int` field on `FishTrack` dataclass (default `DEFAULT_VELOCITY_WINDOW`)
- `velocity_history: deque[np.ndarray]` field (init=False, repr=False) initialized via `__post_init__` with `maxlen=velocity_window`
- `update_from_claim` now: computes raw `delta = centroid_3d - prev`, appends to `velocity_history`, sets `self.velocity = np.mean(list(velocity_history), axis=0)`
- `update_position_only` unchanged — velocity and velocity_history remain frozen for single-view updates
- `FishTracker.__init__` accepts `velocity_window` parameter, stores as `self.velocity_window`, passes to `_create_track`

### Task 2: Unit tests

Added 5 tests after `test_coasting_velocity_damping`:

1. `test_windowed_velocity_smoothing_averages_deltas` — window=3, verifies mean of [1,1,3] deltas = [5/3, 0, 0]
2. `test_windowed_velocity_window_1_matches_raw_delta` — window=1 reproduces last-delta-only behaviour
3. `test_windowed_velocity_single_view_does_not_update_history` — velocity_history length stays at 1 after update_position_only
4. `test_windowed_velocity_prediction_uses_smoothed` — predict() returns last_pos + smoothed_velocity
5. `test_windowed_velocity_coasting_uses_smoothed` — mark_missed advances coasting by smoothed velocity (damping=1.0)

## Test Results

- All 17 pre-existing tracker tests: PASS
- All 5 new windowed velocity tests: PASS
- All associate tests: PASS
- Total tracking suite: 421 passed

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

Files exist:
- `src/aquapose/tracking/tracker.py` — FOUND
- `tests/unit/tracking/test_tracker.py` — FOUND

Commits exist:
- b73b123 (Task 1: feat) — FOUND
- 0cf5295 (Task 2: test) — FOUND

All 5 required test names present in test file:
- `test_windowed_velocity_smoothing_averages_deltas` — FOUND
- `test_windowed_velocity_window_1_matches_raw_delta` — FOUND
- `test_windowed_velocity_single_view_does_not_update_history` — FOUND
- `test_windowed_velocity_prediction_uses_smoothed` — FOUND
- `test_windowed_velocity_coasting_uses_smoothed` — FOUND

`velocity_history` in tracker.py — FOUND
