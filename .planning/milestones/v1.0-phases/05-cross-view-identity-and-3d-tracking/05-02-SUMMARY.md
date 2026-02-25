---
phase: 05-cross-view-identity-and-3d-tracking
plan: 02
subsystem: tracking
tags: [hungarian-assignment, sort-tracker, track-lifecycle, temporal-tracking]
dependency_graph:
  requires:
    - aquapose.tracking.associate (AssociationResult, FrameAssociations from Plan 01)
    - scipy.optimize.linear_sum_assignment
  provides:
    - aquapose.tracking.FishTrack
    - aquapose.tracking.FishTracker
  affects:
    - Phase 05 Plan 03 (HDF5 serialization consumes confirmed tracks)
tech_stack:
  added: []
  patterns:
    - SORT-derived track lifecycle (birth confirmation, grace period, death)
    - XY-only Hungarian cost matrix to avoid Z-noise ID swaps
    - Constant-velocity prediction from 2-frame deque
    - TRACK-04 population constraint: dead track IDs recycled to new observations
    - First-frame batch init sorted by X for deterministic IDs
key_files:
  created:
    - src/aquapose/tracking/tracker.py
    - tests/unit/tracking/test_tracker.py
  modified:
    - src/aquapose/tracking/__init__.py
decisions:
  - "XY-only cost matrix (ignoring Z) prevents Z-noise induced ID swaps in a top-down 13-camera rig where Z uncertainty is 132x larger than XY"
  - "TRACK-04 population constraint: dead track fish_id recycled to first unmatched observation in same frame — prevents ID inflation when fish are temporarily lost"
  - "First-frame batch init sorted by X centroid for deterministic ID assignment (ID 0 = leftmost fish)"
  - "mark_missed() resets hit_streak to 0 so re-appearing tracks must re-confirm — prevents ghost confirmation"
metrics:
  duration: "4 minutes"
  completed: "2026-02-21"
  tasks_completed: 2
  files_created: 2
  files_modified: 1
---

# Phase 05 Plan 02: Hungarian Tracker and Track Lifecycle Summary

SORT-derived temporal fish tracker with Hungarian XY-only assignment, constant-velocity prediction, and population-constraint ID recycling.

## What Was Built

### `src/aquapose/tracking/tracker.py`

**`FishTrack` dataclass:**
- `positions: deque[np.ndarray]` (maxlen=2) for constant-velocity prediction
- `predict()`: extrapolates from two positions; returns copy at single position
- `update(association)`: appends centroid, resets `frames_since_update`, increments `hit_streak`, sets `is_confirmed` once streak >= min_hits
- `mark_missed()`: increments `frames_since_update`, resets `hit_streak` to 0
- `is_dead` property: True when `frames_since_update > max_age`
- Carries per-frame metadata: `camera_detections`, `reprojection_residual`, `confidence`, `n_cameras`

**`FishTracker` class:**
- Constructor parameters: `min_hits=2`, `max_age=7`, `max_distance=0.1m`, `expected_count=9`
- `update(frame_associations)` — full pipeline per frame:
  1. Frame 0: batch-init tracks sorted by X for deterministic IDs
  2. Predict positions for all active tracks via `track.predict()`
  3. Build `(n_tracks x n_obs)` XY-only cost matrix (Euclidean distance in metres)
  4. `linear_sum_assignment` (Hungarian solve)
  5. Gate: reject matches where cost > max_distance
  6. Update matched tracks; mark_missed for unmatched
  7. TRACK-04: zip dead_ids with unmatched_obs_indices — recycle IDs to new observations
  8. Create fresh tracks for any remaining unmatched observations
  9. Prune dead tracks from internal list
  10. Return confirmed tracks only
- `get_all_tracks()`: all tracks including tentative
- `get_seed_points()`: centroids of confirmed tracks for prior-guided RANSAC in next frame; None if none confirmed

### `tests/unit/tracking/test_tracker.py`

13 unit tests, all synthetic (no GPU, no real data):

| Test | What it verifies |
|------|-----------------|
| `test_single_fish_track_across_frames` | Stable fish_id across 5 linear frames |
| `test_single_fish_confirmed_after_min_hits` | Not confirmed frame 0; confirmed frame 1 |
| `test_two_fish_no_swap` | Stable IDs for two separated fish over 10 frames |
| `test_birth_confirmation` | Tentative → confirmed at min_hits threshold |
| `test_grace_period_and_death` | Survives max_age misses; pruned at max_age+1 |
| `test_population_constraint_relinking` | Dead ID recycled to new observation (TRACK-04) |
| `test_constant_velocity_prediction` | [0,0,0] + [1,0,0] predicts [2,0,0] |
| `test_constant_velocity_single_position` | Zero-velocity fallback with one position |
| `test_xy_only_cost_matrix` | Correct match despite Z swap between frames |
| `test_max_distance_gate` | 1.0m observation creates new track, old track in grace |
| `test_first_frame_batch_init` | 9 tracks IDs assigned in ascending X order |
| `test_get_seed_points` | None before confirmation; centroids after |
| `test_get_seed_points_multiple_confirmed` | One seed per confirmed track |

## Deviations from Plan

None — plan executed exactly as written. The plan listed 10 tests; 13 were implemented (3 additional: `test_single_fish_confirmed_after_min_hits`, `test_constant_velocity_single_position`, `test_get_seed_points_multiple_confirmed`) for more complete coverage.

## Verification Results

```
hatch run check                              → ruff: All checks passed; basedpyright: 0 errors in tracking/
hatch run test tests/unit/tracking/         → 264 passed (20 associate + 13 tracker + 231 other)
python -c "from aquapose.tracking import FishTracker, FishTrack"  → OK
```

Pre-existing basedpyright errors in `detector.py` (4 errors) are out of scope.

## Self-Check: PASSED

Files exist:
- FOUND: src/aquapose/tracking/tracker.py
- FOUND: tests/unit/tracking/test_tracker.py
- FOUND: src/aquapose/tracking/__init__.py (updated)

Commits exist:
- FOUND: 226545e (feat(05-02): implement FishTracker with Hungarian assignment and track lifecycle)
- FOUND: ad302b3 (test(05-02): add unit tests for FishTracker lifecycle and Hungarian assignment)
