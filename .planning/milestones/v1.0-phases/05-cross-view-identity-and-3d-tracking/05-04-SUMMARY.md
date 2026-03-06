---
phase: 05-cross-view-identity-and-3d-tracking
plan: 04
subsystem: tracking
tags: [track-driven-association, refactor, claiming, lifecycle, velocity]
dependency_graph:
  requires:
    - aquapose.tracking.tracker (FishTrack, FishTracker from Plan 02)
    - aquapose.tracking.associate (ransac_centroid_cluster, _cast_rays_for_detections from Plan 01)
    - aquapose.calibration.projection (RefractiveProjectionModel, triangulate_rays)
    - aquapose.segmentation.detector (Detection)
  provides:
    - aquapose.tracking.TrackState
    - aquapose.tracking.TrackHealth
    - aquapose.tracking.ClaimResult
    - aquapose.tracking.claim_detections_for_tracks
    - aquapose.tracking.discover_births
    - Refactored FishTracker.update(detections, models) API
  affects:
    - All scripts consuming FishTracker (visualize_tracking, reprojection_diagnostic, etc.)
    - TrackingWriter (is_confirmed now a property — writer unchanged, tests updated)
tech_stack:
  added: []
  patterns:
    - Track-driven claiming replaces two-stage RANSAC+Hungarian pipeline
    - Greedy assignment sorted by (pixel_distance, priority) for conflict resolution
    - State machine lifecycle: PROBATIONARY → CONFIRMED → COASTING → DEAD
    - Velocity with exponential damping during coasting
    - Residual validation with history floor to prevent false rejection
key_files:
  created: []
  modified:
    - src/aquapose/tracking/tracker.py
    - src/aquapose/tracking/associate.py
    - src/aquapose/tracking/__init__.py
    - tests/unit/tracking/test_tracker.py
    - tests/unit/tracking/test_associate.py
    - tests/unit/tracking/test_writer.py
    - scripts/visualize_tracking.py
    - scripts/reprojection_diagnostic.py
    - scripts/visualize_reprojection_diagnostic.py
key_decisions:
  - "Replaced two-stage pipeline (anonymous RANSAC → Hungarian) with track-driven claiming — existing tracks project into cameras and claim detections directly via reprojection proximity"
  - "RANSAC demoted to birth-only — only runs on unclaimed detections when track deficit detected or on periodic birth_interval"
  - "Residual validation requires ≥3 history entries and uses max(mean * factor, threshold * 0.5) floor — prevents false rejection when baseline residual is near zero (e.g., overhead cameras with sub-pixel RANSAC error)"
  - "Probationary tracks have a short leash: die after 2 missed frames (vs max_age for confirmed/coasting)"
  - "is_confirmed kept as @property returning state in (CONFIRMED, COASTING) for writer backward compatibility"
  - "Removed max_distance parameter (Hungarian matching gone); replaced with reprojection_threshold (pixel-space)"
patterns_established:
  - "Track-driven claiming: project predicted 3D → pixel coords per camera → greedy nearest-detection assignment"
  - "Priority-based conflict resolution: confirmed tracks (priority 0) claim before probationary (priority 1)"
  - "Single-view penalty: update position but freeze velocity, increment degraded_frames"
  - "Birth rate limiting: RANSAC only when frame_count==0, confirmed_count < expected, excess unclaimed detections, or periodic interval"
requirements_completed:
  - TRACK-01 (temporal identity preserved via track-driven claiming)
  - TRACK-02 (birth confirmation via PROBATIONARY → CONFIRMED state machine)
  - TRACK-03 (grace period via CONFIRMED → COASTING → DEAD with velocity damping)
  - TRACK-04 (population constraint — dead track ID recycling preserved)
duration: 30min
completed: 2026-02-21
---

# Phase 05 Plan 04: Track-Driven Association Refactor Summary

Replaced the two-stage anonymous RANSAC + Hungarian matching pipeline with track-driven claiming, where existing tracks project into cameras and claim detections directly via reprojection proximity. RANSAC demoted to birth-only for discovering new fish.

## Motivation

The original pipeline produced 22 unique IDs for 9 fish over 60 frames. Root cause: stochastic RANSAC destroyed identity information, forcing the tracker to re-discover it via XY proximity — which failed when prediction drifted during coverage gaps.

## Architecture Change

- **Old**: `detections → RANSAC → anonymous 3D → Hungarian matching → lifecycle`
- **New**: `detections → track claiming (project + greedy) → triangulate → update | unclaimed → birth RANSAC → probationary tracks`

## Accomplishments

### New Types
- `TrackState` enum: `PROBATIONARY | CONFIRMED | COASTING | DEAD`
- `TrackHealth` dataclass: rolling residual/camera history, degraded_frames counter
- `ClaimResult` dataclass: per-track claiming output with triangulated centroid

### Refactored FishTrack
- `is_confirmed` changed from stored field to `@property` (writer compat)
- Explicit `velocity` vector with exponential damping during coasting
- `update_from_claim()` / `update_position_only()` replace old `update(AssociationResult)`
- Probationary short leash: dead after 2 misses (vs `max_age` for confirmed)

### New Association Functions
- `claim_detections_for_tracks()`: project predicted positions → greedy nearest-detection assignment sorted by (pixel_distance, priority)
- `discover_births()`: filter to unclaimed detections, run RANSAC, return multi-view associations

### Rewritten FishTracker.update()
- Single method: `update(detections_per_camera, models, frame_index)` — no more external RANSAC call
- Phase 1 (claiming) → Phase 2 (birth discovery) → Phase 3 (lifecycle pruning)
- Residual validation with history-based rejection (≥3 entries, absolute floor)
- Birth rate limiting: triggers on deficit, excess unclaimed, or periodic interval

### Tests
- **test_tracker.py**: 14 tests rewritten for new API with synthetic `RefractiveProjectionModel` rigs
- **test_associate.py**: 5 new tests in `TestClaimDetectionsForTracks` class
- **test_writer.py**: Updated for `state`-based API, removed stale integration test

### Scripts Updated
- `visualize_tracking.py`, `reprojection_diagnostic.py`, `visualize_reprojection_diagnostic.py`: replaced 3-line RANSAC+update pattern with single `tracker.update(dets, models)`

## Files Modified

| File | Change |
|------|--------|
| `src/aquapose/tracking/tracker.py` | `TrackState`, `TrackHealth`, refactored `FishTrack`, rewritten `FishTracker.update()` |
| `src/aquapose/tracking/associate.py` | `ClaimResult`, `claim_detections_for_tracks()`, `discover_births()` |
| `src/aquapose/tracking/__init__.py` | 5 new exports |
| `tests/unit/tracking/test_tracker.py` | 14 tests rewritten for new API |
| `tests/unit/tracking/test_associate.py` | 5 new claiming tests |
| `tests/unit/tracking/test_writer.py` | Updated for state-based FishTrack |
| `scripts/visualize_tracking.py` | Simplified to single update() call |
| `scripts/reprojection_diagnostic.py` | Same simplification |
| `scripts/visualize_reprojection_diagnostic.py` | Same simplification |

## Decisions Made

1. **Residual validation floor**: With overhead cameras producing near-zero RANSAC residuals, a naive `residual > mean * 3.0` rejects legitimate claims. Fixed with `max(mean * factor, threshold * 0.5)` and requiring ≥3 history entries.
2. **Removed max_distance parameter**: Hungarian XY-distance matching is gone; replaced entirely by pixel-space `reprojection_threshold`.
3. **Single-view penalty**: Updates position but freezes velocity to prevent noisy single-camera observations from corrupting motion prediction.
4. **Birth interval**: Default 5 frames between periodic RANSAC. Immediate triggers for deficit or first frame.

## Verification

- **279 tests pass** (full unit suite)
- **Lint clean** (ruff check — 0 errors)
- **Typecheck**: only pre-existing `detector.py` errors (unrelated to changes)

## Next Steps

- Run `scripts/visualize_tracking.py` on real data to verify unique IDs approach 9
- Tune `reprojection_threshold` and `birth_interval` based on real camera geometry
- Consider adding track merge logic for cases where two probationary tracks converge on the same fish

---
*Phase: 05-cross-view-identity-and-3d-tracking*
*Completed: 2026-02-21*
