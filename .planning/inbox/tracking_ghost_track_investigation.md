# Ghost Track Investigation — Synthetic Tracking Diagnostic

## Summary

The `diagnose_tracking.py` script produces significantly more tracks than ground truth fish, with persistent phantom tracks. Two bugs were found and fixed, but results remain poor even with the real calibration rig.

## Bugs Found and Fixed (committed in 1fac02d)

### 1. `diagnose_tracking.py` — bad parameter defaults (PRIMARY BUG)
The script hardcoded `min_cameras_birth=2`, overriding FishTracker's default of 3. With only 2 cameras required, RANSAC can pair detections from different fish across cameras and triangulate phantom 3D points that by construction satisfy the 2-camera check. Fixed by deferring to FishTracker defaults.

### 2. `tracker.py` — birth proximity check only looked at confirmed tracks
During the probationary window (first ~5 frames), no tracks are confirmed, so the proximity check that prevents duplicate births was ineffective. Fixed by checking all non-dead tracks (`not t.is_dead` instead of `t.is_confirmed`).

### 3. `associate.py` — seed_points not forwarded to discover_births
`ransac_centroid_cluster` accepts `seed_points` and `FishTracker.get_seed_points()` exists, but the wire through `discover_births()` was missing. Fixed.

## Remaining Problem

Even with fixes and using the real calibration JSON (12 cameras, real geometry), tracking results are still poor — too many phantom tracks persist.

## Uninvestigated Leads

### Tank geometry mismatch
`TankConfig` defaults: radius=1.0m, depth=1.0m. When using real calibration, only `water_z` is patched (line 1164 of diagnose_tracking.py) — radius and depth remain at defaults. If the real tank dimensions differ, synthetic fish may swim in regions with poor camera coverage or outside the calibrated volume entirely. **This could cause many detections to fall in degenerate geometry zones where RANSAC easily creates phantoms.**

### Fabricated rig camera count
Default `--n-cameras 4` creates a 4x4=16 camera grid. The real rig has 12 usable cameras (13 minus the skipped center camera). More cameras means more cross-fish detection pairs available to RANSAC, increasing phantom birth probability. When using real calibration this is moot, but worth noting for fabricated-rig testing.

### Synthetic detection quality
`generate_detection_dataset` produces noise-free centroid detections with `confidence=1.0` and `mask=None`. The tracker's claiming and RANSAC logic may behave differently with perfect detections vs real noisy ones — worth checking if any codepath depends on confidence or detection spread.

### RANSAC susceptibility with many cameras
With 12+ cameras all seeing both fish, RANSAC samples 2 cameras and 1 detection each. The probability of sampling one detection from each of two different fish is high when there are many cameras with multiple detections. The `min_cameras=3` check helps but may not be sufficient when fish are close together (crossing scenarios) — a phantom midpoint between two real fish could reproject reasonably onto detections of either fish in multiple cameras.

## Key Files
- `scripts/diagnose_tracking.py` — diagnostic script, tracker setup and visualization
- `src/aquapose/tracking/tracker.py` — FishTracker, birth logic, proximity check
- `src/aquapose/tracking/associate.py` — RANSAC centroid clustering, claiming, discover_births
- `src/aquapose/synthetic/trajectory.py` — TankConfig, fish trajectory generation
- `src/aquapose/synthetic/detection.py` — synthetic detection generation
- `src/aquapose/synthetic/rig.py` — fabricated camera rig builder
