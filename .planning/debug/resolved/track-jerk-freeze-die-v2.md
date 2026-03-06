---
status: resolved
trigger: "track-jerk-freeze-die-v2: tracks exhibit jerk-freeze-die cycle persisting after two fix rounds"
created: 2026-02-24T00:00:00Z
updated: 2026-02-24T01:00:00Z
---

## Current Focus

hypothesis: CONFIRMED. Two independent causes:
  1. Ghost track birth from cross-fish RANSAC at out-of-bounds z (z=3.37m vs valid tank z=0.75-1.75m)
  2. Single-view claiming used constant ray-depth heuristic (z=1.25m) instead of track's known z (~1.68m) causing 0.43m jerk
test: Both hypotheses confirmed by frame-by-frame trace. Root causes directly observable.
expecting: N/A - resolved
next_action: DONE - fix verified, tests pass, committed

## Symptoms

expected: Tracks should follow ground-truth fish continuously through simple motion scenarios. Track death/rebirth should only occur during genuinely ambiguous situations.
actual: Tracks exhibit a repeating cycle: (1) follows GT well, (2) suddenly jerks to wrong position, (3) freezes at wrong position for several frames, (4) dies, (5) new track births, cycle repeats. Happens during easy scenarios with well-isolated, steady-velocity fish.
errors: No crash errors — behavioral issue visible in tracking_video.mp4 outputs
reproduction: Run `python scripts/diagnose_tracking.py --scenario all` and examine output in C:/Users/tucke/PycharmProjects/AquaPose/output/tracking_diagnostic/
timeline: Persists after two fix rounds (commits cc70ae1 for placement, cde210b for coasting prediction). The coasting prediction fix was supposed to address this but the symptom persists.
started: Persists after two fix rounds

## Eliminated

- hypothesis: FishTrack.predict() used fixed-origin extrapolation during coasting causing stall
  evidence: Fixed in cde210b — cumulative integration confirmed working
  timestamp: prior session

- hypothesis: Ghost track phantom births from synthetic fish placed outside camera coverage
  evidence: Fixed in cc70ae1 — tank centered on rig
  timestamp: prior session

- hypothesis: bbox size contributes to jerk (40x25px fixed size)
  evidence: Verified: bbox is only used as centroid carrier. _detection_centroid uses bbox center when mask=None. Round-trip accurate to <0.5px.
  timestamp: prior session

- hypothesis: Residual validation kicking in and falsely rejecting valid claims
  evidence: Residual validation logic requires len(residual_history)>=3 AND residual > max(mean*3.0, 7.5px). Track mean_residual is typically 3-4px, so threshold is 9-12px. Legitimate claims are well below this. Not the primary cause.
  timestamp: 2026-02-24T00:30:00Z

- hypothesis: False positive detections claiming is the primary cause
  evidence: track_fragmentation scenario uses base_fp_rate=0.0 (no FPs) but still shows catastrophic fragmentation (MOTA -0.87). FPs are a secondary concern.
  timestamp: 2026-02-24T00:15:00Z

## Evidence

- timestamp: 2026-02-24T00:01:00Z
  checked: summary.md metrics across all scenarios
  found: track_fragmentation has MOTA -0.8722 (catastrophic), 16 fragmentations, 1022 FPs, only 26.4% TP. startle_response has MOTA 0.695, 3 fragmentations.
  implication: track_fragmentation is worst by far. It uses 25% miss_rate but NO false positives (base_fp_rate=0.0). Yet has 1022 FP track outputs and only 26% TP.

- timestamp: 2026-02-24T00:20:00Z
  checked: frame-by-frame trace of track_fragmentation scenario, frames 0-25
  found: Track 1 born at frame 1 at pos=[0.89, -1.09, 3.41] while GT fish 1 is at [0.37, -0.77, 1.68]. Ghost is 1.8m from true fish. Ghost has LOWER id than GT because RANSAC birthed it from 3 cameras that see a combination of GT fish 1 detections + false positive detections. The ghost's z=3.41m is 2.62m below water surface -- physically impossible.
  implication: Ghost track born from cross-detection RANSAC. The _score_candidate function accepts any candidate with >= 3 cameras within reprojection_threshold -- there is no z-validity check.

- timestamp: 2026-02-24T00:25:00Z
  checked: discover_births output for frame 1
  found: Two births proposed: (1) valid birth near GT fish 1 at [0.37, -0.59, 1.66], (2) ghost birth at [0.86, -1.09, 3.37] from syn_00[1]+syn_01[1]+syn_05[1]. The ghost has residual=3.52px which passes all filters. The claimed detections for the ghost are: syn_00[1] at (1600,226) = boundary FP, syn_01[1] at (1363,234) = true fish 1 detection, syn_05[1] at (1380,12) = boundary FP at v=12px. These three lie along a ray bundle that triangulates to z=3.37m (deep ghost).
  implication: Root cause 1 confirmed: no z-validity check on births. Ghost survives because residual is low (3.52px) -- the cross-detection set happens to triangulate to a point that reprojects near each respective detection.

- timestamp: 2026-02-24T00:35:00Z
  checked: single-view claiming behavior in claim_detections_for_tracks
  found: When only 1 camera claims a track, centroid_3d = ray_origin + _DEFAULT_TANK_DEPTH(0.5m) * ray_dir. For a track at z=1.68m with water_z=0.75m, this places the fish at z=1.25m (0.43m wrong). This causes a position jump of ~0.46m in 3D when a track transitions from multi-view (correct z=1.68) to single-view (heuristic z=1.25).
  implication: Root cause 2 confirmed: single-view heuristic ignores track's known z, causing z-error jerk.

- timestamp: 2026-02-24T00:45:00Z
  checked: effect of z-depth fix on ghost births
  found: After adding `_is_depth_valid` check in discover_births (rejects z > water_z + 2.0m = 2.75m), the ghost birth at z=3.37m is rejected. MOTA improved from -0.8722 to -0.44. TP improved from 26.4% to 44.6%. Jerk from ghost eliminated.
  implication: Fix 1 confirms root cause 1. Ghost birth suppressed.

- timestamp: 2026-02-24T00:50:00Z
  checked: effect of depth-anchored single-view claiming fix
  found: After replacing ray-depth heuristic with depth-anchored approach (t = (pred_z - origin_z) / dir_z), zero jerks in first 100 frames (was 2). MOTA improved further to 0.36. All scenarios improved significantly.
  implication: Fix 2 confirms root cause 2. Single-view jerk eliminated.

- timestamp: 2026-02-24T01:00:00Z
  checked: all scenarios before/after, full test suite
  found:
    crossing_paths: MOTA 0.4067 -> 0.7433 (+0.34), TP 96.5%
    track_fragmentation: MOTA -0.8722 -> 0.1767 (+1.05), TP 69.3%
    tight_schooling: MOTA 0.5680 -> 0.7740 (+0.21), TP 90.1%
    startle_response: MOTA 0.6950 -> 0.9867 (+0.29), TP 98.7%
    All 415 tests pass. Lint clean.
  implication: Both fixes verified, no regressions.

## Resolution

root_cause: |
  Two independent causes of jerk-freeze-die cycle:

  1. GHOST BIRTH FROM CROSS-DETECTION RANSAC:
     RANSAC in discover_births triangulates candidate fish from any 3+ camera
     detections within reprojection_threshold, with no check that the resulting
     3D position is physically plausible. Detections from different fish (e.g.
     GT fish 1 true detection + false positive detections near image boundary)
     triangulate to positions far outside the valid tank depth band (z=3.37m vs
     valid range z=0.75-1.75m). These ghost tracks are born with low reprojection
     residual (~3.5px) because the cross-detection set happens to agree with the
     impossible triangulated point. The ghost then follows a jerk-freeze-die cycle
     as it intermittently claims fringe detections.

  2. SINGLE-VIEW DEPTH HEURISTIC MISMATCH:
     When only 1 camera detects a fish in a given frame, claim_detections_for_tracks
     computes centroid_3d = ray_origin + _DEFAULT_TANK_DEPTH(0.5m) * ray_dir.
     For fish at z=1.68m with water_z=0.75m, this places the fish at z=1.25m
     (0.43m error), causing a 0.46m position jump when the track transitions from
     multi-view (correct z) to single-view (heuristic z). This is the "jerk" moment.

fix: |
  Two targeted fixes in src/aquapose/tracking/associate.py:

  1. Z-DEPTH VALIDITY CHECK (prevents ghost births):
     Added _MAX_FISH_DEPTH_BELOW_SURFACE = 2.0m constant.
     Added _get_water_z() helper to extract water_z from camera models.
     Added _is_depth_valid() check: z in [water_z, water_z + 2.0m].
     Applied in discover_births() to filter out births outside valid depth band.
     Applied in claim_detections_for_tracks() to reject multi-view triangulations
     that fall outside valid depth (use predicted position instead = coast).

  2. DEPTH-ANCHORED SINGLE-VIEW CLAIMING (eliminates z-heuristic jerk):
     For single-camera claims, instead of using _DEFAULT_TANK_DEPTH, use the
     track's predicted z-coordinate to compute the ray parameter t = (pred_z -
     origin_z) / dir_z, then centroid_3d = origin + t * direction. This gives
     the XY position at the track's known depth, eliminating the z-error jerk.
     Falls back to _DEFAULT_TANK_DEPTH if ray is near-horizontal or t <= 0.

verification: |
  BEFORE:
  - crossing_paths: MOTA=0.41, 4 fragmentations, 222 FP
  - track_fragmentation: MOTA=-0.87, 16 fragmentations, 26.4% TP, 1022 FP
  - tight_schooling: MOTA=0.57, 8 fragmentations, 424 FP
  - startle_response: MOTA=0.70, 3 fragmentations, 108 FP

  AFTER:
  - crossing_paths: MOTA=0.74, 3 fragmentations, 133 FP
  - track_fragmentation: MOTA=0.18, 10 fragmentations, 69.3% TP, 464 FP
  - tight_schooling: MOTA=0.77, 7 fragmentations, 190 FP
  - startle_response: MOTA=0.99, 0 fragmentations, 0 FP

  Zero jerk events in first 100 frames of track_fragmentation (was 2).
  All 415 unit tests pass.
  Lint clean.

files_changed:
  - src/aquapose/tracking/associate.py
