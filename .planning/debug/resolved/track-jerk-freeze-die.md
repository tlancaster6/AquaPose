---
status: resolved
trigger: "track-jerk-freeze-die"
created: 2026-02-24T00:00:00Z
updated: 2026-02-24T01:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — The coasting prediction model accumulates error faster than reprojection_threshold allows re-acquisition. During coasting, predict() applies damping^frames_since_update to velocity from the FIXED last observed position, not from a cumulative coasting trajectory. After max_age=7 frames at 0.05m/s fish speed with 3111px/m scale, prediction error = ~36px >> reprojection_threshold=15px. Fish cannot be re-acquired; track dies. New track births immediately. Cycle repeats.
test: Compute prediction error analytically for track_fragmentation params
expecting: Error after 7 coasting frames = last_pos + vel*0.8^7 vs true_pos = last_pos + vel*7 → error = vel*(7 - 0.8^7) ≈ vel*6.79 ≈ 11.5mm ≈ 36px. Confirmed.
next_action: Fix predict() to use cumulative position integration during coasting (NOT single-step projection from fixed last_pos). Use constant velocity (no damping) during coasting so the prediction tracks the fish.

## Symptoms

expected: Tracks should follow ground-truth fish continuously through simple motion scenarios (isolated fish, consistent velocity). Track death/rebirth should only occur during genuinely ambiguous situations (crossings, occlusions).
actual: Tracks exhibit a repeating cycle: (1) track follows GT fish well with minor jitter, (2) track marker suddenly jerks to a wrong position, (3) track freezes at wrong position for several frames, (4) track dies, (5) new track births and cycle repeats. This happens during easy scenarios (isolated, steady motion).
errors: No crash errors — behavioral issue visible in tracking_video.mp4 outputs
reproduction: Run diagnose_tracking.py and examine output videos in output/tracking_diagnostic/
timeline: Visible after the phantom birth fix (commit cc70ae1). The previous issue (too many phantom tracks) was masking this subtler problem.

## Eliminated

- hypothesis: Single-view ray-depth heuristic centroid causes jerk (update_position_only snaps to wrong XY)
  evidence: With all overhead cameras, rays are nearly vertical; ray_origin + 0.5*ray_dir gives correct XY (wrong Z only). For claim matching via 2D reprojection, the Z error doesn't affect 2D projection in overhead cameras. XY from single-view is accurate enough. Not the root cause.
  timestamp: 2026-02-24T00:10:00Z

- hypothesis: expected_count=9 always triggers birth causing phantom tracks every frame
  evidence: diagnose_tracking.py sets expected_count=n_fish (line 1199). For track_fragmentation with 2 fish, expected_count=2, so birth only triggers when non_dead_count < 2. Not always-on.
  timestamp: 2026-02-24T00:11:00Z

- hypothesis: False positives cause bad multi-view claims (jerk)
  evidence: track_fragmentation uses base_false_positive_rate=0.0. No FPs in this scenario. FPs are present in other scenarios but are not the primary fragmentation mechanism here.
  timestamp: 2026-02-24T00:12:00Z

## Evidence

- timestamp: 2026-02-24T00:01:00Z
  checked: tracking_report.md for track_fragmentation scenario
  found: -0.96 MOTA, 16 confirmed tracks for 2 GT fish, 586 FN (65%), 1182 FP (131%), 9 fragmentations. Partially tracked 38% and 32%.
  implication: Extreme fragmentation — each GT fish is being tracked in short bursts (~30-50 frames) before a jerk+die cycle. 16 tracks for 2 fish = ~8 track fragments per fish over 450 frames.

- timestamp: 2026-02-24T00:02:00Z
  checked: tracker.py claim rejection logic (lines 417-431)
  found: Rejection condition: `claim.reprojection_residual > max(mean_res * residual_reject_factor, residual_floor)` where residual_floor = reprojection_threshold * 0.5 = 7.5px (DEFAULT_REPROJECTION_THRESHOLD=15.0, DEFAULT_RESIDUAL_REJECT_FACTOR=3.0). A track is only rejected if residual exceeds the LARGER of (mean * 3) or 7.5px.
  implication: This check should be PROTECTING against jerk, but the jerk still happens. The jerk must be a different event — a claim IS accepted with bad data.

- timestamp: 2026-02-24T00:03:00Z
  checked: claim_detections_for_tracks in associate.py
  found: Claiming is purely greedy by pixel_distance: sorted by (dist, priority) ascending. A track claims the nearest detection within reprojection_threshold in each camera. After single-view claiming, centroid is placed at DEFAULT_TANK_DEPTH=0.5m along the ray — this is a heuristic position NOT the true fish 3D location. If the track happens to claim the WRONG detection in one camera (close in 2D projection space but from a different fish or noise), the triangulated 3D centroid could jump far from the true fish position.
  implication: The "jerk" is likely a bad multi-view claim: two cameras each contribute a detection, but one or both detections belong to a different fish / noise artifact. The triangulated position is wrong and lands far from the true fish location.

- timestamp: 2026-02-24T00:04:00Z
  checked: associate.py claim_detections_for_tracks — what happens after a bad claim is accepted
  found: After the jerk (bad centroid accepted), `update_from_claim` is called: velocity is recomputed as `centroid_3d - prev_pos` = large jump vector. On subsequent frames the PREDICTION is now wrong: it predicts far from the fish. The real detections are outside reprojection_threshold from the wrong prediction. So the track misses on the next frame → mark_missed → COASTING. It keeps predicting away from reality → frames_since_update grows → dies after max_age=7 frames.
  implication: One bad accepted claim → wrong velocity → no future matches → death. This explains the exact jerk→freeze→die pattern.

- timestamp: 2026-02-24T00:05:00Z
  checked: track_fragmentation scenario parameters
  found: miss_rate=0.25 means 25% of detections are randomly dropped. With 12 cameras, a fish typically visible in ~4 cameras at any moment. With 25% miss rate, expect ~1 camera's detection to be missing per frame. This means most frames still have 3+ cameras with detections. The bad claim should come from a different source.
  implication: The miss rate creates occasional single-camera frames. But the "jerk" seems to happen even with multiple cameras present. The issue is likely in the claiming logic when two fish are nearby in 2D projection space.

- timestamp: 2026-02-24T00:06:00Z
  checked: claim_detections_for_tracks candidate sorting — does a track claim the nearest detection PER camera independently?
  found: YES. The code builds all (dist, priority, track_id, cam_id, det_idx) tuples across all cameras, sorts globally by distance, then greedily assigns. This means each track gets AT MOST ONE detection per camera (enforced by `if cam_id in track_claims.get(track_id, {}): continue`). But a detection can only go to ONE track (enforced by `if det_idx in taken_detections.get(cam_id, set()): continue`).
  implication: The claim mechanism is sound for well-separated fish. But if the 25% miss rate causes a frame where only ONE camera sees a fish, the claimed 3D centroid uses the ray-depth heuristic (0.5m along ray) — this is inaccurate! The heuristic can be METERS off from the true position in XY if the camera is oblique.

- timestamp: 2026-02-24T00:07:00Z
  checked: Single-view claim handling (lines 804-828 in associate.py)
  found: Single-camera claim places centroid at `ray_origin + 0.5 * ray_dir`. _DEFAULT_TANK_DEPTH=0.5m. Then in tracker.py lines 433-446: `if claim.n_cameras == 1: track.update_position_only(...)` — this updates POSITION but freezes velocity. So the position gets the bad ray-depth centroid, but velocity is frozen.
  implication: A single-view frame does a bad position update but at least velocity is frozen. This could produce a position "snap" to the wrong location on single-camera frames. But velocity is frozen so NEXT frame prediction is still based on old velocity + bad position → predicts from wrong base location → may miss real detections.

- timestamp: 2026-02-24T00:08:00Z
  checked: The "freeze" behavior — what happens when track position snaps but velocity is frozen
  found: After `update_position_only(bad_centroid)`, the track's last position is now at bad_centroid. Next predict() = bad_centroid + frozen_velocity. If this predicted position is far from real fish (because bad_centroid was far), the real fish detection will be outside reprojection_threshold from the predicted position. Track misses, goes COASTING, eventually dies. The "freeze" in the video is the track coasting: it stays at bad_centroid + damped_velocity which is still in the wrong part of the tank.
  implication: CONFIRMED: single-view claims with ray-depth heuristic ARE the "jerk" event. The position update_position_only() is the root cause of the jerk-freeze-die cycle.

## Resolution

root_cause: FishTrack.predict() computed `last_pos + vel * damping^frames_since_update` from the FIXED last observed position on every call. During coasting (missed detection frames), this stalls the prediction near the last known position while the fish keeps moving. After max_age=7 frames at 0.05m/s fish speed with pixel scale ~3111 px/m, prediction error = ~36px >> reprojection_threshold=15px. Track cannot re-acquire the fish → dies → new track births immediately → jerk-freeze-die cycle.

fix: Changed predict() to return a running `_coasting_prediction` field. mark_missed() now advances this prediction by `vel * damping` each coasting frame (cumulative integration), keeping the prediction tracking the fish trajectory. _coasting_prediction is reset to None on update_from_claim/update_position_only. Non-coasting predict() unchanged: `last_pos + velocity`.

verification: All 415 unit tests pass. track_fragmentation MOTA improved from -0.96 to -0.08. New regression test `test_coasting_prediction_tracks_fish` confirms prediction error < 5mm after 7 coasting frames (was ~11mm / ~36px with old code).

files_changed:
  - src/aquapose/tracking/tracker.py: FishTrack._coasting_prediction field + predict() + mark_missed() + reset in update_from_claim/update_position_only
  - tests/unit/tracking/test_tracker.py: updated test_coasting_velocity_damping + new test_coasting_prediction_tracks_fish
