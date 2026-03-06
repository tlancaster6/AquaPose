---
status: resolved
trigger: "Spline reconstruction via direct triangulation yields very few fish, and the ones it does yield are noticeably offset from all fish in all views."
created: 2026-02-27T00:00:00Z
updated: 2026-02-27T00:10:00Z
---

## Current Focus

hypothesis: THREE COMPOUNDING ROOT CAUSES IDENTIFIED (see Hypotheses section below). The primary root cause is H1: the overlap-reliability weight (w = t_shared / t_saturate) in score_tracklet_pair collapses scores to ~0.30 max when only 30 frames are available (t_saturate=100). This prevents most cross-camera edges from being formed. The few fish that do get grouped (fish_9) then fail either in MidlineStage centroid lookup (5px tolerance) or the final spline has only 1 valid frame.
test: H1 proven mathematically by the scoring formula. H2 is secondary and needs runtime data to confirm. H3 (spline offset) is a CONSEQUENCE of H1/H2, not a separate root cause.
expecting: Fixing H1 alone (lower t_saturate or lower score_min to match 30-frame window) should unlock cross-camera groupings, producing multiple fish groups with 3+ cameras each.
next_action: Present findings to user — investigation complete, fix direction identified. No code changes made.

## Symptoms

expected: The direct triangulation pipeline should reconstruct 3D splines that, when reprojected, overlay accurately on the fish in each camera view. Multiple fish should be reconstructed.
actual: Only 1 fish reconstructed (fish_9). Only 1 valid frame (frame 2, confidence=0.37). The reconstructed spline is offset from all fish in all camera views.
errors: No crash errors -- pipeline runs to completion.
reproduction: Run the pipeline on C:\Users\tucke\aquapose\runs\run_20260227_205841
timeline: v2.1 pipeline after phases 22-24, 27-28 implemented. This is a real-data run with 4 test cameras and 30 frames.

## Eliminated

- hypothesis: "Stub stages (AssociationStage or MidlineStage) are causing empty output"
  evidence: timing.txt shows all 5 stages ran with meaningful times (AssociationStage 6.3s, MidlineStage 16.7s, ReconstructionStage 4.8s). These are real implementations, not stubs.
  timestamp: 2026-02-27T00:00:00Z

- hypothesis: "_run_legacy path running (tracklet_groups is None)"
  evidence: _assemble_midline_set_legacy assigns one fish_id per detection (sequential counter). Fish_ids would each have only 1 camera. ALL would fail min_cameras=3. Output would be entirely empty (no fish at all). But the output HAS fish_9 with 1 valid frame. Therefore _run_with_tracklet_groups ran -- tracklet_groups was non-empty.
  timestamp: 2026-02-27T00:00:00Z

## Evidence

- timestamp: 2026-02-27T00:00:00Z
  checked: outputs.h5
  found: Only 1 fish group (fish_9). Confidence array has only frame 2 with confidence=0.37; all other frames are 0.0 or NaN. Spline XYZ for frame 2: approx (-1.07, 0.52, 1.55) world metres.
  implication: Catastrophically underproducing fish -- 9 expected, 1 produced, valid in only 1 of 30 frames.

- timestamp: 2026-02-27T00:00:00Z
  checked: timing.txt
  found: All 5 real stages ran (not stubs).
  implication: The bug is in logic, not missing implementations.

- timestamp: 2026-02-27T00:00:00Z
  checked: config.yaml against scoring formula in scoring.py
  found: The scoring formula is: score = f * (1 - mean_ghost) * w, where w = min(t_shared, t_saturate) / t_saturate. Config: t_saturate=100, stop_frame=30. This means w_max = 30/100 = 0.30. Even with perfect inlier fraction (f=1.0) and no ghost penalty (ghost=0.0), maximum achievable score is 0.30. But score_min=0.30. So any slight imperfection (f<1.0 or ghost>0) pushes the score below threshold.
  implication: The overlap reliability weight was designed for long videos (hundreds of frames). With only 30 frames, it nearly guarantees all pairwise scores fall below the minimum threshold. This is the PRIMARY ROOT CAUSE.

- timestamp: 2026-02-27T00:00:00Z
  checked: config.yaml tracking params
  found: n_init=3, max_coast_frames=30, stop_frame=30. With n_init=3, tracks only confirm after 3 consecutive detections. Combined with 30 total frames, tracklets have a max of ~27 detected frames. t_min=10 means pairs must share at least 10 frames to even attempt scoring. This is feasible, but the w penalty still caps scores at 0.30.
  implication: The short-video regime fundamentally conflicts with the t_saturate=100 design assumption.

- timestamp: 2026-02-27T00:00:00Z
  checked: ReconstructionStage._run_with_tracklet_groups logic
  found: The stage filters to "detected" frame_status only. For each fish per frame, it needs min_cameras=3. With 4 test cameras and likely 1-2 camera groups (due to H1), most frames will lack 3 cameras even if a group exists.
  implication: Even if Association produces a few multi-camera groups, the camera coverage may still be insufficient.

- timestamp: 2026-02-27T00:00:00Z
  checked: MidlineStage._find_closest_detection_idx tolerance
  found: tolerance=5.0 pixels (hardcoded). This matches tracklet centroids (OC-SORT Kalman-filtered) to YOLO detection centroids. OC-SORT Kalman predictions can deviate from raw YOLO detections by more than 5px, especially after coasting frames or with fast-moving fish. If the centroid mismatch exceeds 5px, the detection is silently excluded from the midline stage.
  implication: Even for correctly associated fish, midlines may fail to be extracted if the tracking centroid doesn't closely match the detection centroid.

- timestamp: 2026-02-27T00:00:00Z
  checked: spline XYZ coordinates in outputs.h5 vs expected tank location
  found: Spline for fish_9 frame 2 is at approx (-1.07, 0.52, 1.55) world metres. The tank diameter is 2.0m, tank height is 1.0m. Spline XYZ values are in the right order of magnitude for an aquarium tank (within ~1-2m range).
  implication: The coordinate space is correct -- the offset is not due to a coordinate frame bug. The offset seen in the overlay is because the spline corresponds to the wrong fish (or an averaged mixture of fish rays from different cameras that were incorrectly associated or poorly triangulated).

- timestamp: 2026-02-27T00:00:00Z
  checked: Association scoring -- ghost penalty mechanism
  found: For each inlier frame, the scoring checks whether a ghost point (midpoint of the two rays) is visible in OTHER cameras. With only 4 cameras total, the "other cameras" set is small (2 cameras). If those 2 cameras show no detection near the ghost point (e.g., fish is schooled on one side, occluded in some cameras), the ghost ratio is 1.0 (fully penalized). This multiplied into the score: score = f * 0.0 * w = 0.0.
  implication: The ghost penalty is a double penalty: it penalizes true fish pairs where a supporting camera has no detection nearby (e.g., fish partially occluded in the 3rd camera). Combined with the t_saturate problem, most valid pairs get zeroed out.

## Root Cause Analysis

THREE compounding issues explain the observed behavior:

### ROOT CAUSE 1 (PRIMARY): t_saturate mismatch for short videos
File: `src/aquapose/core/association/scoring.py`, function `score_tracklet_pair()`
Formula: `w = min(t_shared, config.t_saturate) / config.t_saturate`
Config: `t_saturate=100`, but only 30 frames available.

Maximum possible score with 30 frames: w_max = 30/100 = 0.30.
Minimum threshold: score_min = 0.30.

For ANY score to pass, the pair must have PERFECT inlier fraction AND zero ghost penalty AND share exactly 30 frames. This is essentially impossible in practice. The result: nearly all cross-camera edges are filtered out by score_min, Leiden clustering produces all singletons, and ReconstructionStage drops them all (min_cameras=3 > 1).

The single fish produced (fish_9) survived by coincidence -- either one pair barely passed 0.30, or it was grouped by a path that bypasses the score gate (e.g., the refinement step uses a different criterion).

### ROOT CAUSE 2 (SECONDARY): MidlineStage centroid tolerance too tight
File: `src/aquapose/core/midline/stage.py`, function `_find_closest_detection_idx()`
Hardcoded `tolerance=5.0` pixels.

OC-SORT Kalman predictions drift from raw YOLO bboxes, especially during coasting. If drift > 5px, the detection is silently excluded from the midline group_det_index, so MidlineStage skips it. This means even a correctly-associated multi-camera fish may have fewer midlines extracted than cameras observed.

### ROOT CAUSE 3 (CONTRIBUTING): Ghost penalty too aggressive for 4-camera rig
File: `src/aquapose/core/association/scoring.py`, `score_tracklet_pair()`

With only 4 cameras, the "other cameras" set for ghost penalty is just 2. Fish not visible in those 2 cameras (partial occlusion, fish on other side of tank) trigger ghost_ratio=1.0, which multiplies the score to 0. This kills valid pairs.

### WHY THE SPLINE IS OFFSET
The spline offset is a CONSEQUENCE of root causes 1-3, not a separate bug:
- Association fails to correctly group tracklets, OR groups fish with wrong cameras
- Triangulation then uses midline points from cameras that don't correspond to the same physical fish
- The resulting 3D spline is an average of rays from different fish, producing a position that matches no specific fish in any view

## Resolution

root_cause: The t_saturate=100 parameter was designed for long production videos (hundreds of frames). The test run uses only 30 frames (stop_frame=30), making the overlap reliability weight (w = t_shared/t_saturate) cap out at 0.30 -- exactly at the score_min threshold. Combined with any imperfection in inlier fraction or ghost penalty, essentially all cross-camera edges score below threshold, producing only singleton TrackletGroups that fail the min_cameras=3 reconstruction requirement. The spline offset is a consequence of incorrect/empty association, not a triangulation bug.

fix: empty
verification: empty
files_changed: []
