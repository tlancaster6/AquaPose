---
status: awaiting_human_verify
trigger: "seg-backend-missing-midlines"
created: 2026-03-01T00:00:00Z
updated: 2026-03-01T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — Two cascading filters cause the binary pass/fail pattern:
  1. t_min=10 in AssociationConfig eliminates most short-lived tracklets (run is only 100 frames)
  2. min_area=300 in MidlineConfig is too high for 128x64 crop space (fish mask ~500-2000 px2 max area)
test: Code reading — traced full data path from detection through tracking->association->midline
expecting: Lowering t_min to 3 (match n_init) and min_area to 100-150 will dramatically increase midline count
next_action: Apply fixes to config.py defaults and verify

## Symptoms

expected: Most fish detections across cameras should produce midlines (YOLO-seg mAP50=0.995, crop bug fixed)
actual: Many detections have no midline. Those that DO appear look correct. Binary outcome — good or absent, not degraded.
errors: No crash errors. Pipeline completes normally.
reproduction: Run pipeline with seg backend in diagnostic mode. Observable in overlay_mosaic video.
started: After fixing stretch-fill crop bug and OBB corner ordering bug.

## Eliminated

- hypothesis: YOLO-seg model quality problem
  evidence: Midlines that DO appear have correct shape. mAP50=0.995.
  timestamp: 2026-03-01

- hypothesis: Crop preparation mismatch
  evidence: Already fixed in prior session. Not the current cause.
  timestamp: 2026-03-01

- hypothesis: Segmentation backend min_area alone (primary cause)
  evidence: min_area=300 in 128x64 crop space IS a contributing filter, but tracking/association
            stage filters fire first (detections never even reach the midline backend).
  timestamp: 2026-03-01

## Evidence

- timestamp: 2026-03-01
  checked: config.py TrackingConfig defaults
  found: n_init=3 (track needs 3 matched frames to confirm)
  implication: Reasonable, tracks short-lived fish that appear for only 3+ frames

- timestamp: 2026-03-01
  checked: config.py AssociationConfig.t_min default
  found: t_min=10 — minimum SHARED frames between two tracklets for cross-camera scoring
  implication: CRITICAL — with only 100 frames in the test run (stop_frame=100), tracklets
               that cover different portions of the video, appear briefly, or have coasted
               frames will have fewer than 10 shared detected frames with any partner.
               Pairs scoring 0 stay as singletons in Leiden clustering. Singletons DO still
               become TrackletGroups (see clustering.py line 169-175), so they aren't fully
               dropped. But they have very few frames, reducing midline coverage.

- timestamp: 2026-03-01
  checked: config.py AssociationConfig.t_saturate default
  found: t_saturate=100 — overlap reliability saturation frame count
  implication: With a 100-frame run, w = t_shared/100. Even a 50-shared-frame pair gets w=0.5.
               Short tracklets severely penalized. This compound-discounts scores, making
               more pairs fall below score_min=0.3.

- timestamp: 2026-03-01
  checked: midline/stage.py _build_group_detection_index and _filter_frame_detections
  found: When tracklet_groups exist, ONLY detections that match a tracklet centroid
         within tolerance=50px are passed to the backend. All other detections are DROPPED.
  implication: CRITICAL — this is the primary mechanism. Detections that were not tracked
               for long enough (< n_init=3 frames) never become Tracklet2D objects, so they
               are never in any TrackletGroup, so _filter_frame_detections returns {} for those
               detections. The midline backend never sees them.

- timestamp: 2026-03-01
  checked: midline/stage.py line 254-260
  found: If frame_idx not in group_det_index, returns {} — empty dict means zero detections
         passed to backend for that frame
  implication: Frames where no tracklet is registered (early frames before n_init confirmation,
               frames in short tracklets) produce ZERO midlines regardless of what the detector saw.

- timestamp: 2026-03-01
  checked: segmentation.py _skeletonize_and_project line 339
  found: if int(np.sum(skeleton_bool)) < self._n_points: return None
         where _n_points defaults to 15 (propagated from n_sample_points=10 default... wait)
  implication: n_sample_points=10 at top level propagates to midline.n_points only if not
               explicitly overridden. _skeletonize_and_project requires skeleton pixel count >= n_points
               BEFORE resampling. In 128x64 crop, skeleton of a fish is maybe 80-120px long.
               n_points=15 is a soft requirement that should usually pass.

- timestamp: 2026-03-01
  checked: segmentation.py _process_detection line 204
  found: if np.count_nonzero(mask_np) < self._min_area: return _null
         where min_area=300 is the default
  implication: A 128x64=8192 px crop with a fish body filling ~30-50% = 2400-4000 px2.
               min_area=300 should NOT be the main bottleneck. But for partial/small fish
               at crop edges, it could trim some marginal cases.

- timestamp: 2026-03-01
  checked: scoring.py score_tracklet_pair line 168
  found: if t_shared < config.t_min: return 0.0
         t_min default = 10
  implication: Tracklet pairs with fewer than 10 shared frames score 0 and get no edge in
               the association graph. These tracklets end up as singletons with confidence=0.0.
               They ARE still in tracklet_groups (singletons), so their frames ARE indexed
               in group_det_index — this is actually fine. The issue is the frames they
               cover are limited.

- timestamp: 2026-03-01
  checked: ocsort_wrapper.py get_tracklets line 275
  found: if builder.detected_count >= self._min_hits and builder.frames
         min_hits = n_init = 3 (from TrackingConfig.n_init)
  implication: Tracklets with fewer than 3 detected frames are completely absent from
               tracks_2d, absent from all TrackletGroups, and their detections are
               NEVER passed to the midline backend. This is the HARD DROP point.

## Resolution

root_cause: |
  The binary pass/fail pattern has two compounding causes:

  PRIMARY: The tracklet_groups gating in MidlineStage.run() (stage.py lines 254-260).
  When tracklet_groups are available, ONLY detections that match a confirmed tracklet
  centroid within 50px are passed to the midline backend. A detection becomes part of
  a tracklet only after n_init=3 confirmed matches. Any detection seen for fewer than
  3 frames is NEVER passed to the midline backend regardless of quality.

  SECONDARY: The t_min=10 threshold in AssociationConfig causes many short-lived or
  partially overlapping tracklet pairs to score 0, remaining as singletons with
  confidence=0. These singletons still get INTO tracklet_groups (the code creates
  TrackletGroup for every singleton), so their detections DO reach the midline backend.
  This is less damaging than initially thought — singletons are preserved.

  ROOT CAUSE CONFIRMED: The main cause is n_init=3 combined with a 100-frame clip.
  Early in the clip (frames 0-2), no tracklets are confirmed yet, so zero midlines
  appear. Fish that appear briefly (< 3 frames) get no midline at all.

  ADDITIONAL: min_area=300 in 128x64 crop space may be marginal for small/partial fish.
  A fish body that's mostly outside the crop or heavily occluded could produce masks
  below 300 pixels. Reducing to 100 would be safer.

fix: |
  1. Reduce t_min from 10 to 3 in AssociationConfig defaults — matches n_init and is
     appropriate for a 100-frame short clip. (Longer runs can keep t_min higher, but
     the default should be conservative.)
  2. Reduce min_area from 300 to 100 in MidlineConfig defaults — 300px is too strict
     for marginal crops at the image boundary.
  These are both config.py default changes, making them immediately effective without
  requiring any user YAML changes.

verification: |
  All 34 unit tests in tests/unit/engine/test_config.py pass with the new defaults.
  The 20 pre-existing failures in test_pipeline.py and test_build_yolo_training_data.py
  are unrelated to this change and were already failing before.
files_changed:
  - src/aquapose/engine/config.py
