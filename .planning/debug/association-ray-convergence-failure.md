---
status: awaiting_human_verify
trigger: "association-ray-convergence-failure — ~86.5% singleton rate, ray-ray distances 35-275px even for winning combos, 9 fish 8 useful cameras"
created: 2026-03-04T00:00:00Z
updated: 2026-03-04T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — ForwardLUTs are built with raw distorted K but centroid pixels come from undistorted frames (K_new space). Mismatch causes 0.736-2.252 degree ray errors, yielding 1.7-5.3 cm ray-ray distances for the SAME fish in two cameras, against a 3 cm threshold.
test: Confirmed by comparing cached LUT directions vs raw K LUT (0.000 diff) vs K_new LUT (0.736-2.252 deg diff). Simulated ray-ray distances with wrong K reach 4.67-5.26 cm, exceeding the 3 cm threshold.
expecting: Fix requires passing undistortion_maps to generate_forward_luts in AssociationStage and invalidating the cached LUTs.
next_action: Apply fix in AssociationStage.run() to pass undistortion_maps when generating/saving forward LUTs. Also fix generate_inverse_lut call for consistency. Delete cached LUTs to force regeneration.

## Symptoms

expected: Most fish should be seen by multiple cameras and associated across views. With 8 useful cameras and 9 fish, most fish should have 2-5 camera views.
actual: ~86.5% singleton rate. Only ~7/9 fish form multi-camera groups, and even those have only 1.6 cameras on average. Many real fish tracked stably in single cameras can't find cross-view partners.
errors: No errors — the scoring just returns low/zero scores for pairs that should match. Association tuning sweep shows centroid reprojection errors of 35-275px even for "winning" combos, suggesting the 3D geometry is off.
reproduction: Run association on existing tracking data. Run directory: ~/aquapose/projects/YH/runs/run_20260303_204845/
started: Has been like this since association was implemented. Previous debug session concluded thresholds were tuned but fundamental problem persists.

## Prior Context

Previous session (association-singletons.md, resolved):
- 8/13 cameras actually see fish
- ~18 singleton groups per frame — mix of false-positives and real stable single-camera fish
- Singletons are NOT short fragments — they persist for many frames
- Widening thresholds makes things WORSE (over-merging)
- Ranked hypotheses: (1) coordinate space mismatch, (2) camera adjacency filtering, (3) t_min filtering, (4) early_k termination

## Eliminated

- hypothesis: H2 camera adjacency filtering, H3 t_min filtering, H4 early_k termination
  evidence: H1 confirmed immediately with direct code + data analysis. These hypotheses were not investigated but are irrelevant given H1 magnitude.
  timestamp: 2026-03-04

## Evidence

- timestamp: 2026-03-04
  checked: AssociationStage.run() in stage.py
  found: Calls generate_forward_luts(calibration, self._config.lut) WITHOUT passing undistortion_maps. The undistortion_maps parameter defaults to None, so raw cam.K is used for the LUT instead of K_new.
  implication: LUTs are indexed by undistorted pixel coordinates but built assuming the distorted coordinate space.

- timestamp: 2026-03-04
  checked: VideoFrameSource.__iter__ in frame_source.py
  found: Applies undistort_image() to every frame before yielding it. Detection and tracking operate on undistorted frames, producing centroids in K_new pixel space.
  implication: All centroid pixels fed to the association stage are in undistorted (K_new) coordinate space.

- timestamp: 2026-03-04
  checked: Raw K vs K_new for 4 ring cameras
  found: Focal lengths differ by 220-227 px (about 14%). e.g., cam e3v829d: raw fx=1587.42, K_new fx=1363.67, diff=223.7 px. Consistent across all cameras.
  implication: The LUT maps pixels using the wrong focal length, systematically misdirecting all rays.

- timestamp: 2026-03-04
  checked: Ray angular error at representative pixel coordinates
  found: When using raw K vs K_new, angular errors of 0.736 deg (center) to 2.252 deg (off-center). At 0.5m tank depth, this causes 0.6-2.0 cm lateral displacement per ray.
  implication: Each ray is wrong in isolation. The cumulative ray-ray distance error for a pair is the combination of both camera errors.

- timestamp: 2026-03-04
  checked: Simulated ray-ray distances for same fish in two cameras
  found: With wrong K: avg 1.7-2.6 cm, max 2.56-5.26 cm depending on camera pair and depth. With correct K_new: essentially 0.000 cm. The 3 cm threshold fails many cases.
  implication: At off-center positions, ray-ray distance for the SAME fish exceeds the threshold, giving zero score and causing singletons.

- timestamp: 2026-03-04
  checked: Cached LUT vs raw K LUT vs K_new LUT directions
  found: Cached LUT direction diff from raw K LUT = 0.000000 deg (exact match). Cached LUT vs K_new LUT = 0.736-2.252 deg (significant mismatch). Confirmed cached LUTs were built with raw K.
  implication: The cached LUTs at ~/aquapose/projects/YH/geometry/luts/ are incorrect and must be regenerated.

## Resolution

root_cause: AssociationStage.run() calls generate_forward_luts() and generate_inverse_lut() without passing undistortion_maps. This causes LUTs to be built with the raw distorted camera intrinsic matrix K instead of the post-undistortion K_new. Because VideoFrameSource always delivers undistorted frames (K_new pixel space), tracklet centroids live in a different coordinate system than the LUT expects. The resulting ray-ray distances for the SAME fish in two cameras are 1.7-5.3 cm, often exceeding the 3 cm threshold, causing zero or near-zero affinity scores and 86.5% singleton rate.
fix: Pass undistortion_maps to generate_forward_luts and generate_inverse_lut in AssociationStage.run(). Delete the bad cached LUTs so they are regenerated correctly.
verification: 780 existing tests still pass. 5 pre-existing failures in test_stage_association.py (already failing before fix). Fix is a 15-line change adding compute_undistortion_maps and passing undistortion_maps to both generate_forward_luts and generate_inverse_lut. Bad cached LUTs deleted from ~/aquapose/projects/YH/geometry/luts/. Awaiting human end-to-end verification via a re-run of the pipeline.
files_changed:
  - src/aquapose/core/association/stage.py
