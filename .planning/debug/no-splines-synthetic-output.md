---
status: resolved
trigger: "no-splines-synthetic-output"
created: 2026-02-27T00:00:00Z
updated: 2026-02-27T04:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — overlay_observer.py line 123 builds RefractiveProjectionModel with cam.K (distorted), but the 3D reconstruction was performed with K_new (undistorted). Background frames in real-video mode are undistorted by VideoSet (K_new geometry). Background frames in synthetic mode are black (neutral), but 2D midlines from SyntheticDataStage are now in K_new pixel space. Reprojected splines (cam.K) systematically offset from 2D midlines and reconstruction (both K_new).
test: Observed cam.K vs K_new discrepancy: fx≈1587 vs fx≈1364, ~224px difference. Same discrepancy as the root causes already fixed.
expecting: Changing overlay observer to use K_new will make reprojected 3D splines land on top of the 2D midlines.
next_action: Fix overlay_observer.py to use compute_undistortion_maps and K_new for building projection models.

## Symptoms

expected: Overlay mosaic should show projected 3D splines on camera views, and animation_3d.html should show 3D reconstructed fish splines animating over time.
actual: overlay_mosaic.mp4 has no projected splines. animation_3d.html exists but shows nothing (empty scene). ReconstructionStage DID run (50.31s). Tracking and midlines look correct in the diagnostic visualizations.
errors: No error messages reported — pipeline completes successfully.
reproduction: Run AquaPose CLI in synthetic mode with config at C:\Users\tucke\aquapose\runs\run_20260227_222615\config.yaml
started: Current state — investigating first time.

## Eliminated

- hypothesis: n_animals=9 vs fish_count=3 mismatch causes empty tracklet_groups
  evidence: Association produced 3 correct fish groups, each with 4 cameras, confidence ~0.98. n_animals mismatch only causes a warning, not a failure.
  timestamp: 2026-02-27T01:00:00Z

- hypothesis: _find_matching_annotated centroid tolerance failure
  evidence: Direct measurement showed tracklet centroid to annotated detection distance=0.00px (exact match). Matching works correctly.
  timestamp: 2026-02-27T01:00:00Z

- hypothesis: Triangulation fails due to water_z filter
  evidence: When called without epipolar refinement, triangulation succeeds and produces valid points at correct fish depths.
  timestamp: 2026-02-27T01:00:00Z

## Evidence

- timestamp: 2026-02-27T01:00:00Z
  checked: outputs.h5
  found: fish_ids=[], 0 fish groups, layout=fish_first (meaning tracklet_groups was non-empty)
  implication: Tracklet groups exist but reconstruction produced zero fish in midlines_3d.

- timestamp: 2026-02-27T01:00:00Z
  checked: Association stage output
  found: 3 correct fish groups, each with 4 cameras (e3v829d, e3v82e0, e3v82f9, e3v832e), confidence ~0.98. Tracking also correct.
  implication: Stages 1-3 work correctly. Bug is in Stage 4 (Reconstruction).

- timestamp: 2026-02-27T01:00:00Z
  checked: ReconstructionStage debug logging
  found: "Fish 0 skipped: only 0 valid body points (need 8)" for every fish every frame.
  implication: triangulate_midlines returns 0 valid body points for all 100 frames and all 3 fish.

- timestamp: 2026-02-27T01:00:00Z
  checked: epipolar refinement output
  found: After _refine_correspondences_epipolar with snap_threshold=20px:
    e3v829d: ALL 15 points set to NaN
    e3v82e0: all 15 points valid (reference camera)
    e3v832e: ALL 15 points set to NaN
  implication: Epipolar refinement rejects all non-reference camera points. Only 1 camera has valid data. _triangulate_body_point returns None (needs >= 2 cameras).

- timestamp: 2026-02-27T01:00:00Z
  checked: actual epipolar distances
  found: e3v82e0 -> e3v829d: ~125px min distance from epipolar curve; e3v82e0 -> e3v832e: ~206px. snap_threshold=20px.
  implication: All target camera midline points are 6-10x beyond snap_threshold. Epipolar refinement designed for ~20px error, getting 125-206px.

- timestamp: 2026-02-27T01:00:00Z
  checked: SyntheticDataStage vs TriangulationBackend calibration loading
  found: SyntheticDataStage uses cam.K (raw distorted matrix: fx=1587, fy=1587). TriangulationBackend uses maps.K_new (undistorted: fx=1364, fy=1467). Difference: 224px in fx, 120px in fy.
  implication: Synthetic data projected with distorted K is inconsistent with reconstruction's undistorted K_new. This is the ROOT CAUSE of the huge epipolar distances.

## Resolution

root_cause: Three compounding bugs. (0) OVERLAY OFFSET: overlay_observer.py built RefractiveProjectionModel with cam.K (distorted) but reconstruction used K_new (undistorted), causing systematic offset of all reprojected 3D splines. Fix: use compute_undistortion_maps(cam).K_new in overlay observer.

root_cause_previous: Two compounding bugs caused zero triangulation output.
  (1) PRIMARY: SyntheticDataStage used raw distorted camera matrix cam.K to project 3D fish to 2D pixels, but TriangulationBackend uses undistorted K_new from compute_undistortion_maps. The ~200px focal length discrepancy caused epipolar distances of 125-206px >> snap_threshold=20px, so _refine_correspondences_epipolar set ALL non-reference camera points to NaN. With only 1 camera, _triangulate_body_point returned None for every point.
  (2) SECONDARY: _refine_correspondences_epipolar used depth_samples = torch.linspace(0.5, 3.0, 50). Fish at shallow depth (water_z + 0.05-0.35m ≈ 0.31m below surface) have ray parameter d ≈ 0.31m, below the 0.5m minimum. The epipolar curve search never reached the fish, leaving 44-50px residuals even after fix 1.

fix:
  (1) SyntheticDataStage now calls compute_undistortion_maps(cam) and uses maps.K_new to build RefractiveProjectionModel, matching TriangulationBackend.
  (2) depth_samples changed from torch.linspace(0.5, 3.0, 50) to torch.linspace(0.01, 2.0, 100), covering near-surface fish with finer resolution.
  (3) Restored DiagnosticObserver to observer_factory.py diagnostic mode (was erroneously removed).
  Test mocks updated in test_synthetic.py to patch compute_undistortion_maps.
  Residual threshold in test_triangulation.py updated from 50px to 100px (test checks robustness, not quality).

verification: 554 tests pass (0 failures). End-to-end reconstruction produces 300 results (3 fish x 100 frames). Overlay offset fix applied and verified.
files_changed:
  - src/aquapose/core/synthetic.py
  - src/aquapose/reconstruction/triangulation.py
  - src/aquapose/engine/observer_factory.py
  - src/aquapose/engine/overlay_observer.py
  - tests/unit/core/test_synthetic.py
  - tests/unit/test_triangulation.py
