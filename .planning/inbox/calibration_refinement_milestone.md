# Milestone Seed: Calibration Refinement via Fish Observations

## Motivation

The initial camera calibration (checkerboard-based, via AquaCal) may have residual extrinsic errors or drift over time. Fish detections from the AquaPose pipeline provide "free" multi-view correspondences that can drive bundle-adjustment-style refinement of extrinsic parameters, improving reconstruction accuracy without additional calibration hardware sessions.

## Core Idea

After a pipeline run, extract triangulated 3D fish positions and their per-camera 2D observations, then feed them into AquaCal's `refine_calibration()` API as `PointCorrespondence` objects. The refined calibration replaces the original, and LUTs are regenerated.

## AquaCal API

AquaCal (upcoming PyPI release) provides `refine_calibration()` in `aquacal.calibration.point_refinement`:

- **Input:** `CalibrationResult` + list of `PointCorrespondence(point_3d, observations: dict[cam_name, pixel], weight)`
- **Refines:** Extrinsics (R, t), water_z. Optionally intrinsics (fx, fy, cx, cy, bounded ±10%).
- **Robust losses:** Huber, Cauchy, Soft-L1 for outlier tolerance.
- **Holdout validation:** Automatic train/test split with accept/reject recommendation and per-camera drift reporting.
- **Minimum data:** ≥10 correspondences required.
- **Returns:** `RefinementResult` with refined calibration, validation report, and accepted flag.

## Observation Source: Pose Keypoints

The primary observation source is **pose model keypoints** from the reconstruction stage. Each keypoint corresponds to a consistent anatomical landmark on the fish (nose, tail, body points 1-5), giving view-independent 2D observations that map to the same physical 3D point across all cameras.

**Why not bbox/mask centroids?** The bbox centroid is a 2D artifact that corresponds to different 3D positions depending on viewing angle. A fish viewed head-on vs. broadside has its bbox center at very different physical locations. This view-dependent bias would introduce systematic noise into the refinement.

**Available data per correspondence:**
- **3D point:** Triangulated body point from reconstruction (7 per fish per frame)
- **2D observations:** Pose model keypoint detections in each camera (calibration-independent — produced by the neural network)
- **Filtering:** Use interior body points (indices 2-4) which have the best multi-view visibility and lowest pose model error. Head/tail keypoints are noisier and can be excluded or down-weighted.
- **Volume:** ~3 fish × 3-5 usable keypoints × thousands of frames × 6+ cameras = massive correspondence set. Will need subsampling.

**Stretch alternative:** Consensus centroids from association (simpler to extract, available without reconstruction, but subject to the view-dependent bias described above).

## Proposed Workflow

1. Run normal AquaPose pipeline → produces association + reconstruction outputs
2. `aquapose calibration refine` CLI command:
   a. Load existing calibration and pipeline outputs
   b. Extract PointCorrespondence objects from pose keypoints (triangulated 3D body points + per-camera 2D keypoint detections)
   c. Call `refine_calibration()` with configurable options (loss function, intrinsics toggle, etc.)
   d. Display validation report (holdout error, per-camera drift)
   e. Save refined calibration (new file, preserving original)
3. Regenerate LUTs for the refined calibration (existing `aquapose prep generate-luts` command, or automatic)
4. Optionally re-run pipeline with refined calibration to verify improvement

## Key Design Decisions

- **Standalone CLI command**, not an integrated pipeline stage. Avoids feedback loops and makes validation easier.
- **Pose keypoints as primary observation source.** Interior body points (indices 2-4) preferred for geometric stability. Centroids as fallback only.
- **Extrinsics-only by default.** Intrinsics refinement available as an opt-in flag.
- **Non-destructive.** Save refined calibration alongside original (e.g., `calibration_refined.json`), never overwrite.
- **Validation-gated.** Show the AquaCal validation report; let the user decide whether to adopt the refined calibration.

## Stretch Goals

- Consensus centroid correspondences as simpler alternative/complement
- Include head/tail keypoints with lower weights
- Iterative refinement (refine → re-run → refine again)
- Automatic quality gating (reject refinement if holdout error worsens)
- Before/after reprojection error comparison visualization

## Prerequisite Changes (Done)

`Midline3D` now stores raw triangulation data needed for correspondence extraction:

- `triangulated_points: np.ndarray | None` — shape (n_body_points, 3), raw triangulated 3D positions before spline fitting. NaN for failed points.
- `per_point_inlier_cameras: list[list[str]] | None` — which cameras were inliers for each body point after outlier rejection.

These are populated by the DLT backend from data it already computed but previously discarded. The corresponding 2D keypoint observations are available via `annotated_detections[frame][camera].midline.points[body_idx]` in the diagnostic cache.

Together, these provide everything needed to build `PointCorrespondence` objects: a 3D point, the set of cameras that observed it, and (via the diagnostic cache) the exact 2D pixel locations in each camera.

## Dependencies

- AquaCal with `point_refinement` module (upcoming PyPI release)
- Diagnostic pipeline run with the updated `Midline3D` fields populated

## Open Questions

- How many frames/correspondences are needed for reliable refinement? AquaCal minimum is 10, but practical minimum for a 12-camera rig is likely higher. Will need subsampling strategy for very long videos (thousands of frames × multiple fish × multiple keypoints).
- Should we filter correspondences by keypoint confidence, reprojection residual, or number of contributing cameras before feeding to AquaCal?
- Which keypoint indices are most reliable? Need to verify that interior body points (2-4) are consistently the best across the dataset.
