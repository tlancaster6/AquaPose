# Phase 61: Z-Denoising - Verification

**Verified:** 2026-03-05

## Goal
Clean z-noise from 3D reconstructions via per-frame plane projection (Component A) and temporal smoothing of plane orientation (Component B).

## Verification Checklist

### Component A: Plane Projection
- [x] `fit_plane_weighted` computes IRLS-weighted SVD plane fit with camera-count weights
- [x] `project_onto_plane` projects points onto best-fit plane, preserving signed residuals
- [x] Degenerate plane detection via singular value ratio threshold (0.01)
- [x] Midline3D extended with plane_normal, plane_centroid, off_plane_residuals, is_degenerate_plane
- [x] PlaneProjectionConfig with enabled toggle (default True)
- [x] DLT backend integrates plane fit between triangulation and spline fitting
- [x] HDF5 writer stores all 4 new plane metadata datasets
- [x] Backward-compatible reader for older HDF5 files

### Component B: Temporal Smoothing
- [x] `smooth_plane_normals` with Gaussian filter, sign consistency, segment detection
- [x] `rotate_control_points_to_plane` with Rodrigues' rotation formula
- [x] Degenerate frames interpolated through (not excluded)
- [x] PlaneSmoothingConfig with enabled and sigma_frames toggles
- [x] `smooth-planes` CLI command with --input, --sigma-frames, --dry-run options
- [x] Writes smoothed_plane_normal dataset and overwrites control_points in-place

### Evaluation Metrics
- [x] ZDenoisingMetrics dataclass with median_z_range_cm, mean_z_profile_rms_cm, per_fish_snr
- [x] compute_z_denoising_metrics evaluates spline z-coordinates across frames
- [x] Integrated into ReconstructionMetrics.to_dict()
- [x] Exported from evaluation stages __init__.py

### Testing
- [x] Unit tests for plane fit (5 test cases)
- [x] Unit tests for temporal smoothing (7 test cases)
- [x] Unit tests for control point rotation (4 test cases)
- [x] Unit tests for HDF5 plane metadata round-trip
- [x] Unit tests for z-denoising eval metrics (5 test cases)
- [x] All Phase 61 tests pass; no new failures introduced

## Result: PASS

---
*Phase: 61-z-denoising*
*Verified: 2026-03-05*
