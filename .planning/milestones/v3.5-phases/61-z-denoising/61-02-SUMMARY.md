---
phase: 61-z-denoising
plan: 02
subsystem: reconstruction
tags: [temporal-smoothing, gaussian-filter, rodrigues-rotation, z-denoising, cli, eval-metrics]

requires:
  - phase: 61
    plan: 01
    provides: Midline3D with plane metadata, PlaneProjectionConfig, HDF5 plane datasets
provides:
  - Temporal smoothing module (smooth_plane_normals, rotate_control_points_to_plane)
  - PlaneSmoothingConfig with enabled and sigma_frames toggles
  - smooth-planes CLI command for post-processing HDF5 files
  - ZDenoisingMetrics dataclass and compute_z_denoising_metrics function
affects: [63, 64]

tech-stack:
  added: []
  patterns: [Gaussian temporal smoothing with sign consistency, Rodrigues rotation formula]

key-files:
  created:
    - src/aquapose/core/reconstruction/temporal_smoothing.py
    - tests/unit/core/reconstruction/test_temporal_smoothing.py
  modified:
    - src/aquapose/core/reconstruction/__init__.py
    - src/aquapose/engine/config.py
    - src/aquapose/cli.py
    - src/aquapose/evaluation/stages/reconstruction.py
    - src/aquapose/evaluation/stages/__init__.py
    - tests/unit/evaluation/test_stage_reconstruction.py

key-decisions:
  - "Temporal smoothing is post-processing (CLI command), not inline in reconstruction pipeline"
  - "Segment boundaries detected by frame index gaps > 1"
  - "Degenerate frames interpolated via linear interp before Gaussian filtering"
  - "SNR defined as std(mean_z_profile) / std(frame-to-frame z diffs)"

patterns-established:
  - "Post-processing CLI commands that read/write HDF5 in-place with --dry-run option"

requirements-completed: [RECON-03, RECON-05, RECON-06]

duration: 15min
completed: 2026-03-05
---

# Plan 61-02: Component B Summary

**Temporal Gaussian smoothing of plane normals per-fish with Rodrigues control point rotation, smooth-planes CLI command, and z-denoising eval metrics**

## Performance

- **Duration:** 15 min
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Temporal smoothing module with sign consistency, segment detection, degenerate interpolation
- Rodrigues' rotation formula for control point reorientation between plane normals
- PlaneSmoothingConfig (enabled=True, sigma_frames=3) in engine config
- smooth-planes CLI command reads HDF5, smooths normals, rotates control points, writes in-place
- ZDenoisingMetrics: median z-range, z-profile RMS, per-fish SNR, residual delta
- compute_z_denoising_metrics evaluates spline z-coordinates across frames
- All new symbols exported from package __init__.py files

## Task Commits

1. **Task 1: Temporal smoothing module and config** - `32a41fd` (feat)
2. **Task 2: CLI command and eval metrics** - `e656961` (feat)

## Decisions Made
- Component B is post-processing (not inline in pipeline) -- separate CLI step
- Segment boundaries at frame gaps > 1 (not fish_id changes, since caller groups by fish)
- Degenerate frames interpolated linearly before Gaussian filter, not excluded
- z-range converted from metres to cm in metrics output

## Deviations from Plan
- CLI smooth-planes command was committed together with Phase 62 calibrate-keypoints changes (cde55f0) due to shared cli.py file; eval metrics committed separately (e656961)

## Issues Encountered
- Pre-existing test failures in test_generate_luts_cli.py and test_training_cli.py unrelated to Phase 61

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Z-denoising metrics available for evaluation harness integration
- Smoothed plane normals stored in HDF5 for downstream analysis

---
*Phase: 61-z-denoising*
*Completed: 2026-03-05*
