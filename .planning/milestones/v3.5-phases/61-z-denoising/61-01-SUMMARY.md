---
phase: 61-z-denoising
plan: 01
subsystem: reconstruction
tags: [svd, plane-fitting, z-denoising, hdf5, midline3d]

requires:
  - phase: 44
    provides: DLT triangulation backend, Midline3D type, HDF5 writer
provides:
  - IRLS-weighted SVD plane fit module (fit_plane_weighted, project_onto_plane)
  - Midline3D extended with plane_normal, plane_centroid, off_plane_residuals, is_degenerate_plane
  - PlaneProjectionConfig with enabled toggle in ReconstructionConfig
  - HDF5 writer/reader with plane metadata datasets
affects: [61-02, 63, 64]

tech-stack:
  added: []
  patterns: [weighted SVD plane fit for z-denoising, nested frozen dataclass config]

key-files:
  created:
    - src/aquapose/core/reconstruction/plane_fit.py
    - tests/unit/core/reconstruction/test_plane_fit.py
  modified:
    - src/aquapose/core/types/reconstruction.py
    - src/aquapose/core/reconstruction/backends/dlt.py
    - src/aquapose/engine/config.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/io/midline_writer.py
    - tests/unit/io/test_midline_writer.py
    - src/aquapose/core/reconstruction/__init__.py

key-decisions:
  - "Degenerate plane threshold set to 0.01 (s[1]/s[0] ratio)"
  - "Normal sign convention: positive z component for consistency"
  - "Off-plane residuals stored at body-point indices with NaN for non-triangulated points"

patterns-established:
  - "Nested frozen dataclass config: PlaneProjectionConfig within ReconstructionConfig, dict-to-dataclass conversion in load_config"

requirements-completed: [RECON-01, RECON-02, RECON-04, RECON-06]

duration: 12min
completed: 2026-03-05
---

# Plan 61-01: Component A Summary

**IRLS-weighted SVD plane fit projects triangulated body points onto best-fit bending plane before spline fitting, with full type/config/IO support**

## Performance

- **Duration:** 12 min
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Plane fit module with camera-count weighted SVD and degenerate detection
- DLT backend integration: plane projection runs between triangulation and spline fitting
- HDF5 writer stores plane_normal, plane_centroid, off_plane_residuals, is_degenerate_plane
- Backward-compatible reader for older HDF5 files without plane datasets
- Config toggle plane_projection.enabled (default True) threaded through pipeline

## Task Commits

1. **Task 1: Plane fit module, Midline3D extension, config** - `2ae8387` (feat)
2. **Task 2: DLT backend integration and HDF5 writer update** - `f3fa8ba` (feat)

## Decisions Made
- Degenerate plane threshold at s[1]/s[0] < 0.01 (empirically conservative)
- Normal sign convention: always positive z component to prevent inter-frame sign flips
- Off-plane residuals indexed by body point position (NaN for non-triangulated)

## Deviations from Plan
None - plan executed exactly as written

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plane metadata (normal, centroid) available in Midline3D for temporal smoothing (Plan 61-02)
- Config infrastructure ready for PlaneSmoothingConfig (separate from PlaneProjectionConfig)

---
*Phase: 61-z-denoising*
*Completed: 2026-03-05*
