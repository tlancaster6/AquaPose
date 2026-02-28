---
phase: 23-refractive-lookup-tables
plan: 02
subsystem: calibration
tags: [lut, refraction, inverse-lut, voxel-grid, numpy, torch, npz, cache, ghost-point, overlap-graph]

requires:
  - phase: 23-refractive-lookup-tables
    plan: 01
    provides: LutConfigLike Protocol, compute_lut_hash(), save/load infrastructure, ForwardLUT

provides:
  - InverseLUT class with voxel_centers, visibility_mask, projected_pixels, O(1) grid index lookup
  - generate_inverse_lut() building cylindrical voxel grid and projecting into all ring cameras
  - camera_overlap_graph() mapping camera pairs to shared-visible voxel counts
  - ghost_point_lookup() returning per-camera visibility and pixels for arbitrary 3D points
  - validate_inverse_lut() accuracy checker against RefractiveProjectionModel
  - save/load_inverse_lut() and save/load_inverse_luts() .npz serialization with hash-based caching
  - Coverage histogram and memory footprint printed during generation
  - 8 new unit tests covering all InverseLUT functionality

affects:
  - 25-association (consumes InverseLUT via camera_overlap_graph and ghost_point_lookup)
  - 26-refinement (uses ghost_point_lookup for fast reprojection during pose refinement)

tech-stack:
  added: []
  patterns:
    - "Cylindrical voxel grid with 1e-6*resolution epsilon on arange stop: avoids float32 overshoot while including exact boundary voxels"
    - "InverseLUT stores _grid_to_voxel_idx dict mapping (ix,iy,iz) integer coords to voxel array indices for O(1) ghost-point lookup"
    - "Scalar metadata (voxel_resolution, grid_bounds) serialized as float64 in .npz for exact roundtrip (float32 loses ~1e-7 relative precision)"
    - "Synthetic CalibrationData mocks in tests: _MockCameraData/_MockCalibrationData dataclasses + _look_at_rotation() helper for 3-camera ring rig"

key-files:
  created: []
  modified:
    - src/aquapose/calibration/luts.py
    - src/aquapose/calibration/__init__.py
    - tests/unit/calibration/test_luts.py

key-decisions:
  - "O(1) grid index via integer dict: store (ix,iy,iz)->voxel_idx mapping built at InverseLUT construction; ghost_point_lookup snaps point to nearest grid cell in O(1) without KD-tree"
  - "float64 for scalar metadata in .npz: voxel_resolution and grid_bounds saved as float64 to preserve exact Python float values; float32 roundtrip error would break equality assertions and cache invalidation"
  - "1e-6*resolution epsilon on arange stop: half-step tolerance (0.5*res) caused one extra z-step to appear beyond z_max due to float32 cumulative error; tiny epsilon includes exact boundary without overshoot"

patterns-established:
  - "Synthetic rig pattern: _MockCameraData + _MockCalibrationData + _look_at_rotation() builds minimal camera rigs for tests without loading real calibration files"
  - "_build_cylindrical_voxel_grid() returns (voxel_centers, grid_bounds) — separate from InverseLUT so the grid can be independently tested and reused"

requirements-completed:
  - LUT-02

duration: 10min
completed: 2026-02-27
---

# Phase 23 Plan 02: Inverse LUT System Summary

**InverseLUT class with cylindrical voxel grid, per-camera visibility/pixel projection, camera overlap graph, ghost-point lookup, and hash-based .npz caching**

## Performance

- **Duration:** ~10 min
- **Completed:** 2026-02-27
- **Tasks:** 2
- **Files modified:** 3 (2 modified, 1 test file expanded)

## Accomplishments
- InverseLUT discretizes the cylindrical tank volume into a voxel grid, projects all voxels into each ring camera via RefractiveProjectionModel.project(), and records visibility masks and pixel coordinates
- camera_overlap_graph() counts shared-visible voxels per camera pair for Phase 25 association graph construction
- ghost_point_lookup() returns which cameras can see any 3D point with O(1) nearest-voxel lookup via integer grid index
- Coverage histogram (1+, 2+, 3+... cameras) and memory footprint printed during generation
- Serialization to single inverse.npz with config_hash for automatic cache invalidation
- 15 total unit tests passing (7 forward LUT from 23-01 + 8 new inverse LUT tests)

## Task Commits

1. **Task 1: Implement InverseLUT with voxel grid, overlap graph, and ghost-point lookup** - `1079dca` (feat)
2. **Task 2: Write unit tests for InverseLUT** - `d37bf80` (test)

## Files Created/Modified
- `src/aquapose/calibration/luts.py` - InverseLUT dataclass, generate_inverse_lut(), camera_overlap_graph(), ghost_point_lookup(), validate_inverse_lut(), save/load_inverse_lut/s()
- `src/aquapose/calibration/__init__.py` - exports InverseLUT and all 8 new public functions
- `tests/unit/calibration/test_luts.py` - 8 new tests with synthetic 3-camera rig, _MockCalibrationData, _look_at_rotation()

## Decisions Made
- **O(1) grid index dict**: Plan suggested KD-tree or vectorized distance for ghost_point_lookup. Implemented integer rounding to grid coordinates + dict lookup instead — O(1) per point, no KD-tree memory overhead, works perfectly for the regular voxel grid.
- **float64 for scalar metadata**: voxel_resolution and grid_bounds serialized as float64, not float32. Float32 precision loss (~1e-7 relative) caused exact equality assertions and cache hash comparisons to fail on reload.
- **1e-6*resolution epsilon on arange stop**: The plan's original half-step tolerance (0.5*resolution) was causing one extra z-step beyond z_max due to float32 cumulative arithmetic error.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Voxel arange boundary overshoot with 0.5*resolution tolerance**
- **Found during:** Task 2 (test_cylindrical_voxel_grid_shape)
- **Issue:** `np.arange(z_min, z_max + voxel_resolution * 0.5, ...)` produced an extra voxel beyond z_max (e.g., z=0.35 when z_max=0.33) due to float32 cumulative arithmetic overshoot
- **Fix:** Changed tolerance from `voxel_resolution * 0.5` to `voxel_resolution * 1e-6` — tiny epsilon includes exact boundary without allowing an extra step
- **Files modified:** src/aquapose/calibration/luts.py
- **Verification:** test_cylindrical_voxel_grid_shape passes; all voxels within z_max + 1e-5
- **Committed in:** d37bf80 (Task 2 commit)

**2. [Rule 1 - Bug] float32 precision loss in scalar .npz serialization**
- **Found during:** Task 2 (test_inverse_lut_serialization_roundtrip)
- **Issue:** voxel_resolution=0.05 saved as float32 reloaded as 0.05000000074505806; grid_bounds showed similar float32 rounding errors; equality assertion failed
- **Fix:** Changed voxel_resolution and grid_bounds arrays to dtype=np.float64 in save_inverse_lut()
- **Files modified:** src/aquapose/calibration/luts.py
- **Verification:** test_inverse_lut_serialization_roundtrip passes with exact equality
- **Committed in:** d37bf80 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs)
**Impact on plan:** Both fixes necessary for correctness. No scope creep. Delivered spec exactly as designed.

## Issues Encountered
None beyond the auto-fixed deviations above.

## Next Phase Readiness
- Both Forward LUT (23-01) and Inverse LUT (23-02) are complete — Phase 23 is done
- Phase 25 (Association) can consume InverseLUT via camera_overlap_graph() and ghost_point_lookup() once Phase 24 (OC-SORT Tracking) is complete
- Phase 26 (Refinement) can use ghost_point_lookup() for fast reprojection without running the refractive model

---
*Phase: 23-refractive-lookup-tables*
*Completed: 2026-02-27*
