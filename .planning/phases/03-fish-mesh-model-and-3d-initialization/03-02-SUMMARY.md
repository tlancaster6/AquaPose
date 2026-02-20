---
phase: 03-fish-mesh-model-and-3d-initialization
plan: 02
subsystem: initialization
tags: [pca, triangulation, keypoints, fish-state, cold-start, refractive]

# Dependency graph
requires:
  - phase: 01-calibration-and-refractive-geometry
    provides: "RefractiveProjectionModel.cast_ray and triangulate_rays for refractive triangulation"
  - phase: 03-fish-mesh-model-and-3d-initialization
    plan: 01
    provides: "FishState dataclass for the initialized pose output"
provides:
  - "extract_keypoints: PCA-based center + major-axis endpoint extraction from binary masks"
  - "extract_keypoints_batch: batch-first wrapper for keypoint extraction"
  - "triangulate_keypoint: multi-camera refractive triangulation of a single 2D keypoint"
  - "init_fish_state: FishState estimation from 3 triangulated 3D keypoints"
  - "init_fish_states_from_masks: full cold-start pipeline masks -> list[FishState]"
affects:
  - 04-differentiable-rendering
  - 05-pose-optimization

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PCA on mask pixel coordinates via np.linalg.eigh: eigenvector of largest eigenvalue = major axis"
    - "Canonical sign enforcement: negate axis if max projection < abs(min projection)"
    - "Refractive triangulation: cast_ray per camera -> stack origins/directions -> triangulate_rays"
    - "FishState angle extraction: psi=atan2(hy, hx), theta=asin(hz.clamp(-1,1)) from unit heading"
    - "Batch-first pipeline: init_fish_states_from_masks([n_cameras][n_fish], models) -> list[FishState]"

key-files:
  created:
    - src/aquapose/initialization/__init__.py
    - src/aquapose/initialization/keypoints.py
    - src/aquapose/initialization/triangulator.py
    - tests/unit/initialization/__init__.py
    - tests/unit/initialization/test_keypoints.py
    - tests/unit/initialization/test_triangulator.py

key-decisions:
  - "Canonical sign: endpoint_a always has max projection onto major axis — deterministic across calls"
  - "Minimum 3 cameras enforced at triangulate_keypoint level (not triangulate_rays which only needs 2)"
  - "kappa initialized to 0.0 (straight fish) — head/tail disambiguation deferred to Phase 4 optimizer"
  - "Synthetic test rig: cameras at Z=0 in world, water at Z=1.0, fish at Z~1.5 (not using aquacal dependency)"
  - "Camera geometry fix: t = (-cam_x, -cam_y, 0.0) with R=I places camera at (cam_x, cam_y, 0) in world"

# Metrics
duration: 9min
completed: 2026-02-20
---

# Phase 3 Plan 02: Cold-Start 3D Initialization Summary

**PCA-based keypoint extraction from binary masks + multi-camera refractive triangulation + FishState initialization — full cold-start pipeline with 29 unit tests passing, <1mm round-trip error, <5mm position error, <10 deg heading error**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-20T01:45:51Z
- **Completed:** 2026-02-20T01:55:43Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- `extract_keypoints`: PCA on foreground pixel coordinates, centroid + major-axis endpoints, canonical sign enforcement for determinism, handles bool masks, single-pixel degenerate case, raises on empty mask
- `extract_keypoints_batch`: batch-first list wrapper
- `triangulate_keypoint`: per-camera `cast_ray` → stack → `triangulate_rays` (refractive, not pinhole), enforces >=3 cameras
- `init_fish_state`: yaw/pitch from unit heading vector, s = endpoint distance, kappa = 0
- `init_fish_states_from_masks`: full pipeline [n_cameras][n_fish] → list[FishState], handles None masks (fish not visible in camera)
- 29 unit tests: 13 keypoint tests + 16 triangulation tests

## Task Commits

1. **Task 1: PCA keypoint extraction from binary masks** - `5122c9c` (feat)
2. **Task 2: Multi-camera refractive triangulation and FishState initialization** - `d356a69` (feat)

## Files Created/Modified

- `src/aquapose/initialization/__init__.py` - Public API: 5 exports with `__all__`
- `src/aquapose/initialization/keypoints.py` - `extract_keypoints`, `extract_keypoints_batch`
- `src/aquapose/initialization/triangulator.py` - `triangulate_keypoint`, `init_fish_state`, `init_fish_states_from_masks`
- `tests/unit/initialization/__init__.py` - Package init
- `tests/unit/initialization/test_keypoints.py` - 13 tests for PCA extraction
- `tests/unit/initialization/test_triangulator.py` - 16 tests for triangulation and FishState init

## Decisions Made

- **Canonical sign (endpoint_a = max projection)**: Ensures deterministic output across calls — same mask always produces same endpoint_a. Important for downstream use in `init_fish_states_from_masks` where all cameras must project the same semantic keypoint.
- **>=3 cameras enforced in triangulate_keypoint**: Plan specified >=3, even though `triangulate_rays` technically works with 2. Added as explicit guard at the initialization layer.
- **kappa=0 at initialization**: Fish assumed straight for cold-start; optimizer will recover curvature in Phase 4 via gradient descent.
- **Synthetic test rig without aquacal**: Built minimal 4-camera overhead rig inline in test file to avoid aquacal dependency in unit tests. Uses `t = (-cam_x, -cam_y, 0.0)` with `R=I`, cameras at Z=0, water at Z=1.0.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed synthetic test camera geometry (height_above_water)**
- **Found during:** Task 2 (first test run)
- **Issue:** Original `_make_overhead_camera` helper set `t[2] = -height_above_water = -1.0`, which placed the camera center at Z=1.0 — exactly at the water surface. `water_z - C[2] = 0` meant no ray-plane intersection distance, causing all projections to be invalid.
- **Fix:** Changed to `t = (-cam_x, -cam_y, 0.0)` placing cameras at Z=0 in world with water at Z=1.0.
- **Files modified:** tests/unit/initialization/test_triangulator.py
- **Verification:** All 16 triangulator tests pass; round-trip error ~0.01mm << 1mm limit.

---

**Total deviations:** 1 auto-fixed (bug in test geometry)
**Impact on plan:** Fix necessary for tests to work; no scope creep.

## Issues Encountered

None beyond the geometry bug (auto-fixed).

## Self-Check

Files verified to exist:
- [x] src/aquapose/initialization/__init__.py
- [x] src/aquapose/initialization/keypoints.py
- [x] src/aquapose/initialization/triangulator.py
- [x] tests/unit/initialization/test_keypoints.py
- [x] tests/unit/initialization/test_triangulator.py

Commits verified: 5122c9c, d356a69

## Self-Check: PASSED

## Next Phase Readiness

- `extract_keypoints` ready for any binary mask input (MOG2 output from Phase 2)
- `triangulate_keypoint` uses the same `cast_ray + triangulate_rays` from Phase 1 — refractive, not pinhole
- `init_fish_state` returns a `FishState` compatible with `build_fish_mesh` from Phase 3 Plan 01
- `init_fish_states_from_masks` is the cold-start entry point for the Phase 4 optimizer
- Phase 4 can call `init_fish_states_from_masks(masks_per_camera, models)` to get initial pose for gradient descent

---
*Phase: 03-fish-mesh-model-and-3d-initialization*
*Completed: 2026-02-20*
