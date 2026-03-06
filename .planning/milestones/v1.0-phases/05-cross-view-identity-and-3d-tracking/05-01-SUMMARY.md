---
phase: 05-cross-view-identity-and-3d-tracking
plan: 01
subsystem: tracking
tags: [ransac, cross-view-association, centroid-clustering, triangulation]
dependency_graph:
  requires:
    - aquapose.calibration.projection (RefractiveProjectionModel, triangulate_rays)
    - aquapose.segmentation.detector (Detection)
  provides:
    - aquapose.tracking.ransac_centroid_cluster
    - aquapose.tracking.AssociationResult
    - aquapose.tracking.FrameAssociations
  affects:
    - Phase 05 Plans 02-03 (tracker, HDF5 serialization)
tech_stack:
  added: []
  patterns:
    - RANSAC with prior-guided seeding for cross-view identity
    - torch.no_grad() context for all inference projection/ray calls
    - Greedy assignment after RANSAC to prevent double-assignment
    - Ray-depth heuristic (default_tank_depth=0.5m) for single-view fallback centroids
key_files:
  created:
    - src/aquapose/tracking/__init__.py
    - src/aquapose/tracking/associate.py
    - tests/unit/tracking/__init__.py
    - tests/unit/tracking/test_associate.py
  modified: []
decisions:
  - "assigned_mask dict tracks which detections have been tentatively claimed — prevents double-assignment across both prior-guided and RANSAC passes"
  - "rays_per_camera uses cam_rays local var (not tuple unpacking) to satisfy basedpyright narrowing for Optional subscript"
  - "Single-view detections use 0.5m default tank depth along refracted ray as centroid heuristic — not None — so FrameAssociations are always fully populated"
metrics:
  duration: "5 minutes"
  completed: "2026-02-21"
  tasks_completed: 2
  files_created: 4
  files_modified: 0
---

# Phase 05 Plan 01: RANSAC Centroid Ray Clustering Summary

RANSAC centroid ray clustering for cross-view fish identity association with prior-guided seeding, greedy assignment, and single-view fallback.

## What Was Built

### `src/aquapose/tracking/associate.py`

Core algorithm: given per-camera detections in a single frame, determine which detections across cameras correspond to the same physical fish.

**Data structures:**
- `AssociationResult`: per-fish dataclass holding fish_id, centroid_3d (shape (3,)), reprojection_residual, camera_detections dict, n_cameras, confidence, is_low_confidence
- `FrameAssociations`: per-frame container with associations list, frame_index, unassigned list

**Algorithm (`ransac_centroid_cluster`):**
1. Cast refractive rays from each detection's mask centroid using `model.cast_ray()`
2. Prior-guided pass: project each seed point into cameras, find nearest unassigned detection within reprojection_threshold, accept if >=min_cameras inliers
3. Random RANSAC (n_iter=200 default): sample 2 cameras, 1 detection each, call `triangulate_rays()`, score consensus by reprojecting candidate into all cameras
4. Greedy assignment: accept candidates sorted by inlier count, each detection assigned to at most one fish
5. Low-confidence fallback: remaining single-view detections placed at default tank depth (0.5m) along refracted ray

**Helper:** `_compute_mask_centroid(mask)` — returns (u, v) foreground center-of-mass using `np.where(mask > 0)`, not bbox center.

### `tests/unit/tracking/test_associate.py`

20 tests across 6 test classes, all using synthetic camera rigs (no GPU, no real data):

| Class | What it tests |
|-------|---------------|
| `TestTwoFishThreeCameras` | Basic 2-fish, 3-camera clustering: associations found, >=2 cameras each, 3D centroids within 0.05m XY, residuals below threshold |
| `TestPriorGuidedSeeding` | Seeds enable n_iter=20 and n_iter=0 to find all fish |
| `TestSingleViewDetectionFlagged` | is_low_confidence=True, n_cameras=1, confidence matches detection |
| `TestNoDoubleAssignment` | No (camera_id, det_idx) pair appears in two associations |
| `TestMaskCentroidNotBboxCenter` | Off-center mask centroid != bbox center; empty mask raises |
| `TestEmptyInput` | Empty dict, all-empty lists, frame_index preserved |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] basedpyright Optional subscript error on `rays_per_camera.get()`**
- **Found during:** Task 1 typecheck
- **Issue:** `origins, dirs = rays_per_camera.get(cam_id, (None, None))` returned `tuple[Tensor, Tensor] | tuple[None, None]`; basedpyright couldn't narrow `dirs` via `origins is not None`
- **Fix:** Changed to `cam_rays = rays_per_camera.get(cam_id)` with `cam_rays[0]` / `cam_rays[1]` access
- **Files modified:** `src/aquapose/tracking/associate.py`
- **Commit:** d546b71 (reformatted by pre-commit hook, same commit)

**2. [Rule 1 - Lint] Ruff B905 and RUF059 in test file**
- **Found during:** Task 2 pre-commit hook
- **Issue:** `zip()` without `strict=` and unused `v` variable in `_v` pattern
- **Fix:** Added `strict=True` to zip; renamed `v` to `_v`
- **Files modified:** `tests/unit/tracking/test_associate.py`
- **Commit:** 04bc500

## Verification Results

```
hatch run check           → ruff: All checks passed; basedpyright: 0 errors in tracking/
hatch run test tests/unit/tracking/   → 20/20 passed
python -c "from aquapose.tracking import ransac_centroid_cluster"   → OK
```

Pre-existing basedpyright errors in `detector.py` (4 errors) are out of scope.

## Self-Check: PASSED

Files exist:
- FOUND: src/aquapose/tracking/__init__.py
- FOUND: src/aquapose/tracking/associate.py
- FOUND: tests/unit/tracking/__init__.py
- FOUND: tests/unit/tracking/test_associate.py

Commits exist:
- FOUND: d546b71 (feat(05-01): implement RANSAC centroid ray clustering)
- FOUND: 04bc500 (test(05-01): add unit tests for RANSAC centroid clustering)
