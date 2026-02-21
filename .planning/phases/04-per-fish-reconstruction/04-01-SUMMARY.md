---
phase: 04-per-fish-reconstruction
plan: 01
subsystem: optimization
tags: [pytorch3d, differentiable-rendering, refractive-projection, silhouette, soft-iou, autograd]

# Dependency graph
requires:
  - phase: 01-calibration-and-refractive-geometry
    provides: "RefractiveProjectionModel.project() — differentiable Snell's law projection"
  - phase: 03-fish-mesh-model-and-3d-initialization
    provides: "build_fish_mesh(list[FishState]) -> Meshes, FishState dataclass"
provides:
  - "RefractiveCamera: PyTorch3D-compatible camera wrapping RefractiveProjectionModel.project() into NDC space"
  - "RefractiveSilhouetteRenderer: vertex pre-projection renderer (world -> NDC -> FoVOrthographicCameras -> alpha map)"
  - "soft_iou_loss: differentiable 1-IoU with optional (y1,x1,y2,x2) crop region restriction"
  - "compute_angular_diversity_weights: per-camera weights from view direction min-angle separation"
  - "multi_objective_loss: weighted IoU + gravity prior + morphological constraints + temporal hook"
affects:
  - 04-02-optimizer

# Tech tracking
tech-stack:
  added:
    - "torch 2.9.1+cu128 (downgraded from 2.10+cu130 to match pytorch3d ABI)"
    - "torchvision 0.24.1+cu128 (matching torchvision for torch 2.9.1)"
    - "pytorch3d.renderer: MeshRasterizer, SoftSilhouetteShader, FoVOrthographicCameras, BlendParams, RasterizationSettings"
  patterns:
    - "Vertex pre-projection: world -> NDC via RefractiveCamera, then FoVOrthographicCameras identity render (no further transform)"
    - "Image size derived from K matrix: H=round(2*cy), W=round(2*cx) when not explicitly provided"
    - "Soft IoU formula: (pred*target).sum() / (pred+target-pred*target).sum() with crop region slicing"
    - "Angular diversity: weight_i = (min_angle_i / max_j(min_angle_j)) ** temperature, numpy-computed once"
    - "Gravity proxy: state.theta^2 — pitch deviation as roll substitute (no explicit roll in FishState)"

key-files:
  created:
    - src/aquapose/optimization/renderer.py
    - src/aquapose/optimization/loss.py
    - tests/unit/optimization/__init__.py
    - tests/unit/optimization/test_renderer.py
    - tests/unit/optimization/test_loss.py
  modified:
    - src/aquapose/optimization/__init__.py

key-decisions:
  - "Vertex pre-projection approach chosen over custom PyTorch3D camera class: MeshRasterizer.transform() calls get_world_to_view_transform() and get_ndc_camera_transform() which our custom class didn't implement; pre-projecting to NDC then using FoVOrthographicCameras as identity is cleaner and fully differentiable"
  - "Torch downgraded from 2.10+cu130 to 2.9.1+cu128: pytorch3d-0.7.9+pt2.9.1cu128 wheel links against cudart64_128.dll absent in CUDA 13.0; torchvision downgraded to 0.24.1+cu128 to match"
  - "Camera image size derived from K: H=round(2*cy), W=round(2*cx) — approximation assuming centered principal point; caller can override via camera_image_sizes parameter"
  - "Higher temperature = more spread in angular diversity weights: weight = (min_angle/max_min_angle)^T, small base values drop faster with higher exponent (inverse of initial docstring intuition)"
  - "Gravity prior uses theta^2 (pitch proxy): FishState has no explicit roll angle; adding phi would require FishState extension; pitch proxy accepted per plan recommendation"

patterns-established:
  - "Pattern: RefractiveCamera.project_to_ndc(world_pts) converts world -> pixel -> NDC; Z passthrough for depth sorting"
  - "Pattern: multi_objective_loss returns dict with total/iou/gravity/morph/temporal for per-term logging"
  - "Pattern: temporal_state=None -> temporal_loss=0 hook for Phase 5 activation"

requirements-completed: [RECON-01, RECON-02]

# Metrics
duration: 17min
completed: 2026-02-21
---

# Phase 4 Plan 01: Differentiable Renderer and Multi-Objective Loss Summary

**Differentiable refractive silhouette renderer (vertex pre-projection via Snell's law + FoVOrthographicCameras) and multi-objective loss (weighted soft IoU + gravity prior + morph constraints + temporal hook) with 38 new tests**

## Performance

- **Duration:** 17 min
- **Started:** 2026-02-21T03:22:42Z
- **Completed:** 2026-02-21T03:39:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Fixed pytorch3d DLL import (ABI mismatch): downgraded torch 2.10+cu130 -> 2.9.1+cu128; torchvision to 0.24.1+cu128
- RefractiveCamera wraps RefractiveProjectionModel.project() into PyTorch3D NDC coordinates; camera image size auto-derived from K matrix
- RefractiveSilhouetteRenderer: pre-projects world-space mesh vertices to NDC via RefractiveCamera, then renders with FoVOrthographicCameras (identity transform); alpha channel gives differentiable silhouette
- soft_iou_loss: differentiable 1-IoU with optional (y1,x1,y2,x2) crop region to focus gradients on fish bounding box
- compute_angular_diversity_weights: per-camera weights from minimum angular separation of view directions (R.T @ [0,0,-1]) raised to temperature exponent
- multi_objective_loss: weighted per-camera IoU + gravity prior (theta^2) + morph constraints (relu-based kappa/s bounds) + temporal hook (inactive at temporal_state=None)
- 38 unit tests: 11 renderer tests + 27 loss tests; all 270 total tests pass

## Task Commits

1. **Task 1: Fix pytorch3d import and build RefractiveCamera + RefractiveSilhouetteRenderer** - `7284115` (feat)
2. **Task 2: Build multi-objective loss with angular diversity weighting** - `d474ec3` (feat)

**Plan metadata:** _(final docs commit — see below)_

## Files Created/Modified

- `src/aquapose/optimization/renderer.py` - RefractiveCamera (project_to_ndc, get_image_size) + RefractiveSilhouetteRenderer (render with identity ortho camera)
- `src/aquapose/optimization/loss.py` - soft_iou_loss, compute_angular_diversity_weights, multi_objective_loss
- `src/aquapose/optimization/__init__.py` - Public API exports for 5 symbols
- `tests/unit/optimization/__init__.py` - Package init (empty)
- `tests/unit/optimization/test_renderer.py` - 11 renderer tests (shape, NDC range, gradient, multi-camera)
- `tests/unit/optimization/test_loss.py` - 27 loss tests (IoU correctness, angular weights, morph bounds, gradient)

## Decisions Made

- **Vertex pre-projection approach**: The plan specified a `RefractiveCamera` with `transform_points()` that `MeshRasterizer` would call. However, `MeshRasterizer.transform()` internally calls `cameras.get_world_to_view_transform()` and `cameras.get_ndc_camera_transform()` — methods not in the simple wrapper class. The cleanest solution (and closest to what the research doc described as "alternative approach") is to pre-project vertices to NDC using `RefractiveCamera.project_to_ndc()`, build an NDC-space Meshes object, then render with `FoVOrthographicCameras` (which acts as identity). Keeps refractive projection in the autograd graph and avoids implementing the full PyTorch3D camera interface.

- **Camera image size from K**: `RefractiveProjectionModel` stores K but not the image dimensions. For NDC conversion we need (H, W). Approximation: H = round(2*cy), W = round(2*cx). Callers can override with explicit `camera_image_sizes` list. Works correctly for AquaPose cameras where cx=800, cy=600 → 1600x1200.

- **Torch downgrade to 2.9.1+cu128**: As identified in the research document, this was required. torchvision 0.20.1+cu128 doesn't exist; 0.24.1+cu128 was used instead (the correct compatible version).

- **Temperature documentation corrected**: Initial implementation and docstring said "higher temperature = more uniform weights" but the formula `w = (min_angle/max)^T` means higher T exaggerates differences (small base values drop faster). Docstring updated to reflect actual behavior.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] MeshRasterizer rejects custom RefractiveCamera class**
- **Found during:** Task 1 (renderer implementation)
- **Issue:** `MeshRasterizer.transform()` calls `cameras.get_world_to_view_transform()` and `cameras.get_ndc_camera_transform()` — the simple custom camera wrapper doesn't implement these. Rendering fails with `AttributeError`.
- **Fix:** Switched to vertex pre-projection approach: `RefractiveCamera.project_to_ndc()` pre-projects all mesh vertices to NDC, builds a pre-projected NDC Meshes, renders with `FoVOrthographicCameras` (identity). Differentiability preserved since refraction is in the forward pass.
- **Files modified:** src/aquapose/optimization/renderer.py (redesigned implementation)
- **Verification:** Renderer produces non-empty alpha maps and gradients flow through all 5 FishState params.
- **Committed in:** 7284115 (Task 1 commit)

**2. [Rule 1 - Bug] NDC values out of range with wrong camera image size**
- **Found during:** Task 1 (test_renderer_produces_nonempty_silhouette failed — all-zero alpha)**
- **Issue:** Test fixture used `RefractiveSilhouetteRenderer(image_size=(200,200))` but the camera has intrinsics for a 1600x1200 sensor. When converting pixel coords (u~63, v~600) to NDC using H=200, W=200, the result is way outside [-1,1]. Fish vertices project outside the rasterizer view frustum → empty alpha map.
- **Fix:** Added `_image_size_from_K(K)` helper deriving (H, W) from camera intrinsics; `RefractiveCamera` uses camera's native image size for NDC conversion; renderer's `image_size` is only the rasterizer output resolution.
- **Files modified:** src/aquapose/optimization/renderer.py, tests/unit/optimization/test_renderer.py
- **Verification:** Alpha map now has nonzero pixels (>0.0) for fish at depth 1.5m.
- **Committed in:** 7284115 (Task 1 commit)

**3. [Rule 1 - Bug] Temperature test assertion inverted**
- **Found during:** Task 2 (test_temperature_affects_spread failed)**
- **Issue:** Test asserted spread_low >= spread_high, but the formula `w=(x)^T` produces MORE spread at higher T (not less). Test had the inequality backwards.
- **Fix:** Corrected test assertion to `spread_high >= spread_low`; updated docstring in `compute_angular_diversity_weights` to accurately describe temperature behavior.
- **Files modified:** tests/unit/optimization/test_loss.py, src/aquapose/optimization/loss.py
- **Verification:** Test passes.
- **Committed in:** d474ec3 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 bugs in implementation, 1 bug in test)
**Impact on plan:** All fixes necessary for correctness. Vertex pre-projection is a valid alternative approach documented in the research. No scope creep.

## Issues Encountered

- torchvision 0.20.1+cu128 doesn't exist; closest compatible version is 0.24.1+cu128 (matches torch 2.9.1+cu128).

## Next Phase Readiness

- `RefractiveSilhouetteRenderer.render(meshes, cameras, camera_ids)` ready for Plan 02 optimizer loop
- `multi_objective_loss()` ready for integration; temporal hook present but inactive
- `compute_angular_diversity_weights()` ready for per-camera weighting
- Gradient flow verified end-to-end: FishState params -> build_fish_mesh -> renderer -> alpha -> loss -> backward
- 270 total tests passing, no regressions from torch downgrade

## Self-Check

Files verified to exist:
- [x] src/aquapose/optimization/renderer.py
- [x] src/aquapose/optimization/loss.py
- [x] src/aquapose/optimization/__init__.py
- [x] tests/unit/optimization/__init__.py
- [x] tests/unit/optimization/test_renderer.py
- [x] tests/unit/optimization/test_loss.py

Commits verified: 7284115, d474ec3 (both present in git log)

## Self-Check: PASSED

---
*Phase: 04-per-fish-reconstruction*
*Completed: 2026-02-21*
