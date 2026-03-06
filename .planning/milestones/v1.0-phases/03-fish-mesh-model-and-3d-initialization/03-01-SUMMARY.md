---
phase: 03-fish-mesh-model-and-3d-initialization
plan: 01
subsystem: mesh
tags: [pytorch3d, differentiable-rendering, parametric-mesh, autograd, fish-pose]

# Dependency graph
requires:
  - phase: 01-calibration-and-refractive-geometry
    provides: "Refractive projection math and world coordinate conventions"
provides:
  - "FishState dataclass with 5 differentiable tensor fields (p, psi, theta, kappa, s)"
  - "build_spine_frames: stable circular arc spine with torch.sinc-based kappa=0 handling"
  - "build_cross_section_verts: elliptical cross-sections with gradient flow through h/w/s"
  - "build_fish_mesh: watertight PyTorch3D Meshes from list[FishState] with full gradient flow"
  - "DEFAULT_CICHLID_PROFILE: 7-section cichlid cross-section profile"
  - "Free cross-section mode: heights/widths as requires_grad tensors"
affects:
  - 03-02-initialization
  - 04-differentiable-rendering
  - 05-pose-optimization

# Tech tracking
tech-stack:
  added:
    - "pytorch3d-0.7.9+pt2.9.1cu128 (miropsota wheel, works with torch 2.10+cu130 on Windows)"
    - "fvcore-0.1.5, iopath-0.1.10 (pytorch3d prerequisites)"
  patterns:
    - "Circular arc spine using torch.sinc for stable sin(kappa*t*s)/kappa at kappa=0"
    - "torch.where branching for (1-cos(kappa*t*s))/kappa stable near kappa=0"
    - "Swept elliptical cross-sections via broadcasting: (N,1,1) * (1,M,1) * (N,1,3)"
    - "Watertight mesh via tube-body quads + head/tail apex fan caps"
    - "Faces precomputed once as LongTensor, reused across batch"
    - "PyTorch3D Meshes(verts=list, faces=list) preserves autograd graph through verts"

key-files:
  created:
    - src/aquapose/mesh/state.py
    - src/aquapose/mesh/profiles.py
    - src/aquapose/mesh/spine.py
    - src/aquapose/mesh/cross_section.py
    - src/aquapose/mesh/builder.py
    - tests/unit/mesh/__init__.py
    - tests/unit/mesh/test_state.py
    - tests/unit/mesh/test_spine.py
    - tests/unit/mesh/test_cross_section.py
    - tests/unit/mesh/test_builder.py
  modified:
    - src/aquapose/mesh/__init__.py
    - pyproject.toml

key-decisions:
  - "torch.sinc used for sin(kappa*t*s)/kappa stability: sinc(x/pi) = sin(x)/x, smooth at x=0"
  - "torch.where branching (not eps-addition) for (1-cos)/kappa — sinc approach not clean for cosine term"
  - "miropsota pytorch3d-0.7.9+pt2.9.1cu128 wheel works with torch 2.10+cu130 on Windows despite version mismatch"
  - "Watertight winding: tube uses (v0,v2,v1)+(v1,v2,v3), head cap uses (apex,j_next,j), tail cap uses (apex,j,j_next)"
  - "7 cross-sections at [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0] — denser at head/tail per RESEARCH recommendation"
  - "Spine centered at t=0.5 (midpoint); builder translates by state.p after spine generation"

patterns-established:
  - "Pattern: All mesh math uses pure PyTorch ops, no numpy/item()/detach() — autograd graph preserved end-to-end"
  - "Pattern: build_fish_mesh(states: list[FishState]) batch-first API from day one"

# Metrics
duration: 10min
completed: 2026-02-20
---

# Phase 3 Plan 01: Fish Mesh Model Summary

**Watertight differentiable parametric fish mesh with pytorch3d Meshes wrapping, full gradient flow through all 5 state params (p/psi/theta/kappa/s), and stable kappa=0 handling via torch.sinc**

## Performance

- **Duration:** 10 min
- **Started:** 2026-02-20T01:32:11Z
- **Completed:** 2026-02-20T01:42:00Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments

- FishState dataclass and CrossSectionProfile with DEFAULT_CICHLID_PROFILE (7 sections, denser at head/tail)
- Differentiable circular arc spine (build_spine_frames) with numerically stable kappa=0 via torch.sinc
- Elliptical cross-section vertices (build_cross_section_verts) with gradient flow through heights/widths/s
- Watertight mesh builder (build_fish_mesh) wrapping into pytorch3d Meshes with batch support
- 33 unit tests covering all geometry properties, gradient flow, and edge cases
- pytorch3d-0.7.9 successfully installed via miropsota wheel on Windows/CUDA13

## Task Commits

1. **Task 1: FishState, profiles, spine, and cross-section modules with tests** - `d22ae89` (feat)
2. **Task 2: Mesh builder, PyTorch3D Meshes wrapping, and integration tests** - `40e4c0d` (feat)

**Plan metadata:** _(final docs commit — see below)_

## Files Created/Modified

- `src/aquapose/mesh/state.py` - FishState dataclass with 5 tensor fields
- `src/aquapose/mesh/profiles.py` - CrossSectionProfile dataclass and DEFAULT_CICHLID_PROFILE
- `src/aquapose/mesh/spine.py` - build_spine_frames with torch.sinc stable kappa=0
- `src/aquapose/mesh/cross_section.py` - build_cross_section_verts with free mode gradient flow
- `src/aquapose/mesh/builder.py` - build_fish_mesh, _build_faces, watertight mesh assembly
- `src/aquapose/mesh/__init__.py` - Public API exports for all 6 symbols
- `tests/unit/mesh/test_state.py` - FishState construction, types, shapes, requires_grad
- `tests/unit/mesh/test_spine.py` - Arc geometry, tangent unit length, gradient flow, near-zero kappa
- `tests/unit/mesh/test_cross_section.py` - Circle/ellipse geometry, symmetry, gradient flow
- `tests/unit/mesh/test_builder.py` - Watertight edges, all 5 grad flows, free mode, batch, kappa=0
- `pyproject.toml` - pytorch3d installation instructions updated with working miropsota command

## Decisions Made

- **torch.sinc for arc formula**: `sin(kappa*t*s)/kappa = t*s * sinc(kappa*t*s/pi)` using `torch.sinc(x) = sin(pi*x)/(pi*x)`. This is smooth and exact at kappa=0 without branching.
- **torch.where for cosine term**: `(1-cos(kappa*t*s))/kappa` handled via `torch.where(|kappa|<1e-4, 0, expr)` since the sinc approach isn't clean for the cosine half. Near kappa=0 the perp term is O(kappa) so using 0 is accurate to first order.
- **pytorch3d miropsota wheel**: Installed `pytorch3d-0.7.9+pt2.9.1cu128` against torch 2.10+cu130. Despite the version mismatch (pt2.9.1 vs 2.10), the Meshes API works correctly for CPU tensor operations used in Phase 3.
- **Watertight winding order**: Tube quad (v0,v2,v1)+(v1,v2,v3) for CCW from outside; head cap (apex,j_next,j); tail cap (apex,j,j_next). Validated by counting edges — all 100% appear in exactly 2 faces.
- **Spine centering**: Midpoint (t=0.5) placed at origin; builder adds state.p translation. Keeps spine generation independent of world position.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_ellipse_aspect_ratio: M=10 doesn't land on ellipse axes**
- **Found during:** Task 1 (cross-section tests)
- **Issue:** The aspect ratio test assumed M=10 vertices would sample exact axis-aligned points (where max/min distance = h/w), but with 10 vertices the angular spacing is 36 degrees, missing the exact 90/270 degree points.
- **Fix:** Used M=1000 vertices in the test so vertices closely approximate the true extrema.
- **Files modified:** tests/unit/mesh/test_cross_section.py
- **Verification:** Test passes with atol=1e-3.

**2. [Rule 1 - Bug] Fixed test_left_right_symmetry: wrong symmetry axis**
- **Found during:** Task 1 (cross-section tests)
- **Issue:** Test checked that Y-component of v[j] and v[M-j] were opposites, but the ellipse formula has cos(angle) on Y (binormal): cos(2pi-x)=cos(x) so Y[j]==Y[M-j] (same, not opposite). Z (normal) has sin: sin(2pi-x)=-sin(x) so Z flips.
- **Fix:** Test now checks Z[j]==-Z[M-j] and Y[j]==Y[M-j] to match the actual math.
- **Files modified:** tests/unit/mesh/test_cross_section.py
- **Verification:** Test passes with atol=1e-5.

**3. [Rule 3 - Blocking] Installed pytorch3d before implementing builder**
- **Found during:** Task 2 start (prerequisite check)
- **Issue:** pytorch3d not installed; builder.py imports from pytorch3d.structures.
- **Fix:** Installed fvcore+iopath then pytorch3d via miropsota wheel.
- **Verification:** `from pytorch3d.structures import Meshes; print('OK')` succeeds.

---

**Total deviations:** 3 auto-fixed (2 bugs in tests, 1 blocking install)
**Impact on plan:** All fixes necessary for correctness. No scope creep.

## Issues Encountered

- pytorch3d miropsota wheel version mismatch (pt2.9.1cu128 vs torch 2.10+cu130): Works correctly for CPU mesh operations used in Phase 3. CUDA-accelerated rasterization (Phase 4) may need re-evaluation.

## Self-Check

Files verified to exist:
- [x] src/aquapose/mesh/state.py
- [x] src/aquapose/mesh/profiles.py
- [x] src/aquapose/mesh/spine.py
- [x] src/aquapose/mesh/cross_section.py
- [x] src/aquapose/mesh/builder.py
- [x] tests/unit/mesh/test_builder.py

Commits verified: d22ae89, 40e4c0d (both present in git log)

## Self-Check: PASSED

## Next Phase Readiness

- `build_fish_mesh(states: list[FishState]) -> Meshes` ready for Phase 4 rasterizer
- Gradient flow verified through all 5 state parameters and free cross-section heights/widths
- pytorch3d Meshes format compatible with Phase 4 SoftSilhouetteShader (pending CUDA verification)
- Phase 3 Plan 02 (initialization/triangulation) can proceed independently

---
*Phase: 03-fish-mesh-model-and-3d-initialization*
*Completed: 2026-02-20*
