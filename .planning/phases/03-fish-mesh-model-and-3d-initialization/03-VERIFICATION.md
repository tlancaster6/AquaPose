---
phase: 03-fish-mesh-model-and-3d-initialization
verified: 2026-02-20T02:00:14Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 3: Fish Mesh Model and 3D Initialization Verification Report

**Phase Goal:** A fully differentiable parametric fish mesh model exists and can produce a plausible first-frame 3D state estimate from coarse keypoint detections, ready to be handed to the optimizer
**Verified:** 2026-02-20T02:00:14Z
**Status:** PASSED
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths -- Plan 01 (Mesh Model)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | build_fish_mesh(list[FishState]) returns a watertight triangle mesh | VERIFIED | test_watertight_mesh passes: edge counter confirms every undirected edge shared by exactly 2 faces |
| 2 | Gradients flow from mesh vertices back through all 5 state parameters (p, psi, theta, kappa, s) | VERIFIED | Live backward() pass: all 5 .grad tensors non-None and finite; test_gradients_flow_through_* tests all pass |
| 3 | Near-zero curvature (kappa ~= 0) produces finite vertices and finite gradients (no NaN) | VERIFIED | test_zero_curvature_no_nan passes; live check at kappa=0.0 exact: kappa.grad=0.000967, all verts finite |
| 4 | Free cross-section mode allows per-section height and width as optimizable parameters | VERIFIED | test_free_cross_section_gradients passes; heights.grad and widths.grad non-None and finite |
| 5 | Batch API: build_fish_mesh accepts list[FishState] and returns Meshes batch | VERIFIED | test_batch_build passes: 3 FishStates returns Meshes with len=3 |

**Score:** 5/5 truths verified

### Observable Truths -- Plan 02 (3D Initialization)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 6 | Given a binary mask, extract_keypoints returns center + 2 endpoint keypoints along major axis | VERIFIED | 13 tests pass including horizontal/vertical/diagonal rectangle cases, canonical sign, batch API |
| 7 | Given keypoints from >=3 cameras, triangulate_keypoint produces a 3D point via refractive ray casting | VERIFIED | test_triangulate_round_trip passes (<1mm error); test_triangulate_requires_3_cameras enforces minimum |
| 8 | Given 3 triangulated keypoints, init_fish_state produces a FishState with plausible position/heading/scale | VERIFIED | 6 init_fish_state tests pass; kappa=0, s=endpoint_distance, psi/theta from unit heading |
| 9 | All initialization APIs accept lists (batch-first design) | VERIFIED | extract_keypoints_batch, init_fish_states_from_masks both implemented and tested |
| 10 | Triangulation uses refractive ray casting from Phase 1, not pinhole | VERIFIED | triangulator.py imports RefractiveProjectionModel and triangulate_rays from aquapose.calibration.projection; model.cast_ray() called per camera |

**Score:** 5/5 truths verified

**Overall Score: 10/10 truths verified**

---

## Required Artifacts

### Plan 01: Mesh Model

| Artifact | Expected | Status | Details |
|----------|----------|--------|------|
| src/aquapose/mesh/state.py | FishState dataclass with 5 tensor fields | VERIFIED | @dataclass with p(3,), psi(), theta(), kappa(), s() all torch.Tensor, Google docstring |
| src/aquapose/mesh/spine.py | Circular arc spine with stable kappa~=0 | VERIFIED | build_spine_frames exported; torch.sinc for sin/kappa; torch.where for (1-cos)/kappa; 147 lines |
| src/aquapose/mesh/cross_section.py | Elliptical cross-section vertex generation | VERIFIED | build_cross_section_verts exported; (N,M,3) via broadcasting; gradient-transparent |
| src/aquapose/mesh/builder.py | Mesh assembly and PyTorch3D Meshes wrapping | VERIFIED | build_fish_mesh + _build_faces; 188 lines; Meshes(verts=verts_list, faces=faces_list) |
| src/aquapose/mesh/profiles.py | Default cichlid profile and CrossSectionProfile | VERIFIED | CrossSectionProfile dataclass + DEFAULT_CICHLID_PROFILE with 7 sections |
| src/aquapose/mesh/__init__.py | Public API exports | VERIFIED | Exports 6 symbols with __all__; all confirmed importable |

### Plan 02: 3D Initialization

| Artifact | Expected | Status | Details |
|----------|----------|--------|------|
| src/aquapose/initialization/keypoints.py | PCA-based keypoint extraction | VERIFIED | extract_keypoints + extract_keypoints_batch; canonical sign; ValueError on empty mask; 89 lines |
| src/aquapose/initialization/triangulator.py | Multi-camera triangulation and FishState estimation | VERIFIED | triangulate_keypoint, init_fish_state, init_fish_states_from_masks; >=3 cameras enforced; 155 lines |
| src/aquapose/initialization/__init__.py | Public API for initialization module | VERIFIED | Exports 5 symbols with __all__; all confirmed importable |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|------|
| mesh/builder.py | mesh/spine.py | build_spine_frames call | VERIFIED | Line 8 import; line 91 called in _build_single_mesh_verts |
| mesh/builder.py | mesh/cross_section.py | build_cross_section_verts call | VERIFIED | Line 6 import; line 117 called with all required args |
| mesh/builder.py | pytorch3d.structures.Meshes | Meshes(verts=...) wrapping | VERIFIED | Line 4 import; line 187 return Meshes(verts=verts_list, faces=faces_list) |
| initialization/triangulator.py | calibration/projection.py | cast_ray and triangulate_rays | VERIFIED | Line 8 imports both; line 41 model.cast_ray() per camera; line 48 triangulate_rays called |
| initialization/triangulator.py | mesh/state.py | FishState construction | VERIFIED | Line 10 import; line 85 FishState(...) constructed from triangulated data |

---

## Test Coverage

| Test File | Tests | Status |
|-----------|-------|--------|
| tests/unit/mesh/test_state.py | 5 | All pass |
| tests/unit/mesh/test_spine.py | 10 | All pass |
| tests/unit/mesh/test_cross_section.py | 9 | All pass |
| tests/unit/mesh/test_builder.py | 11 | All pass |
| tests/unit/initialization/test_keypoints.py | 13 | All pass |
| tests/unit/initialization/test_triangulator.py | 16 | All pass |
| **Total Phase 3** | **64** | **All pass** |

Full test run result: 173 passed, 21 deselected (slow) in 20.49s.

---

## Anti-Patterns Found

None. Scan of all Phase 3 source files:

- No TODO/FIXME/PLACEHOLDER/HACK comments in any source file
- No empty implementations (return null, return {}, placeholder bodies)
- The single `return []` at triangulator.py line 114 is a valid early-exit for n_cameras==0, not a stub
- All public functions have type hints and Google-style docstrings

---

## Human Verification Required

None. All aspects of the goal are verifiable programmatically:

- Watertight geometry: verified by edge-counting algorithm in test_watertight_mesh (every edge in exactly 2 faces)
- Gradient flow: verified by live backward() pass through all 5 parameters including kappa=0.0 exact
- Refractive vs. pinhole: verified by import tracing -- triangulator calls model.cast_ray() from calibration module

---

## Implementation Quality Notes

Notable design choices that increase robustness:

1. **kappa=0 stability via torch.sinc**: sin(kappa*t*s)/kappa computed as t*s * torch.sinc(kappa*t*s/pi), exact and smooth at kappa=0 without branching on the main arc direction.

2. **torch.where branching for cosine term**: (1-cos(kappa*t*s))/kappa uses torch.where(|kappa|<1e-4, 0, expr). First-order approximation is O(kappa) accurate. kappa.grad confirmed finite at kappa=0.0 exact.

3. **Gram-Schmidt fallback**: When heading dot world-up exceeds 0.99, orthogonalization switches to [1,0,0] as reference, preventing degenerate dorsal frames for near-vertical fish.

4. **Spine centering at t=0.5**: World translation by state.p happens in builder, not in spine generation. Keeps spine geometry independent of world position.

5. **Faces precomputed once**: _build_faces creates LongTensor once per build_fish_mesh call, reused across all batch members.

6. **Refractive triangulation**: model.cast_ray() applies Snell law at the air-water interface, not a pinhole approximation.

---

## Commits Verified

| Commit | Description | Exists |
|--------|-------------|--------|
| d22ae89 | feat(03-01): FishState, profiles, spine, and cross-section modules with tests | Yes |
| 40e4c0d | feat(03-01): mesh builder, PyTorch3D Meshes wrapping, and integration tests | Yes |
| 5122c9c | feat(03-02): implement PCA keypoint extraction from binary masks | Yes |
| d356a69 | feat(03-02): implement refractive triangulation and FishState initialization | Yes |

---

_Verified: 2026-02-20T02:00:14Z_
_Verifier: Claude (gsd-verifier)_
