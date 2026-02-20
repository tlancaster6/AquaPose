# Phase 3: Fish Mesh Model and 3D Initialization - Research

**Researched:** 2026-02-19
**Domain:** PyTorch differentiable parametric mesh, swept surface generation, PCA-based keypoint extraction, refractive triangulation
**Confidence:** HIGH — PyTorch3D API verified via official docs; codebase examined directly; one MEDIUM-confidence area (PyTorch3D installation with torch 2.10+CUDA13, flagged clearly)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Mesh geometry**
- Swept cross-sections along a spine curve (not deformable template)
- Body trunk only — no fins modeled; fin silhouette contribution treated as noise in Phase 4 loss
- 5-8 cross-sections along the spine (coarse resolution)
- Cross-sections are left-right symmetric about their local normal plane, but overall mesh is asymmetric when the spine is curved

**Cross-section profiles**
- Elliptical cross-sections — each defined by height and width (2 parameters per section)
- Free cross-section mode: both height and width are optimizable per section
- Hardcoded default profile ships with the mesh (cichlid-like: tapered head, wider mid-body, narrow tail) as starting point for optimization
- 8-12 vertices around each ellipse to generate mesh surface

**State vector & posing**
- State vector: {p, ψ/θ, κ, s} — position (3D), heading as yaw + pitch (2 angles), curvature (single constant arc), scale (uniform multiplier)
- Curvature (κ): single constant arc — entire spine bends as a circular arc (1 parameter)
- Heading: yaw (ψ) + pitch (θ) — allows fish to tilt nose-up/down, not just rotate in XY plane
- Scale (s): uniform scale factor on a unit-length template — s=0.15 means a 15cm fish; cross-section positions defined as fractions along [0, 1]

**Keypoint initialization**
- Keypoints derived from binary mask pixels (not manual clicks, not a learned detector):
  1. Get binary mask pixel coordinates as Nx2 array
  2. Centroid (mean) = center keypoint
  3. PCA on coordinates → first component = major axis direction
  4. Project all pixels onto first component → min/max = two endpoint keypoints
- Head vs tail disambiguation deferred to downstream (Phase 4's 2-start forward/180° flip)
- Minimum 3 cameras required for triangulation (matching success criterion)
- Triangulation uses refractive ray casting from Phase 1 (not pinhole approximation)
- Testing uses synthetic masks at known positions (Phase 2 not required)

### Claude's Discretion
- Exact cross-section positions along the spine (spacing strategy)
- Default cichlid profile dimensions (height/width ratios at each section)
- Triangulation algorithm (DLT, midpoint, etc.)
- PyTorch3D mesh format specifics and conversion details
- Watertight mesh closure at head and tail tips

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 3 has three distinct technical components: (1) parametric mesh construction via swept elliptical cross-sections, (2) PyTorch3D integration for downstream rasterization, and (3) PCA-based keypoint extraction and refractive triangulation for cold-start initialization. All three are implementable with existing PyTorch primitives — no specialized new libraries are required beyond PyTorch3D itself.

The parametric mesh is a pure PyTorch tensor computation: given a state vector, generate a spine curve as a circular arc, position elliptical cross-sections along it, and connect them into a triangle mesh. The key insight is that all operations (rotation matrices from angles, cross-section vertex positions, face connectivity) must use differentiable PyTorch ops so gradients flow back through the state vector to Phase 4's optimizer. The faces tensor (connectivity) is integer-valued and non-differentiable, but that is fine — faces are computed once from geometry (number of cross-sections, vertices per ellipse) and reused across forward passes.

The critical installation risk for this phase is PyTorch3D compatibility. The current hatch environment has PyTorch 2.10+CUDA13. PyTorch3D's officially supported range is PyTorch 2.1–2.4 (as of v0.7.9, released November 2024). Third-party prebuilt wheels (miropsota) cover torch 2.3–2.9.1 but have no Windows+CUDA13 builds. Building from source on Windows with CUDA13 is possible but the compilation may hit issues with newer CUDA headers. The safest approach for Phase 3 is to first attempt the miropsota third-party wheel for the closest matching torch version, then fall back to CPU-only source build (`FORCE_CUDA=0`) if CUDA compilation fails — CPU-only pytorch3d is sufficient for mesh validity testing and will still expose the Meshes API and SoftSilhouetteShader used in Phase 4.

**Primary recommendation:** Implement the parametric mesh as pure differentiable PyTorch tensor ops, wrap results in `pytorch3d.structures.Meshes`, and test installation of pytorch3d early as a blocking first task. For the triangulation, reuse `aquapose.calibration.projection.triangulate_rays` which already exists and is exactly the right algorithm.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10+cu130 (current env) | Differentiable spine curve, cross-section vertices, rotation matrices | All mesh math runs in autograd graph |
| pytorch3d | 0.7.9 (target) | `Meshes` container for Phase 4 rasterizer; `SoftSilhouetteShader` in Phase 4 | The industry standard for differentiable mesh rendering in PyTorch |
| numpy | >=1.24 (already installed) | PCA via `np.linalg.eigh` on covariance of pixel coords | scipy.linalg.eigh also works; numpy is already a dependency |
| aquapose.calibration.projection | (local, Phase 1) | `triangulate_rays`, `RefractiveProjectionModel.cast_ray` | Already implemented and tested; refractive triangulation is the Phase 1 deliverable |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | >=1.11 (already installed) | Alternative for PCA if numpy covariance path has numerical issues | Only if numpy path has problems with degenerate masks |
| fvcore | latest | Required pytorch3d dependency for `iopath` | Install alongside pytorch3d |
| iopath | latest | Required pytorch3d dependency | Install alongside pytorch3d |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Swept cross-sections (locked) | Deformable template mesh | Template requires a mesh to start with; swept cross-sections are fully parametric and require zero pre-existing assets |
| PCA on mask pixels (locked) | Learned keypoint detector | PCA works on any binary mask without training data; slower per-frame but zero data requirement |
| `triangulate_rays` (already in codebase) | DLT / midpoint / RANSAC | Linear least-squares on (I - dd^T) is the standard multi-view formulation; already implemented; RANSAC overkill for 3 clean keypoints |
| pytorch3d Meshes (locked for Phase 4 compat) | Open3D TriangleMesh | Open3D not importable in hatch env (AquaMVS CLAUDE.md confirms); pytorch3d is the Phase 4 requirement |

**Installation (pytorch3d — see pitfalls for CUDA13 details):**
```bash
# Option A: miropsota third-party wheel (try first, may not have torch 2.10)
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
    "pytorch3d==0.7.9+pt<CLOSEST_VER>cu130"

# Option B: CPU-only source build (guaranteed to work, no CUDA ops)
FORCE_CUDA=0 pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Option C: Source build with CUDA (may fail on CUDA13 headers — see pitfalls)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Required dependencies before any option:
pip install fvcore iopath
```

---

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/
├── mesh/
│   ├── __init__.py         # Public API: FishMesh, FishState, build_fish_mesh
│   ├── state.py            # FishState dataclass {p, psi, theta, kappa, s}
│   ├── spine.py            # Circular arc spine generation from state
│   ├── cross_section.py    # Elliptical cross-section vertex generation
│   ├── builder.py          # FishMesh: assembles spine + cross-sections into Meshes
│   └── profiles.py         # DEFAULT_CICHLID_PROFILE (hardcoded height/width ratios)
├── calibration/
│   └── (existing: projection.py has triangulate_rays)
└── (new) initialization/
    ├── __init__.py         # Public API: extract_keypoints, triangulate_keypoints, init_fish_state
    ├── keypoints.py        # PCA-based keypoint extraction from binary masks
    └── triangulator.py     # Multi-camera keypoint triangulation → FishState

tests/
└── unit/
    └── mesh/
        ├── __init__.py
        ├── test_state.py           # FishState construction/validation
        ├── test_spine.py           # Circular arc properties, gradient flow
        ├── test_cross_section.py   # Ellipse vertex generation, symmetry
        ├── test_builder.py         # Watertight mesh, gradient flow to state
        └── test_profiles.py        # Default profile shape sanity
    └── initialization/
        ├── __init__.py
        ├── test_keypoints.py       # PCA extracts correct axis, synthetic masks
        └── test_triangulator.py    # Triangulation round-trip with synthetic cameras
```

### Pattern 1: FishState Dataclass
**What:** A plain Python dataclass (not `nn.Module`) holding the 7 raw parameters of the state vector. Keeps the API clean — the mesh builder takes a `FishState`, optimizers hold tensors separately.

**When to use:** Whenever passing pose information between modules.

```python
# src/aquapose/mesh/state.py
from dataclasses import dataclass
import torch

@dataclass
class FishState:
    """Fish pose state vector {p, ψ, θ, κ, s}.

    Attributes:
        p: 3D position (center of fish) in world frame, shape (3,), float32.
        psi: Yaw angle (rotation about world Z axis), radians, shape (), float32.
        theta: Pitch angle (nose-up/down tilt), radians, shape (), float32.
        kappa: Spine curvature (1/radius of circular arc), shape (), float32.
            Positive bends toward the fish's dorsal side.
        s: Uniform scale factor; s=0.15 means fish length is 0.15m, shape (), float32.
    """
    p: torch.Tensor      # (3,)
    psi: torch.Tensor    # scalar
    theta: torch.Tensor  # scalar
    kappa: torch.Tensor  # scalar
    s: torch.Tensor      # scalar
```

### Pattern 2: Circular Arc Spine Generation
**What:** Given yaw ψ, pitch θ, curvature κ, and scale s, generate N cross-section center points and local coordinate frames along a circular arc in 3D. The arc lies in the plane defined by the fish's heading and dorsal vectors.

**Key math:** A circular arc parameterized by arc length t ∈ [0, s] with curvature κ:
- If κ ≈ 0 (straight): positions are evenly spaced along the heading direction
- If κ ≠ 0: use the Frenet-Serret frame; positions trace a circle of radius 1/κ in the heading-dorsal plane

**Differentiability:** All ops — `torch.sin`, `torch.cos`, rotation matrix construction, cumulative sum for arc integration — are differentiable. The number of cross-sections N is a fixed integer (hyperparameter), not a tensor.

```python
# Heading vector from yaw + pitch (AquaPose world: +Z down)
# Yaw ψ rotates in XY plane; pitch θ tilts from XY toward Z
heading = torch.stack([
    torch.cos(psi) * torch.cos(theta),   # X component
    torch.sin(psi) * torch.cos(theta),   # Y component
    torch.sin(theta),                     # Z component (positive = nose tilted down)
])  # (3,)

# Dorsal vector: perpendicular to heading, in the plane of bending
# For zero-pitch, dorsal = [0, 0, 1] (pointing down into water)
# Use Gram-Schmidt to find a consistent "up" vector orthogonal to heading
```

**Arc position at fraction t along [0,1]:**
```python
# For kappa != 0, arc center is at:
# arc_center = p + (1/kappa) * dorsal
# position(t) = arc_center - (1/kappa) * (cos(kappa*t*s)*dorsal + sin(kappa*t*s)*heading)
# For kappa ~= 0 (straight):
# position(t) = p + t*s*heading
```

The transition between arc and straight must be smooth for autograd — use a `torch.where` with a small epsilon threshold or directly use the Taylor expansion of the arc formula near κ=0.

### Pattern 3: Elliptical Cross-Section Vertex Generation
**What:** For each cross-section center (with associated local frame), generate M evenly spaced vertices around an ellipse of given height h and width w.

```python
# M vertices per ellipse (e.g., M=10), angles at uniform spacing
angles = torch.linspace(0, 2*torch.pi, M+1)[:-1]  # (M,)
# Ellipse in local frame (lateral, dorsoventral)
local_verts = torch.stack([
    width * torch.cos(angles),   # lateral axis
    height * torch.sin(angles),  # dorsoventral axis
    torch.zeros(M),              # along-spine axis
], dim=-1)  # (M, 3)
# Transform to world frame: local_verts @ R_local.T + center
```

`height` and `width` are either fixed (from default profile) or `torch.nn.Parameter` tensors (free cross-section mode). Both paths are differentiable.

### Pattern 4: Triangle Face Connectivity (Fixed, Non-Differentiable)
**What:** Given N cross-sections and M vertices per cross-section, the triangle connectivity is fixed geometry — it does not change across forward passes. Build it once as a `torch.LongTensor` and reuse.

**Mesh topology:**
- Tube body: Connect adjacent cross-sections with quad strips, split into 2 triangles each
- Head cap: Fan triangulation from a single apex vertex at the head tip
- Tail cap: Fan triangulation from a single apex vertex at the tail tip

```python
def _build_faces(n_sections: int, verts_per_section: int) -> torch.LongTensor:
    """Build triangle face indices for swept surface mesh.

    Returns:
        faces: shape (F, 3), LongTensor. F = 2*(N-1)*M + 2*(M-1) for tube + caps.
    """
    faces = []
    M = verts_per_section
    # Tube: for each pair of adjacent sections (i, i+1), M quads → 2M triangles
    for i in range(n_sections - 1):
        base = i * M
        for j in range(M):
            j_next = (j + 1) % M
            v0, v1 = base + j, base + j_next
            v2, v3 = v0 + M, v1 + M
            faces.extend([[v0, v1, v2], [v1, v3, v2]])
    # Head cap: apex at index n_sections * M
    apex_head = n_sections * M
    for j in range(M):
        j_next = (j + 1) % M
        faces.append([apex_head, j, j_next])
    # Tail cap: apex at index n_sections * M + 1
    apex_tail = n_sections * M + 1
    tail_base = (n_sections - 1) * M
    for j in range(M):
        j_next = (j + 1) % M
        faces.append([apex_tail, tail_base + j, tail_base + j_next])
    return torch.tensor(faces, dtype=torch.long)
```

### Pattern 5: PyTorch3D Meshes Container
**What:** Wrap the (verts, faces) output into a `pytorch3d.structures.Meshes` object. This is the format Phase 4's rasterizer expects.

```python
from pytorch3d.structures import Meshes

# Single mesh
mesh = Meshes(verts=[verts], faces=[faces])
# verts: (V, 3) float32 tensor WITH requires_grad=True (or computed from grad-enabled params)
# faces: (F, 3) long tensor (non-differentiable, fixed connectivity)

# Batch of meshes (batch-first API)
meshes = Meshes(verts=verts_list, faces=faces_list)
# verts_list: list of (V_i, 3) tensors — can vary per mesh if topology changes
# faces_list: list of (F_i, 3) tensors

# Access vertices (stays in autograd graph)
v = meshes.verts_list()   # list of (V, 3) tensors
v = meshes.verts_packed() # (sum(V_i), 3) — packed across batch

# Joining meshes into a batch
from pytorch3d.structures import join_meshes_as_batch
batch = join_meshes_as_batch([mesh_a, mesh_b])
```

**Gradient flow:** `Meshes.verts_list()` and `verts_packed()` preserve the autograd graph through the vertex tensors. Gradients flow from the rasterizer loss back through `verts` to the state parameters that produced them.

### Pattern 6: PyTorch3D Camera Convention for Phase 4 Prep
**What:** AquaPose uses OpenCV convention (K matrix, R/t extrinsics). PyTorch3D uses a different coordinate convention (+X left, +Y up, +Z out). The conversion is handled by `pytorch3d.utils.camera_conversions.cameras_from_opencv_projection`.

**This is a Phase 4 task**, but Phase 3 must produce meshes in the correct world-frame coordinate system so they render correctly in Phase 4 without coordinate flip. AquaPose world frame is +Z down (into water). PyTorch3D's camera convention is independent of world frame — it converts through the camera extrinsics. The conversion utility handles the sign flips.

```python
# Phase 4 will use this pattern (document here for awareness):
from pytorch3d.utils import cameras_from_opencv_projection

cameras = cameras_from_opencv_projection(
    R=R_batch,          # (N, 3, 3) rotation matrices
    tvec=t_batch,       # (N, 3) translation vectors
    camera_matrix=K_batch,  # (N, 3, 3) intrinsic matrices
    image_size=image_size,  # (N, 2) [H, W]
)
```

### Pattern 7: PCA Keypoint Extraction from Binary Mask
**What:** Given a binary mask (H, W) uint8 array, extract 3 keypoints: center (centroid), and two endpoints along the major axis.

```python
import numpy as np

def extract_keypoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract center and endpoint keypoints from a binary mask using PCA.

    Args:
        mask: Binary mask, shape (H, W), uint8 (0 or 255) or bool.

    Returns:
        center: Centroid pixel coordinate, shape (2,), float32. (u, v) = (col, row).
        endpoint_a: First endpoint along major axis, shape (2,), float32.
        endpoint_b: Second endpoint along major axis, shape (2,), float32.
        Note: head/tail assignment is ambiguous — caller handles disambiguation.
    """
    ys, xs = np.where(mask > 0)         # row (v), col (u) of foreground pixels
    coords = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2) in (u, v)
    center = coords.mean(axis=0)         # (2,) centroid
    centered = coords - center           # (N, 2)
    cov = (centered.T @ centered) / len(coords)  # (2, 2) covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # sorted ascending
    major_axis = eigenvectors[:, -1]     # (2,) first PC = largest eigenvalue
    projections = centered @ major_axis  # (N,) signed projections
    endpoint_a = center + major_axis * projections.max()
    endpoint_b = center + major_axis * projections.min()
    return center, endpoint_a, endpoint_b
```

**Edge cases:** Empty mask (no foreground pixels) → raise `ValueError`. Single-pixel mask → degenerate covariance; return same point for center and both endpoints. Circular fish (belly view) → PCA major axis still produces endpoints but they may be short.

### Pattern 8: Refractive Triangulation for Initialization
**What:** Given 3 keypoints (center, endpoint_a, endpoint_b) from each of ≥3 cameras, triangulate them into 3D using the existing `RefractiveProjectionModel.cast_ray` + `triangulate_rays`.

```python
# For each keypoint type (center, endpoint_a, endpoint_b):
# 1. Collect pixel coords from all cameras that have a valid mask
# 2. Cast refractive rays from each pixel
# 3. Call triangulate_rays(origins, directions) → 3D point

from aquapose.calibration.projection import triangulate_rays, RefractiveProjectionModel

def triangulate_keypoint(
    pixel_coords: list[tuple[float, float]],   # [(u, v), ...] per camera
    models: list[RefractiveProjectionModel],    # one per camera
) -> torch.Tensor:
    """Triangulate a single 2D keypoint from multiple camera views.

    Args:
        pixel_coords: List of (u, v) pixel coords, one per camera. Length >= 3.
        models: List of RefractiveProjectionModel, one per camera.

    Returns:
        point_3d: Triangulated 3D point in world frame, shape (3,), float32.
    """
    assert len(pixel_coords) >= 3, "Need at least 3 cameras"
    all_origins, all_dirs = [], []
    for (u, v), model in zip(pixel_coords, models):
        pixel = torch.tensor([[u, v]], dtype=torch.float32)
        origins, directions = model.cast_ray(pixel)  # (1,3), (1,3)
        all_origins.append(origins[0])
        all_dirs.append(directions[0])
    origins_t = torch.stack(all_origins)    # (N, 3)
    dirs_t = torch.stack(all_dirs)          # (N, 3)
    return triangulate_rays(origins_t, dirs_t)  # (3,)
```

### Pattern 9: State Estimation from 3 Triangulated Keypoints
**What:** Given 3D center, and two 3D endpoints (head/tail ambiguous), compute an initial FishState.

```python
def keypoints_to_fish_state(
    center_3d: torch.Tensor,       # (3,) world position
    endpoint_a_3d: torch.Tensor,   # (3,) one end
    endpoint_b_3d: torch.Tensor,   # (3,) other end
) -> FishState:
    """Convert triangulated keypoints to a FishState estimate.

    Heading is taken from endpoint_a to endpoint_b (disambiguation deferred).
    Curvature is initialized to 0 (straight fish). Scale is estimated from
    the distance between endpoints.

    Args:
        center_3d: Fish center in world frame.
        endpoint_a_3d: One endpoint (head or tail, unknown which).
        endpoint_b_3d: Other endpoint.

    Returns:
        FishState with p=center, psi=estimated yaw, theta=estimated pitch,
        kappa=0, s=distance between endpoints.
    """
    axis = endpoint_b_3d - endpoint_a_3d          # (3,)
    length = torch.linalg.norm(axis)
    heading = axis / (length + 1e-8)              # unit vector
    psi = torch.atan2(heading[1], heading[0])     # yaw from XY
    theta = torch.asin(heading[2].clamp(-1, 1))   # pitch from Z component
    return FishState(p=center_3d, psi=psi, theta=theta,
                     kappa=torch.zeros(()), s=length)
```

### Anti-Patterns to Avoid
- **In-place ops on vertex tensors:** Use `torch.cat`, `torch.stack` instead of `.copy_()` or `+=` on tensors that require grad. In-place ops on leaf tensors with `requires_grad=True` raise errors in autograd.
- **Fixed face connectivity recomputed every forward pass:** Precompute faces tensor once during `__init__` and store it. Recomputing it every call wastes CPU time (faces are integer-valued, not differentiable, no need to trace through their construction).
- **Using `np.linalg.eig` instead of `eigh` for covariance:** The covariance matrix is symmetric; `eigh` is faster and numerically stable for symmetric matrices. `eig` may return complex eigenvalues for nearly-singular cases.
- **Passing `kappa=0` exactly into arc formula:** Division by zero if using `1/kappa` for radius. Use a branch (`torch.where(|kappa| < eps, straight_path, arc_path)`) or the Taylor-expanded arc formula valid near κ=0.
- **Not normalizing ray directions before `triangulate_rays`:** `triangulate_rays` assumes unit directions (the `d.unsqueeze(1) @ d.unsqueeze(0)` assumes unit d). `cast_ray` already returns unit directions so this is only a risk if directions come from elsewhere.
- **Batch-first design omitted:** The phase requires all APIs to accept lists of FishState, even for single-fish use. Design `build_fish_mesh(states: list[FishState]) -> Meshes` from the start — do not add batch support later.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-view triangulation | Custom DLT or midpoint method | `aquapose.calibration.projection.triangulate_rays` | Already implemented, tested, correct for N rays |
| Refractive ray casting | Custom Snell's law from pixels | `RefractiveProjectionModel.cast_ray` | Phase 1 deliverable, cross-validated against AquaCal |
| Triangle mesh container | Custom mesh class | `pytorch3d.structures.Meshes` | Phase 4 rasterizer requires this exact interface |
| PCA eigenvectors | Power iteration or manual SVD | `numpy.linalg.eigh` on covariance | Numerically stable, fast, correct for 2x2 symmetric matrix |
| Coordinate frame rotation matrices | Custom angle-to-matrix code | `torch.stack` + `torch.cos/sin` with the standard ZYX Euler formula | PyTorch autograd traces through trig ops; no separate library needed |
| OpenCV-to-PyTorch3D camera conversion | Manual sign flips | `pytorch3d.utils.cameras_from_opencv_projection` | Official utility handles the +X-left/+Y-up convention correctly |

**Key insight:** All the hard geometric refractive math already exists from Phase 1. Phase 3's initialization is a data flow problem: mask pixels → PCA → pixel keypoints → cast_ray → triangulate_rays → FishState. The hard work is the parametric mesh builder itself, which is a pure PyTorch tensor computation with no existing library shortcut.

---

## Common Pitfalls

### Pitfall 1: PyTorch3D CUDA Compilation Failure with torch 2.10+CUDA13

**What goes wrong:** Attempting to build pytorch3d from source with CUDA 13.0 may fail with errors in CUDA header files (e.g., `vector_types.h` template parsing errors). This has been documented for CUDA 12.8 + torch 2.7 (GitHub issue #1970, unresolved as of early 2025). The same class of error is likely with CUDA 13.

**Why it happens:** PyTorch3D's CUDA kernel source files use C++17 template patterns that conflict with newer CUDA stdlibc headers. The ABI boundary between PyTorch and PyTorch3D is strict — a pytorch3d wheel compiled against torch 2.4 will not load with torch 2.10.

**How to avoid:**
1. First try miropsota third-party wheels: `pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "pytorch3d==0.7.9+pt<VER>cu130"` — check their releases page for the closest torch version to 2.10.
2. If CUDA compilation fails, install CPU-only from source: `FORCE_CUDA=0 pip install "git+https://github.com/facebookresearch/pytorch3d.git"`. CPU-only pytorch3d still provides `Meshes`, `MeshRasterizer`, and `SoftSilhouetteShader` — CUDA ops only accelerate the rasterization, which is needed in Phase 4 not Phase 3.
3. As a last resort: downgrade the hatch env to torch 2.4 + CUDA 12.4 for which prebuilt wheels exist. This is a significant env change — document it clearly.

**Warning signs:** `ImportError: CUDA-compiled PyTorch3D does not match installed PyTorch` or `undefined symbol: _ZN...` on `import pytorch3d`.

### Pitfall 2: Gradient Flow Broken by Faces Tensor in Autograd

**What goes wrong:** If vertex positions are computed correctly but gradients are `None` at the state parameters, the likely cause is a detach or in-place operation somewhere in the mesh construction path.

**Why it happens:** `torch.cat` over a list that includes a `.detach()` tensor silently breaks the graph. Non-differentiable path: building the vertex list in a loop with `.numpy()` / `.item()` calls converts to CPU numpy, breaking the autograd tape.

**How to avoid:** Keep all vertex math in PyTorch tensors with float32 dtype and no `.detach()`. Verify with `verts.requires_grad` check. After `mesh = Meshes(verts=[verts], faces=[faces])`, confirm `mesh.verts_list()[0].requires_grad == True`.

**Verification test:**
```python
state = FishState(p=torch.zeros(3, requires_grad=True), ...)
mesh = build_fish_mesh([state])
verts = mesh.verts_list()[0]
loss = verts.sum()
loss.backward()
assert state.p.grad is not None, "Gradients not flowing to state.p"
```

### Pitfall 3: Curvature κ ≈ 0 Division by Zero in Arc Formula

**What goes wrong:** The standard circular arc formula uses `radius = 1/kappa`. When kappa → 0 (straight fish), this produces infinity, leading to NaN vertex positions and NaN gradients.

**Why it happens:** Initialization sets `kappa=0`. The optimizer may keep kappa near zero for straight fish.

**How to avoid:** Use a numerically stable formulation. Two strategies:
- **Taylor expansion near zero:** The arc displacement `(1/κ) * sin(κ*t)` has the Taylor expansion `t - κ²t³/6 + ...`. Use `torch.where(|κ| < 1e-4, t * s, (1/κ) * torch.sin(κ * t * s))` — but `torch.where` still evaluates both branches; use `torch.sinc` if available, or split into a smooth function.
- **Stable rewrite:** Express arc positions using only `sin(κ*t*s)` and `(1-cos(κ*t*s))` divided by κ, then use `torch.special.sinc` equivalents. Simplest: `x = (sin(κ*t*s) / (κ + ε)) * heading + ((1-cos(κ*t*s)) / (κ + ε)) * dorsal`, where ε is a small stabilizer (1e-8). This is smooth everywhere including κ=0.

### Pitfall 4: Cross-Section Normal Frame Discontinuity (Torsion)

**What goes wrong:** When computing the local frame at each cross-section center along the spine, naïve use of `cross(heading, world_up)` to get the lateral vector fails when the heading is parallel to world_up (fish pointing straight down/up). The result is a zero vector, causing NaN in normalization.

**Why it happens:** The standard Gram-Schmidt approach for building a local frame has a singularity along the reference axis.

**How to avoid:** Use an axis-aligned fallback: if `|heading · [0,0,1]| > 0.99`, use `[1,0,0]` as the reference instead of `[0,0,1]`. Or use the Rodrigues / quaternion-based parallel transport frame which avoids the singularity entirely (but is more complex). For the cichlid-like fish use case (mostly horizontal swimming), the simple Gram-Schmidt approach with a world-up fallback is sufficient.

### Pitfall 5: PCA Axis Direction Sign Ambiguity in 2D

**What goes wrong:** `np.linalg.eigh` returns eigenvectors with arbitrary sign convention (the vector and its negation are both valid eigenvectors). Running the same mask twice may return a flipped axis direction, making endpoint_a and endpoint_b swap.

**Why it happens:** Eigenvector sign is undefined — both `v` and `-v` are valid. Different NumPy versions or CPU states may return different signs.

**How to avoid:** Enforce a canonical sign: after computing `major_axis`, check if `projections.max() > abs(projections.min())` — if not, negate `major_axis`. This ensures endpoint_a is always the "more positive" endpoint in a consistent sense. Since head/tail disambiguation is deferred to Phase 4, the exact choice doesn't matter as long as it is deterministic.

### Pitfall 6: PyTorch3D World Coordinate Convention Mismatch

**What goes wrong:** AquaPose world frame is +Z down (into water, standard computer vision for top-down cameras). PyTorch3D's internal convention for cameras is +X left, +Y up, +Z out of image plane. Passing AquaPose world-frame vertices directly into a naively-constructed PyTorch3D PerspectiveCameras renders a flipped or rotated image.

**Why it happens:** The coordinate axes are genuinely different between the two systems, but `cameras_from_opencv_projection` handles this conversion correctly. The mesh builder produces vertices in AquaPose world frame — that is correct. The camera construction for rendering must use `cameras_from_opencv_projection` (Phase 4 responsibility) to handle the conversion.

**How to avoid:** In Phase 3, produce vertex positions in AquaPose world frame (+Z down). In Phase 4, use `cameras_from_opencv_projection` to create PyTorch3D cameras from the AquaCal K/R/t. Do not flip vertex coordinates in the mesh builder.

---

## Code Examples

### Complete Mesh Builder Skeleton
```python
# src/aquapose/mesh/builder.py
# Source: this design (pattern established in research)

import torch
from pytorch3d.structures import Meshes
from .state import FishState
from .spine import build_spine_frames
from .cross_section import build_cross_section_verts
from .profiles import DEFAULT_CICHLID_PROFILE

N_SECTIONS = 6   # cross-sections along spine (within 5-8 range)
M_VERTS = 10     # vertices per ellipse (within 8-12 range)

def build_fish_mesh(
    states: list[FishState],
    profile: dict | None = None,
) -> Meshes:
    """Build a differentiable parametric fish mesh from a batch of states.

    Args:
        states: List of FishState objects, one per fish.
        profile: Per-section height/width ratios. Defaults to DEFAULT_CICHLID_PROFILE.

    Returns:
        Meshes: PyTorch3D Meshes object with batch size len(states).
            Gradients flow through verts back to state parameters.
    """
    if profile is None:
        profile = DEFAULT_CICHLID_PROFILE
    faces = _build_faces(N_SECTIONS, M_VERTS)  # precomputed, non-differentiable
    verts_list, faces_list = [], []
    for state in states:
        verts = _build_single_mesh_verts(state, profile)  # (V, 3)
        verts_list.append(verts)
        faces_list.append(faces)
    return Meshes(verts=verts_list, faces=faces_list)
```

### Default Cichlid Profile (Discretionary Dimensions)
```python
# src/aquapose/mesh/profiles.py
# Cichlid-like proportions: tapered nose, widest at ~40% body, narrow caudal peduncle
# Heights and widths as fractions of body length s
# Sections at fractions 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 along [0, 1]
DEFAULT_CICHLID_PROFILE = {
    "section_positions": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 6 sections
    "heights": [0.06, 0.12, 0.14, 0.10, 0.06, 0.03],       # height / body_length
    "widths":  [0.04, 0.08, 0.10, 0.07, 0.04, 0.02],       # width / body_length
}
# Note: height > width since cichlids are taller than wide
# These are placeholders — values should be adjusted to match actual cichlid morphology
# during profile self-calibration (Phase 3 MESH-02 free cross-section mode)
```

### Gradient Verification Test Pattern
```python
# tests/unit/mesh/test_builder.py
import torch
import pytest
from aquapose.mesh.state import FishState
from aquapose.mesh.builder import build_fish_mesh

def test_gradients_flow_through_position():
    """Verify gradients propagate from mesh verts back to state.p."""
    p = torch.tensor([0.0, 0.0, 1.2], requires_grad=True)
    state = FishState(
        p=p,
        psi=torch.tensor(0.0),
        theta=torch.tensor(0.0),
        kappa=torch.tensor(0.0),
        s=torch.tensor(0.15),
    )
    mesh = build_fish_mesh([state])
    loss = mesh.verts_list()[0].sum()
    loss.backward()
    assert p.grad is not None
    assert torch.all(p.grad.isfinite())

def test_gradients_flow_through_curvature():
    """Gradients reach kappa — tests the near-zero stability."""
    kappa = torch.tensor(1e-6, requires_grad=True)  # near-zero curvature
    state = FishState(
        p=torch.zeros(3), psi=torch.zeros(()), theta=torch.zeros(()),
        kappa=kappa, s=torch.tensor(0.15),
    )
    mesh = build_fish_mesh([state])
    loss = mesh.verts_list()[0].sum()
    loss.backward()
    assert kappa.grad is not None
    assert kappa.grad.isfinite()
```

### PCA Keypoint Test Pattern
```python
# tests/unit/initialization/test_keypoints.py
import numpy as np
from aquapose.initialization.keypoints import extract_keypoints

def test_horizontal_rectangle():
    """PCA correctly finds major axis of a horizontal rectangle."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:60, 20:180] = 255   # horizontal bar: wider than tall
    center, ep_a, ep_b = extract_keypoints(mask)
    # Center should be near (100, 50)
    assert abs(center[0] - 100) < 5   # u ~ 100
    assert abs(center[1] - 50) < 5    # v ~ 50
    # Major axis should be horizontal (endpoints differ in u, not v)
    axis = ep_a - ep_b
    assert abs(axis[0]) > abs(axis[1])  # more u-variation than v-variation
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|---|---|---|
| Deformable template meshes (SMPL-style) | Swept parametric cross-sections | Template requires a pre-rigged mesh; swept cross-sections only require shape parameters |
| Manual keypoint annotation for initialization | PCA on binary mask pixels | Zero annotation cost, works with any mask quality |
| AquaCal NumPy triangulation (Brent search) | PyTorch `triangulate_rays` (linear least squares) | Phase 1 already implemented this; direct reuse |
| SoftRasterizer (external) | PyTorch3D SoftSilhouetteShader | PyTorch3D is the current standard; SoftRasterizer is the predecessor it was benchmarked against |

**Deprecated/outdated in PyTorch3D:**
- `SfMPerspectiveCameras`, `SfMOrthographicCameras`, `OpenGLPerspectiveCameras`, `OpenGLOrthographicCameras`: Deprecated aliases retained for backward compat. Use `PerspectiveCameras` and `cameras_from_opencv_projection` instead.

---

## Open Questions

1. **PyTorch3D installation with torch 2.10+CUDA13**
   - What we know: Official support tops out at torch 2.4. Third-party miropsota wheels go to 2.9.1 but may not have Windows+CUDA13. Source builds with CUDA 12.8 have compilation failures (issue #1970, unresolved). CUDA 13 is even newer.
   - What's unclear: Whether miropsota has a wheel for torch 2.10+cu130; whether the source build compilation errors affect CUDA 13 in the same way.
   - Recommendation: **Make pytorch3d installation the first task in this phase.** Try miropsota wheel first. If it fails, do CPU-only source build. If CPU-only works, proceed with development and note that Phase 4 may need to address CUDA rendering separately. Do not block mesh model development on this.

2. **Exact cross-section spacing strategy (Claude's Discretion)**
   - What we know: 5-8 cross-sections. Fish body shape changes fastest near head/tail (tapered), slowest at mid-body.
   - Recommendation: Use non-uniform spacing with denser sections at head and tail. Specifically: `[0.0, 0.1, 0.25, 0.50, 0.75, 0.90, 1.0]` (7 sections, denser at ends). This better captures the head taper and caudal peduncle narrowing that matter most for silhouette matching.

3. **Watertight cap closure method (Claude's Discretion)**
   - What we know: The tube body is open at both ends and must be capped for watertight mesh.
   - Recommendation: Single-apex fan triangulation at each end. Place the apex vertex at the cross-section center ± a small extension along the spine direction (e.g., half the inter-section spacing). This produces a cone-like head tip, which is geometrically reasonable for a fish snout and tail.

4. **Default cichlid profile dimensions (Claude's Discretion)**
   - What we know: Cichlids are roughly 1:2 aspect ratio (height:length) for the body, tapered at head and tail.
   - Recommendation: The profile values in the Code Examples section are reasonable starting points but should be validated visually by rendering the default mesh and comparing to reference images of the target cichlid species. The free cross-section mode (MESH-02) exists specifically to self-calibrate these from data.

5. **Batch-first for FishState: list vs. stacked tensors**
   - What we know: Phase success criterion 4 requires all APIs to accept lists of fish states.
   - Recommendation: Accept `list[FishState]` as the batch type (not stacked tensors). This is simpler to implement and aligns with how PyTorch3D's `Meshes` accepts `list[Tensor]`. The optimizer in Phase 4 can hold individual `FishState` objects in a list.

---

## Sources

### Primary (HIGH confidence — official docs and local codebase examined directly)
- `pytorch3d.readthedocs.io/en/latest/modules/structures.html` — Meshes class API, verts_list/padded/packed representations, join_meshes_as_batch
- `pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/shader.html` — SoftSilhouetteShader, BlendParams, sigma convention
- `pytorch3d.readthedocs.io/en/latest/notes/cameras.html` — Coordinate convention (+X left, +Y up, +Z out)
- `pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html` — PerspectiveCameras, cameras_from_opencv_projection
- `C:/Users/tucke/PycharmProjects/AquaPose/src/aquapose/calibration/projection.py` — triangulate_rays (existing implementation), RefractiveProjectionModel.cast_ray
- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/triangulation.py` — Reference triangulation implementation (same algorithm)

### Secondary (MEDIUM confidence — GitHub issues and discussions, verified against multiple sources)
- `github.com/facebookresearch/pytorch3d/issues/522` — OpenCV to PyTorch3D camera convention conversion (negative focal length approach, confirmed by developers)
- `github.com/facebookresearch/pytorch3d/issues/1962` — CUDA 12.8 + PyTorch 2.8 source build workaround (source build works on Linux, Windows WSL2)
- `github.com/facebookresearch/pytorch3d/issues/1970` — CUDA 12.8 + PyTorch 2.7 compilation failure on Windows (unresolved as of early 2025)
- `github.com/facebookresearch/pytorch3d/discussions/1752` — miropsota third-party wheel repository (wheels up to torch 2.9.1, no Windows+CUDA13 confirmed)
- `github.com/facebookresearch/pytorch3d/releases` — v0.7.9 latest release (November 2024), supports torch 2.1-2.4 officially

### Tertiary (LOW confidence — verify before acting)
- WebSearch results on torch 2.10+CUDA13 installation: No direct confirmation that pytorch3d can be installed with torch 2.10+CUDA13 found. CPU-only source build fallback remains the safe path.

---

## Metadata

**Confidence breakdown:**
- Parametric mesh architecture (swept cross-sections, PyTorch3D Meshes): HIGH — PyTorch3D API verified via official docs; math is standard differential geometry
- PyTorch3D installation with torch 2.10+CUDA13: MEDIUM-LOW — No confirmed working configuration found; CPU-only fallback is HIGH confidence
- PCA keypoint extraction: HIGH — standard algorithm, numpy API stable
- Triangulation reuse from Phase 1: HIGH — code exists and is tested
- Default cichlid profile dimensions: LOW — placeholder values, need visual validation

**Research date:** 2026-02-19
**Valid until:** pytorch3d install strategy should be re-evaluated if miropsota releases a torch 2.10 wheel; mesh architecture patterns are stable
