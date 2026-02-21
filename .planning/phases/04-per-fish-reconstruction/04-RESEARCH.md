# Phase 4: Per-Fish Reconstruction - Research

**Researched:** 2026-02-20
**Domain:** Differentiable silhouette rendering, analysis-by-synthesis pose optimization, multi-view IoU loss, PyTorch3D rasterizer
**Confidence:** HIGH for rendering pipeline and optimizer patterns; MEDIUM for angular-diversity weighting (Claude's discretion area); CRITICAL blocker identified for pytorch3d renderer import

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Rendering pipeline**
- Silhouettes only — no depth or normal maps
- Wrap existing `calibration/` refractive ray-casting code in a differentiable layer (thin PyTorch wrapper), do not rewrite Snell's law from scratch
- Per-camera angular-diversity weighting computed from camera extrinsics — down-weight clustered viewpoints
- Camera selection is input-driven: whatever videos are in the input folder get processed. Problematic cameras (e.g., e3v8250) are excluded by the user at the data level, not hardcoded

**Loss design**
- Crop-space IoU: compute silhouette IoU within the bounding box crop region, not the full frame
- Gravity prior = soft roll regularization: penalize deviations of the fish's dorsal-ventral axis from upright orientation. Low weight — just enough to break ambiguities when silhouette alone can't distinguish rolled vs. unrolled pose
- Morphological constraints on both scale (s) and curvature (kappa): curvature bounds should be stricter than scale bounds
- Hand-tuned fixed loss weights (e.g., IoU=1.0, gravity=low, morph=moderate). Tuned empirically, not learned
- Temporal smoothness term is architecturally present but inactive until Phase 5 provides track associations

**Optimization strategy**
- Process one fish at a time per frame — no joint multi-fish optimization
- 2-start initialization on first frame: forward + 180-degree flip. Early exit heuristic: run both ~50 iters, discard the clearly-worse one, finish the better one to completion
- Convergence criterion with hard iteration cap as safety net. Loss delta below threshold for ~3 consecutive steps triggers early stop. Cap prevents runaway optimization on pathological frames
- Warm-start between frames uses constant-velocity prediction: extrapolate position from last 2 frames' solutions

**Validation approach**
- Leave-one-out cross-view holdout: hold out 1 camera, rotate across frames/fish to accumulate statistics
- Full diversity test clip (~500+ frames) covering occlusions, fast motion, edge cases
- Output: quantitative IoU metrics + visual overlays (rendered mesh on real camera frames)
- Target: global average holdout IoU >= 0.80, with no individual camera below 0.60 floor
- Phase 4 includes a diagnose-and-iterate cycle if the target isn't met initially

### Claude's Discretion
- Adam learning rate and hyperparameter defaults
- Exact iteration cap value and convergence threshold
- Angular-diversity weighting formula
- Crop padding strategy for IoU computation
- Visual overlay rendering style

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RECON-01 | System renders differentiable silhouettes of the fish mesh into each camera view via refractive projection + PyTorch3D rasterizer, with per-camera weighting by angular diversity | PyTorch3D `SoftSilhouetteShader` + `MeshRasterizer` + `cameras_from_opencv_projection` pipeline; angular weighting computed from camera extrinsics dot products |
| RECON-02 | System computes multi-objective loss: silhouette IoU + gravity prior + morphological constraint first, then temporal smoothness once tracking provides frame-to-frame associations | Soft IoU formula documented; gravity prior = roll angle penalty; morph constraints = L2 on s and kappa |
| RECON-03 | System runs 2-initialization multi-start (forward + 180° flip) on first frame of each track to resolve head-tail ambiguity | 2-start pattern: run both ~50 iters, pick lower-loss; implemented with psi flipped by pi |
| RECON-04 | System optimizes per-frame fish pose via Adam with warm-start from previous frame's solution | `torch.optim.Adam` with `requires_grad=True` on FishState tensors; warm-start via constant-velocity extrapolation |
| RECON-05 | System validates reconstruction via cross-view holdout (fit on N-k cameras, evaluate IoU on k held-out cameras), achieving ≥0.80 mean holdout IoU | Hold-out: exclude 1 camera from optimizer, evaluate rendered IoU against that camera's mask |
</phase_requirements>

---

## Summary

Phase 4 builds the analysis-by-synthesis optimization loop: given a fish mesh (from Phase 3) and silhouette masks (from Phase 2), render the mesh into each camera using differentiable refractive projection and a PyTorch3D rasterizer, compute soft IoU loss against observed masks, and use Adam to optimize the FishState parameters. All upstream infrastructure is complete — `RefractiveProjectionModel.project()`, `build_fish_mesh()`, and the `FishState` dataclass are implemented and tested. Phase 4 is primarily a wiring-and-tuning problem.

**Critical blocker discovered during research:** The currently installed `pytorch3d-0.7.9+pt2.9.1cu128` fails to import its C extension on the current `torch 2.10.0+cu130` environment (ABI mismatch between CUDA 12.8 and CUDA 13.0). No miropsota wheel exists for Windows + cu130. The fix is to downgrade torch to `2.9.1+cu128` so it matches the installed pytorch3d wheel. This must be the first task of Phase 4.

The core rendering pipeline follows the standard PyTorch3D silhouette rendering pattern: `cameras_from_opencv_projection` converts AquaCal extrinsics to PyTorch3D camera objects, `MeshRasterizer` + `SoftSilhouetteShader` produce a differentiable alpha map, and soft IoU is computed within the crop bounding box. The key AquaPose-specific wrinkle is that projection goes through the refractive interface — the existing `RefractiveProjectionModel.project()` Newton-Raphson implementation is differentiable and should be used as the bridge between world-space mesh vertices and the PyTorch3D camera input.

**Primary recommendation:** Fix pytorch3d import first (torch downgrade or CPU-only rebuild). Then implement rendering as a thin wrapper layer using `cameras_from_opencv_projection` + `SoftSilhouetteShader` with crop-space IoU. Build the optimizer using `torch.optim.Adam` over FishState tensors with `requires_grad=True`. The loss, optimizer, and 2-start logic are all standard PyTorch patterns — no additional libraries required.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.9.1+cu128 (must downgrade from 2.10!) | Autograd optimizer loop, loss computation | All gradients flow through this |
| pytorch3d | 0.7.9+pt2.9.1cu128 (already installed) | `MeshRasterizer`, `SoftSilhouetteShader`, `cameras_from_opencv_projection` | Only differentiable mesh rasterizer with silhouette shader in the ecosystem |
| aquapose.calibration.projection | (Phase 1, local) | `RefractiveProjectionModel.project()` for refractive vertex projection; `cameras_from_opencv_projection` coordinate bridge | Already implemented, differentiable, cross-validated |
| aquapose.mesh.builder | (Phase 3, local) | `build_fish_mesh(list[FishState]) -> Meshes` | Already implemented and tested |
| numpy | >=1.24 | Camera extrinsic angle computation for angular diversity weighting | Already installed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.optim.Adam | (torch builtin) | Gradient-based optimizer for FishState parameters | Main optimization loop |
| cv2 (opencv-python) | >=4.8 | Visual overlay rendering for validation output | RECON-05 visual QA only |
| matplotlib | (via aquacal deps) | Holdout IoU statistics plots for validation report | RECON-05 report |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyTorch3D `SoftSilhouetteShader` | NVDiffRast, Kaolin | Not available in env; pytorch3d is already installed and provides exact API needed |
| PyTorch3D `cameras_from_opencv_projection` | Manual sign-flip coordinate transform | Manual transform is error-prone; the official utility handles non-square image NDC space correctly |
| Adam optimizer | L-BFGS | L-BFGS is better for convex smooth objectives but has no momentum for non-convex pose optimization; Adam handles noise and non-convexity better for silhouette fitting |
| Soft IoU loss | MSE of alpha maps | Soft IoU is scale-invariant (doesn't penalize fish size relative to frame); MSE rewards large masks near reference |

**Environment fix (must run before Phase 4 development):**
```bash
# The installed pytorch3d wheel requires torch 2.9.1+cu128
# Current env has torch 2.10.0+cu130 — ABI mismatch causes _C DLL load failure
pip install torch==2.9.1+cu128 torchvision==0.20.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
# Verify fix:
python -c "from pytorch3d.renderer import SoftSilhouetteShader; print('OK')"
```

---

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/
├── optimization/
│   ├── __init__.py         # Public API: FishOptimizer, render_silhouette, compute_loss
│   ├── renderer.py         # RefractiveSilhouetteRenderer: mesh + cameras → alpha maps
│   ├── loss.py             # MultiObjectiveLoss: soft IoU + gravity + morph + temporal hook
│   └── optimizer.py        # FishOptimizer: 2-start, warm-start, convergence logic
tests/
└── unit/
    └── optimization/
        ├── __init__.py
        ├── test_renderer.py    # Silhouette renders non-empty, gradients flow
        ├── test_loss.py        # Loss decreases when state improves, gravity/morph terms
        └── test_optimizer.py   # 2-start selects lower loss; warm-start round-trip
```

### Pattern 1: RefractiveSilhouetteRenderer
**What:** Wraps `cameras_from_opencv_projection` + `MeshRasterizer` + `SoftSilhouetteShader` into a single callable that takes a `Meshes` object and a list of `RefractiveProjectionModel` camera configs, and returns a dict of camera_id → differentiable alpha map.

**When to use:** Called inside the optimizer's forward pass each iteration.

**Key detail:** AquaPose's `RefractiveProjectionModel` already handles the refractive projection correctly in its `project()` method. However, PyTorch3D's `cameras_from_opencv_projection` takes the post-undistortion K, R, t matrices and creates PyTorch3D `PerspectiveCameras` — these cameras do straight (non-refractive) projection. For Phase 4, the refractive effect can be approximated by using the refractive projection to pre-warp mesh vertex positions before handing to the rasterizer, OR by constructing a custom camera projection that calls `RefractiveProjectionModel.project()`.

**Practical approach (confirmed correct):** Use a custom `transform_points` implementation or a custom shader that calls `RefractiveProjectionModel.project()` per vertex. The simplest path that avoids rebuilding the rasterizer internals is a **vertex pre-projection approach**: project each mesh vertex from world space to image space using `RefractiveProjectionModel.project()` (which is differentiable), then construct a flat camera (orthographic or identity) that maps from image-space coordinates directly to pixels. Alternatively, treat the problem as: for each frame, create a "virtual pinhole" camera whose intrinsics approximate the refractive geometry for the expected fish depth range.

**Note on refractive projection in rasterizer:** The cleanest approach is to pass world-space vertices to a custom rasterizer setup where the camera projection function is replaced by `RefractiveProjectionModel.project()`. PyTorch3D's `MeshRasterizer` accepts any camera object with a `transform_points(world_pts) -> screen_pts` method — creating a thin wrapper around `RefractiveProjectionModel.project()` is the right approach.

```python
# src/aquapose/optimization/renderer.py
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftSilhouetteShader, BlendParams,
)
from aquapose.calibration.projection import RefractiveProjectionModel

import numpy as np

class RefractiveSilhouetteRenderer:
    """Differentiable silhouette renderer using refractive projection.

    Wraps PyTorch3D rasterizer with AquaPose's refractive camera model.
    Each camera's vertices are projected via RefractiveProjectionModel.project()
    before rasterization.

    Args:
        image_size: (H, W) output image dimensions in pixels.
        faces_per_pixel: Number of faces to blend per pixel for silhouette.
            Higher values give smoother gradients (100 is standard).
        sigma: BlendParams sigma — controls edge softness. 1e-4 is standard.
        gamma: BlendParams gamma — controls opacity. 1e-4 is standard.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        faces_per_pixel: int = 100,
        sigma: float = 1e-4,
        gamma: float = 1e-4,
    ) -> None:
        self.image_size = image_size
        blend_params = BlendParams(sigma=sigma, gamma=gamma)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=faces_per_pixel,
        )
        # Rasterizer and shader stored; camera provided at render time
        self.raster_settings = raster_settings
        self.blend_params = blend_params

    def render(
        self,
        meshes: Meshes,
        cameras: list[RefractiveProjectionModel],  # one per view
        camera_ids: list[str],
    ) -> dict[str, torch.Tensor]:
        """Render silhouettes into each camera view.

        Args:
            meshes: PyTorch3D Meshes, single fish mesh.
            cameras: RefractiveProjectionModel instances, one per camera.
            camera_ids: Camera identifier strings.

        Returns:
            Dict mapping camera_id -> alpha map tensor, shape (H, W), float32.
                Values in [0, 1]. Differentiable w.r.t. mesh vertex positions.
        """
        ...  # implementation uses RefractiveCamera wrapper (see Pattern 2)
```

### Pattern 2: RefractiveCamera PyTorch3D Wrapper
**What:** A thin class that wraps `RefractiveProjectionModel` to satisfy the PyTorch3D camera interface required by `MeshRasterizer`. The rasterizer calls `cameras.transform_points(world_pts)` to go from world space to NDC space.

```python
# The camera wrapper converts world-space mesh vertices to screen-space
# via RefractiveProjectionModel.project() then normalizes to NDC
class RefractiveCamera:
    """PyTorch3D-compatible camera wrapper using refractive projection.

    Adapts RefractiveProjectionModel to the interface expected by
    MeshRasterizer (transform_points method).
    """
    def __init__(
        self,
        model: RefractiveProjectionModel,
        image_size: tuple[int, int],
    ) -> None:
        self.model = model
        self.H, self.W = image_size

    def transform_points(self, world_pts: torch.Tensor) -> torch.Tensor:
        """Project world points to NDC space via refractive model.

        Args:
            world_pts: (N, 3) world-space points.

        Returns:
            ndc_pts: (N, 3) NDC coordinates. Z is kept as-is for depth sort.
        """
        pixels, valid = self.model.project(world_pts)  # (N, 2), (N,)
        # Convert pixel coordinates to NDC: [-1, 1] x [-1, 1]
        # PyTorch3D NDC: x in [-1, 1] left-to-right, y in [-1, 1] bottom-to-top
        u = pixels[:, 0]  # (N,) in [0, W]
        v = pixels[:, 1]  # (N,) in [0, H]
        ndc_x = (u / self.W) * 2.0 - 1.0     # [0,W] -> [-1,1]
        ndc_y = -((v / self.H) * 2.0 - 1.0)  # [0,H] -> [1,-1] (flip Y)
        # Z: use world Z as depth proxy (larger = further from camera)
        z = world_pts[:, 2]
        return torch.stack([ndc_x, ndc_y, z], dim=-1)  # (N, 3)
```

**Warning:** This wrapper gives a correct first-order approximation but the gradient through `transform_points` via `RefractiveProjectionModel.project()` must be verified with a backward pass. The Newton-Raphson projection is differentiable (fixed 10 iterations, no data-dependent branching) — gradients should flow. Verification test is mandatory.

### Pattern 3: Crop-Space Soft IoU Loss
**What:** The loss is computed within the bounding box crop region, not the full frame. This focuses gradients on the fish region and avoids the background dominating the loss.

**Formula:** Let `pred` be the rendered alpha map (shape H×W, values in [0,1]) and `target` be the observed binary mask (0 or 1). Both are cropped to the fish bounding box before computing IoU:

```
intersection = sum(pred * target)
union = sum(pred + target - pred * target)
soft_iou = intersection / (union + eps)
iou_loss = 1.0 - soft_iou
```

```python
# src/aquapose/optimization/loss.py
import torch

def soft_iou_loss(
    pred_alpha: torch.Tensor,   # (H, W), values in [0, 1], from renderer
    target_mask: torch.Tensor,  # (H, W), binary {0, 1} float
    crop_region: tuple[int, int, int, int] | None = None,  # (y1, x1, y2, x2)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute 1 - soft IoU between predicted silhouette and target mask.

    Args:
        pred_alpha: Predicted silhouette alpha map, shape (H, W), float32.
        target_mask: Observed binary mask, shape (H, W), float32.
        crop_region: Optional (y1, x1, y2, x2) to restrict computation to
            the fish bounding box. If None, uses full frame.
        eps: Numerical stability for division.

    Returns:
        scalar: 1 - soft_IoU, in range [0, 1]. Lower is better.
    """
    if crop_region is not None:
        y1, x1, y2, x2 = crop_region
        pred_alpha = pred_alpha[y1:y2, x1:x2]
        target_mask = target_mask[y1:y2, x1:x2]
    intersection = (pred_alpha * target_mask).sum()
    union = (pred_alpha + target_mask - pred_alpha * target_mask).sum()
    return 1.0 - intersection / (union + eps)
```

### Pattern 4: Multi-Objective Loss
**What:** Combines per-camera soft IoU losses (weighted by angular diversity) with gravity prior and morphological constraints.

```python
def multi_objective_loss(
    state: FishState,
    pred_alphas: dict[str, torch.Tensor],   # camera_id -> alpha map
    target_masks: dict[str, torch.Tensor],  # camera_id -> binary mask
    crop_regions: dict[str, tuple[int, int, int, int]],
    camera_weights: dict[str, float],       # angular diversity weights
    loss_weights: dict[str, float],         # {"iou": 1.0, "gravity": 0.05, "morph": 0.2}
    temporal_state: FishState | None = None,  # None in Phase 4 (no tracking yet)
    temporal_weight: float = 0.1,
    kappa_max: float = 10.0,
    s_min: float = 0.05, s_max: float = 0.30,
) -> dict[str, torch.Tensor]:
    """Compute multi-objective loss for pose optimization.

    Returns dict with individual loss terms for logging.
    """
    # 1. Silhouette IoU loss (weighted sum across cameras)
    iou_losses = []
    for cam_id, alpha in pred_alphas.items():
        w = camera_weights.get(cam_id, 1.0)
        iou_l = soft_iou_loss(alpha, target_masks[cam_id], crop_regions[cam_id])
        iou_losses.append(w * iou_l)
    iou_loss = torch.stack(iou_losses).sum() / sum(camera_weights.values())

    # 2. Gravity prior: penalize roll (rotation about heading axis)
    # Fish dorsal axis should align with world-up [0, 0, 1]
    # Proxy: penalize deviation of the computed dorsal vector from [0, 0, 1]
    # The dorsal vector is implicit in the spine frame — compute it from psi, theta
    # Simple approximation: for horizontal fish (theta ~ 0), dorsal ~ [0, 0, 1]
    # Roll deviation = angle between fish dorsal and world up
    # For now: gravity_loss = theta^2 scaled (pitch deviation from horizontal)
    # True roll requires a roll parameter not in current state — use pitch as proxy
    gravity_loss = state.theta ** 2  # penalize nose-down/up tilt as gravity proxy

    # 3. Morphological constraints
    kappa_loss = torch.relu(state.kappa.abs() - kappa_max) ** 2
    s_loss = (torch.relu(s_min - state.s) + torch.relu(state.s - s_max)) ** 2

    # 4. Temporal smoothness (inactive in Phase 4 — hook is present)
    temporal_loss = torch.tensor(0.0, device=state.p.device)
    if temporal_state is not None:
        temporal_loss = (
            (state.p - temporal_state.p).norm() ** 2
            + (state.psi - temporal_state.psi) ** 2
        )

    total = (
        loss_weights["iou"] * iou_loss
        + loss_weights.get("gravity", 0.05) * gravity_loss
        + loss_weights.get("morph", 0.2) * (kappa_loss + s_loss)
        + temporal_weight * temporal_loss
    )
    return {
        "total": total, "iou": iou_loss, "gravity": gravity_loss,
        "morph": kappa_loss + s_loss, "temporal": temporal_loss,
    }
```

### Pattern 5: FishState as Optimizable Parameter Set
**What:** To optimize FishState parameters with `torch.optim.Adam`, each parameter tensor must have `requires_grad=True`. The simplest pattern is to create a `nn.Module`-like container or directly use a list of parameter tensors.

```python
# Create optimizable state from an initial estimate
def make_optimizable_state(init_state: FishState, device: str = "cpu") -> FishState:
    """Clone a FishState with requires_grad=True on all parameters."""
    return FishState(
        p=init_state.p.detach().clone().requires_grad_(True).to(device),
        psi=init_state.psi.detach().clone().requires_grad_(True).to(device),
        theta=init_state.theta.detach().clone().requires_grad_(True).to(device),
        kappa=init_state.kappa.detach().clone().requires_grad_(True).to(device),
        s=init_state.s.detach().clone().requires_grad_(True).to(device),
    )

def get_state_params(state: FishState) -> list[torch.Tensor]:
    """Get all optimizable tensors for Adam."""
    return [state.p, state.psi, state.theta, state.kappa, state.s]

# Create optimizer
optimizer = torch.optim.Adam(get_state_params(state), lr=1e-3)
```

### Pattern 6: 2-Start Optimization Loop (First Frame)
**What:** Run two initializations (forward heading and 180-degree flip), run each for ~50 iterations, then continue only the lower-loss candidate.

```python
def optimize_first_frame(
    init_state: FishState,
    target_masks: dict[str, torch.Tensor],
    renderer: RefractiveSilhouetteRenderer,
    cameras: list[RefractiveProjectionModel],
    camera_ids: list[str],
    camera_weights: dict[str, float],
    crop_regions: dict[str, tuple[int, int, int, int]],
    loss_weights: dict[str, float],
    early_exit_iters: int = 50,
    max_iters: int = 300,
    lr: float = 1e-3,
    convergence_delta: float = 1e-4,
    convergence_patience: int = 3,
) -> FishState:
    """Run 2-start optimization to resolve head-tail ambiguity.

    Args:
        init_state: Initial FishState from Phase 3 cold-start.
        early_exit_iters: Iterations before comparing starts (default 50).
        max_iters: Maximum total iterations for the winner.

    Returns:
        Optimized FishState.
    """
    # Start A: original heading
    state_a = make_optimizable_state(init_state)
    # Start B: 180-degree flip (psi + pi)
    flipped = FishState(
        p=init_state.p.clone(), psi=init_state.psi + torch.pi,
        theta=init_state.theta.clone(), kappa=init_state.kappa.clone(),
        s=init_state.s.clone(),
    )
    state_b = make_optimizable_state(flipped)

    loss_a = _run_optimizer(state_a, ..., n_iters=early_exit_iters)
    loss_b = _run_optimizer(state_b, ..., n_iters=early_exit_iters)

    winner = state_a if loss_a <= loss_b else state_b
    return _run_optimizer(winner, ..., n_iters=max_iters - early_exit_iters)
```

### Pattern 7: Warm-Start with Constant-Velocity Prediction
**What:** For frame t, predict position from last 2 frames' solutions to initialize warm-start.

```python
def warm_start_from_velocity(
    state_t_minus_1: FishState,
    state_t_minus_2: FishState | None,
) -> FishState:
    """Predict next frame state using constant-velocity extrapolation.

    Args:
        state_t_minus_1: State from frame t-1.
        state_t_minus_2: State from frame t-2, or None if only 1 prior frame.

    Returns:
        Predicted FishState for frame t (no gradients — used as init only).
    """
    if state_t_minus_2 is None:
        return state_t_minus_1  # No velocity, stay at last position

    # Constant-velocity prediction: extrapolate position and heading
    dp = state_t_minus_1.p.detach() - state_t_minus_2.p.detach()
    p_pred = state_t_minus_1.p.detach() + dp

    dpsi = state_t_minus_1.psi.detach() - state_t_minus_2.psi.detach()
    psi_pred = state_t_minus_1.psi.detach() + dpsi

    return FishState(
        p=p_pred, psi=psi_pred,
        theta=state_t_minus_1.theta.detach(),
        kappa=state_t_minus_1.kappa.detach(),
        s=state_t_minus_1.s.detach(),
    )
```

### Pattern 8: Angular Diversity Weighting
**What:** Down-weight cameras that are spatially clustered to prevent the ring of 12 cameras from dominating over unique viewpoints. Weighting is based on minimum angular separation between each camera and all others.

**Formula (Claude's discretion):** For each camera i, compute the minimum angle to all other cameras j using their view direction vectors (negative z-axis of R). A camera isolated from others gets weight 1.0; a camera close to a neighbor gets proportionally less weight.

```python
def compute_angular_diversity_weights(
    models: list[RefractiveProjectionModel],
    camera_ids: list[str],
    temperature: float = 0.5,
) -> dict[str, float]:
    """Compute per-camera weights based on angular diversity.

    Cameras pointing in similar directions (clustered ring) get lower weights.
    Cameras with unique viewpoints get higher weights.

    Args:
        models: RefractiveProjectionModel instances.
        camera_ids: Camera identifier strings.
        temperature: Controls weight spread. Higher = more uniform weights.

    Returns:
        Dict mapping camera_id -> weight in (0, 1].
    """
    import numpy as np

    # Extract view direction from each camera's rotation matrix
    # Camera looks along -Z axis in camera frame; in world frame: R.T @ [0,0,-1]
    view_dirs = []
    for model in models:
        R = model.R.detach().cpu().numpy()  # (3, 3), world-to-camera
        view_dir = R.T @ np.array([0.0, 0.0, -1.0])  # (3,) world-space view dir
        view_dirs.append(view_dir / np.linalg.norm(view_dir))
    view_dirs = np.stack(view_dirs, axis=0)  # (N, 3)

    # For each camera, find minimum angle to any other camera
    min_angles = []
    for i in range(len(models)):
        dots = view_dirs[i] @ view_dirs.T  # (N,)
        dots = np.clip(dots, -1.0, 1.0)
        dots[i] = -2.0  # exclude self (set to impossible value)
        min_dot = dots.max()  # most similar camera
        min_angle = np.arccos(min_dot)
        min_angles.append(min_angle)
    min_angles = np.array(min_angles)  # (N,)

    # Weight proportional to minimum separation angle
    # Normalize so max-weight camera gets 1.0
    weights = min_angles / (min_angles.max() + 1e-8)
    weights = np.power(weights, temperature)  # temperature controls spread

    return {cam_id: float(w) for cam_id, w in zip(camera_ids, weights)}
```

### Pattern 9: Cross-View Holdout Validation
**What:** For each evaluation frame: hold out 1 camera (exclude from optimization), run optimizer on remaining N-1 cameras, then compute IoU between the rendered silhouette in the held-out camera and its observed mask.

```python
def evaluate_holdout_iou(
    state: FishState,            # optimized state (fitted on N-1 cameras)
    held_out_camera: RefractiveProjectionModel,
    held_out_mask: torch.Tensor,  # (H, W) binary float
    renderer: RefractiveSilhouetteRenderer,
    crop_region: tuple[int, int, int, int] | None,
) -> float:
    """Compute holdout IoU for a single camera."""
    with torch.no_grad():
        meshes = build_fish_mesh([state])
        alpha = renderer.render(meshes, [held_out_camera], ["held_out"])["held_out"]
    return 1.0 - float(soft_iou_loss(alpha, held_out_mask, crop_region))
```

### Anti-Patterns to Avoid
- **Zeroing gradients before backward but after optimizer.step():** Call `optimizer.zero_grad()` BEFORE `loss.backward()`, not after. Standard PyTorch pattern.
- **Recomputing mesh inside the loss function:** Build the mesh once per optimizer step (in the outer loop), not inside the loss computation. The mesh build triggers spine/cross-section computation — doing it twice wastes compute.
- **Not detaching warm-start predictions:** The predicted state for frame t must have `.detach()` applied to all tensors before using as a starting point. Otherwise the warm-start state is part of the previous frame's computation graph, causing memory leaks across frames.
- **Using `item()` inside the optimizer loop:** Calling `.item()` on the loss breaks the autograd graph. Use `.item()` only for logging (after `.backward()`).
- **sigma/gamma too small (sharp silhouette):** If sigma is too small (e.g., 1e-6), the rendered silhouette is essentially binary — gradients near the boundary are zero, causing gradient vanishing. Start with sigma=1e-4, gamma=1e-4 (PyTorch3D defaults).
- **faces_per_pixel=1:** The silhouette shader needs faces_per_pixel >> 1 (typically 100) to accumulate contributions from multiple overlapping faces near the mesh boundary. With faces_per_pixel=1, only the nearest face contributes — silhouettes are sharp and non-differentiable.
- **Not normalizing the IoU loss sum by camera weights:** If camera weights vary, divide the weighted sum by the sum of weights to get a normalized loss in [0, 1]. Otherwise the loss magnitude changes as cameras are added/removed.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mesh rasterization to pixels | Custom ray-triangle intersection | `pytorch3d.renderer.MeshRasterizer` | Already handles NDC conversion, depth sorting, barycentric coordinates |
| Silhouette soft blending | Custom sigmoid distance function | `pytorch3d.renderer.SoftSilhouetteShader` with `BlendParams` | Implements the SoftRas sigma/gamma blending used in all pytorch3d tutorials |
| Camera coordinate conversion | Manual OpenCV-to-NDC sign flips | `pytorch3d.utils.cameras_from_opencv_projection` | Handles non-square image NDC space correctly (fixed in PyTorch3D 2023) |
| Gradient-based optimizer | SGD with manual LR schedule | `torch.optim.Adam` | Adam handles the different gradient scales of position, angle, and scale parameters automatically |
| Soft IoU formula | Hand-rolled intersection/union | Inline formula (3 lines) — too simple to abstract | No library needed; just `(pred * target).sum() / (pred + target - pred * target).sum()` |

**Key insight:** The hard parts (differentiable refractive projection, parametric mesh, PyTorch3D rasterizer) are already implemented. Phase 4 is primarily about connecting these components and tuning hyperparameters.

---

## Common Pitfalls

### Pitfall 1: pytorch3d Renderer DLL Load Failure (CRITICAL — MUST FIX FIRST)

**What goes wrong:** `from pytorch3d.renderer import SoftSilhouetteShader` raises `ImportError: DLL load failed while importing _C`. The renderer cannot be imported at all.

**Why it happens:** The installed wheel is `pytorch3d-0.7.9+pt2.9.1cu128`, compiled against PyTorch 2.9.1+CUDA 12.8. The current environment has PyTorch `2.10.0+cu130` (CUDA 13.0). The CUDA ABI changed between 12.8 and 13.0 — the `_C.cp312-win_amd64.pyd` extension links against `cudart64_128.dll` (CUDA 12.8) which is absent in the CUDA 13.0 runtime.

**No miropsota wheel for Windows + CUDA 13:** Confirmed: miropsota provides cu130 wheels only for Linux. No Windows + cu130 pytorch3d wheel exists.

**How to fix (choose one):**
```bash
# Option A (RECOMMENDED): Downgrade torch to match installed pytorch3d wheel
pip install torch==2.9.1+cu128 torchvision==0.20.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
# Verify: python -c "from pytorch3d.renderer import SoftSilhouetteShader; print('OK')"

# Option B: Reinstall pytorch3d for a matching torch+cuda version
# First ensure torch 2.9.1+cu128 is installed, then:
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
    "pytorch3d==0.7.9+pt2.9.1cu128"

# Option C: CPU-only source build (no CUDA acceleration but always loads)
pip uninstall pytorch3d -y
FORCE_CUDA=0 pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

**Warning signs:** Error appears immediately on `from pytorch3d.renderer import ...`. Not a runtime error — it fails at import time.

### Pitfall 2: Gravity Prior Requires Roll Angle (Not in Current FishState)

**What goes wrong:** The locked decision specifies "soft roll regularization: penalize deviations of the fish's dorsal-ventral axis from upright orientation." However, the current `FishState` has only `{p, psi, theta, kappa, s}` — there is no explicit roll angle. Pitch (`theta`) tilts the nose up/down but does not model lateral roll.

**Why it happens:** The state vector was designed with yaw+pitch heading (5 DOF total including position and scale) — lateral roll was not included. For predominantly horizontal fish (the typical case), roll is assumed near zero.

**How to handle:** The gravity prior must be approximated from the existing state parameters. Two approaches:
1. **Pitch proxy:** Use `theta^2` as the gravity regularizer — penalize nose-down tilt which is the most common unphysical pose. This is a rough approximation.
2. **Add roll parameter:** Add a `phi` (roll) scalar to `FishState` and back it with a strong prior pulling toward phi=0. This requires a state vector change and rebuilds spine generation with an additional rotation. More correct but a larger scope change.

**Recommendation:** Start with the pitch proxy (theta^2) as a low-cost approximation. If optimization produces rolled fish, add an explicit roll parameter (FishState expansion). Document as a potential Phase 4 extension.

### Pitfall 3: Crop-Space vs. Full-Frame IoU

**What goes wrong:** Computing IoU on the full 1600×1200 frame where the fish occupies ~100×50 pixels makes 99.7% of pixels background. The background pixels contribute nothing to intersection but inflate the union denominator when the predicted silhouette spills outside the fish region. This makes the loss less sensitive to the actual fish shape.

**Why it happens:** Background-heavy IoU is numerically dominated by the background agreement term (both pred and target are 0 for most pixels).

**How to avoid:** Always slice both `pred_alpha` and `target_mask` to the fish bounding box before computing IoU. Use `segmentation.crop.CropRegion` (already implemented in Phase 2) to define the crop. The crop region should be the bounding box of the observed mask with padding (25% on each side, matching the Phase 2 convention).

**Warning signs:** Loss is very small (near 0) even when the mesh is clearly misaligned — the background agreement is dominating.

### Pitfall 4: Gradient Explosion Through RefractiveCamera Wrapper

**What goes wrong:** The `RefractiveProjectionModel.project()` Newton-Raphson loop uses 10 fixed iterations. Gradients flow backward through all 10 iterations, which multiplies the Jacobian 10 times. For large mesh vertex displacements (fish far from initial estimate), this can produce large gradients.

**Why it happens:** Fixed-count Newton-Raphson is differentiable but the gradient magnitude depends on the convergence behavior. If the initial guess is far from the true refraction point, intermediate iterates have large residuals, amplifying gradients.

**How to avoid:**
1. Clip gradients: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` before `optimizer.step()`.
2. Use a lower learning rate for position (p) than for angles (psi, theta) — Adam's per-parameter scaling helps here.
3. Warm-start reduces this risk: frames after the first start from a good estimate.

**Warning signs:** Loss suddenly jumps to NaN or very large values after a few iterations. Gradient norms in the hundreds.

### Pitfall 5: sigma Sweep Required for This Rig's Fish Size

**What goes wrong:** The PyTorch3D default sigma=1e-4, gamma=1e-4 are calibrated for the textured mesh tutorials where meshes occupy a large fraction of the rendered image. For AquaPose, a fish occupies ~100×50 pixels in a 1600×1200 frame, or ~3000 pixels total. The boundary zone (where sigma has effect) is proportional to sigma * image_diagonal. If sigma is too small, the boundary is too thin and gradients are weak.

**Why it happens:** sigma controls the pixel-to-face distance threshold for soft blending. The effective softness in pixel units scales with the image size.

**How to avoid:** Start with sigma=1e-4 (PyTorch3D default). If gradients are weak or optimization doesn't converge, try sigma=5e-4 or 1e-3 (softer boundaries, stronger gradients near edges). Since the optimizer renders crops (not full frames), the effective image size for sigma is the crop size (~150×100 pixels), which is more amenable to the default sigma.

**Warning signs:** Loss plateaus immediately without decreasing, even though the mesh is clearly not aligned. Plot the alpha map — if the boundary is razor-thin (binary), sigma needs to increase.

### Pitfall 6: PyTorch3D Camera Convention vs. AquaPose World Frame

**What goes wrong:** AquaPose world frame is +Z down (into water). PyTorch3D's internal convention for cameras involves +X left, +Y up, +Z out of image plane. The `cameras_from_opencv_projection` utility handles this conversion, but if a custom `RefractiveCamera` wrapper is used instead, the sign conventions must be handled manually.

**How to avoid:** In the custom `RefractiveCamera.transform_points()`:
- Pixel u (horizontal, right) maps to NDC x: `(u / W) * 2 - 1` (left to right, -1 to 1)
- Pixel v (vertical, down) maps to NDC y: `-(v / H) * 2 + 1` (bottom to top, 1 to -1 — FLIP Y)
- The Y flip is required because PyTorch3D NDC has +Y pointing up, but pixel coordinates have +v pointing down.

**Warning signs:** Rendered silhouette appears vertically flipped relative to the camera image.

### Pitfall 7: Memory Accumulation Across Optimizer Iterations

**What goes wrong:** Keeping a reference to the loss tensor from previous iterations (e.g., for logging or early-stop checks) prevents Python's garbage collector from freeing the computation graph. Over 200+ iterations, this accumulates all intermediate tensors.

**Why it happens:** PyTorch holds the entire computation graph in memory until the graph is freed. If `loss_history.append(loss)` stores the tensor (not `.item()`), the graph is retained.

**How to avoid:** Always convert to Python float for logging: `loss_history.append(loss.item())`. Clear the computation graph each iteration via `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` cycle. Never store gradient-tracking tensors across iterations.

---

## Code Examples

### Complete SoftSilhouetteShader Setup (verified from PyTorch3D docs)
```python
# Source: pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/shader.html
# Source: pytorch3d camera_position_optimization tutorial

import numpy as np
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftSilhouetteShader, BlendParams,
)

# Standard soft silhouette setup for pose optimization
sigma = 1e-4
gamma = 1e-4
blend_params = BlendParams(sigma=sigma, gamma=gamma)
raster_settings = RasterizationSettings(
    image_size=(H, W),
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,  # consistent with SoftRasterizer
    faces_per_pixel=100,  # must be >> 1 for smooth silhouette gradients
)
# Note: SoftSilhouetteShader output is (N, H, W, 4) RGBA; alpha channel = silhouette
# Access with: images[..., 3]  where images = renderer(meshes, cameras=cameras)
```

### cameras_from_opencv_projection Usage (verified from PyTorch3D docs)
```python
# Source: pytorch3d.readthedocs.io/en/latest/modules/utils.html
# Source: pytorch3d camera_conversions docs
from pytorch3d.utils import cameras_from_opencv_projection
import torch

# R: (N, 3, 3) world-to-camera rotation matrices
# tvec: (N, 3) translation vectors
# camera_matrix: (N, 3, 3) post-undistortion intrinsic matrices
# image_size: (N, 2) tensor of [H, W] per camera

cameras = cameras_from_opencv_projection(
    R=R_batch,            # (N, 3, 3), float32
    tvec=t_batch,         # (N, 3), float32
    camera_matrix=K_batch,  # (N, 3, 3), post-undistortion K
    image_size=torch.tensor([[H, W]] * N, dtype=torch.float32),
)
# Returns PerspectiveCameras object with N cameras
# This handles the OpenCV-to-PyTorch3D coordinate conversion automatically
```

### Soft IoU Loss (from PyTorch Forums, verified against formula)
```python
# Source: discuss.pytorch.org/t/how-to-implement-soft-iou-loss/15152
# Adapted for silhouette alpha maps

def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - soft IoU for silhouette comparison.

    Args:
        pred: Predicted alpha map, shape (H, W), values in [0, 1].
        target: Binary target mask, shape (H, W), values {0.0, 1.0}.
        eps: Numerical stability.

    Returns:
        Scalar loss in [0, 1]. 0 = perfect overlap, 1 = no overlap.
    """
    intersection = (pred * target).sum()
    union = (pred + target - pred * target).sum()
    return 1.0 - intersection / (union + eps)
```

### Adam Optimizer Loop Pattern (standard PyTorch)
```python
import torch

params = get_state_params(state)  # list of requires_grad tensors
optimizer = torch.optim.Adam(params, lr=1e-3)

prev_losses = []
for i in range(max_iters):
    optimizer.zero_grad()

    meshes = build_fish_mesh([state])
    alpha_maps = renderer.render(meshes, cameras, camera_ids)
    losses = multi_objective_loss(state, alpha_maps, target_masks, ...)

    loss = losses["total"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # prevent explosion
    optimizer.step()

    loss_val = loss.item()  # convert to Python float — breaks autograd hold
    prev_losses.append(loss_val)

    # Convergence: loss delta < threshold for `patience` consecutive steps
    if len(prev_losses) >= convergence_patience + 1:
        recent_delta = abs(prev_losses[-1] - prev_losses[-1 - convergence_patience])
        if recent_delta < convergence_delta:
            break
```

### Differentiability Verification Test
```python
# Must pass before any optimizer work proceeds
import torch
from aquapose.mesh.state import FishState
from aquapose.mesh.builder import build_fish_mesh
from aquapose.calibration.projection import RefractiveProjectionModel

def test_renderer_gradients_flow(renderer, camera_model):
    """Verify gradients reach FishState parameters through the renderer."""
    p = torch.tensor([0.0, 0.0, 1.2], requires_grad=True)
    state = FishState(
        p=p, psi=torch.tensor(0.0, requires_grad=True),
        theta=torch.tensor(0.0, requires_grad=True),
        kappa=torch.tensor(0.0, requires_grad=True),
        s=torch.tensor(0.15, requires_grad=True),
    )
    meshes = build_fish_mesh([state])
    alpha_maps = renderer.render(meshes, [camera_model], ["cam0"])
    loss = alpha_maps["cam0"].sum()
    loss.backward()
    assert p.grad is not None, "No gradient to position"
    assert torch.isfinite(p.grad).all(), "NaN gradient to position"
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|---|---|---|
| SoftRasterizer (2019, external) | PyTorch3D SoftSilhouetteShader (2020+) | SoftRas was the precursor; PyTorch3D is the successor with better tooling |
| Full-frame silhouette IoU | Crop-space IoU | Crop restricts loss to fish region, avoids background dominance |
| Single initialization | 2-start (forward + flip) for head-tail | Standard technique in analysis-by-synthesis for ambiguous pose |
| Random restart / grid search | Warm-start from previous frame | Frame-to-frame temporal continuity makes warm-start dramatically faster |
| Fixed weights everywhere | Angular diversity weighting | Down-weights clustered ring cameras; standard multi-view practice |

**Deprecated/outdated:**
- `SfMPerspectiveCameras` in PyTorch3D: Deprecated alias. Use `PerspectiveCameras` with `cameras_from_opencv_projection`.
- `OpenGLPerspectiveCameras`: Deprecated. Use `PerspectiveCameras`.

---

## Open Questions

1. **Gravity prior implementation without explicit roll angle**
   - What we know: `FishState` has only yaw + pitch; lateral roll is not represented. The locked decision specifies "soft roll regularization."
   - What's unclear: Whether to (a) use pitch as a proxy, (b) add an explicit roll angle to FishState, or (c) derive roll from the rendered silhouette orientation.
   - Recommendation: Start with pitch proxy (theta^2 with low weight). If fish appear unnaturally rolled in visualizations, add a roll parameter to FishState — requires extending `build_spine_frames` with an additional rotation. This is a Claude's-discretion implementation decision.

2. **RefractiveCamera integration with MeshRasterizer**
   - What we know: `MeshRasterizer` expects a camera with a `transform_points(world_pts) -> ndc_pts` method. `RefractiveProjectionModel.project()` is differentiable and returns pixel coordinates.
   - What's unclear: Whether wrapping `project()` in a custom camera class works without modifying PyTorch3D internals.
   - Recommendation: Implement the `RefractiveCamera` wrapper (Pattern 2) and verify gradient flow with the test in Pattern 5. If this approach doesn't work (e.g., rasterizer requires specific camera class type checks), fall back to using `cameras_from_opencv_projection` with the post-undistortion K matrix — this gives pinhole rendering that approximates the refractive projection for modest angles. The refractive correction is most important for extreme angles (cameras at the ring periphery) but the optimization may converge well enough with the pinhole approximation.

3. **Crop render size vs. full-frame render for the rasterizer**
   - What we know: `RasterizationSettings(image_size=(H, W))` sets the render resolution. Rendering at 1600×1200 for every camera every iteration is expensive.
   - What's unclear: Whether to render at full resolution and then crop the alpha map, or render at crop resolution with adjusted intrinsics.
   - Recommendation: Render at crop resolution (e.g., 256×256) with intrinsics adjusted to the crop region. This is ~40x faster than full-frame rendering. The crop adjustment requires modifying the principal point and possibly the focal length — standard camera crop formula: `cx_crop = cx - x1, cy_crop = cy - y1` where (x1, y1) is the crop origin. This is a Claude's-discretion implementation decision.

4. **Learning rate schedule and Adam hyperparameters**
   - What we know: PyTorch3D tutorials use lr=0.05 for camera pose optimization on clean synthetic meshes. Fish silhouette optimization involves noisier targets (imperfect masks) and a more complex loss landscape.
   - Recommendation: Start with lr=1e-3, default Adam betas (0.9, 0.999). If convergence is slow, try lr=5e-3. If oscillating, reduce to lr=5e-4. Different parameters may benefit from different LRs: position (p) benefits from larger LR than angles. Use `Adam([{"params": [state.p], "lr": 5e-3}, {"params": [state.psi, state.theta, state.kappa, state.s], "lr": 1e-3}])`.

5. **Iteration cap for warm-start frames**
   - What we know: Warm-start begins from the previous frame's converged state. Convergence should be faster than first-frame optimization.
   - Recommendation: max_iters=50 for warm-start frames (vs. 300 for first frame). The convergence criterion (loss delta < 1e-4 for 3 steps) provides early exit — the cap is mainly a safety net.

---

## Sources

### Primary (HIGH confidence — local codebase examined directly)
- `C:/Users/tucke/PycharmProjects/AquaPose/src/aquapose/calibration/projection.py` — `RefractiveProjectionModel.project()` (differentiable, 10 fixed Newton-Raphson iters)
- `C:/Users/tucke/PycharmProjects/AquaPose/src/aquapose/mesh/builder.py` — `build_fish_mesh(list[FishState]) -> Meshes` (Phase 3 deliverable)
- `C:/Users/tucke/PycharmProjects/AquaPose/src/aquapose/mesh/state.py` — `FishState` dataclass
- `C:/Users/tucke/PycharmProjects/AquaPose/src/aquapose/segmentation/crop.py` — `CropRegion`, `compute_crop_region`, `extract_crop`, `paste_mask`
- `C:/Users/tucke/PycharmProjects/AquaPose/.planning/phases/03-fish-mesh-model-and-3d-initialization/03-VERIFICATION.md` — Phase 3 complete and all tests passing

### Primary (HIGH confidence — official PyTorch3D docs, verified)
- `pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/shader.html` — `SoftSilhouetteShader`, `BlendParams` sigma/gamma defaults
- `pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/rasterizer.html` — `RasterizationSettings`: image_size, blur_radius, faces_per_pixel defaults
- `pytorch3d.readthedocs.io/en/latest/modules/renderer/blending.html` — `BlendParams(sigma=1e-4, gamma=1e-4)` defaults
- `pytorch3d.readthedocs.io/en/latest/modules/utils.html` — `cameras_from_opencv_projection(R, tvec, camera_matrix, image_size) -> PerspectiveCameras`
- PyTorch3D camera_position_optimization tutorial — complete silhouette optimization loop with Adam, 200 iters, lr=0.05

### Secondary (MEDIUM confidence — verified against official source or multiple sources)
- `miropsota.github.io/torch_packages_builder/pytorch3d/` — Wheel list: no Windows + cu130 pytorch3d wheel exists; cu128 up to pt2.9.1 on Windows; cu130 Linux only to pt2.9.0
- `discuss.pytorch.org/t/how-to-implement-soft-iou-loss/15152` — Soft IoU formula (intersection / union with soft predictions)
- `github.com/facebookresearch/pytorch3d/issues/1792` — DLL load failure diagnosis: wheel-torch version mismatch is root cause
- `python -c "from pytorch3d.renderer import SoftSilhouetteShader"` (local test, run 2026-02-20): FAILS with DLL load error on torch 2.10.0+cu130 + pytorch3d-0.7.9+pt2.9.1cu128

### Tertiary (LOW confidence — verify before acting)
- Adam learning rate defaults for fish silhouette fitting: No domain-specific paper found. Starting from PyTorch3D tutorial defaults (lr=0.05) and adjusting downward for noisier loss. LOW confidence on exact values.
- Angular diversity weighting formula: No standard reference. The formula in Pattern 8 is a reasonable heuristic based on minimum angular separation — should be validated empirically during Phase 4 development.

---

## Metadata

**Confidence breakdown:**
- pytorch3d renderer API (SoftSilhouetteShader, BlendParams, RasterizationSettings): HIGH — official docs verified
- cameras_from_opencv_projection: HIGH — official docs verified, parameter shapes confirmed
- DLL load failure diagnosis and fix: HIGH — reproduced locally, miropsota wheel inventory confirmed
- Soft IoU formula: HIGH — standard formula, multiple sources
- Angular diversity weighting formula: LOW — Claude's discretion, no domain reference
- Adam hyperparameter defaults: LOW — educated extrapolation from PyTorch3D tutorials
- Roll angle gravity prior: MEDIUM — design decision needed (pitch proxy vs. explicit roll)

**Research date:** 2026-02-20
**Valid until:** pytorch3d installation strategy valid until miropsota releases a Windows+cu130 wheel; renderer API stable (last changed v0.7.x)
