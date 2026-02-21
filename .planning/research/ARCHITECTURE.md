# Architecture Research

**Domain:** Multi-view 3D animal pose estimation (analysis-by-synthesis, differentiable rendering)
**Researched:** 2026-02-19
**Confidence:** HIGH (system design confirmed by detailed project spec in `.planning/inbox/proposed_pipeline.md`; PyTorch3D rendering architecture verified via official docs)

---

## Standard Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE / PRE-RUN                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  CalibrationLoader (AquaCal)                                    │   │
│  │  • Per-camera intrinsics + extrinsics                           │   │
│  │  • Flat-plane refraction parameters (air-glass-water)           │   │
│  │  • Produces: RefractiveProjector Π_ref per camera               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         PHASE I — SEGMENTATION                         │
│  ┌──────────────┐   ┌──────────────────────┐   ┌───────────────────┐  │
│  │ VideoReader  │──▶│  InstanceSegmenter   │──▶│  MaskStore        │  │
│  │ (13 cameras) │   │  (Detectron2 /       │   │  M_i^(j) per      │  │
│  │ 30 fps, sync │   │   SAM2 fine-tuned)   │   │  camera per fish  │  │
│  └──────────────┘   └──────────────────────┘   └─────────┬─────────┘  │
│                                                           │            │
│                                                           ▼            │
│                                              ┌────────────────────┐    │
│                                              │ KeypointExtractor  │    │
│                                              │ (head / body /     │    │
│                                              │  tail from mask    │    │
│                                              │  PCA / geometry)   │    │
│                                              └────────┬───────────┘    │
└───────────────────────────────────────────────────────┼────────────────┘
                                                        │
                                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      PHASE II — 3D INITIALIZATION                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  EpipolarInitializer                                             │  │
│  │  • Cast refractive rays per keypoint per camera                  │  │
│  │  • Least-squares centroid solve (scipy) → p*, ψ*, s*            │  │
│  │  • κ = 0 (straight body) first frame; warm-start thereafter     │  │
│  │  Fallback: VoxelCarver (open3d) when keypoints are ambiguous    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  Output: initial FishState S = {p, ψ, κ, s} per fish                  │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    PHASE III — DIFFERENTIABLE REFINEMENT               │
│                                                                        │
│  ┌──────────────────┐   ┌────────────────────────────────────────┐     │
│  │  FishMeshBuilder │──▶│  RefractiveRenderer                   │     │
│  │  • Midline spline│   │  • Pre-project vertices through        │     │
│  │    from ψ + κ    │   │    Π_ref (AquaKit, differentiable)    │     │
│  │  • Swept ellipse │   │  • PyTorch3D MeshRasterizer            │     │
│  │    cross-sections│   │    (zbuf + SoftSilhouetteShader)       │     │
│  │  • Watertight    │   │  • One render per camera per fish      │     │
│  │    triangle mesh │   └────────────────┬───────────────────────┘     │
│  └──────────────────┘                   │                              │
│                                         ▼                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  LossComputer                                                    │  │
│  │  • L_sil  : IoU(rendered silhouette, observed mask) per camera   │  │
│  │  • L_grav : dorsal-up orientation prior                          │  │
│  │  • L_shape: cross-section aspect ratio vs. species norm          │  │
│  │  • L_temp : position/curvature acceleration penalty (frame ≥ 2) │  │
│  │  • Camera weighting: angular diversity downweights clustered cams│  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                         │                              │
│                                         ▼                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  PoseOptimizer (Adam, ~50–100 iters/frame)                       │  │
│  │  • Gradients flow: Loss → Renderer → Π_ref → FishState params   │  │
│  │  • All fish batched in parallel on GPU                           │  │
│  │  Output: refined FishState S* per fish per frame                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    PHASE IV — TRACKING AND IDENTITY                    │
│  ┌─────────────────────┐   ┌───────────────────────────────────────┐   │
│  │  MotionPredictor    │   │  AssignmentSolver                    │   │
│  │  (EKF per track)    │──▶│  • Mahalanobis cost matrix           │   │
│  │  • State: p, dp     │   │  • Hungarian algorithm               │   │
│  │  • Anisotropic Q    │   │  • Sex-classification cost boost     │   │
│  │    (σ_z > σ_xy)     │   │  • Gating threshold                  │   │
│  └─────────────────────┘   └───────────────────────────────────────┘   │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  InteractionHandler                                              │  │
│  │  • Merge overlapping detections → "interaction event"            │  │
│  │  • Split on re-separation; re-assign by trajectory + sex        │  │
│  │  • Global constraint: enforce exactly N fish at all times        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      PHASE V — OUTPUT AND VISUALIZATION                │
│  ┌──────────────────────┐   ┌──────────────────────────────────────┐   │
│  │  TrajectoryWriter    │   │  Visualizer                         │   │
│  │  (HDF5 via h5py,     │   │  • 2D overlay: project mesh into    │   │
│  │   parquet via        │   │    cameras via Π_ref + OpenCV       │   │
│  │   pyarrow/pandas)    │   │  • 3D scene: rerun-sdk or PyVista   │   │
│  └──────────────────────┘   │  • Analysis plots: matplotlib       │   │
│                             └──────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| CalibrationLoader | Parse AquaCal JSON; expose per-camera intrinsics, extrinsics, refraction params | `aquacal` library; read once at startup |
| RefractiveProjector | Map 3D world point → 2D image point via Snell's law at flat water surface | Differentiable PyTorch; Newton-Raphson + bisection fallback (from AquaMVS reference) |
| VideoReader | Load synchronized multi-camera frames; present as batched tensors | `cv2.VideoCapture` or `torchvision.io`; one reader per camera |
| InstanceSegmenter | Produce binary body masks M_i^(j) per camera per fish per frame | Detectron2 Mask R-CNN (ResNet-50-FPN); SAM2 for pseudo-label bootstrapping |
| KeypointExtractor | Derive coarse 2D head/body/tail points from mask geometry | PCA of mask pixels; endpoint detection |
| EpipolarInitializer | Estimate initial FishState from multi-view refractive ray intersections | `scipy.optimize.least_squares`; ray-to-point distance minimization |
| VoxelCarver | Fallback initializer via multi-view visual hull in a local bounding box | `open3d` voxel grid; intersection of back-projected masks |
| FishMeshBuilder | Generate watertight triangle mesh from FishState {p, ψ, κ, s} | Pure PyTorch; spline evaluation + swept ellipse cross-sections |
| RefractiveRenderer | Render per-camera silhouettes from mesh via Π_ref + PyTorch3D rasterizer | `pytorch3d.renderer.MeshRasterizer` → `SoftSilhouetteShader`; vertices pre-projected through Π_ref |
| LossComputer | Compute weighted multi-objective loss from rendered vs. observed masks | PyTorch operations; `kornia` for differentiable IoU if needed |
| PoseOptimizer | Run gradient descent to minimize loss over FishState parameters | `torch.optim.Adam`; ~50–100 iterations per frame; batched over all fish |
| MotionPredictor | Maintain per-fish EKF; predict next-frame position and covariance | `filterpy.kalman.ExtendedKalmanFilter`; 3D position + velocity state |
| AssignmentSolver | Match optimized detections to tracks via Mahalanobis distance + Hungarian | `scipy.optimize.linear_sum_assignment`; sex-penalty augmented cost matrix |
| InteractionHandler | Detect, manage, and resolve fish grouping events | Custom logic; topology constraint (N fish always) |
| TrajectoryWriter | Persist per-fish per-frame state vectors and metadata | `h5py` HDF5 + `pandas`/`pyarrow` for analysis |
| Visualizer | 2D video overlay and 3D scene rendering for QA and publication | `rerun-sdk` (live debug), `pyvista` (publication), `opencv` (video overlay) |

---

## Recommended Project Structure

```
src/aquapose/
├── calibration/           # Calibration loading and refractive geometry
│   ├── __init__.py
│   ├── loader.py          # AquaCal JSON parsing → camera model objects
│   └── refractive.py      # RefractiveProjector: 3D→2D via Snell's law (PyTorch, differentiable)
│
├── segmentation/          # Phase I: instance segmentation and keypoint extraction
│   ├── __init__.py
│   ├── segmenter.py       # Detectron2 / SAM2 inference wrapper
│   └── keypoints.py       # Mask-geometry keypoint extraction (head/body/tail)
│
├── initialization/        # Phase II: 3D pose initialization
│   ├── __init__.py
│   ├── epipolar.py        # Refractive ray-based centroid / heading solve
│   └── voxel_carving.py   # Visual hull fallback via open3d
│
├── mesh/                  # Parametric fish body model
│   ├── __init__.py
│   ├── fish_model.py      # FishMeshBuilder: spline + swept ellipses → triangle mesh
│   └── species.py         # Species-specific morphological parameters (aspect ratios, etc.)
│
├── rendering/             # Phase III: differentiable rendering
│   ├── __init__.py
│   ├── renderer.py        # RefractiveRenderer: Π_ref pre-projection + PyTorch3D rasterizer
│   └── loss.py            # LossComputer: silhouette IoU, gravity, shape, temporal
│
├── optimization/          # Phase III: pose refinement loop
│   ├── __init__.py
│   └── optimizer.py       # PoseOptimizer: Adam loop, warm-start, per-fish batching
│
├── tracking/              # Phase IV: EKF + Hungarian assignment + interaction handling
│   ├── __init__.py
│   ├── kalman.py          # MotionPredictor: per-fish EKF (filterpy)
│   ├── assignment.py      # AssignmentSolver: Mahalanobis cost matrix + Hungarian
│   └── interactions.py    # InteractionHandler: merge/split + population constraint
│
├── output/                # Phase V: storage and visualization
│   ├── __init__.py
│   ├── writer.py          # TrajectoryWriter: HDF5 + parquet export
│   └── visualizer.py      # 2D overlay, rerun-sdk 3D scene, matplotlib analysis
│
├── pipeline.py            # Top-level orchestrator: wires phases together, frame loop
├── state.py               # FishState dataclass: {p, ψ, κ, s} + metadata
└── __init__.py
```

### Structure Rationale

- **calibration/**: Isolated because it is loaded once and shared read-only by all phases. The `refractive.py` module is the single source of truth for Π_ref and must be differentiable — keeping it separate prevents accidental non-differentiable reimplementations elsewhere.
- **mesh/**: Separate from `rendering/` because the mesh builder is pure geometry (no rendering concern). Species parameters belong here so they can be swapped without touching the rendering or optimization code.
- **rendering/**: Contains both the renderer and the loss because the loss is tightly coupled to rendered outputs (IoU of rendered silhouette vs. observed mask). Keeping them co-located avoids awkward data passing.
- **optimization/**: Thin layer — just the Adam loop and warm-start logic. All complexity lives in `rendering/loss.py` and `mesh/`.
- **tracking/**: Completely decoupled from rendering. Takes optimized 3D states as input, outputs identity-labeled states.
- **output/**: Pure I/O. No domain logic. Can be extended without touching pipeline logic.
- **pipeline.py**: The only file that imports across all sub-packages. Acts as the wiring point for integration tests.

---

## Architectural Patterns

### Pattern 1: Pre-Project Vertices Through Refraction, Then Rasterize

**What:** Before passing a mesh to PyTorch3D's rasterizer, apply the refractive projection Π_ref to transform all vertices from 3D world coordinates into the distorted 2D camera space (as if the camera had standard pinhole intrinsics but in a refractive medium). The rasterizer then operates on these pre-projected vertices.

**When to use:** Always. PyTorch3D's rasterizer expects camera-space coordinates; Π_ref cannot be expressed as a standard projection matrix because refraction is nonlinear. Pre-projecting vertices is the correct architectural boundary.

**Trade-offs:** Vertices are pre-projected in world space, so the rasterizer depth buffer (zbuf) reflects camera-frame depths, not world depths. Gradients flow back through the rasterizer's zbuf/bary_coords tensors into the pre-projected vertex positions, and from there through Π_ref into FishState parameters. This chain is differentiable as long as Π_ref has valid gradients (Newton-Raphson with PyTorch autograd satisfies this).

**Example:**
```python
def render_silhouette(mesh: Meshes, proj: RefractiveProjector, camera_idx: int) -> torch.Tensor:
    # Pre-project all mesh vertices through refraction
    verts_world = mesh.verts_padded()               # (B, V, 3)
    verts_2d = proj.project(verts_world, camera_idx) # (B, V, 2) — differentiable
    # Feed pre-projected vertices to PyTorch3D rasterizer
    raster_out = rasterizer(build_camera_space_mesh(verts_2d, mesh.faces_padded()))
    # SoftSilhouetteShader uses alpha channel from raster_out
    return silhouette_shader(raster_out)             # (B, H, W, 1)
```

### Pattern 2: Warm-Start Initialization (Per-Frame and Per-Track)

**What:** For all frames after the first, initialize the optimizer from the previous frame's optimized FishState rather than re-running Phase II. For first frame of each track only, run epipolar initialization.

**When to use:** Always at 30 fps. Fish move ~1–3 mm/frame (<3% body length), so the previous solution is a near-optimal starting point. This reduces optimizer iterations from ~500 (cold start) to ~50–100.

**Trade-offs:** Creates a temporal dependency that means Phase III cannot process frames in arbitrary order. For parallelism across time, frames must be processed sequentially per-fish, or a batch of recent frames must be available. This is not a concern for online streaming processing but complicates random-access reprocessing.

**Example:**
```python
state = initial_state_from_phase_ii(frame_0)
for frame_t in video_frames[1:]:
    masks = segmenter.infer(frame_t)
    state = optimizer.refine(state, masks, warm_start=True)  # previous state is init
    writer.write(frame_t, state)
```

### Pattern 3: Batched Per-Fish GPU Parallelism

**What:** Pack all N fish into a single GPU batch for the rendering and optimization step. Each fish occupies one batch element; all Adam steps are vectorized across the batch dimension.

**When to use:** Phase III refinement. Fish are independent given fixed masks, so this is embarrassingly parallel. PyTorch3D Meshes and cameras support batched operations natively.

**Trade-offs:** Requires all fish meshes and masks to fit in GPU memory simultaneously. With 9 fish, ~500-vertex meshes, and 13 cameras at moderate resolution, this is tractable. Watch memory usage when scaling to 9 fish × 13 cameras × high-resolution masks — each rasterizer forward pass stores ~48 bytes per face per pixel (per PyTorch3D docs).

### Pattern 4: Cross-View Holdout Validation

**What:** Reserve one or more cameras as held-out test views during optimization. Silhouette loss is computed only on training cameras; held-out camera IoU is logged but not used in the gradient.

**When to use:** During development and hyperparameter tuning. Measures generalization of the 3D reconstruction to unseen viewpoints — a direct proxy for whether the 3D shape is correct rather than merely fitting the training views.

**Trade-offs:** Reduces the number of gradient-informing views. With 13 cameras and ~3–5 cameras covering each fish, holding out 1–2 cameras per fish is safe. The held-out camera should be selected to maximally stress-test the weakest geometric constraint (typically a camera orthogonal to the fish's primary viewing plane).

---

## Data Flow

### Per-Frame Processing Flow

```
Video frames (13 cameras, synchronized)
    │
    ▼
[InstanceSegmenter]
    │ M_i^(j): binary masks per camera per fish
    │ kp_i^(j): 2D keypoints (head, body, tail) per camera per fish
    ▼
[EpipolarInitializer]  ←── calibration (Π_ref per camera)
    │ FishState S_0^(j) = {p, ψ, κ=0, s}  (frame 0 only)
    │ OR: S_{t-1}^(j)  (warm start, frames 1+)
    ▼
[FishMeshBuilder]
    │ Meshes M^(j): watertight triangle mesh per fish
    ▼
[RefractiveRenderer]  ←── calibration (Π_ref per camera)
    │ Silhouettes R_i^(j): rendered alpha mask per camera per fish
    ▼
[LossComputer]  ←── M_i^(j) (observed masks)
    │ L_total^(j): scalar loss per fish
    ▼
[PoseOptimizer] (Adam, ~50–100 iters, autograd through full chain)
    │ S*^(j): refined FishState per fish
    ▼
[AssignmentSolver]  ←── EKF predictions {p̂^(j), P̂^(j)}
    │ identity-labeled states: (fish_id, S*)
    ▼
[TrajectoryWriter / Visualizer]
    │ HDF5 rows, 2D overlay video, 3D scene
```

### Gradient Flow (Phase III Critical Path)

```
L_total
  └── L_sil (IoU loss on alpha channel)
        └── SoftSilhouetteShader
              └── MeshRasterizer (zbuf, bary_coords — gradients supported)
                    └── pre-projected vertices (2D camera space)
                          └── Π_ref (RefractiveProjector — PyTorch autograd)
                                └── FishState parameters {p, ψ, κ, s}
                                      ← gradients arrive here; Adam updates
```

L_grav, L_shape, L_temp feed directly into FishState parameters without passing through the renderer.

### Key Data Structures

| Structure | Shape / Type | Produced By | Consumed By |
|-----------|-------------|-------------|-------------|
| `masks` | `dict[cam_id, (H, W) bool tensor]` per fish | InstanceSegmenter | EpipolarInitializer, LossComputer |
| `keypoints_2d` | `dict[cam_id, (3, 2) float]` per fish | KeypointExtractor | EpipolarInitializer |
| `FishState` | dataclass: `p(3), ψ(1), κ(K), s(1)` | Initializer / Optimizer | MeshBuilder, TrajectoryWriter, MotionPredictor |
| `Meshes` | PyTorch3D `Meshes` object, batched (N_fish, V, 3) | FishMeshBuilder | RefractiveRenderer |
| `rendered_silhouettes` | `(N_fish, N_cams, H, W, 1)` float | RefractiveRenderer | LossComputer |
| `track_state` | EKF state `[p, dp]` + covariance per track | MotionPredictor | AssignmentSolver |

---

## Build Order (Phase Dependencies)

The system has strict data dependencies that dictate build order:

```
CalibrationLoader ──────────────────────────────────────┐
(must exist first; everything depends on it)            │
                                                        ▼
InstanceSegmenter ──────┐                      RefractiveProjector
(provides masks)        │                      (must be differentiable;
                        │                       built before renderer)
KeypointExtractor ──────┤
(provides coarse 2D     │
 keypoints from masks)  │
                        ▼
             EpipolarInitializer
             (depends on keypoints + Π_ref)
                        │
                        ▼
             FishMeshBuilder ──────────────────┐
             (parametric mesh; pure geometry)   │
                                               ▼
             RefractiveRenderer ◄──── Π_ref + FishMeshBuilder
                        │
                        ▼
             LossComputer ◄──── masks (from InstanceSegmenter)
                        │
                        ▼
             PoseOptimizer
             (wires autograd chain; depends on everything above)
                        │
                        ▼
             MotionPredictor / AssignmentSolver
             (depends on optimized FishState; independent of renderer)
                        │
                        ▼
             TrajectoryWriter / Visualizer
             (pure I/O; built last)
```

**Recommended build stages:**

| Stage | Components to Build | Validation Gate |
|-------|--------------------|--------------------|
| 1 | CalibrationLoader + RefractiveProjector | Round-trip: project known 3D points, check reprojection error < 1px |
| 2 | InstanceSegmenter + KeypointExtractor | Mask IoU ≥ 0.90 on held-out frames |
| 3 | EpipolarInitializer | 3D centroid within 5mm of hand-measured ground truth on test frames |
| 4 | FishMeshBuilder | Mesh is watertight; spline + ellipses produce visually correct shape |
| 5 | RefractiveRenderer | Rendered silhouette visually aligns with observed mask for known pose |
| 6 | LossComputer + PoseOptimizer (single fish) | L_sil converges; optimized pose matches GT better than initialization |
| 7 | VoxelCarver (fallback) | Produces plausible initialization when epipolar fails |
| 8 | MotionPredictor + AssignmentSolver | Zero identity swaps on non-interacting sequences |
| 9 | InteractionHandler | Identity preserved through a simulated merge-split event |
| 10 | TrajectoryWriter + Visualizer | HDF5 loads cleanly; 2D overlay looks correct |

---

## Anti-Patterns

### Anti-Pattern 1: Non-Differentiable Refractive Projection

**What people do:** Implement Π_ref using numpy, scipy, or non-PyTorch solvers for performance, then call `.detach()` when passing to the renderer.

**Why it's wrong:** Breaks the gradient chain. The optimizer cannot backpropagate through the refractive projection into {p, ψ, κ, s}. The optimization will still converge to local minima of whatever is differentiable, but the physically meaningful gradients from the silhouette are lost. This is silent — the optimizer runs without error but produces wrong results.

**Do this instead:** Implement Π_ref entirely in PyTorch. The Newton-Raphson solver for Snell's law at a flat interface can be unrolled as a fixed-iteration loop with autograd-compatible operations. The AquaMVS reference implementation should be verified to use PyTorch operations throughout, not numpy fallbacks.

### Anti-Pattern 2: Per-Camera Sequential Rendering

**What people do:** Loop over cameras in Python, calling the renderer once per camera per fish per frame.

**Why it's wrong:** Leaves GPU throughput on the table. With 13 cameras and 9 fish, a sequential loop invokes the renderer 117 times per frame. Python loop overhead and kernel launch latency dominate runtime.

**Do this instead:** Batch cameras and fish into a single rasterizer call by stacking into the batch dimension of PyTorch3D's `Meshes` and `Cameras` objects. One forward pass per frame, not 117.

### Anti-Pattern 3: Applying the Full Loss to All Cameras Equally

**What people do:** Sum silhouette losses over all cameras with equal weight (w_i = 1 for all i).

**Why it's wrong:** With 12 ring cameras spaced ~30° apart, cameras on the same side of the tank see nearly identical projections of the fish. Equal weighting lets these clusters dominate the gradient, pulling the optimizer toward a pose that satisfies the local cluster rather than the full 3D reconstruction.

**Do this instead:** Apply angular diversity weighting — downweight cameras whose viewing directions cluster together relative to the fish centroid. The proposed formula in `proposed_pipeline.md` (exponential repulsion in viewing direction space) achieves this.

### Anti-Pattern 4: Deferring Calibration Validation

**What people do:** Assume the calibration library is correct and build Phase II and III before checking reprojection errors.

**Why it's wrong:** If the refractive camera model has a systematic error (wrong glass thickness, wrong refractive index, wrong air-glass interface position), every downstream component inherits the error. A 1mm calibration error can produce 5–10mm pose errors at the far side of the 2m tank.

**Do this instead:** Build CalibrationLoader and RefractiveProjector first. Validate by projecting known 3D points (e.g., calibration target locations) into all cameras and measuring reprojection error. This must pass before building any optimization code.

### Anti-Pattern 5: Optimizing All Loss Terms From Frame 1

**What people do:** Enable all four loss terms (L_sil, L_grav, L_shape, L_temp) from the very first frame.

**Why it's wrong:** L_temp requires at least two previous frames. On frame 0, L_temp is undefined; activating it forces arbitrary initialization of the temporal buffer, which can destabilize the optimizer. More subtly, λ values tuned on frame 0 (cold start) behave differently on warm-started frames.

**Do this instead:** Activate L_temp only from frame 2 onward. Tune λ_sil, λ_grav, λ_shape on frame 0 in isolation, then validate that L_temp does not dominate when introduced.

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| v1: 1 fish, 5–30 min clips | Sequential frame processing; single-fish optimizer; everything fits in memory |
| v2: 9 fish, 5–30 min clips | Batched optimizer (9 fish per GPU call); EKF + Hungarian tracking; streaming frame reading |
| v3: 9 fish, full-day recordings (hours) | Streaming with HDF5 checkpointing; cannot hold full video in RAM; must resume mid-clip after interruption |
| v4: real-time or near-real-time | Profiling required; CUDA kernel optimization for Π_ref; potential TorchScript compilation of renderer |

**First bottleneck (9 fish × 13 cameras):** GPU memory for batched rendering. With ~500 vertices per fish and 480×480px masks, memory per rasterizer pass ≈ 48 bytes × faces_per_pixel × H × W × N_fish × N_cams. Profile before assuming it fits.

**Second bottleneck (full-day recordings):** Disk I/O for mask loading. Pre-compute and store all masks to disk after segmentation; do not re-run the segmenter during optimization.

---

## Integration Points

### External Dependencies

| Dependency | Integration Pattern | Notes |
|------------|---------------------|-------|
| AquaCal (calibration library) | Import at startup; load JSON config; expose `RefractiveProjector` objects | Must verify: is Π_ref exposed as a differentiable PyTorch function, or does it require re-implementation? See PITFALLS.md |
| PyTorch3D | `MeshRasterizer` + `SoftSilhouetteShader`; batched Meshes + Cameras API | Version-sensitive: zbuf gradient support and batch API have changed across versions. Pin to a tested version. |
| Detectron2 | Inference only at runtime; training is offline | Large dependency; consider containerizing separately from the optimization pipeline |
| SAM2 | Offline pseudo-label generation only; not in the live inference path | |
| filterpy | `ExtendedKalmanFilter` class | Lightweight; no GPU dependency |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Phase I → Phase II | `dict[fish_id, dict[cam_id, mask_tensor]]` + `dict[fish_id, dict[cam_id, keypoints]]` | Masks stay on CPU until needed for loss computation |
| Phase II → Phase III | `FishState` dataclass with `.requires_grad_(True)` parameters | State must have autograd enabled before entering optimizer |
| Phase III → Phase IV | `list[FishState]` (detached, CPU) — plain 3D positions and orientations | Phase IV uses numpy/scipy; do not pass autograd-enabled tensors |
| Phase IV → Phase V | `list[FishTrajectoryRecord]` — fish_id, position, heading, curvature, scale, metadata | Pure data; no library-specific types |
| CalibrationLoader → all phases | `CameraModel` objects with `project()` method | Shared read-only; no mutable state after initialization |

---

## Sources

- PyTorch3D renderer architecture: [https://pytorch3d.org/docs/renderer](https://pytorch3d.org/docs/renderer) — HIGH confidence (official docs)
- PyTorch3D rasterizer output tensors (zbuf, bary_coords, gradients): [https://pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/rasterizer.html](https://pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/rasterizer.html) — HIGH confidence (official docs)
- PyTorch3D SoftSilhouetteShader: verified via official renderer getting started guide — HIGH confidence
- AquaPose proposed pipeline (authoritative project spec): `.planning/inbox/proposed_pipeline.md` — HIGH confidence (project owner document)
- Refractive underwater camera calibration (2024): [https://arxiv.org/abs/2405.18018](https://arxiv.org/abs/2405.18018) — MEDIUM confidence
- Multi-view 3D pose estimation with triangulation + iterative refinement patterns (CVPR 2024): [MVGFormer](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.pdf) — MEDIUM confidence
- Multi-fish tracking with EKF + Hungarian: [SOD-SORT](https://arxiv.org/html/2507.06400v3) — MEDIUM confidence (confirms EKF + Hungarian as standard for fish tracking)
- Analysis-by-synthesis with differentiable rendering: [VoGE](https://arxiv.org/abs/2205.15401), [SkelSplat](https://skelsplat.github.io/) — MEDIUM confidence (confirms pattern is well-established)

---

*Architecture research for: AquaPose — 3D fish pose estimation via multi-view analysis-by-synthesis*
*Researched: 2026-02-19*
