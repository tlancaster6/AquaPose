# Architecture Research

**Domain:** Multi-view 3D fish pose estimation (direct triangulation with optional analysis-by-synthesis refinement)
**Researched:** 2026-02-21
**Confidence:** HIGH (pipeline design confirmed by pivot proposal in `.planning/inbox/fish-reconstruction-pivot.md`; refractive geometry and calibration verified in Phase 01)

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
│  │    (ray casting: 2D→3D ray; forward projection: 3D→2D pixel)   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE I — DETECTION & SEGMENTATION                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │ VideoReader  │──▶│  Detector    │──▶│ UNetSegmentor│              │
│  │ (12 cameras) │   │ (YOLO /      │   │ (binary mask │              │
│  │ 30 fps, sync │   │  MOG2)       │   │  per crop)   │              │
│  └──────────────┘   └──────────────┘   └──────┬───────┘              │
│                                                │                      │
│                                                ▼                      │
│                                       ┌───────────────┐              │
│                                       │   MaskStore   │              │
│                                       │ M_i per cam   │              │
│                                       │ per detection │              │
│                                       └───────┬───────┘              │
└───────────────────────────────────────────────┼──────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE II — CROSS-VIEW IDENTITY                          │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  CentroidExtractor                                               │ │
│  │  • 2D centroid per detection per camera                          │ │
│  │  • Cast refractive ray per centroid into 3D                      │ │
│  └──────────────────────────────────┬───────────────────────────────┘ │
│                                     ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  RANSACIdentityAssociator                                        │ │
│  │  • Sample camera subsets, triangulate centroids                   │ │
│  │  • Score by reprojection error → consensus sets                  │ │
│  │  • Each cluster = one physical fish                               │ │
│  │  Output: (camera_id, det_id) → fish_id mapping                   │ │
│  └──────────────────────────────────┬───────────────────────────────┘ │
│                                     ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  HungarianTracker (3D)                                           │ │
│  │  • Match per-frame 3D centroids to persistent fish IDs           │ │
│  │  • Nearest-neighbor in 3D with Hungarian assignment              │ │
│  │  • Warm-start: previous frame's 3D positions seed association    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│  Output: identity_map + 3D centroid per fish                         │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE III — MIDLINE EXTRACTION                          │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  MaskSmoother                                                    │ │
│  │  • Morphological closing + opening (adaptive kernel)             │ │
│  │  • Required at current U-Net IoU ~0.62                           │ │
│  └──────────────────────────────────┬───────────────────────────────┘ │
│                                     ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Skeletonizer                                                    │ │
│  │  • skimage.morphology.skeletonize → 1px skeleton                 │ │
│  │  • Distance transform for local half-width                       │ │
│  │  • LongestPathBFS: two-pass BFS → head-to-tail pruned midline   │ │
│  └──────────────────────────────────┬───────────────────────────────┘ │
│                                     ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  ArcLengthSampler                                                │ │
│  │  • Cumulative arc length → normalize to [0, 1]                   │ │
│  │  • Resample at N fixed positions (e.g., N=15)                    │ │
│  │  • Consistent cross-view correspondence via slender-body approx  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│  Output per fish per camera: N × (x, y, half_width) at fixed t vals │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE IV — 3D RECONSTRUCTION                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  MultiViewTriangulator                                           │ │
│  │  • Per body point: RANSAC over camera subsets                    │ │
│  │  • Refractive ray intersection (existing code)                   │ │
│  │  • View-angle weighting: downweight cameras looking along body   │ │
│  │  • Output: N 3D points + half-widths + residuals per fish        │ │
│  └──────────────────────────────────┬───────────────────────────────┘ │
│                                     ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  SplineFitter                                                    │ │
│  │  • Cubic B-spline through N triangulated points (5–8 controls)   │ │
│  │  • Separate 1D spline for width profile                          │ │
│  │  • Output: continuous 3D midline + width "tube model"            │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│       PHASE V — OPTIONAL REFINEMENT (add only if baseline fails)      │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  LMRefiner                                                       │ │
│  │  • Reprojection-based optimization of spline control points      │ │
│  │  • scipy.optimize.least_squares (method='lm')                    │ │
│  │  • Warm-start from SplineFitter output → 5–20 iterations         │ │
│  │  • Optional: smoothness, temporal, width regularization           │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE VI — OUTPUT AND VISUALIZATION                     │
│  ┌──────────────────────┐   ┌──────────────────────────────────────┐  │
│  │  TrajectoryWriter    │   │  Visualizer                         │  │
│  │  (HDF5 via h5py,     │   │  • 2D overlay: project 3D midline   │  │
│  │   parquet via        │   │    into cameras via Π_ref + OpenCV  │  │
│  │   pyarrow/pandas)    │   │  • 3D scene: rerun-sdk              │  │
│  └──────────────────────┘   │  • Analysis plots: matplotlib       │  │
│                             └──────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| CalibrationLoader | Parse AquaCal JSON; expose per-camera intrinsics, extrinsics, refraction params | `aquacal` library; read once at startup |
| RefractiveProjector | Ray casting (2D→3D ray) and forward projection (3D→2D) via Snell's law at flat water surface | PyTorch for forward projection; Newton-Raphson + bisection fallback (from AquaMVS reference) |
| VideoReader | Load synchronized multi-camera frames; present as batched tensors | `cv2.VideoCapture`; one reader per camera |
| Detector | Produce bounding-box detections per camera per frame | YOLOv8 (primary) or MOG2 (fallback); `make_detector("yolo", ...)` |
| UNetSegmentor | Produce binary body masks per detection crop | MobileNetV3-Small encoder U-Net; 128×128 input |
| CentroidExtractor | Compute 2D centroid per detection; cast refractive ray into 3D | Mask/bbox centroid + `RefractiveProjector.cast_ray()` |
| RANSACIdentityAssociator | Cluster rays across cameras into per-fish identity groups | RANSAC over camera subsets; reprojection error scoring |
| HungarianTracker | Assign persistent fish IDs across frames using 3D centroid matching | `scipy.optimize.linear_sum_assignment`; Euclidean cost in 3D |
| MaskSmoother | Morphological preprocessing to stabilize skeletonization on noisy masks | `cv2.morphologyEx`; adaptive kernel radius |
| Skeletonizer | Extract 1px skeleton + distance transform half-widths from smoothed mask | `skimage.morphology.skeletonize` + `scipy.ndimage.distance_transform_edt` |
| ArcLengthSampler | Resample 2D midline at N fixed normalized arc-length positions | Cumulative arc length + linear interpolation; N=15 default |
| MultiViewTriangulator | Triangulate each body point across cameras with RANSAC + view-angle weighting | Refractive ray intersection code; per-point RANSAC |
| SplineFitter | Fit cubic B-spline to triangulated 3D points + 1D width profile | `scipy.interpolate.splprep`; 5–8 control points |
| LMRefiner | Optional reprojection-based refinement of spline control points | `scipy.optimize.least_squares` (LM); 15–24 parameters |
| TrajectoryWriter | Persist per-fish per-frame spline control points and metadata | `h5py` HDF5 + `pandas`/`pyarrow` for analysis |
| Visualizer | 2D midline overlay and 3D scene rendering for QA and publication | `rerun-sdk` (live debug), `opencv` (video overlay), `matplotlib` (analysis) |

---

## Recommended Project Structure

```
src/aquapose/
├── calibration/           # Calibration loading and refractive geometry
│   ├── __init__.py
│   ├── loader.py          # AquaCal JSON parsing → camera model objects
│   └── refractive.py      # RefractiveProjector: ray casting (2D→3D) + forward projection (3D→2D)
│
├── segmentation/          # Phase I: detection and segmentation
│   ├── __init__.py
│   ├── detection.py       # Detector factory: YOLO / MOG2
│   ├── unet.py            # UNetSegmentor: binary mask from crop
│   └── crop.py            # Shared crop utilities (bbox → crop → full-frame)
│
├── identity/              # Phase II: cross-view identity + 3D tracking
│   ├── __init__.py
│   ├── centroid_clustering.py   # CentroidExtractor + RANSACIdentityAssociator
│   └── hungarian_tracker.py     # HungarianTracker: persistent 3D fish ID assignment
│
├── reconstruction/        # Phases III–IV: midline extraction + 3D reconstruction
│   ├── __init__.py
│   ├── medial_axis.py     # MaskSmoother + Skeletonizer + LongestPathBFS
│   ├── arc_length.py      # ArcLengthSampler: normalize + resample midline
│   ├── triangulation.py   # MultiViewTriangulator: RANSAC + view-angle weighting
│   └── spline.py          # SplineFitter: B-spline midline + width profile
│
├── refinement/            # Phase V: optional LM refinement (add only if needed)
│   ├── __init__.py
│   └── lm_refiner.py      # LMRefiner: reprojection-based spline optimization
│
├── mesh/                  # Parametric fish body model (shelved pipeline; retained for reference)
│   ├── __init__.py
│   ├── fish_model.py      # FishMeshBuilder: spline + swept ellipses → triangle mesh
│   └── species.py         # Species-specific morphological parameters
│
├── rendering/             # Differentiable rendering (shelved pipeline; retained for reference)
│   ├── __init__.py
│   ├── renderer.py        # RefractiveRenderer: Π_ref pre-projection + PyTorch3D rasterizer
│   └── loss.py            # LossComputer: silhouette IoU, gravity, shape, temporal
│
├── optimization/          # Adam pose refinement (shelved pipeline; retained for reference)
│   ├── __init__.py
│   └── optimizer.py       # PoseOptimizer: Adam loop, warm-start, per-fish batching
│
├── output/                # Phase VI: storage and visualization
│   ├── __init__.py
│   ├── writer.py          # TrajectoryWriter: HDF5 + parquet export
│   └── visualizer.py      # 2D midline overlay, rerun-sdk 3D scene, matplotlib analysis
│
├── pipeline.py            # Top-level orchestrator: wires phases together, frame loop
├── state.py               # Fish data structures: spline controls, identity map, etc.
└── __init__.py
```

### Structure Rationale

- **calibration/**: Isolated because it is loaded once and shared read-only by all phases. The `refractive.py` module is the single source of truth for both ray casting (2D→3D) and forward projection (3D→2D). Used by identity association (ray casting), triangulation (ray casting), and optional LM refinement (forward projection).
- **segmentation/**: Detection (YOLO/MOG2) and U-Net segmentation. Produces binary masks that feed all downstream phases.
- **identity/**: Cross-view association and temporal tracking. Decoupled from reconstruction because identity must be resolved before midline correspondence can be established.
- **reconstruction/**: The core of the new pipeline. Four files corresponding to the four sub-steps: medial axis → arc-length → triangulation → spline. Each is independently testable.
- **refinement/**: Optional. Only added if baseline triangulation is insufficient. Kept separate to avoid coupling the primary pipeline to an optimization dependency.
- **mesh/, rendering/, optimization/**: Shelved pipeline code. Retained in the codebase for potential future use (e.g., analysis-by-synthesis refinement as an alternative to LM refinement). Not imported by the primary pipeline.
- **output/**: Pure I/O. No domain logic. Can be extended without touching pipeline logic.
- **pipeline.py**: The only file that imports across all sub-packages. Acts as the wiring point for integration tests.

---

## Architectural Patterns

### Pattern 1: Pre-Project Vertices Through Refraction, Then Rasterize (Shelved Pipeline Only)

**What:** Before passing a mesh to PyTorch3D's rasterizer, apply the refractive projection Π_ref to transform all vertices from 3D world coordinates into the distorted 2D camera space. The rasterizer then operates on these pre-projected vertices.

**When to use:** Only in the shelved analysis-by-synthesis pipeline. Not used in the primary direct triangulation pipeline.

**Trade-offs:** Vertices are pre-projected in world space, so the rasterizer depth buffer (zbuf) reflects camera-frame depths, not world depths. Gradients flow back through the rasterizer's zbuf/bary_coords tensors into the pre-projected vertex positions, and from there through Π_ref into FishState parameters. This chain is differentiable as long as Π_ref has valid gradients.

### Pattern 2: Temporal Warm-Start of Identity Association

**What:** For all frames after the first, seed the Hungarian tracker's cost matrix with the previous frame's 3D centroid positions. Fish move ~1–3 mm/frame (<3% body length) at 30 fps, so the previous positions are near-optimal starting points for identity assignment.

**When to use:** Always in the primary pipeline. The HungarianTracker maintains a persistent state of 3D positions per fish ID and uses these as predictions for the next frame.

**Trade-offs:** Creates a temporal dependency — frames must be processed sequentially. For random-access reprocessing, the tracker state must be serialized/restored. Identity association on the first frame (or after a long gap) requires a cold-start heuristic (e.g., spatial clustering only).

**Example:**
```python
tracker = HungarianTracker(n_fish=9)
for frame_t in video_frames:
    masks = segmenter.infer(frame_t)
    centroids_3d = identity_associator.cluster(masks, calibration)
    fish_ids = tracker.assign(centroids_3d)  # warm-start from previous positions
    midlines = reconstruct(masks, fish_ids, calibration)
    writer.write(frame_t, midlines)
```

### Pattern 3: Parallelize Across Fish, Not Pipeline Stages

**What:** All pipeline stages (midline extraction, arc-length sampling, triangulation, spline fitting) are independent across fish within a single frame. Parallelize by processing multiple fish simultaneously rather than pipelining stages.

**When to use:** When processing 8–9 fish per frame. Each fish's reconstruction is embarrassingly parallel once cross-view identity is resolved.

**Trade-offs:** Thread pool or process pool over fish. With 9 fish and CPU-bound triangulation, expect near-linear speedup up to the number of physical cores. No GPU synchronization concerns since the primary pipeline is CPU-bound.

### Pattern 4: Cross-View Holdout Validation

**What:** Reserve one or more cameras as held-out test views. Triangulate using only the remaining cameras; measure reprojection error on the held-out camera as a quality metric.

**When to use:** During development and hyperparameter tuning. Measures whether the 3D reconstruction generalizes to unseen viewpoints — a direct proxy for reconstruction accuracy.

**Trade-offs:** Reduces the number of cameras available for triangulation. With 12 cameras and typically 4–8 cameras seeing each fish, holding out 1–2 cameras per fish is safe. The held-out camera should be selected to maximally stress-test the weakest geometric constraint (typically a camera viewing along the fish's body axis).

### Pattern 5: Arc-Length Correspondence (Slender-Body Assumption)

**What:** Fish are slender bodies with a single dominant axis. The arc-length parameterization of the 2D midline projection is approximately preserved across views: point at normalized arc-length t=0.3 in camera A corresponds to the same physical body position as t=0.3 in camera B.

**When to use:** Always. This is the fundamental assumption enabling cross-view correspondence without explicit keypoint detection. It replaces the mesh-based correspondence used in the shelved pipeline.

**Trade-offs:** The approximation breaks down for significantly curved fish viewed from very different angles — foreshortening compresses the arc-length mapping unevenly. Cameras viewing along the fish's body axis are the worst offenders. RANSAC per body point (Pattern 4 in MultiViewTriangulator) and view-angle weighting mitigate this. If the assumption fails badly, epipolar-guided correspondence refinement can replace it (see pivot proposal, "Key Implementation Notes").

---

## Data Flow

### Per-Frame Processing Flow

```
Video frames (12 cameras, synchronized)
    │
    ▼
[Detector + UNetSegmentor]
    │ Per camera: binary masks + bounding boxes per detection
    ▼
[CentroidExtractor]  ←── calibration (Π_ref ray casting)
    │ Per camera per detection: 2D centroid + 3D ray
    ▼
[RANSACIdentityAssociator]  ←── calibration (Π_ref reprojection scoring)
    │ identity_map: (camera_id, det_id) → fish_id
    │ 3D centroid per fish
    ▼
[HungarianTracker]  ←── previous frame's 3D positions (warm-start)
    │ Persistent fish_id assignment
    ▼
[MaskSmoother → Skeletonizer → ArcLengthSampler]
    │ Per fish per camera: N × (x, y, half_width) at fixed arc-length t
    ▼
[MultiViewTriangulator]  ←── calibration (Π_ref ray casting + reprojection)
    │ Per fish: N 3D points + half-widths + residuals
    ▼
[SplineFitter]
    │ Per fish: 3D midline spline (5–8 control points) + width profile
    ▼
[LMRefiner]  ←── (OPTIONAL, only if baseline insufficient)
    │ Per fish: refined spline control points
    ▼
[TrajectoryWriter / Visualizer]
    │ HDF5 rows, 2D midline overlay video, 3D scene
```

### Key Data Structures

| Structure | Shape / Type | Produced By | Consumed By |
|-----------|-------------|-------------|-------------|
| `masks` | `dict[cam_id, (H, W) bool tensor]` per detection | UNetSegmentor | CentroidExtractor, Skeletonizer |
| `identity_map` | `dict[(cam_id, det_id), fish_id]` | RANSACIdentityAssociator | all downstream per-fish stages |
| `midline_2d` | `(N, 3) float array` (x, y, half_width) per fish per camera | ArcLengthSampler | MultiViewTriangulator, LMRefiner |
| `midline_3d` | `(N, 3) float array` per fish | MultiViewTriangulator | SplineFitter |
| `spline_controls` | `(5–8, 3) float array` per fish | SplineFitter / LMRefiner | TrajectoryWriter, Visualizer |
| `width_profile` | 1D spline (scipy tck tuple) per fish | SplineFitter | TrajectoryWriter, Visualizer |

---

## Build Order (Phase Dependencies)

The system has strict data dependencies that dictate build order:

```
CalibrationLoader ──────────────────────────────────────┐
(must exist first; everything depends on it)            │
                                                        ▼
Detector + UNetSegmentor ──────┐               RefractiveProjector
(provides masks + bboxes)      │               (ray casting + forward
                               │                projection; built before
                               │                identity or triangulation)
CentroidExtractor ─────────────┤
(provides 2D centroids +       │
 3D rays from masks)           │
                               ▼
             RANSACIdentityAssociator
             (depends on centroids + Π_ref)
                               │
                               ▼
             HungarianTracker
             (depends on 3D centroids per frame)
                               │
                               ▼
             MaskSmoother + Skeletonizer
             (depends on masks + identity_map)
                               │
                               ▼
             ArcLengthSampler
             (depends on pruned 2D midlines)
                               │
                               ▼
             MultiViewTriangulator ◄──── Π_ref + identity_map
                               │
                               ▼
             SplineFitter
             (depends on triangulated 3D points)
                               │
                               ▼
             LMRefiner (optional)
             (depends on spline + 2D midlines + Π_ref)
                               │
                               ▼
             TrajectoryWriter / Visualizer
             (pure I/O; built last)
```

**Recommended build stages:**

| Stage | Components to Build | Validation Gate |
|-------|--------------------|--------------------|
| 1 | CalibrationLoader + RefractiveProjector | Round-trip: project known 3D points, check reprojection error < 1px |
| 2 | Detector + UNetSegmentor | Mask IoU on held-out frames (current: ~0.62; sufficient to unblock) |
| 3 | CentroidExtractor + RANSACIdentityAssociator | Identity assignment matches manual annotation on test frames |
| 4 | HungarianTracker | Zero identity swaps on non-interacting sequences |
| 5 | MaskSmoother + Skeletonizer | Skeleton is single-path, head-to-tail, stable across minor mask perturbations |
| 6 | ArcLengthSampler | Sampled points at same t-value correspond visually across cameras |
| 7 | MultiViewTriangulator | Per-point triangulation residual < 2mm on majority of points |
| 8 | SplineFitter | Reprojection of 3D spline into held-out camera visually aligns with mask |
| 9 | LMRefiner (optional) | Reprojection error decreases vs. baseline; no overfitting to noisy masks |
| 10 | TrajectoryWriter + Visualizer | HDF5 loads cleanly; 2D overlay looks correct |

---

## Anti-Patterns

### Anti-Pattern 1: Non-Differentiable Refractive Projection

**What people do:** Implement Π_ref using numpy, scipy, or non-PyTorch solvers for performance, then lose differentiability.

**Why it matters:** In the primary pipeline, differentiability through Π_ref is not required — triangulation and spline fitting use it as a black-box function. However, if the optional LM refiner (Phase V) is added, the forward projection must provide accurate Jacobians for the Levenberg-Marquardt solver. Additionally, if the shelved analysis-by-synthesis pipeline is ever revived, full PyTorch autograd differentiability is essential.

**Do this instead:** Maintain a PyTorch implementation of Π_ref for potential future use. For the primary pipeline, a numpy implementation is acceptable and may be faster. If LM refinement is needed, `scipy.optimize.least_squares` can compute Jacobians via finite differences, but analytic Jacobians are more efficient.

### Anti-Pattern 2: Per-Fish Sequential Processing

**What people do:** Loop over fish in Python, processing each fish's midline extraction, triangulation, and spline fitting sequentially.

**Why it's wrong:** With 8–9 fish and CPU-bound computation, sequential processing leaves cores idle. All stages are independent across fish within a frame.

**Do this instead:** Use a thread pool or process pool across fish. With `concurrent.futures.ProcessPoolExecutor`, achieve near-linear speedup. Batch triangulations into vectorized calls where possible (~15 points × 8 fish = 120 triangulations per frame).

### Anti-Pattern 3: Equal Camera Weighting in Triangulation

**What people do:** Triangulate each body point using all visible cameras with equal weight.

**Why it's wrong:** Cameras whose viewing ray is nearly parallel to the fish's local body axis at that point suffer the worst arc-length correspondence errors. Equal weighting lets these cameras introduce systematic bias into the triangulation.

**Do this instead:** Apply view-angle weighting: for each body point in each camera, compute the angle between the camera's viewing ray and the local tangent direction of the 2D midline. Downweight or exclude cameras where this angle is small (i.e., looking along the body axis). RANSAC per body point provides a second line of defense.

### Anti-Pattern 4: Deferring Calibration Validation

**What people do:** Assume the calibration library is correct and build downstream components before checking reprojection errors.

**Why it's wrong:** If the refractive camera model has a systematic error (wrong glass thickness, wrong refractive index, wrong air-glass interface position), every downstream component inherits the error. A 1mm calibration error can produce 5–10mm pose errors at the far side of the 2m tank.

**Do this instead:** Build CalibrationLoader and RefractiveProjector first. Validate by projecting known 3D points (e.g., calibration target locations) into all cameras and measuring reprojection error. This must pass before building any reconstruction code.

### Anti-Pattern 5: Skipping Mask Smoothing Before Skeletonization

**What people do:** Run `skimage.morphology.skeletonize` directly on noisy U-Net masks (IoU ~0.62) without morphological preprocessing.

**Why it's wrong:** Noisy mask boundaries produce spurious skeleton branches and unstable midline paths. The two-pass BFS longest-path pruning handles some of this, but severe boundary noise creates false endpoints that defeat the pruning. The result is frame-to-frame skeleton jitter that propagates through arc-length sampling into triangulation errors.

**Do this instead:** Apply morphological closing then opening with an adaptive kernel radius proportional to the mask's minor axis width (e.g., `max(3, minor_axis_width // 8)` pixels). This removes boundary noise without eroding thin body regions like the caudal peduncle.

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| v1: 1 fish, 5–30 min clips | Sequential frame processing; single-fish reconstruction; everything fits in memory |
| v2: 9 fish, 5–30 min clips | Process-pool across fish; Hungarian tracking; streaming frame reading |
| v3: 9 fish, full-day recordings (hours) | Streaming with HDF5 checkpointing; cannot hold full video in RAM; must resume mid-clip after interruption |
| v4: near-real-time | Profile triangulation bottleneck; potential Cython/numba for ray intersection; pre-compute camera visibility maps |

**Primary scaling factors:** Number of fish × number of cameras × frame rate. With 9 fish × 12 cameras × 15 body points = 1,620 triangulations per frame at 30 fps = ~48,600 triangulations/second. The refractive ray intersection code must be fast — this is the CPU-bound bottleneck.

**No GPU memory concern** for the primary pipeline. Detection and segmentation use GPU (YOLO + U-Net inference), but all reconstruction is CPU-bound (skeletonization, triangulation, spline fitting). GPU memory is only a concern if the shelved analysis-by-synthesis pipeline is revived.

**Disk I/O bottleneck (full-day recordings):** Pre-compute and store all masks to disk after segmentation; do not re-run the segmenter during reconstruction.

---

## Integration Points

### External Dependencies

| Dependency | Integration Pattern | Notes |
|------------|---------------------|-------|
| AquaCal (calibration library) | Import at startup; load JSON config; expose `RefractiveProjector` objects | Provides both ray casting (2D→3D) and forward projection (3D→2D) |
| scikit-image | `skimage.morphology.skeletonize` for medial axis extraction | Lightweight; CPU only |
| scipy | `scipy.ndimage.distance_transform_edt` (widths), `scipy.interpolate.splprep` (B-spline), `scipy.optimize.least_squares` (LM refiner), `scipy.optimize.linear_sum_assignment` (Hungarian) | Core dependency for reconstruction and tracking |
| ultralytics | YOLOv8 detection inference | GPU inference; model weights loaded at startup |
| PyTorch | U-Net segmentation inference; RefractiveProjector (forward projection) | Required for segmentation; optional for reconstruction |
| SAM2 | Offline pseudo-label generation only; not in the live inference path | Training-time dependency only |

**No longer required for primary pipeline:** PyTorch3D (shelved rendering), Detectron2 (replaced by U-Net), filterpy (no EKF).

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Phase I → Phase II | `dict[cam_id, list[DetectionResult]]` — masks + bboxes per camera | Masks stay on CPU |
| Phase II → Phase III | `identity_map: dict[(cam_id, det_id), fish_id]` + `centroids_3d: dict[fish_id, (3,) array]` | Identity resolved before midline extraction |
| Phase III → Phase IV | `dict[fish_id, dict[cam_id, (N, 3) array]]` — arc-length sampled 2D midlines | Fixed N ensures cross-view correspondence |
| Phase IV → Phase V (optional) | `dict[fish_id, SplineResult]` — control points + width profile | Used as warm-start for LM refiner |
| Phase IV/V → Phase VI | `dict[fish_id, FishTrajectoryRecord]` — spline controls, width, metadata | Pure data; no library-specific types |
| CalibrationLoader → all phases | `CameraModel` objects with `project()` and `cast_ray()` methods | Shared read-only; no mutable state after initialization |

---

## Shelved Architecture: Analysis-by-Synthesis

The original AquaPose architecture used differentiable mesh rendering with Adam optimization: a parametric fish mesh (midline spline + swept ellipse cross-sections) was rendered via PyTorch3D's rasterizer after pre-projecting vertices through the refractive model, and silhouette IoU loss drove gradient-based optimization of the fish state vector {p, ψ, κ, s}. This approach was architecturally sound but took 30+ minutes per second of video, making it impractical for the target dataset (hours of multi-fish recordings).

The mesh building code (`mesh/`), differentiable renderer (`rendering/`), and Adam optimizer (`optimization/`) are retained in the codebase. They may be useful as:
- A high-fidelity refinement alternative to the LM refiner
- A validation tool (render reconstructed splines as meshes to visually verify)
- A reference for the width profile model (swept ellipse cross-sections)

See `.planning/inbox/proposed_pipeline.md` for the original analysis-by-synthesis design and `.planning/inbox/fish-reconstruction-pivot.md` for the pivot rationale.

---

## Sources

- AquaPose pivot proposal (authoritative): `.planning/inbox/fish-reconstruction-pivot.md` — HIGH confidence (project owner document)
- AquaPose original pipeline spec: `.planning/inbox/proposed_pipeline.md` — HIGH confidence (project owner document)
- Refractive underwater camera calibration (2024): [https://arxiv.org/abs/2405.18018](https://arxiv.org/abs/2405.18018) — MEDIUM confidence
- Multi-view 3D pose estimation with triangulation + iterative refinement patterns (CVPR 2024): [MVGFormer](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.pdf) — MEDIUM confidence
- Slender-body midline tracking: Butail & Paley 2012, Voesenek et al. 2016 — MEDIUM confidence (confirms arc-length correspondence assumption)
- scikit-image skeletonize: [https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize) — HIGH confidence (official docs)

---

*Architecture research for: AquaPose — 3D fish pose estimation via direct multi-view triangulation*
*Researched: 2026-02-21*
