# Stack Research

**Domain:** Multi-view 3D fish pose estimation via direct triangulation with refractive ray casting
**Researched:** 2026-02-19 | **Updated:** 2026-02-21 (reconstruction pipeline pivot)
**Confidence:** HIGH (primary pipeline has no exotic build dependencies; all choices are stable PyPI packages)

> **Pivot note (2026-02-21):** AquaPose pivoted from analysis-by-synthesis (differentiable mesh rendering + Adam optimization) to a direct triangulation pipeline. The primary pipeline uses skeletonization, spline fitting, multi-view triangulation, and Levenberg-Marquardt refinement — no differentiable rendering. The old analysis-by-synthesis pipeline is shelved but retained; its additional dependencies are listed in a dedicated section below.

---

## Critical Version Warning

~~PyTorch is currently at **2.10.0** (January 2026). PyTorch3D's latest release (v0.7.9, November 2024) officially supports only up to **PyTorch 2.4.x**.~~

**Post-pivot (2026-02-21):** The primary pipeline has **no PyTorch3D dependency**. PyTorch is used only for U-Net segmentation inference and YOLO detection — both work on any recent PyTorch version. **PyTorch can be upgraded freely.** The version-pinning constraint (PyTorch 2.4.1) applies only if you need to run the shelved analysis-by-synthesis pipeline.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11 | Runtime | Project already targets >=3.11; stable, widely supported. 3.12+ is also fine for the primary pipeline. |
| PyTorch | latest stable | U-Net inference, YOLO detection, tensor ops | No longer pinned to 2.4.1 — primary pipeline has no PyTorch3D coupling. Use latest stable for best performance. Only pin to 2.4.1 if using shelved pipeline. |
| torchvision | match PyTorch | Paired with PyTorch | Follows PyTorch version; needed for image transforms in segmentation. |
| CUDA | 12.x | GPU acceleration | U-Net and YOLO inference only. No PyTorch3D CUDA ops in primary pipeline, so any CUDA version supported by your PyTorch is fine. |
| scikit-image | >=0.22 | Skeletonization, morphology, distance transform | Core of the reconstruction pipeline: `skimage.morphology.skeletonize` extracts midlines from masks, `distance_transform_edt` provides width profiles, morphological ops clean masks. |
| scipy | >=1.13 | Spline fitting, LM refinement, Hungarian assignment | `splprep`/`splev` for midline splines, `least_squares` (Levenberg-Marquardt) for refractive triangulation refinement, `linear_sum_assignment` for frame-to-frame fish tracking. |
| numpy | >=2.0 | Array operations, calibration interface | Foundation for all non-tensor computation. AquaCal is numpy-based. |

### Perception Pipeline Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| opencv-python | 4.13.x | MOG2 background subtraction, video I/O, 2D overlays | Detection (Stage 1), video frame reading, 2D visualization overlays. Use headless variant (`opencv-python-headless`) for server environments. |
| ultralytics (YOLOv8) | >=8.3 | Primary fish detector | Fine-tuned YOLOv8-det on project data. Primary detector — MOG2 is fallback/bootstrap only. |
| U-Net (custom) | — | Binary mask segmentation on crops | Custom MobileNetV3-Small encoder, ~2.5M params, trained on SAM2 pseudo-labels. Replaces Mask R-CNN / Detectron2. Weights at `unet/best_model.pth`. |
| sam2 (segment-anything-2) | latest from source | Zero-shot pseudo-label generation via box prompts | Offline annotation tool only — not deployed at inference time. Uses crop+box-only approach (no mask prompt). |
| supervision | >=0.24 | Annotation format conversion, mask utilities | Convert between annotation formats. Utility library for glue code between tools. |

### Reconstruction Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-image | >=0.22 | Skeletonize masks, morphological cleanup, distance-based width profiles | Every frame: extract midline skeleton from binary mask, compute width via `distance_transform_edt`, prune skeleton branches. |
| scipy | >=1.13 | Spline fitting (`splprep`), LM refinement (`least_squares`), Hungarian tracking (`linear_sum_assignment`) | Every frame: fit smooth spline to skeleton points, refine 3D triangulation via Levenberg-Marquardt through refractive model, associate fish across frames. |
| numpy | >=2.0 | Array operations, refractive ray casting, calibration data | Foundation for all geometric computation. Refractive projection code operates on numpy arrays. |

### Tracking and Association Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy (`linear_sum_assignment`) | >=1.13 | Hungarian algorithm for frame-to-frame fish association | Primary tracking solution. Solves assignment between predicted and observed fish positions each frame. |

### Storage and Visualization Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| h5py | >=3.11 | HDF5 trajectory data storage | Primary output format. Efficient random access to per-fish, per-frame data. |
| pandas | >=2.2 | Tabular analysis, per-frame statistics | Analysis workflows downstream of HDF5 storage. Not used in the hot path. |
| pyarrow | >=17 | Parquet export for columnar analysis | Optional secondary export for users who prefer columnar formats over HDF5. |
| rerun-sdk | >=0.22 | Real-time 3D visualization of trajectories, skeletons | Primary debugging and QA tool. Synchronized multi-camera 2D + 3D visualization. |
| pyvista | >=0.44 | Publication-quality 3D renders | Offline, publication-quality renders. Not needed until Phase V polish. |
| matplotlib | >=3.9 | Trajectory plots, loss curves, analysis figures | Standard scientific plotting throughout all phases. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| hatch | Project management, virtual envs, test/lint/typecheck scripts | Already configured in `pyproject.toml`. Use `hatch run test` / `hatch run lint` throughout. |
| ruff | Linting and formatting | Already configured. Line length 88, targets py311. |
| basedpyright | Type checking | Already configured in basic mode. Tighten as codebase matures. |
| pytest | Testing | Already configured with `slow` marker for skipping GPU-dependent tests in CI. |

---

## Shelved Pipeline Dependencies

The following libraries are required **only** for the shelved analysis-by-synthesis pipeline (differentiable mesh rendering + Adam optimization). They are not needed for the primary triangulation pipeline.

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| PyTorch | **2.4.1** (pinned) | Autograd for differentiable rendering | Must be pinned to 2.4.1 for PyTorch3D 0.7.9 compatibility. |
| torchvision | 0.19.1 | Paired with PyTorch 2.4.1 | Must match PyTorch version exactly. |
| CUDA | 12.1 | GPU acceleration for PyTorch3D CUDA ops | Supported by PyTorch 2.4.1; CUDA 12.6+ requires PyTorch >=2.6 which breaks PyTorch3D. |
| PyTorch3D | 0.7.9 (source) | Differentiable silhouette rendering, mesh rasterization | Install from source (`pip install -e .`) against PyTorch 2.4.1. Do not delete the cloned directory. |
| kornia | >=0.7 | Differentiable IoU loss (Lovasz hinge), image morphological ops | Pure PyTorch, no CUDA extension compile needed. |
| filterpy | 1.4.4 | Extended Kalman Filter for 3D fish tracking | Stable but unmaintained. EKF is simple enough to reimplement with scipy if needed. |
| detectron2 | latest from source | Mask R-CNN instance segmentation | Replaced by custom U-Net in primary pipeline. Install from `git+https://github.com/facebookresearch/detectron2.git`. |
| scikit-learn | >=1.5 | Sex classification (color histogram features) | Deferred to v2. Not needed for reconstruction. |

---

## Installation

### Primary Pipeline

```bash
# Step 1: Create environment with Python 3.11+
# Using hatch (preferred):
pip install hatch
hatch env create

# Or using conda:
conda create -n aquapose python=3.11
conda activate aquapose

# Step 2: Install PyTorch (latest stable with CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install SAM2 from source (for pseudo-label generation only)
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
# Download model checkpoints separately (see sam2 README)

# Step 4: Install remaining dependencies
pip install \
  opencv-python-headless \
  scikit-image \
  scipy \
  numpy \
  h5py \
  pandas \
  pyarrow \
  rerun-sdk \
  pyvista \
  matplotlib \
  supervision \
  "ultralytics>=8.3"

# Step 5: Install AquaCal (internal dependency)
# pip install -e /path/to/aquacal
```

### Shelved Pipeline (additional dependencies)

Only install these if you need to run the analysis-by-synthesis pipeline.

```bash
# Pin PyTorch to 2.4.1 (REPLACES latest PyTorch from primary install)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch3D from source (editable install — do not delete the directory)
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..

# Install Detectron2 from source
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install additional libraries
pip install kornia filterpy scikit-learn
```

> **Warning:** Installing the shelved pipeline downgrades PyTorch to 2.4.1, which affects the primary pipeline. Use a separate virtual environment if you need both.

---

## Alternatives Considered

| Recommended | Alternative | Status Post-Pivot |
|-------------|-------------|-------------------|
| Direct triangulation + LM refinement | PyTorch3D analysis-by-synthesis | Shelved. Direct triangulation is simpler, faster, and avoids PyTorch3D build issues. Analysis-by-synthesis retained as fallback if triangulation accuracy is insufficient. |
| Custom U-Net (MobileNetV3-Small) | Detectron2 Mask R-CNN | **Resolved.** U-Net is lighter, faster, and sufficient for binary mask segmentation on crops. Detectron2 dependency eliminated. |
| Custom U-Net | Ultralytics YOLOv8-seg | U-Net trained on SAM2 pseudo-labels gives adequate masks on crops. YOLOv8-seg is an option if U-Net quality degrades. |
| scipy `linear_sum_assignment` | filterpy EKF | **Resolved.** Simple Hungarian assignment is sufficient for frame-to-frame tracking. EKF shelved with analysis-by-synthesis pipeline. |
| SAM2 | SAM (v1) | SAM2 adds video propagation for consistent pseudo-labels across frames. Use SAM2. |
| rerun-sdk | Open3D visualization | rerun's synchronized timeline + multi-stream playback is better for debugging multi-view reconstruction. |
| scikit-image skeletonize | Custom midline extraction | scikit-image's `skeletonize` is well-tested and handles branching; writing a custom version is unnecessary. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Gaussian Splatting renderers (3DGS, SkelSplat) | Primary pipeline does not use rendering at all. Even for the shelved pipeline, Gaussians cannot represent a parametric mesh with controllable topology. | Direct triangulation (primary) or PyTorch3D mesh rasterizer (shelved). |
| NeRF-based rendering | No rendering in primary pipeline. For shelved pipeline, NeRF is orders of magnitude slower than rasterization. | Direct triangulation (primary) or PyTorch3D (shelved). |
| Open3D for voxel carving | Heavy dependency with its own CUDA requirements. Not needed — primary pipeline uses triangulation, not volumetric methods. | scipy + numpy for any volumetric fallback. |
| TensorFlow / JAX | Project is PyTorch-native. Segmentation (U-Net), detection (YOLO), and refractive projection code are all PyTorch/numpy. | PyTorch + numpy throughout. |
| COLMAP / photogrammetry pipelines | AquaCal already provides refraction-aware calibration with sub-millimeter accuracy. COLMAP does not support refractive camera models. | AquaCal (existing internal library). |
| torchsparse / MinkowskiEngine | These are for learning on point clouds / voxel grids. AquaPose uses geometric reconstruction, not a learned 3D representation. | scikit-image + scipy for geometric reconstruction. |

---

## Stack Patterns by Variant

**If U-Net segmentation quality degrades on new cameras:**
- Retrain on expanded SAM2 pseudo-labels from the new camera
- Consider switching to `segmentation-models-pytorch` U-Net++ if custom U-Net architecture is limiting
- Ensure crops are 128x128 and use the shared crop utilities in `src/aquapose/segmentation/crop.py`

**If YOLO detection misses females (low contrast):**
- Augment training data with more female examples
- Lower confidence threshold and rely on downstream mask quality filtering
- MOG2 remains available as a complementary detector for stationary-camera scenarios

**If triangulation accuracy is insufficient (>5mm reprojection error):**
- Check refractive model calibration — most error comes from interface normal/distance
- Increase number of camera views used in triangulation
- Consider reactivating shelved analysis-by-synthesis pipeline for difficult cases

**If scipy `least_squares` LM refinement diverges:**
- Tighten bounds on the 3D search space using multi-view consensus
- Use robust loss (`loss='soft_l1'`) to handle outlier views
- Verify refractive projection Jacobians numerically

**For Windows development:**
- Primary pipeline installs cleanly on Windows via standard pip (no source builds required)
- PyTorch3D source builds on Windows (shelved pipeline only) require Visual Studio Build Tools — use WSL2 or a Linux VM

---

## Version Compatibility

### Primary Pipeline

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| PyTorch (latest) | torchvision (matching) | Standard pairing; no special constraints |
| PyTorch (latest) | ultralytics >=8.3 | ultralytics tracks PyTorch releases |
| PyTorch (latest) | SAM2 (main branch) | SAM2 targets current PyTorch |
| scikit-image >=0.22 | numpy >=2.0 | scikit-image 0.22+ supports numpy 2.x |
| scipy >=1.13 | numpy >=2.0 | scipy 1.13+ supports numpy 2.x |
| Python 3.11+ | All listed libraries | Safe baseline; 3.12 also works for primary pipeline |

### Shelved Pipeline (additional constraints)

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| PyTorch 2.4.1 | PyTorch3D 0.7.9 | CONFIRMED via official INSTALL.md |
| PyTorch 2.4.1 | torchvision 0.19.1 | Must match exactly |
| PyTorch 2.4.1 | detectron2 (main branch) | Source install from main |
| PyTorch 2.4.1 | kornia >=0.7 | Pure PyTorch; no version coupling |
| CUDA 12.1 | PyTorch 2.4.1 | Officially supported combination |
| Python 3.11 | PyTorch3D source build | Confirmed; 3.12 untested for PyTorch3D |

---

## Sources

- scikit-image morphology docs — skeletonize, distance_transform_edt: https://scikit-image.org/docs/stable/api/skimage.morphology.html
- scipy interpolate docs — splprep, splev for spline fitting: https://docs.scipy.org/doc/scipy/reference/interpolate.html
- scipy optimize docs — least_squares (Levenberg-Marquardt): https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
- scipy linear_sum_assignment — Hungarian algorithm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
- PyTorch3D INSTALL.md — version compatibility matrix: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- PyTorch3D releases page — v0.7.9 (November 2024): https://github.com/facebookresearch/pytorch3d/releases
- PyTorch official get-started — current stable: https://pytorch.org/get-started/locally/
- OpenCV releases — v4.13.0 (February 2026): https://opencv.org/releases/
- SAM2 GitHub — video segmentation, pseudo-label generation: https://github.com/facebookresearch/sam2
- rerun-sdk PyPI — active development: https://pypi.org/project/rerun-sdk/
- ultralytics docs — YOLOv8 detection and training: https://docs.ultralytics.com/
- AquaPose fish-reconstruction-pivot.md — pivot rationale: see `.planning/inbox/fish-reconstruction-pivot.md`

---

*Stack research for: 3D fish pose estimation via direct triangulation with refractive ray casting*
*Researched: 2026-02-19 | Updated: 2026-02-21*
