# Stack Research

**Domain:** Multi-view 3D animal pose estimation via analysis-by-synthesis with differentiable refractive rendering
**Researched:** 2026-02-19
**Confidence:** MEDIUM (PyTorch3D compatibility gap is a real risk; all other choices are HIGH)

---

## Critical Version Warning

PyTorch is currently at **2.10.0** (January 2026). PyTorch3D's latest release (v0.7.9, November 2024) officially supports only up to **PyTorch 2.4.x**. There is a confirmed 5-version gap. Building PyTorch3D from source against PyTorch 2.5–2.10 is possible but requires manual compilation and an editable install that must not be deleted. This is the single largest installation risk in the stack. **Pin PyTorch to 2.4.1 for initial development, then evaluate upgrading once PyTorch3D publishes compatibility updates.**

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11 | Runtime | Project already targets ≥3.11; stable, widely supported by all listed libraries. Do not use 3.13 yet — PyTorch3D source builds are untested there. |
| PyTorch | 2.4.1 | Autograd, tensor ops, optimizer | The newest version officially supported by PyTorch3D 0.7.9. Pinning here avoids source-build pain for all team members. Upgrade to 2.x+ once PyTorch3D follows. |
| torchvision | 0.19.1 | Paired with PyTorch 2.4.1 | Follows PyTorch version; needed for Detectron2 and feature transforms. Must match PyTorch version exactly. |
| CUDA | 12.1 | GPU acceleration | Supported by PyTorch 2.4.1; broadly available on modern NVIDIA cards. CUDA 12.6+ requires PyTorch ≥2.6 which breaks PyTorch3D. |
| PyTorch3D | 0.7.9 (source) | Differentiable silhouette rendering, mesh rasterization | The only production-grade differentiable rasterizer with SoftSilhouetteShader, MeshRasterizer, and PyTorch-native mesh structures. Silhouette loss via `pytorch3d.renderer` is the exact primitive AquaPose needs. No other library combines this with clean PyTorch mesh ops. Install from source (`pip install -e .`) against PyTorch 2.4.1. |

### Perception Pipeline Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| opencv-python | 4.13.x | MOG2 background subtraction, video I/O, 2D overlays | All detection (Stage 1), video frame reading, 2D visualization overlays. Latest stable; use headless variant (`opencv-python-headless`) for server environments. |
| detectron2 | latest from source | Mask R-CNN instance segmentation | Stage 2 segmentation after annotation. Install from `git+https://github.com/facebookresearch/detectron2.git` — there is no published PyPI wheel. Requires matching PyTorch 2.4.1. |
| sam2 (segment-anything-2) | latest from source | Zero-shot pseudo-label generation via point/box prompts | Annotation bootstrap only — not deployed at inference time. Use video mode for temporal propagation across frames, significantly reducing annotation burden vs. single-frame SAM. |
| ultralytics (YOLOv8) | >=8.3 | Detection fallback if MOG2 recall < 95% | Only deploy if MOG2 fails the 95% recall acceptance threshold. A fine-tuned YOLOv8-det model on project annotations. |
| supervision | >=0.24 | Annotation format conversion, mask utilities | Convert between COCO, Detectron2, and Label Studio annotation formats. Roboflow's utility library — actively maintained, fills the glue-code gap between annotation tools and model training. |

### Differentiable Pose Reconstruction Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| kornia | >=0.7 | Differentiable IoU loss, image morphological ops | Use `kornia.losses.lovasz_hinge_loss` for differentiable binary mask IoU during silhouette optimization. Avoids writing custom differentiable IoU from scratch. Pure PyTorch — no CUDA extension compile needed. |
| scipy | >=1.13 | Epipolar consensus via `least_squares`, Hungarian algorithm via `linear_sum_assignment`, spline generation | Phase II initialization (ray intersection), Phase IV tracking association. Also needed for any non-differentiable numerical work (profile fitting). |
| numpy | >=2.0 | Array operations, calibration data handling, AquaCal interface | Foundation for all non-tensor computation. AquaCal is numpy-based — data flows from AquaCal into numpy then into PyTorch tensors. |

### Tracking and Association Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| filterpy | 1.4.4 | Extended Kalman Filter for 3D fish tracking | Phase IV tracking. Stable, no active development, but the EKF API is complete and suitable. If filterpy maintenance is a concern, the EKF is simple enough to reimplement in ~100 lines using scipy. |
| scikit-learn | >=1.5 | Sex classification (color histogram features, simple classifier) | Phase IV identity disambiguation. No heavy modeling here — a linear SVM or random forest on HSV histograms is sufficient. |

### Storage and Visualization Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| h5py | >=3.11 | HDF5 trajectory data storage | Primary output format. Supports efficient random access to per-fish, per-frame data across arbitrarily long recordings. Essential when recordings scale beyond RAM. |
| pandas | >=2.2 | Tabular analysis, per-frame statistics | Analysis workflows downstream of HDF5 storage. Not used in the hot path. |
| pyarrow | >=17 | Parquet export for columnar analysis | Optional secondary export for users who prefer columnar formats over HDF5. |
| rerun-sdk | >=0.22 | Real-time 3D visualization of fish meshes, trajectories | Primary debugging and QA tool during development. Supports synchronized multi-camera 2D + 3D visualization. Far faster to integrate than PyVista for interactive debugging. |
| pyvista | >=0.44 | Publication-quality 3D mesh renders | Offline, publication-quality renders. Not needed until Phase V polish. |
| matplotlib | >=3.9 | Trajectory plots, loss curves, analysis figures | Standard scientific plotting throughout all phases. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| hatch | Project management, virtual envs, test/lint/typecheck scripts | Already configured in `pyproject.toml`. Use `hatch run test` / `hatch run lint` throughout. |
| ruff | Linting and formatting | Already configured. Line length 88, targets py311. |
| basedpyright | Type checking | Already configured in basic mode. Tighten as codebase matures. |
| pytest | Testing | Already configured with `slow` marker for skipping GPU-dependent tests in CI. |
| Label Studio | Human-in-the-loop annotation correction | Self-hosted or cloud. Accepts COCO JSON format for import/export. Use `supervision` for format conversion. |

---

## Installation

```bash
# Step 1: Create environment with Python 3.11 (via hatch or conda)
# If using conda for PyTorch3D build compatibility:
conda create -n aquapose python=3.11
conda activate aquapose

# Step 2: Install PyTorch 2.4.1 with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install PyTorch3D from source (MUST stay on editable install — do not delete the directory)
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..

# Step 4: Install Detectron2 from source
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Step 5: Install SAM2 from source
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
# Download model checkpoints separately (see sam2 README for checkpoint URLs)

# Step 6: Install remaining dependencies
pip install \
  opencv-python-headless \
  kornia \
  scipy \
  numpy \
  filterpy \
  scikit-learn \
  h5py \
  pandas \
  pyarrow \
  rerun-sdk \
  pyvista \
  matplotlib \
  supervision \
  "ultralytics>=8.3"

# Step 7: Install AquaCal (internal dependency — adjust path or install method as appropriate)
# pip install -e /path/to/aquacal
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| PyTorch3D (SoftSilhouetteShader) | nvdiffrast | If silhouette rendering performance becomes a bottleneck. nvdiffrast is faster but provides hard rasterization gradients only at boundaries, not the soft probabilistic blending needed for analysis-by-synthesis. PyTorch3D's soft rasterizer is the right primitive for silhouette-fitting from scratch. |
| PyTorch3D | DEODR | DEODR has excellent boundary differentiability but minimal ecosystem support and requires C compilation. PyTorch3D's CUDA ops are faster for the mesh sizes in AquaPose (~1000–2000 vertices). |
| PyTorch3D | kaolin (NVIDIA) | Kaolin is actively maintained by NVIDIA and has silhouette support via `kaolin.metrics.render.mask_iou`. It could replace PyTorch3D if PyTorch3D becomes unmaintained, but the API is less documented for the specific analysis-by-synthesis use case. |
| Detectron2 (Mask R-CNN) | Ultralytics YOLOv8-seg | YOLOv8-seg is faster and simpler to train but less accurate on small objects with precise mask boundaries. For 80–100px fish on crops, Mask R-CNN's RoI-Align gives better boundary accuracy. Use YOLOv8-det (detection only) as MOG2 fallback, not for segmentation. |
| SAM2 | SAM (v1) | SAM2 adds video propagation, which is directly useful for generating consistent pseudo-labels across many frames of the same fish. SAM v1 is single-frame only. Use SAM2 for pseudo-label generation. |
| filterpy EKF | pykalman | pykalman is more actively maintained but has a less ergonomic API for nonlinear (extended) filters. If filterpy causes issues with Python 3.12+, reimplement the EKF directly using scipy and numpy — it is ~80 lines. |
| rerun-sdk | Open3D visualization | Open3D has 3D mesh rendering but lacks rerun's synchronized timeline + multi-stream playback. For debugging pose estimates alongside camera frames, rerun's time-indexed logging is substantially more useful. |
| kornia (differentiable IoU) | Custom PyTorch IoU | Writing differentiable IoU from scratch via soft blending is ~30 lines, but kornia's Lovász hinge loss is numerically better-behaved and directly optimizes the IoU surrogate. Prefer kornia unless a dependency conflict arises. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| PyTorch > 2.4.x (initially) | PyTorch3D 0.7.9 officially supports only up to 2.4.x. Community reports confirm 2.6+2.8 source builds are finicky and require source patches. Introducing this debt before the core pipeline works will cause debugging confusion. | PyTorch 2.4.1 pinned until PyTorch3D publishes a new release with ≥2.5 support. |
| Gaussian Splatting renderers (3DGS, SkelSplat) | Gaussian splatting renders differentiable alpha-composited point clouds, not meshes. AquaPose needs a parametric mesh with controllable topology (swept cross-sections, midline spline). Gaussians cannot represent this parameterization. | PyTorch3D mesh rasterizer. |
| NeRF-based rendering | NeRF requires per-scene training and is orders of magnitude slower than rasterization for the analysis-by-synthesis optimization loop running 50–100 iterations per frame. | PyTorch3D differentiable rasterization. |
| Open3D for voxel carving (Phase II) | Open3D is a heavy dependency with its own CUDA requirements. The voxel carving fallback in Phase II is only triggered rarely (first frame of each track after track loss). Implement voxel carving with numpy + PyTorch directly to avoid the dependency. | numpy + PyTorch for any voxel carving fallback. |
| TensorFlow / JAX | Project is PyTorch-native end to end. Detectron2, PyTorch3D, SAM2, kornia, and the refractive projection code (from AquaMVS) are all PyTorch. Mixed-framework gradient flow is not possible. | PyTorch throughout. |
| COLMAP / photogrammetry pipelines | AquaCal already provides refraction-aware calibration with sub-millimeter accuracy. COLMAP does not support refractive camera models and would produce worse calibration. | AquaCal (existing internal library). |
| torchsparse / MinkowskiEngine (sparse 3D convolutions) | These are for learning on point clouds / voxel grids. AquaPose uses a parametric mesh, not a learned 3D representation. They add significant compilation complexity with no benefit. | PyTorch3D mesh structures. |

---

## Stack Patterns by Variant

**If PyTorch3D build fails against your CUDA version:**
- Build PyTorch3D from the `stable` branch tag: `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"`
- Check `pytorch3d/INSTALL.md` for per-CUDA-version workarounds
- As a last resort, downgrade to CUDA 11.8 + PyTorch 2.4.1, which has the most pre-built wheel availability

**If MOG2 detection recall < 95% for females (expected risk):**
- Train YOLOv8-det (`ultralytics`) on 500+ annotated frames
- Deploy as replacement for MOG2, keeping output format identical (bounding boxes)
- Do not use YOLOv8-seg here — segmentation happens downstream in Mask R-CNN on crops

**If Mask R-CNN boundary quality is insufficient (IoU < 0.90):**
- Add PointRend head to Detectron2 config (~25% inference overhead, significant boundary improvement)
- Alternative: Train a binary U-Net on crops using `segmentation-models-pytorch`; lighter and faster if Detectron2 is overkill

**If filterpy fails on Python 3.12+:**
- Reimplement EKF directly: `scipy.linalg.solve` for the Kalman update; takes ~80 lines
- No external EKF dependency is strictly needed given the simple constant-velocity motion model

**For Windows development:**
- PyTorch3D source builds on Windows require Visual Studio Build Tools and may need PyTorch header patches (see GitHub issues). Use WSL2 with CUDA passthrough or a Linux VM for the initial build.
- All other libraries install cleanly on Windows via standard pip.

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| PyTorch 2.4.1 | PyTorch3D 0.7.9 | CONFIRMED via official INSTALL.md |
| PyTorch 2.4.1 | torchvision 0.19.1 | CONFIRMED via pytorch/vision releases |
| PyTorch 2.4.1 | detectron2 (main branch) | Detectron2 tracks latest PyTorch; source install from main is the supported method |
| PyTorch 2.4.1 | SAM2 (main branch) | SAM2 is Meta-maintained alongside Detectron2; main branch targets current PyTorch |
| PyTorch 2.4.1 | kornia ≥0.7 | kornia is pure PyTorch; no version coupling |
| Python 3.11 | All listed libraries | Safe baseline; PyTorch3D source builds are confirmed on 3.11 |
| Python 3.12 | PyTorch3D | Untested for source builds; filterpy may also have issues. Do not use 3.12 until confirmed. |
| CUDA 12.1 | PyTorch 2.4.1 | Officially supported combination |

---

## Sources

- PyTorch3D INSTALL.md — version compatibility matrix, installation instructions: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- PyTorch3D releases page — v0.7.9 release date (November 28, 2024): https://github.com/facebookresearch/pytorch3d/releases
- PyTorch official get-started — current stable PyTorch 2.10.0, CUDA support: https://pytorch.org/get-started/locally/
- PyTorch3D GitHub issue #1962 — confirmed source build workaround for PyTorch 2.8+CUDA 12.8: https://github.com/facebookresearch/pytorch3d/issues/1962
- pytorch3d renderer docs — SoftSilhouetteShader, MeshRasterizer, RasterizationSettings: https://pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/shader.html
- OpenCV releases — v4.13.0 is current stable (February 2026): https://opencv.org/releases/
- SAM2 GitHub — video segmentation, pseudo-label generation: https://github.com/facebookresearch/sam2
- Detectron2 GitHub — Mask R-CNN, PointRend, source install: https://github.com/facebookresearch/detectron2
- kornia losses docs — Lovász hinge loss for differentiable binary IoU: https://kornia.readthedocs.io/en/latest/losses.html
- rerun-sdk PyPI — v0.22.1, active 2025 development: https://pypi.org/project/rerun-sdk/0.22.1/
- filterpy GitHub — v1.4.4, EKF API stable: https://github.com/rlabbe/filterpy
- AquaPose proposed_pipeline.md — project-specific library selections already validated by author: internal document
- AquaPose detection_segmentation_brief.md — MOG2 → Detectron2 decision rationale: internal document

---

*Stack research for: 3D fish pose estimation via analysis-by-synthesis with differentiable refractive rendering*
*Researched: 2026-02-19*
