# AquaPose

## What This Is

AquaPose is a 3D fish pose estimation system that reconstructs the position, orientation, and body shape of cichlids from a 13-camera aquarium rig. The primary pipeline detects fish via YOLO/MOG2, segments them with U-Net trained on SAM2 pseudo-labels, extracts 2D midlines via skeletonization, associates identities across cameras with RANSAC centroid clustering, triangulates 3D midline points through a refractive camera model, and fits continuous B-splines — producing dense 3D trajectories and midline kinematics for behavioral research. A curve-based optimizer provides an alternative correspondence-free reconstruction method. An analysis-by-synthesis pipeline (differentiable mesh rendering) is shelved but retained.

## Core Value

Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation — if this doesn't work, nothing downstream matters.

## Requirements

### Validated

- ✓ Load calibration data from AquaCal (CALIB-01) — v1.0
- ✓ Differentiable refractive projection 3D→pixel and ray casting pixel→ray (CALIB-02, CALIB-03) — v1.0
- ✓ Z-reconstruction uncertainty quantified for 13-camera rig (CALIB-04) — v1.0 (132x Z/XY anisotropy)
- ✓ MOG2 detection with ≥95% recall + YOLOv8 alternative detector (SEG-01) — v1.0
- ✓ SAM2 pseudo-label generation from box prompts with quality filtering (SEG-02, SEG-03) — v1.0
- ✓ U-Net segmentation on cropped detections, IoU 0.623 (SEG-04, SEG-05) — v1.0
- ✓ Parametric fish mesh: spine + swept cross-sections, differentiable (MESH-01, MESH-02) — v1.0
- ✓ Epipolar initialization from coarse keypoints (MESH-03) — v1.0
- ✓ 2D medial axis extraction via skeletonization + BFS pruning (RECON-01) — v1.0
- ✓ Arc-length normalized resampling for cross-view correspondence (RECON-02) — v1.0
- ✓ Multi-view triangulation with RANSAC + view-angle weighting (RECON-03) — v1.0
- ✓ B-spline fitting with width profile (RECON-04) — v1.0
- ✓ Curve-based optimizer as alternative to triangulation (RECON-05) — v1.0
- ✓ RANSAC cross-view identity association (TRACK-01, TRACK-02) — v1.0
- ✓ Hungarian 3D tracking with population constraint (TRACK-03, TRACK-04) — v1.0
- ✓ HDF5 output with spline control points and metadata (OUT-01) — v1.0
- ✓ 2D reprojection overlay visualization (OUT-02) — v1.0
- ✓ 3D midline animation via matplotlib (OUT-03) — v1.0

### Shelved (Analysis-by-Synthesis Pipeline)

- ✓ Differentiable silhouette rendering (RECON-ABS-01) — v1.0 (shelved)
- ✓ Multi-objective loss function (RECON-ABS-02) — v1.0 (shelved)
- ✓ Single-fish pose optimization via Adam (RECON-ABS-03, RECON-ABS-04) — v1.0 (shelved)
- ✓ Cross-view holdout validation (RECON-ABS-05) — v1.0 (shelved)
- Shelved 2026-02-21: functionally complete but 30+ min/sec runtime is impractical

### Active

(Fresh for next milestone — define via `/gsd:new-milestone`)

### Out of Scope

- Merge-and-split interaction handling — future milestone
- Sex classification — deferred to v2
- Full-day recording processing — future milestone (v1 targets 5–30 min clips)
- Real-time processing — batch only
- Fin segmentation — body-only masks
- AquaKit centralized library — backburnered; AquaCal is the dependency, AquaMVS is reference
- Voxel carving initialization fallback — epipolar consensus is sufficient
- Mobile or web interface — CLI/script-based pipeline
- Offline mode / edge deployment — lab workstation only

## Context

### Current State (v1.0 shipped)

- **Codebase:** 50,802 LOC Python across `src/aquapose/` (calibration, core, segmentation, mesh, initialization, io, utils)
- **Tech stack:** Python 3.11, PyTorch, PyTorch3D, scikit-image, OpenCV, h5py, ultralytics (YOLO), hatch build system
- **Two reconstruction methods:** Direct triangulation (primary, fast) and curve optimization (experimental, correspondence-free)
- **Segmentation quality:** U-Net IoU 0.623 — sufficient for pipeline but noisy 2D midlines are the primary quality bottleneck for real data
- **Known limitation:** Z-reconstruction uncertainty 132x larger than XY due to top-down camera geometry

### Rig Geometry

- 13 cameras: 12 in a ring at ~0.6m radius + 1 center, mounted ~1m above a cylindrical tank (2m diameter, 1m tall)
- All cameras oriented straight down through a flat water surface (air-water interface, no glass)
- 30 fps, synchronized, 1600x1200 resolution
- 25-50 deg best triangulation angle; 3-5 camera coverage everywhere; X-Y strong, Z weaker

### Subjects

- 9 cichlids (3 male, 6 female), ~10cm body length
- Clear water with controlled diffuse lighting
- Females are low-contrast against background — a known segmentation challenge

### Library Ecosystem

- **AquaCal**: Numpy-based refractive calibration library (dependency — import for calibration loading)
- **AquaMVS**: PyTorch-based multi-view stereo reconstruction (reference only — not imported)
- **AquaPose**: This repo — 3D pose estimation

## Constraints

- **Dependency**: AquaCal library must be importable (calibration data loading)
- **Hardware**: GPU required for PyTorch3D and curve optimization
- **Data**: Real multi-camera recordings and calibration data available
- **Processing**: Batch mode only; v1 targets 5-30 minute clips

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Direct triangulation over analysis-by-synthesis | ABS 30+ min/sec impractical; direct triangulation orders of magnitude faster | ✓ Good — primary pipeline works |
| AquaCal as dependency, AquaMVS as reference | Avoid fragile cross-repo imports; reimplement refractive projection | ✓ Good |
| YOLO as primary detector, MOG2 as fallback | YOLOv8n trained on 150 frames; recall 0.78 | ✓ Good |
| U-Net over Mask R-CNN for segmentation | Lightweight (~2.5M params), trains on SAM2 pseudo-labels | ⚠️ Revisit — IoU 0.623 is bottleneck |
| RANSAC centroid clustering for cross-view identity | Cast refractive rays, triangulate minimal subsets, score consensus | ✓ Good |
| Arc-length normalized correspondence | Slender-body assumption preserves cross-view correspondence | ✓ Good |
| Analysis-by-synthesis retained as optional route | Shelved, not deleted — available for advanced work | ✓ Good |
| Curve optimizer as alternative to triangulation | Correspondence-free B-spline fitting via chamfer distance | — Pending real-data validation |
| XY-only tracking cost matrix | Z uncertainty 132x larger; XY-only prevents Z-noise ID swaps | ✓ Good |
| Population-constrained tracking | 9 fish always; dead tracks recycled to unmatched observations | ✓ Good |

---
*Last updated: 2026-02-25 after v1.0 milestone*
