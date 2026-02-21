# AquaPose

## What This Is

AquaPose is a 3D fish pose estimation system that reconstructs the position, orientation, and body shape of cichlids from a multi-camera aquarium rig. The primary pipeline extracts 2D medial axes from segmentation masks, establishes cross-view correspondences via arc-length normalization, triangulates 3D midline points through a refractive camera model, and fits continuous 3D splines — producing dense 3D trajectories and midline kinematics for behavioral research. An alternative analysis-by-synthesis pipeline (differentiable mesh rendering + optimization) is shelved but retained for advanced use.

## Core Value

Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation — if this doesn't work, nothing downstream matters.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Completed

- [x] Load calibration data from AquaCal (intrinsics, extrinsics, refractive model)
- [x] Refractive projection (3D→pixel) and ray casting (pixel→ray) in PyTorch
- [x] Background subtraction detection (MOG2) + YOLOv8 alternative detector
- [x] SAM2-based pseudo-label generation from bounding box prompts (box-only, no mask prompt)
- [x] U-Net segmentation on cropped detections (replaced Mask R-CNN; best val IoU: 0.623)
- [x] Parametric fish mesh: midline spline + swept cross-sections, differentiable in PyTorch

### Active (Direct Triangulation Pipeline)

- [ ] Cross-view identity association via RANSAC centroid ray clustering
- [ ] Persistent 3D fish tracking via Hungarian assignment on 3D centroids
- [ ] 2D medial axis extraction from masks (skeletonize + longest-path BFS)
- [ ] Arc-length normalized resampling for cross-view correspondence
- [ ] Multi-view triangulation with RANSAC and view-angle weighting
- [ ] 3D spline fitting from triangulated midline points
- [ ] Per-frame trajectory output (spline control points, width profile, centroid) in HDF5
- [ ] 2D overlay visualization (reprojected midline on camera views)

### Shelved (Analysis-by-Synthesis Pipeline)

- [x] Differentiable silhouette rendering via PyTorch3D with refractive projection
- [x] Multi-objective loss function (silhouette, gravity prior, morphological constraint)
- [x] Single-fish pose optimization via Adam with warm-start
- [x] Cross-view holdout validation framework
- Shelved 2026-02-21: functionally complete but 30+ min/sec runtime is impractical

### Out of Scope

- Merge-and-split interaction handling — future milestone
- Sex classification — deferred to v2
- Full-day recording processing — future milestone (v1 targets 5–30 min clips)
- Real-time processing — batch only
- Fin segmentation — body-only masks
- AquaKit centralized library — backburnered; AquaCal is the dependency, AquaMVS is reference
- Voxel carving initialization fallback — epipolar consensus is sufficient
- Mobile or web interface — CLI/script-based pipeline

## Context

### Rig Geometry

- 13 cameras: 12 in a ring at ~0.6m radius + 1 center, mounted ~1m above a cylindrical tank (2m diameter, 1m tall)
- All cameras oriented straight down through a flat water surface (air-water interface, no glass)
- 30 fps, synchronized
- 1600×1200 resolution
- 25–50° best triangulation angle; 3–5 camera coverage everywhere; X-Y strong, Z weaker

### Subjects

- 9 cichlids (3 male, 6 female), ~10cm body length
- Clear water with controlled diffuse lighting
- Females are low-contrast against background — a known segmentation challenge
- Males and females may differ in body shape (separate cross-section profiles may be needed)

### Library Ecosystem

- **AquaCal**: Numpy-based refractive calibration library (dependency — import for calibration loading)
- **AquaMVS**: PyTorch-based multi-view stereo reconstruction (reference — port refractive projection code)
- **AquaPose**: This repo — 3D pose estimation via analysis-by-synthesis

### Calibration

- Refraction-aware, sub-millimeter reprojection error
- Flat air-water interface (no glass), single Snell's law boundary
- Calibration data stored as JSON, loaded via AquaCal

### Annotation Workflow

1. YOLO detection (or MOG2 fallback) → bounding boxes across all frames
2. Feed bounding boxes as box-only prompts to SAM2 → pseudo-label masks with quality filtering
3. Train U-Net on pseudo-labels directly (no manual annotation step — Label Studio removed)
4. U-Net inference on detection crops → binary masks for downstream reconstruction

### Cross-Section Profile Self-Calibration

Species-specific cross-section profiles are self-calibrated from data rather than literature:
- Select frames where a fish is straight-bodied, well-isolated, and visible in 5+ cameras
- Run pose optimization with morphological prior disabled, per-cross-section height/width as free parameters
- Average recovered profiles across multiple fish and frames
- Validate against published cichlid morphometrics as sanity check
- Fit separate male/female profiles if shapes differ

### Refractive Projection (from AquaMVS reference)

- `project` (3D→pixel): Decomposes geometry into radial distances, Newton-Raphson (10 fixed iterations) for Snell's law, then pinhole projection. Currently returns (pixels, valid) — **needs depth output added** for PyTorch3D rasterizer integration.
- `cast_ray` (pixel→ray): Pinhole back-projection, ray-plane intersection at water surface, vectorized Snell's law for refracted direction.
- Design choices for differentiability: fixed iteration count, torch.clamp, epsilon terms, torch.where for validity masking.

## Constraints

- **Dependency**: AquaCal library must be importable (calibration data loading)
- **Hardware**: GPU required for PyTorch3D rendering and pose optimization
- **Data**: Real multi-camera recordings and calibration data already available
- **Processing**: Batch mode only; v1 targets 5–30 minute clips
- **Long-term**: Pipeline must eventually scale to full-day recordings (streaming/checkpointing), but not in v1

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Direct triangulation over analysis-by-synthesis | Analysis-by-synthesis (30+ min/sec) impractical; medial axis + triangulation is orders of magnitude faster with acceptable quality | Decided 2026-02-21 |
| AquaCal as dependency, AquaMVS as reference | Avoid fragile cross-repo imports; reimplement refractive projection in AquaPose | Decided |
| YOLO as primary detector, MOG2 as fallback | YOLOv8n trained on 150 frames; recall 0.78, sufficient for pipeline | Decided |
| RANSAC centroid clustering for cross-view identity | Cast refractive rays from 2D centroids, triangulate minimal subsets, score consensus | Decided 2026-02-21 |
| Arc-length normalized correspondence | Slender-body assumption: normalized arc-length along 2D midline projection approximately preserves cross-view correspondence | Decided 2026-02-21 |
| U-Net over Mask R-CNN for segmentation | Lightweight (~2.5M params), trains on SAM2 pseudo-labels, best val IoU 0.623 | Decided |
| Cross-view holdout for validation | Avoids need for manual 3D ground truth; uses camera geometry as its own validation | Decided |
| Analysis-by-synthesis retained as optional route | Shelved, not deleted — available for advanced work requiring mesh-level reconstruction | Decided 2026-02-21 |

---
*Last updated: 2026-02-21 — Reconstruction pivot: direct triangulation replaces analysis-by-synthesis as primary pipeline*
