# AquaPose

## What This Is

AquaPose is a 3D fish pose estimation system that uses analysis-by-synthesis to reconstruct the position, orientation, and body shape of cichlids in a multi-camera aquarium rig. It fits a parametric fish mesh to multi-view silhouettes through differentiable refractive rendering, producing dense 3D trajectories and midline kinematics for behavioral research.

## Core Value

Accurate single-fish 3D reconstruction from multi-view silhouettes via differentiable refractive rendering — if this doesn't work, nothing downstream matters.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

- [ ] Load calibration data from AquaCal (intrinsics, extrinsics, refractive model)
- [ ] Refractive projection (3D→pixel) and ray casting (pixel→ray) in PyTorch, with depth output
- [ ] Background subtraction detection (MOG2) with bounding box output
- [ ] SAM-based pseudo-label generation from bounding box prompts
- [ ] Label Studio annotation pipeline integration
- [ ] Instance segmentation model (Mask R-CNN on crops) trained on corrected annotations
- [ ] Parametric fish mesh: midline spline + swept cross-sections, differentiable in PyTorch
- [ ] Free cross-section mode for self-calibrating body shape profiles from data
- [ ] Epipolar consensus 3D initialization from coarse keypoints
- [ ] Differentiable silhouette rendering via PyTorch3D with refractive projection
- [ ] Multi-objective loss function (silhouette, gravity prior, morphological constraint, temporal smoothness)
- [ ] Cross-view holdout validation framework
- [ ] Single-fish pose optimization on real data with validated accuracy

### Out of Scope

- Multi-fish tracking and identity assignment — future milestone
- Merge-and-split interaction handling — future milestone
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

1. MOG2 background subtraction → bounding boxes across all frames
2. Sample ~100 frames per camera (~1300 total annotated frames across 13 cameras)
3. Feed bounding boxes as prompts to SAM (single-frame) → pseudo-label masks
4. Import images + pseudo-labels into Label Studio for human correction
5. Most frames are confirm-and-skip; effort on bad boundaries and merged fish

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
| Analysis-by-synthesis over keypoint triangulation | Dense silhouette constraints are more robust under refraction than sparse point correspondences | — Pending |
| AquaCal as dependency, AquaMVS as reference | Avoid fragile cross-repo imports; reimplement refractive projection in AquaPose | — Pending |
| MOG2 detection before learned detector | Simpler, faster, may be sufficient; YOLO fallback if recall < 95% | — Pending |
| Epipolar consensus only (no voxel carving) | Simpler initialization; voxel carving is expensive and rarely needed with warm-start | — Pending |
| Self-calibrated cross-section profiles | Data-driven is more accurate than literature values; needs free-parameter mesh mode | — Pending |
| Cross-view holdout for validation | Avoids need for manual 3D ground truth; uses camera geometry as its own validation | — Pending |
| Pre-project vertices then PyTorch3D rasterize | Compose refractive projection with standard rasterizer; need to add depth output | — Pending |

---
*Last updated: 2026-02-19 after initialization*
