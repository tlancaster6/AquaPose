# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Accurate single-fish 3D reconstruction from multi-view silhouettes via differentiable refractive rendering
**Current focus:** Phase 3 — Fish Mesh Model and 3D Initialization

## Current Position

Phase: 3 of 6 (Fish Mesh Model and 3D Initialization)
Plan: 1 of 2 in current phase
Status: Executing
Last activity: 2026-02-20 — Completed 03-01: Parametric fish mesh with pytorch3d Meshes

Progress: [████░░░░░░] 45%

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 17 min
- Total execution time: 1.7 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-calibration-and-refractive-geometry | 2 | 50 min | 25 min |
| 02-segmentation-pipeline | 3 | 35 min | 12 min |
| 03-fish-mesh-model-and-3d-initialization | 1 | 10 min | 10 min |

**Recent Trend:**
- Last 5 plans: 5 min, 8 min, 12 min, 15 min, 10 min
- Trend: stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: AquaCal is numpy-based — refractive projection must be reimplemented in PyTorch (AquaCal used only for loading calibration JSON, not for forward projection)
- [Init]: Phase 3 (Fish Mesh) depends only on Phase 1, not Phase 2 — can develop in parallel with segmentation if calendar time matters
- [Init]: Temporal smoothness loss (RECON-02) built in Phase 4 but only activates in Phase 5 when tracking provides associations — build the hook, wire it in Phase 5
- [Init]: All APIs are batch-first (list of fish states) from day one — even Phase 4 single-fish code uses single-element lists
- [01-01]: Cross-validation compares against AquaCal NumPy (not AquaMVS PyTorch) — AquaMVS not importable in hatch env due to missing open3d/lightglue; AquaCal NumPy is the actual ground truth
- [01-01]: K_inv float32 inversion tolerance set to atol=1e-4 — float32 with fx=1400 produces ~6e-5 error, expected float32 precision not a bug
- [01-02]: Z/XY anisotropy is 132x mean (30x-577x range) at 0.5px noise — Phase 4 optimizer should weight Z loss approximately 100x smaller than XY
- [01-02]: build_synthetic_rig uses water_z = height_above_water (0.75m) since AquaCal places cameras at world Z=0 with Z increasing downward into water
- [02-01]: Shadow exclusion via threshold at 254 (MOG2 outputs 127 for shadows, 255 for foreground)
- [02-01]: Detection.mask is full-frame sized (not cropped to bbox) to feed directly into SAM as mask prompt
- [02-02]: SAM2 predictor lazily loaded on first predict() call to avoid GPU allocation on import
- [02-02]: Label Studio uses its own RLE variant (mask2rle) not pycocotools RLE
- [02-03]: torchvision maskrcnn_resnet50_fpn_v2 instead of Detectron2 (unmaintained, Windows-incompatible)
- [02-03]: Custom collate_fn with tuple(zip(*batch)) for Mask R-CNN's list-of-dicts format
- [03-01]: torch.sinc used for sin(kappa*t*s)/kappa stability at kappa=0 — sinc(x/pi)=sin(x)/x, smooth everywhere
- [03-01]: miropsota pytorch3d-0.7.9+pt2.9.1cu128 works with torch 2.10+cu130 on Windows (CUDA mesh ops not needed in Phase 3)
- [03-01]: Watertight winding: tube (v0,v2,v1)+(v1,v2,v3), head cap (apex,j_next,j), tail cap (apex,j,j_next) — all edges shared exactly 2x
- [03-01]: Spine centered at t=0.5 midpoint; builder translates by state.p — keeps spine generation position-independent

### Pending Todos

None.

### Blockers/Concerns

- [Phase 1 - RESOLVED]: Z-uncertainty budget quantified: Z error is 132x larger than XY for top-down 13-camera rig (see docs/reports/z_uncertainty_report.md)
- [Phase 2]: MOG2 female recall under worst-case conditions (stationary, low contrast) not yet measured — most likely operational failure mode
- [Phase 4]: PyTorch3D sigma/gamma hyperparameters for this rig's fish pixel sizes unknown — empirical sweep needed during Phase 4 development

## Session Continuity

Last session: 2026-02-20
Stopped at: Completed 03-01: Parametric fish mesh model
Resume file: .planning/phases/03-fish-mesh-model-and-3d-initialization/03-01-SUMMARY.md
