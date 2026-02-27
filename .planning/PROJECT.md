# AquaPose

## What This Is

AquaPose is a 3D fish pose estimation system that reconstructs the position, orientation, and body shape of cichlids from a 13-camera aquarium rig. Built as an event-driven computation engine with strict 3-layer architecture (Core Computation → PosePipeline → Observers), the pipeline executes 5 stages — Detection (YOLO), Midline (U-Net + skeletonization), Association (RANSAC centroid clustering), Tracking (Hungarian 3D), and Reconstruction (multi-view triangulation + B-spline fitting) — producing dense 3D trajectories and midline kinematics for behavioral research. A curve-based optimizer provides an alternative correspondence-free reconstruction backend. Invoked via `aquapose run --config path.yaml` with production, diagnostic, synthetic, and benchmark execution modes.

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

- ✓ Event-driven pipeline architecture (3-layer: Core Computation → PosePipeline → Observers) — v2.0
- ✓ Stage Protocol interface with strongly typed PipelineContext — v2.0
- ✓ Frozen dataclass configuration system with YAML + CLI overrides — v2.0
- ✓ Structured lifecycle event system — v2.0
- ✓ Observer-based diagnostics, timing, visualization, and export (5 observers) — v2.0
- ✓ CLI entrypoint (`aquapose run`) as thin wrapper over PosePipeline — v2.0
- ✓ Clean-room stage migrations preserving numerical equivalence (5 stages) — v2.0
- ✓ Execution modes via configuration (production, diagnostic, synthetic, benchmark) — v2.0
- ✓ Golden data verification framework and regression test suite — v2.0
- ✓ AST-based import boundary checker with pre-commit enforcement — v2.0

### Active

<!-- v2.1 Identity — pipeline reorder for robust cross-view correspondence -->

- [ ] Precomputed refractive lookup tables (forward + inverse) for fast association
- [ ] OC-SORT 2D tracking per camera replacing 3D bundle-claiming tracker
- [ ] Cross-camera tracklet association via ray-ray scoring, Leiden clustering, 3D refinement
- [ ] Pipeline reordering: Detection → 2D Tracking → Association → Midline → Reconstruction
- [ ] Tracklet and association diagnostic visualization

## Current Milestone: v2.1 Identity

**Goal:** Reorder the pipeline to track in 2D first, then associate tracklets across cameras using trajectory-level geometric evidence — fixing the root cause of broken 3D reconstruction.

**Target features:**
- Precomputed refractive LUTs (pixel→ray, voxel→pixel) eliminating per-frame refraction math
- OC-SORT per-camera 2D tracking replacing 3D bundle-claiming tracker
- Cross-camera tracklet association (pairwise scoring, graph clustering, 3D refinement)
- Pipeline integration (new PipelineContext fields, deferred midline, reconstruction from tracklet groups)
- Diagnostic visualization for tracklets and associations

### Out of Scope

- Merge-and-split interaction handling — future milestone
- Sex classification — deferred
- Full-day recording processing — future milestone (v1 targets 5–30 min clips)
- Real-time processing — batch only
- Fin segmentation — body-only masks
- AquaKit centralized library — backburnered; AquaCal is the dependency, AquaMVS is reference
- Voxel carving initialization fallback — epipolar consensus is sufficient
- Mobile or web interface — CLI/script-based pipeline
- Offline mode / edge deployment — lab workstation only
- Pydantic for config — frozen dataclasses already decided and shipped in v2.0

## Context

### Current State (v2.0 Alpha shipped)

- **Codebase:** 18,660 LOC source + 14,826 LOC tests across `src/aquapose/` (calibration, core/, engine/, segmentation, tracking, reconstruction, io, visualization)
- **Architecture:** Event-driven 3-layer — Core Computation (5 stages) → PosePipeline (orchestrator) → Observers (5 side-effect handlers)
- **Tech stack:** Python 3.11, PyTorch, PyTorch3D, scikit-image, OpenCV, h5py, ultralytics (YOLO), Click (CLI), Plotly (3D viz), hatch build system
- **Test suite:** 514 unit tests, 7 regression tests (data-dependent), 1 E2E smoke test
- **Two reconstruction backends:** Triangulation (primary, fast) and curve optimizer (experimental, correspondence-free) — selected via config
- **Segmentation quality:** U-Net IoU 0.623 — sufficient for pipeline but noisy 2D midlines are the primary quality bottleneck for real data
- **Known limitation:** Z-reconstruction uncertainty 132x larger than XY due to top-down camera geometry
- **Import boundary:** Automated AST-based checker enforced via pre-commit hook — core/ never imports engine/ at runtime

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
| Stage Protocol via structural typing (not ABC) | typing.Protocol with runtime_checkable — no inheritance required | ✓ Good — clean 5-stage architecture |
| Frozen dataclasses for config (not Pydantic) | Simpler, stdlib-only, hierarchical nesting | ✓ Good — defaults→YAML→CLI→freeze works well |
| PipelineContext in core/, not engine/ | Pure data contracts belong in core/ layer | ✓ Good — resolved IB-003 violations |
| Observers as event subscribers (not stage hooks) | Zero coupling to stages; fault-tolerant dispatch | ✓ Good — adding/removing observers has no effect on computation |
| Port behavior, not rewrite logic | Numerical equivalence is the acceptance bar | ✓ Good — golden data framework validates |
| Canonical 5-stage model (not 7) | Detection, Midline, Association, Tracking, Reconstruction — aligned to guidebook | ✓ Good — simplified planning and implementation |
| TrackingStage consumes Stage 3 bundles | Stage 3 is hard dependency; bundles-aware backend | ✓ Good — clean data flow, no re-derivation |
| Import boundary via AST checker + pre-commit | Automated enforcement prevents architectural regression | ✓ Good — 0 violations at milestone completion |

---
*Last updated: 2026-02-27 after v2.1 Identity milestone started*
