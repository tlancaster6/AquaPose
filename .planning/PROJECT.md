# AquaPose

## What This Is

AquaPose is a 3D fish pose estimation system that reconstructs the position, orientation, and body shape of cichlids from a 13-camera aquarium rig. Built as an event-driven computation engine with strict 3-layer architecture (Core Computation → PosePipeline → Observers), the pipeline executes 5 stages — Detection (YOLO-OBB), 2D Tracking (OC-SORT per-camera), Association (ray-ray scoring + Leiden clustering), Midline (YOLO-seg or YOLO-pose backends), and Reconstruction (confidence-weighted DLT triangulation + B-spline fitting) — producing dense 3D trajectories and midline kinematics for behavioral research. Precomputed refractive lookup tables (forward pixel→ray and inverse voxel→pixel) eliminate per-frame refraction math during association. Per-stage pickle caching enables offline evaluation via `aquapose eval` (multi-stage quality reports) and `aquapose tune` (parameter sweeps with two-tier validation). Invoked via `aquapose run --config path.yaml` with production, diagnostic, synthetic, and benchmark execution modes.

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

- ✓ Precomputed refractive lookup tables (forward pixel→ray + inverse voxel→pixel) for fast association (LUT-01, LUT-02) — v2.1
- ✓ OC-SORT 2D tracking per camera replacing 3D bundle-claiming tracker (TRACK-01) — v2.1
- ✓ Cross-camera tracklet association via ray-ray scoring, Leiden clustering, 3D refinement (ASSOC-01/02/03) — v2.1
- ✓ Pipeline reordering: Detection → 2D Tracking → Association → Midline → Reconstruction (PIPE-01/02/03) — v2.1
- ✓ Tracklet and association diagnostic visualization (DIAG-01) — v2.1
- ✓ E2E smoke tests on synthetic data (EVAL-01) — v2.1

- ✓ Custom U-Net and keypoint regression code stripped; Ultralytics-only codebase — v3.0
- ✓ YOLO-seg training wrapper with COCO polygon data converter (DATA-01, TRAIN-01) — v3.0
- ✓ YOLO-pose training wrapper with CLI subcommand (TRAIN-02) — v3.0
- ✓ SegmentationBackend (YOLO-seg + skeletonization) as selectable midline backend (PIPE-01) — v3.0
- ✓ PoseEstimationBackend (YOLO-pose + spline interpolation) as selectable midline backend (PIPE-02) — v3.0
- ✓ Config backend selection via midline.backend field (PIPE-03) — v3.0
- ✓ Standard YOLO txt+yaml training data format (STAB-01) — v3.0
- ✓ Consolidated weights_path config field (STAB-02) — v3.0
- ✓ Legacy dirs reorganized into core/ submodules with core/types/ package (REORG-01) — v3.0

- ✓ Diagnostic fixture system: MidlineFixture + NPZ serialization for offline evaluation (DIAG-01, DIAG-02) — v3.1
- ✓ Offline evaluation harness with CalibBundle, frame selection, Tier 1/Tier 2 metrics (EVAL-01 through EVAL-06) — v3.1
- ✓ Confidence-weighted DLT triangulation with outlier rejection, single strategy (RECON-01 through RECON-07) — v3.1
- ✓ DLT validated against baseline; outlier threshold tuned 50→10 (RECON-08) — v3.1
- ✓ Association parameter sweep infrastructure (ASSOC-01 through ASSOC-04) — v3.1
- ✓ Dead reconstruction code removed: old triangulation, curve optimizer, epipolar/orientation (CLEAN-01 through CLEAN-03) — v3.1

- ✓ Per-stage pickle caching with StaleCacheError and ContextLoader (INFRA-01 through INFRA-04) — v3.2
- ✓ Five typed stage evaluators with frozen metric dataclasses and DEFAULT_GRIDs (EVAL-01 through EVAL-05, TUNE-06) — v3.2
- ✓ `aquapose eval` CLI for multi-stage quality reports (EVAL-06, EVAL-07) — v3.2
- ✓ `aquapose tune` CLI with grid sweeps, two-tier validation, config diff output (TUNE-01 through TUNE-05) — v3.2
- ✓ Legacy evaluation machinery removed: harness.py, NPZ export, standalone scripts (CLEAN-01 through CLEAN-05) — v3.2
- ✓ Partial pipeline execution via --resume-from and initial_context (INFRA-02, INFRA-03) — v3.2

- ✓ FrameSource protocol + VideoFrameSource replacing direct VideoSet usage in stages (FRAME-01, FRAME-02, FRAME-03) — v3.3
- ✓ ChunkOrchestrator processing video in fixed-size temporal chunks above PosePipeline (CHUNK-01 through CHUNK-05) — v3.3
- ✓ ChunkHandoff carrying tracker state + identity map across chunk boundaries with atomic serialization (CHUNK-04, CHUNK-05) — v3.3
- ✓ Identity stitching mapping chunk-local fish IDs to globally consistent IDs (IDENT-01, IDENT-02) — v3.3
- ✓ Per-chunk HDF5 flush with global frame offsets; HDF5Observer removed (OUT-01, OUT-02) — v3.3
- ✓ CLI delegates to ChunkOrchestrator; degenerate and multi-chunk output correctness validated (INTEG-01, INTEG-02, INTEG-03) — v3.3
- ✓ Per-chunk diagnostic caches with manifest.json; EvalRunner/TuningOrchestrator chunk-aware — v3.3
- ✓ Visualization migrated from engine observers to `aquapose viz` CLI in evaluation suite — v3.3

### Active

## Current Milestone: v3.4 Performance Optimization

**Goal:** Reduce per-chunk pipeline processing time by optimizing the four profiled bottlenecks (inference batching, frame I/O, reconstruction, association), prioritized by impact-to-complexity ratio.

**Target features:**
- Batched YOLO inference for detection and midline stages (GPU utilization currently ~30%)
- Optimized frame I/O (video decoding ~12% of wall time)
- Vectorized DLT reconstruction across body points (~9% of wall time)
- Vectorized association scoring (~5% of wall time)

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

### Current State (v3.3 shipped)

- **Codebase:** 21,634 LOC source across `src/aquapose/` (calibration, core/, engine/, io, evaluation)
- **Architecture:** Event-driven 3-layer — Core Computation (5 stages) → PosePipeline (orchestrator) → Observers (3: console, timing, diagnostic). ChunkOrchestrator sits above PosePipeline managing chunk loop, identity stitching, and HDF5 output.
- **Pipeline order:** Detection (YOLO-OBB) → 2D Tracking (OC-SORT) → Association (Leiden) → Midline (YOLO-seg or YOLO-pose) → Reconstruction (DLT triangulation + B-spline)
- **Chunk processing:** ChunkOrchestrator processes video in fixed-size temporal chunks (default 1000 frames). ChunkHandoff carries tracker state + identity map across boundaries. Per-chunk HDF5 flush with global frame offsets.
- **Tech stack:** Python 3.11, PyTorch, PyTorch3D, scikit-image, OpenCV, h5py, ultralytics (YOLO), Click (CLI), Plotly (3D viz), boxmot (OC-SORT), leidenalg/igraph, hatch build system
- **Midline backends:** SegmentationBackend (YOLO-seg + skeletonization) and PoseEstimationBackend (YOLO-pose + spline), selectable via `midline.backend` config field
- **Reconstruction:** Single DLT backend — confidence-weighted triangulation with outlier rejection (threshold=10.0), B-spline fitting (7 control points)
- **Evaluation:** Per-chunk pickle caching (chunk_NNN/cache.pkl + manifest.json), five typed stage evaluators, `aquapose eval` (quality reports), `aquapose tune` (parameter sweeps with two-tier validation), `aquapose viz` (overlay, animation, trails from cached data)
- **Training infrastructure:** `aquapose train {yolo-obb, seg, pose}` CLI subcommands with standard YOLO txt+yaml data format
- **Core organization:** Shared types in `core/types/`, implementations in `core/<stage>/`, legacy top-level dirs eliminated
- **Known limitation:** Z-reconstruction uncertainty 132x larger than XY due to top-down camera geometry; ~70% singleton rate in association (upstream detection/tracking bottleneck)
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
- **Hardware**: GPU required for YOLO inference and PyTorch operations
- **Data**: Real multi-camera recordings and calibration data available
- **Processing**: Batch mode only; v1 targets 5-30 minute clips

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Direct triangulation over analysis-by-synthesis | ABS 30+ min/sec impractical; direct triangulation orders of magnitude faster | ✓ Good — primary pipeline works |
| AquaCal as dependency, AquaMVS as reference | Avoid fragile cross-repo imports; reimplement refractive projection | ✓ Good |
| YOLO as primary detector, MOG2 as fallback | YOLOv8n trained on 150 frames; recall 0.78 | ✓ Good |
| U-Net over Mask R-CNN for segmentation | Lightweight (~2.5M params), trains on SAM2 pseudo-labels | ✗ Replaced — IoU 0.623 insufficient; replaced by YOLO-seg in v3.0 |
| Custom keypoint regression for midline | U-Net encoder + regression head, per-point confidence | ✗ Replaced — poor performance even with augmentation; replaced by YOLO-pose in v3.0 |
| Ultralytics unification over custom models | Two custom U-Net models failed; Ultralytics provides pretrained backbones, battle-tested training, unified architecture | ✓ Good — v3.0 shipped, all 16 requirements satisfied |
| Standard YOLO txt+yaml over NDJSON | NDJSON was adopted mid-v3.0 then reverted; standard format has better tooling support | ✓ Good — all three training modes use txt+yaml |
| Legacy dirs reorganized into core/ submodules | reconstruction/, segmentation/, tracking/ had misleading names and cross-package imports | ✓ Good — core/types/ shared types, core/<stage>/ implementations |
| RANSAC centroid clustering for cross-view identity | Cast refractive rays, triangulate minimal subsets, score consensus | Superseded by Leiden clustering in v2.1 |
| Arc-length normalized correspondence | Slender-body assumption preserves cross-view correspondence | ✓ Good |
| Analysis-by-synthesis retained as optional route | Shelved, not deleted — available for advanced work | ✓ Good |
| Curve optimizer as alternative to triangulation | Correspondence-free B-spline fitting via chamfer distance | ✗ Removed v3.1 — must beat DLT baseline on eval harness to justify reintroduction |
| Reconstruction rebuild from minimal baseline | Both backends over-engineered, poor real-data results; rebuild with eval harness measuring every change | ✓ Good — DLT meets baseline, ~3,200 lines dead code removed |
| Pose estimation backend only for reconstruction | Ordered keypoints eliminate correspondence/orientation machinery in reconstruction | ✓ Good — v3.1 shipped, DLT is sole backend |
| Confidence-weighted DLT over RANSAC triangulation | Single strategy regardless of camera count; no branching, no orientation alignment | ✓ Good — simpler and matches baseline quality |
| Outlier rejection threshold 10.0 (not 50.0) | Empirical grid search on real data via evaluation harness | ✓ Good — best Tier 1 reprojection |
| NPZ fixtures for offline evaluation | Flat slash-separated keys for numpy.load compatibility; versioned (v1.0/v2.0) | ✓ Good — enables data-driven parameter tuning |
| Association params: keep defaults | Sweep showed marginal gains (~1% yield); ~70% singleton rate is upstream bottleneck | ✓ Good — no over-tuning |
| XY-only tracking cost matrix | Z uncertainty 132x larger; XY-only prevents Z-noise ID swaps | Superseded — OC-SORT per-camera in v2.1 |
| Population-constrained tracking | 9 fish always; dead tracks recycled to unmatched observations | Superseded — Leiden clustering handles identity in v2.1 |
| Stage Protocol via structural typing (not ABC) | typing.Protocol with runtime_checkable — no inheritance required | ✓ Good — clean 5-stage architecture |
| Frozen dataclasses for config (not Pydantic) | Simpler, stdlib-only, hierarchical nesting | ✓ Good — defaults→YAML→CLI→freeze works well |
| PipelineContext in core/, not engine/ | Pure data contracts belong in core/ layer | ✓ Good — resolved IB-003 violations |
| Observers as event subscribers (not stage hooks) | Zero coupling to stages; fault-tolerant dispatch | ✓ Good — adding/removing observers has no effect on computation |
| Port behavior, not rewrite logic | Numerical equivalence is the acceptance bar | ✓ Good — golden data framework validates |
| Canonical 5-stage model (not 7) | Detection, Midline, Association, Tracking, Reconstruction — aligned to guidebook | ✓ Good — reordered to Det→Track→Assoc→Mid→Recon in v2.1 |
| TrackingStage consumes Stage 3 bundles | Stage 3 is hard dependency; bundles-aware backend | Superseded — TrackingStage is now Stage 2, consumes detections directly |
| Pipeline reorder: track-first then associate | Frame-level RANSAC association failed; trajectory-level evidence needed | ✓ Good — root cause fix for broken 3D reconstruction |
| OC-SORT for 2D tracking (not Hungarian) | Per-camera independence, IoU+Kalman, handles occlusion via virtual trajectories | ✓ Good — robust coasting, clean state roundtrip |
| Leiden clustering for cross-camera association | Graph-based with must-not-link constraints; handles variable fish counts | ✓ Good — replaces population-constrained RANSAC |
| Precomputed LUTs over per-frame refraction | Forward+inverse LUTs eliminate ~ms/frame refraction math during association | ✓ Good — enables trajectory-level scoring at scale |
| Auto-generate LUTs on first pipeline run | No separate CLI subcommand; LUTs built lazily in AssociationStage | ✓ Good — zero setup friction |
| Import boundary via AST checker + pre-commit | Automated enforcement prevents architectural regression | ✓ Good — 0 violations at milestone completion |
| Per-stage pickle caching over monolithic NPZ | One file per StageComplete event; ContextLoader for sweep isolation | ✓ Good — enables efficient parameter sweeps |
| Stage evaluators with zero engine imports | Pipeline config passes as explicit function params | ✓ Good — clean separation |
| No automatic config mutation from tuning | Config diff block for manual application preserves reproducibility | ✓ Good — researcher reviews all changes |
| Legacy evaluation fully removed (not shimmed) | harness.py, NPZ export, 3 standalone scripts deleted | ✓ Good — no dead code maintenance |
| Two-tier validation for parameter sweeps | Fast sweep (few frames) → thorough top-N validation (many frames) | ✓ Good — balances speed and reliability |

---
| ChunkOrchestrator above PosePipeline for chunk loop | Fixed-size temporal chunks bound association complexity; orchestrator owns HDF5 output and identity stitching | ✓ Good — enables reliable long-video processing |
| ChunkHandoff in core/context.py (not engine/) | core/tracking/stage.py must construct it; core must not import from engine | ✓ Good — avoids circular import |
| Identity stitching via track ID continuity | Lightweight majority-vote using OC-SORT carry-forward track IDs | ✓ Good — simple, effective |
| Diagnostic + chunk mode co-existence | Originally mutual exclusion (Phase 53); removed in Phase 54 to support multi-chunk diagnostic runs | ✓ Good — per-chunk cache layout makes it work |
| Visualization migrated to eval suite | Overlay, animation, trails operate on cached data post-run instead of during pipeline execution | ✓ Good — decouples viz from pipeline; enables multi-chunk continuous output |
| Per-chunk single cache (not per-stage) | One cache.pkl per chunk containing full PipelineContext; simpler than per-stage files | ✓ Good — reduces file count, enables chunk-aware eval/tuning |

---
*Last updated: 2026-03-05 after v3.4 Performance Optimization milestone start*
