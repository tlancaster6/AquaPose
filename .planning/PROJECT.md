# AquaPose

## What This Is

AquaPose is a 3D fish pose estimation system that reconstructs the position, orientation, and body shape of cichlids from a 13-camera aquarium rig. Built as an event-driven computation engine with strict 3-layer architecture (Core Computation → PosePipeline → Observers), the pipeline executes 5 stages — Detection (YOLO-OBB), Pose Estimation (YOLO-pose with 6 keypoints), 2D Tracking (custom OKS-based keypoint tracker per-camera), Association (ray-ray scoring + Leiden clustering), and Reconstruction (confidence-weighted DLT triangulation + B-spline fitting) — producing dense 3D trajectories and midline kinematics for behavioral research. The keypoint tracker uses a 24-dim Kalman filter over keypoint positions and velocities, with OKS cost, OCM direction consistency, ORU/OCR recovery mechanisms, and cubic spline gap interpolation. Precomputed refractive lookup tables (forward pixel→ray and inverse voxel→pixel) eliminate per-frame refraction math during association. Per-stage pickle caching enables offline evaluation via `aquapose eval` (multi-stage quality reports) and `aquapose tune` (parameter sweeps with two-tier validation). Invoked via `aquapose run --config path.yaml` with production, diagnostic, synthetic, and benchmark execution modes.

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

- ✓ Vectorized association scoring via NumPy broadcasting replacing per-pair Python loop (ASSOC-01, ASSOC-02) — v3.4
- ✓ Vectorized DLT reconstruction via batched `torch.linalg.lstsq` replacing per-body-point loop (RECON-01, RECON-02) — v3.4
- ✓ Background-thread frame prefetch in ChunkFrameSource (FIO-01, FIO-02) — v3.4
- ✓ Batched YOLO inference for detection and midline stages with OOM retry (BATCH-01 through BATCH-04) — v3.4
- ✓ End-to-end performance validation: 8.2x total speedup confirmed (VAL-01 through VAL-03) — v3.4

- Z-denoising: centroid z-flattening + temporal Gaussian smoothing for clean 3D reconstructions (ZDenoisingConfig) -- v3.5
- Prep infrastructure: `aquapose prep calibrate-keypoints` and `aquapose prep generate-luts` CLIs with fail-fast enforcement -- v3.5
- Pseudo-label generation: Source A (consensus reprojection) and Source B (gap-fill) with confidence scoring in YOLO format -- v3.5
- Gap detection: inverse LUT visibility cross-referencing with failure reason tagging (no-detection, no-tracklet, failed-midline) -- v3.5
- Frame selection and dataset assembly: temporal subsampling, curvature-diversity sampling, pooled assembly with confidence thresholds -- v3.5
- Training run management: timestamped run dirs, `aquapose train compare` with cross-run metrics and provenance tracking -- v3.5
- Elastic midline deformation augmentation: TPS-based C-curve/S-curve generation reducing curvature bias (OKS slope -0.71 to -0.30) -- v3.5
- SQLite sample store: content-hash dedup, provenance tracking, symlink-based dataset assembly, model lineage with config auto-update -- v3.5
- Workflow-oriented CLI: project-aware path resolution (`--project`), run shorthand (latest, timestamp, negative index), deprecated command removal -- v3.5

- ✓ Extended evaluation metrics: percentiles (reprojection, confidence, camera count), per-keypoint breakdown, curvature-stratified quality, track fragmentation -- v3.6
- ✓ Data store bootstrap: temporal split, tagged exclusions, baseline model training and registration -- v3.6
- ✓ Pseudo-label iteration loop: generate, diversity-select, CVAT correct, retrain, evaluate with A/B curation comparison -- v3.6
- ✓ eval-compare CLI for round-over-round pipeline metric comparison with directional highlighting -- v3.6
- ✓ Training module consolidation: unified train_yolo(), shared _run_training() orchestrator, seg registration fix -- v3.6
- ✓ Final validation: 5-minute pipeline run with round 1 models, overlay mosaic, metrics comparison report -- v3.6

- ✓ Occlusion investigation: GO recommendation — no keypoint identity jumps, no OBB merging, confidence threshold 0.25 — v3.7
- ✓ Production OBB and Pose models retrained with all-source stratified data (OBB +7.4pts, Pose +0.6pts mAP50-95) — v3.7
- ✓ Pipeline reordered: Detection → Pose → Tracking → Association → Reconstruction — v3.7
- ✓ Segmentation midline backend fully removed (backends/segmentation.py, skeletonization, orientation) — v3.7
- ✓ Custom OKS-based keypoint tracker: 24-dim KF, OCM direction, ORU/OCR recovery, gap interpolation, chunk handoff — v3.7
- ✓ OC-SORT/BoxMot dependency completely removed — v3.7
- ✓ Tracker tuned to 27 tracks (vs OC-SORT 30) with 95% coverage on benchmark clip — v3.7
- ✓ Zero type errors, 1,159 tests passing, full CLI smoke test — v3.7

### Active

## Current Milestone: v3.8 Improved Association

**Goal:** Replace single-centroid ray scoring with multi-keypoint association, add swap-aware group validation and singleton recovery, and tune on real data to reduce the ~27% singleton rate.

**Target features:**
- Multi-keypoint pairwise scoring (6 keypoints per detection, confidence-filtered)
- Group validation with temporal changepoint detection for upstream ID swap splitting
- Singleton recovery with swap-aware split-and-assign
- Fragment merging and full refinement removed (simplification)
- Parameter tuning pass with real-data evaluation against v3.7 baseline

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

### Current State (v3.7 shipped)

- **Codebase:** 29,525 LOC source across `src/aquapose/` (calibration, core/, engine/, io, evaluation, training, visualization)
- **Architecture:** Event-driven 3-layer — Core Computation (5 stages) -> PosePipeline (orchestrator) -> Observers (3: console, timing, diagnostic). ChunkOrchestrator sits above PosePipeline managing chunk loop, identity stitching, and HDF5 output.
- **Pipeline order:** Detection (YOLO-OBB) -> Pose Estimation (YOLO-pose, 6 keypoints) -> 2D Tracking (custom OKS keypoint tracker) -> Association (Leiden) -> Reconstruction (DLT triangulation + B-spline + z-denoising)
- **Tracker:** Custom `KeypointTracker` with 24-dim KF (6 kpts x 2D pos+vel), OKS cost matrix, OCM direction consistency, ORU/OCR recovery, cubic spline gap interpolation, chunk handoff via serialized KF state. Configurable via `TrackingConfig` (base_r, lambda_ocm, max_gap_frames, match_cost_threshold, ocr_threshold, det_thresh, max_age).
- **Chunk processing:** ChunkOrchestrator processes video in fixed-size temporal chunks (default 200 frames). ChunkHandoff carries tracker state + identity map across boundaries. Per-chunk HDF5 flush with global frame offsets.
- **Performance:** 8.2x total pipeline speedup (914s -> 112s per chunk). Batched YOLO inference (detection 11.5x, pose 8.1x), background-thread frame prefetch, vectorized DLT reconstruction (7.0x), vectorized association scoring (3.8x). OOM retry with automatic batch halving.
- **Tech stack:** Python 3.11, PyTorch, PyTorch3D, scikit-image, OpenCV, h5py, ultralytics (YOLO), Click (CLI), Plotly (3D viz), scipy (CubicSpline), leidenalg/igraph, hatch build system
- **Reconstruction:** Single DLT backend -- vectorized confidence-weighted triangulation with outlier rejection (threshold=10.0), B-spline fitting (7 control points), z-denoising via centroid z-flattening + temporal Gaussian smoothing. Reads 6-keypoint poses and interpolates to 15-point midlines.
- **Evaluation:** Per-chunk pickle caching (chunk_NNN/cache.pkl + manifest.json), five typed stage evaluators, `aquapose eval` (quality reports), `aquapose tune` (parameter sweeps with two-tier validation), `aquapose viz` (overlay, animation, trails from cached data)
- **Pseudo-labeling:** Source A (consensus reprojection) + Source B (gap-fill) pseudo-labels with confidence scoring, elastic augmentation for curvature bias reduction
- **Training infrastructure:** `aquapose train {obb, pose}` CLI subcommands; SQLite sample store with content-hash dedup, symlink-based assembly, model lineage tracking
- **CLI:** Workflow-oriented command groups (`run`, `eval`, `viz`, `tune`, `data`, `prep`, `pseudo-label`, `train`), project-aware path resolution (`--project`), run shorthand (latest, timestamp, negative index)
- **Core organization:** Shared types in `core/types/`, implementations in `core/<stage>/`, legacy top-level dirs eliminated
- **Production models:** OBB (mAP50-95=0.781, run_20260310_115419), Pose (mAP50-95=0.974, run_20260310_171543) — retrained with all-source stratified data
- **Tracking metrics:** 27 tracks on benchmark clip (vs 9-fish target, vs OC-SORT 30), 95% detection coverage, 0 gaps, continuity=1.000
- **Known limitation:** Z-reconstruction uncertainty 132x larger than XY due to top-down camera geometry; ~27% singleton rate in association; ASSOC-01 keypoint centroid not active in current tracker (deferred to TRACK-V2-04)
- **Import boundary:** Automated AST-based checker enforced via pre-commit hook -- core/ never imports engine/ at runtime

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
| Vectorized association scoring (NumPy broadcasting) | Per-pair Python loop is the bottleneck; broadcasting eliminates it | ✓ Good — 3.8x speedup, numerically identical |
| Vectorized DLT via batched lstsq (not per-point SVD) | C camera loops only, no body-point loop; single lstsq call for all N points | ✓ Good — 7.0x speedup, within 1e-4 m equivalence |
| Drop 2-camera ray-angle filter in vectorized DLT | Masking per-point would require a body-point loop, defeating vectorization | ✓ Good — negligible yield impact |
| Background-thread frame prefetch (not multiprocessing) | IPC overhead for large frame arrays; daemon thread + bounded queue is simpler | ✓ Good — eliminates seek overhead and GPU idle gaps |
| Queue maxsize=2 for prefetch buffer | Balances memory (2 frames x 12 cameras ~144MB) vs prefetch benefit | ✓ Good — sufficient overlap without memory pressure |
| Mutable BatchState for OOM retry | Needs to persist batch size reductions across calls within a chunk | ✓ Good — effective adaptive behavior |
| batch_size=0 means no limit | Simplest default — send all inputs in one call | ✓ Good — works well for typical 12-camera setup |
| CPU crop extraction separated from GPU batch predict | Clean OOM retry boundary — only retry the GPU call, not crop extraction | ✓ Good — correct retry granularity |
| GPU non-determinism accepted for batched inference | 1-detection delta cascading through stages is inherent to batched vs serial GPU execution | ✓ Good — not an algorithmic regression |

| Centroid z-flattening over IRLS plane fit | Simpler approach; z-denoising goal achieved without full plane projection | ✓ Good -- z-noise cleaned for pseudo-label quality |
| Confidence composite: 50% residual + 30% camera + 20% variance | Empirically balanced pseudo-label quality scoring | ✓ Good -- enables threshold-based filtering |
| SQLite sample store over directory-based management | Content-hash dedup, provenance tracking, reproducible assembly via symlinks | ✓ Good -- eliminates ad-hoc directory management |
| Elastic TPS deformation for curvature augmentation | C-curve/S-curve variants reduce straight-fish training bias | ✓ Good -- OKS slope improved -0.71 to -0.30 |
| Workflow-oriented CLI over module-oriented | `--project` resolution replaces per-command `--config`; run shorthand | ✓ Good -- cleaner UX, less boilerplate |
| Source priority upsert (manual > corrected > pseudo) | Higher-quality labels always win in dedup | ✓ Good -- correct precedence |
| Skip round 2, accept round 1 models | All primary metrics improved (singleton -12.5%, p50 reproj -28.4%); diminishing returns expected | ✓ Good -- shipped faster without quality compromise |
| eval-compare as top-level CLI command | Avoids refactoring eval into a group; keeps format_comparison_table out of __init__ to prevent collision | ✓ Good -- clean CLI surface |
| Consolidated train_yolo() replacing 3 wrappers | Eliminates duplication; seg registration bug fixed by shared _run_training() orchestrator | ✓ Good -- 3 files deleted, single entry point |
| CVAT for pseudo-label curation (not Label Studio) | CVAT has better OBB editing tools; Label Studio integration built but not used for final workflow | — Pending |
| A/B curation comparison as standard practice | Light human curation (+9.2pts mAP50-95 on held-out data) justified the CVAT review step | ✓ Good -- quantified value of human oversight |

---
| Pose before tracking (not after) | Keypoints needed for OKS-based tracker; reorder eliminates AnnotatedDetection wrapper | ✓ Good — simpler data flow, Detection enriched in-place |
| Custom OKS tracker over OC-SORT/BoxMot | OC-SORT uses IoU on OBBs which fails under occlusion; OKS on keypoints is anatomically meaningful | ✓ Good — 27 tracks vs OC-SORT 30, with better cost model |
| 24-dim KF state (not 60-dim) | 6 kpts x (x,y,vx,vy) = 24; conceptual "60-dim" was over-specification | ✓ Good — tractable state dimension |
| BoxMot fully removed (not retained as fallback) | Single tracker backend simplifies maintenance; OC-SORT can be reimplemented if needed | ✓ Good — ~1,200 lines removed, zero external tracker dependency |
| Segmentation backend removed (not kept selectable) | YOLO-pose is sole midline source; segmentation/skeletonization path unused since v3.0 | ✓ Good — cleaner codebase, no dead backends |
| BYTE-style secondary pass deferred (TRACK-10) | Coverage 93.6% above 90% trigger; root cause is occlusion reacquisition, not low-conf misses | ✓ Good — avoided premature complexity |
| ASSOC-01 keypoint centroid deferred | Implemented in ocsort_wrapper, deleted with BoxMot; fix would be throwaway before TRACK-V2-04 | — Deferred to next milestone |
| Bidirectional merge removed | Forward+backward merge added complexity without reducing fragmentation (44 vs 42 tracks) | ✓ Good — simpler single-pass architecture |

---
*Last updated: 2026-03-11 after v3.8 Improved Association milestone started*
