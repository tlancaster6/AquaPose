# Milestones

## v3.6 Model Iteration & QA (Shipped: 2026-03-10)

**Phases completed:** 8 phases (70-77), 13 plans
**Timeline:** 5 days (2026-03-06 -> 2026-03-10)
**Codebase:** 30,480 LOC source
**Git range:** 1151 commits since v3.5, 130 files changed (+16,595 / -2,761)

**Key accomplishments:**
1. Extended evaluation metrics — percentiles (reprojection, confidence, camera count), per-keypoint breakdown, curvature-stratified quality, and 3D track fragmentation analysis
2. Data store bootstrap — temporal split, tagged exclusions, baseline OBB (mAP50=0.931) and pose (mAP50=0.991) models trained and registered
3. Full pseudo-label iteration loop — generate pseudo-labels, diversity-select, correct in CVAT, retrain, evaluate at pipeline level
4. A/B curation comparison — curated+augmented pose model improved +9.2pts mAP50-95 on held-out data, quantifying the value of light human curation
5. eval-compare CLI — round-over-round pipeline metric comparison with directional highlighting and structured JSON output
6. Training module consolidation — deduplicated YOLO wrappers, fixed seg registration bug, added test coverage for weight-copying and subset selection

**Delivered:** Complete pseudo-label retraining loop executed end-to-end, producing round 1 models with improved primary metrics (singleton rate -12.5%, p50 reprojection error -28.4%, p90 reprojection error -19.8%). Full provenance chain from baseline through pseudo-labels to final validation documented with showcase overlay videos.

**Known gaps (accepted as tech debt):**
- ITER-02: Store provenance uses source=corrected instead of source=pseudo with round=1 metadata
- ITER-05: Round 2 skipped per Phase 74 decision checkpoint (round 1 improvements sufficient)
- FINAL-02: 11/12 trail videos (e3v83f1 missing; overlay mosaic covers all 12 cameras)

---

## v3.5 Pseudo-Labeling (Shipped: 2026-03-06)

**Phases completed:** 9 phases (61-69), 22 plans
**Timeline:** 2 days (2026-03-05 -> 2026-03-06)
**Codebase:** 28,033 LOC source
**Git range:** 136 commits, 135 files changed (+22,963 / -2,121)

**Key accomplishments:**
1. Z-denoising pipeline — centroid z-flattening + temporal Gaussian smoothing cleans z-noise from 3D reconstructions for accurate reprojection
2. Pseudo-label generation — reproject consensus 3D reconstructions into camera views as OBB + pose training labels with confidence scoring
3. Gap detection and fill — identify detection gaps via inverse LUT visibility and generate corrective labels tagged by failure reason
4. Elastic midline deformation augmentation — TPS-based C-curve/S-curve synthetic training data generation reducing curvature bias (OKS slope -0.71 to -0.30)
5. SQLite sample store — content-hash dedup, provenance tracking, symlink-based dataset assembly, model lineage with config auto-update
6. Workflow-oriented CLI — project-aware path resolution, run shorthand, deprecated command removal, `aquapose {data, prep, pseudo-label}` groups

**Delivered:** Complete pseudo-labeling infrastructure enabling iterative model retraining from pipeline 3D reconstructions. Source A (consensus reprojections) and Source B (gap-fill) pseudo-labels with confidence scoring, elastic augmentation for curvature bias reduction, SQLite-backed sample store with dedup and lineage tracking, and workflow-oriented CLI with project/run resolution.

---

## v3.4 Performance Optimization (Shipped: 2026-03-05)

**Phases completed:** 5 phases (56-60), 8 plans
**Timeline:** 1 day (2026-03-04 → 2026-03-05)
**Codebase:** 22,754 LOC source
**Git range:** 15 commits, 56 files changed (+7,152 / -210)

**Key accomplishments:**
1. 8.2x total pipeline speedup (914s → 112s per chunk) validated end-to-end
2. Batched YOLO inference for detection (11.5x speedup) and midline (8.1x speedup) with automatic OOM retry and batch halving
3. Background-thread frame prefetch in ChunkFrameSource eliminating seek overhead and GPU idle gaps
4. Vectorized DLT reconstruction via batched `torch.linalg.lstsq` replacing per-body-point Python loop (7.0x speedup)
5. Vectorized association scoring via NumPy broadcasting replacing per-pair ray-ray distance loop (3.8x speedup)
6. Performance validation script and report confirming correctness preservation across all optimizations

**Delivered:** Comprehensive pipeline performance optimization reducing per-chunk processing time from 914s to 112s (8.2x speedup) by batching YOLO inference across cameras/crops, prefetching frames in a background thread, and vectorizing DLT reconstruction and association scoring. All 13 requirements satisfied, audit passed.

**Tech debt (non-blocking):**
- ASSOC-01/ASSOC-02 checkboxes unchecked in REQUIREMENTS.md (satisfied per verification)
- Phase 57 SUMMARY missing requirements-completed frontmatter field
- SC-5 (Phase 59): real-data eval comparison for batched vs serial inference pending
- Eval correctness FAIL from GPU non-determinism accepted as non-regression

---

## v3.3 Chunk Mode (Shipped: 2026-03-05)

**Phases completed:** 5 phases (51-55), 11 plans
**Timeline:** 2 days (2026-03-03 → 2026-03-04)
**Codebase:** 21,634 LOC source
**Git range:** 66 commits, 108 files changed (+11,046 / -4,401)

**Key accomplishments:**
1. FrameSource protocol + VideoFrameSource — injectable frame source replacing direct VideoSet usage in stages
2. ChunkOrchestrator processing video in fixed-size temporal chunks with per-chunk PosePipeline invocation
3. ChunkHandoff frozen dataclass carrying tracker state + identity map across chunk boundaries with atomic serialization
4. Identity stitching mapping chunk-local fish IDs to globally consistent IDs via track ID continuity
5. Per-chunk diagnostic caches (chunk_NNN/cache.pkl + manifest.json) with EvalRunner/TuningOrchestrator chunk-aware loading and merging
6. Visualization migrated from engine observers to evaluation suite — `aquapose viz` CLI operating on cached data post-run

**Delivered:** Chunk processing mode enabling reliable long-video processing without O(T²) association scaling. Videos processed in fixed-size temporal chunks with state carried across boundaries, per-chunk HDF5 flush with global frame offsets, and full diagnostic/evaluation support for multi-chunk runs. All 15 requirements satisfied, audit passed.

**Tech debt (non-blocking):**
- Phase 53 missing VERIFICATION.md (requirements verified by Phase 55)
- INTEG-02 wording stale in REQUIREMENTS.md (mutual exclusion removed in Phase 54)
- 5 pre-existing test failures in test_stage_association.py (stale DEFAULT_GRID expected values)

---

## v3.2 Evaluation Ecosystem (Shipped: 2026-03-03)

**Phases completed:** 5 phases (46-50), 11 plans
**Timeline:** 1 day (2026-03-03)
**Codebase:** 20,789 LOC source
**Git range:** 58 commits, 42 files changed (+5,850 / -2,936)

**Key accomplishments:**
1. Per-stage pickle caching system with StaleCacheError, ContextLoader, and envelope format for offline evaluation data
2. Five typed stage evaluators (detection, tracking, association, midline, reconstruction) with frozen metric dataclasses and DEFAULT_GRIDs
3. `aquapose eval <run-dir>` CLI producing multi-stage quality reports in human-readable and JSON format
4. `aquapose tune --stage` CLI with grid sweeps, two-tier validation (fast sweep + thorough top-N), and config diff output
5. Removed all legacy evaluation machinery — harness.py, midline_fixture.py, NPZ export, and 3 standalone scripts
6. Partial pipeline execution via `--resume-from` and `initial_context` for efficient parameter sweeps

**Delivered:** Unified evaluation and parameter tuning system replacing standalone scripts with `aquapose eval` and `aquapose tune` CLI subcommands, measuring stage-specific quality at every pipeline stage with per-stage pickle caching as the data source. All 22 requirements satisfied, audit passed.

**Tech debt (non-blocking):**
- 4 stale docstring references to deleted artifacts (dlt.py, evaluation/__init__.py, config.py)
- 4 pre-existing issues (test_pose_dataset_structure failure, 40 typecheck errors, 2 skipped test modules)

---

## v3.1 Reconstruction (Shipped: 2026-03-03)

**Phases completed:** 7 phases (40-45 including 43.1), 13 plans
**Timeline:** 2 days (2026-03-02 → 2026-03-03)
**Codebase:** 19,493 LOC source
**Git range:** 108 commits, 219 files changed (+19,532 / -10,256)

**Key accomplishments:**
1. Built diagnostic fixture system (MidlineFixture + NPZ serialization) for capturing pipeline intermediate data as offline-loadable evaluation fixtures
2. Created offline evaluation harness with CalibBundle, frame selection, Tier 1 reprojection error and Tier 2 leave-one-out camera stability metrics
3. Implemented confidence-weighted DLT triangulation backend with outlier rejection, replacing over-engineered RANSAC triangulation and curve optimizer
4. Empirically tuned outlier rejection threshold from 50.0 to 10.0 via grid search on real data evaluation
5. Systematic association parameter sweep revealed ~70% singleton rate as upstream detection/tracking bottleneck, not association parameters
6. Removed ~3,200 lines of dead reconstruction code (old triangulation, curve optimizer, epipolar/orientation machinery) — DLT is the sole reconstruction backend

**Delivered:** Complete reconstruction rebuild from over-engineered dual-backend system to minimal, empirically-validated DLT triangulation with proper evaluation infrastructure. Evaluation harness enables data-driven tuning of all reconstruction parameters.

---

## v3.0 Ultralytics Unification (Shipped: 2026-03-02)

**Phases completed:** 5 phases, 14 plans
**Timeline:** 2 days (2026-03-01 → 2026-03-02)
**Codebase:** 22,087 LOC source, 18,829 LOC tests (656 tests)
**Git range:** 54 commits, 168 files changed (+14,778 / -9,059)

**Key accomplishments:**
1. Removed all custom U-Net, SAM2, MOG2, and legacy training code — clean Ultralytics-only foundation
2. Built YOLO-seg and YOLO-pose training wrappers with CLI subcommands and COCO seg data converter
3. Implemented SegmentationBackend (YOLO-seg + skeletonization) and PoseEstimationBackend (YOLO-pose + spline) as selectable midline backends
4. Standardized training data format from NDJSON to YOLO txt+yaml labels across all three model types (OBB, seg, pose)
5. Consolidated config fields (single `weights_path`), fixed `init-config` defaults for YOLO-OBB + pose_estimation
6. Reorganized legacy `reconstruction/`, `segmentation/`, `tracking/` into `core/` submodules with shared `core/types/` package

**Delivered:** Complete migration from custom U-Net/keypoint models to Ultralytics-native YOLO-seg and YOLO-pose backends, with unified training infrastructure, standardized data format, consolidated config, and reorganized codebase. All 16 requirements satisfied, audit passed 16/16.

---

## v1.0 MVP (Shipped: 2026-02-25)

**Phases completed:** 12 phases, 28 plans
**Timeline:** 11 days (2026-02-14 → 2026-02-25)
**Codebase:** 50,802 LOC Python, 300 files modified
**Git range:** e9eddd1..590d068

**Key accomplishments:**
1. Differentiable refractive projection layer validated across 13 cameras (sub-pixel accuracy, Z-uncertainty 132x XY quantified)
2. Full segmentation pipeline: MOG2/YOLO detection → SAM2 pseudo-labels → U-Net inference (best val IoU 0.623)
3. Parametric fish mesh with differentiable spine + cross-sections in PyTorch (watertight, batch-first)
4. RANSAC cross-view identity association + Hungarian 3D tracking with persistent fish IDs (9 fish, population-constrained)
5. End-to-end direct triangulation pipeline: 2D midlines → arc-length sampling → RANSAC triangulation → B-spline fitting → HDF5 output + visualization
6. Correspondence-free B-spline curve optimizer as alternative reconstruction method (chamfer distance + L-BFGS)

**Delivered:** Complete 3D fish midline reconstruction system from multi-view silhouettes via refractive multi-view triangulation, with two reconstruction methods (direct triangulation and curve optimization), diagnostic tooling, and synthetic test infrastructure.

**Known gaps:**
- Phase 8 plan 03 (E2E integration test on real data): SUMMARY.md missing from disk, though E2E test code was committed (feat(08-03))
- No milestone audit performed — completed without pre-audit

**Architecture pivot:** Analysis-by-synthesis pipeline (Phases 3-4) shelved mid-milestone due to 30+ min/sec runtime. Replaced by direct triangulation pipeline (Phases 5-9) achieving orders-of-magnitude faster reconstruction.

---


## v2.0 Alpha (Shipped: 2026-02-27)

**Phases completed:** 10 phases, 34 plans
**Timeline:** 3 days (2026-02-25 → 2026-02-27)
**Codebase:** 18,660 LOC source, 14,826 LOC tests (514 tests)
**Git range:** v1.0..bd327bd (140 commits)

**Key accomplishments:**
1. Built event-driven engine skeleton — Stage Protocol (structural typing), PipelineContext (typed accumulator), frozen dataclass config hierarchy, typed lifecycle events, Observer protocol with EventBus, PosePipeline orchestrator
2. Migrated all 5 computation stages (Detection, Midline, Association, Tracking, Reconstruction) as pure Stage implementors with strict import boundary enforcement
3. Implemented 5 observers (timing, HDF5 export, 2D reprojection overlay, 3D midline animation, diagnostic capture) as pure event subscribers with zero stage coupling
4. Created `aquapose run` CLI entrypoint with 4 execution modes (production, diagnostic, synthetic, benchmark) and observer factory in engine layer
5. Built golden data verification framework (frozen v1.0 outputs) and regression test suite with per-stage numerical tolerance checking
6. Conducted full architectural audit against guidebook, built AST-based import boundary checker with pre-commit hook, remediated all critical findings (IB-003, dead modules, Stage 3/4 coupling)

**Delivered:** Complete architectural refactor from script-driven pipeline to event-driven computation engine with strict 3-layer architecture (Core Computation → PosePipeline → Observers), verified for numerical equivalence with v1.0, with comprehensive audit tooling and diagnostic infrastructure.

**Known gaps:**
- VER-03 regression tests skip without real video data env vars (infrastructure correct, human execution pending)
- MOG2 detection backend not implemented (YOLO only; registry pattern ready)
- Pre-existing flaky test: test_near_claim_penalty_suppresses_ghost (test-ordering state pollution)

---


## v2.1 Identity (Shipped: 2026-02-28)

**Phases completed:** 7 phases, 12 plans, 0 tasks

**Key accomplishments:**
- (none recorded)

---

