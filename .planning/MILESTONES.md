# Milestones

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

