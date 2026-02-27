# Milestones

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

