# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- ✅ **v2.0 Alpha** — Phases 13-21 (shipped 2026-02-27)
- ✅ **v2.1 Identity** — Phases 22-28 (shipped 2026-02-28)
- ✅ **v2.2 Backends** — Phases 29-33.1 (shipped 2026-03-01)
- 🚧 **v3.0 Ultralytics Unification** — Phases 35-37 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-9) — SHIPPED 2026-02-25</summary>

- [x] Phase 1: Calibration and Refractive Geometry (2/2 plans) — complete
- [x] Phase 2: Segmentation Pipeline (4/4 plans) — complete
- [x] Phase 02.1: Segmentation Troubleshooting (3/3 plans) — complete (INSERTED)
- [x] Phase 02.1.1: Object-detection alternative to MOG2 (3/3 plans) — complete (INSERTED)
- [x] Phase 3: Fish Mesh Model and 3D Initialization (2/2 plans) — complete
- [x] Phase 4: Per-Fish Reconstruction (3/3 plans) — shelved (ABS too slow)
- [x] Phase 04.1: Isolate phase4-specific code (1/1 plan) — complete (INSERTED)
- [x] Phase 5: Cross-View Identity and 3D Tracking (3/3 plans) — complete
- [x] Phase 6: 2D Medial Axis and Arc-Length Sampling (1/1 plan) — complete
- [x] Phase 7: Multi-View Triangulation (1/1 plan) — complete
- [x] Phase 8: End-to-End Integration Testing (3 plans, 2 summaries) — complete
- [x] Phase 9: Curve-Based Optimization (2/2 plans) — complete

**12 phases, 28 plans total**
Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v2.0 Alpha (Phases 13-21) — SHIPPED 2026-02-27</summary>

- [x] Phase 13: Engine Core (4/4 plans) — completed 2026-02-25
- [x] Phase 14: Golden Data and Verification Framework (2/2 plans) — completed 2026-02-25
- [x] Phase 14.1: Fix Critical Mismatch (2/2 plans) — completed 2026-02-25 (INSERTED)
- [x] Phase 15: Stage Migrations (5/5 plans) — completed 2026-02-26
- [x] Phase 16: Numerical Verification and Legacy Cleanup (2/2 plans) — completed 2026-02-26
- [x] Phase 17: Observers (5/5 plans) — completed 2026-02-26
- [x] Phase 18: CLI and Execution Modes (3/3 plans) — completed 2026-02-26
- [x] Phase 19: Alpha Refactor Audit (4/4 plans) — completed 2026-02-26
- [x] Phase 20: Post-Refactor Loose Ends (5/5 plans) — completed 2026-02-27
- [x] Phase 21: Retrospective, Prospective (2/2 plans) — completed 2026-02-27

**10 phases, 34 plans total**
Full details: `.planning/milestones/v2.0-ROADMAP.md`

</details>

<details>
<summary>✅ v2.1 Identity (Phases 22-28) — SHIPPED 2026-02-28</summary>

- [x] Phase 22: Pipeline Scaffolding (2/2 plans) — completed 2026-02-27
- [x] Phase 23: Refractive Lookup Tables (2/2 plans) — completed 2026-02-27
- [x] Phase 24: Per-Camera 2D Tracking (1/1 plan) — completed 2026-02-27
- [x] Phase 25: Association Scoring and Clustering (2/2 plans) — completed 2026-02-27
- [x] Phase 26: Association Refinement and Pipeline Wiring (3/3 plans) — completed 2026-02-27
- [x] Phase 27: Diagnostic Visualization (1/1 plan) — completed 2026-02-27
- [x] Phase 28: E2E Testing (1/1 plan) — completed 2026-02-27

**7 phases, 12 plans total**
Full details: `.planning/milestones/v2.1-ROADMAP.md`

</details>

<details>
<summary>✅ v2.2 Backends (Phases 29-33.1) — SHIPPED 2026-03-01</summary>

- [x] **Phase 29: Guidebook Audit** — complete 2026-02-28
- [x] **Phase 30: Config and Contracts** — complete 2026-02-28
- [x] **Phase 31: Training Infrastructure** — complete 2026-02-28
- [x] **Phase 32: YOLO-OBB Detection Backend** — complete 2026-02-28
- [x] **Phase 33: Keypoint Midline Backend** — complete 2026-03-01
- [x] **Phase 33.1: Keypoint Training Data Augmentation** — complete 2026-03-01

**6 phases (including inserted 33.1), Phase 34 (Stabilization) deferred**
Full details: `.planning/phases/29-*` through `.planning/phases/33.1-*`

</details>

### 🚧 v3.0 Ultralytics Unification (In Progress)

**Milestone Goal:** Replace custom U-Net segmentation and keypoint regression models with Ultralytics-native YOLO26n-seg and YOLO26n-pose, unifying detection, segmentation, and midline extraction on one framework. Starts by stripping the failed custom code, then builds training wrappers for the new models, then integrates them as backends within the existing Stage architecture.

**⚠ Cross-cutting concern — Coordinate spaces:** Full-image ↔ crop-space conversions are a pervasive source of error, especially with OBB affine warps. Mismatches between training-time and inference-time crop preparation cause silent accuracy failures. Every phase must explicitly verify coordinate round-trips at each boundary (training labels, inference output, back-projection to full frame). Existing crop utilities should be reused with extreme care.

- [x] **Phase 35: Codebase Cleanup** — Remove custom U-Net, SAM2 pipeline, old midline backends, MOG2 backend, and legacy training CLI commands (completed 2026-03-01)
- [x] **Phase 36: Training Wrappers** — Add NDJSON seg data converter and YOLO-seg/pose training wrappers following existing yolo_obb.py pattern (completed 2026-03-01)
- [ ] **Phase 37: Pipeline Integration** — Implement YOLOSegBackend and YOLOPoseBackend as selectable midline backends with instance matching and config support

## Phase Details

### Phase 35: Codebase Cleanup — COMPLETE (2026-03-01)
**Goal**: The codebase contains no custom U-Net, SAM2 pseudo-label, old midline backend, MOG2 detection, or legacy training CLI code — only Ultralytics-based models and the new training wrappers remain, leaving a clean foundation for v3.0 backends
**Depends on**: Nothing (cleanup precedes building)
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04, CLEAN-05
**Plans**: 2/2 complete
Plans:
- [x] 35-01-PLAN.md — Remove custom models, SAM2, MOG2, and old training CLI
- [x] 35-02-PLAN.md — Stub midline backends as no-ops, correct planning docs
**Summaries**: 35-01-SUMMARY.md, 35-02-SUMMARY.md (in progress)

### Phase 36: Training Wrappers — COMPLETE (2026-03-01)
**Goal**: A COCO-to-NDJSON segmentation data converter and training wrappers for YOLO26n-seg and YOLO26n-pose are available from the CLI, following the same pattern as the existing `yolo_obb.py` training wrapper
**Depends on**: Phase 35 (clean codebase, no legacy training commands to conflict)
**Requirements**: DATA-01, TRAIN-01, TRAIN-02
**Plans**: 2/2 complete
Plans:
- [x] 36-01-PLAN.md — Add --mode seg to build_yolo_training_data.py (COCO polygon converter)
- [x] 36-02-PLAN.md — YOLO-seg and YOLO-pose training wrappers with CLI subcommands
**Summaries**: 36-02-SUMMARY.md

### Phase 37: Pipeline Integration
**Goal**: The pipeline supports `yolo_seg` and `yolo_pose` as selectable midline backends; running either end-to-end produces `Midline2D` objects compatible with the reconstruction stages
**Depends on**: Phase 36 (trained models exist), Phase 35 (custom model code removed; existing segment_then_extract and direct_pose backends are no-op stubs awaiting YOLO model wiring)
**Requirements**: PIPE-01, PIPE-02, PIPE-03
**Success Criteria** (what must be TRUE):
  1. Setting `midline.backend: yolo_seg` in pipeline config runs the full pipeline end-to-end; the MidlineStage produces binary masks per detection that feed skeletonization the same way U-Net masks did
  2. Setting `midline.backend: yolo_pose` in pipeline config runs the full pipeline end-to-end; the MidlineStage produces `Midline2D` objects with 6-keypoint coordinates resampled to `n_sample_points` and per-point confidence scores
  3. Both backends produce `Midline2D` instances with identical shape and field structure — the reconstruction stages require no backend-specific branching
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-9 | v1.0 | 28/28 | Complete | 2026-02-25 |
| 13-21 | v2.0 | 34/34 | Complete | 2026-02-27 |
| 22-28 | v2.1 | 12/12 | Complete | 2026-02-28 |
| 29-33.1 | v2.2 | 12/12 | Complete | 2026-03-01 |
| 35. Codebase Cleanup | v3.0 | 2/2 | Complete | 2026-03-01 |
| 36. Training Wrappers | v3.0 | Complete    | 2026-03-01 | 2026-03-01 |
| 37. Pipeline Integration | v3.0 | 0/TBD | Not started | - |
