# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- ✅ **v2.0 Alpha** — Phases 13-21 (shipped 2026-02-27)
- ✅ **v2.1 Identity** — Phases 22-28 (shipped 2026-02-28)
- ✅ **v2.2 Backends** — Phases 29-33.1 (shipped 2026-03-01)
- ✅ **v3.0 Ultralytics Unification** — Phases 35-39 (shipped 2026-03-02)
- ✅ **v3.1 Reconstruction** — Phases 40-45 (shipped 2026-03-03)
- ✅ **v3.2 Evaluation Ecosystem** — Phases 46-50 (shipped 2026-03-03)
- ✅ **v3.3 Chunk Mode** — Phases 51-55 (shipped 2026-03-05)
- ✅ **v3.4 Performance Optimization** — Phases 56-60 (shipped 2026-03-05)
- ✅ **v3.5 Pseudo-Labeling** — Phases 61-69 (shipped 2026-03-06)
- ✅ **v3.6 Model Iteration & QA** — Phases 70-77 (shipped 2026-03-10)
- ✅ **v3.7 Improved Tracking** — Phases 78-86 (shipped 2026-03-11)
- ✅ **v3.8 Improved Association** — Phases 87-92 (shipped 2026-03-12)
- 🚧 **v3.9 Reconstruction Modernization** — Phases 93-96 (in progress)

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

<details>
<summary>✅ v3.0 Ultralytics Unification (Phases 35-39) — SHIPPED 2026-03-02</summary>

- [x] Phase 35: Codebase Cleanup (2/2 plans) — completed 2026-03-01
- [x] Phase 36: Training Wrappers (2/2 plans) — completed 2026-03-01
- [x] Phase 37: Pipeline Integration (2/2 plans) — completed 2026-03-01
- [x] Phase 38: Stabilization and Tech Debt Cleanup (3/4 plans, 1 deferred) — completed 2026-03-02
- [x] Phase 39: Core Reorganization (4/4 plans) — completed 2026-03-02

**5 phases, 14 plans total**
Full details: `.planning/milestones/v3.0-ROADMAP.md`

</details>

<details>
<summary>✅ v3.1 Reconstruction (Phases 40-45) — SHIPPED 2026-03-03</summary>

- [x] Phase 40: Diagnostic Capture (2/2 plans) — completed 2026-03-02
- [x] Phase 41: Evaluation Harness (2/2 plans) — completed 2026-03-02
- [x] Phase 42: Baseline Measurement (1/1 plan) — completed 2026-03-02
- [x] Phase 43: Triangulation Rebuild (2/2 plans) — completed 2026-03-02
- [x] Phase 43.1: Association Tuning (2/2 plans) — completed 2026-03-03 (INSERTED)
- [x] Phase 44: Validation and Tuning (2/2 plans) — completed 2026-03-03
- [x] Phase 45: Dead Code Cleanup (2/2 plans) — completed 2026-03-03

**7 phases, 13 plans total**
Full details: `.planning/milestones/v3.1-ROADMAP.md`

</details>

<details>
<summary>✅ v3.2 Evaluation Ecosystem (Phases 46-50) — SHIPPED 2026-03-03</summary>

- [x] Phase 46: Engine Primitives (3/3 plans) — completed 2026-03-03
- [x] Phase 47: Evaluation Primitives (3/3 plans) — completed 2026-03-03
- [x] Phase 48: EvalRunner and eval CLI (2/2 plans) — completed 2026-03-03
- [x] Phase 49: TuningOrchestrator and tune CLI (2/2 plans) — completed 2026-03-03
- [x] Phase 50: Cleanup and Replacement (1/1 plan) — completed 2026-03-03

**5 phases, 11 plans total**
Full details: `.planning/milestones/v3.2-ROADMAP.md`

</details>

<details>
<summary>✅ v3.3 Chunk Mode (Phases 51-55) — SHIPPED 2026-03-05</summary>

- [x] Phase 51: Frame Source Refactor (2/2 plans) — completed 2026-03-03
- [x] Phase 52: Chunk Orchestrator and Handoff (3/3 plans) — completed 2026-03-03
- [x] Phase 53: Integration and Validation (1/1 plan) — completed 2026-03-04
- [x] Phase 54: Chunk-Aware Diagnostics and Eval Migration (4/4 plans) — completed 2026-03-04
- [x] Phase 55: Chunk Validation and Gap Closure (1/1 plan) — completed 2026-03-05

**5 phases, 11 plans total**
Full details: `.planning/milestones/v3.3-ROADMAP.md`

</details>

<details>
<summary>✅ v3.4 Performance Optimization (Phases 56-60) — SHIPPED 2026-03-05</summary>

- [x] Phase 56: Vectorized Association Scoring (2/2 plans) — completed 2026-03-05
- [x] Phase 57: Vectorized DLT Reconstruction (1/1 plan) — completed 2026-03-05
- [x] Phase 58: Frame I/O Optimization (1/1 plan) — completed 2026-03-05
- [x] Phase 59: Batched YOLO Inference (3/3 plans) — completed 2026-03-05
- [x] Phase 60: End-to-End Performance Validation (1/1 plan) — completed 2026-03-05

**5 phases, 8 plans total**
Full details: `.planning/milestones/v3.4-ROADMAP.md`

</details>

<details>
<summary>✅ v3.5 Pseudo-Labeling (Phases 61-69) — SHIPPED 2026-03-06</summary>

- [x] Phase 61: Z-Denoising (2/2 plans) — completed 2026-03-05
- [x] Phase 62: Prep Infrastructure (2/2 plans) — completed 2026-03-05
- [x] Phase 63: Pseudo-Label Generation (2/2 plans) — completed 2026-03-05
- [x] Phase 64: Gap Detection and Fill (2/2 plans) — completed 2026-03-05
- [x] Phase 65: Frame Selection and Dataset Assembly (3/3 plans) — completed 2026-03-05
- [x] Phase 66: Training Run Management (2/2 plans) — completed 2026-03-05
- [x] Phase 67: Elastic Deformation Augmentation (2/2 plans) — completed 2026-03-06
- [x] Phase 68: Training Data Storage (4/4 plans) — completed 2026-03-06
- [x] Phase 69: CLI Workflow Cleanup (3/3 plans) — completed 2026-03-06

**9 phases, 22 plans total**
Full details: `.planning/milestones/v3.5-ROADMAP.md`

</details>

<details>
<summary>✅ v3.6 Model Iteration & QA (Phases 70-77) — SHIPPED 2026-03-10</summary>

- [x] Phase 70: Metrics & Comparison Infrastructure (2/2 plans) — completed 2026-03-06
- [x] Phase 71: Data Store Bootstrap (2/2 plans) — completed 2026-03-07
- [x] Phase 72: Baseline Pipeline Run & Metrics (1/1 plan) — completed 2026-03-07
- [x] Phase 73: Round 1 Pseudo-Labels & Retraining (3/3 plans) — completed 2026-03-09
- [x] Phase 74: Round 1 Evaluation & Decision (2/2 plans) — completed 2026-03-09
- [ ] Phase 75: Round 2 (Conditional) — skipped per Phase 74 decision
- [x] Phase 76: Final Validation (1/1 plan) — completed 2026-03-10
- [x] Phase 77: Training Module Code Quality (2/2 plans) — completed 2026-03-09

**8 phases (1 skipped), 13 plans total**
Full details: `.planning/milestones/v3.6-ROADMAP.md`

</details>

<details>
<summary>✅ v3.7 Improved Tracking (Phases 78-86) — SHIPPED 2026-03-11</summary>

- [x] Phase 78: Occlusion Investigation (2/2 plans) — completed 2026-03-10
- [x] Phase 78.1: OBB & Pose Production Retrain (2/2 plans) — completed 2026-03-10 (INSERTED)
- [x] Phase 79: Occlusion Remediation (Conditional) — skipped (GO decision)
- [x] Phase 80: Baseline Metrics (1/1 plan) — completed 2026-03-11
- [x] Phase 81: Pipeline Reorder & Segmentation Removal (2/2 plans) — completed 2026-03-11
- [x] Phase 82: Association Upgrade — Keypoint Centroid (1/1 plan) — completed 2026-03-11
- [x] Phase 83: Custom Tracker Implementation (2/2 plans) — completed 2026-03-11
- [x] Phase 84: Integration & Evaluation (2/2 plans) — completed 2026-03-11
- [x] Phase 84.1: Tracker Tuning (2/2 plans) — completed 2026-03-11 (INSERTED)
- [x] Phase 85: Code Quality Audit & CLI Smoke Test (2/2 plans) — completed 2026-03-11
- [x] Phase 86: Cleanup (Conditional) (2/2 plans) — completed 2026-03-11

**11 phases (1 skipped, 2 inserted), 18 plans total**
Full details: `.planning/milestones/v3.7-ROADMAP.md`

</details>

<details>
<summary>✅ v3.8 Improved Association (Phases 87-92) — SHIPPED 2026-03-12</summary>

- [x] Phase 87: Tracklet2D Keypoint Propagation (1/1 plans) — completed 2026-03-11
- [x] Phase 88: Multi-Keypoint Pairwise Scoring (1/1 plans) — completed 2026-03-11
- [x] Phase 89: Fragment Merging Removal (1/1 plans) — completed 2026-03-11
- [x] Phase 90: Group Validation with Changepoint Detection (2/2 plans) — completed 2026-03-11
- [x] Phase 91: Singleton Recovery (2/2 plans) — completed 2026-03-11
- [x] Phase 91.1: Association Bottleneck Investigation & Remediation (3/3 plans) — completed 2026-03-11 (INSERTED)
- [x] Phase 92: Parameter Tuning Pass (2/2 plans) — completed 2026-03-12

**7 phases (1 inserted), 12 plans total**
Full details: `.planning/milestones/v3.8-ROADMAP.md`

</details>

### v3.9 Reconstruction Modernization (In Progress)

**Milestone Goal:** Make reconstruction keypoint-native — raw anatomical keypoints as primary output, B-spline as optional post-processing, dead code and stale config plumbing cleaned up.

- [x] **Phase 93: Config Plumbing** - Wire `n_sample_points` through to ReconstructionStage; default to 6 (completed 2026-03-13)
- [x] **Phase 94: Dead Code Removal** - Remove scalar `_triangulate_body_point()` fallback and its comments (completed 2026-03-13)
- [ ] **Phase 95: Spline Refactoring** - Move B-spline fitting to optional post-processing; raw keypoints as primary output
- [ ] **Phase 96: Z-Denoising and Documentation** - Adapt z-denoising for keypoint arrays; update all stale docstrings and type docs

## Phase Details

### Phase 93: Config Plumbing
**Goal**: `n_sample_points` is a first-class config value that flows from ReconstructionConfig through the pipeline to ReconstructionStage, defaulting to 6 to reflect the 6-keypoint identity mapping
**Depends on**: Nothing (first phase of milestone)
**Requirements**: CFG-01, CFG-02
**Success Criteria** (what must be TRUE):
  1. `ReconstructionConfig.n_sample_points` exists with default value 6
  2. Setting `n_sample_points` in project YAML overrides the default and the pipeline uses the new value without any code changes
  3. The previous hardcoded value of 15 no longer appears in reconstruction logic
**Plans:** 1/1 plans complete
Plans:
- [ ] 93-01-PLAN.md — Wire n_sample_points end-to-end and change default to 6

### Phase 94: Dead Code Removal
**Goal**: The scalar `_triangulate_body_point()` fallback path and all comments referencing it are deleted from dlt.py
**Depends on**: Nothing (independent of config plumbing)
**Requirements**: CLEAN-01, CLEAN-02
**Success Criteria** (what must be TRUE):
  1. `_triangulate_body_point()` does not exist anywhere in the codebase
  2. No comments in dlt.py or related files reference the scalar fallback or the removed function
  3. All tests pass after deletion (no test relied on the removed path)
**Plans**: 1 plan
Plans:
- [ ] 94-01-PLAN.md — Delete scalar fallback, helper, constants, stale comments, and equivalence tests

### Phase 95: Spline Refactoring
**Goal**: B-spline fitting is no longer in the core reconstruction path — reconstruction produces raw triangulated keypoints directly, and spline fitting is available as a separate optional utility
**Depends on**: Phase 93
**Requirements**: SPL-01, SPL-02, SPL-03
**Success Criteria** (what must be TRUE):
  1. A pipeline run with spline disabled produces a valid Midline3D containing raw triangulated keypoints (not interpolated or fitted)
  2. The B-spline fitting function is callable as a standalone utility on a raw-keypoint Midline3D
  3. Midline3D can represent both raw-keypoint and spline-fitted states without type errors
  4. A pipeline run with spline enabled produces output numerically equivalent to the pre-refactor baseline
**Plans**: 2 plans
Plans:
- [ ] 95-01-PLAN.md — Extend Midline3D type, add spline_enabled config, refactor DltBackend
- [ ] 95-02-PLAN.md — Update HDF5 writer, evaluation, and pipeline wiring for raw-keypoint mode

### Phase 96: Z-Denoising and Documentation
**Goal**: Z-denoising operates correctly on raw 6-keypoint arrays, and all docstrings and type documentation reflect the keypoint-native, variable-point-count reconstruction output
**Depends on**: Phase 95
**Requirements**: ZDEN-01, DOC-01, DOC-02
**Success Criteria** (what must be TRUE):
  1. Running z-denoising on a raw 6-keypoint reconstruction array produces a denoised array with the same shape (no shape errors or silent dimension mismatches)
  2. The stage.py module docstring describes N-point keypoint-native output, not a fixed 15-point midline
  3. Midline2D and Midline3D type docstrings describe variable point counts and distinguish raw-keypoint vs spline-fitted representations
**Plans**: TBD
Plans:
- TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 93 → 94 → 95 → 96

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 93. Config Plumbing | 1/1 | Complete    | 2026-03-13 |
| 94. Dead Code Removal | 1/1 | Complete    | 2026-03-13 |
| 95. Spline Refactoring | 1/2 | In Progress|  |
| 96. Z-Denoising and Documentation | 0/TBD | Not started | - |
