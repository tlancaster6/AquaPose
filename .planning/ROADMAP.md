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
- ✅ **v3.9 Reconstruction Modernization** — Phases 93-96 (shipped 2026-03-14)
- 🚧 **v3.10 Publication Metrics** — Phases 97-101 (in progress)

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

<details>
<summary>✅ v3.9 Reconstruction Modernization (Phases 93-96) — SHIPPED 2026-03-14</summary>

- [x] Phase 93: Config Plumbing (1/1 plans) — completed 2026-03-13
- [x] Phase 94: Dead Code Removal (1/1 plans) — completed 2026-03-13
- [x] Phase 95: Spline Refactoring (2/2 plans) — completed 2026-03-13
- [x] Phase 96: Z-Denoising and Documentation (1/1 plans) — completed 2026-03-13

**4 phases, 5 plans total**
Full details: `.planning/milestones/v3.9-ROADMAP.md`

</details>

### v3.10 Publication Metrics (In Progress)

**Milestone Goal:** Run full 5-minute diagnostic pipeline and produce comprehensive, publication-ready performance and accuracy metrics.

- [x] **Phase 97: Full Pipeline Run** - Execute 9,000-frame diagnostic run with production models
- [x] **Phase 98: Performance Metrics** - Extract per-stage timing breakdown and throughput (completed 2026-03-15)
- [x] **Phase 99: Reconstruction Quality Metrics** - Report reprojection error distributions and camera coverage (completed 2026-03-15)
- [ ] **Phase 100: Tracking and Association Metrics** - Report fragmentation, identity consistency, singleton rate, and wall-time
- [ ] **Phase 101: Results Document** - Compile all metrics into updated performance-accuracy.md

## Phase Details

### Phase 97: Full Pipeline Run
**Goal**: A complete 9,000-frame diagnostic pipeline run completes successfully with production models and all chunk caches written to disk
**Depends on**: Nothing (first phase of milestone)
**Requirements**: RUN-01
**Success Criteria** (what must be TRUE):
  1. `aquapose run` completes all chunks without crashing
  2. Per-chunk cache files exist for every chunk in the run directory
  3. HDF5 output is present and contains 3D reconstruction data for the full run
  4. Run used the production models configured in config.yaml
**Plans:** 1/1 plans complete
Plans:
- [x] 97-01-PLAN.md — Launch full diagnostic pipeline run and verify outputs

### Phase 98: Performance Metrics
**Goal**: Per-stage timing and end-to-end throughput numbers are extracted from the run and recorded
**Depends on**: Phase 97
**Requirements**: RUN-02, RUN-03
**Success Criteria** (what must be TRUE):
  1. Per-stage wall-time reported for detection, pose, tracking, association, and reconstruction
  2. End-to-end throughput reported as frames/sec and total wall-time for the full 9,000-frame run
  3. Numbers are drawn from the actual Phase 97 run (not synthetic or estimated)
**Plans:** 1/1 plans complete
Plans:
- [ ] 98-01-PLAN.md — Parse timing data and record pipeline performance metrics

### Phase 99: Reconstruction Quality Metrics
**Goal**: Reprojection error statistics and camera visibility statistics are measured and recorded from the full run
**Depends on**: Phase 97
**Requirements**: RECON-01, RECON-02, RECON-03
**Success Criteria** (what must be TRUE):
  1. Reprojection error distribution reported with mean, p50, p90, p99 across all frames in the run
  2. Per-keypoint reprojection error breakdown shows all 6 keypoints individually
  3. Camera visibility statistics reported (mean cameras per fish, distribution) across all frames
  4. Metrics derived from `aquapose eval` output on Phase 97 run caches
**Plans:** 1/1 plans complete
Plans:
- [x] 99-01-PLAN.md — Add p99/camera visibility stats and run full evaluation

### Phase 100: Tracking and Association Metrics
**Goal**: Track fragmentation, identity consistency, detection coverage, singleton rate, and association wall-time are all measured from the full run
**Depends on**: Phase 97
**Requirements**: TRACK-01, TRACK-02, TRACK-03, ASSOC-01, ASSOC-02
**Success Criteria** (what must be TRUE):
  1. Track count and fragmentation metrics reported (total tracks, fragments per fish, longest continuous track)
  2. Identity consistency across chunk boundaries reported (ID swap count or consistency rate)
  3. Detection coverage reported as % frames with detections per camera across all cameras
  4. Singleton rate reported for the full run
  5. Association wall-time reported as seconds per chunk and total for full run
**Plans:** 1 plan
Plans:
- [ ] 100-01-PLAN.md — Run eval and record tracking/association metrics from full run

### Phase 101: Results Document
**Goal**: A single performance-accuracy.md document contains all current full-run metrics with stale results replaced
**Depends on**: Phase 98, Phase 99, Phase 100
**Requirements**: DOC-01, DOC-02
**Success Criteria** (what must be TRUE):
  1. performance-accuracy.md exists and contains all metrics from Phases 98, 99, and 100
  2. All previously stale or placeholder metric values are replaced with Phase 97 measurements
  3. Supporting CSV files (if any) are present alongside the document
  4. Document clearly attributes all numbers to the Phase 97 run (date, run directory, model versions)
**Plans**: TBD


| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 93. Config Plumbing | v3.9 | 1/1 | Complete | 2026-03-13 |
| 94. Dead Code Removal | v3.9 | 1/1 | Complete | 2026-03-13 |
| 95. Spline Refactoring | v3.9 | 2/2 | Complete | 2026-03-13 |
| 96. Z-Denoising and Documentation | v3.9 | 1/1 | Complete | 2026-03-13 |
| 97. Full Pipeline Run | v3.10 | 1/1 | Complete | 2026-03-15 |
| 98. Performance Metrics | 1/1 | Complete    | 2026-03-15 | - |
| 99. Reconstruction Quality Metrics | v3.10 | 1/1 | Complete | 2026-03-15 |
| 100. Tracking and Association Metrics | v3.10 | 0/1 | Not started | - |
| 101. Results Document | v3.10 | 0/TBD | Not started | - |
