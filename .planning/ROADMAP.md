# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** - Phases 1-9 (shipped 2026-02-25)
- ✅ **v2.0 Alpha** - Phases 13-21 (shipped 2026-02-27)
- ✅ **v2.1 Identity** - Phases 22-28 (shipped 2026-02-28)
- ✅ **v2.2 Backends** - Phases 29-33.1 (shipped 2026-03-01)
- ✅ **v3.0 Ultralytics Unification** - Phases 35-39 (shipped 2026-03-02)
- ✅ **v3.1 Reconstruction** - Phases 40-45 (shipped 2026-03-03)
- ✅ **v3.2 Evaluation Ecosystem** - Phases 46-50 (shipped 2026-03-03)
- ✅ **v3.3 Chunk Mode** - Phases 51-55 (shipped 2026-03-05)
- ✅ **v3.4 Performance Optimization** - Phases 56-60 (shipped 2026-03-05)
- ✅ **v3.5 Pseudo-Labeling** - Phases 61-69 (shipped 2026-03-06)
- ✅ **v3.6 Model Iteration & QA** - Phases 70-77 (shipped 2026-03-10)
- ✅ **v3.7 Improved Tracking** - Phases 78-86 (shipped 2026-03-11)
- ✅ **v3.8 Improved Association** - Phases 87-92 (shipped 2026-03-12)
- ✅ **v3.9 Reconstruction Modernization** - Phases 93-96 (shipped 2026-03-14)
- ✅ **v3.10 Publication Metrics** - Phases 97-101 (shipped 2026-03-15)
- 🚧 **v3.11 Appearance-Based ReID** - Phases 102-106 (in progress)

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

<details>
<summary>✅ v3.10 Publication Metrics (Phases 97-101) — SHIPPED 2026-03-15</summary>

- [x] Phase 97: Full Pipeline Run (1/1 plans) — completed 2026-03-15
- [x] Phase 98: Performance Metrics (1/1 plans) — completed 2026-03-15
- [x] Phase 99: Reconstruction Quality Metrics (1/1 plans) — completed 2026-03-15
- [x] Phase 100: Tracking and Association Metrics (1/1 plans) — completed 2026-03-15
- [x] Phase 101: Results Document (1/1 plans) — completed 2026-03-15

**5 phases, 5 plans total**
Full details: `.planning/milestones/v3.10-ROADMAP.md`

</details>

### 🚧 v3.11 Appearance-Based ReID (In Progress)

**Milestone Goal:** Add post-hoc appearance-based re-identification to resolve identity swaps that geometry cannot catch — specifically the known female-female cichlid swap. A new `aquapose reid` command group operates on completed pipeline output, embeds fish crops through a fine-tuned MegaDescriptor-T backbone, detects swaps at occlusion events via cosine similarity, and writes corrected trajectories to `midlines_reid.h5`.

#### Phases

- [x] **Phase 102: Embedding Infrastructure** - Crop extraction, backbone wrapper, and batch embed runner producing `embeddings.h5` for a completed run (completed 2026-03-25)
- [x] **Phase 103: Training Data Mining** - High-confidence trajectory mining with swap-buffer contamination filtering to produce quality-gated training crops (completed 2026-03-25)
- [x] **Phase 104: Backbone Fine-Tuning** - Metric learning fine-tune of MegaDescriptor-T with discriminability gate (female-female AUC >= 0.75) (completed 2026-03-25)
- [ ] **Phase 105: Swap Detection and Repair** - Cosine-similarity swap detection at occlusion events and margin-gated repair writing `midlines_reid.h5`
- [ ] **Phase 106: CLI Integration** - `aquapose reid` command group wiring all ReID subcommands into the existing CLI pattern

## Phase Details

### Phase 102: Embedding Infrastructure
**Goal**: Users can extract L2-normalized embeddings for every fish detection in a completed run and inspect them
**Depends on**: Nothing (first phase of milestone)
**Requirements**: EMBED-01, EMBED-02, EMBED-03
**Success Criteria** (what must be TRUE):
  1. Given a completed run directory, `embed_runner` iterates all (frame, fish_id, camera) tuples from `midlines_stitched.h5` and writes `embeddings.h5` without error
  2. Embeddings are L2-normalized 768-dim vectors (MegaDescriptor-T output) verifiable by checking that each row has unit norm
  3. Crops are extracted using the existing stretch-fill affine warp — same convention as YOLO training — verifiable by visual inspection of saved crops
  4. Zero-shot cosine similarity between two crops of the same fish from different cameras is measurably higher than between two crops of different fish on a sample of 50 frames
**Plans**: 2 plans
- [ ] 102-01-PLAN.md — Config + FishEmbedder backbone wrapper (timm, ReidConfig, embedder module)
- [ ] 102-02-PLAN.md — EmbedRunner with crop extraction, NPZ output, and zero-shot eval

### Phase 103: Training Data Mining
**Goal**: A quality-controlled training crop dataset is available, free of swap contamination and camera bias
**Depends on**: Phase 102
**Requirements**: TRAIN-01, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. Training data extractor produces a `reid_crops/<fish_id>/` directory structure covering all 9 fish identities
  2. No crops appear within 150 frames of a detected changepoint or swap event (contamination filter is verifiable by checking frame indices against the swap event list)
  3. Quality filter parameters (min cameras, min duration, max residual) are configurable and documented; filtering logs show how many segments were accepted vs rejected
  4. The dataset contains at least one valid segment per fish identity, or the run exits with a clear diagnostic message
**Plans**: 2 plans
- [ ] 103-01-PLAN.md — TrainingDataMiner core logic with quality gates, temporal windowing, sampling, and unit tests
- [ ] 103-02-PLAN.md — mine-reid-crops CLI command wiring

### Phase 104: Backbone Fine-Tuning
**Goal**: A fine-tuned backbone produces embeddings that discriminate female cichlids well enough to gate swap repair
**Depends on**: Phase 103
**Requirements**: TRAIN-03
**Success Criteria** (what must be TRUE):
  1. Fine-tuning loop runs to completion and saves `best_reid_model.pt` with training loss curves visible in logs
  2. Female-female pair AUC on the temporal holdout set is measured and reported; a value >= 0.75 is required to proceed to Phase 105
  3. Re-embedding all detections with the fine-tuned model produces updated `embeddings.h5` with measurably tighter within-identity cosine similarity compared to zero-shot baseline
**Plans**: 2 plans
- [x] 104-01-PLAN.md — ReID training module (ProjectionHead, feature caching, SubcenterArcFace training loop, AUC evaluation, unit tests)
- [x] 104-02-PLAN.md — Standalone driver script (end-to-end workflow with AUC gate and conditional re-embedding)

### Phase 105: Swap Detection and Repair
**Goal**: Known identity swap events are detected and repaired, producing corrected output with no increase in reprojection error
**Depends on**: Phase 104
**Requirements**: SWAP-01, SWAP-02
**Success Criteria** (what must be TRUE):
  1. Swap detector identifies both known swap events (male-female and female-female) from the persisted `/swap_events/` in `midlines_stitched.h5`
  2. Repair is applied only at detected occlusion events with cosine margin > 0.15 — stable segments are not modified
  3. `midlines_reid.h5` is written as a corrected copy; mean reprojection error in repaired segments does not increase versus the pre-repair baseline
  4. False positive rate on confirmed-clean segments is below 5% (measurable by running repair on segments with no known swaps)
**Plans**: 2 plans
- [ ] 105-01-PLAN.md — Core swap detector module (SwapDetector class, ReidEvent, cross-pattern confirmation, seeded + scan modes, repair, unit tests)
- [ ] 105-02-PLAN.md — Standalone validation script (driver on YH data, validates known swaps, FP rate, produces midlines_reid.h5)

### Phase 106: CLI Integration
**Goal**: All ReID capabilities are accessible via `aquapose reid` subcommands following existing CLI patterns
**Depends on**: Phase 105
**Requirements**: CLI-01
**Success Criteria** (what must be TRUE):
  1. `aquapose reid embed -p YH` runs the batch embed runner on a completed run and writes `embeddings.h5` with progress output
  2. `aquapose reid repair -p YH` runs swap detection and repair, writing `midlines_reid.h5` with a summary of events detected and repaired
  3. `aquapose reid train-data -p YH` extracts and filters training crops, reporting accept/reject counts per fish identity
  4. All subcommands follow existing CLI patterns (project flag `-p`, run resolution, consistent error messaging) — smoke test passes without errors
**Plans**: TBD

## Progress

**Execution Order:** Phases execute in numeric order: 102 → 103 → 104 → 105 → 106

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 102. Embedding Infrastructure | 2/2 | Complete    | 2026-03-25 | - |
| 103. Training Data Mining | 2/2 | Complete    | 2026-03-25 | - |
| 104. Backbone Fine-Tuning | v3.11 | 2/2 | Complete | 2026-03-25 |
| 105. Swap Detection and Repair | v3.11 | 0/TBD | Not started | - |
| 106. CLI Integration | v3.11 | 0/TBD | Not started | - |
