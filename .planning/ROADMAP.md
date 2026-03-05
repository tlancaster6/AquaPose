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
- **v3.5 Pseudo-Labeling** — Phases 61-66 (in progress)

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

### v3.5 Pseudo-Labeling (In Progress)

**Milestone Goal:** Use pipeline 3D reconstructions to generate training labels at scale, enabling iterative model retraining to improve detection and pose estimation quality.

- [x] **Phase 61: Z-Denoising** - Plane projection and temporal smoothing to clean z-noise from 3D reconstructions (completed 2026-03-05)
- [x] **Phase 62: Prep Infrastructure** - CLI wiring for calibrate-keypoints and LUT generation extraction (completed 2026-03-05)
- [x] **Phase 63: Pseudo-Label Generation (Source A)** - Reproject consensus 3D reconstructions into camera views as training labels (completed 2026-03-05)
- [ ] **Phase 64: Gap Detection and Fill (Source B)** - Identify and label detection gaps via inverse LUT visibility cross-referencing
- [ ] **Phase 65: Frame Selection and Dataset Assembly** - Temporal subsampling, pose-diversity sampling, and pooled dataset construction
- [ ] **Phase 66: Training Run Management** - Run organization, cross-run comparison, and iterative retraining support

## Phase Details

### Phase 61: Z-Denoising
**Goal**: 3D reconstructions have clean in-plane spines suitable for accurate reprojection into camera views
**Depends on**: Nothing (first phase of v3.5)
**Requirements**: RECON-01, RECON-02, RECON-03, RECON-04, RECON-05, RECON-06
**Success Criteria** (what must be TRUE):
  1. Component A (plane projection): IRLS-weighted SVD plane fit projects triangulated points before spline fitting; plane normal + centroid stored per fish per frame; reprojection residuals increase by no more than ~0.5 px (do-no-harm check)
  2. Component B (temporal smoothing): plane normals smoothed per-fish within continuous track segments; control points rotated via stored normal/centroid; median z-range drops below ~1 cm; frame-to-frame z-profile RMS < 0.1 cm; SNR > 1 for most fish
  3. Signed off-plane residuals stored per body point (no hard bypass; real out-of-plane structure preserved in residuals for future Component C)
  4. Separate config toggles: `plane_projection.enabled` (reconstruction-time) and `plane_smoothing.enabled` / `plane_smoothing.sigma_frames` (post-processing). A can run without B; B requires A
  5. HDF5 writer and Midline3D type updated with plane normal, centroid, and off-plane residual fields
**Plans:** 2/2 plans complete
Plans:
- [ ] 61-01-PLAN.md — Component A: plane fit, Midline3D extension, config, HDF5 writer
- [ ] 61-02-PLAN.md — Component B: temporal smoothing CLI, eval metrics

### Phase 62: Prep Infrastructure
**Goal**: Users can prepare calibrated keypoint t-values and pre-generated LUTs before running pseudo-label generation
**Depends on**: Nothing (independent of Phase 61, but both must complete before Phase 63)
**Requirements**: PREP-01, PREP-02, PREP-03, PREP-04
**Success Criteria** (what must be TRUE):
  1. User can run `aquapose prep calibrate-keypoints` and obtain anatomical t-values from manual annotations
  2. Pseudo-label generation fails fast with a clear error if keypoint t-values are not configured (no silent uniform fallback)
  3. User can run `aquapose prep generate-luts` to pre-generate forward and inverse LUTs from calibration data
  4. AssociationStage requires pre-generated LUTs as input and fails fast if missing (lazy generation removed)
**Plans:** 2 plans
Plans:
- [x] 62-01-PLAN.md — Rework calibrate-keypoints CLI, fail-fast enforcement, init-config reminders
- [x] 62-02-PLAN.md — Add generate-luts CLI, remove lazy LUT generation, early pipeline validation

### Phase 63: Pseudo-Label Generation (Source A)
**Goal**: Users can generate high-confidence OBB and pose training labels from consensus 3D reconstructions
**Depends on**: Phase 61 (clean splines), Phase 62 (keypoint t-values, LUTs available)
**Requirements**: LABEL-01, LABEL-02, LABEL-03, LABEL-04
**Success Criteria** (what must be TRUE):
  1. User can generate OBB pseudo-labels from a pipeline run's diagnostic caches via CLI
  2. User can generate pose pseudo-labels with keypoints placed at calibrated anatomical positions along the 3D spline
  3. Each pseudo-label carries a confidence score derived from reconstruction quality metrics (residual, camera count, per-view residual)
  4. Labels are output in standard YOLO txt+yaml format with a confidence metadata sidecar
**Plans:** 2/2 plans complete
Plans:
- [ ] 63-01-PLAN.md — Promote geometry functions, build core pseudo-label module (reprojection, confidence, label generation)
- [ ] 63-02-PLAN.md — CLI command, diagnostic cache iteration, frame extraction, YOLO dataset output

### Phase 64: Gap Detection and Fill (Source B)
**Goal**: Users can identify where the model fails to detect visible fish and generate corrective training labels for those gaps
**Depends on**: Phase 63 (shares reprojection and label generation logic)
**Requirements**: GAP-01, GAP-02, GAP-03, GAP-04
**Success Criteria** (what must be TRUE):
  1. System identifies cameras where a reconstructed fish should be visible (via inverse LUT) but was not detected
  2. Each gap is tagged with a failure reason: no-detection, no-tracklet, or failed-midline
  3. Gap-fill pseudo-labels are generated by reprojecting 3D reconstructions into the gap cameras
  4. Source B labels are stored separately from Source A with distinct metadata, enforcing a configurable minimum camera floor (default 3)
**Plans:** 2 plans
Plans:
- [ ] 64-01-PLAN.md — Core gap detection, classification, and gap-fill label generation functions
- [ ] 64-02-PLAN.md — CLI refactoring with --consensus/--gaps flags and directory restructure

### Phase 65: Frame Selection and Dataset Assembly
**Goal**: Users can build a training dataset from manual annotations plus filtered pseudo-labels with controlled diversity and validation splits
**Depends on**: Phase 63 (Source A labels), Phase 64 (Source B labels)
**Requirements**: FRAME-01, FRAME-02, FRAME-03, DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. User can apply temporal subsampling (every Kth frame) and frames with zero reconstructions are automatically filtered
  2. Pose-diversity sampling selects frames that maximize coverage of body configurations (curved, straight, turning)
  3. User can assemble a training dataset pooling manual + Source A + Source B with independent confidence thresholds per source
  4. Assembled dataset includes a fixed manual validation set and a separate pseudo-label validation set broken down by source and gap reason
**Plans**: TBD

### Phase 66: Training Run Management
**Goal**: Users can track, compare, and iterate on training runs with full provenance of which pseudo-label round and thresholds produced each model
**Depends on**: Phase 65 (assembled datasets to train on)
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03
**Success Criteria** (what must be TRUE):
  1. Each training run outputs to a unique timestamped directory with a config snapshot and metric summary
  2. User can run `aquapose train compare` to generate a cross-run comparison table with both Ultralytics training metrics and aquapose eval pipeline metrics
  3. Comparison report tracks which pseudo-label round and confidence thresholds were used for each run
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 61 -> 62 -> 63 -> 64 -> 65 -> 66

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-9 | v1.0 | 28/28 | Complete | 2026-02-25 |
| 13-21 | v2.0 | 34/34 | Complete | 2026-02-27 |
| 22-28 | v2.1 | 12/12 | Complete | 2026-02-28 |
| 29-33.1 | v2.2 | 12/12 | Complete | 2026-03-01 |
| 35-39 | v3.0 | 14/14 | Complete | 2026-03-02 |
| 40-45 | v3.1 | 13/13 | Complete | 2026-03-03 |
| 46-50 | v3.2 | 11/11 | Complete | 2026-03-03 |
| 51-55 | v3.3 | 11/11 | Complete | 2026-03-05 |
| 56-60 | v3.4 | 8/8 | Complete | 2026-03-05 |
| 61. Z-Denoising | 2/2 | Complete    | 2026-03-05 | - |
| 62. Prep Infrastructure | v3.5 | 2/2 | Complete | 2026-03-05 |
| 63. Pseudo-Label Generation (Source A) | 2/2 | Complete   | 2026-03-05 | - |
| 64. Gap Detection and Fill (Source B) | v3.5 | 0/2 | Not started | - |
| 65. Frame Selection and Dataset Assembly | v3.5 | 0/? | Not started | - |
| 66. Training Run Management | v3.5 | 0/? | Not started | - |
