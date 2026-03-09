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
- 🚧 **v3.6 Model Iteration & QA** — Phases 70-76 (in progress)

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

### v3.6 Model Iteration & QA (In Progress)

**Milestone Goal:** Run the pseudo-label retraining loop end-to-end, producing demonstrably better OBB detection and pose estimation models with full provenance tracking.

- [x] **Phase 70: Metrics & Comparison Infrastructure** - Extend evaluation with percentiles, per-keypoint breakdown, curvature-stratified quality, and track fragmentation
- [x] **Phase 71: Data Store Bootstrap** - Import manual annotations, train and register baseline models through store workflow (completed 2026-03-07)
- [x] **Phase 72: Baseline Pipeline Run & Metrics** - Establish quantitative "before" snapshot on short iteration clip (completed 2026-03-07)
- [ ] **Phase 73: Round 1 Pseudo-Labels & Retraining** - Generate pseudo-labels, manually correct in CVAT, train round 1 models with A/B curation comparison
- [ ] **Phase 74: Round 1 Evaluation & Decision** - Compare round 1 pipeline metrics to baseline; decide whether to proceed to round 2
- [ ] **Phase 75: Round 2 (Conditional)** - Second iteration if round 1 shows clear improvement with headroom remaining
- [ ] **Phase 76: Final Validation** - Full 5-minute pipeline run with best models, showcase overlay videos, summary document

## Phase Details

### Phase 70: Metrics & Comparison Infrastructure
**Goal**: All evaluation metrics extended and ready before the iteration loop starts, so every round is measured consistently
**Depends on**: Nothing (independent of data store work)
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06
**Success Criteria** (what must be TRUE):
  1. `aquapose eval` output includes reprojection error percentiles (p50, p90, p95), midline confidence percentiles (p10, p50, p90), and camera count percentiles (p50, p90)
  2. `aquapose eval` output includes per-keypoint reprojection error breakdown (mean + p90 per body point index) recomputed from cached splines
  3. `aquapose eval` output includes curvature-stratified reconstruction quality (reprojection error per curvature quantile bin with sample counts)
  4. `aquapose eval` output includes 3D track fragmentation analysis (gap count, gap duration stats, continuity ratio)
  5. All new metrics appear in both human-readable text and JSON output formats
**Plans:** 2/2 plans complete
Plans:
- [x] 70-01-PLAN.md — Percentile metrics (EVAL-01/02/03) and track fragmentation evaluator (EVAL-06)
- [x] 70-02-PLAN.md — Per-keypoint reprojection error (EVAL-04) and curvature-stratified quality (EVAL-05)

### Phase 71: Data Store Bootstrap
**Goal**: All existing manual annotations imported into the data store with baseline OBB and pose models trained, registered, and sanity-checked
**Depends on**: Nothing (parallel with Phase 70)
**Requirements**: BOOT-01, BOOT-02, BOOT-03, BOOT-04, BOOT-05
**Success Criteria** (what must be TRUE):
  1. `aquapose data convert` converts COCO-JSON annotations to both YOLO-OBB and YOLO-pose formats
  2. Manual annotations are in the data store as `source=manual` with correct provenance, and `data status` shows the expected sample counts
  3. Baseline OBB and pose models are trained from store-assembled datasets and registered with model lineage
  4. Train/val split respects temporal holdout convention (no near-duplicate leakage between splits)
  5. `aquapose data exclude --reason TAG` applies reason-tagged exclusions and `data status` shows breakdown by reason
**Plans:** 2/2 plans complete
Plans:
- [ ] 71-01-PLAN.md — Temporal split, val tagging, tagged assemble, exclusion reasons, training defaults
- [ ] 71-02-PLAN.md — End-to-end convert-import-assemble-train workflow execution

### Phase 72: Baseline Pipeline Run & Metrics
**Goal**: Quantitative "before" snapshot established on short iteration clip using baseline models from the store
**Depends on**: Phase 70, Phase 71
**Requirements**: ITER-01
**Success Criteria** (what must be TRUE):
  1. Pipeline completes a diagnostic-mode run on a short clip (~1 min) using store-registered baseline models
  2. `aquapose eval` produces a full metric report including all Phase 70 extended metrics
  3. Baseline metric numbers (singleton rate, reprojection error percentiles, track continuity, per-keypoint breakdown) are recorded as the benchmark for improvement
**Plans:** 1/1 plans complete
Plans:
- [ ] 72-01-PLAN.md — Pre-flight checks, baseline pipeline run, evaluation, and metric snapshot review

### Phase 73: Round 1 Pseudo-Labels & Retraining
**Goal**: Pseudo-labels generated from baseline run, manually corrected in CVAT, imported into store, and round 1 models trained with A/B curation comparison quantified
**Depends on**: Phase 72
**Requirements**: ITER-02, ITER-03, ITER-06
**Success Criteria** (what must be TRUE):
  1. Pseudo-labels (OBB + pose) generated from baseline run caches, diversity-selected, and imported into store as `source=pseudo, round=1`
  2. Selected subset manually corrected in CVAT; corrected labels imported as `source=manual` with correction magnitude quantified
  3. Round 1 OBB and pose models trained on manual + pseudo-label datasets (elastic augmentation on manual only) and registered with model lineage
  4. A/B comparison completed: model trained on CVAT-corrected labels vs model trained on uncorrected pseudo-labels, with curation value quantified via training metrics
  5. `aquapose train compare` shows training metric comparison between baseline and round 1 models
**Plans:** 3 plans
Plans:
- [ ] 73-01-PLAN.md — Generate pseudo-labels, diversity selection, label-studio curation checkpoint
- [ ] 73-02-PLAN.md — Import corrections, quantify, assemble datasets (uncurated before corrections, curated after)
- [ ] 73-03-PLAN.md — Train 4 models, A/B curation comparison

### Phase 74: Round 1 Evaluation & Decision
**Goal**: Round 1 models evaluated at pipeline level against baseline; informed decision on whether to proceed to round 2
**Depends on**: Phase 73
**Requirements**: ITER-04
**Success Criteria** (what must be TRUE):
  1. Pipeline re-run on same short clip with round 1 models produces comparable eval report
  2. Round 0 vs round 1 metric comparison is documented (singleton rate, reprojection error, track continuity, per-keypoint breakdown)
  3. Decision checkpoint completed: proceed to round 2, or skip to final validation, with rationale recorded
**Plans**: TBD

### Phase 75: Round 2 (Conditional)
**Goal**: Second iteration of the pseudo-label loop executed if round 1 showed clear improvement with remaining headroom
**Depends on**: Phase 74 (conditional on decision to proceed)
**Requirements**: ITER-05
**Success Criteria** (what must be TRUE):
  1. Round 2 pseudo-labels generated from round 1 pipeline run and imported as `source=pseudo, round=2`
  2. Round 2 models trained (with evaluation of whether elastic augmentation is still needed given pseudo-label curvature diversity)
  3. Round 0 vs round 1 vs round 2 metric comparison documented with clear trend assessment
**Plans**: TBD

### Phase 76: Final Validation
**Goal**: Best iteration models confirmed at full 5-minute scale with showcase outputs produced
**Depends on**: Phase 74 or Phase 75 (whichever is the last completed iteration phase)
**Requirements**: FINAL-01, FINAL-02, FINAL-03
**Success Criteria** (what must be TRUE):
  1. Full 5-minute pipeline run completes with best models and produces a complete `aquapose eval` report
  2. Overlay videos generated for all 12 cameras from the final run
  3. Summary document produced with metrics table (round 0 vs round 1 vs round 2 vs final), key observations, and known limitations
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 70 and 71 (parallel) -> 72 -> 73 -> 74 -> 75 (conditional) -> 76

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
| 61-69 | v3.5 | 22/22 | Complete | 2026-03-06 |
| 70. Metrics & Comparison Infrastructure | v3.6 | Complete    | 2026-03-06 | 2026-03-06 |
| 71. Data Store Bootstrap | 2/2 | Complete    | 2026-03-07 | - |
| 72. Baseline Pipeline Run & Metrics | 1/1 | Complete    | 2026-03-07 | - |
| 73. Round 1 Pseudo-Labels & Retraining | v3.6 | 0/2 | Not started | - |
| 74. Round 1 Evaluation & Decision | v3.6 | 0/TBD | Not started | - |
| 75. Round 2 (Conditional) | v3.6 | 0/TBD | Not started | - |
| 76. Final Validation | v3.6 | 0/TBD | Not started | - |

### Phase 77: Training module code quality: deduplicate YOLO wrappers and CLI commands, consolidate shared functions, fix seg registration bug, add tests for diverse subset selection and weight copying

**Goal:** Eliminate code duplication in the training module, fix seg CLI registration bug, and add test coverage for untested critical paths
**Requirements**: CQ-01, CQ-02, CQ-03, CQ-04, CQ-05, CQ-06, CQ-07, CQ-08
**Depends on:** None (independent refactoring, can run anytime)
**Plans:** 2/2 plans complete

Plans:
- [ ] 77-01-PLAN.md — Consolidate YOLO wrappers, CLI orchestrator, and deduplicate shared functions
- [ ] 77-02-PLAN.md — Add tests for weight-copying logic and diverse subset selection
