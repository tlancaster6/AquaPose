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
- 🚧 **v3.8 Improved Association** — Phases 87-92 (in progress)

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

### v3.8 Improved Association (In Progress)

**Milestone Goal:** Replace single-centroid ray scoring with multi-keypoint association, add swap-aware group validation and singleton recovery, and tune on real data to reduce the ~27% singleton rate.

- [x] **Phase 87: Tracklet2D Keypoint Propagation** - Extend Tracklet2D with per-frame keypoint and confidence arrays from tracker (completed 2026-03-11)
- [x] **Phase 88: Multi-Keypoint Pairwise Scoring** - Replace single-centroid ray casting with K-keypoint vectorized scoring (completed 2026-03-11)
- [x] **Phase 89: Fragment Merging Removal** - Delete merge_fragments and max_merge_gap; pipeline still runs end-to-end (completed 2026-03-11)
- [x] **Phase 90: Group Validation with Changepoint Detection** - Add validation.py replacing refinement.py; temporal ID swap splitting and outlier eviction (completed 2026-03-11)
- [ ] **Phase 91: Singleton Recovery** - Assign or split-assign singletons to existing groups; enforce same-camera overlap constraint
- [ ] **Phase 92: Parameter Tuning Pass** - Calibrate new config parameters on real data; confirm improvement over v3.7 baseline

## Phase Details

### Phase 87: Tracklet2D Keypoint Propagation
**Goal**: Tracklet2D carries full per-frame keypoint and confidence data from the tracking stage to the association stage
**Depends on**: Nothing (first phase of milestone)
**Requirements**: DATA-01, DATA-02
**Success Criteria** (what must be TRUE):
  1. `Tracklet2D` has `keypoints` field holding a (T, K, 2) array of per-frame keypoint coordinates
  2. `Tracklet2D` has `keypoint_conf` field holding a (T, K) array of per-frame confidence values, with 0.0 on coasted frames
  3. Both fields are None when the tracking stage does not produce keypoint data, and all existing association consumers are unaffected by the None default
  4. Unit tests confirm the keypoint arrays round-trip correctly through the tracker's `to_tracklet2d()` method
**Plans**: TBD

### Phase 88: Multi-Keypoint Pairwise Scoring
**Goal**: Association scoring casts rays from all confident keypoints per detection per frame instead of one centroid, producing richer pairwise affinity scores
**Depends on**: Phase 87
**Requirements**: SCORE-01, SCORE-02, SCORE-03, SCORE-04
**Success Criteria** (what must be TRUE):
  1. For a tracklet pair sharing frames, the scorer produces one aggregate score derived from multiple keypoint rays rather than a single centroid ray
  2. Keypoints below the configurable confidence floor are excluded from scoring on a per-frame basis — low-confidence frames cast fewer rays, not zero
  3. Per-frame keypoint distances are aggregated into a single pairwise score via the configured method (default: arithmetic mean); the aggregation method is selectable via `AssociationConfig`
  4. Scoring produces numerically identical results to a reference loop-based implementation; a round-trip unit test confirms LUT coordinate correctness (3D point projects, ray passes within 2mm of source)
  5. Benchmark shows no regression in scoring wall time relative to the added ray count (< 3x slowdown on the same chunk)
**Plans**: TBD

### Phase 89: Fragment Merging Removal
**Goal**: Fragment merging code is deleted and the pipeline still runs end-to-end without it
**Depends on**: Phase 87
**Requirements**: CLEAN-01
**Success Criteria** (what must be TRUE):
  1. `merge_fragments` function and all helpers are gone from `clustering.py`
  2. `max_merge_gap` field is absent from `AssociationConfig` and all YAML configs
  3. End-to-end pipeline run completes without errors on the benchmark clip
**Plans**: 1 plan
Plans:
- [ ] 89-01-PLAN.md — Delete merge_fragments code, config, tests, and interpolated references

### Phase 90: Group Validation with Changepoint Detection
**Goal**: After clustering, each group is audited for temporal ID swaps and outliers; swapped tracklets are split or evicted; refinement.py is deleted
**Depends on**: Phase 88, Phase 89
**Requirements**: VALID-01, VALID-02, VALID-03, VALID-04, CLEAN-02
**Success Criteria** (what must be TRUE):
  1. For each tracklet in a group, per-frame multi-keypoint residuals against the group consensus are computed and accessible for inspection
  2. A temporal changepoint in a tracklet's residual series causes the tracklet to be split at the detected swap point; the consistent segment stays in the group and the inconsistent segment becomes a singleton candidate
  3. A tracklet with uniformly high residual (no changepoint found) is evicted from the group and becomes a singleton candidate
  4. `refinement.py` is deleted; all downstream consumers of `per_frame_confidence` and `consensus_centroids` either use equivalent outputs from `validation.py` or handle None correctly
  5. False positive rate on confirmed-correct tracklets from the v3.7 benchmark is below 30% (measured during plan execution)
**Plans**: 2 plans
Plans:
- [ ] 90-01-PLAN.md — Create validation.py with multi-keypoint residuals, changepoint detection, and split/evict logic
- [ ] 90-02-PLAN.md — Wire validation into pipeline, migrate config, delete refinement.py

### Phase 91: Singleton Recovery
**Goal**: Singletons (including those created by Phase 90) are scored against existing groups and assigned, split-assigned, or left as true singletons
**Depends on**: Phase 90
**Requirements**: RECOV-01, RECOV-02, RECOV-03, RECOV-04
**Success Criteria** (what must be TRUE):
  1. Each singleton is scored against all existing groups using per-frame multi-keypoint residuals computed fresh against each group
  2. A singleton with strong overall match to one group is assigned to that group; the group membership constraint (no duplicate cameras in overlapping frames) is enforced before assignment
  3. A singleton with no strong overall match but a temporal split matching two distinct groups is split and each segment assigned to its matching group (swap-aware recovery)
  4. A singleton with no match after split analysis remains a singleton; the pipeline does not force assignment
  5. End-to-end pipeline run completes; measured singleton rate is lower than the v3.7 baseline (27%)
**Plans**: 2 plans
Plans:
- [ ] 91-01-PLAN.md — Create recovery.py with scoring, assignment, split-assign, and same-camera constraint logic
- [ ] 91-02-PLAN.md — Add config fields, wire into stage.py, update __init__.py exports

### Phase 92: Parameter Tuning Pass
**Goal**: New config parameters are empirically calibrated on real data; the final v3.8 association configuration is documented and validated against the v3.7 baseline
**Depends on**: Phase 91
**Requirements**: EVAL-01, EVAL-02
**Success Criteria** (what must be TRUE):
  1. `aquapose tune --stage association` runs a parameter grid over the new config fields (confidence floor, changepoint threshold, minimum segment length, singleton assignment threshold) on cached tracking outputs
  2. Tuned parameters are applied and an end-to-end pipeline run produces a measurable reduction in singleton rate vs. the v3.7 baseline (target: ~15%, floor: better than 27%)
  3. Reprojection error and grouping quality metrics are not degraded relative to v3.7 baseline
  4. Tuned config defaults are committed; a brief tuning results document records the sweep ranges, selected values, and metric comparison
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 87 → 88 → 89 → 90 → 91 → 92
Note: Phases 88 and 89 depend only on Phase 87 and can be executed in either order (they touch different files).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 87. Tracklet2D Keypoint Propagation | 1/1 | Complete    | 2026-03-11 |
| 88. Multi-Keypoint Pairwise Scoring | 1/1 | Complete    | 2026-03-11 |
| 89. Fragment Merging Removal | 1/1 | Complete    | 2026-03-11 |
| 90. Group Validation with Changepoint Detection | 2/2 | Complete   | 2026-03-11 |
| 91. Singleton Recovery | 0/2 | Not started | - |
| 92. Parameter Tuning Pass | 0/TBD | Not started | - |
