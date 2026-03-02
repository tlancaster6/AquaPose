# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- ✅ **v2.0 Alpha** — Phases 13-21 (shipped 2026-02-27)
- ✅ **v2.1 Identity** — Phases 22-28 (shipped 2026-02-28)
- ✅ **v2.2 Backends** — Phases 29-33.1 (shipped 2026-03-01)
- ✅ **v3.0 Ultralytics Unification** — Phases 35-39 (shipped 2026-03-02)
- 🚧 **v3.1 Reconstruction** — Phases 40-45 (in progress)

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

### 🚧 v3.1 Reconstruction (In Progress)

**Milestone Goal:** Tear down over-engineered reconstruction backends and rebuild from a minimal, empirically-validated triangulation baseline with a proper evaluation harness.

## Phases

- [x] **Phase 40: Diagnostic Capture** - Expand diagnostic observer to capture and serialize MidlineSet data as loadable fixtures (completed 2026-03-02)
- [ ] **Phase 41: Evaluation Harness** - Build offline evaluation framework with real-data fixtures, frame selection, and Tier 1/2 metrics
- [ ] **Phase 42: Baseline Measurement** - Run evaluation against current reconstruction backend to establish reference metrics
- [ ] **Phase 43: Triangulation Rebuild** - Implement stripped-down confidence-weighted DLT triangulation with outlier rejection and B-spline fitting
- [ ] **Phase 44: Validation and Tuning** - Confirm new backend meets or beats baseline on Tier 1 and Tier 2 metrics
- [ ] **Phase 45: Dead Code Cleanup** - Remove old triangulation backend, curve optimizer, and other dead reconstruction code

## Phase Details

### Phase 40: Diagnostic Capture
**Goal**: MidlineSet data from pipeline runs can be captured and loaded independently for offline evaluation
**Depends on**: Phase 39 (v3.0 codebase)
**Requirements**: DIAG-01, DIAG-02
**Success Criteria** (what must be TRUE):
  1. Running the pipeline in diagnostic mode serializes MidlineSet data to disk alongside existing diagnostic outputs
  2. Serialized MidlineSet fixtures can be loaded into a Python session without running the pipeline
  3. Loaded fixtures contain the same per-camera, per-fish midline data that the reconstruction stage receives
**Plans**: 40-01 (Serialization), 40-02 (Loader)

### Phase 41: Evaluation Harness
**Goal**: An offline evaluation framework exists that loads fixtures and computes Tier 1 and Tier 2 metrics without running the full pipeline
**Depends on**: Phase 40
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**Success Criteria** (what must be TRUE):
  1. The harness loads a MidlineSet fixture + calibration data and runs reconstruction metrics without a video or live pipeline
  2. Frame selection produces 15-20 frames from a ~300-frame window via uniform temporal sampling
  3. Tier 1 output shows per-fish, per-camera reprojection error with mean, max, and overall aggregates
  4. Tier 2 output shows leave-one-out camera stability as max control-point displacement across dropout runs
  5. Results are printed as a human-readable summary table and saved as machine-diffable regression data
**Plans**: 2 plans
- [ ] 41-01-PLAN.md — Extend fixture format with CalibBundle for self-contained evaluation
- [ ] 41-02-PLAN.md — Evaluation harness core: frame selection, metrics, output

### Phase 42: Baseline Measurement
**Goal**: Reference metrics from the current reconstruction backend are recorded and available for comparison
**Depends on**: Phase 41
**Requirements**: EVAL-06
**Success Criteria** (what must be TRUE):
  1. The evaluation harness runs against the current (pre-rebuild) reconstruction backend without error
  2. Baseline Tier 1 and Tier 2 metric values are saved to disk as the regression reference
  3. The baseline report is human-readable and identifies per-fish and per-camera outliers in the current backend
**Plans**: 1 plan
- [ ] 42-01-PLAN.md — Baseline measurement script with outlier flagging and regression persistence

### Phase 43: Triangulation Rebuild
**Goal**: A new reconstruction backend exists that uses confidence-weighted DLT triangulation with outlier rejection and B-spline fitting
**Depends on**: Phase 42
**Requirements**: RECON-01, RECON-02, RECON-03, RECON-04, RECON-05, RECON-06, RECON-07
**Success Criteria** (what must be TRUE):
  1. Each body point is triangulated via confidence-weighted DLT using all available cameras (single strategy, no branching on camera count)
  2. Cameras whose reprojection residual exceeds the rejection threshold are flagged as outliers and the point is re-triangulated with inlier cameras only
  3. A B-spline with 7 control points is fit to the triangulated points; frames where fewer than the minimum valid-point threshold are available are skipped
  4. Points with Z at or below water_z are rejected before fitting
  5. Reconstructions where a configurable fraction of body points had fewer than 3 inlier cameras are flagged as low-confidence
  6. Half-widths from upstream are passed through to the output without being used in triangulation logic
**Plans**: TBD

### Phase 44: Validation and Tuning
**Goal**: The new triangulation backend is confirmed to meet or beat the baseline on Tier 1 and Tier 2 metrics
**Depends on**: Phase 43
**Requirements**: RECON-08
**Success Criteria** (what must be TRUE):
  1. Running the evaluation harness against the new backend produces Tier 1 reprojection error at or below the Phase 42 baseline
  2. Running the evaluation harness against the new backend produces Tier 2 leave-one-out stability at or below the Phase 42 baseline
  3. The outlier rejection threshold has been empirically set based on evaluation output (value recorded in PROJECT.md decisions)
**Plans**: TBD

### Phase 45: Dead Code Cleanup
**Goal**: The codebase contains only the new triangulation backend with no orphaned reconstruction code
**Depends on**: Phase 44
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03
**Success Criteria** (what must be TRUE):
  1. The old triangulation backend module is deleted and no code imports from it
  2. The curve optimizer backend module is deleted and no code imports from it
  3. The refine_midline_lm stub and unused orientation/epipolar code paths are removed
  4. All existing tests pass after cleanup and no test references deleted modules
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 40 → 41 → 42 → 43 → 44 → 45

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-9 | v1.0 | 28/28 | Complete | 2026-02-25 |
| 13-21 | v2.0 | 34/34 | Complete | 2026-02-27 |
| 22-28 | v2.1 | 12/12 | Complete | 2026-02-28 |
| 29-33.1 | v2.2 | 12/12 | Complete | 2026-03-01 |
| 35-39 | v3.0 | 14/14 | Complete | 2026-03-02 |
| 40. Diagnostic Capture | 2/2 | Complete    | 2026-03-02 | - |
| 41. Evaluation Harness | v3.1 | 0/2 | Not started | - |
| 42. Baseline Measurement | v3.1 | 0/? | Not started | - |
| 43. Triangulation Rebuild | v3.1 | 0/? | Not started | - |
| 44. Validation and Tuning | v3.1 | 0/? | Not started | - |
| 45. Dead Code Cleanup | v3.1 | 0/? | Not started | - |
