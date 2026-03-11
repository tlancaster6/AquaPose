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
- 🚧 **v3.7 Improved Tracking** — Phases 78-86 (in progress)

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

### v3.7 Improved Tracking (In Progress)

**Milestone Goal:** Replace OC-SORT on OBB centroids with a custom bidirectional keypoint tracker. Reorder the pipeline so pose estimation precedes tracking, drop the segmentation midline backend, and upgrade cross-view association to use anatomical keypoints. Target: 9-track, zero-fragmentation output on the 20-second perfect-tracking clip.

- [x] **Phase 78: Occlusion Investigation** - Script + written findings characterizing detector/pose behavior under occlusion, with go/no-go recommendation — completed 2026-03-10, **GO**
- [x] **Phase 78.1: OBB & Pose Production Retrain** - Retrain detector/pose with corrected pseudo-labels in train/val split, more epochs for white-wall recall (INSERTED) — 2/2 plans complete
- [x] **Phase 79: Occlusion Remediation (Conditional)** - skipped per Phase 78 GO decision
- [x] **Phase 80: Baseline Metrics** - Quantitative OC-SORT tracking metrics on the perfect-tracking clip, establishing numbers to beat (completed 2026-03-11)
- [x] **Phase 81: Pipeline Reorder & Segmentation Removal** - Pose runs before tracking; segmentation backend removed; stage interfaces updated (completed 2026-03-11)
- [x] **Phase 82: Association Upgrade — Keypoint Centroid** - Cross-view association uses mid-body keypoint instead of OBB centroid (completed 2026-03-11)
- [x] **Phase 83: Custom Tracker Implementation** - Bidirectional batched keypoint tracker with OKS cost, OCM direction, KF state, gap interpolation (completed 2026-03-11)
- [x] **Phase 84: Integration & Evaluation** - New tracker wired into pipeline, evaluated against Phase 80 baselines (completed 2026-03-11)
- [ ] **Phase 85: Code Quality Audit & CLI Smoke Test** - Dead code removed, type errors fixed, pipeline runs end-to-end from CLI
- [ ] **Phase 86: Cleanup (Conditional)** - Address issues found in Phase 85 — skip if Phase 85 is clean

## Phase Details

### Phase 78: Occlusion Investigation
**Goal**: Understand how the OBB detector and pose model behave when fish partially occlude each other, and produce a go/no-go recommendation for proceeding to tracker implementation
**Depends on**: Phase 77 (v3.6 complete)
**Requirements**: INV-01, INV-02, INV-04
**Success Criteria** (what must be TRUE):
  1. A standalone script in `scripts/` runs detection + pose on a configurable camera/frame range and produces an annotated video with per-track-ID colors, gray/red untracked detections, and confidence-encoded keypoints
  2. The video covers the occlusion events at the ~13-14 second mark of `e3v831e-20260218T145915-150429.mp4` in the crop region (263,225)-(613,525)
  3. A written summary exists characterizing OBB and pose behavior under occlusion — specifically whether boxes merge, keypoints jump fish, and how per-keypoint confidence behaves
  4. The summary includes a concrete go/no-go recommendation on whether occlusion handling is acceptable for proceeding to tracker implementation
  5. A confidence threshold recommendation exists based on observed quality vs false-positive tradeoff across tested confidence levels
**Plans:** 2/2 plans complete
Plans:
- [x] 78-01-PLAN.md -- Build occlusion investigation script
- [x] 78-02-PLAN.md -- Execute investigation and produce findings

### Phase 78.1: OBB & Pose Production Retrain (INSERTED)

**Goal:** Retrain OBB detector and pose model with corrected pseudo-labels in all-source stratified train/val split, with 300 epochs and patience=50 for white-wall recall improvement. Terminal retrain producing production models for v3.7 tracker milestone.
**Requirements**: RETRAIN-01, RETRAIN-02, RETRAIN-03, RETRAIN-04
**Depends on:** Phase 78
**Plans:** 2/2 plans complete

Plans:
- [x] 78.1-01-PLAN.md -- Assemble datasets and hand off training commands to user (complete 2026-03-10)
- [x] 78.1-02-PLAN.md -- Evaluate new models, update config, visual white-wall check

### Phase 79: Occlusion Remediation (Conditional)
**Goal**: Address occlusion-related failure modes identified in Phase 78 before building the tracker — this phase is skipped entirely if Phase 78 yields a go recommendation
**Depends on**: Phase 78
**Requirements**: REM-01
**Success Criteria** (what must be TRUE):
  1. Each failure mode listed in the Phase 78 no-go finding has a corresponding fix (retraining, NMS tuning, filtering heuristic, or explicit deferral with justification)
  2. Re-running the Phase 78 investigation script on the same clip shows the failure modes are resolved or reduced to an acceptable level
**Plans**: TBD (scope set by Phase 78 findings)

### Phase 80: Baseline Metrics
**Goal**: Establish quantitative OC-SORT tracking baselines on the 20-second perfect-tracking target clip so post-overhaul improvements are measurable
**Depends on**: Phase 78.1
**Requirements**: INV-03
**Success Criteria** (what must be TRUE):
  1. A baseline metrics document exists recording track count, track duration distribution, fragmentation count, and total coverage for the current OC-SORT tracker on `e3v83eb-20260218T145915-150429.mp4` frames 3300-4500 (1:50-2:30, 40 seconds)
  2. The document states the gap to the zero-fragmentation, 9-track target explicitly as numbers (e.g., "8 tracks found, 3 ID switches, 94% coverage")
**Plans:** 1/1 plans complete
Plans:
- [ ] 80-01-PLAN.md — Build baseline tracking script, add 2D fragmentation evaluator, run and record metrics

### Phase 81: Pipeline Reorder & Segmentation Removal
**Goal**: Pose estimation runs immediately after detection (before tracking), and the segmentation midline backend is fully removed from the codebase
**Depends on**: Phase 80
**Requirements**: PIPE-01, PIPE-02, PIPE-03
**Success Criteria** (what must be TRUE):
  1. The pipeline executes in order Detection -> Pose -> Tracking -> Association -> Reconstruction without errors on a test clip
  2. `backends/segmentation.py`, skeletonization code, and orientation resolution logic that only applied to segmentation are gone from the codebase — no dead imports or stale references
  3. `PipelineContext` and all stage interfaces reflect the new stage ordering and accept pose outputs from Stage 2
  4. All existing unit tests pass; any tests that depended on the old stage order or segmentation backend are updated or removed
**Plans:** 2/2 plans complete
Plans:
- [ ] 81-01-PLAN.md — Rename core/midline to core/pose, reorder pipeline, update PipelineContext and Detection type
- [ ] 81-02-PLAN.md — Delete segmentation/orientation code, update consumers, fix all tests

### Phase 82: Association Upgrade — Keypoint Centroid
**Goal**: Cross-view association uses the mid-body keypoint position instead of the OBB centroid, making ray-based matching more stable under partial occlusion and frame-edge clipping
**Depends on**: Phase 81
**Requirements**: ASSOC-01
**Success Criteria** (what must be TRUE):
  1. `Tracklet2D.centroids` is populated from the selected mid-body keypoint (empirically determined highest-confidence keypoint index) rather than OBB center
  2. The association stage runs end-to-end without modification to the downstream LUT/ray-ray scoring/Leiden clustering machinery
  3. A brief note documents which keypoint index was selected and why (confidence statistics)
**Plans:** 1/1 plans complete
Plans:
- [ ] 82-01-PLAN.md — Add keypoint centroid config, implement extraction in tracker, document selection

### Phase 83: Custom Tracker Implementation
**Goal**: A bidirectional batched keypoint tracker replaces OC-SORT, using OKS cost, OCM direction consistency, Kalman filter over keypoint positions, asymmetric birth/death, ORU/OCR mechanisms, bidirectional merge, chunk handoff, and gap interpolation
**Depends on**: Phase 81
**Requirements**: TRACK-01, TRACK-02, TRACK-03, TRACK-04, TRACK-05, TRACK-06, TRACK-07, TRACK-08, TRACK-09, TRACK-10
**Success Criteria** (what must be TRUE):
  1. The tracker runs a forward and backward OC-SORT pass over each chunk and merges the resulting tracklets
  2. Association cost uses OKS (keypoint similarity) rather than IoU on OBBs, with OCM direction consistency as an additive term using the spine heading vector
  3. The Kalman filter tracks keypoint positions and velocities; the state dimension (60-dim or 24-dim) is explicitly chosen and documented
  4. Track birth and death apply asymmetric rules based on frame-edge proximity (edge tracks born/die more easily)
  5. Chunk boundary handoff serializes and restores KF mean, covariance, and observation history so tracks survive chunk transitions
  6. Small tracklet gaps are filled via spline interpolation
  7. If INV-04 findings reveal significant low-confidence valid detections, a secondary BYTE-style pass for those detections is implemented; otherwise TRACK-10 is explicitly deferred
**Plans:** 2/2 plans complete
Plans:
- [ ] 83-01-PLAN.md — KF engine, OKS/OCM cost, single-pass tracker, sigma computation
- [ ] 83-02-PLAN.md — Bidirectional merge, gap interpolation, chunk handoff, config + stage wiring

### Phase 84: Integration & Evaluation
**Goal**: The new tracker is wired into the reordered pipeline and evaluated against the Phase 80 baselines, with iteration on parameters if needed
**Depends on**: Phase 83, Phase 82
**Requirements**: INTEG-01, INTEG-02
**Success Criteria** (what must be TRUE):
  1. `aquapose run` with the new pipeline order and custom tracker completes end-to-end on the perfect-tracking 20-second clip without errors
  2. Post-run tracking metrics (track count, duration distribution, fragmentation, coverage) are compared directly against the Phase 80 OC-SORT baselines in a written evaluation note
  3. The tracker shows measurable improvement on at least one primary metric (fragmentation count or track count closer to 9) compared to OC-SORT baseline
**Plans:** 2/2 plans complete
Plans:
- [ ] 84-01-PLAN.md — Pipeline wiring, evaluation script, metrics comparison

### Phase 85: Code Quality Audit & CLI Smoke Test
**Goal**: The overhaul leaves no dead code, broken cross-references, or type errors; the full pipeline runs cleanly from the CLI with the new stage ordering
**Depends on**: Phase 84
**Requirements**: INTEG-03, INTEG-04
**Success Criteria** (what must be TRUE):
  1. A code quality audit finds zero dead code from the removed segmentation backend — no unused imports, unreachable functions, or stale references
  2. `hatch run typecheck` produces no new type errors introduced by the v3.7 overhaul
  3. `aquapose run` completes end-to-end on a test clip with the new pipeline, and all config options for the new tracker are documented
  4. A documented decision exists on whether BoxMot is removed as a dependency or retained as an OC-SORT fallback
**Plans:** TBD


### Phase 86: Cleanup (Conditional)
**Goal**: Address issues found during the Phase 85 audit — this phase is skipped entirely if Phase 85 finds nothing actionable
**Depends on**: Phase 85
**Requirements**: (none — scope set by Phase 85 findings)
**Success Criteria** (what must be TRUE):
  1. Each issue listed in the Phase 85 audit report has a corresponding fix or an explicit justification for deferral
  2. `hatch run check` and `hatch run test` pass cleanly after all fixes
**Plans**: TBD (scope set by Phase 85 findings)

## Progress

**Execution Order:**
Phases execute in numeric order: 78 -> 79 (conditional) -> 80 -> 81 -> 82 -> 83 -> 84 -> 85 -> 86 (conditional)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 78. Occlusion Investigation | v3.7 | 2/2 | Complete | 2026-03-10 |
| 78.1 OBB & Pose Production Retrain | 2/2 | Complete    | 2026-03-10 | - |
| 79. Occlusion Remediation (Conditional) | v3.7 | 0/0 | Skipped | 2026-03-10 |
| 80. Baseline Metrics | 1/1 | Complete    | 2026-03-11 | - |
| 81. Pipeline Reorder & Segmentation Removal | 2/2 | Complete    | 2026-03-11 | - |
| 82. Association Upgrade — Keypoint Centroid | 1/1 | Complete    | 2026-03-11 | - |
| 83. Custom Tracker Implementation | 2/2 | Complete    | 2026-03-11 | - |
| 84. Integration & Evaluation | 2/2 | Complete   | 2026-03-11 | - |
| 85. Code Quality Audit & CLI Smoke Test | v3.7 | 0/TBD | Not started | - |
| 86. Cleanup (Conditional) | v3.7 | 0/TBD | Not started | - |
