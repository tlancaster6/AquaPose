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
- 🚧 **v3.4 Performance Optimization** — Phases 56-59 (in progress)

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

### v3.4 Performance Optimization (In Progress)

**Milestone Goal:** Reduce per-chunk pipeline processing time by optimizing the four profiled bottlenecks — batched YOLO inference (~26% of wall time across detection + midline), frame I/O (~12%), vectorized DLT reconstruction (~9%), and vectorized association scoring (~5%). All changes are correctness-neutral and verified against the existing `aquapose eval` harness.

- [x] **Phase 56: Vectorized Association Scoring** — Replace per-pair ray-ray distance loop with NumPy broadcasting (completed 2026-03-05)
- [x] **Phase 57: Vectorized DLT Reconstruction** — Replace per-body-point SVD loop with batched `torch.linalg.svd` (completed 2026-03-05)
- [ ] **Phase 58: Frame I/O Optimization** — Add background-thread prefetch source to eliminate seek overhead and GPU idle gaps
- [ ] **Phase 59: Batched YOLO Inference** — Batch detection and midline YOLO calls across cameras and crops per frame

## Phase Details

### Phase 56: Vectorized Association Scoring
**Goal**: Association pairwise scoring runs without per-pair Python loops, using batched NumPy operations that produce numerically identical results
**Depends on**: Phase 55 (v3.3 complete)
**Requirements**: ASSOC-01, ASSOC-02
**Success Criteria** (what must be TRUE):
  1. `score_tracklet_pair()` inner frame loop is replaced with batched NumPy ops using `ray_ray_closest_point_batch()`
  2. `aquapose eval` association metrics on a real YH chunk are identical before and after the change
  3. Early-termination semantics (score threshold short-circuit) are preserved — no spurious extra scoring after threshold hit
**Plans**: TBD

### Phase 57: Vectorized DLT Reconstruction
**Goal**: DLT triangulation processes all body points simultaneously via batched SVD rather than iterating one point at a time
**Depends on**: Phase 56
**Requirements**: RECON-01, RECON-02
**Success Criteria** (what must be TRUE):
  1. `DltBackend._triangulate_body_point()` scalar loop is replaced by a vectorized `_triangulate_fish_vectorized()` operating over all body points in one `torch.linalg.svd` call
  2. Inlier camera sets for each body point match the scalar baseline exactly on real YH chunk cache data
  3. 3D point positions from the vectorized path are within 1e-4 m of the scalar baseline
  4. `aquapose eval` reconstruction metrics on a real YH chunk are unchanged
**Plans:** 1/1 plans complete
Plans:
- [ ] 57-01-PLAN.md — Implement vectorized triangulation and equivalence tests

### Phase 58: Frame I/O Optimization
**Goal**: Frame decoding overlaps with GPU inference via a background-thread producer-consumer queue, eliminating seek overhead and GPU idle time between frames
**Depends on**: Phase 57
**Requirements**: FIO-01, FIO-02
**Success Criteria** (what must be TRUE):
  1. ChunkFrameSource internals replaced with background prefetch thread and bounded queue (per user decision — same class name, no new BatchFrameSource)
  2. ChunkFrameSource is a drop-in replacement — all stage code is unaffected and `aquapose eval` produces identical results
  3. Frame identity is correct across all 12 cameras for every frame in a multi-chunk run (no seek inaccuracy or thread-safety corruption)
  4. Prefetch buffer depth is hardcoded at 2 frames (~144 MB for 12 cameras)
**Plans:** 1 plan
Plans:
- [ ] 58-01-PLAN.md — Implement background prefetch in ChunkFrameSource and fix DetectionStage missing-camera guard

### Phase 59: Batched YOLO Inference
**Goal**: Detection and midline YOLO models receive batched inputs instead of one image at a time, increasing GPU utilization from ~30% toward its practical ceiling
**Depends on**: Phase 58
**Requirements**: BATCH-01, BATCH-02, BATCH-03, BATCH-04
**Success Criteria** (what must be TRUE):
  1. `DetectionStage.run()` collects all 12 camera frames for a timestep and calls `YOLOOBBBackend.detect_batch()` once per timestep instead of 12 times
  2. `MidlineStage.run()` collects all crops across all cameras for a frame and calls the midline backend's `process_batch()` once, correctly redistributing results by `(cam_id, det_idx)`
  3. `detection_batch_frames` and `midline_batch_crops` config fields exist and control batch sizes
  4. A CUDA OOM during `model.predict()` triggers an automatic retry with halved batch size rather than crashing the pipeline
  5. `aquapose eval` detection and midline metrics on a real YH chunk are identical to the pre-batching baseline
**Plans**: TBD

## Progress

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
| 56. Vectorized Association Scoring | v3.4 | Complete    | 2026-03-05 | - |
| 57. Vectorized DLT Reconstruction | 1/1 | Complete    | 2026-03-05 | - |
| 58. Frame I/O Optimization | v3.4 | 0/1 | Not started | - |
| 59. Batched YOLO Inference | v3.4 | 0/? | Not started | - |

### Phase 60: End-to-End Performance Validation

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 59
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 60 to break down)
