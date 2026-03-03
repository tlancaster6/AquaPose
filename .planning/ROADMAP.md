# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- ✅ **v2.0 Alpha** — Phases 13-21 (shipped 2026-02-27)
- ✅ **v2.1 Identity** — Phases 22-28 (shipped 2026-02-28)
- ✅ **v2.2 Backends** — Phases 29-33.1 (shipped 2026-03-01)
- ✅ **v3.0 Ultralytics Unification** — Phases 35-39 (shipped 2026-03-02)
- ✅ **v3.1 Reconstruction** — Phases 40-45 (shipped 2026-03-03)
- ✅ **v3.2 Evaluation Ecosystem** — Phases 46-50 (shipped 2026-03-03)
- 🔄 **v3.3 Chunk Mode** — Phases 51-53 (active)

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

### v3.3 Chunk Mode (Phases 51-53) — ACTIVE

- [x] **Phase 51: Frame Source Refactor** - Extract video I/O from stages into injectable frame source (1/2 plans complete)
- [ ] **Phase 52: Chunk Orchestrator and Handoff** - Implement chunk loop, ChunkHandoff, identity stitching, per-chunk HDF5 flush (0/3 plans)
- [ ] **Phase 53: Integration and Validation** - Wire orchestrator into CLI, disable HDF5 observer under chunk mode, validate output equivalence

## Phase Details

### Phase 51: Frame Source Refactor
**Goal**: Stages receive frames from an injectable source instead of opening VideoSet directly
**Depends on**: Nothing (first phase of v3.3)
**Requirements**: FRAME-01, FRAME-02, FRAME-03
**Success Criteria** (what must be TRUE):
  1. DetectionStage and MidlineStage accept a frame source via constructor; neither opens a VideoSet internally
  2. A frame source yields `(frame_idx, dict[str, ndarray])` — local frame index plus per-camera undistorted frames
  3. Running the pipeline with the new frame source produces identical outputs to the prior direct-VideoSet path
  4. `stop_frame` is absent from PipelineConfig; existing tests and configs that relied on it are updated or removed
**Plans**: 2 plans
Plans:
- [x] 51-01-PLAN.md — Define FrameSource protocol + VideoFrameSource, migrate stages and build_stages
- [ ] 51-02-PLAN.md — Migrate observers, remove stop_frame, delete VideoSet, update tests

### Phase 52: Chunk Orchestrator and Handoff
**Goal**: Videos can be processed in fixed-size temporal chunks with state carried across boundaries
**Depends on**: Phase 51
**Requirements**: CHUNK-01, CHUNK-02, CHUNK-03, CHUNK-04, CHUNK-05, IDENT-01, IDENT-02, OUT-01
**Success Criteria** (what must be TRUE):
  1. ChunkOrchestrator loops over fixed-size chunks, invoking PosePipeline once per chunk with a windowed frame source
  2. ChunkHandoff (frozen dataclass) carries tracker state and identity map from one chunk to the next; it is written atomically to `handoff.pkl` after each chunk
  3. Chunk-local fish IDs are mapped to globally consistent IDs via track ID continuity; unmatched tracklet groups receive fresh global IDs
  4. Per-chunk 3D midlines are flushed to the HDF5 output file with the correct global frame offset after each chunk completes
  5. Setting `chunk_size: 0` or `chunk_size: null` produces a single-chunk degenerate run with no behavioral change; setting `chunk_size < 100` emits a warning
**Plans**: 3 plans:
- [ ] 52-01-PLAN.md — ChunkHandoff dataclass, ChunkFrameSource, chunk_size config, write_handoff
- [ ] 52-02-PLAN.md — ChunkOrchestrator core loop, identity stitching, HDF5 flush, progress
- [ ] 52-03-PLAN.md — Delete CarryForward, migrate all references, fix tests

### Phase 53: Integration and Validation
**Goal**: Chunk mode is the production path in `aquapose run` and produces correct output verified against non-chunked baseline
**Depends on**: Phase 52
**Requirements**: OUT-02, INTEG-01, INTEG-02, INTEG-03
**Success Criteria** (what must be TRUE):
  1. `aquapose run` delegates to ChunkOrchestrator; the HDF5 observer is disabled automatically when chunk mode is active
  2. Specifying both `chunk_size > 0` and diagnostic mode in the config raises a validation error before any processing begins
  3. Running the same video with chunked mode (chunk_size=1000) and non-chunked mode (chunk_size=null) produces numerically equivalent HDF5 output
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
| 51 | 2/2 | Complete    | 2026-03-03 | - |
| 52 | v3.3 | 0/3 | Not started | - |
| 53 | v3.3 | 0/TBD | Not started | - |
