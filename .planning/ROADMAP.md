# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- 🚧 **v2.0 Alpha** — Phases 13-18 (in progress)

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

### 🚧 v2.0 Alpha (In Progress)

**Milestone Goal:** Transform AquaPose from a script-driven scientific pipeline into an event-driven scientific computation engine with strict architectural layering, verified for numerical equivalence with v1.0.

- [x] **Phase 13: Engine Core** - Stage Protocol, PipelineContext, events, observer base, config, orchestrator, import boundary (completed 2026-02-25)
- [x] **Phase 14: Golden Data and Verification Framework** - Generate frozen reference outputs from v1.0, define interface test harness (completed 2026-02-25)
- [ ] **Phase 15: Stage Migrations** - Port all 5 computation stages as pure Stage implementors
- [ ] **Phase 16: Numerical Verification and Legacy Cleanup** - Regression tests against golden data, archive legacy scripts
- [ ] **Phase 17: Observers** - Timing, HDF5 export, 2D reprojection, 3D animation, diagnostic observers
- [ ] **Phase 18: CLI and Execution Modes** - `aquapose run` entrypoint, production/diagnostic/synthetic/benchmark modes

## Phase Details

### Phase 13: Engine Core
**Goal**: The architectural skeleton exists — protocol interfaces, typed context, event system, observer base, config hierarchy, pipeline orchestrator, and enforced import boundary — ready for stages to be plugged in
**Depends on**: Nothing (first phase of v2.0)
**Requirements**: ENG-01, ENG-02, ENG-03, ENG-04, ENG-05, ENG-06, ENG-07, ENG-08
**Success Criteria** (what must be TRUE):
  1. A class can implement Stage Protocol via structural typing and be recognized without inheriting a base class
  2. PipelineContext accumulates typed fields set by each stage with no implicit shared state
  3. Firing a lifecycle event delivers it synchronously to all subscribed observers
  4. A frozen config object can be constructed from defaults, overridden by YAML, then overridden by CLI kwargs, and raises on post-freeze mutation
  5. The full serialized run config is written as the first artifact when PosePipeline.run() is called
**Plans**: 4 plans

Plans:
- [ ] 13-01-PLAN.md — Stage Protocol, PipelineContext, and import boundary (wave 1)
- [ ] 13-02-PLAN.md — Config dataclass hierarchy with YAML and CLI override support (wave 1)
- [ ] 13-03-PLAN.md — Event system and Observer protocol (wave 1)
- [ ] 13-04-PLAN.md — PosePipeline orchestrator skeleton (wave 2)

### Phase 14: Golden Data and Verification Framework
**Goal**: Frozen reference outputs from the v1.0 pipeline exist on disk as a committed snapshot, and an interface test harness can assert that a Stage produces correct output from a given context
**Depends on**: Phase 13
**Requirements**: VER-01, VER-02
**Success Criteria** (what must be TRUE):
  1. Running the v1.0 pipeline on a fixed clip produces outputs that are committed as golden data in a standalone commit
  2. A test can instantiate any Stage, call stage.run(context), and assert output fields in PipelineContext
  3. The golden data generation script is deterministic — re-running on the same clip produces bit-identical outputs
**Plans**: 2 plans

Plans:
- [ ] 14-01-PLAN.md — Golden data generation script and committed snapshot (wave 1)
- [ ] 14-02-PLAN.md — Interface test harness for stage output correctness (wave 2)

### Phase 14.1: Fix Critical Mismatch Between Old and Proposed Pipeline Structures (INSERTED)

**Goal:** Align all active planning documents and Phase 13/14 code to the guidebook's canonical 5-stage pipeline model (Detection, Midline, Association, Tracking, Reconstruction). The guidebook is the single source of truth. This phase does NOT port any stages — it corrects the planning foundation so Phase 15 starts from a consistent, accurate model.
**Requirements**: None (correction phase — updates existing requirements)
**Depends on:** Phase 14
**Plans:** 2/2 plans complete

Plans:
- [ ] 14.1-01-PLAN.md — Update planning documents (ROADMAP, REQUIREMENTS) and delete redundant inbox files (wave 1)
- [ ] 14.1-02-PLAN.md — Update engine code (PipelineContext, config) and golden test harness to match 5-stage model (wave 1)

### Phase 15: Stage Migrations
**Goal**: All 5 computation stages exist as pure Stage implementors with no side effects, wired into PosePipeline and producing context fields that downstream stages consume
**Depends on**: Phase 14
**Requirements**: STG-01, STG-02, STG-03, STG-04, STG-05
**Success Criteria** (what must be TRUE):
  1. Detection stage can be swapped between model-based detection backends (YOLO or MOG2) via config with no code change
  2. Each stage accepts only PipelineContext as input and writes only PipelineContext fields — no filesystem reads/writes inside stage logic
  3. PosePipeline.run() on a real clip completes all 5 stages without error
  4. Interface tests pass for each of the 5 stages individually
**Plans**: 5 plans

Plans:
- [ ] 15-01-PLAN.md — Detection stage (model-based backend: YOLO/MOG2) (wave 1)
- [ ] 15-02-PLAN.md — Midline stage (segment-then-extract backend: U-Net/SAM + skeletonization + BFS pruning) (wave 2)
- [ ] 15-03-PLAN.md — Cross-view association stage (RANSAC centroid clustering) (wave 3)
- [ ] 15-04-PLAN.md — Tracking stage (Hungarian 3D with population constraint) (wave 4)
- [ ] 15-05-PLAN.md — Reconstruction stage (triangulation backend: RANSAC + view-angle weighting + B-spline fitting) (wave 5)

### Phase 16: Numerical Verification and Legacy Cleanup
**Goal**: The migrated pipeline is confirmed numerically equivalent to v1.0 on real data, and all legacy scripts are archived and removed from active paths
**Depends on**: Phase 15
**Requirements**: VER-03, VER-04
**Success Criteria** (what must be TRUE):
  1. Regression tests run the new pipeline on the golden-data clip and confirm outputs match golden data within accepted tolerance (or document known intentional bug fixes)
  2. All legacy pipeline scripts have been moved to scripts/legacy/ and are no longer on any active import path
  3. The test suite passes with no references to the old script-based execution path
**Plans**: TBD

Plans:
- [ ] 16-01: Numerical regression tests against golden data
- [ ] 16-02: Legacy script archival and removal

### Phase 17: Observers
**Goal**: All diagnostic, export, and visualization side effects are implemented as Observers that subscribe to pipeline events and produce their outputs independently of stage logic
**Depends on**: Phase 16
**Requirements**: OBS-01, OBS-02, OBS-03, OBS-04, OBS-05
**Success Criteria** (what must be TRUE):
  1. Attaching the timing observer to a run produces a per-stage and total timing report without modifying any stage code
  2. Attaching the HDF5 export observer writes spline control points and metadata to disk after the pipeline completes
  3. Attaching the diagnostic observer captures intermediate stage outputs in memory without any stage being aware
  4. Removing all observers from a run produces identical numerical outputs (observers are purely additive side effects)
**Plans**: TBD

Plans:
- [ ] 17-01: Timing observer
- [ ] 17-02: HDF5 export observer
- [ ] 17-03: 2D reprojection overlay visualization observer
- [ ] 17-04: 3D midline animation visualization observer
- [ ] 17-05: Diagnostic observer

### Phase 18: CLI and Execution Modes
**Goal**: `aquapose run` is a working CLI entrypoint that accepts a config path and mode flag, assembles the correct observer set, and runs the pipeline — with no pipeline logic living in the CLI layer
**Depends on**: Phase 17
**Requirements**: CLI-01, CLI-02, CLI-03, CLI-04, CLI-05
**Success Criteria** (what must be TRUE):
  1. `aquapose run --config path.yaml` runs the full pipeline on a real clip and exits 0
  2. `aquapose run --mode diagnostic` activates the diagnostic observer and produces extra artifacts without any code change to stages or core observers
  3. `aquapose run --mode synthetic` runs the pipeline using injected synthetic data via a stage adapter, not a pipeline bypass
  4. `aquapose run --mode benchmark` runs with timing observer only and reports total and per-stage time
  5. The CLI layer contains no reconstruction logic — it only parses args, builds config, assembles observers, and calls PosePipeline.run()
**Plans**: TBD

Plans:
- [ ] 18-01: CLI entrypoint and production mode
- [ ] 18-02: Diagnostic and benchmark modes
- [ ] 18-03: Synthetic mode via stage adapter

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Calibration and Refractive Geometry | v1.0 | 2/2 | Complete | 2026-02-19 |
| 2. Segmentation Pipeline | v1.0 | 4/4 | Complete | 2026-02-20 |
| 02.1 Segmentation Troubleshooting | v1.0 | 3/3 | Complete | 2026-02-20 |
| 02.1.1 Object-detection alternative to MOG2 | v1.0 | 3/3 | Complete | 2026-02-20 |
| 3. Fish Mesh Model and 3D Initialization | v1.0 | 2/2 | Complete | 2026-02-20 |
| 4. Per-Fish Reconstruction | v1.0 | 3/3 | Shelved | 2026-02-21 |
| 04.1 Isolate phase4-specific code | v1.0 | 1/1 | Complete | 2026-02-21 |
| 5. Cross-View Identity and 3D Tracking | v1.0 | 3/3 | Complete | 2026-02-21 |
| 6. 2D Medial Axis and Arc-Length Sampling | v1.0 | 1/1 | Complete | 2026-02-21 |
| 7. Multi-View Triangulation | v1.0 | 1/1 | Complete | 2026-02-22 |
| 8. End-to-End Integration Testing | v1.0 | 2/3 | Complete | 2026-02-23 |
| 9. Curve-Based Optimization | v1.0 | 2/2 | Complete | 2026-02-25 |
| 13. Engine Core | 4/4 | Complete    | 2026-02-25 | - |
| 14. Golden Data and Verification Framework | 2/2 | Complete    | 2026-02-25 | - |
| 15. Stage Migrations | v2.0 | 0/5 | Not started | - |
| 16. Numerical Verification and Legacy Cleanup | v2.0 | 0/2 | Not started | - |
| 17. Observers | v2.0 | 0/5 | Not started | - |
| 18. CLI and Execution Modes | v2.0 | 0/3 | Not started | - |
