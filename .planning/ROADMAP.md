# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- ✅ **v2.0 Alpha** — Phases 13-21 (shipped 2026-02-27)
- ✅ **v2.1 Identity** — Phases 22-28 (shipped 2026-02-28)
- ✅ **v2.2 Backends** — Phases 29-33.1 (shipped 2026-03-01)
- ✅ **v3.0 Ultralytics Unification** — Phases 35-39 (shipped 2026-03-02)
- ✅ **v3.1 Reconstruction** — Phases 40-45 (shipped 2026-03-03)
- 🚧 **v3.2 Evaluation Ecosystem** — Phases 46-50 (in progress)

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

### 🚧 v3.2 Evaluation Ecosystem (In Progress)

**Milestone Goal:** Unified evaluation and parameter tuning system that replaces standalone scripts with `aquapose eval` and `aquapose tune` CLI subcommands, measures stage-specific quality at every pipeline stage, supports single-stage sweeps with proper caching, and removes all legacy evaluation machinery.

- [x] **Phase 46: Engine Primitives** — Per-stage pickle caching and pre-populated context support (completed 2026-03-03)
- [x] **Phase 47: Evaluation Primitives** — ContextLoader, five stage evaluators, and DEFAULT_GRIDS (completed 2026-03-03)
- [x] **Phase 48: EvalRunner and `aquapose eval` CLI** — Multi-stage evaluation report with human-readable and JSON output (completed 2026-03-03)
- [x] **Phase 49: TuningOrchestrator and `aquapose tune` CLI** — Grid sweeps, cascade tuning, and retirement of standalone scripts (completed 2026-03-03)
- [x] **Phase 50: Cleanup and Replacement** — Remove monolithic NPZ machinery and retire harness.py (completed 2026-03-03)

## Phase Details

### Phase 46: Engine Primitives
**Goal**: The pipeline emits per-stage pickle cache files on each StageComplete event, and PosePipeline accepts a pre-populated context to skip upstream stages during sweeps
**Depends on**: Phase 45 (v3.1 complete)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04
**Success Criteria** (what must be TRUE):
  1. Running `aquapose run --mode diagnostic` produces `diagnostics/<stage>_cache.pkl` files alongside existing outputs — one file per pipeline stage
  2. ContextLoader can deserialize any stage's pickle file into a fresh PipelineContext without touching other stages' data
  3. PosePipeline.run() accepts an initial_context parameter and skips stages whose outputs are already populated
  4. Deserializing a pickle file from an incompatible class version raises StaleCacheError with a clear human-readable message
**Plans**: TBD

### Phase 47: Evaluation Primitives
**Goal**: Pure-function stage evaluators for all five pipeline stages return typed metric dataclasses from stage snapshot data, with DEFAULT_GRIDS colocated in evaluator modules for tunable stages
**Depends on**: Phase 46
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, TUNE-06
**Success Criteria** (what must be TRUE):
  1. Each of the five stage evaluators (detection, tracking, association, midline, reconstruction) accepts a stage snapshot and returns a typed frozen dataclass of metrics
  2. No stage evaluator imports from `engine/` — all pipeline config needed (e.g., n_animals) passes as explicit function parameters
  3. DEFAULT_GRIDS for association and reconstruction parameters live in the same module as their evaluators, covering the parameter ranges previously in standalone scripts
  4. All five metric dataclasses can be constructed from synthetic test data without a real pipeline run
**Plans**: TBD

### Phase 48: EvalRunner and `aquapose eval` CLI
**Goal**: Users can evaluate any diagnostic run directory and receive a multi-stage quality report in human-readable or JSON format, replacing the functionality of `scripts/measure_baseline.py`
**Depends on**: Phase 47
**Requirements**: EVAL-06, EVAL-07, CLEAN-03
**Success Criteria** (what must be TRUE):
  1. `aquapose eval <run-dir>` prints a multi-stage quality report to stdout covering all five pipeline stages present in the run directory
  2. `aquapose eval <run-dir> --report json` produces machine-readable JSON output with the same metric content
  3. `scripts/measure_baseline.py` is deleted from the repository — its functionality is fully covered by `aquapose eval`
**Plans**: 2 plans
- [ ] 48-01-PLAN.md — EvalRunner class, EvalRunnerResult dataclass, cache discovery, unit tests
- [ ] 48-02-PLAN.md — Multi-stage report formatters, CLI eval command, delete measure_baseline.py

### Phase 49: TuningOrchestrator and `aquapose tune` CLI
**Goal**: Users can sweep association and reconstruction parameters from the CLI with proper upstream caching, top-N validation, and a config diff block showing recommended changes — and the two standalone tuning scripts are retired
**Depends on**: Phase 48
**Requirements**: TUNE-01, TUNE-02, TUNE-03, TUNE-04, TUNE-05, CLEAN-01, CLEAN-02
**Success Criteria** (what must be TRUE):
  1. `aquapose tune --stage association` executes a grid sweep using per-stage pickle caches as the upstream input, skipping detection, tracking, and midline re-execution for each combo
  2. `aquapose tune --stage reconstruction` executes a grid sweep using association and midline caches as upstream input, skipping those stages for each combo
  3. Both sweep commands support `--n-frames` (fast sweep) and `--n-frames-validate` (thorough validation) with top-N candidates re-evaluated at the higher frame count before a winner is declared
  4. Sweep output includes a before/after metric comparison and a config diff block the researcher can apply manually — no automatic config file mutation occurs
  5. `scripts/tune_association.py` and `scripts/tune_threshold.py` are deleted from the repository
**Plans**: 2 plans
- [ ] 49-01-PLAN.md — TuningOrchestrator class with sweep logic, two-tier validation, and output formatting
- [ ] 49-02-PLAN.md — Wire `aquapose tune` CLI command, update exports, delete legacy scripts

### Phase 50: Cleanup and Replacement
**Goal**: The old evaluation machinery (monolithic NPZ and standalone harness) is removed, leaving the per-stage pickle cache system as the sole evaluation data source
**Depends on**: Phase 49
**Requirements**: CLEAN-04, CLEAN-05
**Success Criteria** (what must be TRUE):
  1. DiagnosticObserver no longer writes `pipeline_diagnostics.npz` automatically — the monolithic NPZ machinery is removed, not deprecated with a shim
  2. `evaluation/harness.py` is deleted — reconstruction evaluation functionality is fully consolidated into `evaluation/stages/reconstruction.py`
  3. All existing tests pass with the legacy evaluation code removed
**Plans**: 1 plan
- [ ] 50-01-PLAN.md — Delete legacy NPZ machinery, harness.py, midline_fixture.py; prune orphaned code; clean up tests

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-9 | v1.0 | 28/28 | Complete | 2026-02-25 |
| 13-21 | v2.0 | 34/34 | Complete | 2026-02-27 |
| 22-28 | v2.1 | 12/12 | Complete | 2026-02-28 |
| 29-33.1 | v2.2 | 12/12 | Complete | 2026-03-01 |
| 35-39 | v3.0 | 14/14 | Complete | 2026-03-02 |
| 40-45 | v3.1 | 13/13 | Complete | 2026-03-03 |
| 46. Engine Primitives | 3/3 | Complete    | 2026-03-03 | - |
| 47. Evaluation Primitives | 3/3 | Complete    | 2026-03-03 | - |
| 48. EvalRunner and eval CLI | 2/2 | Complete    | 2026-03-03 | - |
| 49. TuningOrchestrator and tune CLI | 2/2 | Complete    | 2026-03-03 | - |
| 50. Cleanup and Replacement | 1/1 | Complete   | 2026-03-03 | - |
