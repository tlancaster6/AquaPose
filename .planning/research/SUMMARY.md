# Project Research Summary

**Project:** AquaPose v3.2 — Evaluation and Tuning Ecosystem
**Domain:** Evaluation/parameter-tuning CLI for a 5-stage multi-view 3D fish pose estimation pipeline
**Researched:** 2026-03-03
**Confidence:** HIGH

## Executive Summary

AquaPose v3.2 adds a unified evaluation and parameter-tuning system to a pipeline that is working at the reconstruction level. The current state: reconstruction-only evaluation is spread across three standalone scripts (`tune_association.py`, `tune_threshold.py`, `measure_baseline.py`) with no unified CLI, no evaluation of the four non-reconstruction stages, and a DiagnosticObserver that writes a single monolithic NPZ file after the full pipeline completes. The research is grounded in a detailed resolved-design seed document plus direct codebase inspection of every component being modified — confidence is high across all four research areas.

The recommended approach is to build a four-phase orchestration layer above the existing PosePipeline without restructuring the pipeline itself. Two surgical changes to the engine layer enable everything else: extend `DiagnosticObserver` to write per-stage pickle caches on `StageComplete` events (replacing the monolithic NPZ as the evaluation data source), and add an `initial_context` parameter to `PosePipeline.run()` (a five-line change that allows the orchestrator to pre-populate upstream stage outputs). On top of these primitives, a new `evaluation/` module tree provides pure-function stage evaluators, a `ContextLoader`, an `EvalRunner`, and a `TuningOrchestrator`. The result is `aquapose eval <run-dir>` for multi-stage reporting and `aquapose tune --stage <name>` / `aquapose tune --cascade` for parameter optimization with a 10-50x speedup via upstream caching.

The primary architectural risk is correctness around shared context across sweep combos: if the same PipelineContext object is reused across parameter combinations, combo N's output silently contaminates combo N+1's input. The mitigation is to reconstruct from pickle caches for each combo (fast for numpy-backed types) rather than deep-copying (prohibitively slow for 13-camera frame buffers) or sharing (silently wrong). The secondary risk is scope creep — the seed document explicitly defers tracking/midline sweeps, Bayesian optimization, and cross-session cache reuse; those decisions should not be revisited until evaluation data identifies them as bottlenecks.

## Key Findings

### Recommended Stack

The v3.2 milestone introduces no new runtime dependencies. All components build on libraries already present: Python `dataclasses` (frozen), `pickle` (stdlib), `click` (existing CLI), and the existing `evaluation/` module infrastructure. The Ultralytics-based pipeline stack (YOLO11n-seg/pose for midline, YOLO-OBB for detection, OC-SORT for tracking, scipy/scikit-image for triangulation) is fully established from v3.0/v3.1 and is not modified in this milestone.

**Core technologies relevant to v3.2:**
- `pickle` (stdlib): Per-stage cache serialization — chosen because it round-trips Python domain objects (Detection, Tracklet2D, TrackletGroup) without a custom serializer; safe because caches are session-scoped within the same process
- `click` (already in stack): CLI subcommand groups for `aquapose eval` and `aquapose tune`; extends the existing `aquapose` Click group
- Frozen `dataclasses` (stdlib): Metric result types (DetectionMetrics, TrackingMetrics, AssociationMetrics, MidlineMetrics, ReconstructionMetrics) — project decision, Pydantic explicitly out of scope per PROJECT.md
- `copy.copy()` (stdlib): Shallow-copy of pre-populated PipelineContext for sweep combo isolation — safe because upstream stage outputs are immutable by convention (each stage writes a new field reference, never mutates upstream fields)

### Expected Features

**Must have (table stakes — v3.2 success criteria):**
- `DiagnosticObserver` refactor: emit per-stage pickle files on each `StageComplete` — the prerequisite for everything else; replaces the monolithic NPZ as the evaluation data source
- `ContextLoader`: deserialize per-stage pickle caches into a pre-populated `PipelineContext` for sweep combo isolation
- `PosePipeline.run(initial_context=None)`: accept optional pre-populated context; enables stage-isolated re-execution without re-running upstream GPU stages
- Per-stage metric evaluator functions for all 5 stages as pure functions with typed frozen dataclass results
- `aquapose eval <run-dir>` CLI: multi-stage report with human-readable stdout and optional `--report json`
- `aquapose tune --stage association`: grid sweep with two-tier frame counts, top-N validation, before/after comparison, config diff block
- `aquapose tune --stage reconstruction`: same structure as association sweep
- `aquapose tune --cascade`: sequences association sweep then reconstruction sweep, threads the association winner's run directory as the upstream cache for reconstruction, emits combined E2E delta report
- Retire `scripts/tune_association.py`, `scripts/tune_threshold.py`, `scripts/measure_baseline.py` after CLI achieves feature parity

**Should have (add within milestone if time permits):**
- `--stage <name>` filter on `aquapose eval` for focused single-stage debugging
- `--param name --range min:max:step` CLI override for custom sweep ranges

**Defer to v3.x / future:**
- `aquapose tune --stage tracking` — add only if evaluation reveals tracking fragmentation as a bottleneck
- `aquapose tune --stage midline` — add only if evaluation reveals midline completion rate as a bottleneck
- Cross-session cache reuse with version tagging to detect stale caches
- Parallel sweep execution across multiple GPU processes
- Sweep results export to CSV/parquet

### Architecture Approach

The new system adds an orchestration layer above the existing engine without restructuring PosePipeline or its stages. A new `evaluation/` module tree contains all new code: `runner.py` (EvalRunner), `tuning.py` (TuningOrchestrator), `context_loader.py` (ContextLoader), and a `stages/` subpackage with one pure-function evaluator per stage. Existing `evaluation/metrics.py`, `evaluation/output.py`, and `evaluation/harness.py` are extended or refactored as thin facades, not replaced. The import boundary rules enforced by the project's AST pre-commit checker are respected throughout: `core/` and `engine/` do not import from `evaluation/`; stage evaluators in `evaluation/stages/` have zero engine imports.

**Major components:**
1. `EvalRunner` (`evaluation/runner.py`) — read-only; loads per-stage pickle files from a run directory, invokes stage evaluators, formats multi-stage report; no pipeline execution
2. `TuningOrchestrator` (`evaluation/tuning.py`) — manages sweep loop, session-scoped cache directory, top-N validation, cascade sequencing; calls PosePipeline as a black box N times with independent observer and context per combo
3. `ContextLoader` (`evaluation/context_loader.py`) — isolated pickle deserializer; the only module touching pickle round-trips; loads upstream stages into a fresh PipelineContext for each sweep combo
4. Stage evaluators (`evaluation/stages/*.py`) — five pure functions mapping StageSnapshot to typed metric dataclasses; `DEFAULT_GRIDS` for tunable stages (association, reconstruction) colocated in same module as their evaluator
5. `DiagnosticObserver` (modified) — writes `diagnostics/<stage>_cache.pkl` on each `StageComplete`; retains existing in-memory `self.stages` dict and `midline_fixtures.npz` write on `PipelineComplete`
6. `PosePipeline.run()` (minimally modified) — gains `initial_context: PipelineContext | None = None`; one conditional replaces the bare `PipelineContext()` constructor call

### Critical Pitfalls

The PITFALLS.md covers v3.0 Ultralytics Unification pitfalls (training data annotation format, model integration) and v2.2 OBB/keypoint integration pitfalls — both resolved concerns for the v3.1-complete codebase. The following are the critical pitfalls for the v3.2 Evaluation Ecosystem milestone, derived from architecture and feature research:

1. **Shared mutable context across sweep combos** — reusing the same PipelineContext object across sweep combos causes combo N's output to contaminate combo N+1's input silently; no exception is raised, metrics are just wrong. Prevention: `ContextLoader.load()` always constructs a fresh PipelineContext by deserializing from per-stage pickle files for each combo; never pass the same context reference to consecutive pipeline runs.

2. **Deep-copying PipelineContext for each sweep combo** — PipelineContext fields contain large numpy arrays (per-frame detections across 13 cameras, potentially thousands of frames); deep copy per combo makes sweeps prohibitively slow. Prevention: shallow copy is safe because stage outputs are immutable by convention. Use `copy.copy()`, never `copy.deepcopy()`.

3. **Stage evaluators importing from `engine/`** — importing PipelineConfig or stage builder functions inside a stage evaluator creates circular import risk and forces engine instantiation to test a pure metric calculation. Prevention: all pipeline configuration needed by an evaluator (e.g., `n_animals` for association yield ratio) is passed as an explicit function parameter; evaluators have zero engine imports. The import boundary checker may need a new rule (SR-003) to enforce this automatically.

4. **Monolithic pickle replacing the monolithic NPZ** — writing the entire PipelineContext as one pickle file after pipeline completion repeats the current NPZ problem: loading one stage's output requires loading all data. Prevention: per-stage pickle files, one per `StageComplete` event; a reconstruction sweep only needs to load association and midline caches, not re-read detection data across 13 cameras.

5. **Implicit cascade config mutation** — applying association winner params directly to the user's `config.yaml` mid-cascade would mutate a file the user expects to control and break auditability. Prevention: the cascade orchestrator manages config propagation internally during the session only; final output is a printed config diff block the researcher applies manually.

## Implications for Roadmap

Based on combined research, the ARCHITECTURE.md's suggested five-phase build order is well-justified by dependency analysis and should be used as the phase structure directly. Each phase produces working, testable code before the next phase begins.

### Phase 1: Engine Primitives — Per-Stage Pickle Caching and Pre-Populated Context

**Rationale:** Every other component depends on either (a) per-stage pickle files existing in the run directory or (b) PosePipeline accepting a pre-populated context. These are internal capabilities with no user-facing surface — they can be built, tested, and verified before any CLI work begins. Building them first eliminates the risk of discovering a compatibility issue after the orchestrator is already written.

**Delivers:** Modified `DiagnosticObserver` writing `diagnostics/<stage>_cache.pkl` after each stage completes; modified `PosePipeline.run(initial_context=None)` (five-line change); unit tests for both; `aquapose run --mode diagnostic` produces per-stage cache files alongside existing outputs.

**Addresses:** Per-stage diagnostic files (table stakes prerequisite); PosePipeline pre-populated context support (table stakes prerequisite)

**Avoids:** Pitfall 4 (monolithic pickle design); pitfall 1 (shared mutable context — foundation for correct isolation)

### Phase 2: Evaluation Primitives — ContextLoader and Stage Evaluators

**Rationale:** Stage evaluators and ContextLoader have no CLI or TuningOrchestrator dependencies. They can be built and unit-tested with synthetic StageSnapshot data in complete isolation. Building them before the CLI ensures metric logic is solid before wiring to user-facing output. ContextLoader integration testing uses the pickle files produced in Phase 1.

**Delivers:** `evaluation/context_loader.py`; `evaluation/stages/detection.py`, `tracking.py`, `midline.py` (evaluate-only, no DEFAULT_GRIDS); `evaluation/stages/association.py` and `reconstruction.py` with DEFAULT_GRIDS (migrating sweep ranges from standalone scripts); all five metric dataclasses added to `evaluation/metrics.py`; unit tests with synthetic StageSnapshot data; ContextLoader integration test.

**Addresses:** Per-stage metric evaluators for all 5 stages; DEFAULT_GRIDS colocated with evaluators; migration of param grids from standalone scripts

**Avoids:** Pitfall 3 (stage evaluators importing from engine/); pitfall 2 (deep-copying context — ContextLoader design uses shallow copy)

### Phase 3: EvalRunner and `aquapose eval` CLI

**Rationale:** EvalRunner assembles Phase 2 components into the first user-observable output and validates the complete read-path (pickle files → evaluators → formatted report) before the write-path (TuningOrchestrator) is built. Retiring `measure_baseline.py` at this phase confirms migration approach before the heavier tuning work begins.

**Delivers:** `evaluation/runner.py` (EvalRunner); extended `evaluation/output.py` with multi-stage report formatter and sweep table formatter; `aquapose eval <run-dir>` CLI subcommand with `--report json` flag; integration test (pipeline in diagnostic mode → `aquapose eval` → verify report structure); retire `scripts/measure_baseline.py`.

**Addresses:** `aquapose eval` CLI (table stakes); human-readable stdout report; JSON output; retire `measure_baseline.py`

**Avoids:** Scope creep into tuning before eval is independently verified

### Phase 4: TuningOrchestrator and `aquapose tune` CLI

**Rationale:** TuningOrchestrator is the most complex component and depends on all prior phases. Building it last means it can be tested against real pipeline outputs from the start. The cascade feature requires both single-stage sweeps to be independently working and testable before the cascade orchestrator is wired.

**Delivers:** `evaluation/tuning.py` (TuningOrchestrator with `sweep_stage` and `cascade` methods); `aquapose tune --stage association` and `--stage reconstruction` CLI subcommands; `aquapose tune --cascade`; before/after metric comparison and config diff block in all tune output; two-tier frame counts (`--n-frames`, `--n-frames-validate`); top-N validation (full pipeline for sweep winners); retire `scripts/tune_association.py` and `scripts/tune_threshold.py`.

**Addresses:** `aquapose tune` CLI (table stakes); stage-isolated parameter sweep (critical efficiency feature delivering 10-50x speedup); two-tier frame counts; top-N validation; cascade tuning; retire standalone tuning scripts

**Avoids:** Pitfall 1 (shared mutable context — ContextLoader.load() called fresh for each combo); pitfall 5 (implicit cascade mutation — config diff block only, no file mutation)

### Phase 5: Cleanup and Deprecation

**Rationale:** Housekeeping that should not block feature work but should not be skipped. The monolithic NPZ deprecation is gated on all prior phases passing, ensuring no backward compat regression from premature removal. Refactoring `harness.py` as a facade locks in the clean architecture for future milestones.

**Delivers:** Removed automatic write of `pipeline_diagnostics.npz` from DiagnosticObserver (public `export_pipeline_diagnostics()` method retained for backward compat); `evaluation/harness.py` refactored as thin facade delegating to `evaluation/stages/reconstruction.py`; updated CLAUDE.md CLI command table and architecture section.

**Addresses:** Technical debt cleanup; consistent internal architecture for future evaluation milestones

### Phase Ordering Rationale

- **Phases 1 and 2 are strictly prerequisite** to Phases 3 and 4: CLI subcommands have nothing to read or run until pickle caches exist and evaluator functions exist.
- **Phase 3 before Phase 4** ensures the read-path (EvalRunner) is independently validated before the write-path (TuningOrchestrator) is built; retiring one script in Phase 3 confirms migration approach before the harder migration.
- **Phase 5 last** because deprecating the monolithic NPZ before Phase 3 integration tests pass could break those tests; deprecation is safe only after the full new pipeline is verified end-to-end.
- **Feature dependency chain from FEATURES.md confirms this order:** per-stage diagnostic files → ContextLoader → stage-isolated sweep → `aquapose tune`; `aquapose eval` → retire `measure_baseline.py`.

### Research Flags

Phases with well-documented patterns (skip research-phase, implement directly):
- **Phase 1:** Surgical changes to two engine components with exact code examples in ARCHITECTURE.md — the specific line to change in `pipeline.py` and the exact DiagnosticObserver handler are already specified.
- **Phase 2:** Pure function evaluators and ContextLoader are data transformations; association and reconstruction metric definitions are migrated from existing scripts; detection/tracking/midline metric definitions are fully specified in FEATURES.md.
- **Phase 3:** EvalRunner and Click extension follow established patterns already in the codebase; `format_summary_table()` and `write_regression_json()` extension paths are specified in ARCHITECTURE.md.
- **Phase 5:** Deprecation and documentation; no novel design decisions.

Phases that may benefit from a targeted pre-implementation sketch during planning:
- **Phase 4 (TuningOrchestrator):** The cascade orchestrator's internal config propagation (threading D1 as the upstream cache for reconstruction sweep) and the two-tier frame count logic have the most moving parts. A brief implementation sketch before coding would reduce risk of context contamination bugs. The seed document specifies the design but the implementation details around `work_dir` management and `copy.copy()` timing warrant explicit pre-planning.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies; all technologies are already in the codebase and working; no version constraints introduced |
| Features | HIGH | Grounded in a detailed resolved-design seed document plus direct inspection of every component being modified; feature boundaries explicitly specified including anti-features and deferral rationale |
| Architecture | HIGH | Based on direct inspection of every component being modified; component contracts, data flow diagrams, and code examples with specific line numbers are all specified in ARCHITECTURE.md |
| Pitfalls | MEDIUM | PITFALLS.md covers v3.0/v2.2 risks (now historical for v3.1-complete codebase); v3.2-specific pitfalls derived from architecture analysis; the five identified risks are structurally sound but not cross-validated against documented precedents in analogous systems |

**Overall confidence:** HIGH

### Gaps to Address

- **v3.2-specific pitfall completeness:** The five pitfalls identified for this milestone are derived from architecture analysis. They are well-reasoned but would benefit from a review of analogous sweep orchestration patterns (sklearn Pipeline, MLflow sweep runs) to confirm no edge cases were missed. Low urgency — mitigations are already specified.

- **Association DEFAULT_GRIDS calibration:** Sweep ranges in `evaluation/stages/association.py` are migrated from `scripts/tune_association.py`. These ranges should be verified against the YH project's known-good parameter neighborhood before the first tuning run, to confirm the grid covers the optimal region. Validation step, not a design gap.

- **`stop_after` field confirmation:** ARCHITECTURE.md states `stop_after` is already present in `PipelineConfig`; this should be confirmed at the start of Phase 4 planning to determine whether TuningOrchestrator can use it directly for upstream-only pipeline runs or whether additional logic is needed.

## Sources

### Primary (HIGH confidence)

- `.planning/inbox/evaluation_and_tuning_system.md` — resolved design seed document; CLI design, caching strategy, cascade flow, stage-specific metric definitions, anti-feature rationale (HIGH — primary design authority)
- `.planning/PROJECT.md` — v3.1 completion state, v3.2 milestone definition, existing decisions table
- `src/aquapose/evaluation/harness.py` — existing reconstruction eval implementation; migration target
- `src/aquapose/evaluation/metrics.py` — existing Tier1Result, Tier2Result, select_frames
- `src/aquapose/evaluation/output.py` — existing ASCII and JSON report formatters
- `src/aquapose/engine/pipeline.py` — current PosePipeline.run() implementation; specific change point identified
- `src/aquapose/engine/diagnostic_observer.py` — current snapshot capture and monolithic NPZ export; modification target
- `src/aquapose/core/context.py` — PipelineContext and CarryForward contracts
- `src/aquapose/cli.py` — existing Click group structure; extension point
- `src/aquapose/engine/config.py` — PipelineConfig and stop_after field
- `tools/import_boundary_checker.py` — import boundary rules IB-001 through SR-002
- `scripts/tune_association.py` — sweep ranges and scoring logic to migrate

### Secondary (MEDIUM confidence)

- Full `src/aquapose/` codebase — pipeline architecture, stage interface contracts, type system
- CV pipeline evaluation patterns — stage-isolated sweeps, proxy metrics without ground truth, cascade tuning; analogous to sklearn Pipeline partial-fit patterns and MLflow sweep orchestration (domain knowledge, not externally verified for this codebase)

### Tertiary (historical context only)

- `.planning/research/PITFALLS.md` — v3.0 Ultralytics Unification and v2.2 OBB/keypoint integration pitfalls; relevant for training infrastructure context but now historical for the v3.1-complete codebase; structural patterns informed v3.2-specific pitfall analysis

---
*Research completed: 2026-03-03*
*Ready for roadmap: yes*
