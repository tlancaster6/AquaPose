# Feature Research

**Domain:** Evaluation and parameter tuning system for multi-stage computer vision pipeline (AquaPose v3.2)
**Researched:** 2026-03-03
**Confidence:** HIGH — project has a detailed resolved-design seed document; existing code is readable; domain patterns for CV pipeline evaluation/tuning are well-understood.

---

## Context: What Already Exists (Do Not Re-Implement)

The following infrastructure is ALREADY BUILT and must be built upon, not replaced:

| Existing Component | Location | What It Does |
|-------------------|----------|--------------|
| `run_evaluation()` | `evaluation/harness.py` | Loads NPZ fixture, runs reconstruction, computes Tier1/Tier2 metrics |
| `generate_fixture()` | `evaluation/harness.py` | Runs full pipeline with overrides, emits NPZ fixture |
| `Tier1Result`, `Tier2Result` | `evaluation/metrics.py` | Reprojection error + leave-one-out displacement dataclasses |
| `compute_tier1()`, `compute_tier2()` | `evaluation/metrics.py` | Reconstruction metric computation functions |
| `select_frames()` | `evaluation/metrics.py` | Deterministic frame sampling via linspace |
| `format_summary_table()` | `evaluation/output.py` | ASCII summary table |
| `write_regression_json()` | `evaluation/output.py` | JSON regression output |
| `DiagnosticObserver`, `StageSnapshot` | `engine/diagnostic_observer.py` | Per-stage context capture (monolithic NPZ currently) |
| `PosePipeline.run()` | `engine/pipeline.py` | Single-pass pipeline executor |
| `tune_association.py` | `scripts/` | Standalone association sweep (retire after milestone) |
| `tune_threshold.py` | `scripts/` | Standalone reconstruction sweep (retire after milestone) |
| `measure_baseline.py` | `scripts/` | Standalone baseline measurement (retire after milestone) |
| `MidlineFixture`, `CalibBundle` | `io/midline_fixture.py` | NPZ v2.0 serialization |

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features a researcher expects from any evaluation/tuning CLI. Missing these makes the system feel broken or incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `aquapose eval <run-dir>` CLI subcommand | Any evaluation system needs a CLI entrypoint; long-argument scripts are not acceptable for regular use | LOW | Thin Click wrapper; existing CLI already uses Click; extends `aquapose` command group |
| Multi-stage metric report covering all 5 stages | Current system evaluates reconstruction only; researchers need to see where in the pipeline quality degrades | MEDIUM | Five metric evaluator functions needed; detection/tracking/association/midline/reconstruction each get dedicated functions |
| Stage filter flag (`--stage <name>`) for eval | Running all stages is slow during focused debugging; targeted eval is essential | LOW | Simple filtering layer over full metric computation; no structural change needed |
| Human-readable stdout report (tabular) | Researchers want numbers immediately without parsing files | LOW | Extend existing `format_summary_table()` pattern to multi-stage output |
| JSON output flag (`--report json`) | Machine-readable output for scripting, comparison, and CI | LOW | Extend existing `write_regression_json()` to multi-stage structure |
| `aquapose tune --stage <name>` CLI subcommand | Direct replacement for standalone tune scripts; all params, ranges, and logic in one command | MEDIUM | Core sweep loop, param grid loading, stage-specific metric selection, top-N validation |
| Per-stage diagnostic files (replacing monolithic NPZ) | Monolithic NPZ cannot represent partial pipeline runs (when `stop_after` is used); per-stage files enable selective loading during sweep | MEDIUM | DiagnosticObserver refactor: emit per-stage files on `StageComplete`; context loader reads stages selectively |
| Stage-isolated parameter sweep (re-run only target stage per combo) | Sweeping the full pipeline per combo is O(N_combos * full_pipeline_time); upstream caching is the critical efficiency gain | HIGH | Requires context loader (pickle) + PosePipeline accepting pre-populated context; this is the architectural centerpiece of the milestone |
| Two-tier frame counts (fast sweep + thorough validation) | Full validation on every combo wastes GPU time; sweep fast with fewer frames, validate winners with more | LOW | `--n-frames` and `--n-frames-validate` CLI flags; already designed in seed doc; configurable defaults |
| Top-N validation (full pipeline for sweep winners) | Stage-specific metrics do not prove E2E quality; winners must be validated end-to-end | MEDIUM | Configurable N (default 3); validation uses more frames than sweep phase; runs full pipeline for each winner |
| Before/after comparison in tuning output | Researcher needs to know whether tuning actually improved things relative to baseline | LOW | Compare baseline (D0) metrics to winner metrics; shown in final report alongside metric deltas |
| Config diff in tuning output | Researcher needs to know what params changed in order to update their `config.yaml` | LOW | Emit recommended override block alongside metrics table; do NOT auto-mutate the user's config |
| Retire standalone tuning scripts | `tune_association.py`, `tune_threshold.py`, `measure_baseline.py` must be fully subsumed so there is one canonical way to tune | LOW | Delete scripts after confirming CLI covers all functionality; migrate domain knowledge (param grids, scoring logic) into evaluation module |

### Differentiators (Competitive Advantage)

Features that go beyond baseline expectations and provide meaningful research value for this domain.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `aquapose tune --cascade` (association then reconstruction in sequence) | Sequential tuning with locked-in winners is the correct approach for dependent stages; naive independent tuning of each stage gives suboptimal results because association quality directly controls reconstruction input quality | HIGH | Cascade orchestrator: run association sweep → validate winner D1 → use D1 as upstream cache → run reconstruction sweep → validate winner D2 → emit final delta report; complex but architecturally correct |
| Per-stage proxy metrics without ground truth | Most CV evaluation frameworks require annotations; self-consistency metrics (yield, fragmentation rates, reprojection error) work on any production run without labels | HIGH | Each stage gets its own evaluator module with distinct metric logic: detection yield, track length distribution, association fish yield ratio, midline completeness/smoothness, reconstruction reprojection error |
| Pickle-based upstream caching during sweeps | Avoids GPU re-execution of stages upstream of the target; turns O(N_combos * full_pipeline_time) into O(upstream_time + N_combos * target_stage_time) — typically 10-50x speedup for reconstruction sweeps | MEDIUM | Per-stage pickle in tuning work directory; discarded when tuning session ends; session-scoped to avoid stale cache bugs |
| Default sweep grids colocated with metric evaluator functions | Keeps sweep ranges as an evaluation concern, not a pipeline config concern; easy to find, easy to modify without touching pipeline code | LOW | `DEFAULT_GRIDS` dict in each stage's evaluator module; grids cover a reasonable neighborhood around current defaults |
| CLI parameter range override (`--param name --range min:max:step`) | Researcher can probe a specific parameter without editing code | LOW | Parses range string into value list; overrides DEFAULT_GRIDS entry for that param; enables targeted investigation after initial sweep |
| `stop_after` support for partial pipeline execution | Enables "run only stages 1-3 and cache the result" without paying for expensive midline/reconstruction; critical for rapid association iteration | MEDIUM | PosePipeline accepts optional `stop_after: str` parameter naming the last stage to execute; DiagnosticObserver emits what's available; context loader reads only the stages present |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Bayesian or Gaussian process optimization | Promises smarter parameter search than grid search | Obscures what the search is doing; requires additional dependencies (scikit-optimize, Optuna); parameter spaces here are small (2-5 params with defined ranges) and grid search is fully interpretable; Bayesian methods add value only at 10+ params | Grid search with overridable ranges; 2D joint grids for pairs of correlated params if needed |
| Automatic config file mutation on tuning completion | Saves a manual YAML editing step | Mutating user config files without explicit consent destroys reproducibility and is surprising behavior; researcher must review config diff before applying | Print recommended config diff block to stdout; let researcher apply it manually to their config.yaml |
| Real-time / streaming evaluation during pipeline execution | Evaluate quality as each stage completes in a live run | Evaluation currently requires the full stage output to be available; partial evaluation adds synchronization complexity and changes the pipeline's execution model | Evaluate in a separate offline step after the diagnostic run completes; the run directory is the evaluation input |
| Pydantic for sweep config validation | Type-safe validation of sweep parameter dicts | Project decision (PROJECT.md) is frozen dataclasses; Pydantic is explicitly out of scope | Frozen dataclasses + Click type annotations on CLI flags; DEFAULT_GRIDS dicts with documented types |
| Cross-session cache persistence | Reuse upstream caches from previous tuning sessions | Pipeline code may change between sessions, making cached pickles invalid; stale pickles produce silent correctness bugs; seed doc explicitly chose session-scoped caches | Fresh baseline run per tuning session; tuning work directory is discarded when session ends |
| Composite weighted scoring for parameter ranking | Combine multiple metrics into one comparable score | Weights are arbitrary and obscure which metric drove the ranking decision; single primary metric with tiebreaker is more auditable and explainable | Primary metric (fish yield or mean reprojection error) with a single tiebreaker; report all metrics in the output table for human review |
| Automatic cascade state propagation (implicit config mutation mid-cascade) | After association sweep, automatically apply winner params before reconstruction sweep begins | Implicit config changes mid-cascade break auditability; researcher should know what changed and why at each step | Cascade orchestrator manages config propagation internally during the session only; the final report shows the accumulated config diff from D0 to D2 |
| Sweep capability for tracking and midline stages at launch | Complete the full tuning surface for all 5 stages | Seed doc explicitly decided these are evaluate-only at launch: OC-SORT tracking defaults are well-understood; midline params are precision/recall filters that reconstruction's own outlier rejection already handles; adding tuning before confirming it's a bottleneck is speculative over-engineering | Implement evaluate-only for tracking and midline; add sweep support in a future milestone only if evaluation data reveals them as bottlenecks |
| Ground-truth-based metrics | More rigorous than proxy metrics | No ground truth is available at pipeline runtime; training data has annotations but those are consumed during model training, not pipeline evaluation | Self-consistency proxy metrics (yield, reprojection error, cross-view consistency) are always available and sufficient for parameter optimization |
| Retroactive evaluation of pre-v3.2 run directories | Evaluate older runs without re-running the pipeline | Pre-v3.2 run directories use the monolithic NPZ format; the new per-stage file format is not backward-compatible; building a translation layer adds complexity for limited benefit | Not required at launch; the seed doc explicitly defers retroactive compatibility; researcher re-runs the pipeline in diagnostic mode to get a v3.2-compatible run directory |

---

## Feature Dependencies

```
[Per-stage diagnostic files]
    └──requires──> [DiagnosticObserver refactor: emit per-stage files on StageComplete]

[Context loader (pickle deserialization into PipelineContext)]
    └──requires──> [Per-stage diagnostic files]  (knows which stages are available)

[PosePipeline: accept pre-populated PipelineContext + optional stop_after]
    └──minimal change──> existing single-pass architecture preserved

[Stage-isolated parameter sweep]
    └──requires──> [Per-stage diagnostic files]
    └──requires──> [Context loader]
    └──requires──> [PosePipeline: accept pre-populated context]

[Per-stage metric evaluator functions (5 stages)]
    └──builds on──> [Tier1Result, Tier2Result] (existing reconstruction metrics reused)
    └──new work──>  detection, tracking, association, midline metric functions

[aquapose eval <run-dir>]
    └──requires──> [Per-stage metric evaluator functions]
    └──requires──> [Per-stage diagnostic files]  (data source for evaluation)
    └──extends──>  [format_summary_table(), write_regression_json()] (existing output utilities)

[aquapose tune --stage association]
    └──requires──> [Stage-isolated parameter sweep]
    └──requires──> [Per-stage metric evaluator functions: association]
    └──requires──> [Two-tier frame counts]
    └──requires──> [Top-N validation]
    └──subsumes──> [scripts/tune_association.py]

[aquapose tune --stage reconstruction]
    └──requires──> [Stage-isolated parameter sweep]
    └──requires──> [Per-stage metric evaluator functions: reconstruction]
    └──requires──> [Two-tier frame counts]
    └──requires──> [Top-N validation]
    └──subsumes──> [scripts/tune_threshold.py]

[aquapose tune --cascade]
    └──requires──> [aquapose tune --stage association]
    └──requires──> [aquapose tune --stage reconstruction]
    └──requires──> [Cascade orchestrator that sequences the two and threads D1 into reconstruction sweep]

[Retire standalone scripts]
    └──requires──> [aquapose tune --stage association]  (full supersession of tune_association.py)
    └──requires──> [aquapose tune --stage reconstruction]  (full supersession of tune_threshold.py)
    └──requires──> [aquapose eval]  (full supersession of measure_baseline.py)
```

### Dependency Notes

- **Per-stage diagnostic files is the critical prerequisite.** Everything else depends on having per-stage serialized outputs in the run directory. DiagnosticObserver currently writes a monolithic NPZ at `PipelineComplete`; it must emit per-stage pickle/structured files at each `StageComplete` event.

- **PosePipeline change is minimal by design.** The orchestrator manages context population externally. PosePipeline only needs to accept an optional pre-populated context (skipping internal context creation) and an optional `stop_after` stage name. The single-pass execution model is preserved; the orchestrator is the new outer loop.

- **cascade requires both stage sweeps to be complete and independently testable** before the cascade orchestrator can be wired. The cascade orchestrator is thin: call association sweep, take winner, use winner's run directory as the upstream cache for reconstruction sweep, collect final delta report.

- **Retire standalone scripts is the last step.** Scripts should be retained until the CLI has been exercised against the same test cases and produces equivalent results. The migration transfers domain knowledge (param grids, scoring formulas) from scripts into the evaluation module's `DEFAULT_GRIDS` dicts and metric functions.

---

## MVP Definition

### Launch With (v3.2 — all of these are success criteria)

- [ ] Per-stage diagnostic files replacing monolithic NPZ (DiagnosticObserver refactor)
- [ ] Context loader: pickle serialization/deserialization of PipelineContext stage outputs
- [ ] PosePipeline: accept optional pre-populated PipelineContext + optional `stop_after`
- [ ] Per-stage metric evaluator functions for all 5 stages (detection, tracking, association, midline, reconstruction)
- [ ] `aquapose eval <run-dir>` CLI with multi-stage report (stdout human-readable + optional `--report json`)
- [ ] `aquapose tune --stage association` with grid sweep, two-tier frame counts, top-N validation
- [ ] `aquapose tune --stage reconstruction` with grid sweep, two-tier frame counts, top-N validation
- [ ] `aquapose tune --cascade` for association-then-reconstruction in sequence with E2E validation between stages
- [ ] Before/after metric comparison + config diff block in all `tune` output
- [ ] Retire `scripts/tune_association.py`, `scripts/tune_threshold.py`, `scripts/measure_baseline.py`

### Add After Validation (v3.x)

- [ ] `--stage <name>` filter for `aquapose eval` — add once multi-stage eval is working and the filtering need is confirmed
- [ ] `--param name --range min:max:step` CLI override for custom sweep ranges — add when researchers want to narrow in on specific params after initial grid
- [ ] `aquapose tune --stage tracking` — add only if evaluation data reveals tracking fragmentation as a bottleneck
- [ ] `aquapose tune --stage midline` — add only if evaluation data reveals midline completion rate as a bottleneck
- [ ] 2D joint grid sweeps (two correlated params simultaneously) — add when 1D sweeps prove insufficient

### Future Consideration (v4+)

- [ ] Cross-session cache reuse with version tagging to detect stale caches
- [ ] Per-stage metric trending across multiple runs (longitudinal quality tracking as data grows)
- [ ] Sweep results export to CSV/parquet for external analysis or plotting
- [ ] Parallel sweep execution across multiple GPU processes

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| DiagnosticObserver refactor: per-stage files | HIGH | MEDIUM | P1 |
| Context loader (pickle round-trip) | HIGH | MEDIUM | P1 |
| PosePipeline: accept pre-populated context + stop_after | HIGH | LOW | P1 |
| Per-stage metric evaluators (5 stages) | HIGH | HIGH | P1 |
| `aquapose eval` CLI, multi-stage report | HIGH | LOW | P1 |
| `aquapose tune --stage association` | HIGH | MEDIUM | P1 |
| `aquapose tune --stage reconstruction` | HIGH | MEDIUM | P1 |
| Two-tier frame counts | HIGH | LOW | P1 |
| Top-N validation (full pipeline for winners) | HIGH | LOW | P1 |
| Before/after comparison + config diff output | HIGH | LOW | P1 |
| `aquapose tune --cascade` | HIGH | MEDIUM | P1 |
| Retire standalone scripts | MEDIUM | LOW | P1 |
| `--stage` filter for eval | MEDIUM | LOW | P2 |
| `--param name --range` CLI override | MEDIUM | LOW | P2 |
| `aquapose tune --stage tracking` | LOW | MEDIUM | P3 |
| `aquapose tune --stage midline` | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must ship for v3.2 success criteria
- P2: Should add within milestone if time permits
- P3: Future milestone, add only after evaluation reveals bottleneck

---

## Existing Infrastructure Refactor Map

Features in this milestone refactor existing components. The changes required are listed here to inform phase planning.

| Existing Component | Change Required | Scope |
|-------------------|----------------|-------|
| `evaluation/harness.py` | Reconstruction-specific logic migrates to reconstruction stage evaluator; harness becomes a thin orchestrator or is retired | MEDIUM refactor |
| `evaluation/metrics.py` | Tier1/Tier2 types become reconstruction-stage metric types; add 4 new result types (one per remaining stage) | MEDIUM expansion |
| `evaluation/output.py` | Generalize format functions to accept multi-stage metric dicts; extend JSON schema | MEDIUM expansion |
| `engine/diagnostic_observer.py` | Add per-stage file emit on `StageComplete`; monolithic NPZ may be retained for backward compat or retired | MEDIUM refactor |
| `engine/pipeline.py` | Accept optional initial `PipelineContext`; accept optional `stop_after: str` | LOW change |
| `engine/config.py` | No change — sweep ranges live in evaluation module, not config | None |
| `scripts/tune_association.py` | Migrate param grids and metric scoring into `evaluation/`; delete script | Migrate + delete |
| `scripts/tune_threshold.py` | Migrate param grids and metric scoring into `evaluation/`; delete script | Migrate + delete |
| `scripts/measure_baseline.py` | Migrate baseline measurement logic into `aquapose eval`; delete script | Migrate + delete |

---

## Sources

- `.planning/PROJECT.md` — v3.1 state, v3.2 milestone definition, existing decisions (HIGH confidence — primary source)
- `.planning/inbox/evaluation_and_tuning_system.md` — resolved design decisions, CLI design, caching strategy, cascade flow, stage-specific metric definitions (HIGH confidence — primary source)
- `src/aquapose/evaluation/harness.py` — existing reconstruction eval implementation (HIGH confidence — direct code inspection)
- `src/aquapose/evaluation/metrics.py` — Tier1/Tier2 metric result types and computation (HIGH confidence — direct code inspection)
- `src/aquapose/engine/diagnostic_observer.py` — StageSnapshot structure, monolithic NPZ pattern (HIGH confidence — direct code inspection)
- `src/aquapose/engine/pipeline.py` — PosePipeline architecture and single-pass execution model (HIGH confidence — direct code inspection)
- CV pipeline evaluation patterns (stage-isolated sweeps, proxy metrics, cascade tuning) — established patterns in ML system design; analogous to sklearn Pipeline partial-fit patterns and MLflow sweep orchestration (MEDIUM confidence — domain knowledge, not externally verified for this codebase)

---

*Feature research for: AquaPose v3.2 Evaluation Ecosystem — unified `aquapose eval` and `aquapose tune` CLI*
*Researched: 2026-03-03*
