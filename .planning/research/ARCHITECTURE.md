# Architecture Research

**Domain:** Evaluation and parameter tuning system for 3D fish pose estimation pipeline
**Researched:** 2026-03-03
**Confidence:** HIGH (based on direct codebase inspection and seed document)

## Standard Architecture

### System Overview

The new evaluation/tuning system adds an orchestration layer above the existing three. PosePipeline and its stages are not restructured — the orchestrator wraps them.

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLI Layer (src/aquapose/cli.py)                                     │
│  aquapose eval <run-dir>          aquapose tune <config.yaml>        │
│       |                                    |                         │
│       v                                    v                         │
│  EvalRunner                        TuningOrchestrator                │
│  (evaluation/runner.py)            (evaluation/tuning.py)            │
├─────────────────────────────────────────────────────────────────────┤
│  Engine Layer (src/aquapose/engine/)                                 │
│  PosePipeline — accepts optional initial_context (minimal change)    │
│  DiagnosticObserver — writes per-stage pickles instead of NPZ        │
│  PipelineConfig — stop_after already present; no new fields needed   │
├─────────────────────────────────────────────────────────────────────┤
│  Core Layer (src/aquapose/core/)                                     │
│  PipelineContext — no change                                         │
│  5 Stages: Detection, Tracking, Association, Midline, Reconstruction │
├─────────────────────────────────────────────────────────────────────┤
│  Evaluation Layer (src/aquapose/evaluation/)                         │
│  Stage evaluators (one module per stage — pure functions, no engine) │
│  ContextLoader — deserializes pickle caches into PipelineContext     │
│  EvalRunner — reads per-stage files, invokes evaluators, reports     │
│  TuningOrchestrator — sweep loop, cache mgmt, cascade logic         │
│  Metric types (existing Tier1/Tier2 + 4 new stage metric types)     │
│  Reporting (output.py extended for multi-stage and sweep tables)     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | New vs Modified |
|-----------|---------------|-----------------|
| `EvalRunner` | Load per-stage diagnostic files, invoke stage evaluators, format multi-stage report | NEW in `evaluation/runner.py` |
| `TuningOrchestrator` | Sweep loop, cache management, top-N validation, cascade sequencing | NEW in `evaluation/tuning.py` |
| `ContextLoader` | Deserialize per-stage pickle caches into PipelineContext for partial pipeline re-entry | NEW in `evaluation/context_loader.py` |
| `DiagnosticObserver` | Capture stage snapshots — extended to write per-stage pickle files on each StageComplete | MODIFIED in `engine/diagnostic_observer.py` |
| `PosePipeline.run()` | Execute stages — gains optional `initial_context` parameter | MODIFIED in `engine/pipeline.py` |
| Stage evaluators | Pure functions mapping StageSnapshot to metric results | NEW in `evaluation/stages/` |
| `evaluation/metrics.py` | Existing Tier1/Tier2 reconstruction metrics — extended with 4 new stage metric dataclasses | MODIFIED |
| `evaluation/output.py` | Existing ASCII + JSON formatters — extended for multi-stage reports and sweep tables | MODIFIED |
| `evaluation/harness.py` | Existing reconstruction-only harness — refactored to delegate to `stages/reconstruction.py` | MODIFIED (thin facade) |
| `cli.py` | Top-level Click group — gains `eval` and `tune` subcommand groups | MODIFIED |

## Recommended Project Structure

```
src/aquapose/
├── evaluation/
│   ├── __init__.py             # Public API: run_evaluation (existing), EvalRunner, TuningOrchestrator
│   ├── harness.py              # EXISTING — refactored to delegate to stages/reconstruction.py
│   ├── metrics.py              # EXISTING — extended with DetectionMetrics, TrackingMetrics, etc.
│   ├── output.py               # EXISTING — extended with sweep table formatter, multi-stage report
│   ├── runner.py               # NEW — EvalRunner: reads per-stage files, invokes evaluators, reports
│   ├── tuning.py               # NEW — TuningOrchestrator: sweep loop, cache mgmt, cascade logic
│   ├── context_loader.py       # NEW — ContextLoader: pickle -> PipelineContext pre-population
│   └── stages/
│       ├── __init__.py         # Public: all 5 evaluator functions + DEFAULT_GRIDS dicts
│       ├── detection.py        # NEW — evaluate_detection(snapshot) -> DetectionMetrics
│       ├── tracking.py         # NEW — evaluate_tracking(snapshot) -> TrackingMetrics
│       ├── association.py      # NEW — evaluate_association(snapshot, n_animals) -> AssociationMetrics + DEFAULT_GRIDS
│       ├── midline.py          # NEW — evaluate_midline(snapshot) -> MidlineMetrics
│       └── reconstruction.py   # NEW — evaluate_reconstruction(fixture_path, ...) -> ReconstructionMetrics + DEFAULT_GRIDS
│                               #        (logic migrated from harness.py)
├── engine/
│   ├── pipeline.py             # MODIFIED — PosePipeline.run() accepts optional initial_context
│   ├── diagnostic_observer.py  # MODIFIED — per-stage pickle files; monolithic NPZ deprecated
│   └── ...                     # all other engine files unchanged
└── cli.py                      # MODIFIED — adds eval_group and tune_group Click command groups
```

### Structure Rationale

- **`evaluation/stages/`:** Each evaluator is a pure function in its own module. No cross-stage imports. `DEFAULT_GRIDS` for tunable stages (association, reconstruction) live in the same module as their evaluator — sweep ranges are an evaluation concern, not a pipeline concern.
- **`evaluation/runner.py` and `evaluation/tuning.py` as separate modules:** EvalRunner is read-only (consumes existing diagnostic files); TuningOrchestrator runs pipelines and writes caches. Different lifecycles, different test strategies, different responsibilities.
- **`evaluation/context_loader.py`:** Isolated because it is the only module that touches pickle deserialization. Keeping it isolated limits the blast radius if the cache format changes.
- **`engine/pipeline.py` modification is minimal:** Only change is `run(initial_context=None)`. PosePipeline's identity as a single-pass executor is preserved.
- **`evaluation/` stays outside `core/`:** Evaluation code imports from engine (to build stages and run the pipeline). The import boundary rule is `core/ -> nothing from engine/`. `evaluation/` is not `core/` — it is a peer of `engine/` — so it may freely import from both.

## Architectural Patterns

### Pattern 1: Orchestrator over PosePipeline

**What:** TuningOrchestrator manages the sweep loop — try N configs, evaluate, compare, repeat. It calls PosePipeline as a black box and reads DiagnosticObserver output as the bridge between execution and evaluation.

**When to use:** Any logic requiring multiple pipeline runs (sweeps, cascade tuning, future batch/chunk processing).

**Trade-offs:** PosePipeline stays testable with simple mocks. The orchestrator is testable by injecting fake pipeline results. The cost is that each combo requires a full constructor-then-run cycle, which is intentional — stage state must not bleed across combos.

**Example:**
```python
# evaluation/tuning.py
class TuningOrchestrator:
    def sweep_stage(
        self,
        config: PipelineConfig,
        stage_name: str,
        param_grid: dict[str, list],
        n_frames: int,
        cache_dir: Path,
    ) -> SweepResults:
        upstream_context = self._context_loader.load(cache_dir, up_to_stage=stage_name)
        results = []
        for combo in self._expand_grid(param_grid):
            patched_config = self._patch_config(config, stage_name, combo)
            stages = [build_target_stage(patched_config, stage_name)]
            observer = DiagnosticObserver()
            pipeline = PosePipeline(stages=stages, config=patched_config, observers=[observer])
            context = pipeline.run(initial_context=copy.copy(upstream_context))
            snapshot = observer.stages[_STAGE_CLASS_NAMES[stage_name]]
            metrics = evaluate_stage(stage_name, snapshot)
            results.append((combo, metrics))
        return self._rank(results, stage_name)
```

### Pattern 2: PipelineContext Pre-Population via Initial Context

**What:** `PosePipeline.run()` accepts an optional `initial_context: PipelineContext | None = None`. When provided, the pipeline uses it instead of constructing a fresh `PipelineContext()`. Stages that consume upstream fields find them already populated. Stages that produce fields write over any pre-existing value.

**When to use:** Any time upstream stage outputs are cached and only a downstream stage needs to re-run with different parameters.

**Trade-offs:** This is a minimal surgical change to PosePipeline — one conditional at the context initialization point. The alternative (a `start_after` parameter to skip stages explicitly) would require suppressing StageStart/StageComplete events for skipped stages, risking observer bugs. Pre-populated context is simpler — the pipeline does not know or care that some fields were pre-loaded vs computed.

**Critical constraint:** The pre-populated context passed to each combo must be a copy, not the original. Each combo's target stage writes to a field — if they share the same object, combo N's output contaminates combo N+1's input.

**Example (minimal change):**
```python
# engine/pipeline.py — the only change required
def run(self, initial_context: PipelineContext | None = None) -> PipelineContext:
    ...
    # Line 155 currently: context = PipelineContext()
    # Change to:
    context = initial_context if initial_context is not None else PipelineContext()
    ...
```

```python
# evaluation/context_loader.py
import copy
import pickle
from pathlib import Path
from aquapose.core.context import PipelineContext

_STAGE_FIELDS = {
    "detection": ["frame_count", "camera_ids", "detections"],
    "tracking": ["tracks_2d"],
    "association": ["tracklet_groups"],
    "midline": ["annotated_detections"],
    "reconstruction": ["midlines_3d"],
}
_STAGE_ORDER = ["detection", "tracking", "association", "midline", "reconstruction"]

class ContextLoader:
    def load(self, cache_dir: Path, up_to_stage: str) -> PipelineContext:
        """Deserialize cached stage outputs into a fresh PipelineContext copy."""
        ctx = PipelineContext()
        stop_idx = _STAGE_ORDER.index(up_to_stage)
        for stage_name in _STAGE_ORDER[:stop_idx]:
            cache_file = cache_dir / f"{stage_name}_cache.pkl"
            with cache_file.open("rb") as f:
                payload: dict = pickle.load(f)
            for field_name, value in payload.items():
                object.__setattr__(ctx, field_name, value)
        # Shallow copy is safe: upstream fields are frozen by convention
        return copy.copy(ctx)
```

### Pattern 3: Per-Stage Pickle Cache Written by DiagnosticObserver

**What:** After each stage completes, DiagnosticObserver writes a per-stage pickle file (e.g., `diagnostics/detection_cache.pkl`) containing the PipelineContext fields produced by that stage. The monolithic `pipeline_diagnostics.npz` is deprecated in favor of per-stage files.

**When to use:** Always in diagnostic mode. Per-stage files enable selective loading — a reconstruction sweep only needs to load association and midline caches, not re-read detection data.

**Trade-offs:** Pickle is not human-readable, but it is the only format that round-trips Python domain objects (Detection, Tracklet2D, TrackletGroup) without a custom serializer. Pickle is safe here because the tuning orchestrator and pipeline run in the same process and session. The caches are discardable after tuning completes.

**File layout per run directory:**
```
<run_dir>/
  config.yaml                     # unchanged
  diagnostics/                    # NEW subdirectory
    detection_cache.pkl           # {frame_count, camera_ids, detections}
    tracking_cache.pkl            # {tracks_2d}
    association_cache.pkl         # {tracklet_groups}
    midline_cache.pkl             # {annotated_detections}
    reconstruction_cache.pkl      # {midlines_3d}
  midline_fixtures.npz            # unchanged (written by DiagnosticObserver._on_pipeline_complete)
  pipeline_diagnostics.npz        # DEPRECATED — stop writing; retain read API for backward compat
  eval_results.json               # written by aquapose eval
```

**DiagnosticObserver change — new StageComplete handler behavior:**
```python
# Appended to existing on_event handler in engine/diagnostic_observer.py
def _write_stage_cache(self, event: StageComplete) -> None:
    """Write per-stage pickle cache after each stage completes."""
    if self._output_dir is None:
        return
    stage_name_lower = _STAGE_NAME_TO_KEY.get(event.stage_name)
    if stage_name_lower is None:
        return
    cache_dir = self._output_dir / "diagnostics"
    cache_dir.mkdir(exist_ok=True)
    payload = {
        field: getattr(event.context, field)
        for field in _STAGE_OUTPUT_FIELDS[stage_name_lower]
        if getattr(event.context, field, None) is not None
    }
    cache_file = cache_dir / f"{stage_name_lower}_cache.pkl"
    with cache_file.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### Pattern 4: Stage Evaluator as Pure Function

**What:** Each stage evaluator is a pure function that takes a StageSnapshot and returns a typed, frozen metric dataclass. No pipeline, no config, no side effects, no engine imports.

**When to use:** Whenever metrics for a single stage are needed — in `aquapose eval` (reading from disk) and in the tuning sweep (reading from in-memory snapshots). Both code paths feed the same function.

**Trade-offs:** Pure functions are trivially unit-testable with synthetic StageSnapshot data. Reconstruction evaluation needs CalibBundle (not just snapshot fields), so `evaluate_reconstruction` takes additional arguments — this is acceptable, the signature documents the dependency explicitly.

**Example signatures:**
```python
# evaluation/stages/detection.py
@dataclass(frozen=True)
class DetectionMetrics:
    mean_detections_per_frame_per_camera: float
    yield_stability_cv: float       # coefficient of variation in frame-level yield
    per_camera_balance_cv: float    # CV across cameras
    confidence_p50: float           # median confidence
    confidence_p10: float           # 10th percentile confidence

def evaluate_detection(snapshot: StageSnapshot) -> DetectionMetrics: ...

# evaluation/stages/association.py
@dataclass(frozen=True)
class AssociationMetrics:
    fish_yield_ratio: float          # primary metric: n_reconstructable / n_animals
    singleton_rate: float            # tiebreaker: fraction of tracklets unassigned
    mean_cameras_per_fish: float
    cluster_confidence_mean: float

DEFAULT_GRIDS: dict[str, list] = {
    "ray_distance_threshold": [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15],
    "score_min": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
    "eviction_reproj_threshold": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10],
    "leiden_resolution": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    "early_k": [5, 10, 15, 20, 25, 30],
}

def evaluate_association(snapshot: StageSnapshot, n_animals: int) -> AssociationMetrics: ...

# evaluation/stages/reconstruction.py
@dataclass(frozen=True)
class ReconstructionMetrics:
    mean_reprojection_error_px: float  # primary metric
    tier2_max_displacement: float | None
    inlier_ratio_mean: float
    reconstruction_rate: float

DEFAULT_GRIDS: dict[str, list] = {
    "outlier_threshold": [5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
    "min_cameras": [2, 3, 4],
    "inlier_threshold": [30.0, 50.0, 75.0, 100.0],
    "n_control_points": [5, 7, 9],
}

def evaluate_reconstruction(
    fixture_path: Path,
    n_frames: int,
    outlier_threshold: float | None = None,
    skip_tier2: bool = False,
) -> ReconstructionMetrics: ...
```

## Data Flow

### `aquapose eval` Flow

```
aquapose eval <run-dir>
    |
    v
EvalRunner.run(run_dir)
    |
    +-- load diagnostics/detection_cache.pkl     -> StageSnapshot
    |       evaluate_detection(snapshot)         -> DetectionMetrics
    |
    +-- load diagnostics/tracking_cache.pkl      -> StageSnapshot
    |       evaluate_tracking(snapshot)          -> TrackingMetrics
    |
    +-- load diagnostics/association_cache.pkl   -> StageSnapshot
    |       evaluate_association(snapshot, n)    -> AssociationMetrics
    |
    +-- load diagnostics/midline_cache.pkl       -> StageSnapshot
    |       evaluate_midline(snapshot)           -> MidlineMetrics
    |
    +-- load midline_fixtures.npz                -> MidlineFixture
    |       evaluate_reconstruction(fixture)     -> ReconstructionMetrics
    |
    +-- format_multi_stage_report(all_metrics)   -> stdout
            optionally: write eval_results.json
```

### `aquapose tune --stage association` Flow

```
aquapose tune <config.yaml> --stage association --n-frames 30 --n-frames-validate 100
    |
    v
TuningOrchestrator
    |
    +-- Step 1: Baseline full pipeline run (diagnostic mode, n_frames=30)
    |     PosePipeline -> DiagnosticObserver -> writes diagnostics/*.pkl + midline_fixtures.npz
    |     EvalRunner.run(baseline_dir) -> baseline metrics (all 5 stages)
    |
    +-- Step 2: Association sweep (target stage only per combo)
    |     For each param combo in DEFAULT_GRIDS (or CLI override):
    |       ContextLoader.load(baseline_dir, up_to_stage="association") -> upstream_context
    |       PosePipeline(stages=[AssociationStage], initial_context=copy(upstream_context)).run()
    |       evaluate_association(snapshot, n_animals) -> AssociationMetrics
    |     rank by fish_yield_ratio, select top-N (default: top-3)
    |
    +-- Step 3: Validation (top-N combos, full pipeline, n_frames=100)
    |     For each top-N combo:
    |       Full pipeline run (all 5 stages) with patched config
    |       EvalRunner.run(validation_dir) -> all metrics
    |     pick winner by primary metric + tiebreaker
    |
    +-- Step 4: Report
          sweep table (all combos ranked by fish_yield_ratio)
          before/after comparison (baseline vs winner)
          config diff (best params vs current config defaults)
```

### `aquapose tune --cascade` Flow

```
TuningOrchestrator.cascade()
    |
    +-- tune_stage("association") -> winner run D1
    |       (report: association metrics improved? E2E regression?)
    |
    +-- tune_stage("reconstruction", using_cached_run=D1) -> winner run D2
    |       (report: reconstruction metrics improved? E2E regression?)
    |
    +-- final_report(D0 -> D2):
            per-stage metric deltas
            E2E metric delta
            combined config diff (association winner + reconstruction winner)
```

### Key Data Flows

1. **Pickle cache as the tuning backbone:** DiagnosticObserver writes per-stage pickles after every stage completes. TuningOrchestrator reads these to build pre-populated contexts. The pickle payload is the same Python objects already in PipelineContext — no transformation required between write and read.

2. **Midline fixture as the reconstruction evaluation input:** The existing NPZ fixture format (`midline_fixtures.npz`) is the input to reconstruction evaluation. This format is kept unchanged. It is produced by DiagnosticObserver on PipelineComplete when association + midline stages have run.

3. **StageSnapshot as the in-memory evaluation surface:** DiagnosticObserver already builds StageSnapshot objects in memory during a live run. Stage evaluators accept StageSnapshot directly, so the TuningOrchestrator can evaluate without disk round-trips during a sweep.

## Integration Points

### Internal Boundaries

| Boundary | Direction | Rule |
|----------|-----------|------|
| `evaluation/` -> `engine/` | Permitted | `evaluation/` imports `pipeline.py`, `config.py`, `diagnostic_observer.py` |
| `evaluation/` -> `core/` | Permitted | `evaluation/` imports `context.py`, stage types |
| `evaluation/stages/` -> `engine/` | FORBIDDEN | Stage evaluators are pure functions; engine imports would break unit testability |
| `core/` -> `evaluation/` | FORBIDDEN | Import boundary rule IB-001 (AST-checked pre-commit) |
| `engine/` -> `evaluation/` | FORBIDDEN | Import boundary rule IB-002 (AST-checked pre-commit) |
| `cli.py` -> `evaluation/` | Permitted | CLI imports EvalRunner, TuningOrchestrator |

**Note on the import boundary checker:** The AST checker in `tools/import_boundary_checker.py` enforces that `core/` does not import `engine/`. The `evaluation/` package is not `core/` — it sits at the same level as `engine/` under `src/aquapose/`. The checker does not currently restrict `evaluation/` from importing `engine/`, which is correct. The checker may need a new rule (SR-003) if any `evaluation/stages/*.py` file accidentally imports from `engine/` — pure stage evaluators must not.

### Modified Component Contracts

#### `PosePipeline.run()` — minimal change

Current: `def run(self) -> PipelineContext`

New: `def run(self, initial_context: PipelineContext | None = None) -> PipelineContext`

The only code change is at line 155 of the current `pipeline.py`:

```python
# Before:
context = PipelineContext()

# After:
context = initial_context if initial_context is not None else PipelineContext()
```

All observers, event emission, stage dispatch, and error handling are unchanged.

#### `DiagnosticObserver` — extended

Current behavior: in-memory snapshots + `_on_pipeline_complete` writes monolithic NPZ.

New behavior:
- On each `StageComplete` event: write `<output_dir>/diagnostics/<stage>_cache.pkl`.
- On `PipelineComplete`: write `midline_fixtures.npz` as before; stop writing `pipeline_diagnostics.npz`.
- In-memory `self.stages` dict is unchanged — evaluators and tests that use the observer in-process continue to work.

The monolithic NPZ export method (`export_pipeline_diagnostics`) is retained as a public method for backward compatibility but is no longer called automatically.

### New Component Contracts

#### `ContextLoader.load(cache_dir, up_to_stage) -> PipelineContext`

- Reads pickle files for all stages before `up_to_stage` in pipeline order.
- Returns a fresh PipelineContext with those fields populated.
- Returns a shallow copy — the caller's context does not share mutable references with the cache payload.
- Raises `FileNotFoundError` with a clear message if a required upstream cache is missing.
- Stage order constant: `["detection", "tracking", "association", "midline", "reconstruction"]`.

#### `EvalRunner.run(run_dir, stage=None) -> MultiStageReport`

- `run_dir`: path to an existing diagnostic run directory.
- `stage`: optional filter; if provided, evaluate only that stage.
- Loads per-stage pickle files (or midline_fixtures.npz for reconstruction), invokes evaluators, returns structured report.
- Report has `.format_text() -> str` and `.to_json() -> dict` methods.
- Does not run any pipeline stages. Read-only.

#### `TuningOrchestrator` key method contracts

```python
class TuningOrchestrator:
    def sweep_stage(
        self,
        config: PipelineConfig,
        stage_name: str,            # "association" or "reconstruction"
        n_frames: int,              # fast sweep frame count
        n_frames_validate: int,     # validation frame count
        top_n: int = 3,             # how many winners to validate fully
        param_grid: dict[str, list] | None = None,  # None = use DEFAULT_GRIDS
        work_dir: Path | None = None,  # None = tempdir
    ) -> TuningReport: ...

    def cascade(
        self,
        config: PipelineConfig,
        stages: list[str] = ["association", "reconstruction"],
        n_frames: int = 30,
        n_frames_validate: int = 100,
        top_n: int = 3,
    ) -> CascadeReport: ...
```

## Suggested Build Order

Each phase produces working, testable code before the next phase begins.

### Phase 1: PosePipeline pre-population + per-stage pickle caching

**Why first:** Everything else depends on these two primitive capabilities. The sweep loop, the context loader, and the stage evaluators all require either (a) pre-populated context or (b) per-stage pickle files.

Deliverables:
- Modify `PosePipeline.run()` to accept `initial_context` (5-line change).
- Extend `DiagnosticObserver` to write per-stage pickle files on `StageComplete`.
- Write unit tests for both changes.
- After this phase: `aquapose run --mode diagnostic` produces `diagnostics/*.pkl` alongside existing outputs.

**No new user-facing behavior yet. Internal capability only.**

### Phase 2: ContextLoader + stage evaluators (all 5 stages)

**Why second:** Stage evaluators and ContextLoader have no dependency on CLI or TuningOrchestrator. They can be built and unit-tested in isolation with synthetic StageSnapshot data.

Deliverables:
- `evaluation/context_loader.py` — ContextLoader.
- `evaluation/stages/detection.py`, `tracking.py`, `midline.py` (evaluate-only, no DEFAULT_GRIDS).
- `evaluation/stages/association.py` with DEFAULT_GRIDS (migrating sweep ranges from `scripts/tune_association.py`).
- `evaluation/stages/reconstruction.py` with DEFAULT_GRIDS (migrating evaluation logic from `harness.py`).
- Metric result dataclasses for all 5 stages added to `evaluation/metrics.py`.
- Unit tests with synthetic StageSnapshot data for all evaluators.
- ContextLoader integration test using pickle files produced by Phase 1.

**All 5 stage evaluators callable from Python. ContextLoader tested end-to-end.**

### Phase 3: EvalRunner + `aquapose eval` CLI subcommand

**Why third:** EvalRunner assembles Phase 2 components into a user-facing tool. The CLI subcommand is a thin wrapper over EvalRunner.

Deliverables:
- `evaluation/runner.py` — EvalRunner.
- Extend `evaluation/output.py` with multi-stage report formatter and sweep table formatter.
- Add `eval_group` Click command group to `cli.py` with `aquapose eval <run-dir>` and `--stage` filter.
- Integration test: run pipeline in diagnostic mode, then `aquapose eval <run_dir>`, verify report structure.
- Retire `scripts/measure_baseline.py` (its functionality is now `aquapose eval`).

**`aquapose eval <run-dir>` produces a working multi-stage report.**

### Phase 4: TuningOrchestrator + `aquapose tune` CLI subcommand

**Why fourth:** TuningOrchestrator depends on all prior phases. This is the most complex component and should be built last so it can be tested against real pipeline outputs.

Deliverables:
- `evaluation/tuning.py` — TuningOrchestrator with `sweep_stage` and `cascade`.
- Add `tune_group` Click command group to `cli.py`.
- Test single-stage sweep (association, then reconstruction) against real data.
- Test cascade mode end-to-end.
- Retire `scripts/tune_association.py` and `scripts/tune_threshold.py`.

**`aquapose tune --stage association` and `aquapose tune --cascade` fully functional.**

### Phase 5: Cleanup and deprecation

Deliverables:
- Remove write-path for `pipeline_diagnostics.npz` from DiagnosticObserver (retain `export_pipeline_diagnostics` as a public method for backward compatibility, but stop calling it automatically).
- Update `evaluation/harness.py` to delegate to `evaluation/stages/reconstruction.py` or deprecate.
- Update docs and CLAUDE.md CLI command table.

## Anti-Patterns

### Anti-Pattern 1: Sweep Logic Inside PosePipeline

**What people do:** Add a `for config in grid` loop inside PosePipeline, or add a `sweep_mode` parameter to `run()`.

**Why it's wrong:** PosePipeline's contract is "execute stages, emit events, return context." Sweep logic (try N configs, compare results, manage caches) is an outer loop with nothing to do with single-pass execution. Putting it inside PosePipeline couples the single-run path to evaluation concerns and makes both harder to test.

**Do this instead:** TuningOrchestrator wraps PosePipeline. PosePipeline is called N times from the outside. Each call is independent with its own observer and context.

### Anti-Pattern 2: Stage Evaluators Importing from engine/

**What people do:** Import `PipelineConfig` or `build_stages` inside a stage evaluator to reconstruct pipeline configuration during metric computation.

**Why it's wrong:** Stage evaluators are pure functions over data (StageSnapshot contents). Importing engine/ creates circular import risk, violates the evaluator's read-only contract, and makes unit testing the evaluators unnecessarily heavy (requires instantiating pipeline objects to test a metric calculation).

**Do this instead:** All pipeline configuration needed by an evaluator (e.g., `n_animals` for association yield ratio) is passed as an explicit parameter to the evaluator function. Configuration is not re-derived inside the evaluator.

### Anti-Pattern 3: Deep-Copying PipelineContext for Each Sweep Combo

**What people do:** Call `copy.deepcopy(upstream_context)` to isolate each sweep combo's starting state.

**Why it's wrong:** PipelineContext fields contain large numpy arrays (per-frame detections across 13 cameras, thousands of frames). Deep copying these for every parameter combo in a grid search wastes memory proportional to `grid_size * data_size`.

**Do this instead:** Shallow-copy the context (or reconstruct it from the pickle cache for each combo). The freeze-on-populate invariant means stage outputs are immutable by convention — a downstream stage does not mutate an upstream field, it writes a new reference. Shallow copy is safe because each combo's target stage writes a fresh reference to the field it owns.

### Anti-Pattern 4: Monolithic Multi-Stage Pickle

**What people do:** Pickle the entire PipelineContext in one file after the pipeline completes.

**Why it's wrong:** A monolithic pickle requires loading the full context just to access one stage's outputs. A reconstruction sweep only needs association and midline cache — loading detection data for 13 cameras across thousands of frames is wasteful. Per-stage files enable selective loading.

**Do this instead:** Write one pickle per stage as each stage completes (DiagnosticObserver StageComplete handler). Load only the stages needed for the current operation.

### Anti-Pattern 5: Sharing Mutable Context Across Sweep Combos

**What people do:** Reuse the same PipelineContext object across sweep combos, overwriting only the target stage's output field.

**Why it's wrong:** If the target stage writes to multiple fields, or if a bug touches a field it shouldn't, combo N's output can contaminate combo N+1's input. The error is silent and produces wrong metrics without raising any exception.

**Do this instead:** Each combo starts from a fresh context built by `ContextLoader.load()`. The context loader always constructs a new PipelineContext and populates it from the upstream pickle caches. Pickle deserialization is fast for the numpy-backed types used here.

## Sources

- Direct codebase inspection (HIGH confidence):
  - `src/aquapose/engine/pipeline.py` — current PosePipeline.run() implementation
  - `src/aquapose/engine/diagnostic_observer.py` — current snapshot capture and NPZ export
  - `src/aquapose/core/context.py` — PipelineContext and CarryForward contracts
  - `src/aquapose/evaluation/harness.py` — existing reconstruction evaluation harness
  - `src/aquapose/evaluation/metrics.py` — existing Tier1Result, Tier2Result, select_frames
  - `src/aquapose/evaluation/output.py` — existing ASCII and JSON report formatters
  - `src/aquapose/cli.py` — existing Click group structure
  - `src/aquapose/engine/config.py` — PipelineConfig and stop_after field
  - `tools/import_boundary_checker.py` — import boundary rules IB-001 through SR-002
  - `scripts/tune_association.py` — existing sweep ranges and sweep logic to migrate
- Seed document (HIGH confidence — resolved design decisions): `.planning/inbox/evaluation_and_tuning_system.md`
- Project history (HIGH confidence): `.planning/PROJECT.md` — key decisions table and v3.2 milestone target

---
*Architecture research for: AquaPose v3.2 Evaluation Ecosystem milestone*
*Researched: 2026-03-03*
