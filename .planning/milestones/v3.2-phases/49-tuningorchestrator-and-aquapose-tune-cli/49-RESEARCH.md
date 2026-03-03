# Phase 49: TuningOrchestrator and `aquapose tune` CLI - Research

**Researched:** 2026-03-03
**Domain:** CLI command design, parameter sweep orchestration, stage cache reuse, Python/Click
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Sweep strategy:**
- Association: Joint 2D grid over `ray_distance_threshold x score_min` (these interact via the soft scoring kernel), then sequential carry-forward for remaining params (`eviction_reproj_threshold`, `leiden_resolution`, `early_k`)
- Reconstruction: 1D grid per param, sequential carry-forward (`outlier_threshold` first, then `n_points`)
- No early stopping: Always sweep all parameter stages regardless of intermediate results
- Grids are hardcoded defaults: Use `ASSOCIATION_DEFAULT_GRID` and `RECONSTRUCTION_DEFAULT_GRID` from `evaluation/stages/`. No CLI flags for custom ranges — researchers edit the code if needed

**Two-tier validation:**
- Fast sweep: Tier 1 only (`skip_tier2=True`) at `--n-frames` count (default 30)
- Top-N validation: Full Tier 1 + Tier 2 at `--n-frames-validate` count (default 100) for the top candidates
- Re-run target stage only during validation — upstream stage caches are reused, only the swept stage is re-executed with candidate params at the higher frame count
- Default `--top-n` is 3

**Output and config diff:**
- Winner vs baseline only: One clean before/after comparison table (yield, mean error, max error, singleton rate, tier 2 stability) with deltas
- 2D yield matrix: Print the joint grid yield % matrix for association sweeps (shows parameter interaction patterns)
- Progress lines always: Print per-combo result line as each completes (yield, error, etc.) — no quiet mode needed
- Config diff as YAML snippet: Print a ready-to-paste YAML block showing only changed keys and new values, matching the config.yaml structure

**CLI design:**
- Command: `aquapose tune --stage association` or `aquapose tune --stage reconstruction`
- `--config` / `-c` required: Path to the run-generated config YAML (exhaustive, not the minimalist user config). The config's parent directory IS the run directory
- No --run or --resume-from: The orchestrator infers the run directory from the config file's parent path and auto-discovers stage cache pickles from the `stages/` subdirectory
- Frame count flags: `--n-frames` (fast sweep, default 30), `--n-frames-validate` (thorough validation, default 100)
- `--top-n` (default 3): Number of candidates for full validation

### Claude's Discretion
- Internal TuningOrchestrator class design and method decomposition
- How stage cache discovery works (glob pattern, naming convention)
- Scoring function design for ranking candidates
- Console formatting details (column widths, separators)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TUNE-01 | `aquapose tune --stage association` sweeps association parameters using grid search with fish yield as primary metric | `evaluate_association()` + `ASSOCIATION_DEFAULT_GRID` already exist; `EvalRunner` provides cache loading pattern; joint 2D grid code from `tune_association.py` can be ported directly |
| TUNE-02 | `aquapose tune --stage reconstruction` sweeps reconstruction parameters using grid search with mean reprojection error as primary metric | `evaluate_reconstruction()` + `RECONSTRUCTION_DEFAULT_GRID` already exist; reconstruction cache (`reconstruction_cache.pkl`) holds `midlines_3d` data; scoring is 1D sequential |
| TUNE-03 | Two-tier frame counts: configurable fast-sweep and thorough-validation frame counts via CLI flags | `select_frames()` in `metrics.py` handles deterministic sampling; `EvalRunner.run(n_frames=N)` pattern is established; `skip_tier2` kwarg maps directly to fast-vs-validate |
| TUNE-04 | Top-N validation runs full pipeline for sweep winners to verify E2E quality | Top-N collection pattern exists in `tune_association.py`; for association re-runs the association stage with cached upstream (detection+tracking from cache); for reconstruction re-runs with association+midline cache |
| TUNE-05 | Tuning output includes before/after metric comparison and recommended config diff block | Comparison table pattern exists in `_print_final_report()` in `tune_association.py`; YAML diff block is new but simple: compare winner params vs config values; use `yaml.dump()` |
| CLEAN-01 | `scripts/tune_association.py` retired after `aquapose tune --stage association` achieves feature parity | Script currently uses NPZ fixtures + `generate_fixture()` (old harness); new implementation uses per-stage pickle caches + `EvalRunner` pattern; feature parity must be verified before deletion |
| CLEAN-02 | `scripts/tune_threshold.py` retired after `aquapose tune --stage reconstruction` achieves feature parity | Script sweeps `outlier_threshold` sequentially; `RECONSTRUCTION_DEFAULT_GRID` already defines that same sweep range plus `n_points`; feature parity extends coverage |
</phase_requirements>

## Summary

Phase 49 builds a `TuningOrchestrator` class and wires it to a new `aquapose tune` Click command in `cli.py`. The key difference from the legacy scripts is that the new implementation uses per-stage pickle caches (written by `DiagnosticObserver`) rather than re-running the full pipeline for every parameter combination. This means for association sweeps, only the association+reconstruction stages re-execute per combo (detection+tracking caches are loaded once); for reconstruction sweeps, only reconstruction re-executes (detection+tracking+association+midline caches are loaded once).

The infrastructure for this phase is largely complete: `ASSOCIATION_DEFAULT_GRID`, `RECONSTRUCTION_DEFAULT_GRID`, `evaluate_association()`, `evaluate_reconstruction()`, `load_stage_cache()`, `EvalRunner` (cache discovery pattern), and the Click CLI group all exist. The primary work is: (1) designing `TuningOrchestrator` that orchestrates combo sweeps using cached upstream data, (2) implementing the two-tier validation loop, (3) formatting output with the comparison table + YAML diff block, and (4) wiring the `tune` command in `cli.py`.

The legacy scripts (`tune_association.py`, `tune_threshold.py`) use the old NPZ fixture + `generate_fixture()` harness, which re-runs the ENTIRE pipeline for each parameter combination. The new architecture skips upstream stages using pickle caches, making sweeps dramatically faster for reconstruction (only the DLT pass re-runs) and somewhat faster for association (detection and tracking are skipped).

**Primary recommendation:** Place `TuningOrchestrator` in `src/aquapose/evaluation/tuning.py`; register the `tune` CLI command in `cli.py` as a thin wrapper; delete the two legacy scripts only after the new command passes an integration smoke-test.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| click | (project dep) | CLI command definition, option parsing | Already used for all CLI commands; `@cli.command("tune")` pattern is established |
| yaml | (project dep via PyYAML) | Printing config diff block | Already used in `cli.py` (`import yaml`) for config serialization |
| dataclasses | stdlib | `dataclasses.replace()` for config patching | Already used in `harness.py::generate_fixture()` and throughout config layer |
| pathlib.Path | stdlib | File discovery, cache path resolution | Already used everywhere in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| aquapose.evaluation.stages | (internal) | `evaluate_association`, `evaluate_reconstruction`, grids | Called per-combo to compute metrics from cached context |
| aquapose.core.context | (internal) | `load_stage_cache`, `StaleCacheError` | Load upstream stage caches before sweep |
| aquapose.evaluation.metrics | (internal) | `select_frames` | Frame sampling for fast sweep vs validation |
| aquapose.engine.config | (internal) | `load_config`, `AssociationConfig`, `ReconstructionConfig` | Patch stage config with sweep params; read baseline values for diff |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `evaluation/tuning.py` | `engine/tuning.py` | Context decision says evaluation code stays in `evaluation/`; stage evaluators are already there |
| Inline sweep in cli.py | Separate TuningOrchestrator class | TuningOrchestrator is testable in isolation; CLI stays thin per project pattern |

## Architecture Patterns

### Recommended Project Structure

```
src/aquapose/evaluation/
├── tuning.py        # NEW: TuningOrchestrator class (association + reconstruction sweepers)
├── runner.py        # existing EvalRunner (unchanged)
├── stages/          # existing evaluators + grids (unchanged)
│   ├── association.py   # ASSOCIATION_DEFAULT_GRID, evaluate_association()
│   └── reconstruction.py  # RECONSTRUCTION_DEFAULT_GRID, evaluate_reconstruction()
└── __init__.py      # update to export TuningOrchestrator

src/aquapose/cli.py  # add @cli.command("tune") at end

scripts/
├── tune_association.py   # DELETE after feature parity
└── tune_threshold.py     # DELETE after feature parity
```

### Pattern 1: Cache-Based Stage Re-execution

The fundamental design for the tuning sweep is:

1. Load the upstream stage cache (e.g., `midline_cache.pkl` for reconstruction sweep)
2. Patch only the swept stage's config parameters via `dataclasses.replace()`
3. Build a partial pipeline (only the swept stage + any downstream stage for metrics)
4. Run the partial pipeline with `initial_context` = loaded upstream cache
5. Extract metrics from resulting context

**For association sweep** — upstream data needed: `tracking_cache.pkl`
- Load `tracking_cache.pkl` → `PipelineContext` with `tracks_2d` populated
- Patch `AssociationConfig` with combo params → build `AssociationStage(config)`
- Run `AssociationStage` → produces `tracklet_groups`
- Run `MidlineStage` → produces `annotated_detections`
- Run `ReconstructionStage` → produces `midlines_3d`
- Call `evaluate_association()` and `evaluate_reconstruction()` on results

**IMPORTANT DESIGN CHOICE**: Looking at the legacy script more carefully, it uses `generate_fixture()` which runs the full pipeline and then calls `run_evaluation()` on the fixture. The NEW approach skips upstream re-execution. For association tuning, the association, midline, and reconstruction stages must re-run from the tracking cache. For reconstruction tuning, only reconstruction re-runs from the midline cache.

However, this requires running partial pipelines with the `PosePipeline` engine or calling stage `.run()` methods directly. The simpler and more robust approach (matching `EvalRunner` pattern) is:
- Load the appropriate upstream cache into `PipelineContext`
- Call the swept stage's `.run(context)` directly (NOT through `PosePipeline`, to avoid writing run artifacts)
- For association sweep: load tracking cache → run AssociationStage → run MidlineStage → run ReconstructionStage → evaluate
- For reconstruction sweep: load midline cache → run ReconstructionStage (with patched params) → evaluate

This avoids the `PosePipeline`'s output_dir / config.yaml writing overhead while still using the real stage implementations.

**Stage parameter patching** (confirmed from harness.py and config.py):
```python
import dataclasses
patched_assoc = dataclasses.replace(config.association, ray_distance_threshold=val)
patched_config = dataclasses.replace(config, association=patched_assoc)
assoc_stage = AssociationStage(patched_config)
```

For `ReconstructionStage`, the constructor signature differs — check `build_stages()`:
```python
reconstruction_stage = ReconstructionStage(
    calibration_path=config.calibration_path,
    backend=config.reconstruction.backend,
    min_cameras=config.reconstruction.min_cameras,
    max_interp_gap=config.reconstruction.max_interp_gap,
    n_control_points=config.reconstruction.n_control_points,
)
```
Note: `outlier_threshold` and `n_points` are NOT passed to `ReconstructionStage` constructor — they appear to be on the backend object. This requires investigation during implementation.

### Pattern 2: Cache Discovery

Cache files are in `<run_dir>/diagnostics/<stage>_cache.pkl`. Run dir is `config_path.parent`. Discovery:
```python
def _discover_cache(run_dir: Path, stage: str) -> Path:
    return run_dir / "diagnostics" / f"{stage}_cache.pkl"
```

This matches `EvalRunner._discover_caches()` exactly.

### Pattern 3: CLI Registration (from cli.py)

```python
@cli.command("tune")
@click.option("--stage", "-s", required=True,
    type=click.Choice(["association", "reconstruction"], case_sensitive=False))
@click.option("--config", "-c", required=True,
    type=click.Path(exists=True))
@click.option("--n-frames", "n_frames", type=int, default=30)
@click.option("--n-frames-validate", "n_frames_validate", type=int, default=100)
@click.option("--top-n", "top_n", type=int, default=3)
def tune_cmd(stage: str, config: str, n_frames: int, n_frames_validate: int, top_n: int) -> None:
    """Sweep stage parameters and output a recommended config diff."""
    from aquapose.evaluation.tuning import TuningOrchestrator
    from aquapose.core.context import StaleCacheError
    ...
```

### Pattern 4: Scoring (from tune_association.py)

Association scoring (carry forward to new code):
```python
def _compute_score(assoc_metrics: AssociationMetrics, recon_metrics: ReconstructionMetrics) -> tuple[float, float]:
    # Primary: negative fish yield (higher yield = better)
    # Secondary: mean reprojection error (lower = better)
    return (-assoc_metrics.fish_yield_ratio, recon_metrics.mean_reprojection_error)
```

Reconstruction scoring: primary is mean reprojection error, secondary is yield (fewer fish dropped by outlier threshold = better).

### Pattern 5: Config Diff Block (YAML snippet output)

```python
def _format_config_diff(stage: str, winner_params: dict[str, float], baseline_config) -> str:
    """Print YAML block of changed params matching config.yaml structure."""
    changed = {}
    for k, v in winner_params.items():
        baseline_val = getattr(baseline_config, k)
        if abs(v - baseline_val) > 1e-9:
            changed[k] = v
    if not changed:
        return f"{stage}:  # No changes recommended"
    return f"{stage}:\n" + yaml.dump(changed, default_flow_style=False, indent=2)
```

### Anti-Patterns to Avoid
- **Running PosePipeline for each combo**: Wastes time writing config.yaml artifacts, setting up observers, etc. Call stage `.run()` directly.
- **Importing from engine at module level in tuning.py**: Follow the inline import pattern from `EvalRunner._read_n_animals()` to avoid coupling.
- **Mutating stage caches between combos**: Each combo must get a fresh copy of the upstream context. Use `copy.copy()` (shallow is fine per project decision: stage outputs are immutable by convention).
- **Generating NPZ fixtures per combo**: Legacy approach; entirely replaced by pickle cache approach.
- **Automatic config file mutation**: Explicitly prohibited; YAML snippet is print-only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Frame sampling | Custom linspace | `select_frames()` from `evaluation/metrics.py` | Already tested, deterministic |
| Cache loading | Custom pickle load | `load_stage_cache()` from `core/context.py` | Handles envelope format, StaleCacheError |
| Config patching | Custom dict merge | `dataclasses.replace()` | Standard pattern used throughout project |
| Grid definitions | Hardcoded in tuning.py | `ASSOCIATION_DEFAULT_GRID` / `RECONSTRUCTION_DEFAULT_GRID` from `evaluation/stages/` | TUNE-06 already complete; grids are there |
| YAML output | Custom string format | `yaml.dump()` | Already imported in cli.py |
| Association metrics | Custom computation | `evaluate_association()` from `evaluation/stages/association.py` | EVAL-03 complete |
| Reconstruction metrics | Custom computation | `evaluate_reconstruction()` from `evaluation/stages/reconstruction.py` | EVAL-05 complete |

## Common Pitfalls

### Pitfall 1: ReconstructionStage Constructor vs Config Fields

**What goes wrong:** `outlier_threshold` and `n_points` (or `n_sample_points`) may not be constructor parameters of `ReconstructionStage` — they may live on the `DltBackend` object built inside the stage. The `build_stages()` factory in `pipeline.py` does NOT pass `outlier_threshold` to `ReconstructionStage` constructor directly.

**Why it happens:** `ReconstructionStage` likely constructs the `DltBackend` internally using `config.reconstruction`. To patch `outlier_threshold`, the implementation may need to pass a patched `ReconstructionConfig` to `ReconstructionStage` or construct the backend directly.

**How to avoid:** Inspect `ReconstructionStage.__init__` signature during planning/implementation before designing the sweep loop. The sweep may need to construct `DltBackend` directly rather than going through `ReconstructionStage`.

**Warning signs:** If `outlier_threshold` is not in the `ReconstructionStage` constructor, verify whether it's accepted via the config object or via the backend constructor.

### Pitfall 2: Context Mutation Between Combos

**What goes wrong:** Loading the upstream cache once and reusing the same `PipelineContext` object for all combos. If any stage mutates the context in place, results from combo N bleed into combo N+1.

**Why it happens:** `PipelineContext` is a plain `@dataclass` (not frozen), so in-place mutation is possible.

**How to avoid:** Call `copy.copy(upstream_ctx)` before passing to the swept stage for each combo. Per the Phase 46 decision, shallow copy is sufficient because stage outputs are immutable by convention.

### Pitfall 3: MidlineStage Needs to Re-run for Association Sweeps

**What goes wrong:** For association sweeps, re-running only `AssociationStage` then calling `evaluate_association()` directly on `tracklet_groups` is sufficient for the fast sweep tier. But top-N validation needs full Tier 2 metrics (from `evaluate_reconstruction()` with `tier2_result`), which requires running the full association→midline→reconstruction chain.

**Why it happens:** The new `evaluate_association()` operates on `MidlineSet` data (from midline stage output), not raw tracklet groups. The `EvalRunner._build_midline_sets()` requires `annotated_detections` from the midline cache AND `tracklet_groups` from association.

**How to avoid:** For each combo in the sweep:
1. Run `AssociationStage(patched_config).run(ctx_copy)` to get `tracklet_groups`
2. Run `MidlineStage.run(ctx)` to get `annotated_detections`
3. Build `MidlineSet` via `_build_midline_sets()` logic → call `evaluate_association()`
4. Run `ReconstructionStage.run(ctx)` → call `evaluate_reconstruction()`

However, running MidlineStage per combo negates the cache speedup. **Alternative:** Since the midline cache already has `annotated_detections` from the original run, for association sweeps we only need to re-run `AssociationStage` to get new `tracklet_groups`, then use the EXISTING `annotated_detections` from the midline cache to build `MidlineSet`. This is the correct cache-based approach.

The scoring can then use:
- `evaluate_association(midline_sets, n_animals)` — builds from new tracklet_groups + existing midline cache
- `evaluate_reconstruction(frame_results)` — needs to re-run reconstruction on the new midline_sets (since midline_sets change when tracklet_groups change)

### Pitfall 4: Cache Path Discovery

**What goes wrong:** The CONTEXT.md says caches are in `stages/` subdirectory, but the code consistently uses `diagnostics/` (confirmed in `diagnostic_observer.py`, `runner.py`).

**Root cause:** CONTEXT.md says "auto-discovers stage cache pickles from the `stages/` subdirectory" but the actual path is `<run_dir>/diagnostics/<stage>_cache.pkl`.

**How to avoid:** Always use `run_dir / "diagnostics" / f"{stage}_cache.pkl"` — consistent with `EvalRunner._discover_caches()`.

### Pitfall 5: Association Grid Uses `early_k` as float

**What goes wrong:** `ASSOCIATION_DEFAULT_GRID["early_k"]` is `[5.0, 10.0, 15.0, 20.0, 25.0, 30.0]` (floats) because `dict[str, list[float]]` was chosen as the type. But `AssociationConfig.early_k` is `int`. Passing `float` to `dataclasses.replace` for an `int` field causes a type error or silent coercion.

**How to avoid:** Cast `early_k` values to `int` before passing to `dataclasses.replace()`. This matches the Phase 47 P02 decision: "early_k values stored as float [5.0, 10.0, ...] to satisfy dict[str, list[float]] type; source uses int."

### Pitfall 6: `n_points` in RECONSTRUCTION_DEFAULT_GRID

**What goes wrong:** `RECONSTRUCTION_DEFAULT_GRID` has key `"n_points"` but `ReconstructionConfig` uses `n_sample_points`. The grid key name may not match the config field name.

**Why it happens:** Looking at `reconstruction.py`'s DEFAULT_GRID: `"n_points": [7.0, 11.0, 15.0, 21.0]`. But `ReconstructionConfig` has `n_sample_points: int = 15` and `n_control_points: int = 7`. Neither is named `n_points`.

**How to avoid:** During implementation, verify the mapping from grid key to config field. The `n_points` grid key likely maps to `n_sample_points` in `ReconstructionConfig`. This must be resolved — likely a field name mismatch introduced in Phase 47.

### Pitfall 7: Association Stage Run Signature

**What goes wrong:** Assuming `AssociationStage.run(context)` → returns `PipelineContext`. Per `pipeline.py`, `TrackingStage` has a different signature `(context, carry) -> (context, carry)` but `AssociationStage` uses the standard `Stage` protocol.

**How to avoid:** Only `TrackingStage` has the special signature. `AssociationStage` follows the standard `run(context) -> context` protocol.

## Code Examples

### Loading an upstream cache and running one stage

```python
# Source: pattern from EvalRunner._discover_caches() + harness.py generate_fixture()
from aquapose.core.context import load_stage_cache, StaleCacheError
from aquapose.engine.config import load_config
import dataclasses
import copy

run_dir = config_path.parent
tracking_cache = run_dir / "diagnostics" / "tracking_cache.pkl"
upstream_ctx = load_stage_cache(tracking_cache)

# Patch association config
config = load_config(config_path)
patched_assoc = dataclasses.replace(config.association, ray_distance_threshold=0.04)
patched_config = dataclasses.replace(config, association=patched_assoc)

# Run swept stage on a copy (shallow copy per Phase 46 decision)
from aquapose.core.association import AssociationStage
ctx_copy = copy.copy(upstream_ctx)
result_ctx = AssociationStage(patched_config).run(ctx_copy)
```

### 2D joint grid yield matrix printing (from tune_association.py, preserve in new code)

```python
def _print_joint_grid_matrix(param_a, values_a, param_b, values_b, results):
    yield_map = {}
    for overrides, assoc_metrics, recon_metrics in results:
        va, vb = overrides[param_a], overrides[param_b]
        yield_map[(va, vb)] = 100.0 * assoc_metrics.fish_yield_ratio

    col_w = 7
    row_label_w = 8
    header_parts = [f"{'':>{row_label_w}}"]
    for vb in values_b:
        header_parts.append(f"{vb:>{col_w}.3g}")
    print(f"  " + " ".join(header_parts))

    for va in values_a:
        row_parts = [f"{va:>{row_label_w}.3g}"]
        for vb in values_b:
            pct = yield_map.get((va, vb), float("nan"))
            cell_str = f"{pct:.0f}%"
            row_parts.append(f"{cell_str:>{col_w}}")
        print("  " + " ".join(row_parts))
```

### YAML config diff output

```python
import yaml

def format_config_diff(stage_name: str, winner_params: dict, baseline_stage_config) -> str:
    """Format winner params as a YAML snippet for manual application."""
    changed = {
        k: (int(v) if isinstance(v, float) and v == int(v) else v)
        for k, v in winner_params.items()
        if abs(float(v) - float(getattr(baseline_stage_config, k))) > 1e-9
    }
    if not changed:
        return f"{stage_name}:  # no changes — baseline is already optimal\n"
    return f"{stage_name}:\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(changed.items())) + "\n"
```

### Final comparison table

```python
def _print_comparison_table(baseline_metrics, winner_metrics, winner_params):
    print(f"\n{'=' * 70}")
    print("TUNING RESULT: Baseline vs Winner")
    print("=" * 70)
    print(f"\n  {'Metric':<25} | {'Baseline':>12} | {'Winner':>12} | {'Delta':>12}")
    print(f"  {'-' * 25}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")
    # ... rows for yield, mean_error, max_error, singleton_rate, tier2_stability
```

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `tune_association.py` generates fresh NPZ fixture per combo (full pipeline re-run) | `TuningOrchestrator` loads `tracking_cache.pkl` and re-runs only association → midline → reconstruction per combo | Dramatically faster: skips GPU detection and tracking for each combo |
| `tune_threshold.py` uses legacy `run_evaluation()` on NPZ fixture | `TuningOrchestrator` loads `midline_cache.pkl` and re-runs only reconstruction per combo | Reconstruction-only re-run is seconds vs minutes |
| Grid definitions scattered in script files | Centralized in `evaluation/stages/association.py` and `evaluation/stages/reconstruction.py` | TUNE-06 already complete; single source of truth |

**Deprecated/outdated:**
- `scripts/tune_association.py`: Deleted after TUNE-01 complete
- `scripts/tune_threshold.py`: Deleted after TUNE-02 complete

## Open Questions

1. **ReconstructionStage constructor and `outlier_threshold`**
   - What we know: `build_stages()` passes `outlier_threshold` is NOT in the constructor signature shown (only `calibration_path`, `backend`, `min_cameras`, `max_interp_gap`, `n_control_points`).
   - What's unclear: How `outlier_threshold` is configured in `ReconstructionStage` — is it a constructor param not shown in `build_stages()`? Is it on the backend object?
   - Recommendation: Read `ReconstructionStage.__init__` during Wave 0 planning task to confirm. May need to construct `DltBackend` directly with `outlier_threshold=val` and pass to `ReconstructionStage` rather than using the high-level factory.

2. **`n_points` grid key vs `ReconstructionConfig.n_sample_points`**
   - What we know: `RECONSTRUCTION_DEFAULT_GRID` has `"n_points"`. `ReconstructionConfig` has `n_sample_points`. These names don't match.
   - What's unclear: Whether `n_points` maps to `n_sample_points` (number of midline sample points) or `n_control_points` (7 B-spline knots).
   - Recommendation: Read `reconstruction.py::DEFAULT_GRID` comments or the stage constructor during implementation. Most likely `n_points` → `n_sample_points` (15 default matches grid range 7-21).

3. **Does `MidlineStage.run()` need to re-run for association sweeps?**
   - What we know: `evaluate_association()` takes `list[MidlineSet]` which requires `annotated_detections` + `tracklet_groups`. When `tracklet_groups` changes, the `MidlineSet` mapping changes too.
   - What's unclear: Whether re-using the midline cache's `annotated_detections` with new `tracklet_groups` from the sweep is valid (i.e., are the midlines already computed per-detection, and the MidlineSet just re-maps them?).
   - Recommendation: Yes — `MidlineSet` is constructed by matching tracklet centroids to existing annotated detections (per `_build_midline_sets()` in `runner.py`). The midline cache's `annotated_detections` already has all midlines; re-mapping with new `tracklet_groups` is valid WITHOUT re-running MidlineStage. This is the correct architecture.

4. **How does top-N validation re-run only the swept stage?**
   - What we know: "Re-run target stage only during validation — upstream stage caches are reused."
   - What's unclear: For association, "target stage only" = AssociationStage only (midline + reconstruction still use cached data for metrics). But if we need reconstruction metrics for `tier2_stability`, reconstruction must re-run.
   - Recommendation: For validation tier, re-run AssociationStage with candidate params, rebuild MidlineSet from midline cache + new tracklet_groups, then re-run ReconstructionStage to get updated midlines_3d, then call `evaluate_reconstruction(frame_results, tier2_result=compute_tier2(...))` for full Tier 2. The "only swept stage re-runs" means: NOT re-running detection, tracking, or midline inference — those use cached data.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `hatch run test` |
| Full suite command | `hatch run test-all` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TUNE-01 | Association sweep produces AssociationMetrics from combo params | unit | `pytest tests/unit/evaluation/test_tuning.py::test_association_sweep -x` | ❌ Wave 0 |
| TUNE-02 | Reconstruction sweep produces ReconstructionMetrics from combo params | unit | `pytest tests/unit/evaluation/test_tuning.py::test_reconstruction_sweep -x` | ❌ Wave 0 |
| TUNE-03 | `--n-frames` / `--n-frames-validate` control frame sampling | unit | `pytest tests/unit/evaluation/test_tuning.py::test_frame_count_flags -x` | ❌ Wave 0 |
| TUNE-04 | Top-N candidates are re-validated at higher frame count | unit | `pytest tests/unit/evaluation/test_tuning.py::test_top_n_validation -x` | ❌ Wave 0 |
| TUNE-05 | Output contains comparison table and YAML diff block | unit | `pytest tests/unit/evaluation/test_tuning.py::test_output_format -x` | ❌ Wave 0 |
| CLEAN-01 | `scripts/tune_association.py` does not exist | n/a (file deletion verified by CI) | `pytest tests/unit/test_smoke.py` (import smoke) | ✅ exists |
| CLEAN-02 | `scripts/tune_threshold.py` does not exist | n/a (file deletion verified by CI) | `pytest tests/unit/test_smoke.py` (import smoke) | ✅ exists |

### Sampling Rate
- **Per task commit:** `hatch run test`
- **Per wave merge:** `hatch run test`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/evaluation/test_tuning.py` — covers TUNE-01 through TUNE-05 with synthetic cache fixtures (no GPU required; mock stage `.run()` calls with canned PipelineContext)
- [ ] No framework install needed — pytest already present

## Sources

### Primary (HIGH confidence)
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/stages/association.py` — `ASSOCIATION_DEFAULT_GRID`, `AssociationMetrics`, `evaluate_association()` verified by direct read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/stages/reconstruction.py` — `RECONSTRUCTION_DEFAULT_GRID`, `ReconstructionMetrics`, `evaluate_reconstruction()` verified by direct read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/runner.py` — `EvalRunner`, cache discovery pattern, `_build_midline_sets()` pattern verified by direct read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/context.py` — `load_stage_cache()`, `StaleCacheError`, `PipelineContext` structure verified by direct read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/cli.py` — Click CLI patterns, `eval_cmd` template verified by direct read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/pipeline.py` — `build_stages()`, `stop_after`, stage constructor signatures verified by direct read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/config.py` — `AssociationConfig`, `ReconstructionConfig`, `PipelineConfig.stop_after` field confirmed
- `/home/tlancaster6/Projects/AquaPose/scripts/tune_association.py` — joint grid sweep algorithm, scoring, 2D matrix print, final report pattern verified by direct read
- `/home/tlancaster6/Projects/AquaPose/scripts/tune_threshold.py` — reconstruction sweep pattern verified by direct read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/diagnostic_observer.py` — cache path pattern `diagnostics/<stage>_cache.pkl` confirmed

### Secondary (MEDIUM confidence)
- `.planning/STATE.md` — Phase 46 decisions: shallow copy for sweep isolation, envelope format, cache path structure

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are existing project dependencies; no new libraries needed
- Architecture: HIGH — patterns are directly verified from existing EvalRunner, harness.py, and tune_association.py source
- Pitfalls: HIGH for items 1-5 (verified from source code); MEDIUM for items 6-7 (inferred from naming discrepancies found in source)

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (stable domain; only invalidated by further phase completions that change stage APIs)
