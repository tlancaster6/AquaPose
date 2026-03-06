# Phase 48: EvalRunner and `aquapose eval` CLI - Research

**Researched:** 2026-03-03
**Domain:** CLI orchestration, multi-stage report formatting, pickle cache discovery
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Report structure**
- Equal-depth sections for all 5 stages (detection, tracking, association, midline, reconstruction)
- Header summary block at the top with run metadata (run ID, date, stages present, frames) and one-line-per-stage key metric values — no pass/fail judgment, user interprets
- Flat metric lists per stage — no tiered sub-sections. Only reconstruction keeps its Tier 1 (reprojection) / Tier 2 (leave-one-out) structure since those are meaningfully different metric categories

**CLI interface**
- Primary input: `aquapose eval <run-dir>` — positional argument, run directory path
- Always evaluates all stages present in the run directory — no `--stages` filter flag
- Keep `--n-frames` flag from measure_baseline.py for quick spot-checks; defaults to all frames if omitted
- `--report json` flag for machine-readable JSON output (matching success criteria EVAL-07)
- Output files (eval_results.json, report text) written inside the run directory alongside stage caches

**Output format**
- Human-readable: same ASCII table style as existing `format_summary_table` — aligned columns, dashes, no color/rich dependency
- Missing stages (from `--stop-after` runs): skip silently, only show sections for stages with cache files
- Drop outlier flagging (`*` markers from format_baseline_report) — eval report presents raw metrics
- JSON schema: Claude's discretion on nesting structure

**EvalRunner orchestration**
- `EvalRunner` class in `src/aquapose/evaluation/` with a `run()` method
- Discovers and loads per-stage pickle caches from the run directory
- Prefer cache-files-only operation; fall back to config.yaml from the run directory if evaluators need parameters like n_animals (config.yaml is always present and consistently located in run dirs)
- Class name: `EvalRunner`

### Claude's Discretion
- JSON schema nesting structure (mirror report vs. flat machine-friendly)
- Internal EvalRunner discovery logic for finding stage caches
- How to pass evaluator parameters when cache-only isn't sufficient
- Exact metric formatting (decimal places, units, column widths)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-06 | `aquapose eval <run-dir>` CLI produces multi-stage human-readable report to stdout | `EvalRunner.run()` orchestrates all stage evaluators; `format_eval_report()` in output.py produces ASCII text; `eval` command added to Click group in cli.py |
| EVAL-07 | `aquapose eval <run-dir> --report json` produces machine-readable JSON output | `--report json` Click option on `eval` command; JSON written via extended `_NumpySafeEncoder` pattern; nested dict with `run_metadata`, `stages` keys |
| CLEAN-03 | `scripts/measure_baseline.py` retired after `aquapose eval` achieves feature parity | Delete `scripts/measure_baseline.py` as a final task after `aquapose eval` passes all success criteria |
</phase_requirements>

## Summary

Phase 48 wires together Phase 47's five stage evaluators into a single `EvalRunner` class and exposes it as `aquapose eval <run-dir>` in the existing Click CLI. The implementation has three tightly constrained sub-problems: (1) cache discovery — finding which `<run-dir>/diagnostics/*_cache.pkl` files exist, loading them with `load_stage_cache()`, and extracting the typed data each evaluator needs; (2) report formatting — extending `evaluation/output.py` with multi-stage ASCII and JSON formatters that follow the existing `format_summary_table` style; and (3) config.yaml fallback — reading `n_animals` from the run directory's `config.yaml` when needed by `evaluate_association()`.

The existing codebase provides all the primitives needed. `load_stage_cache()` in `core/context.py` handles pickle deserialization with `StaleCacheError` propagation. The five evaluators from Phase 47 each accept typed arguments extracted from `PipelineContext` fields. `cli.py` already has a Click group pattern with `@cli.command()` and `click.Path(exists=True)`. `evaluation/output.py` has the ASCII formatting style to extend and `_NumpySafeEncoder` for JSON serialization.

The main discretionary work is: (1) deciding the JSON schema structure (recommended: `{run_metadata: {...}, stages: {detection: {...}, tracking: {...}, ...}}`), (2) implementing `EvalRunner` discovery and n_animals fallback logic, and (3) extending `output.py` with multi-stage formatters. No new dependencies are required.

**Primary recommendation:** Follow `PosePipeline` as the structural model for `EvalRunner` (class with `run()` method returning a result dataclass). Use `load_stage_cache()` directly for cache loading. Read `n_animals` from `config.yaml` via `load_config()` only when association cache is present.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Click | project dep | CLI `eval` command, `--report`, `--n-frames` options | All existing CLI commands use Click; pattern is locked |
| Python dataclasses (frozen=True) | stdlib | `EvalRunnerResult` multi-stage result type | Established project pattern for result types |
| json | stdlib | JSON output (`--report json`) | `_NumpySafeEncoder` already handles numpy scalar conversion |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pathlib.Path` | stdlib | Cache discovery (`<run-dir>/diagnostics/*.pkl` glob) | All path handling in this codebase uses pathlib |
| `load_config()` | internal | Reading `n_animals` from run-dir `config.yaml` | Only when association cache present and n_animals needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Flat `stages` dict in JSON | Nested `stage_name.tier1/tier2` mirror of ASCII | Flat is machine-friendly; mirror is more symmetric. Recommend flat with explicit `reconstruction.tier1` and `reconstruction.tier2` sub-keys since that structure maps cleanly to ReconstructionMetrics fields |
| `load_config()` for n_animals | Parsing YAML directly | `load_config()` is the canonical reader; direct YAML parsing would duplicate logic and miss defaults resolution |

**Installation:** No new dependencies required.

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/evaluation/
├── __init__.py              # update to export EvalRunner + EvalRunnerResult
├── metrics.py               # UNCHANGED
├── harness.py               # UNCHANGED (Phase 50 handles removal)
├── output.py                # EXTENDED: add format_eval_report(), format_eval_json()
├── runner.py                # NEW: EvalRunner class, EvalRunnerResult dataclass
└── stages/
    └── ...                  # Phase 47 outputs (detection, tracking, etc.)
```

### Pattern 1: EvalRunner Mirrors PosePipeline
**What:** `EvalRunner` is a class initialized with a `run_dir: Path`, with a `run(n_frames: int | None = None) -> EvalRunnerResult` method.
**When to use:** This is the sole orchestration pattern.
**Example:**
```python
# Following PosePipeline pattern from src/aquapose/engine/pipeline.py
@dataclass(frozen=True)
class EvalRunnerResult:
    """Multi-stage evaluation results.

    Attributes:
        run_id: Run identifier from the cache envelope metadata.
        stages_present: Set of stage names for which caches were found.
        detection: DetectionMetrics or None if cache not present.
        tracking: TrackingMetrics or None if cache not present.
        association: AssociationMetrics or None if cache not present.
        midline: MidlineMetrics or None if cache not present.
        reconstruction: ReconstructionMetrics or None if cache not present.
        frames_evaluated: Number of frames evaluated (after n_frames sampling).
        frames_available: Total frames available in caches.
    """
    run_id: str
    stages_present: frozenset[str]
    detection: DetectionMetrics | None
    tracking: TrackingMetrics | None
    association: AssociationMetrics | None
    midline: MidlineMetrics | None
    reconstruction: ReconstructionMetrics | None
    frames_evaluated: int
    frames_available: int


class EvalRunner:
    """Orchestrates per-stage evaluation from a diagnostic run directory."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)

    def run(self, n_frames: int | None = None) -> EvalRunnerResult:
        """Discover caches, run evaluators, return multi-stage results."""
        ...
```

### Pattern 2: Cache Discovery with Silent Skip
**What:** Probe for each of the 5 expected cache filenames. Load those present with `load_stage_cache()`. Skip missing files silently.
**When to use:** Core of EvalRunner's discovery logic.
**Example:**
```python
# Cache files written by DiagnosticObserver._write_stage_cache():
# Stage name "DetectionStage" -> stage_key "detection" -> "detection_cache.pkl"
_STAGE_CACHE_NAMES: dict[str, str] = {
    "detection": "DetectionStage",
    "tracking": "TrackingStage",
    "association": "AssociationStage",
    "midline": "MidlineStage",
    "reconstruction": "ReconstructionStage",
}

def _discover_caches(run_dir: Path) -> dict[str, PipelineContext]:
    """Load all available stage caches from <run-dir>/diagnostics/."""
    diagnostics_dir = run_dir / "diagnostics"
    loaded: dict[str, PipelineContext] = {}
    for stage_key in _STAGE_CACHE_NAMES:
        cache_path = diagnostics_dir / f"{stage_key}_cache.pkl"
        if cache_path.exists():
            loaded[stage_key] = load_stage_cache(cache_path)
    return loaded
```

### Pattern 3: n_animals Fallback via load_config()
**What:** When the association cache is present, `evaluate_association()` requires `n_animals`. Read it from `<run-dir>/config.yaml` using `load_config()`.
**When to use:** Only when association cache is present. config.yaml is always written by PosePipeline before any stage runs.
**Example:**
```python
from aquapose.engine.config import load_config  # inline import to avoid top-level coupling

def _read_n_animals(run_dir: Path) -> int:
    """Read n_animals from config.yaml in the run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in run directory: {run_dir}")
    config = load_config(config_path)
    return config.n_animals
```

Note: follow the Phase 46 pattern of inline import (`from aquapose.engine.config import load_config` inside the method) to avoid top-level coupling between `evaluation/` and `engine/`.

### Pattern 4: Multi-Stage ASCII Report
**What:** Extend `output.py` with `format_eval_report(result: EvalRunnerResult) -> str` that produces an ASCII report. Header block, then one section per present stage.
**When to use:** Called by CLI for human-readable output.
**Style guidelines from existing `format_summary_table`:**
- Header: `"Evaluation Report"` + `"="*N` underline
- Metadata block: run_id, stages present, frames evaluated
- Summary line per stage (one key metric inline)
- Stage sections: `"Stage: Detection"` + `"-"*50` + table rows
- Column widths: `f"{'Metric':<30} {'Value':>12}"` pattern

### Pattern 5: JSON Output Schema
**What:** `format_eval_json(result: EvalRunnerResult) -> str` produces JSON string. Schema mirrors the report sections but machine-friendly.
**Recommended schema:**
```json
{
  "run_id": "run_20260303_003159",
  "stages_present": ["detection", "tracking", "association", "midline", "reconstruction"],
  "frames_evaluated": 100,
  "frames_available": 500,
  "stages": {
    "detection": { ...DetectionMetrics.to_dict()... },
    "tracking": { ...TrackingMetrics.to_dict()... },
    "association": { ...AssociationMetrics.to_dict()... },
    "midline": { ...MidlineMetrics.to_dict()... },
    "reconstruction": {
      "tier1": { ...tier1 fields... },
      "tier2": { ...tier2 fields... },
      ...other ReconstructionMetrics fields...
    }
  }
}
```
Uses `_NumpySafeEncoder` from `output.py` for numpy scalar conversion.

### Pattern 6: Click `eval` Command
**What:** New `@cli.command()` in `cli.py` following existing `run` and `init-config` commands.
**Example:**
```python
@cli.command("eval")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--report",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format (default: text).",
)
@click.option(
    "--n-frames",
    "n_frames",
    type=int,
    default=None,
    help="Number of frames to evaluate (default: all frames).",
)
def eval_cmd(run_dir: str, report: str, n_frames: int | None) -> None:
    """Evaluate a diagnostic run directory and print a quality report."""
    from aquapose.evaluation.runner import EvalRunner
    from aquapose.evaluation.output import format_eval_report, format_eval_json

    runner = EvalRunner(Path(run_dir))
    result = runner.run(n_frames=n_frames)

    if report == "json":
        click.echo(format_eval_json(result))
    else:
        click.echo(format_eval_report(result))
```

Note: Click does not allow a command named `eval` (Python built-in). Use `@cli.command("eval")` with the function named `eval_cmd`. Verify this works — if Click rejects "eval" as a name, use `"evaluate"` as the command name but the success criteria specifies `aquapose eval`, so "eval" must be verified.

### Anti-Patterns to Avoid
- **Top-level `from aquapose.engine import ...` in runner.py:** Use inline imports (same pattern as CLI `--resume-from` logic in `cli.py` line 128). This keeps `evaluation/` decoupled from `engine/` at module import time.
- **Re-implementing frame sampling in EvalRunner:** `select_frames()` from `evaluation/metrics.py` handles deterministic sampling. EvalRunner should call it when `n_frames` is provided, using the detection or reconstruction cache frame count.
- **Writing output files eagerly:** The report and JSON are written to the run directory (decision: `eval_results.json` inside run dir). But stdout always gets the output too. Don't silently suppress stdout.
- **Importing `format_summary_table` and `format_baseline_report` for the new report:** These are reconstruction-only formatters. The new `format_eval_report()` is a separate function that covers all five stages.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pickle deserialization with error handling | Custom unpickle + error wrapper | `load_stage_cache()` from `core/context.py` | Already handles `StaleCacheError`, `FileNotFoundError`, shape validation |
| Numpy scalar → JSON conversion | New encoder class | `_NumpySafeEncoder` from `evaluation/output.py` (import it) | Already handles `np.floating` and `np.integer` |
| Frame sampling | `np.linspace`-based sampler | `select_frames()` from `evaluation/metrics.py` | Deterministic, handles edge cases (fewer frames than requested) |
| Config reading | Direct YAML parsing | `load_config()` from `engine/config.py` | Handles defaults, path resolution, n_animals validation |
| Click command structure | New Click group | Append `@cli.command("eval")` to existing group in `cli.py` | Consistent with `run` and `init-config` patterns |

**Key insight:** EvalRunner's job is orchestration only — it glues existing primitives (load_stage_cache, stage evaluators, select_frames) together. No new algorithms are needed.

## Common Pitfalls

### Pitfall 1: PipelineContext Field Extraction for Each Evaluator
**What goes wrong:** Each evaluator accepts different typed inputs extracted from PipelineContext. The caller (EvalRunner) must correctly unpack context fields for each evaluator.
**Why it happens:** PipelineContext fields use generic types (`list`, `dict`) without inner type parameters due to engine import boundary constraints. The mapping is:
- `evaluate_detection()` ← `ctx.detections` (`list[dict[str, list[Detection]]]`)
- `evaluate_tracking()` ← flatten `ctx.tracks_2d` (`dict[str, list[Tracklet2D]]` → `list[Tracklet2D]`)
- `evaluate_association()` ← **need to know Phase 47's exact input type** (likely `list[MidlineSet]` or equivalent from `ctx.tracklet_groups` and `ctx.annotated_detections`)
- `evaluate_midline()` ← **need to know Phase 47's exact input type** (likely `ctx.annotated_detections`)
- `evaluate_reconstruction()` ← **need to know Phase 47's exact input type** (likely `ctx.midlines_3d`)
**How to avoid:** Read Phase 47 evaluator signatures before implementing EvalRunner. The Phase 47 PLAN files specify exact evaluator signatures. If Phase 47 is not yet complete when implementing Phase 48, the planner must specify the interface contract explicitly.
**Warning signs:** If you write `ctx.detections` and get a typing error about inner types, add a runtime type check or cast via `typing.cast`.

### Pitfall 2: `eval` as Click Command Name
**What goes wrong:** Python's built-in `eval` function name may conflict with Click command naming or function definition.
**Why it happens:** `def eval(...)` is valid Python but shadows the built-in. Click registers commands by string name independently of the function name.
**How to avoid:** Name the function `eval_cmd` but register as `@cli.command("eval")`. Test that `aquapose eval --help` works after registration.

### Pitfall 3: n_frames Sampling Requires a Frame Count Reference
**What goes wrong:** `select_frames()` takes `frame_indices: tuple[int, ...]` from a fixture. EvalRunner works from PipelineContext, not a fixture. The frame indices must come from `ctx.frame_count` or be reconstructed from detection data.
**Why it happens:** The existing `select_frames()` was designed for NPZ fixtures with explicit `frame_indices` tuples.
**How to avoid:** Construct frame indices as `tuple(range(ctx.frame_count))` when `ctx.frame_count` is set. This is valid because PipelineContext represents a contiguous frame sequence starting at 0. Apply `n_frames` sampling at the context level before passing data to evaluators — or pass frame indices to evaluators that need them.

### Pitfall 4: Association Evaluator Requires Data from Multiple Caches
**What goes wrong:** The association stage cache stores `ctx.tracklet_groups` (the output of Stage 3). But `evaluate_association()` per Phase 47's design accepts `list[MidlineSet]` — which is built from both `tracklet_groups` and `annotated_detections`. The exact input may require the midline cache too, or the Phase 47 evaluator may accept `list[TrackletGroup]` directly.
**Why it happens:** There's ambiguity in what the association evaluator accepts — Phase 47's CONTEXT.md says "fish yield ratio, singleton rate, camera coverage, cluster quality" which can all be derived from `TrackletGroup` data alone without midlines.
**How to avoid:** Check Phase 47's actual evaluator signatures once implemented. Most association metrics (singleton rate, camera coverage, yield ratio) derive from `TrackletGroup.tracklets` — which is in the association cache. EvalRunner should use the association cache's `ctx.tracklet_groups` and call `evaluate_association(tracklet_groups, n_animals)`.

### Pitfall 5: StaleCacheError Propagation to Click
**What goes wrong:** If a cache file is present but stale (class evolution), `load_stage_cache()` raises `StaleCacheError`. Unhandled, this produces a Python traceback instead of a clean error message.
**Why it happens:** `StaleCacheError` is not a `click.ClickException`, so Click won't format it nicely.
**How to avoid:** Catch `StaleCacheError` in the Click command handler and re-raise as `click.ClickException(str(exc))`. Follow the existing pattern in `cli.py` line 133.

### Pitfall 6: Output Files and stdout Both Required
**What goes wrong:** The decision says output files are written inside the run directory. But stdout must also receive the report (for EVAL-06/07 success criteria).
**Why it happens:** The success criteria say "prints to stdout" — file writing is additional.
**How to avoid:** `click.echo()` the report to stdout AND write `eval_results.json` to `<run-dir>/`. The text report can optionally also be written to `<run-dir>/eval_report.txt` but stdout is mandatory.

## Code Examples

Verified patterns from existing sources:

### Cache File Location (from DiagnosticObserver._write_stage_cache)
```python
# Source: src/aquapose/engine/diagnostic_observer.py lines 189-194
# Stage names map as: "DetectionStage".removesuffix("Stage").lower() -> "detection"
# Written to: <output_dir>/diagnostics/detection_cache.pkl

_CACHE_STAGE_KEYS: tuple[str, ...] = (
    "detection", "tracking", "association", "midline", "reconstruction"
)

diagnostics_dir = run_dir / "diagnostics"
for stage_key in _CACHE_STAGE_KEYS:
    cache_path = diagnostics_dir / f"{stage_key}_cache.pkl"
    if cache_path.exists():
        ctx = load_stage_cache(cache_path)
        # ctx is a full PipelineContext with all fields up to that stage populated
```

### PipelineContext Fields for Each Stage
```python
# Source: src/aquapose/core/context.py PipelineContext docstring
# Stage 1 Detection output:  ctx.detections  -- list[dict[str, list[Detection]]]
# Stage 2 Tracking output:   ctx.tracks_2d   -- dict[str, list[Tracklet2D]]
# Stage 3 Association output: ctx.tracklet_groups -- list[TrackletGroup]
# Stage 4 Midline output:    ctx.annotated_detections -- list[dict[str, list[AnnotatedDetection]]]
# Stage 5 Reconstruction output: ctx.midlines_3d -- list[dict[int, Midline3D]]
# Always set after detection: ctx.frame_count, ctx.camera_ids
```

### Extracting Tracklets for evaluate_tracking()
```python
# tracks_2d is dict[str, list[Tracklet2D]] — cam_id -> tracklets
# evaluate_tracking() accepts list[Tracklet2D] — flatten all cameras
tracking_ctx = loaded_caches.get("tracking")
if tracking_ctx and tracking_ctx.tracks_2d:
    all_tracklets = [
        t
        for tracklets in tracking_ctx.tracks_2d.values()
        for t in tracklets
    ]
    tracking_metrics = evaluate_tracking(all_tracklets)
```

### Reading n_animals from config.yaml
```python
# Source: src/aquapose/engine/pipeline.py line 179 shows config.yaml location
# config.yaml is always at <run_dir>/config.yaml (not inside diagnostics/)

def _read_n_animals(run_dir: Path) -> int:
    # Inline import to avoid top-level engine/ coupling (per Phase 46 pattern)
    from aquapose.engine.config import load_config
    config = load_config(run_dir / "config.yaml")
    return config.n_animals
```

### Existing Click Command Pattern
```python
# Source: src/aquapose/cli.py @cli.command() pattern
@cli.command("eval")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--report", type=click.Choice(["text", "json"]), default="text")
@click.option("--n-frames", "n_frames", type=int, default=None)
def eval_cmd(run_dir: str, report: str, n_frames: int | None) -> None:
    """Evaluate a diagnostic run directory and print a quality report."""
    ...
```

### ASCII Report Style (from format_summary_table)
```python
# Source: src/aquapose/evaluation/output.py format_summary_table
lines: list[str] = []
lines.append("Evaluation Report")
lines.append("=================")
lines.append(f"Run: {result.run_id}  |  Frames: {result.frames_evaluated}/{result.frames_available}")
lines.append(f"Stages: {', '.join(sorted(result.stages_present))}")
lines.append("")
# Per-stage sections:
lines.append("Detection")
lines.append("-" * 50)
lines.append(f"{'Metric':<30} {'Value':>12}")
lines.append(f"{'------':<30} {'-----':>12}")
# ... rows from detection_metrics.to_dict()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `scripts/measure_baseline.py` — argparse, NPZ fixture input, reconstruction-only | `aquapose eval <run-dir>` — Click, per-stage pickle caches, 5-stage coverage | Phase 48 | Replaces legacy script with integrated CLI; all 5 stages evaluated; CLEAN-03 |
| `format_baseline_report()` with `*` outlier markers | `format_eval_report()` — raw metrics, no outlier flagging | Phase 48 | Simpler output; user interprets thresholds without automatic flagging |
| `EvalResults` dataclass (reconstruction-only) | `EvalRunnerResult` dataclass (5-stage) | Phase 48 | Multi-stage result type |
| NPZ fixtures (v2.0 midline_fixtures.npz) | Per-stage pickle caches (`diagnostics/<stage>_cache.pkl`) | Phase 46 | Evaluation no longer depends on NPZ format; caches are richer |

**Deprecated/outdated:**
- `scripts/measure_baseline.py`: Deleted in this phase (CLEAN-03). Its `format_baseline_report()` outlier logic stays in `output.py` but is not used by the new report.
- `EvalResults` from `harness.py`: Not deleted (Phase 50 handles `harness.py`), but `EvalRunnerResult` supersedes it for new code.

## Open Questions

1. **Phase 47 evaluator signatures — exact inputs for association, midline, reconstruction**
   - What we know: Phase 47 CONTEXT.md specifies inputs conceptually. The PLAN files specify function signatures. Association takes `list[MidlineSet]` or `list[TrackletGroup]`.
   - What's unclear: Phase 48 planner must confirm exact Phase 47 evaluator signatures before specifying EvalRunner's unpacking logic. If Phase 47 hasn't run yet, the planner should specify the interface and EvalRunner implementation assumes that interface.
   - Recommendation: Planner should read Phase 47 PLAN files and specify the exact input unpacking in EvalRunner tasks. The most likely mapping: `evaluate_association(tracklet_groups, n_animals)` takes `list[TrackletGroup]` (from association cache); `evaluate_midline(annotated_per_frame)` takes the annotated_detections field; `evaluate_reconstruction(midlines_3d)` takes the midlines_3d field.

2. **Frame count for n_frames sampling — which cache to use?**
   - What we know: `frame_count` and `frame_indices` are set on PipelineContext by the detection stage. All later stage caches also have `frame_count` populated.
   - What's unclear: Should EvalRunner use the most-complete cache's frame_count, or the earliest present cache?
   - Recommendation: Use whichever cache is earliest in the pipeline (detection first, then tracking, etc.) for `frame_count`. Construct frame indices as `tuple(range(frame_count))` and apply `select_frames(frame_indices, n_frames)` when `n_frames` is not None.

3. **Whether `eval` is a valid Click command name**
   - What we know: `eval` is a Python built-in but valid as a string identifier and Click command name. The function should be named `eval_cmd` to avoid shadowing.
   - What's unclear: Whether Click's entry-point machinery has issues with `eval` as a command name.
   - Recommendation: Use `@cli.command("eval")` with function `eval_cmd`. This is a low-risk use of Click's command naming. Verify with `aquapose eval --help` in the test suite.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via hatch) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `hatch run test tests/unit/evaluation/` |
| Full suite command | `hatch run test` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-06 | `aquapose eval <run-dir>` prints multi-stage human-readable report to stdout | unit | `hatch run test tests/unit/evaluation/test_runner.py -x` | No — Wave 0 |
| EVAL-07 | `--report json` produces machine-readable JSON output | unit | `hatch run test tests/unit/evaluation/test_runner.py -x` | No — Wave 0 |
| CLEAN-03 | `scripts/measure_baseline.py` deleted | N/A (file deletion) | `python -c "import aquapose; assert not Path('scripts/measure_baseline.py').exists()"` or just verify in git status | No |

### Sampling Rate
- **Per task commit:** `hatch run test tests/unit/evaluation/ -x`
- **Per wave merge:** `hatch run test`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/evaluation/test_runner.py` — covers EVAL-06, EVAL-07 with synthetic cache fixtures (no real pipeline run)
- [ ] `tests/unit/evaluation/test_eval_output.py` — covers `format_eval_report()` and `format_eval_json()` output formatting

*(Existing test infrastructure: `tests/unit/evaluation/__init__.py`, `test_harness.py`, `test_metrics.py`, `test_output.py` exist and establish patterns for synthetic fixture construction.)*

## Sources

### Primary (HIGH confidence)
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/diagnostic_observer.py` — cache file naming convention (`<stage_key>_cache.pkl` in `diagnostics/`), cache envelope format
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/context.py` — `load_stage_cache()`, `StaleCacheError`, `PipelineContext` fields and their types
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/cli.py` — Click group pattern, `@cli.command()` usage, inline import pattern for `StaleCacheError`
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/output.py` — `format_summary_table()`, `_NumpySafeEncoder`, ASCII formatting style
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/harness.py` — `run_evaluation()` as structural model; `EvalResults` as prior-art result dataclass
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/pipeline.py` — `config.yaml` written at `<run_dir>/config.yaml` before any stage runs
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/config.py` — `load_config()`, `PipelineConfig.n_animals` field
- `/home/tlancaster6/Projects/AquaPose/scripts/measure_baseline.py` — feature reference for CLEAN-03 parity check
- `/home/tlancaster6/Projects/AquaPose/.planning/phases/47-evaluation-primitives/47-CONTEXT.md` — evaluator interface contracts (input types, return types)
- `/home/tlancaster6/Projects/AquaPose/.planning/phases/47-evaluation-primitives/47-01-PLAN.md` — exact evaluator function signatures for detection and tracking

### Secondary (MEDIUM confidence)
- `/home/tlancaster6/Projects/AquaPose/.planning/phases/47-evaluation-primitives/47-RESEARCH.md` — architecture decisions locked for Phase 47 that Phase 48 depends on
- `/home/tlancaster6/Projects/AquaPose/tests/unit/evaluation/test_harness.py` — synthetic fixture pattern for EvalRunner tests

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are existing project deps, no new additions
- Architecture: HIGH — directly based on existing CLI + evaluation patterns; EvalRunner structure mirrors PosePipeline
- Pitfalls: HIGH — all identified from direct code inspection of context.py, cli.py, harness.py
- Open questions: MEDIUM — three open questions, all with clear recommendations; Phase 47 evaluator signatures are the only real unknown

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (stable — no external dependencies changing; validity depends on Phase 47 completing as designed)
