# Phase 74: Round 1 Evaluation & Decision - Research

**Researched:** 2026-03-09
**Domain:** Pipeline evaluation comparison CLI + pipeline re-run with round 1 models
**Confidence:** HIGH

## Summary

Phase 74 requires two main deliverables: (1) a pipeline re-run with the round 1 winner models from Phase 73 to produce a new `eval_results.json`, and (2) a reusable `aquapose eval compare RUN_A RUN_B` CLI command that loads two `eval_results.json` files and produces a side-by-side terminal table with deltas and a machine-readable `eval_comparison.json`.

The codebase already has strong foundations for both: `EvalRunner` + `format_eval_report()` handle single-run evaluation, `training/compare.py::format_comparison_table()` demonstrates the exact click.style best-value highlighting pattern to follow, `resolve_run()` handles run directory resolution by timestamp/path/latest, and the project config already points to the round 1 winner model paths. The Phase 72 baseline run (`run_20260307_140127`) ran with `mode=diagnostic`, `chunk_size=300`, no max_chunks limit (all 9000 frames), and its `eval_results.json` is in place.

**Primary recommendation:** Build `eval compare` as a new Click subcommand in `cli.py` that loads two `eval_results.json` files via `resolve_run()`, computes deltas, and outputs a terminal table following the `training/compare.py` pattern. The pipeline re-run is a straightforward `aquapose run` with the current config (which already has round 1 model paths).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Build `aquapose eval compare RUN_A RUN_B` command that loads both eval_results.json files
- Output: side-by-side terminal table with deltas (absolute + percentage change) and direction indicators (up/down arrows)
- Also writes comparison JSON (`eval_comparison.json`) to the later run's directory
- All metrics shown in comparison, with primary metrics visually highlighted as decision-drivers
- Primary metrics: reprojection error percentiles (p50, p90) and singleton rate
- Decision is purely directional -- any improvement in primary metrics = proceed to round 2
- No hard numeric thresholds or cutoffs
- `74-DECISION.md` written manually in the phase directory after reviewing CLI comparison output
- Mirror Phase 72's exact run parameters (check Phase 72 run and match)
- Only difference: model paths point to round 1 models instead of baseline
- Diagnostic mode (writes chunk caches needed if round 2 proceeds)
- Model paths via SampleStore lookup (consistent with store-based workflow)
- Light pre-flight validation: verify round 1 models are registered in store and .pt files exist before starting the run

### Claude's Discretion
- Exact terminal table layout and column formatting for the comparison CLI
- How primary metrics are visually highlighted (bold, color, marker)
- Pre-flight validation implementation details
- Whether eval compare accepts run directories or run IDs

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ITER-04 | Round 1 pipeline run evaluated against baseline metrics; decision checkpoint on whether to proceed to round 2 | Pipeline re-run with round 1 models, `eval compare` CLI for metric comparison, `74-DECISION.md` for recording the decision |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| click | (existing) | CLI subcommand framework | Already used throughout `cli.py` and `training/cli.py` |
| json | stdlib | Load/write `eval_results.json` and `eval_comparison.json` | Standard library, already used |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| click.style | (existing) | Bold/color highlighting for primary metrics | Terminal table formatting |
| click.unstyle | (existing) | Strip ANSI for width calculation | Column alignment |
| `_NumpySafeEncoder` | (existing) | JSON serialization of numpy scalars | Writing `eval_comparison.json` |

### Alternatives Considered
None -- this phase uses entirely existing infrastructure.

## Architecture Patterns

### Recommended Module Structure
```
src/aquapose/
├── evaluation/
│   ├── compare.py          # NEW: load_eval_results(), compute_deltas(), format_comparison_table()
│   └── output.py           # Existing: _NumpySafeEncoder (reuse for JSON output)
├── cli.py                  # Add `eval-compare` subcommand
└── cli_utils.py            # Existing: resolve_run() for run directory resolution
```

### Pattern 1: Two-Run Comparison Table (follows training/compare.py)
**What:** Load two `eval_results.json` files, compute metric deltas, format as side-by-side terminal table with best-value highlighting.
**When to use:** `aquapose eval compare RUN_A RUN_B`
**Key design points:**
- `resolve_run()` already handles timestamp prefix, path, or "latest" -- use it for both RUN_A and RUN_B arguments
- Flatten the nested `eval_results.json` stages dict into a flat list of `(stage, metric_name, value_a, value_b)` tuples for table rendering
- Primary metrics (singleton_rate, p50/p90 reprojection error) get `click.style(bold=True, fg="green")` or similar highlight
- Delta column shows absolute change and percentage: `+0.05 (+3.2%)` with directional arrow
- For metrics where lower is better (reprojection error, singleton rate), a decrease is "good" (green); for metrics where higher is better (yield, continuity), an increase is "good"

### Pattern 2: eval_comparison.json Schema
**What:** Machine-readable comparison output written to the later run's directory.
**Schema:**
```json
{
  "run_a": {"run_id": "...", "path": "..."},
  "run_b": {"run_id": "...", "path": "..."},
  "metrics": {
    "association": {
      "singleton_rate": {"a": 0.313, "b": 0.25, "delta": -0.063, "pct_change": -20.1, "primary": true},
      ...
    },
    "reconstruction": {
      "p50_reprojection_error": {"a": 3.02, "b": 2.8, "delta": -0.22, "pct_change": -7.3, "primary": true},
      ...
    }
  }
}
```

### Pattern 3: Pre-flight Validation Before Pipeline Re-run
**What:** Before starting the long pipeline run, verify that round 1 models exist and are accessible.
**Implementation:** A simple function that reads the config, checks that `detection.weights_path` and `midline.weights_path` exist on disk, and prints a confirmation or error. This can be a helper called at the start of the re-run script/instructions, not necessarily a CLI command.

### Anti-Patterns to Avoid
- **Re-running EvalRunner instead of loading JSON:** The `eval compare` command should load pre-computed `eval_results.json` files, not re-run `EvalRunner`. Evaluation is expensive (loads chunk caches, does projection math). The JSON files already exist from `aquapose eval`.
- **Hardcoding Phase 72 run parameters:** The CONTEXT.md says to check Phase 72's run and mirror it, not assume specific values. Read the baseline run's `config.yaml` to verify chunk_size, max_chunks, etc.
- **Building a click group instead of a simple command:** The eval comparison is a single command (`aquapose eval-compare` or `aquapose eval compare`). Since `eval` is currently a plain command (not a group), either add a new top-level command `eval-compare`, or refactor `eval` into a group with subcommands `run` and `compare`. The simplest approach is a new top-level command.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Run directory resolution | Custom path parsing | `cli_utils.resolve_run()` | Handles timestamps, paths, "latest" |
| JSON encoding of numpy | Custom float conversion | `_NumpySafeEncoder` | Already handles np.floating, np.integer |
| ANSI-safe column alignment | Manual string padding | `click.unstyle()` for width calc | Pattern from `training/compare.py` |
| Best-value highlighting | Custom color codes | `click.style(bold=True, fg="green")` | Consistent with existing table output |

**Key insight:** Nearly every building block exists. The new code is primarily glue: load two JSON files, compute deltas, format output.

## Common Pitfalls

### Pitfall 1: Metric Directionality
**What goes wrong:** Showing "green" for increases in metrics where lower is better (e.g., reprojection error, singleton rate).
**Why it happens:** Default assumption is "higher = better" but several key metrics are "lower = better".
**How to avoid:** Define a `LOWER_IS_BETTER` set containing `singleton_rate`, `mean_reprojection_error`, `p50_reprojection_error`, `p90_reprojection_error`, `p95_reprojection_error`, `max_reprojection_error`, `total_gaps`, `mean_gap_duration`, `max_gap_duration`, `coast_frequency`, `low_confidence_flag_rate`, etc. Use this to determine arrow direction and color.
**Warning signs:** Green arrows pointing up on error metrics.

### Pitfall 2: Division by Zero in Percentage Change
**What goes wrong:** Division by zero when baseline metric is 0.0.
**Why it happens:** Some metrics (e.g., `total_gaps`) could be 0 in the baseline.
**How to avoid:** Guard with `if abs(baseline_value) < 1e-12: pct = None` and display "N/A" or just the absolute delta.

### Pitfall 3: Nested Dict Metrics
**What goes wrong:** Trying to compute deltas on dict-valued metrics like `per_camera_error`, `camera_distribution`, `per_point_error`, `curvature_stratified`.
**Why it happens:** These are nested structures, not scalar values.
**How to avoid:** Only compute deltas for scalar (int/float) metrics. Skip or specially handle dict-valued metrics. For the comparison table, show only scalar metrics. Dict-valued metrics can be mentioned in the JSON but without delta computation, or flattened (e.g., `per_point_error.0.mean_px`).

### Pitfall 4: Pipeline Run Might Fail or Produce Different Chunk Count
**What goes wrong:** If the round 1 models produce different detection counts, chunk processing might differ subtly.
**Why it happens:** Different OBB model may detect more/fewer fish, changing downstream association/reconstruction.
**How to avoid:** This is expected and part of the evaluation. Just ensure the run completes. Use `mode=diagnostic` to get chunk caches. The eval comparison handles different frame counts gracefully (both runs report `frames_evaluated`).

### Pitfall 5: Stale Config Model Paths
**What goes wrong:** The project config already points to round 1 models (updated during Phase 73), but the Phase 72 baseline run's config.yaml still has the baseline model paths.
**Why it happens:** Each run copies config.yaml at execution time.
**How to avoid:** This is correct behavior -- the current project config has round 1 paths, so a new `aquapose run` will use them. The baseline run's config.yaml correctly records what models were used. No issue here, just be aware.

## Code Examples

### Loading eval_results.json
```python
import json
from pathlib import Path

def load_eval_results(run_dir: Path) -> dict:
    """Load eval_results.json from a run directory."""
    results_path = run_dir / "eval_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No eval_results.json in {run_dir}")
    with results_path.open() as f:
        return json.load(f)
```

### Computing Scalar Deltas
```python
LOWER_IS_BETTER = {
    "singleton_rate", "mean_reprojection_error", "max_reprojection_error",
    "p50_reprojection_error", "p90_reprojection_error", "p95_reprojection_error",
    "total_gaps", "mean_gap_duration", "max_gap_duration",
    "coast_frequency", "low_confidence_flag_rate", "mean_jitter",
}

PRIMARY_METRICS = {
    ("association", "singleton_rate"),
    ("reconstruction", "p50_reprojection_error"),
    ("reconstruction", "p90_reprojection_error"),
}

def compute_delta(stage: str, metric: str, val_a: float, val_b: float) -> dict:
    delta = val_b - val_a
    pct = (delta / val_a * 100) if abs(val_a) > 1e-12 else None
    lower_better = metric in LOWER_IS_BETTER
    improved = (delta < 0) if lower_better else (delta > 0)
    return {"a": val_a, "b": val_b, "delta": delta, "pct_change": pct, "improved": improved}
```

### Click Style Highlighting (from training/compare.py pattern)
```python
import click

# For primary metrics that improved:
cell = click.style(f"{value:.4f}", fg="green", bold=True)

# For primary metrics that regressed:
cell = click.style(f"{value:.4f}", fg="red", bold=True)

# Direction indicator:
arrow = "\u2193" if delta < 0 else "\u2191"  # down/up arrow
```

## Existing Codebase Reference

### Phase 72 Baseline Run Parameters (from config.yaml)
- **Run:** `run_20260307_140127`
- **mode:** diagnostic
- **chunk_size:** 300
- **max_chunks:** None (all chunks processed -- 9000 frames total)
- **n_animals:** 9
- **OBB model:** `run_20260307_094353/best_model.pt` (baseline)
- **Pose model:** `run_20260307_113057/best_model.pt` (baseline)

### Phase 73 Round 1 Winners (from 73-RESULTS.md and current config.yaml)
- **OBB curated:** `run_20260309_120659/best_model.pt`
- **Pose curated+aug:** `run_20260309_152248/best_model.pt`
- Both already set in the project `config.yaml`

### Phase 72 Baseline Primary Metrics
- singleton_rate: 0.3127 (31.3%)
- fish_yield_ratio: 0.8574 (85.7%)
- p50_reprojection_error: 3.020 px
- p90_reprojection_error: 5.203 px
- mean_reprojection_error: 3.520 px
- mean_continuity_ratio: 0.947 (94.7%)
- total_gaps: 6

### eval_results.json Schema (from baseline run)
```
{
  "run_id": "",
  "stages_present": ["association", "detection", "midline", "reconstruction", "tracking"],
  "frames_evaluated": 9000,
  "frames_available": 9000,
  "stages": {
    "detection": { total_detections, mean_confidence, std_confidence, mean_jitter, per_camera_counts },
    "tracking": { track_count, length_*, coast_frequency, detection_coverage },
    "association": { fish_yield_ratio, singleton_rate, camera_distribution, total_fish_observations, frames_evaluated, p50_camera_count, p90_camera_count },
    "midline": { total_midlines, mean_confidence, std_confidence, p10/p50/p90_confidence, completeness, temporal_smoothness },
    "reconstruction": { mean/max/p50/p90/p95_reprojection_error, fish_reconstructed, fish_available, inlier_ratio, low_confidence_flag_rate, tier2_stability, per_camera_error, per_fish_error, z_denoising, per_point_error, curvature_stratified },
    "fragmentation": { total_gaps, mean/max_gap_duration, mean_continuity_ratio, unique_fish_ids, expected_fish, track_births, track_deaths, mean/median_track_lifespan }
  }
}
```

### CLI Command Structure
The current `eval` is a plain `@cli.command("eval")`, not a group. Options for adding compare:
1. **New top-level command** `@cli.command("eval-compare")` -- simplest, no refactoring
2. **Convert eval to a group** with `eval run` (existing) and `eval compare` (new) -- cleaner but breaking change

Recommendation: Use option 1 (`eval-compare`) for simplicity, since the CONTEXT.md says `aquapose eval compare` but the implementation can be `eval-compare` as a single hyphenated command to avoid refactoring `eval` into a group.

Actually, looking at the CONTEXT.md more carefully: "Build `aquapose eval compare RUN_A RUN_B`" -- this implies a subcommand structure. Converting `eval` to a `click.Group` with subcommands `run` (the current eval behavior) and `compare` (new) is the cleanest approach. This is a small refactor: rename `eval_cmd` to be under a group.

## Open Questions

1. **`eval` as group vs new command**
   - What we know: CONTEXT.md says `aquapose eval compare RUN_A RUN_B`. Current `eval` is a plain command.
   - What's unclear: Whether to refactor `eval` into a group (breaking `aquapose eval RUN` syntax) or use `eval-compare`.
   - Recommendation: Claude's discretion per CONTEXT.md. Refactoring to a group is cleaner but changes existing `aquapose eval` to `aquapose eval run`. A compromise: make `eval` a group with `invoke_without_command=True` that defaults to the existing behavior, plus add a `compare` subcommand. This preserves backward compatibility.

2. **Pipeline run duration**
   - What we know: Phase 72 baseline processed 9000 frames (all chunks). With chunk_size=300 that is 30 chunks. This likely took 30-60+ minutes.
   - What's unclear: Exact duration. The run should use `TaskCreate` per project workflow preferences.
   - Recommendation: Document that the pipeline run is long-running and should be executed via `TaskCreate` or interactively in a separate terminal.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `src/aquapose/cli.py`, `src/aquapose/cli_utils.py`, `src/aquapose/evaluation/output.py`, `src/aquapose/evaluation/runner.py`, `src/aquapose/training/compare.py`
- Phase 72 baseline: `~/aquapose/projects/YH/runs/run_20260307_140127/eval_results.json` and `config.yaml`
- Phase 73 results: `.planning/phases/73-round-1-pseudo-labels-retraining/73-RESULTS.md`
- Current project config: `~/aquapose/projects/YH/config.yaml`

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in use
- Architecture: HIGH - follows established codebase patterns exactly
- Pitfalls: HIGH - identified from direct codebase analysis

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable -- no external dependencies changing)
