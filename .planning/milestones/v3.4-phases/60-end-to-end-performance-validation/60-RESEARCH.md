# Phase 60: End-to-End Performance Validation - Research

**Researched:** 2026-03-05
**Domain:** Pipeline benchmarking, eval metric comparison, report generation
**Confidence:** HIGH

## Summary

Phase 60 is a validation-and-reporting phase, not a code-change phase. The goal is to run the post-optimization pipeline on the same workload as the pre-optimization baseline, capture timing and eval metrics, compare them, and produce a markdown report documenting the results.

All required infrastructure already exists: `TimingObserver` captures per-stage wall-clock time automatically in diagnostic mode, `EvalRunner` + `aquapose eval` compute per-stage quality metrics, and the baseline timing is already saved. The work is scripting the run, parsing outputs, computing deltas, and writing the report.

**Primary recommendation:** Write a single Python script (or inline plan tasks) that: (1) runs the post-optimization pipeline, (2) runs eval on both baseline and post-optimization runs, (3) compares timing and eval metrics, (4) generates the markdown report.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use existing TimingObserver for per-stage wall-clock timing
- Baseline is saved from pre-optimization run `run_20260304_180748` (single chunk, 200 frames, 12 cameras, total 914.2s) -- already copied to `baseline-timing.txt` in the phase directory
- Post-optimization run uses identical YH project config, single chunk, same 200-frame window
- Correctness validated by running `aquapose eval` on both runs and comparing metrics
- Run mode: `--mode diagnostic` to generate caches for eval
- Markdown summary saved in `.planning/phases/60-end-to-end-performance-validation/`
- Per-stage speedup ratios: before time, after time, speedup factor
- Focus on four optimized stages: detection, association, midline, reconstruction
- Total wall-clock speedup ratio
- Eval correctness: pass/fail summary only (not full metrics side-by-side)
- No hard minimum speedup target -- document whatever improvement was achieved
- Correctness: threshold tolerance for eval metric differences (small floating-point variance acceptable)
- If correctness regression detected beyond tolerance, milestone is blocked
- Single chunk, same YH config as baseline
- No GPU warm-up run -- baseline didn't have one either
- Single iteration, no averaging

### Claude's Discretion
- Exact floating-point tolerance thresholds for eval metric comparison
- Report formatting details and section organization
- How to present the eval baseline (capture from existing run or reference prior eval output)

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib | 3.x | File I/O, JSON parsing, string formatting | No external deps needed |
| aquapose CLI | local | `aquapose run` and `aquapose eval` | Already built, tested, handles all pipeline orchestration |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | - | Parse eval_results.json | Comparing eval metrics programmatically |
| re (stdlib) | - | Parse timing.txt format | Extracting per-stage seconds from timing report |
| pathlib (stdlib) | - | Path manipulation | All file operations |

**Installation:** No new dependencies required.

## Architecture Patterns

### Recommended Workflow Structure

The phase decomposes into three sequential steps:

1. **Run post-optimization pipeline** -- `aquapose run --config ~/aquapose/projects/YH/config.yaml --mode diagnostic --max-chunks 1`
2. **Run eval on both runs** -- `aquapose eval <baseline_run_dir>` and `aquapose eval <post_run_dir>` (both write `eval_results.json`)
3. **Generate comparison report** -- Parse timing files + eval JSONs, compute deltas, write markdown

### Timing Data Format

The `TimingObserver` writes a fixed-format text file:

```
Timing Report -- run: run_YYYYMMDD_HHMMSS
==================================================
  DetectionStage                   303.45s ( 33.2%)
  TrackingStage                      3.47s (  0.4%)
  AssociationStage                  35.91s (  3.9%)
  MidlineStage                     451.50s ( 49.4%)
  ReconstructionStage              119.89s ( 13.1%)
--------------------------------------------------
  TOTAL                            914.22s
```

Parsing: each stage line matches `^\s+(\w+Stage)\s+([\d.]+)s`. TOTAL line matches `^\s+TOTAL\s+([\d.]+)s`.

### Eval Data Format

`aquapose eval` writes `eval_results.json` to the run directory. Structure:

```json
{
  "run_id": "...",
  "stages_present": ["association", "detection", "midline", "reconstruction", "tracking"],
  "frames_evaluated": 200,
  "frames_available": 200,
  "stages": {
    "detection": { "total_detections": N, "mean_confidence": F, ... },
    "tracking": { "track_count": N, "detection_coverage": F, ... },
    "association": { "fish_yield_ratio": F, "singleton_rate": F, ... },
    "midline": { "total_midlines": N, "mean_confidence": F, ... },
    "reconstruction": { "mean_reprojection_error": F, "fish_reconstructed": N, ... }
  }
}
```

### Key Eval Metrics for Correctness Check

| Stage | Metric | Type | Tolerance Recommendation |
|-------|--------|------|--------------------------|
| detection | total_detections | int | Exact match (deterministic YOLO inference) |
| detection | mean_confidence | float | Exact match (same model, same input) |
| tracking | track_count | int | Exact match (deterministic OC-SORT) |
| tracking | detection_coverage | float | Exact match |
| association | fish_yield_ratio | float | +/- 0.02 (GPU non-determinism in upstream detections) |
| association | singleton_rate | float | +/- 0.05 |
| midline | total_midlines | int | Exact match |
| midline | mean_confidence | float | +/- 0.001 |
| reconstruction | mean_reprojection_error | float | +/- 0.5 px |
| reconstruction | fish_reconstructed | int | +/- 2 (RANSAC non-determinism) |

**Rationale for tolerances:** YOLO inference is deterministic for the same input on the same GPU. Batched inference should produce identical results to sequential inference (same model weights, same preprocessing). However, GPU floating-point non-determinism in cuDNN convolutions can cause tiny differences that propagate through downstream stages. Detection and midline stages should be nearly exact; association and reconstruction have more room for cascading variance.

**Correctness pass/fail rule:** If any metric exceeds its tolerance, flag as FAIL. The report should list which metrics failed and by how much.

### Report Structure (Recommended)

```markdown
# v3.4 Performance Validation Report

**Date:** YYYY-MM-DD
**Baseline run:** run_20260304_180748 (pre-optimization)
**Post-optimization run:** run_YYYYMMDD_HHMMSS

## Timing Comparison

| Stage | Before (s) | After (s) | Speedup |
|-------|-----------|----------|---------|
| Detection | 303.45 | X.XX | Y.Yx |
| Tracking | 3.47 | X.XX | Y.Yx |
| Association | 35.91 | X.XX | Y.Yx |
| Midline | 451.50 | X.XX | Y.Yx |
| Reconstruction | 119.89 | X.XX | Y.Yx |
| **TOTAL** | **914.22** | **X.XX** | **Y.Yx** |

## Correctness Validation

**Result: PASS / FAIL**

[If PASS: "All eval metrics within tolerance."]
[If FAIL: table of failing metrics with expected, actual, tolerance, delta]
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pipeline execution | Custom stage runner | `aquapose run --mode diagnostic --max-chunks 1` | Already handles all stage wiring, observer attachment, output |
| Eval metrics | Custom metric computation | `aquapose eval <run_dir>` | Already computes all per-stage metrics, writes JSON |
| Timing capture | Manual timeit blocks | `TimingObserver` (auto-attached in diagnostic mode) | Already records per-stage wall-clock, writes `timing.txt` |
| JSON comparison | Custom diff logic | Load two `eval_results.json`, compare field-by-field | Simple dict comparison with tolerance |

## Common Pitfalls

### Pitfall 1: Stale Cache Error
**What goes wrong:** If codebase changed since baseline was run, `aquapose eval` on the baseline run raises `StaleCacheError` because the pickle format changed.
**Why it happens:** `load_chunk_cache` validates cache compatibility with current code.
**How to avoid:** Run eval on the baseline *before* making code changes, or ensure the baseline `eval_results.json` already exists from a prior eval run. The baseline run directory at `~/aquapose/projects/YH/runs/run_20260304_180748/` may already have `eval_results.json` if eval was run previously.
**Warning signs:** `StaleCacheError` exception when running `aquapose eval` on old run.

### Pitfall 2: Wrong Run Directory
**What goes wrong:** Comparing timing from a multi-chunk run against the single-chunk baseline.
**Why it happens:** Forgetting `--max-chunks 1`.
**How to avoid:** Always use `--max-chunks 1` for the post-optimization run.

### Pitfall 3: Timing File Location
**What goes wrong:** Looking for `timing.txt` in the wrong place.
**Why it happens:** TimingObserver writes to the run output directory, which is under `~/aquapose/projects/YH/runs/run_YYYYMMDD_HHMMSS/`.
**How to avoid:** The timing file path is `<run_dir>/timing.txt`.

### Pitfall 4: Baseline Eval May Not Exist
**What goes wrong:** Trying to load `eval_results.json` from the baseline run when it was never generated.
**Why it happens:** `aquapose eval` must be explicitly run on each run directory.
**How to avoid:** Check if `eval_results.json` exists in the baseline run dir. If not, run `aquapose eval` on it first. But beware of Pitfall 1 (stale cache) -- if the cache is stale, the eval cannot be regenerated. In that case, run eval on a different pre-optimization run, or skip the eval comparison and note it in the report.

### Pitfall 5: Diagnostic Mode Timing Overhead
**What goes wrong:** Diagnostic mode adds cache-writing overhead that inflates timing numbers.
**Why it happens:** Diagnostic mode serializes pipeline context to disk after each chunk.
**How to avoid:** Both baseline and post-optimization use diagnostic mode, so the overhead is consistent (apples-to-apples comparison, as specified in CONTEXT.md).

## Code Examples

### Parsing timing.txt
```python
import re
from pathlib import Path

def parse_timing(path: Path) -> dict[str, float]:
    """Parse a TimingObserver timing.txt into {stage_name: seconds}."""
    result = {}
    text = path.read_text()
    for match in re.finditer(r'^\s+(\S+)\s+([\d.]+)s', text, re.MULTILINE):
        result[match.group(1)] = float(match.group(2))
    return result
```

### Comparing eval metrics with tolerance
```python
import json
from pathlib import Path

def compare_eval(baseline_path: Path, post_path: Path) -> list[dict]:
    """Compare two eval_results.json files, return list of failures."""
    TOLERANCES = {
        "detection": {"total_detections": 0, "mean_confidence": 0.0},
        "tracking": {"track_count": 0, "detection_coverage": 0.0},
        "association": {"fish_yield_ratio": 0.02, "singleton_rate": 0.05},
        "midline": {"total_midlines": 0, "mean_confidence": 0.001},
        "reconstruction": {"mean_reprojection_error": 0.5, "fish_reconstructed": 2},
    }

    baseline = json.loads(baseline_path.read_text())
    post = json.loads(post_path.read_text())

    failures = []
    for stage, metrics in TOLERANCES.items():
        b_stage = baseline["stages"].get(stage, {})
        p_stage = post["stages"].get(stage, {})
        for metric, tol in metrics.items():
            b_val = b_stage.get(metric)
            p_val = p_stage.get(metric)
            if b_val is None or p_val is None:
                continue
            delta = abs(p_val - b_val)
            if delta > tol:
                failures.append({
                    "stage": stage, "metric": metric,
                    "baseline": b_val, "post": p_val,
                    "tolerance": tol, "delta": delta,
                })
    return failures
```

### Running the pipeline
```bash
# Post-optimization run (single chunk, diagnostic mode)
aquapose run --config ~/aquapose/projects/YH/config.yaml --mode diagnostic --max-chunks 1

# Eval on baseline (if eval_results.json doesn't already exist)
aquapose eval ~/aquapose/projects/YH/runs/run_20260304_180748/

# Eval on post-optimization run
aquapose eval ~/aquapose/projects/YH/runs/run_YYYYMMDD_HHMMSS/
```

## State of the Art

No technology changes relevant -- this phase uses existing project infrastructure exclusively.

## Open Questions

1. **Baseline eval_results.json availability**
   - What we know: The baseline run exists at `~/aquapose/projects/YH/runs/run_20260304_180748/`
   - What's unclear: Whether `eval_results.json` already exists there, or whether the cache is still compatible with current code
   - Recommendation: Check at plan execution time. If stale, try other pre-optimization runs in the `runs/` directory. If all stale, skip eval comparison and note it in the report as a known limitation.

2. **Post-optimization run output directory**
   - What we know: `aquapose run` creates a timestamped directory under `~/aquapose/projects/YH/runs/`
   - What's unclear: The exact path (generated at runtime)
   - Recommendation: Capture the run directory path from CLI output or glob for the most recent run directory after execution.

## Sources

### Primary (HIGH confidence)
- Project source code: `src/aquapose/engine/timing.py` -- TimingObserver implementation
- Project source code: `src/aquapose/evaluation/runner.py` -- EvalRunner implementation
- Project source code: `src/aquapose/evaluation/output.py` -- Report formatting, JSON output
- Project source code: `src/aquapose/cli.py` -- CLI command signatures
- Phase context: `.planning/phases/60-end-to-end-performance-validation/60-CONTEXT.md`
- Baseline timing: `.planning/phases/60-end-to-end-performance-validation/baseline-timing.txt`

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all tools are existing project infrastructure, fully inspected
- Architecture: HIGH - straightforward run-compare-report workflow
- Pitfalls: HIGH - based on direct code inspection of error handling and file formats

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (stable -- no external dependencies)
