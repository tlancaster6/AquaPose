# Phase 48: EvalRunner and `aquapose eval` CLI - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can evaluate any diagnostic run directory and receive a multi-stage quality report in human-readable or JSON format, replacing the functionality of `scripts/measure_baseline.py`. The EvalRunner orchestrates Phase 47's per-stage evaluators into a unified report. Input is a run directory containing Phase 46's per-stage pickle caches.

</domain>

<decisions>
## Implementation Decisions

### Report structure
- Equal-depth sections for all 5 stages (detection, tracking, association, midline, reconstruction)
- Header summary block at the top with run metadata (run ID, date, stages present, frames) and one-line-per-stage key metric values — no pass/fail judgment, user interprets
- Flat metric lists per stage — no tiered sub-sections. Only reconstruction keeps its Tier 1 (reprojection) / Tier 2 (leave-one-out) structure since those are meaningfully different metric categories

### CLI interface
- Primary input: `aquapose eval <run-dir>` — positional argument, run directory path
- Always evaluates all stages present in the run directory — no `--stages` filter flag
- Keep `--n-frames` flag from measure_baseline.py for quick spot-checks; defaults to all frames if omitted
- `--report json` flag for machine-readable JSON output (matching success criteria EVAL-07)
- Output files (eval_results.json, report text) written inside the run directory alongside stage caches

### Output format
- Human-readable: same ASCII table style as existing format_summary_table — aligned columns, dashes, no color/rich dependency
- Missing stages (from `--stop-after` runs): skip silently, only show sections for stages with cache files
- Drop outlier flagging (`*` markers from format_baseline_report) — eval report presents raw metrics
- JSON schema: Claude's discretion on nesting structure

### EvalRunner orchestration
- `EvalRunner` class in `src/aquapose/evaluation/` with a `run()` method
- Discovers and loads per-stage pickle caches from the run directory
- Prefer cache-files-only operation; fall back to config.yaml from the run directory if evaluators need parameters like n_animals (config.yaml is always present and consistently located in run dirs)
- Class name: `EvalRunner`

### Claude's Discretion
- JSON schema nesting structure (mirror report vs. flat machine-friendly)
- Internal EvalRunner discovery logic for finding stage caches
- How to pass evaluator parameters when cache-only isn't sufficient
- Exact metric formatting (decimal places, units, column widths)

</decisions>

<specifics>
## Specific Ideas

- `scripts/measure_baseline.py` must be deleted as part of this phase (CLEAN-03)
- The existing `evaluation/harness.py` is NOT deleted in this phase — that's Phase 50. EvalRunner coexists with it for now.
- Config.yaml is written to every run directory automatically, so if evaluators need config values, it's a reliable source

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluation/output.py`: `format_summary_table`, `write_regression_json`, `_NumpySafeEncoder` — can be extended for multi-stage formatting
- `evaluation/harness.py`: `run_evaluation()` and `EvalResults` dataclass — reference for how reconstruction evaluation currently works
- `evaluation/metrics.py`: `Tier1Result`, `Tier2Result`, `compute_tier1`, `compute_tier2`, `select_frames` — existing metric computation

### Established Patterns
- CLI uses Click with `@cli.group()` and `@cli.command()` — new `eval` command follows this pattern
- `PosePipeline` is a class with `run()` — EvalRunner mirrors this pattern
- Frozen dataclasses for results (`EvalResults`) — multi-stage results should follow same pattern

### Integration Points
- `cli.py`: Add `eval` command to the existing Click group
- Phase 46 pickle caches: EvalRunner reads these from `<run-dir>/diagnostics/<stage>_cache.pkl`
- Phase 47 stage evaluators: EvalRunner calls these with loaded cache data

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 48-evalrunner-and-aquapose-eval-cli*
*Context gathered: 2026-03-03*
