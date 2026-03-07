# Phase 74: Round 1 Evaluation & Decision - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Re-run the pipeline with round 1 models (winner of Phase 73's A/B comparison), compare pipeline-level metrics against the Phase 72 baseline, and make an informed go/no-go decision on whether to proceed to round 2. Builds a reusable `aquapose eval compare` CLI command.

</domain>

<decisions>
## Implementation Decisions

### Comparison CLI
- Build `aquapose eval compare RUN_A RUN_B` command that loads both eval_results.json files
- Output: side-by-side terminal table with deltas (absolute + percentage change) and direction indicators (up/down arrows)
- Also writes comparison JSON (`eval_comparison.json`) to the later run's directory
- All metrics shown in comparison, with primary metrics visually highlighted as decision-drivers

### Decision criteria
- Primary metrics: reprojection error percentiles (p50, p90) and singleton rate
- Decision is purely directional — any improvement in primary metrics = proceed to round 2; regression or stagnation = skip to final validation
- No hard numeric thresholds or cutoffs
- A/B winner from Phase 73 is recorded — comparison documents which models (curated vs uncurated) were used

### Decision recording
- `74-DECISION.md` written manually in the phase directory after reviewing CLI comparison output
- Contains: metric comparison table (from CLI output) and model provenance (which round 1 models, which A/B arm)
- Go/no-go verdict is conversational (not in the doc) — the doc is the data, the decision happens in discussion

### Pipeline re-run setup
- Mirror Phase 72's exact run parameters (check Phase 72 run and match — do not hardcode clip length or chunk settings)
- Only difference: model paths point to round 1 models instead of baseline
- Diagnostic mode (writes chunk caches needed if round 2 proceeds)
- Model paths via SampleStore lookup (consistent with store-based workflow)
- Light pre-flight validation: verify round 1 models are registered in store and .pt files exist before starting the run

### Claude's Discretion
- Exact terminal table layout and column formatting for the comparison CLI
- How primary metrics are visually highlighted (bold, color, marker)
- Pre-flight validation implementation details
- Whether eval compare accepts run directories or run IDs

</decisions>

<specifics>
## Specific Ideas

- Comparison CLI should follow the same pattern as `train compare` (side-by-side table with best-value highlighting)
- Phase 72 run parameters may change (e.g., max-chunks could be bumped) — Phase 74 should check and mirror, not assume specific values

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `EvalRunner` + `EvalRunnerResult.to_dict()`: loads chunk caches, runs all evaluators, produces JSON-serializable results
- `format_eval_report()` / `format_eval_json()`: existing text and JSON formatters for single-run eval
- `training/compare.py`: `format_comparison_table()` with best-value highlighting via `click.style()` — pattern to follow for eval comparison
- `SampleStore`: model registration and path lookup
- `output.py._NumpySafeEncoder`: handles numpy scalars in JSON output

### Established Patterns
- `eval_results.json` written to run directory by `aquapose eval` CLI
- Click-based CLI with `@click.argument` for run directories
- `click.style(bold=True, fg="green")` for best-value highlighting in tables
- Run identity via timestamp-based directory names

### Integration Points
- Phase 72 baseline `eval_results.json` is one input to the comparison
- Phase 73 produces round 1 models registered in SampleStore
- Round 1 run's chunk caches feed into Phase 75 (round 2 pseudo-label generation) if decision is to proceed
- `eval_comparison.json` in run directory provides machine-readable comparison record

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 74-round-1-evaluation-decision*
*Context gathered: 2026-03-07*
