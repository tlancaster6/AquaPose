# Phase 72: Baseline Pipeline Run & Metrics - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish a quantitative "before" snapshot by running the pipeline on a short iteration clip with baseline models and recording pipeline-level metrics as the benchmark for improvement. No new evaluation tooling — uses existing `aquapose run` and `aquapose eval` commands.

</domain>

<decisions>
## Implementation Decisions

### Iteration clip
- Use the first ~1 minute of the YH project video (no specific segment — behavioral diversity is uniform)
- chunk_size=300 frames (10s per chunk at 30fps)
- Use `--max-chunks 6` to get ~1800 frames (~60s)
- Clip defined via existing `--max-chunks` CLI flag, no new config fields

### Run mode and model selection
- Diagnostic mode (writes chunk caches needed by Phase 73 for pseudo-label generation)
- Baseline models pointed at pipeline via `--set` config overrides (e.g., `--set detection.model_path=... --set midline.model_path=...`)
- Model paths come from store registration (Phase 71 output)

### Metric recording
- eval_results.json written to run directory (existing behavior, no changes)
- JSON only — no markdown summary file
- No special round tagging in eval_results.json — run directory timestamp is the identifier
- Human-readable report goes to stdout only

### Metric comparison
- Comparison CLI deferred to Phase 74 ("Round 1 Evaluation & Decision")
- Phase 72 only records the baseline — no comparison tooling needed yet

### Key metrics for iteration loop
- Reprojection error percentiles (p50, p90, p95)
- Singleton rate (fraction of unassociated tracklets)
- Track continuity ratio (3D track fragmentation)
- OKS curvature slope tracked separately as a model-level metric during training (Phase 73), not as part of pipeline eval

### Success thresholds
- Purely directional — no hard go/no-go cutoffs
- Phase 74 is the decision point where numbers are reviewed

### Reproducibility
- Document the exact `aquapose run` + `aquapose eval` commands (no shell script)
- Recorded in phase verification or README

### Claude's Discretion
- Exact `--set` override syntax for model paths (depends on Phase 71 store output format)
- Whether to add any lightweight validation that the run completed successfully before eval
- Config file adjustments for chunk_size=300

</decisions>

<specifics>
## Specific Ideas

- The first minute of the video is representative — no need to cherry-pick a segment
- chunk_size=300 is preferred over the default 200 for fewer chunks and faster iteration

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `aquapose run` CLI command: fully functional with `--max-chunks`, `--set` overrides, mode selection
- `aquapose eval` CLI command: produces full metric report from diagnostic run dirs, writes eval_results.json
- `EvalRunner`: orchestrates per-stage evaluation from chunk caches (detection, tracking, association, midline, reconstruction, fragmentation)
- `SampleStore`: SQLite-backed training data management with model registration (Phase 71)

### Established Patterns
- Diagnostic mode writes chunk caches to `diagnostics/chunk_NNN/cache.pkl`
- `eval_results.json` is written to run directory on every eval invocation
- Config overrides via `--set key=value` on CLI
- Run identity: timestamp-based (`run_YYYYMMDD_HHMMSS`)

### Integration Points
- Phase 71 (Data Store Bootstrap) provides store-registered baseline model paths
- Phase 70 extended metrics (percentiles, per-keypoint, curvature-stratified, fragmentation) are already in eval
- Run directory caches feed into Phase 73 (pseudo-label generation)
- eval_results.json feeds into Phase 74 (round comparison)

</code_context>

<deferred>
## Deferred Ideas

- `aquapose eval compare RUN_A RUN_B` comparison CLI — belongs in Phase 74
- Val-set OKS integration into `aquapose eval` — potential future phase or Phase 73 scope
- Named model profiles (`--models baseline`) for cleaner invocation — future improvement

</deferred>

---

*Phase: 72-baseline-pipeline-run-metrics*
*Context gathered: 2026-03-07*
