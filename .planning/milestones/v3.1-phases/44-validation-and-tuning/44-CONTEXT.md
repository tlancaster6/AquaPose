# Phase 44: Validation and Tuning - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Confirm the new DltBackend meets or beats the TriangulationBackend baseline on Tier 1 (reprojection error) and Tier 2 (leave-one-out stability) metrics. Empirically tune the outlier rejection threshold and record the final value in the centralized config system and PROJECT.md.

</domain>

<decisions>
## Implementation Decisions

### Threshold Tuning Strategy
- Grid search over outlier_threshold: 10-100px in steps of 5 (19 evaluations)
- Optimization balances reprojection error AND fish count (target: 9 fish per frame)
- Fish count is the primary yield metric — a threshold that drops fish is penalized even if per-fish error is low
- Single global threshold (not per-camera)
- During sweep, only compute Tier 1 + fish count (skip Tier 2 for speed)
- After sweep, run full Tier 1 + Tier 2 on the top 3-5 candidate thresholds

### Pass/Fail Criteria
- No automated pass/fail gate — qualitative judgment by the user
- Accuracy degradation is acceptable if DLT reconstructs more fish than baseline
- Tier 2 (leave-one-out stability) is informational, not gated
- Comparison report shows per-fish breakdown (error + reconstruction status per fish_id) plus aggregates
- Side-by-side display: DLT at best threshold vs TriangulationBackend baseline

### Comparison Workflow
- Standalone script (e.g. scripts/tune_threshold.py), separate from measure_baseline.py
- Grid search output is ephemeral (printed to console, not committed)
- Final tuned threshold is recorded in two places:
  1. ReconstructionConfig default (centralized config system)
  2. PROJECT.md decisions section (as required by success criteria)
- Updating ReconstructionConfig to include the outlier threshold default is part of this phase

### Tuning Scope
- Only tune outlier_threshold (1D search)
- low_confidence_fraction (0.2) stays at default — it only flags, doesn't reject
- min_ray_angle (5°) stays at default — geometric edge-case filter for 2-camera pairs
- If no threshold in 10-100px meets baseline, flag for manual review (no auto-expand)

### Claude's Discretion
- Exact composite scoring formula for balancing error vs fish count during sweep
- How many "top N" candidates to run full Tier 2 on (3-5)
- Console output formatting for the grid search results table
- How to surface the Pareto trade-off in the comparison report

</decisions>

<specifics>
## Specific Ideas

- "A low reprojection error is useless if we get there by throwing away all of our data" — completeness matters as much as accuracy
- Fish count target is 9 per frame (known fish population in the rig)
- The threshold should flow through the centralized config system (frozen dataclasses, YAML → freeze), not be a module-level constant

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `run_evaluation()` in `evaluation/harness.py`: Already supports `backend="dlt"` and `backend="triangulation"`, returns `EvalResults` with tier1/tier2/summary
- `scripts/measure_baseline.py`: Runs eval harness, saves baseline_results.json + baseline_report.txt. Grid search script can follow same pattern
- `DltBackend.from_models()`: Accepts `outlier_threshold` parameter — grid search can construct fresh backends per threshold
- `format_baseline_report()` and `format_summary_table()` in `evaluation/output.py`: Existing formatters for human-readable output

### Established Patterns
- Reconstruction backends are constructed via `from_models()` classmethod with config params
- Eval harness returns structured `EvalResults` dataclass with `tier1: Tier1Result`, `tier2: Tier2Result`
- Config system uses frozen dataclasses with YAML → freeze precedence
- Scripts live in `scripts/` directory, use argparse

### Integration Points
- `DltBackend.DEFAULT_OUTLIER_THRESHOLD` (currently 50.0) needs to be replaced by config-driven default
- `ReconstructionConfig` in `engine/config.py` is where the new default belongs
- `DltBackend.__init__()` needs to read from config rather than module constant
- `PROJECT.md` decisions section for recording the empirical threshold value

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 44-validation-and-tuning*
*Context gathered: 2026-03-02*
