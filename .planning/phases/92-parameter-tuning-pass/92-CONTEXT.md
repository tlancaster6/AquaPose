# Phase 92: Parameter Tuning Pass - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Empirically calibrate new v3.8 association config parameters on real data using the existing tuning harness. Validate improvement in singleton rate over v3.7 baseline without degrading reprojection error. Commit tuned defaults and a brief results document.

</domain>

<decisions>
## Implementation Decisions

### Tuning grid
- Sweep `keypoint_confidence_floor` at 3 points: 0.2, 0.3, 0.4 (narrow range around default)
- Re-sweep `ray_distance_threshold` and `score_min` alongside the new param (multi-keypoint scoring may shift optimal values)
- Hold all other params at current defaults (validation, recovery, leiden, etc.)
- Validation and recovery params (min_segment_length, eviction_reproj_threshold, recovery thresholds) stay at implementation defaults — not swept

### Baseline comparison
- Run v3.7 config through the same tuning harness on the same cached data (apples-to-apples)
- Add a config toggle (e.g. `use_multi_keypoint_scoring: bool`) to disable multi-keypoint scoring for v3.7 baseline
- Generate a fresh diagnostic cache before tuning (current code with keypoint data in tracklets)
- Use the same short clip as Phase 72 (~1 min, --max-chunks 6, YH project)

### Metrics and targets
- Singleton rate is the primary metric for ranking grid candidates
- Reproj error and grouping quality are guardrails (must not degrade)
- Reproj error guardrail: must stay under 5px mean
- Singleton rate target: ~15%, floor: must beat 27% (v3.7 baseline)

### Tuning harness scope
- Harness replays the full association pipeline: scoring -> clustering -> validation -> recovery
- Both association and singleton recovery are included since recovery affects singleton rate
- Same tuning CLI entry point: `aquapose tune --stage association`
- Extend DEFAULT_GRID in association evaluator with the new params

### Final validation
- After selecting best config, run one full end-to-end pipeline run with tuned params (satisfies EVAL-02)
- Use the same short clip for the E2E run

### Claude's Discretion
- Exact grid values for ray_distance_threshold and score_min re-sweep
- Results document format and level of detail
- Whether to add additional metrics to the comparison beyond what's already computed
- Implementation details of the v3.7 scoring toggle

</decisions>

<specifics>
## Specific Ideas

- Keep it simple — the machinery exists, the defaults were chosen deliberately, this is "run what we have, measure, adjust if needed"
- Don't over-engineer the tuning protocol; the user hasn't seen v3.8 outputs yet and wants data before investing in elaborate methodology

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TuningOrchestrator` in `cli.py`: existing `tune` CLI command with grid search over cached diagnostic data
- `DEFAULT_GRID` in `evaluation/stages/association.py`: current grid for ray_distance_threshold, score_min, eviction_reproj_threshold, leiden_resolution, early_k
- `AssociationMetrics` dataclass: already computes singleton_rate, fish_yield_ratio, camera_distribution, percentiles
- `evaluate_association()`: pure-function evaluator consuming MidlineSet lists

### Established Patterns
- Tuning harness replays from cached diagnostic outputs (no live pipeline per config point)
- Grid search produces ranked candidates with metric comparison
- `AssociationConfig` is a frozen dataclass — all params are YAML-tunable

### Integration Points
- `AssociationConfig` in `engine/config.py`: add scoring toggle here
- `DEFAULT_GRID` in `evaluation/stages/association.py`: extend with new params
- `aquapose tune --stage association` CLI command: no changes needed to CLI itself
- Phase 72 baseline run: `~/aquapose/projects/YH/runs/run_20260307_140127/`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 92-parameter-tuning-pass*
*Context gathered: 2026-03-11*
