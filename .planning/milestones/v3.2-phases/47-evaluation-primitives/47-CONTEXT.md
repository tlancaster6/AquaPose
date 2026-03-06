# Phase 47: Evaluation Primitives - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Pure-function stage evaluators for all five pipeline stages (detection, tracking, association, midline, reconstruction). Each evaluator accepts typed arguments and returns a frozen metric dataclass. DEFAULT_GRIDS for association and reconstruction parameters are colocated with their evaluator modules. No CLI, no runner, no reporting — those are Phase 48+.

</domain>

<decisions>
## Implementation Decisions

### Metric Selection
- Detection: yield (total detections), confidence distribution (mean/std), **short-term jitter** (mean absolute frame-to-frame change in detection count per camera — distinguishes detector instability from fish legitimately leaving frame), per-camera balance
- Tracking: track count, track length summary stats (median, mean, min, max), coast frequency (fraction of frames where track is coasting), detection coverage (fraction of detections assigned to tracks)
- Association: fish yield ratio (observed/expected using n_animals param), singleton rate, camera coverage distribution (how many cameras see each fish), cluster quality
- Midline: keypoint confidence stats, midline completeness, temporal smoothness
- Reconstruction: new ReconstructionMetrics dataclass (fresh, not reusing Tier1Result/Tier2Result); computes mean reprojection error, Tier 2 stability, inlier ratio, low-confidence flag rate
- All metric dataclasses include a `to_dict()` method for easy JSON serialization

### Evaluator Input Contracts
- Each evaluator accepts explicit typed parameters — no PipelineContext import, no engine/ dependency
- Caller (Phase 48 EvalRunner) is responsible for unpacking context and gathering cross-stage data
- Association evaluator accepts `n_animals: int` as explicit param (read from config by caller, not by evaluator)
- All evaluators must be constructible from synthetic test data without a real pipeline run

### DEFAULT_GRIDS Design
- Flat `dict[str, list[float]]` — no primary/secondary/joint structure
- Phase 49's TuningOrchestrator decides sweep order independently
- Association grid: match current tune_association.py ranges exactly (ray_distance_threshold, score_min, eviction_reproj_threshold, leiden_resolution, early_k)
- Reconstruction grid: outlier_threshold (10–100, step 5) AND n_points as sweepable parameters
- Grid values copied from existing scripts as starting point

### Module Organization
- New `evaluation/stages/` subpackage with one file per stage
- Each file contains: frozen metrics dataclass + evaluator function + DEFAULT_GRID (for tunable stages)
- File layout:
  - `evaluation/stages/__init__.py`
  - `evaluation/stages/detection.py` — DetectionMetrics, evaluate_detection()
  - `evaluation/stages/tracking.py` — TrackingMetrics, evaluate_tracking()
  - `evaluation/stages/association.py` — AssociationMetrics, evaluate_association(), DEFAULT_GRID
  - `evaluation/stages/midline.py` — MidlineMetrics, evaluate_midline()
  - `evaluation/stages/reconstruction.py` — ReconstructionMetrics, evaluate_reconstruction(), DEFAULT_GRID
- Stage-specific function names (evaluate_detection, evaluate_tracking, etc.) — not a generic evaluate()
- Existing Tier1Result/Tier2Result stay in metrics.py untouched — Phase 50 handles cleanup

### Claude's Discretion
- Reconstruction evaluator implementation strategy (refactor existing compute_tier1/tier2 logic or fresh implementation producing same quality metrics)
- Exact midline metric formulations (keypoint confidence aggregation, temporal smoothness measure)
- Internal helper functions and shared utilities within evaluation/stages/
- Exact parameter types and defaults for each evaluator function signature

</decisions>

<specifics>
## Specific Ideas

- Detection jitter metric: mean absolute frame-to-frame change in detection count per camera. A camera going 9→9→9→0→0→0 has low jitter (~1.5, one transition); a flickering camera going 9→3→8→2→9→4 has high jitter (~5). This distinguishes legitimate fish movement from detector instability.
- Association evaluator must accept n_animals as explicit function parameter (not import from engine config), but the intent is that the caller reads it from the existing config infrastructure.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluation/metrics.py`: Tier1Result, Tier2Result, compute_tier1(), compute_tier2(), select_frames() — reconstruction evaluator can reuse computation logic internally
- `evaluation/harness.py`: run_evaluation() orchestrates Tier 1 + Tier 2 — reference for how reconstruction metrics are computed from MidlineSet data
- `evaluation/output.py`: format_summary_table(), format_baseline_report(), flag_outliers() — Phase 48 will use these patterns
- `scripts/tune_association.py`: SWEEP_RANGES, SECONDARY_RANGES, _compute_association_metrics() — source for DEFAULT_GRID values and association metric formulas
- `scripts/tune_threshold.py`: threshold range [10, 100] step 5 — source for reconstruction DEFAULT_GRID

### Established Patterns
- Frozen dataclasses for immutable result types (Tier1Result, Tier2Result pattern)
- Pure functions that accept data and return typed results (compute_tier1/compute_tier2 pattern)
- Per-camera and per-fish aggregation patterns in metrics.py

### Integration Points
- `core/types/detection.py`, `core/types/midline.py`, `core/types/reconstruction.py` — stage output types that evaluators will accept as inputs
- Phase 46 ContextLoader will deserialize pickle caches → Phase 48 EvalRunner unpacks → evaluator functions receive typed args
- `evaluation/__init__.py` will need updating to export new stage evaluator symbols

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 47-evaluation-primitives*
*Context gathered: 2026-03-03*
