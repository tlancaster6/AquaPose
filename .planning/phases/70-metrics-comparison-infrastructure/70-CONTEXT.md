# Phase 70: Metrics & Comparison Infrastructure - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend `aquapose eval` output with percentiles, per-keypoint reprojection error breakdown, curvature-stratified reconstruction quality, and 3D track fragmentation analysis. All new metrics appear in both text and JSON output. No new CLI commands or pipeline changes -- this phase only adds metrics computation and output formatting within the existing evaluation system.

</domain>

<decisions>
## Implementation Decisions

### Curvature Stratification (EVAL-05)
- Compute curvature from 2D midlines in the diagnostic cache, not from 3D splines
- Use the 2D midline from the camera with highest mean keypoint confidence as the representative curvature per fish-frame
- Reuse existing `compute_curvature()` from `training/pseudo_labels.py` (works on 2D control points)
- Stratify into 4 quantile bins (quartiles) -- adaptive to each run's curvature distribution
- Report reprojection error (mean + p90) per quartile bin with sample counts

### Track Fragmentation (EVAL-06)
- Two levels of fragmentation analysis:
  - **Frame-level gaps:** Within a fish ID's lifespan, frames where 3D reconstruction is missing. Metrics: gap count, mean gap duration (frames), max gap duration. Per-fish continuity ratio (frames with 3D midline / frames in track lifespan) + overall mean continuity ratio.
  - **Track-level fragmentation:** Total unique fish IDs observed vs expected n_animals, track birth/death count, mean/median track lifespan. Detects cases where tracks die entirely and new ones are born.
- Any single missing frame counts as a gap (no minimum threshold)

### Per-Keypoint Reprojection Error (EVAL-04)
- "Body point index" means the N arc-length sample points on the spline (e.g., 15 points), not the 6 anatomical keypoints
- Compute by re-projecting each 3D sample point through each observing camera via refractive projection, measuring pixel distance to corresponding 2D midline point
- Requires calibration data at eval time -- load from the run's config.yaml (extends existing EvalRunner pattern that already reads config for n_animals)
- Report mean + p90 reprojection error for each of the N points individually

### Percentile Metrics (EVAL-01, EVAL-02, EVAL-03)
- Reprojection error percentiles (p50, p90, p95) alongside existing mean/max in reconstruction section
- Midline confidence percentiles (p10, p50, p90) alongside existing mean/std in midline section
- Camera count percentiles (p50, p90) alongside existing distribution in association section

### Report Organization
- Extend existing stage sections rather than creating new top-level sections (except track fragmentation, which gets its own section)
- Percentiles appear alongside existing mean/max/std values in their respective stage sections
- Per-keypoint breakdown: compact table in text (Point Index | Mean px | P90 px), one row per sample point
- Curvature stratification: sub-section under reconstruction with quartile table
- Track fragmentation: new top-level section with frame-level and track-level sub-sections
- JSON: flat keys within existing stage dicts (e.g., `p50_reprojection_error`, `per_point_error`, `curvature_stratified`), no nesting under sub-keys

### Claude's Discretion
- Exact dataclass structure for new metric types
- How to handle edge cases (e.g., fish with no 2D midline confidence, single-frame tracks)
- Whether to add LUT loading to EvalRunner or use direct refractive projection for per-keypoint error

</decisions>

<specifics>
## Specific Ideas

- User wants to distinguish "gappy tracks that survive" from "tracks that die and get reborn" -- the two fragmentation levels address fundamentally different failure modes
- 2D curvature for stratification was chosen because it matches what the model "sees" and aligns with the elastic augmentation work (OKS-vs-curvature analysis from v3.5)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `compute_curvature()` in `training/pseudo_labels.py`: computes 2D curvature from control points -- reusable for curvature stratification
- `EvalRunner` in `evaluation/runner.py`: orchestrates per-stage evaluation, already reads config.yaml for n_animals
- `compute_tier1()` in `evaluation/metrics.py`: aggregates per-camera/per-fish reprojection errors -- extend for percentiles
- `format_eval_report()` / `format_eval_json()` in `evaluation/output.py`: text and JSON formatters to extend
- `ReconstructionMetrics`, `MidlineMetrics`, `AssociationMetrics`: frozen dataclasses with `to_dict()` methods

### Established Patterns
- Stage evaluators are pure functions in `evaluation/stages/` that take pipeline data and return frozen dataclass metrics
- `EvalRunnerResult` aggregates all stage metrics with `to_dict()` for JSON serialization
- `_NumpySafeEncoder` handles numpy types in JSON output
- Per-camera and per-fish breakdowns use `dict[str, dict[str, float]]` pattern

### Integration Points
- `EvalRunner.run()` calls each stage evaluator and assembles `EvalRunnerResult`
- `Midline3D` has `control_points`, `knots`, `per_camera_residuals`, `mean_residual`, `max_residual`
- Calibration loading via `load_config()` + AquaCal loader already exists
- `PipelineContext` in diagnostic caches has `midlines_3d`, `annotated_detections`, `tracklet_groups`

</code_context>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 70-metrics-comparison-infrastructure*
*Context gathered: 2026-03-06*
