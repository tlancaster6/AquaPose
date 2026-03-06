# Architecture Patterns

**Domain:** Evaluation metrics extension, analysis infrastructure, and model iteration QA for 3D fish pose estimation pipeline
**Researched:** 2026-03-06
**Confidence:** HIGH (based on direct codebase inspection of all evaluation and core type modules)

## Existing Evaluation Architecture

The evaluation system has three layers:

```
CLI: `aquapose eval`
    |
    v
EvalRunner (evaluation/runner.py)
    |-- load_run_context(): discovers chunk_NNN/cache.pkl, merges into single PipelineContext
    |-- run(): unpacks PipelineContext fields, calls per-stage evaluators
    |-- returns EvalRunnerResult (frozen dataclass, 5 optional stage metrics)
    |
    +---> evaluate_detection(frames)           -> DetectionMetrics
    +---> evaluate_tracking(tracklets)         -> TrackingMetrics
    +---> evaluate_association(midline_sets, n) -> AssociationMetrics
    +---> evaluate_midline(frames)             -> MidlineMetrics
    +---> evaluate_reconstruction(results, n)  -> ReconstructionMetrics
    |
    v
Output formatters (evaluation/output.py)
    |-- format_eval_report(result) -> ASCII text
    |-- format_eval_json(result)   -> JSON string
```

### Key Design Properties

1. **Stage evaluators are pure functions** -- they take typed data extracted from `PipelineContext` fields and return frozen metric dataclasses. They never import from `engine/`.

2. **Frozen dataclasses with `to_dict()`** -- every metric type has a `to_dict()` method returning JSON-serializable dicts. New fields must follow this pattern.

3. **Frame sampling** -- `EvalRunner.run(n_frames)` uses `select_frames()` to uniformly sample frames before passing data to evaluators. New analysis functions must respect this sampling.

4. **MidlineSet assembly** -- `EvalRunner._build_midline_sets()` reconstructs per-frame `MidlineSet` (fish_id -> camera_id -> Midline2D) from `tracklet_groups` + `annotated_detections` via centroid matching. This is the bridge between association identity and midline data.

### Cached Data Available in PipelineContext

| Field | Type | Producer Stage | Available Data Per Element |
|-------|------|---------------|--------------------------|
| `detections` | `list[dict[str, list[Detection]]]` | Detection | bbox, confidence, camera_id |
| `tracks_2d` | `dict[str, list[Tracklet2D]]` | Tracking | frames, centroids, frame_status ("detected"/"coasted") |
| `tracklet_groups` | `list[TrackletGroup]` | Association | fish_id, per-camera tracklets, consensus_centroids |
| `annotated_detections` | `list[dict[str, list[AnnotatedDetection]]]` | Midline | detection + Midline2D (points, half_widths, point_confidence) |
| `midlines_3d` | `list[dict[int, Midline3D]]` | Reconstruction | control_points, knots, mean_residual, max_residual, per_camera_residuals, n_cameras, centroid_z, z_offsets |

### What Midline3D Stores vs. What It Does NOT

**Stored:** `mean_residual` (scalar), `max_residual` (scalar), `per_camera_residuals` (per-camera mean across all body points), `control_points` + `knots` (B-spline), `n_cameras` (minimum across body points).

**NOT stored:** Per-body-point residuals, per-body-point camera counts, per-body-point camera IDs. These are computed during reconstruction in `DltBackend._reconstruct_fish()` but discarded after spline fitting.

## Recommended Architecture for v3.6 New Features

### Principle: Extend Evaluators for Simple Aggregates, Add Analysis Module for Cross-Stage Computations

Two categories of work:

**Category A -- Evaluator extensions** (add fields + percentile computation to existing evaluator functions):
- Reprojection error percentiles (p50/p90/p95) in `ReconstructionMetrics`
- Midline confidence percentiles (p10/p50/p90) in `MidlineMetrics`
- Camera count percentiles (p50/p90) in `AssociationMetrics`

**Category B -- New analysis functions** (cross-field or multi-step computations requiring data from multiple context fields):
- Per-keypoint reprojection error breakdown
- Curvature-stratified reconstruction quality
- 3D track fragmentation

Category A belongs inside existing evaluator dataclasses and functions. Category B belongs in a new `evaluation/analysis.py` module.

### Component Boundaries

| Component | Responsibility | Data Source |
|-----------|---------------|-------------|
| `evaluation/stages/reconstruction.py` | Add reproj error percentiles | `Midline3D.mean_residual` (already iterated) |
| `evaluation/stages/midline.py` | Add confidence percentiles | `Midline2D.point_confidence` (already collected) |
| `evaluation/stages/association.py` | Add camera count percentiles | `MidlineSet` n_cams (already counted) |
| **`evaluation/analysis.py` (NEW)** | Per-keypoint, curvature, fragmentation | Multiple `PipelineContext` fields |
| `evaluation/runner.py` | Call analysis functions, extend `EvalRunnerResult` | Orchestrates everything |
| `evaluation/output.py` | Format new metrics + analyses | `EvalRunnerResult` |

### Data Flow for New Features

```
PipelineContext (merged from chunks)
    |
    +---> evaluate_reconstruction(frame_results, ...)
    |         NOW ALSO: p50/p90/p95 reproj error from flat residual list
    |         -> ReconstructionMetrics (extended)
    |
    +---> evaluate_midline(frames)
    |         NOW ALSO: p10/p50/p90 confidence from conf_array
    |         -> MidlineMetrics (extended)
    |
    +---> evaluate_association(midline_sets, n_animals)
    |         NOW ALSO: p50/p90 camera count from n_cams list
    |         -> AssociationMetrics (extended)
    |
    +---> analyze_per_keypoint(frame_results, midline_sets, calib_bundle)
    |         -> KeypointAnalysis (NEW)
    |
    +---> analyze_curvature_quality(frame_results)
    |         -> CurvatureAnalysis (NEW)
    |
    +---> analyze_track_fragmentation(tracklet_groups, midlines_3d, ...)
              -> FragmentationAnalysis (NEW)
```

## Detailed Design: Category A (Evaluator Extensions)

### 1. Reprojection Error Percentiles

**What changes:** Add `p50_reprojection_error`, `p90_reprojection_error`, `p95_reprojection_error` to `ReconstructionMetrics`.

**Data already available:** `evaluate_reconstruction()` iterates all `Midline3D` objects to count low-confidence flags. In the same loop, collect `midline3d.mean_residual` into a flat list, then `np.percentile(residuals, [50, 90, 95])`.

**Implementation:** ~10 lines of change in `evaluate_reconstruction()`, 3 new fields with `float` defaults in `ReconstructionMetrics`, 3 new lines in `to_dict()`.

### 2. Midline Confidence Percentiles

**What changes:** Add `p10_confidence`, `p50_confidence`, `p90_confidence` to `MidlineMetrics`.

**Data already available:** `evaluate_midline()` already builds `conf_array = np.array(all_confidences)`. One `np.percentile()` call.

**Implementation:** ~5 lines of change, 3 new fields, 3 new lines in `to_dict()`.

### 3. Camera Count Percentiles

**What changes:** Add `camera_count_p50`, `camera_count_p90` to `AssociationMetrics`.

**Data already available:** `evaluate_association()` already computes `n_cams = len(cam_map)` per observation. Collect into a list, `np.percentile()`.

**Implementation:** ~5 lines of change, 2 new fields, 2 new lines in `to_dict()`.

### Backward Compatibility

All new fields use default values (`0.0`), so existing code that constructs these dataclasses without the new fields continues to work. The frozen dataclass pattern already supports this via `field(default=0.0)` or keyword defaults. Existing tests constructing these dataclasses remain valid.

## Detailed Design: Category B (New Analysis Module)

### New File: `evaluation/analysis.py`

Contains three frozen dataclasses and three pure analysis functions. No engine imports.

### 4. Per-Keypoint Reprojection Error Breakdown

**Goal:** Mean and p90 reprojection error broken down by body point index (0=head through 14=tail). Reveals whether head/tail points are systematically worse.

**Data challenge:** `Midline3D` does NOT store per-body-point residuals. Two approaches:

**(A) Recompute from spline (RECOMMENDED):** Evaluate the cached B-spline at N uniform positions, reproject into each observing camera using `CalibBundle`, compute per-point pixel error against the observed 2D midline points from `MidlineSet`. This replicates what `dlt.py` lines 353-380 do at reconstruction time.

**(B) Store per-body-point residuals in Midline3D:** Add array field, populate during reconstruction. Simpler analysis but **invalidates all existing caches** (StaleCacheError on deserialize due to `context_fingerprint` mismatch).

**Recommendation: Approach A.** Avoids cache invalidation, keeps analysis self-contained. The function needs `CalibBundle` (for reprojection), confirming it belongs in `analysis.py` not in a stage evaluator.

**Dependencies:** `EvalRunner` must load `CalibBundle` from the run's config. It already reads `config.yaml` for `n_animals`; extending it to load calibration is a small addition.

**Type:**

```python
@dataclass(frozen=True)
class KeypointAnalysis:
    per_keypoint_mean: dict[int, float]   # body_point_index -> mean px error
    per_keypoint_p90: dict[int, float]    # body_point_index -> p90 px error
    n_observations: dict[int, int]        # body_point_index -> observation count
```

### 5. Curvature-Stratified Reconstruction Quality

**Goal:** Bin reconstructions by 3D spline curvature, report reprojection error per bin. Answers: "does reconstruction quality degrade for curved fish?"

**Data available:** `Midline3D.control_points` + `knots` allow spline evaluation. 3D curvature = `|r' x r''| / |r'|^3`. The codebase already has `compute_curvature()` in `training/frame_selection.py` (v3.5) for 2D -- the 3D version is analogous.

**No external dependencies:** Curvature from 3D spline, reprojection error from `Midline3D.mean_residual`. Pure function.

**Implementation:**
1. For each fish-frame: evaluate spline, compute curvature at sample points, take max curvature as scalar summary.
2. Compute curvature quantile bin edges (quartiles) from all fish-frames.
3. Report mean/p90 reprojection error per bin.

**Type:**

```python
@dataclass(frozen=True)
class CurvatureAnalysis:
    bins: list[tuple[float, float]]       # (low, high) curvature edges
    mean_error_per_bin: list[float]        # mean reproj error per bin
    p90_error_per_bin: list[float]         # p90 reproj error per bin
    count_per_bin: list[int]              # observations per bin
    overall_curvature_p50: float
    overall_curvature_p90: float
```

### 6. 3D Track Fragmentation

**Goal:** Post-association track continuity metrics -- how often does a visible fish fail to get reconstructed?

**Data sources:**
- `tracklet_groups`: `TrackletGroup` objects with per-camera tracklets (frame ranges + status)
- `midlines_3d`: per-frame `dict[int, Midline3D]` showing which fish were actually reconstructed
- Cross-reference: for each `fish_id`, compare frames where tracklet_groups indicate >=3 camera views vs frames where `midlines_3d` has an entry

**Implementation:**
1. From `tracklet_groups`: for each fish_id, compute per-frame camera count from union of tracklets' frame ranges (counting only "detected" status frames).
2. From `midlines_3d`: for each fish_id, collect set of frames with reconstruction.
3. Gap = frame where camera_count >= 3 but no reconstruction exists.
4. Continuity = reconstructed_frames / visible_frames.

**Type:**

```python
@dataclass(frozen=True)
class FragmentationAnalysis:
    unique_fish_ids: int
    expected_fish: int
    mean_continuity_ratio: float
    per_fish_continuity: dict[int, float]
    total_gaps: int
    mean_gap_duration: float
    max_gap_duration: int
```

## Integration with EvalRunnerResult

Extend `EvalRunnerResult` with three optional analysis fields:

```python
@dataclass(frozen=True)
class EvalRunnerResult:
    # ... existing 10 fields ...
    keypoint_analysis: KeypointAnalysis | None = None
    curvature_analysis: CurvatureAnalysis | None = None
    fragmentation_analysis: FragmentationAnalysis | None = None
```

The `to_dict()` method adds these under a top-level `"analyses"` key (separate from `"stages"`) to distinguish per-stage metrics from cross-cutting analyses. Formatters add new report sections when non-None.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Cross-Stage Analysis in a Stage Evaluator
**What:** Adding per-keypoint analysis (needs CalibBundle + 2D observations + 3D splines) into `reconstruction.py`.
**Why bad:** Stage evaluators are pure functions of their stage's output. Adding calibration dependency breaks the evaluator contract.
**Instead:** Put it in `analysis.py` with explicit CalibBundle parameter. EvalRunner orchestrates loading.

### Anti-Pattern 2: Storing Per-Body-Point Data in Midline3D
**What:** Adding `per_point_residuals`, `per_point_camera_ids` arrays to Midline3D to simplify analysis.
**Why bad:** Invalidates all existing caches (context_fingerprint changes trigger StaleCacheError warnings). Bloats cache size. Couples storage format to analysis needs.
**Instead:** Recompute from spline + 2D observations at analysis time. Spline evaluation is cheap.

### Anti-Pattern 3: Analysis Functions That Load Their Own Data
**What:** Having analysis functions call `load_run_context()` or read config files internally.
**Why bad:** Makes them untestable in isolation, couples analysis to file layout.
**Instead:** Analysis functions take typed data as parameters. EvalRunner handles loading and passes data in.

### Anti-Pattern 4: Breaking Frozen Dataclass Backward Compatibility
**What:** Adding required (non-default) fields to existing frozen dataclasses.
**Why bad:** All existing construction sites break. Tests break. Any code that constructs `ReconstructionMetrics(...)` positionally will get wrong field mapping.
**Instead:** Always add new fields with defaults (e.g., `p50_reprojection_error: float = 0.0`). Place new fields after existing fields.

## Suggested Build Order

Three dependency chains, ordered by risk and dependency:

### Chain 1: Evaluator Extensions (independent, parallelizable)
- **1a.** Reprojection error percentiles (reconstruction evaluator) -- ~30 min
- **1b.** Midline confidence percentiles (midline evaluator) -- ~30 min
- **1c.** Camera count percentiles (association evaluator) -- ~30 min

These are leaf changes with no cross-dependencies. Do them first to establish the pattern. Update `format_eval_report()` and `format_eval_json()` for each.

### Chain 2: Analysis Module + Simple Analyses
- **2a.** Create `evaluation/analysis.py` with `CurvatureAnalysis` + `analyze_curvature_quality()` -- no calibration dependency, reads only `Midline3D` fields. Most directly relevant to model iteration loop (validates curved-fish reconstruction quality).
- **2b.** `FragmentationAnalysis` + `analyze_track_fragmentation()` -- reads `tracklet_groups` + `midlines_3d`, no calibration needed.
- **2c.** Wire analysis results into `EvalRunnerResult` (3 new optional fields), update `output.py` formatters.

### Chain 3: Per-Keypoint Analysis (requires calibration plumbing)
- **3a.** Add CalibBundle loading to `EvalRunner` (read calibration path from config.yaml, load via AquaCal).
- **3b.** Implement `analyze_per_keypoint()` with spline evaluation + multi-camera reprojection + per-point error.
- **3c.** Wire into `EvalRunnerResult` and formatters.

### Recommended Order

```
1a + 1b + 1c (parallel, trivial)
    |
    v
2a: CurvatureAnalysis (creates analysis.py, most impactful for iteration loop)
    |
    v
2b: FragmentationAnalysis (second analysis, same module)
    |
    v
2c: Wire analyses into EvalRunnerResult + formatters
    |
    v
3a: CalibBundle loading in EvalRunner
    |
    v
3b + 3c: Per-keypoint analysis + wiring
```

**Rationale:**
- Chain 1 first because it is trivial and establishes the frozen-dataclass-extension pattern.
- CurvatureAnalysis (2a) before FragmentationAnalysis (2b) because curvature analysis is the primary signal for whether retrained models handle curved fish better -- the core question of the iteration loop.
- Per-keypoint (Chain 3) last because it introduces a new dependency (CalibBundle in EvalRunner), requires the most complex computation, and is less critical for the iteration loop's primary question.

## File Change Summary

| File | Change Type | What |
|------|-------------|------|
| `evaluation/stages/reconstruction.py` | MODIFY | +3 percentile fields to `ReconstructionMetrics`, compute in `evaluate_reconstruction()` |
| `evaluation/stages/midline.py` | MODIFY | +3 percentile fields to `MidlineMetrics`, compute in `evaluate_midline()` |
| `evaluation/stages/association.py` | MODIFY | +2 percentile fields to `AssociationMetrics`, compute in `evaluate_association()` |
| `evaluation/analysis.py` | **NEW** | 3 frozen dataclasses + 3 analysis functions |
| `evaluation/runner.py` | MODIFY | +3 optional fields on `EvalRunnerResult`, call analysis functions in `run()`, add CalibBundle loading |
| `evaluation/output.py` | MODIFY | Extend formatters for new metrics and analyses |
| `evaluation/__init__.py` | MODIFY | Export new analysis types and functions |
| `evaluation/stages/__init__.py` | NO CHANGE | Field additions are backward-compatible |

## Sources

- Direct code inspection: `src/aquapose/evaluation/stages/reconstruction.py` -- ReconstructionMetrics fields, evaluate_reconstruction() loop structure
- Direct code inspection: `src/aquapose/evaluation/stages/midline.py` -- MidlineMetrics fields, evaluate_midline() confidence collection
- Direct code inspection: `src/aquapose/evaluation/stages/association.py` -- AssociationMetrics fields, evaluate_association() camera counting
- Direct code inspection: `src/aquapose/evaluation/stages/detection.py` -- DetectionMetrics (no changes needed)
- Direct code inspection: `src/aquapose/evaluation/stages/tracking.py` -- TrackingMetrics (no changes needed)
- Direct code inspection: `src/aquapose/evaluation/runner.py` -- EvalRunner.run() orchestration, _build_midline_sets() assembly, EvalRunnerResult structure
- Direct code inspection: `src/aquapose/evaluation/output.py` -- format_eval_report() and format_eval_json() patterns
- Direct code inspection: `src/aquapose/evaluation/metrics.py` -- Tier1Result, compute_tier1() residual aggregation
- Direct code inspection: `src/aquapose/core/types/reconstruction.py` -- Midline3D fields (what is/isn't stored)
- Direct code inspection: `src/aquapose/core/types/midline.py` -- Midline2D fields
- Direct code inspection: `src/aquapose/core/reconstruction/backends/dlt.py` -- per-body-point residual computation (lines 255-287), spline residual computation (lines 353-380)
- Direct code inspection: `src/aquapose/core/association/types.py` -- TrackletGroup structure
- Direct code inspection: `src/aquapose/core/context.py` -- PipelineContext fields, context_fingerprint cache versioning
- v3.6 seed document: Phase 70 specification

---
*Architecture research for: AquaPose v3.6 Model Iteration & QA (Phase 70: Metrics & Comparison Infrastructure)*
*Researched: 2026-03-06*
