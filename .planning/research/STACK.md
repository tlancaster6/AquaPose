# Technology Stack

**Project:** AquaPose v3.6 Model Iteration & QA
**Researched:** 2026-03-06
**Scope:** Stack additions for evaluation metrics extensions, curvature-stratified analysis, per-keypoint breakdown, track fragmentation, and iteration loop orchestration

## Key Finding: No New Dependencies Required

Every feature in the v3.6 milestone can be implemented using libraries already in the dependency set. The new metrics are pure numerical computations over data structures that already exist in the codebase.

**Rationale:** The existing stack (NumPy, SciPy, B-spline evaluation, frozen dataclasses, JSON output) provides everything needed. Adding dependencies for simple percentile/binning/counting operations would be over-engineering.

---

## Recommended Stack (Existing -- No Changes)

### Core Computation Libraries (already installed)

| Technology | Version | Purpose in v3.6 | Why Sufficient |
|------------|---------|-----------------|----------------|
| NumPy | >=1.24 | Percentile computation, per-keypoint aggregation, curvature binning | `np.percentile`, `np.digitize`, `np.histogram` cover all binning/percentile needs |
| SciPy | >=1.11 | B-spline evaluation for 3D curvature from `Midline3D` control points | `scipy.interpolate.BSpline` already used in reconstruction evaluator |
| Python stdlib `dataclasses` | 3.11+ | Extended frozen metric dataclasses | Existing pattern: `ReconstructionMetrics`, `MidlineMetrics`, etc. |
| Python stdlib `json` | 3.11+ | JSON serialization of extended metrics | Existing `_NumpySafeEncoder` handles numpy scalars |
| Python stdlib `sqlite3` | 3.11+ | Model lineage queries for comparison | Already used by SQLite sample store |

### CLI and Output (already installed)

| Technology | Version | Purpose in v3.6 | Why Sufficient |
|------------|---------|-----------------|----------------|
| Click | >=8.1 | No new CLI subcommands needed; metrics appear in existing `aquapose eval` output | Existing `eval` and `train compare` commands |

### Libraries Explicitly NOT Needed

| Library | Why Considered | Why Not Adding |
|---------|---------------|----------------|
| pandas | Tabular metric comparison across rounds | Overkill. Dict-of-dicts + JSON output is the existing pattern. Adding pandas for a few comparison tables adds a heavy transitive dependency tree for no real benefit. |
| matplotlib/seaborn | Plotting curvature-vs-error, regression charts | The milestone SEED doc specifies text/JSON output via `aquapose eval` and overlay video via `aquapose viz`. No new plot types are specified. If needed later, matplotlib is already transitively available via ultralytics. |
| polars | Fast dataframe operations | Same reasoning as pandas -- the data volumes (hundreds of fish-frames per eval run) don't justify a dataframe library. |
| rich | Pretty terminal tables | Click + plain ASCII formatting is the established pattern in `output.py`. Consistency matters more than pretty printing. |
| pydantic | Metric schema validation | Frozen dataclasses are the project's decided pattern (KEY DECISION from PROJECT.md). |

---

## Integration Points for New Features

### 1. Reprojection Error Percentiles

**Where:** Extend `ReconstructionMetrics` dataclass in `evaluation/stages/reconstruction.py`

**Data source:** `Midline3D.mean_residual` per fish-frame (already collected in `evaluate_reconstruction`)

**Computation:** `np.percentile(all_residuals, [50, 90, 95])` -- one line of NumPy

**New fields on `ReconstructionMetrics`:**
```python
reprojection_p50: float
reprojection_p90: float
reprojection_p95: float
```

### 2. Per-Keypoint Reprojection Error Breakdown

**Where:** New analysis function in `evaluation/stages/reconstruction.py` or new file `evaluation/stages/keypoint_analysis.py`

**Data source:** `Midline3D.per_camera_residuals` gives per-camera mean, but per-keypoint requires re-evaluating the spline at body point positions and computing per-point residuals. The spline (`control_points` + `knots`) and projection models are available in the cached `PipelineContext`.

**Key insight:** The DLT backend computes per-body-point residuals internally (`_TriangulationResult.mean_residuals`) but only stores the aggregate on `Midline3D.mean_residual`. Two approaches:
1. **Post-hoc recomputation** (recommended): Evaluate the stored B-spline at N sample points, reproject into each camera, compute per-point errors. This matches the existing evaluator pattern of operating on cached `Midline3D` objects without needing raw triangulation intermediates.
2. **Store per-point residuals on Midline3D**: Would require adding a field and changing the DLT backend. More invasive, but avoids recomputation.

**Recommendation:** Post-hoc recomputation. It keeps the core types stable and follows the established evaluator pattern. The computation is cheap (15 body points x 12 cameras x N frames, all vectorizable).

**Computation:** NumPy + SciPy B-spline eval + projection model (already available via `CalibBundle` / `DltBackend.from_models`)

### 3. Curvature-Stratified Reconstruction Quality

**Where:** New analysis function, likely in `evaluation/stages/reconstruction.py`

**Data source:** `Midline3D.control_points` + `Midline3D.knots` for 3D curvature; `Midline3D.mean_residual` for quality

**Existing curvature function:** `training.pseudo_labels.compute_curvature()` computes mean absolute curvature from control points via finite differences. This can be reused directly or extracted to a shared utility.

**Computation:**
1. Compute curvature per fish-frame using `compute_curvature(midline3d.control_points)`
2. Bin into quantiles: `np.percentile(curvatures, [25, 50, 75])` to define bin edges
3. Report mean reprojection error per bin: `np.digitize` + groupby aggregation

**New output structure:**
```python
@dataclass(frozen=True)
class CurvatureStratifiedMetrics:
    bin_edges: tuple[float, ...]  # curvature quantile boundaries
    per_bin_mean_error: dict[str, float]  # "q1", "q2", "q3", "q4" -> mean px
    per_bin_count: dict[str, int]
    curvature_error_correlation: float  # Pearson r
```

**Dependency note:** `compute_curvature` currently lives in `training.pseudo_labels`. It should be moved to a shared location (e.g., `core/types/reconstruction.py` or a new `core/geometry.py`) to avoid the evaluation module importing from training. This is a code organization move, not a dependency addition.

### 4. Midline Confidence Percentiles

**Where:** Extend `MidlineMetrics` dataclass in `evaluation/stages/midline.py`

**Data source:** Already collected as `all_confidences` list in `evaluate_midline()`

**Computation:** `np.percentile(conf_array, [10, 50, 90])`

**New fields on `MidlineMetrics`:**
```python
confidence_p10: float
confidence_p50: float
confidence_p90: float
```

### 5. Camera Count Percentiles

**Where:** Extend `AssociationMetrics` dataclass in `evaluation/stages/association.py`

**Data source:** Already available from `camera_distribution` histogram

**Computation:** Reconstruct per-observation camera counts from the histogram, then `np.percentile`

**New fields on `AssociationMetrics`:**
```python
camera_count_p50: float
camera_count_p90: float
```

### 6. 3D Track Fragmentation Analysis

**Where:** New section in reconstruction evaluator or new file `evaluation/stages/track_analysis.py`

**Data source:** `frame_results: list[tuple[int, dict[int, Midline3D]]]` -- the same input already consumed by `evaluate_reconstruction`. Track identity comes from `Midline3D.fish_id`, frame index from `Midline3D.frame_index`.

**Computation:**
1. Group by `fish_id` to get per-fish frame sequences
2. Detect gaps: consecutive frame indices with missing reconstructions
3. Count fragments: number of contiguous runs per fish
4. Continuity ratio: frames with reconstruction / total frames in range

**All pure Python/NumPy -- no new dependencies.**

**New output structure:**
```python
@dataclass(frozen=True)
class TrackFragmentationMetrics:
    total_tracks: int
    mean_fragments_per_track: float
    mean_gap_duration: float  # frames
    max_gap_duration: int
    mean_continuity_ratio: float  # 0-1
    per_fish_continuity: dict[int, float]
```

### 7. Model Comparison / Regression Detection

**Where:** Existing `aquapose train compare` CLI + extended `aquapose eval` JSON output

**Approach:** Compare JSON eval outputs from different rounds. No new infrastructure needed -- the user runs `aquapose eval` on each round's diagnostic caches and compares the JSON files. The SEED doc confirms this is the intended workflow ("before/after comparison").

**For automated regression detection:** Simple threshold checks on key metrics (reprojection error, singleton rate). Implementable as a comparison function that takes two `EvalRunnerResult.to_dict()` outputs and flags regressions. Pure Python dict comparison.

---

## Installation

No changes to `pyproject.toml` dependencies. All features use existing packages:

```toml
# Already in pyproject.toml -- NO CHANGES NEEDED
dependencies = [
    "numpy>=1.24",
    "scipy>=1.11",
    # ... all other existing deps unchanged
]
```

---

## Version Verification

| Library | Required | Current Constraint | Verified Feature Availability |
|---------|----------|--------------------|-------------------------------|
| NumPy | `np.percentile`, `np.digitize` | >=1.24 | Both available since NumPy 1.0. HIGH confidence. |
| SciPy | `BSpline` evaluation | >=1.11 | `scipy.interpolate.BSpline` available since SciPy 0.19. Already used throughout codebase. HIGH confidence. |
| Python | `dataclasses`, `json`, `sqlite3` | >=3.11 | All stdlib. HIGH confidence. |

---

## Code Organization Recommendations

### Move `compute_curvature` to shared location

**Current location:** `src/aquapose/training/pseudo_labels.py`
**Problem:** Evaluation code cannot import from training without violating module boundaries
**Recommended location:** `src/aquapose/core/types/reconstruction.py` (alongside `Midline3D`) or new `src/aquapose/core/geometry.py`
**Impact:** One function move + import update in pseudo_labels.py

### Keep new analysis functions alongside existing evaluators

**Pattern:** Each stage evaluator is a single file in `evaluation/stages/`. New analysis functions (per-keypoint breakdown, curvature stratification, track fragmentation) should either:
- Extend existing evaluator files (if tightly coupled to existing metrics), or
- Live in new files in the same directory (if logically independent)

**Recommendation:** Per-keypoint and curvature-stratified metrics extend the reconstruction evaluator. Track fragmentation gets its own file since it operates on a different abstraction level (temporal sequences vs. per-frame quality).

### Extend `to_dict()` and output formatting

Every new metric field needs:
1. Addition to the frozen dataclass
2. Entry in `to_dict()` for JSON serialization
3. Row in `format_eval_report()` for ASCII output

This is mechanical but must not be forgotten -- it's the pattern that makes `aquapose eval --json` work.

---

## Sources

- Existing codebase analysis (HIGH confidence -- direct code reading)
- NumPy documentation for `np.percentile`, `np.digitize` (HIGH confidence -- stable API since NumPy 1.x)
- SciPy documentation for `scipy.interpolate.BSpline` (HIGH confidence -- already used in codebase)
- No external sources needed -- all recommendations are based on extending existing patterns with existing tools

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| No new dependencies | HIGH | All computations are simple NumPy/SciPy operations on existing data structures |
| Integration points | HIGH | Direct code reading of evaluators, types, and DLT backend |
| Per-keypoint approach | MEDIUM | Post-hoc recomputation vs. storing on Midline3D is a design choice; either works |
| Track fragmentation | HIGH | Straightforward groupby + gap detection on frame indices |
| Code organization | HIGH | Follows established patterns visible in codebase |
