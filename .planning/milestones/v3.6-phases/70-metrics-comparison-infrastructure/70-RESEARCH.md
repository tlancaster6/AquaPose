# Phase 70: Metrics & Comparison Infrastructure - Research

**Researched:** 2026-03-06
**Domain:** Evaluation metrics extension (numpy/scipy statistics, refractive projection, frozen dataclass patterns)
**Confidence:** HIGH

## Summary

This phase extends the existing `aquapose eval` system with five categories of new metrics: percentiles for reprojection error / midline confidence / camera count (EVAL-01/02/03), per-keypoint reprojection error breakdown (EVAL-04), curvature-stratified reconstruction quality (EVAL-05), and 3D track fragmentation analysis (EVAL-06). All metrics must appear in both text and JSON output formats.

The existing evaluation architecture is well-structured and highly extensible. Stage evaluators are pure functions in `evaluation/stages/` that return frozen dataclasses with `to_dict()` methods. The `EvalRunner` orchestrates evaluation and assembles an `EvalRunnerResult`. Output formatting lives in `evaluation/output.py` with separate `format_eval_report()` (text) and `format_eval_json()` (JSON) functions. The key implementation challenge is EVAL-04 (per-keypoint reprojection error), which requires loading calibration data and refractive projection models at eval time -- but the existing reconstruction backend (`dlt.py`) already demonstrates this exact pattern (lines 349-377).

**Primary recommendation:** Extend existing frozen dataclasses with new fields (using defaults for backward compatibility), add new computation functions following the pure-function evaluator pattern, and extend the two output formatters. No new CLI commands or architectural changes needed.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Curvature stratification (EVAL-05): Compute curvature from 2D midlines in the diagnostic cache, not from 3D splines. Use the 2D midline from the camera with highest mean keypoint confidence as the representative curvature per fish-frame. Reuse existing `compute_curvature()` from `training/pseudo_labels.py`. Stratify into 4 quantile bins (quartiles). Report reprojection error (mean + p90) per quartile bin with sample counts.
- Track fragmentation (EVAL-06): Two levels -- frame-level gaps (within a fish ID's lifespan) and track-level fragmentation (total unique fish IDs vs expected n_animals, birth/death count, lifespan stats). Any single missing frame counts as a gap.
- Per-keypoint reprojection error (EVAL-04): "Body point index" means the N arc-length sample points on the spline (e.g., 15 points). Compute by re-projecting each 3D sample point through each observing camera. Requires calibration data at eval time. Report mean + p90 per point index.
- Percentile metrics (EVAL-01/02/03): Reprojection error p50/p90/p95; midline confidence p10/p50/p90; camera count p50/p90.
- Report organization: Extend existing stage sections (except track fragmentation gets its own section). JSON uses flat keys within existing stage dicts.

### Claude's Discretion
- Exact dataclass structure for new metric types
- How to handle edge cases (e.g., fish with no 2D midline confidence, single-frame tracks)
- Whether to add LUT loading to EvalRunner or use direct refractive projection for per-keypoint error

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | Reprojection error percentiles (p50, p90, p95) | Extend `ReconstructionMetrics` with percentile fields; compute from existing `all_residuals` in `compute_tier1()` |
| EVAL-02 | Midline confidence percentiles (p10, p50, p90) | Extend `MidlineMetrics` with percentile fields; compute from existing `all_confidences` list in `evaluate_midline()` |
| EVAL-03 | Camera count percentiles (p50, p90) | Extend `AssociationMetrics` with percentile fields; compute from existing `camera_distribution` data |
| EVAL-04 | Per-keypoint reprojection error (mean + p90 per body point index) | New computation in reconstruction evaluator; load calibration + build `RefractiveProjectionModel` per camera; reproject spline sample points; pattern exists in `dlt.py` lines 349-377 |
| EVAL-05 | Curvature-stratified reconstruction quality | New computation combining `compute_curvature()` on best-camera 2D midlines with per-fish reprojection errors; quartile binning via `np.quantile` |
| EVAL-06 | 3D track fragmentation analysis | New evaluator or extension; analyze `midlines_3d` frame presence per fish_id for gaps; analyze `tracklet_groups` for track-level metrics |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (existing) | Percentile computation, array operations | Already used throughout evaluation |
| scipy.interpolate | (existing) | BSpline evaluation for spline sample points | Already used in reconstruction evaluator |
| dataclasses | (stdlib) | Frozen metric dataclasses | Existing pattern in all stage evaluators |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch | (existing) | RefractiveProjectionModel.project() | EVAL-04 per-keypoint reprojection |
| aquapose.calibration | (internal) | CalibrationData, RefractiveProjectionModel | EVAL-04 loading camera models |
| aquapose.training.pseudo_labels | (internal) | compute_curvature() | EVAL-05 curvature stratification |

No new external dependencies needed.

## Architecture Patterns

### Existing Evaluation Architecture (follow exactly)
```
evaluation/
├── metrics.py          # Tier1Result, Tier2Result, compute_tier1(), select_frames()
├── runner.py           # EvalRunner orchestrates stage evaluators, EvalRunnerResult aggregates
├── output.py           # format_eval_report() (text), format_eval_json() (JSON), _NumpySafeEncoder
├── stages/
│   ├── __init__.py     # Re-exports all stage evaluators and metrics
│   ├── detection.py    # DetectionMetrics + evaluate_detection()
│   ├── tracking.py     # TrackingMetrics + evaluate_tracking()
│   ├── association.py  # AssociationMetrics + evaluate_association()
│   ├── midline.py      # MidlineMetrics + evaluate_midline()
│   └── reconstruction.py  # ReconstructionMetrics + evaluate_reconstruction()
└── tuning.py           # Tuning sweep (not relevant)
```

### Pattern 1: Frozen Dataclass with `to_dict()`
**What:** Every metrics result is a `@dataclass(frozen=True)` with a `to_dict()` method that returns JSON-serializable dicts
**When to use:** All new metric types
**Example:**
```python
# Source: evaluation/stages/reconstruction.py
@dataclass(frozen=True)
class ReconstructionMetrics:
    mean_reprojection_error: float
    # ... fields ...

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_reprojection_error": float(self.mean_reprojection_error),
            # ... cast numpy scalars to Python builtins ...
        }
```

### Pattern 2: Pure Function Evaluator
**What:** Stage evaluators are pure functions that take data, return frozen dataclass. No side effects.
**When to use:** All new metric computations
**Example:**
```python
# Source: evaluation/stages/midline.py
def evaluate_midline(frames: list[dict[int, Midline2D]]) -> MidlineMetrics:
    # compute metrics from data
    return MidlineMetrics(...)
```

### Pattern 3: EvalRunner Orchestration
**What:** `EvalRunner.run()` calls each stage evaluator with unpacked `PipelineContext` data, assembles `EvalRunnerResult`
**When to use:** Adding new evaluation stages or passing new data to evaluators
**Key insight:** EvalRunner already reads `config.yaml` for `n_animals`. For EVAL-04, extend it to also load calibration data from the run's config.

### Pattern 4: Dual Output Formatting
**What:** `format_eval_report()` builds ASCII text line-by-line; `format_eval_json()` calls `result.to_dict()` + JSON dumps
**When to use:** All new metrics must be formatted in both functions

### Anti-Patterns to Avoid
- **Breaking frozen dataclass backward compat:** All new fields on existing dataclasses MUST have defaults (e.g., `new_field: float | None = None`) so existing code that constructs them without the new fields still works.
- **Putting computation in output formatters:** Keep computation in evaluators, formatting in output.py.
- **Using `.numpy()` without `.cpu()`:** When working with torch tensors from RefractiveProjectionModel.project(), always use `.detach().cpu().numpy()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Percentile computation | Manual sorting/indexing | `np.percentile(arr, [50, 90, 95])` | Handles edge cases, interpolation |
| Curvature from 2D midlines | New curvature function | `compute_curvature()` from `training/pseudo_labels.py` | Already validated, uses finite difference tangent approach |
| Refractive 3D-to-pixel projection | Custom projection math | `RefractiveProjectionModel.project()` | Handles Snell's law, Newton-Raphson iterations |
| BSpline evaluation | Manual spline math | `scipy.interpolate.BSpline` with existing knots/control_points | Existing pattern in reconstruction evaluator |
| Quantile binning | Manual bin edge computation | `np.quantile(values, [0.25, 0.5, 0.75])` + `np.digitize()` | Robust edge handling |

## Common Pitfalls

### Pitfall 1: CUDA Tensors from RefractiveProjectionModel
**What goes wrong:** `RefractiveProjectionModel.project()` returns CUDA tensors if calibration was loaded on GPU
**Why it happens:** Projection models store camera parameters on GPU device
**How to avoid:** Always use `.detach().cpu().numpy()` on projection outputs
**Warning signs:** `RuntimeError: can't convert cuda:0 device type tensor to numpy`

### Pitfall 2: Empty/Missing Data in Metric Computation
**What goes wrong:** Division by zero or empty array percentile when a fish has no observations
**Why it happens:** Some frames may have zero reconstructions, zero midlines, or single-camera-only observations
**How to avoid:** Guard every percentile/mean computation with `if len(arr) > 0` checks; return sentinel values (0.0 or None) for empty data
**Warning signs:** numpy warnings about empty slice

### Pitfall 3: Frozen Dataclass Default Mutability
**What goes wrong:** Adding a mutable default (like `list`) to a frozen dataclass field
**Why it happens:** Python dataclass restriction on mutable defaults
**How to avoid:** Use `field(default_factory=dict)` or `None` defaults with optional types

### Pitfall 4: JSON Serialization of New Types
**What goes wrong:** `TypeError: Object of type ndarray is not JSON serializable`
**Why it happens:** numpy scalars/arrays leak into `to_dict()` output
**How to avoid:** Cast ALL values to Python builtins in `to_dict()` methods (int(), float(), str()). Existing `_NumpySafeEncoder` catches numpy scalars but not arrays.

### Pitfall 5: Curvature Computation with Too Few Points
**What goes wrong:** `compute_curvature()` requires at least 3 control points for finite differences (tangents need `diff` twice)
**Why it happens:** Some midlines may have very few valid points
**How to avoid:** Guard with `if control_points.shape[0] < 3: return None` before calling compute_curvature

### Pitfall 6: Track Fragmentation Frame Range Ambiguity
**What goes wrong:** Computing gaps incorrectly when chunk merging creates non-contiguous frame indices
**Why it happens:** `midlines_3d` is a list indexed by frame position, but after chunk merging frame indices are globally offset
**How to avoid:** Use `Midline3D.frame_index` attribute (which has global frame index) rather than list position. For gap analysis, build `fish_id -> set[frame_index]` mapping from midlines_3d entries.

## Code Examples

### Computing Percentiles from Existing Data (EVAL-01/02/03)
```python
# Source: numpy documentation, pattern used in evaluation/metrics.py
import numpy as np

# For reprojection error percentiles (EVAL-01)
all_residuals = [...]  # collected from frame_results
if all_residuals:
    p50, p90, p95 = np.percentile(all_residuals, [50, 90, 95])
else:
    p50 = p90 = p95 = 0.0
```

### Refractive Reprojection for Per-Keypoint Error (EVAL-04)
```python
# Source: core/reconstruction/backends/dlt.py lines 349-377
import scipy.interpolate
from aquapose.core.reconstruction.utils import SPLINE_K

# Evaluate spline at sample points
spline = scipy.interpolate.BSpline(
    midline3d.knots.astype(np.float64),
    midline3d.control_points.astype(np.float64),
    SPLINE_K,
)
u_sample = np.linspace(0.0, 1.0, n_body_points)
spline_pts_3d = torch.from_numpy(spline(u_sample).astype(np.float32))

# Project through each camera's refractive model
proj_px, valid = model.project(spline_pts_3d)
proj_np = proj_px.detach().cpu().numpy()  # CRITICAL: .cpu() before .numpy()
valid_np = valid.detach().cpu().numpy()

# Compare to observed 2D midline points
for j in range(n_body_points):
    if valid_np[j] and not np.any(np.isnan(proj_np[j])):
        err = float(np.linalg.norm(proj_np[j] - obs_pts[j]))
```

### Curvature Stratification (EVAL-05)
```python
# Source: training/pseudo_labels.py compute_curvature()
from aquapose.training.pseudo_labels import compute_curvature

# For each fish-frame: pick camera with highest mean confidence
best_cam_midline = max(cam_midlines.values(), key=lambda m: np.mean(m.point_confidence or np.ones(1)))
curvature = compute_curvature(best_cam_midline.points)  # works on 2D (N,2)

# Bin into quartiles
bin_edges = np.quantile(all_curvatures, [0.25, 0.5, 0.75])
bin_indices = np.digitize(all_curvatures, bin_edges)  # 0-3
```

### Track Fragmentation Analysis (EVAL-06)
```python
# Build per-fish frame presence from midlines_3d
fish_frames: dict[int, set[int]] = {}
for frame_data in midlines_3d:
    if frame_data is None:
        continue
    for fish_id, m3d in frame_data.items():
        fish_frames.setdefault(fish_id, set()).add(m3d.frame_index)

# Frame-level gap analysis per fish
for fish_id, frames in fish_frames.items():
    sorted_frames = sorted(frames)
    lifespan = sorted_frames[-1] - sorted_frames[0] + 1
    continuity_ratio = len(frames) / lifespan
    gaps = [sorted_frames[i+1] - sorted_frames[i] - 1
            for i in range(len(sorted_frames)-1)
            if sorted_frames[i+1] - sorted_frames[i] > 1]
```

## Implementation Strategy

### Recommended Task Ordering

1. **Percentiles (EVAL-01/02/03):** Simplest -- add fields to existing dataclasses, compute from data already collected. Touches `ReconstructionMetrics`, `MidlineMetrics`, `AssociationMetrics`, their `to_dict()`, and both output formatters.

2. **Track fragmentation (EVAL-06):** Independent new evaluator. Create a `FragmentationMetrics` dataclass and `evaluate_fragmentation()` function. New stage section in EvalRunner. New section in both output formatters.

3. **Curvature stratification (EVAL-05):** Moderate complexity. Needs access to both 2D midlines (for curvature) and 3D midlines (for reprojection error). Can be a sub-computation within the reconstruction evaluator or a separate function called from EvalRunner.

4. **Per-keypoint reprojection error (EVAL-04):** Most complex. Requires loading calibration data in EvalRunner, building RefractiveProjectionModel instances, and evaluating splines. The exact pattern already exists in `dlt.py` -- extract and reuse.

### Key Design Decision: Per-Keypoint Error Approach

**Recommendation: Use direct refractive projection (not LUTs).**

Rationale:
- The exact code pattern exists in `dlt.py` lines 349-377 (spline evaluation + refractive projection + residual computation)
- LUTs are for fast pixel-to-ray lookup (inverse direction) -- not needed for 3D-to-pixel (forward direction)
- `RefractiveProjectionModel.project()` handles forward projection directly
- Loading calibration is already demonstrated in `EvalRunner._read_n_animals()` pattern (loads config.yaml)

To get calibration data:
1. `EvalRunner` already has `self._run_dir` with `config.yaml`
2. Load config -> get calibration path -> `load_calibration_data()` -> build `RefractiveProjectionModel` per camera
3. Pass models to the per-keypoint error computation function

### Dataclass Extension Strategy

For EVAL-01/02/03, add optional fields with defaults to existing dataclasses:

```python
@dataclass(frozen=True)
class ReconstructionMetrics:
    # ... existing fields ...
    p50_reprojection_error: float | None = None
    p90_reprojection_error: float | None = None
    p95_reprojection_error: float | None = None
    per_point_error: dict[int, dict[str, float]] | None = None  # EVAL-04
    curvature_stratified: dict[str, dict[str, float]] | None = None  # EVAL-05
```

For EVAL-06, create a new dataclass:

```python
@dataclass(frozen=True)
class FragmentationMetrics:
    # Frame-level
    total_gaps: int
    mean_gap_duration: float
    max_gap_duration: int
    mean_continuity_ratio: float
    per_fish_continuity: dict[int, float]
    # Track-level
    unique_fish_ids: int
    expected_fish: int
    track_births: int
    track_deaths: int
    mean_track_lifespan: float
    median_track_lifespan: float
```

## Open Questions

1. **Number of spline sample points for EVAL-04**
   - What we know: The reconstruction backend uses a configurable `n_body_points` (typically 15). `Midline3D` stores control points (7) and knots (11), not the sample count.
   - What's unclear: Whether to hardcode 15 or detect from the 2D midline point count
   - Recommendation: Default to 15 (matching pipeline default), document as a constant. The 2D midlines in the cache will have the same N points the spline was fit to.

2. **Track-level fragmentation: defining "birth" and "death"**
   - What we know: TrackletGroups have fish_ids assigned by association. midlines_3d has per-frame dicts keyed by fish_id.
   - What's unclear: Whether a fish_id that appears in frame 0 counts as a "birth" (it was always there) vs a fish_id appearing at frame 50 (genuinely new)
   - Recommendation: Count any fish_id whose first appearance is after the global first frame as a "birth"; any fish_id whose last appearance is before the global last frame as a "death". This captures the fragmentation signal without requiring ground-truth fish count.

## Sources

### Primary (HIGH confidence)
- `src/aquapose/evaluation/runner.py` -- EvalRunner orchestration, EvalRunnerResult structure
- `src/aquapose/evaluation/output.py` -- format_eval_report(), format_eval_json() patterns
- `src/aquapose/evaluation/stages/reconstruction.py` -- ReconstructionMetrics, evaluate_reconstruction()
- `src/aquapose/evaluation/stages/midline.py` -- MidlineMetrics, evaluate_midline()
- `src/aquapose/evaluation/stages/association.py` -- AssociationMetrics, evaluate_association()
- `src/aquapose/evaluation/stages/tracking.py` -- TrackingMetrics, evaluate_tracking()
- `src/aquapose/evaluation/metrics.py` -- compute_tier1(), Tier1Result
- `src/aquapose/core/reconstruction/backends/dlt.py` lines 349-377 -- spline reprojection pattern
- `src/aquapose/calibration/projection.py` -- RefractiveProjectionModel.project() API
- `src/aquapose/training/pseudo_labels.py` -- compute_curvature() function
- `src/aquapose/core/types/reconstruction.py` -- Midline3D dataclass fields
- `src/aquapose/core/types/midline.py` -- Midline2D dataclass fields

### Secondary (MEDIUM confidence)
- numpy percentile computation -- standard API, well-known

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use in the project
- Architecture: HIGH -- extending existing well-documented patterns, no new architecture needed
- Pitfalls: HIGH -- identified from direct code reading of existing patterns and known project pitfalls (CUDA tensors, etc.)

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable internal codebase patterns)
