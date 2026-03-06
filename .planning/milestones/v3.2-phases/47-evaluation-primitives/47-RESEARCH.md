# Phase 47: Evaluation Primitives - Research

**Researched:** 2026-03-03
**Domain:** Pure-function stage evaluators, frozen metric dataclasses, DEFAULT_GRIDS
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Metric Selection**
- Detection: yield (total detections), confidence distribution (mean/std), short-term jitter (mean absolute frame-to-frame change in detection count per camera), per-camera balance
- Tracking: track count, track length summary stats (median, mean, min, max), coast frequency (fraction of frames where track is coasting), detection coverage (fraction of detections assigned to tracks)
- Association: fish yield ratio (observed/expected using n_animals param), singleton rate, camera coverage distribution (how many cameras see each fish), cluster quality
- Midline: keypoint confidence stats, midline completeness, temporal smoothness
- Reconstruction: new ReconstructionMetrics dataclass (fresh, not reusing Tier1Result/Tier2Result); computes mean reprojection error, Tier 2 stability, inlier ratio, low-confidence flag rate
- All metric dataclasses include a `to_dict()` method for easy JSON serialization

**Evaluator Input Contracts**
- Each evaluator accepts explicit typed parameters — no PipelineContext import, no engine/ dependency
- Caller (Phase 48 EvalRunner) is responsible for unpacking context and gathering cross-stage data
- Association evaluator accepts `n_animals: int` as explicit param
- All evaluators must be constructible from synthetic test data without a real pipeline run

**DEFAULT_GRIDS Design**
- Flat `dict[str, list[float]]` — no primary/secondary/joint structure
- Association grid: match current tune_association.py ranges exactly (ray_distance_threshold, score_min, eviction_reproj_threshold, leiden_resolution, early_k)
- Reconstruction grid: outlier_threshold (10–100, step 5) AND n_points as sweepable parameters
- Grid values copied from existing scripts as starting point

**Module Organization**
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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | Detection stage evaluator computes yield, confidence distribution, yield stability, and per-camera balance metrics | DetectionMetrics frozen dataclass + evaluate_detection() in evaluation/stages/detection.py; input type: list of per-frame per-camera Detection lists |
| EVAL-02 | Tracking stage evaluator computes track count, track length distribution, coast frequency, and detection coverage metrics | TrackingMetrics frozen dataclass + evaluate_tracking() in evaluation/stages/tracking.py; input type: list of Tracklet2D objects |
| EVAL-03 | Association stage evaluator computes fish yield ratio, singleton rate, camera coverage, and cluster quality metrics | AssociationMetrics frozen dataclass + evaluate_association() in evaluation/stages/association.py; existing _compute_association_metrics() in tune_association.py is direct prior art; n_animals as explicit param |
| EVAL-04 | Midline stage evaluator computes keypoint confidence, midline completeness, and temporal smoothness metrics | MidlineMetrics frozen dataclass + evaluate_midline() in evaluation/stages/midline.py; input type: per-frame per-fish Midline2D dicts with point_confidence fields |
| EVAL-05 | Reconstruction stage evaluator computes mean reprojection error, Tier 2 stability, inlier ratio, and low-confidence flag rate | ReconstructionMetrics frozen dataclass + evaluate_reconstruction() in evaluation/stages/reconstruction.py; can reuse compute_tier1/compute_tier2 logic internally |
| TUNE-06 | DEFAULT_GRIDS for association and reconstruction parameters colocated with stage evaluator modules | DEFAULT_GRID in evaluation/stages/association.py (flat dict merging SWEEP_RANGES + SECONDARY_RANGES from tune_association.py) and in evaluation/stages/reconstruction.py (outlier_threshold 10–100 step 5, n_points) |
</phase_requirements>

## Summary

Phase 47 creates a `evaluation/stages/` subpackage with five pure evaluator modules. Each module owns: a frozen metric dataclass, an evaluator function with no engine/ imports, and (for tunable stages) a DEFAULT_GRID. This is purely an internal API addition — no CLI, no runner, no reporting. The only dependency additions are the existing core types (Detection, Tracklet2D, TrackletGroup, Midline2D, Midline3D/MidlineSet) that the evaluators accept as inputs.

The codebase already has a mature pattern for this work. `evaluation/metrics.py` has frozen dataclasses (`Tier1Result`, `Tier2Result`) and pure compute functions (`compute_tier1`, `compute_tier2`) that serve as direct templates. `scripts/tune_association.py` has `_compute_association_metrics()` that contains exactly the association metric logic to migrate into `evaluate_association()`, and its `SWEEP_RANGES` / `SECONDARY_RANGES` are the verbatim source for `DEFAULT_GRID`. `scripts/tune_threshold.py` defines the reconstruction threshold range (10–100, step 5) that becomes the `outlier_threshold` column in reconstruction's `DEFAULT_GRID`.

The main discretionary work is: (1) deciding how evaluate_reconstruction() handles the Tier 2 leave-one-out computation (reuse compute_tier1/compute_tier2 directly or extract their logic), (2) defining midline metric formulations for temporal smoothness, and (3) choosing exactly what inputs each evaluator accepts as its signature. The existing test infrastructure for evaluation (`tests/unit/evaluation/`) provides clear patterns for the unit tests Wave 0 will need.

**Primary recommendation:** Follow the existing `metrics.py` frozen-dataclass + pure-function pattern exactly. Copy association metrics logic directly from `tune_association.py`'s `_compute_association_metrics()`. Use `compute_tier1` and `compute_tier2` internally in `evaluate_reconstruction()` rather than reimplementing them.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python dataclasses (frozen=True) | stdlib | Immutable metric result types | Established project pattern; prevents mutation; `to_dict()` method easily added |
| numpy | project dep | Metric aggregations (mean, std, median, cumsum) | All existing metric code uses numpy; no alternatives needed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `from __future__ import annotations` | stdlib | Forward references in type hints | Required in all new modules per project pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `to_dict()` method on dataclass | `dataclasses.asdict()` | `to_dict()` is more explicit and handles non-serializable types (numpy arrays) explicitly; `asdict()` recurses into nested dataclasses which is fine but requires numpy→float casts; project preference is explicit `to_dict()` per CONTEXT.md |
| Flat `DEFAULT_GRID` | Primary/secondary/joint structure | Phase 49 TuningOrchestrator decides sweep order independently; flat dict is simpler and more composable |

**Installation:** No new dependencies required — all stdlib + existing project deps.

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/evaluation/
├── __init__.py              # update to export new stage evaluator symbols
├── metrics.py               # UNCHANGED — Tier1Result, Tier2Result stay here
├── harness.py               # UNCHANGED until Phase 50
├── output.py                # UNCHANGED
└── stages/
    ├── __init__.py          # exports all five evaluators + dataclasses
    ├── detection.py         # DetectionMetrics, evaluate_detection()
    ├── tracking.py          # TrackingMetrics, evaluate_tracking()
    ├── association.py       # AssociationMetrics, evaluate_association(), DEFAULT_GRID
    ├── midline.py           # MidlineMetrics, evaluate_midline()
    └── reconstruction.py    # ReconstructionMetrics, evaluate_reconstruction(), DEFAULT_GRID
```

### Pattern 1: Frozen Dataclass with to_dict()
**What:** Each metric type is a `@dataclass(frozen=True)` with a `to_dict()` method returning `dict[str, object]` with all numpy scalars converted to Python builtins.
**When to use:** Every metric dataclass in this phase.
**Example** (from existing `metrics.py` pattern):
```python
# Source: src/aquapose/evaluation/metrics.py
@dataclass(frozen=True)
class Tier1Result:
    per_camera: dict[str, dict[str, float]]
    per_fish: dict[int, dict[str, float]]
    overall_mean_px: float
    overall_max_px: float
    fish_reconstructed: int
    fish_available: int
```

New pattern with `to_dict()`:
```python
@dataclass(frozen=True)
class DetectionMetrics:
    """Metrics for the detection stage."""
    total_detections: int
    mean_confidence: float
    std_confidence: float
    mean_jitter: float          # mean abs frame-to-frame delta in detection count per camera
    per_camera_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable dict."""
        return {
            "total_detections": self.total_detections,
            "mean_confidence": float(self.mean_confidence),
            "std_confidence": float(self.std_confidence),
            "mean_jitter": float(self.mean_jitter),
            "per_camera_counts": dict(self.per_camera_counts),
        }
```

### Pattern 2: Pure Evaluator Function Signature
**What:** Each evaluator accepts concrete stage data, no context imports. Returns typed metric dataclass.
**When to use:** All five evaluator functions.
**Example** (following `compute_tier1` signature style):
```python
# Source: src/aquapose/evaluation/metrics.py compute_tier1 pattern
def evaluate_association(
    midline_sets: list[MidlineSet],
    n_animals: int,
) -> AssociationMetrics:
    """Compute association quality metrics from MidlineSet data.

    Args:
        midline_sets: Per-frame association results (fish_id -> camera_id -> Midline2D).
        n_animals: Expected number of fish (used to compute yield ratio).

    Returns:
        AssociationMetrics with yield ratio, singleton rate, and camera coverage.
    """
    ...
```

### Pattern 3: DEFAULT_GRID as Flat Dict
**What:** `DEFAULT_GRID: dict[str, list[float]]` constant at module level, matching all sweepable parameters for that stage.
**When to use:** `evaluation/stages/association.py` and `evaluation/stages/reconstruction.py`.
**Example** (from `tune_association.py` SWEEP_RANGES + SECONDARY_RANGES):
```python
# Source: scripts/tune_association.py
DEFAULT_GRID: dict[str, list[float]] = {
    "ray_distance_threshold": [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15],
    "score_min": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
    "eviction_reproj_threshold": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10],
    "leiden_resolution": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    "early_k": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
}
```

```python
# Source: scripts/tune_threshold.py (10..100 step 5)
DEFAULT_GRID: dict[str, list[float]] = {
    "outlier_threshold": [float(t) for t in range(10, 105, 5)],
    "n_points": [7.0, 11.0, 15.0, 21.0],  # typical spline sample counts
}
```

### Anti-Patterns to Avoid
- **Importing from engine/:** Evaluators must never `from aquapose.engine import ...`. All config values needed (like `n_animals`) must be explicit function parameters.
- **Reusing Tier1Result/Tier2Result in ReconstructionMetrics:** The new `ReconstructionMetrics` dataclass is a fresh type — it can internally call `compute_tier1`/`compute_tier2` to derive its data, but it should not expose those legacy types in its public interface.
- **Mutable DEFAULT_GRID:** Define as a module-level constant, not a mutable function default. Use a top-level assignment so Phase 49's TuningOrchestrator can import and introspect it.
- **Returning raw dicts instead of typed dataclasses:** All evaluators must return their specific typed dataclass, not a `dict`. The `to_dict()` method handles serialization.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Association metrics logic | New from scratch | Migrate directly from `tune_association.py::_compute_association_metrics()` | Exactly the logic needed; already tested by real runs |
| Reconstruction metrics | Full re-implementation | Call `compute_tier1()` / `compute_tier2()` from `evaluation/metrics.py` internally | These functions are correct and tested; wrap them, don't copy |
| Numpy scalar → Python serialization | Custom JSON encoder per module | Explicit `float()` casts in `to_dict()` | Simpler and consistent with project style (avoids the `_NumpySafeEncoder` class in output.py, which is overkill for a pure dict method) |

**Key insight:** The existing `evaluation/metrics.py` compute functions are already the correct implementation for reconstruction metrics. Phase 47's job is to package them in a new typed wrapper with a clean interface, not rewrite them.

## Common Pitfalls

### Pitfall 1: evaluate_reconstruction() Input Contract
**What goes wrong:** The reconstruction evaluator requires calibration/projection models to reproject 3D points back to 2D for reprojection error. The existing `harness.py` builds these from a `CalibBundle`. In the new architecture, projections models may need to be passed as an explicit parameter or computed from a `CalibBundle` passed as input.
**Why it happens:** The new evaluator cannot import from `engine/`, but it may still need calibration data to reproject Midline3D control points.
**How to avoid:** Pass projection models (or a `CalibBundle`) as an explicit parameter to `evaluate_reconstruction()`. The Midline3D type already stores `mean_residual`, `max_residual`, `per_camera_residuals`, and `is_low_confidence` — these can be read directly without reprojection. The Tier 2 stability metric requires calling DltBackend, which means passing models is required for that metric only. Consider making Tier 2 optional (a `skip_tier2: bool` flag) or accepting pre-computed Tier2Result data.
**Warning signs:** If you find yourself importing `DltBackend` and `RefractiveProjectionModel` in the evaluator, verify they are engine-free (they are in `core/` and `calibration/` respectively — which is acceptable since the constraint is only on `engine/`).

### Pitfall 2: Tracklet2D Coast Fraction Computation
**What goes wrong:** Coast frequency requires per-frame status information from `Tracklet2D.frame_status`. If no frames are available (empty tracklet), division by zero occurs.
**Why it happens:** The `frame_status` field is a plain `tuple` of strings. The evaluator must sum `"coasted"` entries over total entries.
**How to avoid:** Guard against zero-length tracklets. Add `n_frames = len(t.frames)` check before computing `coast_freq = coasted_count / n_frames`.

### Pitfall 3: Jitter Metric Requires Frame-Ordered Data
**What goes wrong:** The detection jitter metric (mean absolute frame-to-frame change in detection count per camera) requires that detection counts are ordered by frame index. If input is an unordered dict or set, the diff is meaningless.
**Why it happens:** Stage output from the pickle cache is a per-frame structure. The evaluator will receive a list of frames; the implementor must sort by frame index before computing diffs.
**How to avoid:** The input should be structured as `list[dict[str, list[Detection]]]` (ordered by frame), or as `dict[int, dict[str, list[Detection]]]` (frame_index → camera_id → detections). Document the expected ordering in the function docstring. Use `sorted()` on frame indices before computing diffs.

### Pitfall 4: __init__.py Update Required
**What goes wrong:** New symbols in `evaluation/stages/` are not importable from `aquapose.evaluation` unless the parent `__init__.py` exports them.
**Why it happens:** Project rule: every package must have an `__init__.py` that explicitly imports its public interface.
**How to avoid:** Update `evaluation/__init__.py` to re-export all five evaluator functions and metric dataclasses. Also create `evaluation/stages/__init__.py` that exports everything from the submodules.

### Pitfall 5: MidlineSet Input for Reconstruction vs Association
**What goes wrong:** Both `evaluate_association()` and `evaluate_reconstruction()` consume data derived from the same `MidlineSet` stage output, but they consume it differently. Association evaluates how many cameras observed each fish (the `len(cam_map)` for each fish). Reconstruction evaluates reprojection error of the triangulated 3D midlines.
**Why it happens:** The `MidlineSet` type (`dict[int, dict[str, Midline2D]]`) is the output of the association stage (Stage 3) and the input to the reconstruction stage (Stage 5). It is not the same as `dict[int, Midline3D]` (the reconstruction output).
**How to avoid:** `evaluate_association()` accepts `list[MidlineSet]` (one per frame). `evaluate_reconstruction()` accepts `list[dict[int, Midline3D]]` (reconstruction results, one per frame) — and optionally a `CalibBundle` or projection models for Tier 2.

## Code Examples

Verified patterns from existing sources:

### Association Metrics Logic (from tune_association.py)
```python
# Source: scripts/tune_association.py::_compute_association_metrics()
from collections import defaultdict

def evaluate_association(
    midline_sets: list[MidlineSet],
    n_animals: int,
) -> AssociationMetrics:
    camera_distribution: dict[int, int] = defaultdict(int)
    total_observations = 0
    singleton_count = 0

    for midline_set in midline_sets:
        for _fish_id, cam_map in midline_set.items():
            n_cams = len(cam_map)
            camera_distribution[n_cams] += 1
            total_observations += 1
            if n_cams == 1:
                singleton_count += 1

    singleton_rate = singleton_count / max(total_observations, 1)
    # fish yield ratio: mean fish observed per frame / n_animals
    frames = len(midline_sets)
    fish_per_frame = total_observations / max(frames, 1)
    yield_ratio = fish_per_frame / max(n_animals, 1)

    return AssociationMetrics(
        fish_yield_ratio=yield_ratio,
        singleton_rate=singleton_rate,
        camera_distribution=dict(camera_distribution),
        total_fish_observations=total_observations,
        frames_evaluated=frames,
    )
```

### Reconstruction Metrics (wrapping existing compute_tier1)
```python
# Source: evaluation/metrics.py compute_tier1 pattern
from aquapose.evaluation.metrics import compute_tier1, compute_tier2

def evaluate_reconstruction(
    frame_results: list[tuple[int, dict[int, Midline3D]]],
    fish_available: int = 0,
) -> ReconstructionMetrics:
    tier1 = compute_tier1(frame_results, fish_available=fish_available)
    low_confidence_count = sum(
        1
        for _fi, midline_dict in frame_results
        for midline3d in midline_dict.values()
        if midline3d.is_low_confidence
    )
    total_fish = sum(len(md) for _, md in frame_results)
    low_confidence_flag_rate = low_confidence_count / max(total_fish, 1)

    return ReconstructionMetrics(
        mean_reprojection_error=tier1.overall_mean_px,
        max_reprojection_error=tier1.overall_max_px,
        fish_reconstructed=tier1.fish_reconstructed,
        fish_available=tier1.fish_available,
        low_confidence_flag_rate=low_confidence_flag_rate,
        per_camera_error=tier1.per_camera,
        per_fish_error=tier1.per_fish,
    )
```

### Detection Jitter Metric
```python
# Detection jitter: mean abs frame-to-frame change in detection count per camera
# Per CONTEXT.md: camera 9→9→9→0→0→0 has low jitter (~1.5); 9→3→8→2→9→4 has high jitter (~5)
import numpy as np

def _compute_jitter(counts_per_camera: dict[str, list[int]]) -> float:
    """Mean absolute frame-to-frame delta in detection count, averaged over cameras."""
    if not counts_per_camera:
        return 0.0
    jitters = []
    for counts in counts_per_camera.values():
        arr = np.array(counts, dtype=float)
        if len(arr) < 2:
            jitters.append(0.0)
        else:
            jitters.append(float(np.mean(np.abs(np.diff(arr)))))
    return float(np.mean(jitters))
```

### Module-level __init__.py Pattern
```python
# Source: src/aquapose/evaluation/__init__.py pattern
"""Stage-level pure-function evaluators for all five pipeline stages."""

from aquapose.evaluation.stages.association import (
    AssociationMetrics,
    DEFAULT_GRID as ASSOCIATION_DEFAULT_GRID,
    evaluate_association,
)
from aquapose.evaluation.stages.detection import DetectionMetrics, evaluate_detection
from aquapose.evaluation.stages.midline import MidlineMetrics, evaluate_midline
from aquapose.evaluation.stages.reconstruction import (
    ReconstructionMetrics,
    DEFAULT_GRID as RECONSTRUCTION_DEFAULT_GRID,
    evaluate_reconstruction,
)
from aquapose.evaluation.stages.tracking import TrackingMetrics, evaluate_tracking

__all__ = [
    "AssociationMetrics",
    "ASSOCIATION_DEFAULT_GRID",
    "DetectionMetrics",
    "MidlineMetrics",
    "RECONSTRUCTION_DEFAULT_GRID",
    "ReconstructionMetrics",
    "TrackingMetrics",
    "evaluate_association",
    "evaluate_detection",
    "evaluate_midline",
    "evaluate_reconstruction",
    "evaluate_tracking",
]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Metric logic scattered across harness.py + scripts | Pure functions in evaluation/stages/ | Phase 47 | Evaluators become unit-testable and importable without engine |
| SWEEP_RANGES / SECONDARY_RANGES in scripts/tune_association.py | DEFAULT_GRID in evaluation/stages/association.py | Phase 47 | Grid is now co-located with the evaluator that interprets it; scripts become thin wrappers |
| Tier1Result/Tier2Result as public reconstruction metrics API | ReconstructionMetrics (new) wrapping the same computations | Phase 47 (legacy types removed Phase 50) | Cleaner interface; legacy types remain untouched until Phase 50 |

## Open Questions

1. **Tier 2 in evaluate_reconstruction() — models required?**
   - What we know: Tier 2 (leave-one-out) requires running DltBackend.reconstruct_frame() with one camera dropped, which needs projection models. Midline3D already stores `mean_residual`, `max_residual`, `per_camera_residuals`, `is_low_confidence` so Tier 1 metrics are available without models.
   - What's unclear: Does Phase 47 include Tier 2 in evaluate_reconstruction(), or just Tier 1 + is_low_confidence metrics? Tier 2 requires `CalibBundle` or projection models as an additional parameter.
   - Recommendation: Include Tier 2 as an optional parameter with a `tier2_data` pass-through (caller can supply pre-computed Tier2Result, or None to skip). This avoids requiring DltBackend in the evaluator while keeping the interface clean. Alternatively, accept `models: dict[str, RefractiveProjectionModel] | None` as an optional param and compute Tier 2 when provided. The reconstruction `DEFAULT_GRID` suggests this stage is tunable, implying Tier 2 must be computable. The planner should decide which approach to lock in.

2. **Midline temporal smoothness — what constitutes a "frame sequence"?**
   - What we know: `Midline2D.point_confidence` provides per-point confidence; `Midline2D.points` is shape (N, 2). Temporal smoothness measures how much midlines change frame-over-frame for the same fish.
   - What's unclear: The input to `evaluate_midline()` must be ordered by (fish_id, frame_index) to compute temporal diffs. The exact input shape needs deciding: `list[dict[int, Midline2D]]` (frame → fish → midline) or `dict[int, list[Midline2D]]` (fish → frames).
   - Recommendation: Use `list[dict[int, Midline2D]]` (list of per-frame dicts, same shape as other stage outputs) so the evaluator interface is consistent. Temporal smoothness computed as mean L2 distance between consecutive frame midline centroid positions per fish.

3. **Detection evaluator input shape**
   - What we know: Detection stage produces `dict[str, list[Detection]]` per frame (camera_id → detections). Stage 46 pickle caches one dict per frame.
   - What's unclear: The exact input type for `evaluate_detection()` — `list[dict[str, list[Detection]]]` (ordered list of per-frame dicts) is the cleanest match to the pickle cache deserialization.
   - Recommendation: `evaluate_detection(frames: list[dict[str, list[Detection]]]) -> DetectionMetrics`. The frame ordering in the list provides the sequence needed for jitter computation.

## Sources

### Primary (HIGH confidence)
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/metrics.py` — frozen dataclass + pure function pattern; compute_tier1/compute_tier2 signatures
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/harness.py` — how reconstruction metrics are computed from MidlineSet; DltBackend usage for Tier 2
- `/home/tlancaster6/Projects/AquaPose/scripts/tune_association.py` — SWEEP_RANGES, SECONDARY_RANGES, _compute_association_metrics() verbatim source for DEFAULT_GRID and association metric formulas
- `/home/tlancaster6/Projects/AquaPose/scripts/tune_threshold.py` — threshold range 10–100 step 5 for reconstruction DEFAULT_GRID
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/types/detection.py` — Detection type with confidence field
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/types/midline.py` — Midline2D type with point_confidence field
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/types/reconstruction.py` — Midline3D type with mean_residual, max_residual, is_low_confidence, per_camera_residuals
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/tracking/types.py` — Tracklet2D with frames, frame_status, centroids fields
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/association/types.py` — TrackletGroup type
- `/home/tlancaster6/Projects/AquaPose/tests/unit/evaluation/test_metrics.py` — test pattern for evaluation unit tests; `_make_midline3d()` synthetic fixture pattern

### Secondary (MEDIUM confidence)
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/__init__.py` — current export list; must be updated to include new stage symbols

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are existing project deps, no new additions
- Architecture: HIGH — directly based on existing evaluation/metrics.py patterns and locked CONTEXT.md decisions
- Pitfalls: HIGH — all identified from direct code inspection of existing harness/evaluator code
- Open questions: MEDIUM — three design decisions left to planner (Tier 2 in reconstruction evaluator, midline input shape, detection input shape)

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (stable — no external dependencies changing)
