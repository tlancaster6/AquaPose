# Phase 41: Evaluation Harness - Research

**Researched:** 2026-03-02
**Domain:** Offline evaluation framework — fixture loading, metric computation, tabular reporting
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Frame Selection**
- Uniform temporal sampling via np.linspace (deterministic, no randomness)
- Same fixture always evaluates the same frames — enables regression comparison
- Default 15 frames, configurable via parameter
- If fixture has fewer frames than requested, evaluate all available with a warning (no error)
- No quality filtering — partial camera coverage per fish is expected and informative

**Metric Computation**
- Tier 1 reprojection error: call existing reconstruction backend's reconstruct_frame() directly, then reproject via RefractiveProjectionModel. Measures the actual backend being evaluated.
- Tier 2 leave-one-out: drop each observing camera in turn (not all 12), re-triangulate, measure max control-point displacement in world metres (Euclidean 3D distance)
- Leave-one-out runs that fail reconstruction (too few cameras after dropout) reported as N/A — useful signal about camera redundancy

**Output Format**
- Human-readable summary: per-camera breakdown as primary axis, per-fish within each camera. Overall aggregates at bottom.
- Machine-diffable regression data: JSON format, aggregated metrics only (no per-frame detail). Per-fish and per-camera aggregates (mean, max).
- Results saved next to fixture file (e.g., fixture_dir/eval_results.json)

**Harness Architecture**
- New `src/aquapose/evaluation/` package — dedicated module for harness, metrics, frame selection
- Python API only (no CLI command this phase)
- Calibration parameters bundled in fixture (intrinsics + extrinsics per camera) — reconstruction backends use RefractiveProjectionModel on-the-fly, no LUTs needed
- Phase 41's first task: extend Phase 40 fixture format to include calibration data

### Claude's Discretion
- Internal module structure within evaluation/ package
- Exact summary table formatting
- JSON schema details for regression data
- Test structure for the harness itself

### Deferred Ideas (OUT OF SCOPE)
- CLI command (`aquapose evaluate`) — add when Python API is stable
- Tier 3 synthetic ground-truth evaluation — tracked as EVAL-T3-01/T3-02 in requirements
- Per-frame detailed output — may add later if aggregated metrics prove insufficient for debugging
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | Evaluation harness loads MidlineSet fixtures + calibration data and runs metrics without the full pipeline | Fixture extension (calib bundling) + harness.run_evaluation() calling reconstruct_frame() directly |
| EVAL-02 | Frame selection produces 15-20 frames from ~300 frame window via uniform temporal sampling | np.linspace on fixture.frame_indices, clamp to available, warn when fewer than requested |
| EVAL-03 | Tier 1 metric: per-fish, per-camera reprojection error (mean and max) with overall aggregates | triangulate_midlines() already returns per_camera_residuals and mean/max_residual on Midline3D; harness aggregates across selected frames |
| EVAL-04 | Tier 2 metric: leave-one-out camera stability (max control-point displacement across dropout runs) | Re-run reconstruct_frame() with one camera removed from each MidlineSet, compute max Euclidean distance between baseline and dropout control_points (shape 7×3) |
| EVAL-05 | Evaluation outputs human-readable summary table and machine-diffable regression data | tabulate or manual table formatting + json.dump to eval_results.json next to fixture |
</phase_requirements>

---

## Summary

Phase 41 builds an offline evaluation framework that is self-contained relative to the live pipeline. The primary design constraint is that the harness must work without video files or live cameras — it reads a MidlineSet fixture (NPZ) plus bundled calibration, selects frames, and feeds them directly into the existing `TriangulationBackend.reconstruct_frame()` / `triangulate_midlines()` machinery.

The first task in this phase extends the Phase 40 NPZ fixture format to bundle calibration parameters (K_new per camera, R, t, water_z, interface_normal, n_air, n_water). This makes fixtures fully self-contained: no external calibration file needed for evaluation. The calibration parameters are the same scalars/matrices already loaded by `TriangulationBackend._load_models()`; bundling them means the harness reconstructs `RefractiveProjectionModel` objects directly from the NPZ rather than from a separate JSON file.

The metric computation reuses existing code heavily. Tier 1 (reprojection error) is already computed inside `triangulate_midlines()` and surfaces on `Midline3D.mean_residual`, `max_residual`, and `per_camera_residuals`. Tier 2 (leave-one-out stability) requires running `reconstruct_frame()` N times (once per observing camera per fish per frame) and measuring max Euclidean distance between baseline and dropout control points. The output module formats results into a human-readable ASCII table and a JSON regression file.

**Primary recommendation:** Implement in three focused sub-tasks: (1) extend fixture format with calibration, (2) implement harness core + metrics, (3) implement output formatters + JSON serialization. Keep evaluation/ as a flat package (3-4 modules) without deep nesting.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | project-standard | Frame selection (np.linspace), array math, NPZ I/O | Already in use throughout |
| scipy | project-standard | BSpline evaluation for control-point displacement | Already used in triangulation.py |
| torch | project-standard | RefractiveProjectionModel reconstruction (CPU tensors) | Already in use; models always CPU for evaluation |
| json | stdlib | Machine-diffable regression output | stdlib, no dep, human-readable diffs |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| warnings | stdlib | Emit warning when fixture has fewer frames than requested | Warn, not raise |
| pathlib.Path | stdlib | Fixture path handling, output path next to fixture | Consistent with rest of codebase |
| dataclasses | stdlib | EvalResults and metric container types | Consistent with Midline3D, MidlineFixture patterns |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| json stdlib | pandas / rich | json is machine-diffable and has zero deps; pandas is overkill for scalar aggregates |
| Manual ASCII table | tabulate | tabulate is not in project deps; manual formatting is trivially sufficient for a fixed schema |
| Calling reconstruct_frame() | Re-implementing triangulation | Never re-implement — reconstruct_frame() IS the backend being tested |

---

## Architecture Patterns

### Recommended Package Structure
```
src/aquapose/evaluation/
├── __init__.py          # public API: run_evaluation, EvalResults
├── harness.py           # run_evaluation() — orchestrates frame selection, metrics
├── metrics.py           # compute_tier1(), compute_tier2() — pure metric logic
└── output.py            # format_summary_table(), write_regression_json()

tests/unit/io/
└── test_midline_fixture.py   # extend with calibration bundle tests (existing file)

tests/unit/evaluation/
├── __init__.py
├── test_harness.py           # run_evaluation() unit tests with synthetic data
├── test_metrics.py           # tier1/tier2 metric computation unit tests
└── test_output.py            # summary table + JSON output unit tests
```

### Pattern 1: Fixture Format Extension (NPZ calibration bundle)
**What:** Add per-camera calibration arrays to the existing NPZ format under a `calib/` key group. The loader reads them back and reconstructs `RefractiveProjectionModel` instances.

**New NPZ key convention:**
```
calib/water_z                        — scalar float32
calib/n_air                          — scalar float32
calib/n_water                        — scalar float32
calib/interface_normal               — shape (3,), float32
calib/{camera_id}/K_new              — shape (3, 3), float32
calib/{camera_id}/R                  — shape (3, 3), float32
calib/{camera_id}/t                  — shape (3,), float32
```

**Loader change:** `load_midline_fixture()` optionally reads these keys and returns `CalibBundle` alongside the existing `MidlineFixture`. Alternatively, a new `load_midline_fixture_with_calib()` function keeps the existing loader untouched.

**Export change:** `DiagnosticObserver.export_midline_fixtures()` receives the `CalibrationData` (or pre-built models dict) and writes calib keys. The `TriangulationBackend` already holds `self._models`; the pipeline wires the calibration data through.

**Design decision (discretion):** Two clean options:
1. Extend `MidlineFixture` with an optional `calib_bundle` field (None when absent) — backward compatible, loader handles both old and new NPZ.
2. Separate `load_calibration_from_fixture(path)` function — keeps concerns separate.

Option 1 is simpler for the harness (single object to pass around). Recommended.

**Version bump:** Increment NPZ_VERSION to "2.0" when calibration keys are added. Loader must handle "1.0" (no calib) and "2.0" (with calib).

### Pattern 2: Frame Selection
**What:** Given `fixture.frame_indices` (tuple of ints), select `n_frames` indices via uniform temporal sampling.

```python
def select_frames(frame_indices: tuple[int, ...], n_frames: int = 15) -> list[int]:
    """Select n_frames frame indices via uniform temporal sampling."""
    available = list(frame_indices)
    if len(available) <= n_frames:
        if len(available) < n_frames:
            warnings.warn(
                f"Fixture has {len(available)} frames, fewer than requested {n_frames}. "
                "Evaluating all available frames.",
                stacklevel=2,
            )
        return available
    # np.linspace on indices into available, then map back to frame values
    positions = np.linspace(0, len(available) - 1, n_frames, dtype=int)
    return [available[int(p)] for p in positions]
```

**Key detail:** `np.linspace(0, len-1, n, dtype=int)` is deterministic and endpoint-inclusive — selects first, last, and evenly-spaced interior frames. This guarantees the same frames are always selected from the same fixture (regression stability).

### Pattern 3: Tier 1 Metric (Reprojection Error)
**What:** For each selected frame, call `backend.reconstruct_frame(frame_idx, midline_set)` and collect `Midline3D.mean_residual`, `max_residual`, `per_camera_residuals` directly from the return value. Aggregate across frames.

```python
# Pseudocode
for frame_idx in selected_frame_indices:
    midline_set = fixture.frames[frame_to_pos[frame_idx]]
    results: dict[int, Midline3D] = backend.reconstruct_frame(frame_idx, midline_set)
    for fish_id, midline3d in results.items():
        # Collect per-camera residuals
        for cam_id, cam_err in (midline3d.per_camera_residuals or {}).items():
            tier1_cam_errors[cam_id][fish_id].append(cam_err)
        # Collect overall
        tier1_overall_errors[fish_id].append(midline3d.mean_residual)
```

**Aggregate output structure:**
- Per-camera, per-fish: mean and max over selected frames
- Per-camera overall: mean and max across all fish in that camera
- Global: mean and max across everything

**Note:** `Midline3D.per_camera_residuals` is `dict[str, float] | None`. The harness must handle `None` gracefully (fish where residuals were not computed). In practice, `triangulate_midlines()` always sets `per_camera_residuals` when it returns a result (confirmed in source).

### Pattern 4: Tier 2 Metric (Leave-One-Out Camera Stability)
**What:** For each selected frame and fish, identify the observing cameras (cameras present in `midline_set[fish_id]`), then for each observing camera, run reconstruction with that camera removed and measure max control-point displacement vs the baseline.

```python
# Pseudocode
for frame_idx in selected_frame_indices:
    midline_set = fixture.frames[frame_to_pos[frame_idx]]
    for fish_id, cam_map in midline_set.items():
        # Baseline reconstruction
        baseline = backend.reconstruct_frame(frame_idx, midline_set).get(fish_id)
        if baseline is None:
            continue  # Fish not reconstructable even with all cameras
        baseline_ctrl = baseline.control_points  # shape (7, 3)

        for dropout_cam in cam_map.keys():
            # Remove one camera
            reduced_midline_set = {
                fid: {c: m for c, m in cams.items() if c != dropout_cam}
                for fid, cams in midline_set.items()
            }
            dropout_result = backend.reconstruct_frame(frame_idx, reduced_midline_set)
            dropout = dropout_result.get(fish_id)
            if dropout is None:
                tier2_results[fish_id][dropout_cam].append(None)  # N/A
                continue
            # Max Euclidean distance between control points
            diffs = np.linalg.norm(
                dropout.control_points - baseline_ctrl, axis=1
            )  # shape (7,)
            max_displacement = float(np.max(diffs))
            tier2_results[fish_id][dropout_cam].append(max_displacement)
```

**Aggregate:** Max displacement across all frames per (fish, dropout_camera) pair.

**N/A handling:** When dropout reconstruction returns no fish (too few cameras), record `None`. In JSON output, serialise as `null`. In summary table, display as `N/A`.

**Performance note:** With 15 frames × 2 fish × 6 cameras = 180 reconstruction calls. Each `reconstruct_frame()` takes ~10-50ms (CPU, no model loading). Total ~2-10 seconds. Acceptable for an offline harness.

### Pattern 5: Harness Constructor — Backend Instantiation from Fixture
**What:** The harness constructs a `TriangulationBackend`-equivalent using calibration data from the fixture (not from a calibration JSON file).

Since `TriangulationBackend.__init__` takes a `calibration_path: str | Path`, the harness cannot directly pass a `CalibBundle`. Two clean options:

**Option A (recommended):** Harness calls `triangulate_midlines()` directly with models built from fixture calib data. Avoids needing a calibration file entirely.

```python
# In harness.py
def _build_models_from_fixture(fixture: MidlineFixture) -> dict[str, RefractiveProjectionModel]:
    """Build per-camera projection models from bundled calibration data."""
    calib = fixture.calib_bundle  # CalibBundle or similar
    models = {}
    for cam_id in calib.camera_ids:
        models[cam_id] = RefractiveProjectionModel(
            K=torch.from_numpy(calib.K_new[cam_id]),
            R=torch.from_numpy(calib.R[cam_id]),
            t=torch.from_numpy(calib.t[cam_id]),
            water_z=calib.water_z,
            normal=torch.from_numpy(calib.interface_normal),
            n_air=calib.n_air,
            n_water=calib.n_water,
        )
    return models
```

**Option B:** Accept an optional `calibration_path` alongside the fixture (fallback if fixture has no calib). Keeps `TriangulationBackend` unchanged.

Option A is cleaner for the self-contained fixture requirement (EVAL-01). Use Option A.

### Pattern 6: Output Formats

**Human-readable summary table** (stdout + optional file):
```
Evaluation Summary
==================
Fixture: midline_fixtures.npz  |  Frames evaluated: 15 / 300

Tier 1: Reprojection Error (pixels)
-------------------------------------
Camera         Fish   Mean     Max
-----------    ----   ------   ------
cam_abc        0      3.21     8.45
               1      4.10     11.23
               ALL    3.65     11.23
cam_def        0      2.87     7.12
               ...
OVERALL               3.44     11.23

Tier 2: Leave-One-Out Stability (mm world displacement)
---------------------------------------------------------
Fish   Dropout Camera   Max Displacement
----   --------------   ----------------
0      cam_abc          1.23
0      cam_def          N/A
...
```

**JSON regression file** (`eval_results.json`):
```json
{
  "fixture": "midline_fixtures.npz",
  "frames_evaluated": 15,
  "frames_available": 300,
  "tier1": {
    "overall_mean_px": 3.44,
    "overall_max_px": 11.23,
    "per_camera": {
      "cam_abc": {"mean_px": 3.21, "max_px": 8.45},
      ...
    },
    "per_fish": {
      "0": {"mean_px": 3.21, "max_px": 8.45},
      ...
    }
  },
  "tier2": {
    "per_fish_dropout": {
      "0": {
        "cam_abc": {"max_displacement_m": 0.00123},
        "cam_def": null
      }
    }
  }
}
```

### Anti-Patterns to Avoid
- **Re-implementing triangulation:** Never rewrite triangulation math in the harness. Call `triangulate_midlines()` or `backend.reconstruct_frame()` directly.
- **Holding all frame results in memory:** The harness processes selected frames sequentially, accumulating only the aggregated metric values. No per-frame result retention needed.
- **Importing TriangulationBackend for evaluation:** The backend requires a calibration file path. Use the lower-level `triangulate_midlines()` + models-from-fixture pattern instead.
- **Mutating the fixture object:** Never modify `fixture.frames` when building reduced midline sets for leave-one-out. Always construct new dicts.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Uniform frame sampling | Custom sampling logic | `np.linspace(0, n-1, k, dtype=int)` | Deterministic, endpoint-inclusive, numpy handles edge cases |
| 3D control point distance | Custom Euclidean distance | `np.linalg.norm(pts_a - pts_b, axis=1)` | Already used in triangulation.py |
| B-spline evaluation for Tier 2 | Custom spline evaluator | `scipy.interpolate.BSpline(knots, ctrl_pts, k)` | Already used in triangulation.py; same pattern |
| JSON serialization with nulls | Custom serializer | `json.dump` with `default=lambda x: None` | stdlib handles None → null correctly |
| Triangulation | Custom DLT implementation | `triangulate_midlines()` from triangulation.py | This is exactly what Tier 1 and Tier 2 measure |

**Key insight:** The evaluation harness is a thin orchestration layer. All the hard math (triangulation, reprojection, spline fitting) is already in existing modules. The harness selects frames, feeds them to existing code, and collects results.

---

## Common Pitfalls

### Pitfall 1: Calibration Data Not Reaching the Harness
**What goes wrong:** The harness loads the fixture but has no calibration data because the fixture was created before Phase 41's format extension.
**Why it happens:** Phase 40 NPZ files (version "1.0") have no `calib/` keys. Phase 41 requires "2.0" fixtures.
**How to avoid:** Version-check in `load_midline_fixture`: if `meta/version == "1.0"`, set `calib_bundle = None` and the harness raises a clear error: "Fixture version 1.0 does not include calibration data. Re-run the pipeline with DiagnosticObserver to generate a version 2.0 fixture."
**Warning signs:** `KeyError` on `calib/water_z` during fixture loading.

### Pitfall 2: Backend Reconstruction Returns Empty on Leave-One-Out
**What goes wrong:** When dropping a camera leaves fewer than 2 observing cameras for a fish's body points, `triangulate_midlines()` returns an empty dict (fish skipped). Harness must record this as N/A, not as zero displacement.
**Why it happens:** `triangulate_midlines()` skips fish with fewer than `min_body_points` valid triangulated points. With only 1 camera, all body points fail triangulation.
**How to avoid:** Check `dropout_result.get(fish_id)` for None before accessing control points. Record `None` / display "N/A".
**Warning signs:** KeyError accessing `dropout_result[fish_id]` when fish has very few cameras.

### Pitfall 3: Frame Index vs Position Index Confusion
**What goes wrong:** `fixture.frames` is indexed by position (0, 1, 2...), while frame indices are the original video frame numbers (e.g., 10, 25, 40...). Passing the original frame index as a list index causes IndexError.
**Why it happens:** `fixture.frame_indices = (10, 25, 40)` means `fixture.frames[0]` is frame 10. Code that does `fixture.frames[10]` will fail or return wrong data.
**How to avoid:** Build a lookup dict: `frame_to_pos = {fi: pos for pos, fi in enumerate(fixture.frame_indices)}`. Always look up position via `frame_to_pos[frame_idx]`.
**Warning signs:** IndexError accessing `fixture.frames[large_number]`.

### Pitfall 4: CUDA Tensors from RefractiveProjectionModel
**What goes wrong:** `RefractiveProjectionModel.project()` returns PyTorch tensors. If any models were moved to CUDA, `.cpu().numpy()` is required before numpy operations.
**Why it happens:** Models built from fixture data default to CPU. But if code inadvertently calls `.to("cuda")`, downstream numpy operations break.
**How to avoid:** Always call `.cpu().numpy()` on tensor results, never bare `.numpy()`. This is stated in CLAUDE.md as a project-wide pitfall.
**Warning signs:** `RuntimeError: can't convert CUDA tensor to numpy`.

### Pitfall 5: per_camera_residuals None When Fish Has Very Few Cameras
**What goes wrong:** `Midline3D.per_camera_residuals` can be None if triangulation produced a result but the residual computation path returned early.
**Why it happens:** In `triangulate_midlines()`, `cam_residuals` is always populated (see source), but the field is typed as `dict[str, float] | None`. Future refactors or edge cases could leave it None.
**How to avoid:** Always guard: `for cam_id, err in (midline3d.per_camera_residuals or {}).items()`.

### Pitfall 6: JSON Non-Serializable Types
**What goes wrong:** `json.dump` raises `TypeError` on numpy float32/int64 values or numpy arrays.
**Why it happens:** Metric aggregation produces numpy scalars (e.g., `np.float32(3.44)`) which are not JSON-serializable by default.
**How to avoid:** Convert all values to Python builtins before serializing: `float(val)`, `int(val)`. Or use a custom `default` function that handles numpy scalars.

---

## Code Examples

### Frame Selection
```python
# Source: project decision (np.linspace pattern)
import warnings
import numpy as np

def select_frames(frame_indices: tuple[int, ...], n_frames: int = 15) -> list[int]:
    """Select up to n_frames from available frame_indices via uniform temporal sampling."""
    available = list(frame_indices)
    if not available:
        return []
    if len(available) <= n_frames:
        if len(available) < n_frames:
            warnings.warn(
                f"Fixture has {len(available)} frames, fewer than requested {n_frames}. "
                "Evaluating all available frames.",
                stacklevel=2,
            )
        return available
    positions = np.linspace(0, len(available) - 1, n_frames, dtype=int)
    return [available[int(p)] for p in positions]
```

### Building RefractiveProjectionModel from Bundled Calibration
```python
# Source: mirrors TriangulationBackend._load_models() pattern
import torch
from aquapose.calibration.projection import RefractiveProjectionModel

def build_models_from_calib(calib_bundle) -> dict[str, RefractiveProjectionModel]:
    """Reconstruct per-camera models from fixture-bundled calibration arrays."""
    models: dict[str, RefractiveProjectionModel] = {}
    normal = torch.from_numpy(calib_bundle.interface_normal)
    for cam_id in calib_bundle.camera_ids:
        models[cam_id] = RefractiveProjectionModel(
            K=torch.from_numpy(calib_bundle.K_new[cam_id]),
            R=torch.from_numpy(calib_bundle.R[cam_id]),
            t=torch.from_numpy(calib_bundle.t[cam_id]),
            water_z=calib_bundle.water_z,
            normal=normal,
            n_air=calib_bundle.n_air,
            n_water=calib_bundle.n_water,
        )
    return models
```

### Tier 2 Control-Point Displacement
```python
# Source: project pattern (np.linalg.norm already used in triangulation.py)
import numpy as np

def max_control_point_displacement(
    baseline_ctrl: np.ndarray,    # shape (7, 3)
    dropout_ctrl: np.ndarray,     # shape (7, 3)
) -> float:
    """Max Euclidean distance between baseline and dropout control points."""
    diffs = np.linalg.norm(dropout_ctrl - baseline_ctrl, axis=1)  # (7,)
    return float(np.max(diffs))
```

### Calibration NPZ Serialization (writer side)
```python
# Source: mirrors existing export_midline_fixtures() pattern in diagnostic_observer.py
def _add_calib_to_npz(
    npz_arrays: dict,
    models: dict,  # dict[str, RefractiveProjectionModel]
    water_z: float,
    interface_normal,  # torch.Tensor (3,)
    n_air: float,
    n_water: float,
) -> None:
    """Write calibration arrays into npz_arrays dict in-place."""
    npz_arrays["calib/water_z"] = np.array(water_z, dtype=np.float32)
    npz_arrays["calib/n_air"] = np.array(n_air, dtype=np.float32)
    npz_arrays["calib/n_water"] = np.array(n_water, dtype=np.float32)
    npz_arrays["calib/interface_normal"] = interface_normal.cpu().numpy().astype(np.float32)
    for cam_id, model in models.items():
        prefix = f"calib/{cam_id}"
        npz_arrays[f"{prefix}/K_new"] = model.K.cpu().numpy().astype(np.float32)
        npz_arrays[f"{prefix}/R"] = model.R.cpu().numpy().astype(np.float32)
        npz_arrays[f"{prefix}/t"] = model.t.cpu().numpy().astype(np.float32)
```

### JSON-safe conversion
```python
# Source: stdlib json pattern
import json
import numpy as np

class _NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return super().default(obj)

# Usage:
json.dump(results_dict, fp, cls=_NumpySafeEncoder, indent=2)
```

---

## Fixture Format Extension Detail

### What Needs to Change in Phase 40 Code

**`src/aquapose/io/midline_fixture.py`:**
- `NPZ_VERSION` bumped to `"2.0"`
- `MidlineFixture` gets an optional `calib_bundle: CalibBundle | None = None` field
- `load_midline_fixture()` reads `calib/` keys when present (version "2.0"), builds `CalibBundle`
- Backward compatibility: version "1.0" loads with `calib_bundle=None`

**`src/aquapose/engine/diagnostic_observer.py`:**
- `export_midline_fixtures()` accepts optional `models: dict[str, RefractiveProjectionModel] | None = None` and calibration scalars
- When `models` is provided, writes `calib/` keys and uses version "2.0"

**`CalibBundle` dataclass** (new, lives in `midline_fixture.py` or `evaluation/`):
```python
@dataclass(frozen=True)
class CalibBundle:
    """Bundled calibration data for self-contained fixture evaluation."""
    camera_ids: tuple[str, ...]
    K_new: dict[str, np.ndarray]       # camera_id -> (3,3) float32
    R: dict[str, np.ndarray]           # camera_id -> (3,3) float32
    t: dict[str, np.ndarray]           # camera_id -> (3,) float32
    water_z: float
    interface_normal: np.ndarray       # (3,) float32
    n_air: float
    n_water: float
```

---

## Integration Points Summary

| Component | Role in Phase 41 | Location |
|-----------|-----------------|----------|
| `MidlineFixture` | Carries frames + calibration bundle | `src/aquapose/io/midline_fixture.py` |
| `load_midline_fixture()` | Deserializes fixture + calib | `src/aquapose/io/midline_fixture.py` |
| `export_midline_fixtures()` | Writes calib/ keys alongside midline/ keys | `src/aquapose/engine/diagnostic_observer.py` |
| `triangulate_midlines()` | Core reconstruction called by harness | `src/aquapose/core/reconstruction/triangulation.py` |
| `RefractiveProjectionModel` | Built from calib bundle, used for reconstruction | `src/aquapose/calibration/projection.py` |
| `Midline3D.per_camera_residuals` | Tier 1 source data | `src/aquapose/core/types/reconstruction.py` |
| `Midline3D.control_points` | Tier 2 displacement measurement | `src/aquapose/core/types/reconstruction.py` |

---

## Open Questions

1. **Where does `export_midline_fixtures()` get the calibration data?**
   - What we know: `DiagnosticObserver` doesn't currently receive calibration data. The `TriangulationBackend` holds `self._models`.
   - What's unclear: How calibration reaches the observer at export time. Options: (a) pass models dict to `export_midline_fixtures()` as argument, (b) have `DiagnosticObserver.__init__` accept optional models, (c) have the pipeline pass calib data through the context.
   - Recommendation: Option (a) — pass models dict as optional argument to `export_midline_fixtures()`. Minimal API change, explicit, no hidden state.

2. **Should `CalibBundle` live in `midline_fixture.py` or `evaluation/`?**
   - What we know: It's read by the fixture loader (io/) and consumed by the harness (evaluation/). Circular import risk if in evaluation/ and referenced from io/.
   - What's unclear: Whether it belongs conceptually to the fixture data contract or to evaluation.
   - Recommendation: Define `CalibBundle` in `midline_fixture.py` alongside `MidlineFixture`. It is part of the fixture data contract.

3. **Backward compatibility for version "1.0" fixtures?**
   - What we know: Existing test fixtures created in Phase 40 are version "1.0" with no calib keys.
   - What's unclear: Whether Phase 40 tests need updating to v2.0.
   - Recommendation: Loader handles both versions. Version "1.0" fixtures load with `calib_bundle=None`. Existing Phase 40 tests remain valid. Harness raises a clear error when `calib_bundle is None`.

---

## Sources

### Primary (HIGH confidence)
- Source inspection of `src/aquapose/core/reconstruction/triangulation.py` — confirmed `triangulate_midlines()` API, `Midline3D` fields, residual computation pattern
- Source inspection of `src/aquapose/io/midline_fixture.py` — confirmed NPZ key convention, `MidlineFixture` structure, loader pattern
- Source inspection of `src/aquapose/engine/diagnostic_observer.py` — confirmed `export_midline_fixtures()` API, NPZ write pattern
- Source inspection of `src/aquapose/calibration/projection.py` — confirmed `RefractiveProjectionModel` constructor args and `.project()` return types
- Source inspection of `src/aquapose/calibration/loader.py` — confirmed `CameraData`, `CalibrationData`, `compute_undistortion_maps()` pattern
- Source inspection of `src/aquapose/core/reconstruction/backends/triangulation.py` — confirmed `TriangulationBackend._load_models()` pattern for building models from calibration
- Source inspection of `tests/unit/io/test_midline_fixture.py` — confirmed test patterns for round-trip fixture tests
- Source inspection of `tests/unit/test_triangulation.py` — confirmed synthetic rig helper pattern for unit tests

### Secondary (MEDIUM confidence)
- CONTEXT.md decisions — locked design choices confirmed against source code compatibility

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are already project dependencies, confirmed from source
- Architecture: HIGH — patterns directly derived from existing code in triangulation.py, midline_fixture.py, diagnostic_observer.py
- Pitfalls: HIGH — derived from source inspection (per_camera_residuals typing, frame vs position indexing, CUDA warning from CLAUDE.md)

**Research date:** 2026-03-02
**Valid until:** 2026-04-01 (stable codebase, low churn expected)
