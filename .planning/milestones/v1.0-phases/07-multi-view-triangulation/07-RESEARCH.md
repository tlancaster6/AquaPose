# Phase 7: Multi-View Triangulation - Research

**Researched:** 2026-02-21
**Domain:** Multi-view 3D triangulation, B-spline fitting, refractive optics
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Triangulation strategy (per body point):**
- Exhaustive pairwise search for <=7 cameras: Enumerate all (C choose 2) camera pairs, triangulate each via SVD least-squares (`triangulate_rays`), score each candidate by **max reprojection error** across held-out cameras. Keep the candidate with the lowest max held-out error.
- Re-triangulate with inliers: After selecting the best pair, re-triangulate using all cameras whose reprojection error against the winning candidate is acceptable — more cameras = better accuracy.
- Residual rejection for >7 cameras: Triangulate using all cameras directly, drop any camera with reprojection error > median + 2σ, re-triangulate once with remaining cameras.
- No view-angle weighting: Camera axes are nearly orthogonal to fish body axes most of the time; view-angle weighting adds complexity without benefit.
- No RANSAC random sampling: Exhaustive search is deterministic and cheap at typical camera counts (4–6 = 6–15 pairs).

**Spline fitting:**
- Fixed 7 control points for all fish, all frames — consistent output shape, no adaptive complexity.
- Fixed smoothing parameter tuned once — predictable, reproducible.
- Width profile stored as discrete array of N half-width values at sample points — no separate width spline.
- Total arc length computed at fit time — every downstream consumer needs it. Other derived quantities (heading, curvature) computed by consumers as needed.

**LM refinement (RECON-05):**
- Stub interface only — define the function signature and data flow, implement as no-op pass-through (returns spline unchanged). Pipeline runs end-to-end without modification. Full LM implementation deferred.

**Quality signals & fallbacks:**
- 2-camera fish: Triangulate anyway using the single pair, flag as low-confidence for downstream filtering.
- Failed body points (only 1 camera at that arc-length position): Drop the point and let `make_lsq_spline` handle the gap — spline naturally interpolates through missing positions. If there are less than n_control_points+2 valid body points, skip and let the previous frame coast through.
- Summary stats only for quality: store mean + max residual per fish, not per-point arrays.
- Batched/vectorized processing across all fish in a frame — not per-fish sequential loops.

### Claude's Discretion

- Inlier threshold (pixel distance) for the re-triangulate-with-inliers step
- `make_lsq_spline` smoothing (achieved via knot placement — no explicit s parameter)
- Minimum number of successfully-triangulated body points required before attempting spline fit
- Internal batching strategy (how to vectorize across variable camera counts per fish)

### Deferred Ideas (OUT OF SCOPE)

- LM reprojection refinement (RECON-05 full implementation) — add only if baseline triangulation + spline quality is insufficient
- Temporal smoothing / Kalman filtering of control point trajectories — separate phase
- Epipolar-guided correspondence refinement (if arc-length correspondence proves too noisy on highly curved fish)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RECON-03 | System triangulates each of the N body positions across cameras via refractive ray intersection with per-point RANSAC and view-angle weighting to reject arc-length correspondence outliers | Exhaustive pairwise triangulation using `triangulate_rays` (already in `calibration/projection.py`) + reprojection scoring via `RefractiveProjectionModel.project` (already implemented). Verified: 2-camera triangulation error = 0mm on noise-free data; 5-camera with 20px outlier correctly routes best pair through 4 remaining clean cameras. Note: requirement mentions "RANSAC and view-angle weighting" but user locked decisions replace both with exhaustive pairwise + inlier re-triangulation (no weighting). |
| RECON-04 | System fits a cubic B-spline (5–8 control points) through the N triangulated 3D points, plus a 1D width-profile spline, producing a continuous 3D midline + tube model per fish per frame | `scipy.interpolate.make_lsq_spline` verified: produces exactly 7 control points from 15 3D input points using explicit knot vector. Width profile stored as discrete array (no spline). Arc length computed via numerical integration of evaluated spline. Minimum valid points for 7-ctrl spline: n_data >= 7 (verified empirically). |
| RECON-05 | *(Optional)* System refines 3D spline control points via Levenberg-Marquardt minimization of reprojection error | Stub only: define function signature, implement as identity pass-through. No implementation needed this phase. |
</phase_requirements>

---

## Summary

Phase 7 triangulates the N=15 arc-length-sampled 2D midline points produced by Phase 6 (one per camera per fish per frame) into N 3D body positions, then fits a cubic B-spline through them to produce a continuous 3D midline. All core building blocks exist in the codebase: `triangulate_rays` (SVD least-squares, in `calibration/projection.py`), `RefractiveProjectionModel.project` (Newton-Raphson refractive forward projection, same file), and scipy's spline fitting machinery. The phase is entirely new code — no existing reconstruction module for triangulation — but all dependencies are already declared in `pyproject.toml` and verified working.

The triangulation strategy is exhaustive pairwise search: for each body point, enumerate all camera pairs (C(n,2) pairs; 6–15 pairs for 4–6 cameras), triangulate each pair with `triangulate_rays`, score each candidate by maximum reprojection error across held-out cameras, keep the lowest-error candidate, then re-triangulate using all cameras within the inlier threshold. This is deterministic, cheap, and correctly handles arc-length correspondence outliers. Verified experimentally: a 5-camera rig with one camera carrying 20px arc-length error correctly routes the best triangulation through the 4 clean cameras.

The critical implementation decision is the spline fitting API. `scipy.interpolate.splprep` does NOT directly accept a control-point count; its smoothing parameter `s` determines control points indirectly and varies with data. `scipy.interpolate.make_lsq_spline` is the correct function for fixed 7 control points — it accepts an explicit knot vector giving exact control over output shape. Verified: `make_lsq_spline` with interior knots `[0.25, 0.5, 0.75]` produces exactly a (7, 3) control-point array from 15 input points. Minimum input data: 7 valid body points (n_data >= n_ctrl).

**Primary recommendation:** Implement as `src/aquapose/reconstruction/triangulation.py` (sibling to existing `midline.py`). Keep triangulation and spline fitting in separate private functions. Expose one public entry point: `triangulate_midlines(midline_set, models) -> dict[int, Midline3D]`.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `aquapose.calibration.projection.triangulate_rays` | (in-project) | SVD least-squares 3D point from N rays | Already implemented, tested, works with 2–N cameras |
| `aquapose.calibration.projection.RefractiveProjectionModel.project` | (in-project) | Refractive forward projection for reprojection scoring | Already implemented with Newton-Raphson, returns (pixels, valid) |
| `scipy.interpolate.make_lsq_spline` | scipy>=1.11 (in pyproject) | Fixed-control-point cubic B-spline fitting | Exact control over output shape (7 ctrl pts); `splprep` does NOT give this |
| `numpy` | >=1.24 (in pyproject) | Array operations, itertools.combinations for pair enumeration | Already project-wide dependency |
| `itertools.combinations` | stdlib | Enumerate camera pairs for exhaustive pairwise search | No extra dep; 6–21 pairs for 4–7 cameras |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scipy.interpolate.BSpline` | scipy>=1.11 | Evaluate fitted spline, compute derivative for arc length | Returned by `make_lsq_spline`; use `.derivative()` for tangent |
| `torch` | >=2.0 (in pyproject) | `triangulate_rays` and `cast_ray` operate on torch tensors | All camera/ray operations are torch; convert numpy↔torch at module boundary |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `make_lsq_spline` | `scipy.interpolate.splprep` | `splprep`'s `s=` parameter controls knots indirectly — empirically 4–7 ctrl pts at s=1e-4 to 1e-6 on typical fish data; NOT deterministic across frame/fish variation. `make_lsq_spline` gives exact 7 ctrl pts every time. |
| `make_lsq_spline` | manual cubic B-spline via `numpy.linalg.lstsq` | Would require building the full design matrix; `make_lsq_spline` does this with correct end conditions |
| Exhaustive pairwise | RANSAC with random sampling | RANSAC is non-deterministic; at 4–6 cameras exhaustive search visits all 6–15 pairs anyway, so RANSAC sampling offers no speedup and loses determinism |

**Installation:** No new dependencies required. All libraries are in `pyproject.toml`. scikit-image was added in Phase 6.

---

## Architecture Patterns

### Recommended Module Location

```
src/aquapose/
├── reconstruction/
│   ├── __init__.py          # add Midline3D, triangulate_midlines exports
│   ├── midline.py           # Phase 6: Midline2D, MidlineExtractor (existing)
│   └── triangulation.py     # Phase 7: Midline3D, triangulate_midlines
tests/
└── unit/
    └── reconstruction/
        ├── __init__.py
        └── test_triangulation.py
```

### Key Data Structures

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Midline3D:
    """Fitted 3D midline for one fish in one frame.

    Attributes:
        fish_id: Track ID from Phase 5.
        frame_index: Frame number.
        control_points: B-spline control points, shape (7, 3), float32.
            Column order is (x, y, z) in world metres.
        knots: B-spline knot vector, shape (11,), float32.
            Fixed: [0,0,0,0, 0.25, 0.5, 0.75, 1,1,1,1] for k=3, n_ctrl=7.
        degree: Spline degree (always 3 for cubic).
        arc_length: Total arc length in metres, float32.
        half_widths: Half-width at each of the N=15 arc-length sample positions,
            shape (N,), float32, in world metres (mean across inlier cameras).
        n_cameras: Number of camera views used for triangulation.
        mean_residual: Mean per-point reprojection residual in pixels, float32.
        max_residual: Max per-point reprojection residual in pixels, float32.
        is_low_confidence: True when only 2 cameras contributed.
    """
    fish_id: int
    frame_index: int
    control_points: np.ndarray   # (7, 3) float32
    knots: np.ndarray            # (11,) float32
    degree: int                  # always 3
    arc_length: float
    half_widths: np.ndarray      # (N,) float32
    n_cameras: int
    mean_residual: float
    max_residual: float
    is_low_confidence: bool = False
```

### Pattern 1: Per-Body-Point Triangulation (<=7 Cameras, Exhaustive Pairwise)

```python
# Source: verified in research (codebase + scipy + manual verification)
from itertools import combinations
import torch
import numpy as np
from aquapose.calibration.projection import triangulate_rays, RefractiveProjectionModel

def _triangulate_body_point(
    pixels: dict[str, torch.Tensor],   # camera_id -> (2,) pixel coord
    models: dict[str, RefractiveProjectionModel],
    inlier_threshold: float = 15.0,    # pixels; Claude's discretion
) -> tuple[torch.Tensor, list[str], float] | None:
    """Triangulate one body position from multiple camera observations.

    Returns (point_3d, inlier_camera_ids, max_residual) or None if
    fewer than 2 cameras are available.

    For 2-camera case: single pair, no held-out scoring, flagged low-confidence.
    For 3+ cameras: exhaustive pairwise search, score vs held-out, re-triangulate inliers.
    """
    cam_ids = list(pixels.keys())
    n_cams = len(cam_ids)

    if n_cams < 2:
        return None  # single-camera body point: dropped

    # Cast rays for all cameras (reuse across pair evaluations)
    origins: dict[str, torch.Tensor] = {}
    directions: dict[str, torch.Tensor] = {}
    for cam_id in cam_ids:
        o, d = models[cam_id].cast_ray(pixels[cam_id].unsqueeze(0))
        origins[cam_id] = o.squeeze(0)
        directions[cam_id] = d.squeeze(0)

    if n_cams == 2:
        # Single pair — no held-out cameras for scoring
        o = torch.stack([origins[cam_ids[0]], origins[cam_ids[1]]])
        d = torch.stack([directions[cam_ids[0]], directions[cam_ids[1]]])
        pt3d = triangulate_rays(o, d)
        return pt3d, cam_ids, 0.0  # max_residual=0.0 (no held-out cameras)

    # Exhaustive pairwise search
    best_pt = None
    best_max_err = float('inf')
    best_pair = None
    for i_id, j_id in combinations(cam_ids, 2):
        o = torch.stack([origins[i_id], origins[j_id]])
        d = torch.stack([directions[i_id], directions[j_id]])
        pt3d = triangulate_rays(o, d)

        # Score: max reprojection error over held-out cameras
        held_out = [k for k in cam_ids if k not in (i_id, j_id)]
        max_err = 0.0
        for k in held_out:
            pix_reproj, val = models[k].project(pt3d.unsqueeze(0))
            if val[0]:
                err = float(torch.norm(pix_reproj[0] - pixels[k]))
                max_err = max(max_err, err)

        if max_err < best_max_err:
            best_max_err = max_err
            best_pt = pt3d
            best_pair = (i_id, j_id)

    # Re-triangulate with inliers
    inlier_o, inlier_d, inlier_ids = [], [], []
    for cam_id in cam_ids:
        pix_reproj, val = models[cam_id].project(best_pt.unsqueeze(0))
        if val[0]:
            err = float(torch.norm(pix_reproj[0] - pixels[cam_id]))
            if err < inlier_threshold:
                inlier_o.append(origins[cam_id])
                inlier_d.append(directions[cam_id])
                inlier_ids.append(cam_id)

    if len(inlier_o) < 2:
        # Fallback: use best_pair result
        return best_pt, list(best_pair), best_max_err  # type: ignore[arg-type]

    final_pt = triangulate_rays(torch.stack(inlier_o), torch.stack(inlier_d))
    # Recompute max residual for quality reporting
    final_max_err = 0.0
    for cam_id in inlier_ids:
        pix_reproj, val = models[cam_id].project(final_pt.unsqueeze(0))
        if val[0]:
            err = float(torch.norm(pix_reproj[0] - pixels[cam_id]))
            final_max_err = max(final_max_err, err)

    return final_pt, inlier_ids, final_max_err
```

### Pattern 2: Spline Fitting with Fixed 7 Control Points

```python
# Source: verified in research (scipy.interpolate.make_lsq_spline in hatch env)
import numpy as np
from scipy.interpolate import make_lsq_spline

# Fixed knot vector for cubic spline with 7 control points
# n_ctrl=7, k=3 (cubic): n_interior = n_ctrl - k - 1 = 3
# Interior knots at 0.25, 0.5, 0.75
SPLINE_K = 3
SPLINE_N_CTRL = 7
_INTERIOR_KNOTS = np.array([0.25, 0.5, 0.75])
SPLINE_KNOTS = np.concatenate([
    np.zeros(SPLINE_K + 1),
    _INTERIOR_KNOTS,
    np.ones(SPLINE_K + 1),
])
# = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]  shape (11,)

def _fit_spline(
    u_param: np.ndarray,   # (M,) arc-length params in [0,1], sorted ascending
    pts_3d: np.ndarray,    # (M, 3) triangulated body positions
) -> tuple[np.ndarray, float] | None:
    """Fit cubic B-spline with 7 control points through M valid 3D body positions.

    Args:
        u_param: Normalized arc-length parameters, shape (M,), sorted [0..1].
        pts_3d: World-frame 3D points, shape (M, 3), float32.

    Returns:
        (control_points, arc_length) where control_points is (7, 3) float32,
        or None if M < SPLINE_N_CTRL (not enough data to fit).
    """
    if len(u_param) < SPLINE_N_CTRL:
        return None  # caller handles skip / coast logic

    try:
        spl = make_lsq_spline(u_param, pts_3d, SPLINE_KNOTS, k=SPLINE_K)
    except Exception:
        return None

    ctrl_pts = spl.c.astype(np.float32)  # (7, 3)

    # Arc length via numerical integration (1000-point uniform sample)
    u_fine = np.linspace(0.0, 1.0, 1000)
    xyz_fine = spl(u_fine)  # (1000, 3)
    arc_len = float(np.sum(np.sqrt(np.sum(np.diff(xyz_fine, axis=0) ** 2, axis=1))))

    return ctrl_pts, arc_len
```

### Pattern 3: Outlier Rejection for >7 Cameras

```python
# Source: project design decision (CONTEXT.md) + numpy verification
import numpy as np
import torch
from aquapose.calibration.projection import triangulate_rays, RefractiveProjectionModel

def _triangulate_body_point_many_cameras(
    pixels: dict[str, torch.Tensor],
    models: dict[str, RefractiveProjectionModel],
) -> tuple[torch.Tensor, list[str], float]:
    """Triangulate using all cameras, then drop residual outliers.

    For n_cameras > 7. Triangulate all, compute per-camera reprojection
    residuals, drop cameras with residual > median + 2*sigma, re-triangulate once.
    """
    cam_ids = list(pixels.keys())
    # Cast all rays
    all_o, all_d = [], []
    for cam_id in cam_ids:
        o, d = models[cam_id].cast_ray(pixels[cam_id].unsqueeze(0))
        all_o.append(o.squeeze(0))
        all_d.append(d.squeeze(0))

    pt3d = triangulate_rays(torch.stack(all_o), torch.stack(all_d))

    # Compute per-camera reprojection residuals
    residuals = []
    for cam_id in cam_ids:
        pix_r, val = models[cam_id].project(pt3d.unsqueeze(0))
        residuals.append(float(torch.norm(pix_r[0] - pixels[cam_id])) if val[0] else float('inf'))

    residuals_arr = np.array(residuals)
    median = np.median(residuals_arr)
    std = np.std(residuals_arr)
    threshold = median + 2.0 * std

    inlier_ids = [cam_id for cam_id, r in zip(cam_ids, residuals) if r <= threshold]

    if len(inlier_ids) < 2:
        return pt3d, cam_ids, float(np.max(residuals_arr))  # fall back to all

    # Re-triangulate with inliers
    inlier_o = torch.stack([all_o[cam_ids.index(c)] for c in inlier_ids])
    inlier_d = torch.stack([all_d[cam_ids.index(c)] for c in inlier_ids])
    final_pt = triangulate_rays(inlier_o, inlier_d)
    max_res = float(max(residuals[cam_ids.index(c)] for c in inlier_ids))
    return final_pt, inlier_ids, max_res
```

### Pattern 4: Public Entry Point

```python
# Source: design decision from CONTEXT.md + Phase 6 output interface
from aquapose.reconstruction.midline import Midline2D
from aquapose.calibration.projection import RefractiveProjectionModel

# Input type from Phase 6:
MidlineSet = dict[int, dict[str, Midline2D]]  # fish_id -> camera_id -> Midline2D

def triangulate_midlines(
    midline_set: MidlineSet,
    models: dict[str, RefractiveProjectionModel],
    frame_index: int = 0,
) -> dict[int, Midline3D]:
    """Triangulate 2D midlines into 3D splines for all fish in a frame.

    Args:
        midline_set: Phase 6 output. fish_id -> camera_id -> Midline2D.
            Only cameras with successful midline extraction are present.
        models: Per-camera refractive projection models.
        frame_index: Current frame index.

    Returns:
        Dict mapping fish_id -> Midline3D for fish with successful triangulation.
        Fish with insufficient cameras or body points are omitted.
    """
    ...
```

### Pattern 5: LM Stub (RECON-05)

```python
# Source: CONTEXT.md decision — stub only, no-op pass-through
def refine_midline_lm(
    midline_3d: Midline3D,
    midline_set: MidlineSet,
    models: dict[str, RefractiveProjectionModel],
) -> Midline3D:
    """Refine 3D spline via Levenberg-Marquardt reprojection minimization.

    STUB: Returns midline_3d unchanged. Full implementation deferred.

    Args:
        midline_3d: Initial 3D midline from triangulation + spline fitting.
        midline_set: 2D midline observations for this fish (all cameras).
        models: Per-camera refractive projection models.

    Returns:
        Refined Midline3D (currently identical to input).
    """
    return midline_3d
```

### Anti-Patterns to Avoid

- **Using `splprep` for fixed 7 control points:** `splprep` selects knots via smoothing parameter `s`; produces 4–15 control points depending on data curvature. On noisy real data, the count is unpredictable frame-to-frame. Use `make_lsq_spline` with explicit knot vector.
- **Passing `u_param = np.linspace(0, 1, 15)` when some body points are missing:** When M < 15 valid body points are present, the u_param values must correspond to the actual arc-length positions of the valid points (indices 0, 2, 7, ... for example), not a re-normalized [0,1] linspace. The correct u_param for point at index i is `i / (N-1)` where N=15.
- **Converting torch tensors to numpy inside the hot loop:** `triangulate_rays` and `project` are torch operations. Avoid repeated `.numpy()` / `torch.tensor()` conversion inside the per-body-point loop. Cast at the module boundary (pixel observations from Midline2D.points are numpy float32; cast once to torch at the top of each fish).
- **Scoring pairs using all cameras including the pair itself:** The held-out scoring must exclude the pair members. Including the pair in the score inflates confidence (a pair perfectly explains its own observations).
- **Not handling the 2-camera case (no held-out cameras):** When only 2 cameras see a fish body point, there are no held-out cameras. Skip held-out scoring; triangulate the single pair and set `is_low_confidence=True` on the Midline3D result.
- **Accumulating per-point residual arrays:** CONTEXT.md requires summary stats only (mean + max per fish). Do not store a (15,) residual array — accumulate into scalar mean and max during the body-point loop.
- **Calling `make_lsq_spline` with u_param values that don't bracket all interior knots:** The Schoenberg-Whitney condition requires at least one data point in each knot interval. With interior knots at [0.25, 0.5, 0.75], if all valid body points are clustered at one arc-length region (e.g., all in [0, 0.3]), the fit will fail. Check that u_param spans [0, 1] (i.e., both endpoints must be valid body points) before calling.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SVD least-squares triangulation | Custom DLT or linear system solver | `triangulate_rays` (in-project) | Already implemented, tested, handles 2–N rays with graceful degeneracy |
| Refractive forward projection | Pinhole approximation or iterative Newton | `RefractiveProjectionModel.project` (in-project) | 10-iteration Newton-Raphson already implemented and differentiable |
| Fixed-knot B-spline least squares | Build design matrix manually | `scipy.interpolate.make_lsq_spline` | Correct Schoenberg-Whitney enforcement, stable linear algebra, returns `BSpline` object with `.derivative()` for arc-length computation |
| Combination enumeration | Nested loops for pair selection | `itertools.combinations` | Correct, readable, zero overhead at 6–21 pairs |

**Key insight:** All geometric primitives are already in the codebase from Phases 1–3. Phase 7 is primarily *orchestration* of existing tools with a new data structure (`Midline3D`) and new control flow (per-body-point pairwise search, aggregation to spline).

---

## Common Pitfalls

### Pitfall 1: `splprep` vs `make_lsq_spline` for Fixed Control Points

**What goes wrong:** Using `splprep(pts, k=3, s=some_value)` hoping for exactly 7 control points. On clean synthetic data at s=1e-4, splprep produces 7 control points. On noisy data or different fish curvatures, it produces 4–15 control points — frame-to-frame inconsistency breaks the "consistent output shape" requirement.
**Why it happens:** `splprep` uses the FITPACK routines that adaptively choose knots to achieve the target residual norm `s`. The number of knots is a result, not an input.
**How to avoid:** Use `make_lsq_spline(u_param, pts_3d, SPLINE_KNOTS, k=3)` with a fixed pre-built knot vector. Control-point count is determined by the knot vector length: n_ctrl = len(SPLINE_KNOTS) - k - 1 = 11 - 3 - 1 = 7. Always exactly 7.
**Warning signs:** `len(tck[1][0])` varies across frames when using `splprep`.

### Pitfall 2: Arc-Length Parameter Misalignment When Body Points Are Missing

**What goes wrong:** Body point at normalized arc-length position 7/14 fails triangulation. Code uses `u_param = np.linspace(0, 1, n_valid)` (re-normalizing), then fits spline. The control points end up shifted along the body axis — point 7 is now mapped to where point 8 should be.
**Why it happens:** The arc-length positions from Phase 6 are fixed indices (i/14 for point i). When some are missing, the remaining points must keep their original u values, not be re-normalized to fill [0,1].
**How to avoid:** Build u_param as `np.array([i / (N-1) for i in valid_indices])` where N=15 and valid_indices are the indices of successfully-triangulated body points. Never re-normalize.
**Warning signs:** Spline endpoint matches arc-length extremes but intermediate points are shifted laterally from camera observations.

### Pitfall 3: Schoenberg-Whitney Violation at Spline Fit

**What goes wrong:** `make_lsq_spline` raises `ValueError: nc = 7 > m = N` when fewer than 7 valid body points exist, or raises silently-corrupt results when u_param doesn't bracket all interior knots.
**Why it happens:** The Schoenberg-Whitney condition requires n_data >= n_ctrl. Also, each knot interval [0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1] must contain at least one data point; if a region is empty (all body points in that arc-length region failed), the design matrix is rank-deficient.
**How to avoid:** (1) Check `n_valid >= SPLINE_N_CTRL + 2 = 9` before calling (per CONTEXT.md decision). (2) Verify u_param[0] == 0.0 and u_param[-1] == 1.0 — endpoints (head and tail) must always be present; if either is missing, skip and coast. (3) Wrap in `try/except` to catch degenerate cases.
**Warning signs:** `ValueError: nc = 7 > m = N` in logs.

### Pitfall 4: Torch/Numpy Dtype Mismatch at Boundary

**What goes wrong:** `RefractiveProjectionModel.project` receives float64 instead of float32, or returns tensors that are then passed into numpy as object arrays.
**Why it happens:** `Midline2D.points` is `np.ndarray` float32. `triangulate_rays` expects float32 tensors. Conversion at wrong location produces float64 or tensor shape mismatches.
**How to avoid:** At the start of processing each fish, convert all pixel coordinates: `pix_tensor = torch.from_numpy(midline.points[i]).float()  # (2,) float32`. The `triangulate_rays` and `project` APIs both use float32 throughout.
**Warning signs:** `RuntimeError: expected scalar type Float but found Double`.

### Pitfall 5: Width Profile in World Units vs Pixel Units

**What goes wrong:** Half-width values from `Midline2D.half_widths` are in full-frame pixel space. They must be converted to world metres for the `Midline3D.half_widths` field.
**Why it happens:** Phase 6 outputs widths in pixels after the crop-to-frame transform. Phase 7 (and downstream HDF5 output) expects widths in metres.
**How to avoid:** For each inlier camera and body point, compute the half-width contribution: cast a second ray from `pixel + width_pixel * perpendicular_direction` and triangulate with the body-point ray to get a world-space width estimate. Average across inlier cameras. This is the geometrically correct conversion for refractive cameras where pixel-to-metre scale varies with depth.
**Alternative (simpler, approximate):** Use the depth of the triangulated body point plus the camera's focal length to compute pixel-to-metre scale: `width_m = half_width_px * depth_m / focal_length_px`. Simpler but ignores refractive distortion.
**Warning signs:** Half-widths in `Midline3D` are 1400x too large (still in pixels) or vary implausibly.

### Pitfall 6: 2-Camera Body Point Not Flagged as Low-Confidence

**What goes wrong:** A fish seen in exactly 2 cameras still gets triangulated. The result has no held-out cameras for quality scoring. If downstream code filters on `is_low_confidence`, it needs the flag to be set.
**Why it happens:** The `_triangulate_body_point` function silently succeeds with 2 cameras. The calling code in `triangulate_midlines` must propagate the 2-camera flag to `Midline3D.is_low_confidence`.
**How to avoid:** Track the minimum `n_inlier_cameras` across all body points for a given fish. If it is ever 2 (or if the fish track's `n_cameras` is 2 from Phase 5), set `Midline3D.is_low_confidence = True`.
**Warning signs:** Fish with only 2 cameras pass downstream quality gates that assume multi-view triangulation quality.

---

## Code Examples

### Verified: splprep vs make_lsq_spline Control Point Count

```python
# Source: live verification in hatch env (research session 2026-02-21)
import numpy as np
from scipy.interpolate import make_lsq_spline, splprep

np.random.seed(0)
t_sample = np.linspace(0, 1, 15)
# Noisy 3D fish spine data
x = t_sample * 0.15 + np.random.normal(0, 0.002, 15)
y = 0.02 * np.sin(2*np.pi * t_sample) + np.random.normal(0, 0.002, 15)
z = 1.5 + 0.01 * np.cos(np.pi * t_sample) + np.random.normal(0, 0.002, 15)
pts_T = np.stack([x, y, z]).T  # (15, 3)

# splprep: unpredictable control point count
for s in [1e-2, 1e-3, 1e-4]:
    tck, u = splprep(pts_T.T, k=3, s=s)
    print(f"splprep s={s:.0e}: n_ctrl={len(tck[1][0])}")  # 4, 4, 7

# make_lsq_spline: always 7
KNOTS = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=float)
spl = make_lsq_spline(t_sample, pts_T, KNOTS, k=3)
print(f"make_lsq_spline: n_ctrl={spl.c.shape[0]}")  # always 7
```

### Verified: Arc Length from BSpline

```python
# Source: live verification in hatch env (research session 2026-02-21)
import numpy as np
from scipy.interpolate import make_lsq_spline

KNOTS = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=float)
spl = make_lsq_spline(t_sample, pts_T, KNOTS, k=3)  # pts_T shape (15, 3)

# Numerical arc length (1000-point integration)
u_fine = np.linspace(0.0, 1.0, 1000)
xyz_fine = spl(u_fine)  # (1000, 3)
arc_len = float(np.sum(np.sqrt(np.sum(np.diff(xyz_fine, axis=0) ** 2, axis=1))))
print(f"arc_length: {arc_len:.4f}m")  # ~0.1744m for noisy data around 0.15m fish

# Control points for storage
ctrl_pts = spl.c.astype(np.float32)  # (7, 3) float32
knots = spl.t.astype(np.float32)     # (11,) float32
```

### Verified: Minimum Valid Body Points for Spline

```python
# Source: live verification (make_lsq_spline with varying n_data)
# n_data = 7: OK (= n_ctrl)
# n_data = 6: ERROR: nc = 7 > m = 6
# CONTEXT.md decision: skip if < n_ctrl + 2 = 9 (conservative, ensures bracket coverage)
MIN_BODY_POINTS = 9  # = SPLINE_N_CTRL + 2
```

### Verified: Pairwise Exhaustive Search Scales

```python
# Source: itertools.combinations verification in research
from itertools import combinations
for n_cam in range(2, 8):
    n_pairs = len(list(combinations(range(n_cam), 2)))
    print(f"n_cam={n_cam}: {n_pairs} pairs")
# n_cam=2: 1 pair
# n_cam=3: 3 pairs
# n_cam=4: 6 pairs
# n_cam=5: 10 pairs
# n_cam=6: 15 pairs
# n_cam=7: 21 pairs
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PCA keypoints (3 keypoints, Phase 3) | Arc-length-sampled midlines (15 points, Phase 6) | Project pivot 2026-02-21 | More information per fish; correspondence is implicit via arc-length |
| Adam optimizer for spline refinement | Direct triangulation + spline fit | Project pivot 2026-02-21 | Orders of magnitude faster; ~1ms vs 30+ min/sec |
| `scipy.interpolate.splprep` | `scipy.interpolate.make_lsq_spline` | Research finding 2026-02-21 | `splprep`'s smoothing parameter does not control control-point count deterministically; `make_lsq_spline` gives exact 7 control points every time |

**`scipy.interpolate.make_lsq_spline` note:** Available since scipy 1.0.0 (stable API). The `BSpline` return type supports `.derivative()`, `.integrate()`, and direct call `spl(u)`. No breaking API changes in scipy>=1.11. Confidence: HIGH (verified in hatch env with scipy installed).

---

## Integration Notes

### Input Interface (what Phase 7 receives from Phase 6)

```python
# MidlineSet: Phase 6 output
MidlineSet = dict[int, dict[str, Midline2D]]
# fish_id -> camera_id -> Midline2D
# Midline2D.points: (15, 2) float32, full-frame pixels, head-to-tail
# Midline2D.half_widths: (15,) float32, full-frame pixels
# Midline2D.is_head_to_tail: True when orientation is established

# models: from Phase 1 (passed through pipeline)
models: dict[str, RefractiveProjectionModel]

# frame_index: from pipeline driver
frame_index: int
```

**Important:** `Midline2D.points[i]` corresponds to normalized arc-length position `i / 14`. This is the u_param for body point i. When body points are dropped, preserve the original u values.

### Output Interface (what Phase 7 delivers to Phase 8)

```python
# Dict of successfully-triangulated fish
result: dict[int, Midline3D]
# fish_id -> Midline3D

# Midline3D fields consumed by Phase 8 (HDF5 writer):
# .control_points: (7, 3) float32 — B-spline control points
# .knots: (11,) float32 — knot vector (fixed, but stored for completeness)
# .arc_length: float — total body length in metres
# .half_widths: (15,) float32 — width profile in metres
# .n_cameras: int — for quality filtering downstream
# .mean_residual: float — for quality filtering
# .max_residual: float — for quality filtering
# .is_low_confidence: bool — for downstream filtering
```

### Interaction with Phase 5 (FishTrack)

Phase 7 does not take `FishTrack` objects directly. It consumes the `MidlineSet` dict (keyed by `fish_id`) and the `models` dict. The `fish_id` in `MidlineSet` comes from Phase 6, which got it from Phase 5's `FishTrack.fish_id`. No direct Phase 5 import needed.

---

## Open Questions

1. **Inlier threshold value (pixel distance)**
   - What we know: CONTEXT.md marks this as Claude's discretion. Reprojection experiments show that 1cm XY error produces ~14px reprojection error at fx=1400. Arc-length correspondence errors are typically 3–20px for well-segmented fish.
   - Recommendation: Start with **15px** as the inlier threshold. This accepts 1cm 3D error (well within fish body radius ~1.5cm) while rejecting major arc-length correspondence failures (20px+). Make it a constructor parameter on the triangulator class for easy tuning.

2. **Width profile in world units**
   - What we know: Midline2D.half_widths are in full-frame pixels. Downstream needs metres. Conversion requires knowing pixel-to-metre scale at each body point depth.
   - Recommendation: Use the approximate formula `width_m = hw_px * depth_m / focal_px` where `depth_m` is the Z coordinate of the triangulated body point relative to the camera. This is the pinhole approximation (ignores refraction). Accurate enough for a width profile that is downstream-only (not used for triangulation itself). If accuracy matters, implement the two-ray method.

3. **Minimum body points threshold (n_ctrl + 2 = 9 vs n_ctrl = 7)**
   - What we know: `make_lsq_spline` requires n_data >= n_ctrl = 7. CONTEXT.md says "if less than n_control_points+2". The "+2" provides margin for the Schoenberg-Whitney bracket coverage.
   - Recommendation: Use **9** as the hard minimum (= n_ctrl + 2). This ensures at least 2 data points in each of the 4 knot intervals on average, reducing ill-conditioned spline fits from clustered data.

4. **Handling coasting frames (fish with no successful triangulation)**
   - What we know: CONTEXT.md says "let the previous frame coast through" when there are insufficient valid body points.
   - What's unclear: Where does the "previous Midline3D" state live? The triangulator is stateless per-frame.
   - Recommendation: The triangulator should return `None` for failed fish (not a Midline3D). The pipeline driver (a thin wrapper calling Phase 6 then Phase 7) holds the last successful Midline3D per fish and forwards it when triangulation fails. This keeps the triangulator stateless and testable.

---

## Sources

### Primary (HIGH confidence)

- **Direct code inspection** — `triangulate_rays` (calibration/projection.py:8–47), `RefractiveProjectionModel.project` (same file:169–257), `RefractiveProjectionModel.cast_ray` (same file:109–167), `Midline2D` interface (reconstruction/midline.py:33–55), `FishTrack.camera_detections` (tracking/tracker.py:127)
- **Live environment verification** — `make_lsq_spline` with 15 3D points → exactly (7,3) control points; `splprep` with varying s → 4–15 control points; `make_lsq_spline` fails for n_data < n_ctrl; arc length computation via `spl(u_fine)` integration verified in hatch env (scipy 1.11+)
- **Algorithm verification** — Pairwise exhaustive search: 5-camera rig with 20px outlier in camera 2 correctly selects pair from clean cameras and excludes outlier in inlier re-triangulation; 2-camera triangulation error 0.00mm on noise-free data; outlier rejection (median + 2σ) correctly excludes 8px outlier from 2.0px-mean data

### Secondary (MEDIUM confidence)

- **CONTEXT.md** — locked decisions on triangulation strategy, spline control points, fallback behavior — source of truth; no independent verification needed
- **Phase 6 RESEARCH.md and midline.py** — confirmed `Midline2D` output interface, arc-length parameter conventions (i/14 for point i), full-frame pixel coordinates

### Tertiary (LOW confidence)

- **Width conversion formula** (`hw_m = hw_px * depth / focal`) — pinhole approximation for refractive camera; not verified against AquaCal ground truth. Flagged for validation in Phase 8 visualization.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified installed in hatch env; all in-project functions verified working
- Architecture: HIGH — module location follows established codebase pattern (`reconstruction/`); data structure design follows `Midline2D` precedent
- Pitfalls: HIGH for code-inspection-verified issues (`splprep` variability, u_param misalignment, Schoenberg-Whitney); MEDIUM for runtime pitfalls (width unit conversion, coasting logic)

**Research date:** 2026-02-21
**Valid until:** 2026-05-21 (stable libraries — scipy, numpy; 90-day validity)
