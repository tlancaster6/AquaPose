# Phase 1: Calibration and Refractive Geometry - Research

**Researched:** 2026-02-19
**Domain:** PyTorch refractive projection, AquaCal/AquaMVS integration
**Confidence:** HIGH — Both reference repos are on local disk; all code examined directly.

---

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Refractive model scope**
- Single flat air-water interface only — no glass panels in the optical path
- All 13 cameras are top-down, looking through the water surface
- Water surface height comes from AquaCal's calibration output (already calibrated, not estimated at runtime)
- Single constant index of refraction (n=1.333) — temperature variation is negligible

**Validation approach**
- Drop the "1px reprojection against ground truth" success criterion — that was already validated in AquaMVS
- Validation means: our adapted PyTorch code produces numerically equivalent output to AquaMVS's reference implementation for the same inputs
- AquaMVS is the primary reference (already PyTorch); AquaCal's numpy projection is not used for validation
- AquaMVS is in a separate repo — researcher should examine it to understand the code to adapt

**Z-uncertainty report**
- Serves dual purpose: inform downstream optimizer weighting AND document system accuracy for the paper
- Uniform depth sampling across the full tank range (e.g., every 5-10cm)
- Purely analytical — geometric ray intersection calculations from camera geometry and Snell's law, no empirical data
- Output as markdown report with embedded matplotlib/seaborn plots (error vs. depth curves for X, Y, Z separately)

**AquaCal integration**
- AquaCal is an importable Python package — use its API to load calibration data
- Provides full camera model: intrinsics (focal length, principal point, distortion), extrinsics (rotation, translation), and water surface plane
- AquaCal is a data loader for this phase — all differentiable projection math comes from adapting AquaMVS's PyTorch code

### Claude's Discretion

- Exact numerical tolerance for "equivalence" with AquaMVS (e.g., 1e-5 or 1e-6)
- Depth sampling resolution for Z-uncertainty report
- Plot styling and report structure details
- How to handle AquaCal's distortion model in the differentiable path

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope

</user_constraints>

---

## Summary

AquaMVS already contains a complete, working PyTorch refractive projection implementation in `src/aquamvs/projection/refractive.py`. AquaMVS also contains a complete calibration bridge in `src/aquamvs/calibration.py` that loads AquaCal JSON into PyTorch tensors. This phase is primarily an **adapt-and-port** task: copy these two files into AquaPose's `calibration/` module, make minor modifications (remove AquaMVS-specific dependencies, adjust imports), and write tests that verify equivalence against the AquaCal NumPy reference. AquaMVS already has cross-validation tests in `tests/test_projection/test_cross_validation.py` that serve as the exact test pattern to follow.

The one entirely new deliverable in this phase is the Z-uncertainty characterization report (CALIB-04). This is a standalone script that runs analytical geometry calculations using the already-implemented ray casting, then generates error-vs-depth curves for X, Y, and Z independently using matplotlib. No additional libraries are needed.

The key open question is distortion handling: AquaMVS's `RefractiveProjectionModel` works post-undistortion (it accepts an undistortion-corrected intrinsic matrix `K_new`). AquaPose will need to either pre-undistort images upstream (as AquaMVS does) or decide whether to handle distortion inside the differentiable path. The former is strongly recommended — it matches AquaMVS's validated approach and avoids making the non-differentiable OpenCV undistortion part of the gradient computation.

**Primary recommendation:** Copy AquaMVS's `projection/refractive.py` and `calibration.py` verbatim into AquaPose's `calibration/` module, then verify equivalence with AquaCal's NumPy implementation using the cross-validation test pattern from AquaMVS.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0 (already in pyproject.toml) | Differentiable projection, autograd | Already project dependency |
| aquacal | 1.4.1 (local editable) | Load calibration JSON, NumPy reference for validation | The calibration system for this rig |
| numpy | >=1.24 (already in pyproject.toml) | Array conversions at AquaCal boundary | Already project dependency |
| opencv-python | >=4.8 (already in pyproject.toml) | Undistortion remap tables (pinhole + fisheye) | Already project dependency |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | (via aquacal's deps) | Z-uncertainty report plots | CALIB-04 report generation only |
| scipy | >=1.11 (already in pyproject.toml) | Not needed for this phase | Already available if needed |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pre-undistortion (AquaMVS pattern) | In-loop distortion in differentiable path | In-loop would require differentiable lens model; OpenCV's undistortPoints is non-differentiable. Pre-undistortion is simpler, already validated, and sufficient since AquaPose processes video frames. |
| Fixed-count Newton-Raphson (AquaMVS) | Convergence-based stopping | Fixed count ensures differentiability (no data-dependent branching). AquaMVS uses 10 iterations, which is documented as more than sufficient for convergence. |

**Installation:**

```bash
# AquaCal must be installed as a local editable dependency
pip install -e "C:/Users/tucke/PycharmProjects/AquaCal"

# Add to pyproject.toml [project] dependencies (local path):
# aquacal @ file:///C:/Users/tucke/PycharmProjects/AquaCal
# OR keep as developer-installed and document separately
```

> Note: AquaCal is not on PyPI. It must be installed as a local editable install, the same pattern AquaMVS uses.

---

## Architecture Patterns

### Recommended Project Structure

```
src/aquapose/calibration/
├── __init__.py          # Public API: load_calibration, RefractiveCamera, etc.
├── loader.py            # AquaCal bridge: load_calibration_data(), CameraData, CalibrationData
├── projection.py        # RefractiveProjectionModel (ported from AquaMVS)
└── uncertainty.py       # Z-uncertainty report generator (CALIB-04)

tests/unit/calibration/
├── __init__.py
├── test_loader.py       # Unit tests for calibration loading
├── test_projection.py   # Unit tests for RefractiveProjectionModel
└── test_uncertainty.py  # Unit tests for uncertainty calculations

tests/integration/
└── test_calibration_cross_validation.py  # Cross-validation vs AquaCal NumPy
```

### Pattern 1: AquaCal Calibration Bridge

**What:** Load AquaCal JSON and convert to PyTorch tensors. Identical to AquaMVS's `calibration.py`.

**When to use:** CALIB-01. Called once at startup, not per-frame.

**Source:** `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/calibration.py` (verified working)

```python
# From AquaMVS calibration.py — port this verbatim into loader.py
from aquacal.io.serialization import load_calibration as aquacal_load_calibration

def load_calibration_data(calibration_path: str | Path) -> CalibrationData:
    result = aquacal_load_calibration(calibration_path)
    cameras = {}
    for name, cam_calib in result.cameras.items():
        K = torch.from_numpy(cam_calib.intrinsics.K).to(torch.float32)
        dist_coeffs = torch.from_numpy(cam_calib.intrinsics.dist_coeffs)  # float64
        R = torch.from_numpy(cam_calib.extrinsics.R).to(torch.float32)
        t_numpy = cam_calib.extrinsics.t
        if t_numpy.ndim == 2:
            t_numpy = t_numpy.squeeze()
        t = torch.from_numpy(t_numpy).to(torch.float32)
        cameras[name] = CameraData(name=name, K=K, dist_coeffs=dist_coeffs,
                                   R=R, t=t, image_size=cam_calib.intrinsics.image_size,
                                   is_fisheye=cam_calib.intrinsics.is_fisheye,
                                   is_auxiliary=cam_calib.is_auxiliary)
    water_z = next(iter(result.cameras.values())).water_z
    interface_normal = torch.from_numpy(result.interface.normal).to(torch.float32)
    if interface_normal.ndim == 2:
        interface_normal = interface_normal.squeeze()
    return CalibrationData(cameras=cameras, water_z=water_z,
                           interface_normal=interface_normal,
                           n_air=result.interface.n_air,
                           n_water=result.interface.n_water)
```

### Pattern 2: RefractiveProjectionModel (CALIB-02 and CALIB-03)

**What:** PyTorch class implementing `project()` (3D→pixel, Newton-Raphson, 10 fixed iters) and `cast_ray()` (pixel→underwater ray, Snell's law). Both methods are differentiable via autograd.

**When to use:** This is the core deliverable. Port from AquaMVS verbatim.

**Source:** `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/projection/refractive.py` (verified working, fully differentiable)

Key implementation details from AquaMVS source:
- `project()`: Uses 10 fixed Newton-Raphson iterations (no convergence check — maintains differentiability). Adds `1e-12` epsilon to `r_q` sqrt to avoid zero-grad. Uses `torch.clamp` and `torch.minimum` (not in-place) to stay in autograd graph.
- `cast_ray()`: Ray-plane intersection then vectorized Snell's law. Returns origins on water surface (Z=water_z) and unit direction vectors pointing into water.
- Both methods accept `(N, 2)` or `(N, 3)` float32 tensors. Return tuples. Device follows input.
- Constructor accepts K, R, t as float32 tensors; precomputes K_inv and C = -R^T @ t.
- The `to(device)` method moves all internal tensors, enabling GPU use.

```python
# Constructor signature (port verbatim):
class RefractiveProjectionModel:
    def __init__(self, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
                 water_z: float, normal: torch.Tensor,
                 n_air: float, n_water: float) -> None: ...

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns: (pixels (N,2), valid (N,) bool)
        # Invalid pixels are NaN
        ...

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns: (origins (N,3), directions (N,3))
        # Origins lie on water surface (Z = water_z)
        ...
```

### Pattern 3: Cross-Validation Test Strategy

**What:** Tests that compare AquaPose's ported PyTorch model against AquaCal's NumPy reference implementation. This is how "numerical equivalence" is defined.

**Source:** `C:/Users/tucke/PycharmProjects/AquaMVS/tests/test_projection/test_cross_validation.py` (verified working)

```python
# Cross-validation pattern from AquaMVS test_cross_validation.py
from aquacal.core.refractive_geometry import refractive_project, trace_ray_air_to_water
from aquapose.calibration import RefractiveProjectionModel  # our port

# For cast_ray cross-validation:
# 1. Build AquaCal Camera + Interface from same params
# 2. Call trace_ray_air_to_water(camera, interface, pixel_np) for each pixel
# 3. Call aquapose_model.cast_ray(pixels_pt) for the batch
# 4. Assert torch.allclose(origins_pt[i], origin_aquacal, atol=1e-5)

# For project cross-validation:
# 1. Call refractive_project(camera, interface, point_np) for each point
# 2. Call aquapose_model.project(points_pt) for the batch
# 3. Assert torch.allclose(pixels_pt[i], pixel_aquacal, atol=1e-5)
```

**Numerical tolerance recommendation:** `atol=1e-5` pixels for cross-validation. This is the tolerance AquaMVS itself uses. The difference between float32 (AquaPose) and float64 (AquaCal NumPy) means tighter tolerances are not appropriate.

### Pattern 4: Z-Uncertainty Analytical Report (CALIB-04)

**What:** Analytical characterization of X/Y/Z reconstruction uncertainty as a function of tank depth, using the multi-view triangulation geometry of the 13-camera rig.

**Method:** For each depth Z in the tank:
1. Project a known 3D point from each camera using `RefractiveProjectionModel.project()`
2. Add ±1 pixel perturbation to the projected pixel in each camera
3. Cast perturbed rays using `cast_ray()` and find the best triangulation
4. Measure X/Y/Z deviation from the ground-truth 3D point
5. Report as X-error(Z), Y-error(Z), Z-error(Z) curves

**Alternatively (fully analytical without real calibration file):** Derive analytically from Snell's law geometry for a known ring configuration (radius 0.635m, water_z≈0.978m). The AquaMVS CLAUDE.md confirms these reference geometry values.

**Sampling:** 5cm intervals from top of tank (water_z + 0.05m) to tank bottom (water_z + 0.5m or full range). This gives ~10 depth samples at 5cm intervals, or ~20 at 2.5cm.

**Output format:** Markdown file with embedded base64 PNG plots (or separate PNG files linked from markdown). Three plots minimum: X-error vs depth, Y-error vs depth, Z-error vs depth. One plot per camera or aggregate stats.

### Anti-Patterns to Avoid

- **In-place tensor operations in autograd path:** Use `torch.clamp` not `r_p.clamp_()`. Use `torch.minimum` not `r_p = min(r_p, r_q)`. AquaMVS correctly avoids these already.
- **Convergence-based loop termination:** The Newton-Raphson loop must use a fixed iteration count (10) so autograd can trace through it. A `break` on convergence introduces data-dependent control flow that can cause autograd issues.
- **float64 in the differentiable path:** K, R, t should be float32. The distortion coefficients are float64 but are only used in the OpenCV undistortion (not differentiable) path.
- **Importing AquaMVS directly:** AquaMVS is reference code only — port the relevant classes, do not add AquaMVS as a dependency of AquaPose.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Calibration JSON loading | Custom JSON parser | `aquacal.io.serialization.load_calibration` | AquaCal already handles version compatibility, backward compat with `interface_distance` legacy key |
| Lens distortion undistortion maps | Custom distortion model | `cv2.getOptimalNewCameraMatrix` + `cv2.initUndistortRectifyMap` (pinhole) or `cv2.fisheye.*` (fisheye) | OpenCV handles both pinhole and fisheye; AquaMVS already has this tested in `compute_undistortion_maps()` |
| Numerical gradient checking | Custom finite differences | `torch.autograd.gradcheck` | PyTorch's built-in works with double precision; use `atol=1e-4, rtol=1e-3` for float32→float64 conversion |
| Snell's law vector form | Custom derivation | Port AquaMVS's vectorized implementation | Already correct, tested, and handles the degenerate nadir-ray case |

**Key insight:** Both the calibration loader and the refractive projection model already exist in AquaMVS. The work is porting + validation, not reimplementation.

---

## Common Pitfalls

### Pitfall 1: Distortion Coefficients and the Differentiable Path

**What goes wrong:** The `K` matrix used in `RefractiveProjectionModel` must be the **post-undistortion** intrinsic matrix (`K_new`), not the raw `K`. Passing the raw distorted `K` will produce wrong projections.

**Why it happens:** AquaMVS separates undistortion (preprocessing, done once with OpenCV) from projection (done per-point with PyTorch). The `K` in AquaCal's calibration is the raw intrinsic. After undistortion, a different `K_new` describes the rectified image geometry.

**How to avoid:** When constructing `RefractiveProjectionModel`, use `compute_undistortion_maps(camera).K_new` as the `K` parameter, not `camera.K`. Pre-undistort the images separately.

**Warning signs:** Points that should project near the principal point project off-center; reprojection errors are large in image corners.

### Pitfall 2: AquaCal's `water_z` is Per-Camera, Not Global

**What goes wrong:** `CalibrationResult.cameras[name].water_z` stores `water_z` per camera (for legacy reasons, even though post-optimization all cameras share the same value). AquaMVS's `load_calibration_data` correctly extracts it from the first camera: `water_z = next(iter(result.cameras.values())).water_z`.

**Why it happens:** AquaCal's schema evolved: `water_z` was originally per-camera during calibration, then unified after optimization. The JSON stores it per-camera for backward compatibility.

**How to avoid:** Port AquaMVS's loading logic verbatim. Don't try to read `water_z` from `result.interface` — it's not stored there (only `normal`, `n_air`, `n_water` are in `InterfaceParams`).

### Pitfall 3: AquaCal Not in AquaPose's pyproject.toml

**What goes wrong:** AquaCal is not on PyPI and is not listed in AquaPose's `pyproject.toml` dependencies. A fresh `hatch env create` will not install it, causing import errors.

**Why it happens:** AquaCal is a local project installed with `pip install -e`. The `file://` path dependency format in pyproject.toml is fragile across machines.

**How to avoid:** Add `aquacal @ file:///path/to/AquaCal` as a dependency in pyproject.toml, OR document it clearly in CLAUDE.md as a prerequisite step (like pytorch3d). The AquaMVS README handles this with explicit install instructions. Consider the same pattern for AquaPose.

### Pitfall 4: Fisheye Camera in 13-Camera Rig

**What goes wrong:** The 13-camera rig has 12 ring cameras (standard pinhole) and 1 auxiliary center camera (fisheye). The fisheye camera uses `cv2.fisheye.*` functions for undistortion, not the standard `cv2.getOptimalNewCameraMatrix` path. Mixing these up produces wrong remap tables.

**Why it happens:** The fisheye equidistant model has 4 distortion coefficients and requires `cv2.fisheye.initUndistortRectifyMap`. The pinhole model uses `cv2.initUndistortRectifyMap`.

**How to avoid:** Port AquaMVS's `compute_undistortion_maps()` function which already dispatches on `camera.is_fisheye`. The auxiliary camera may not need to be included in the projection path for Phase 1 (it's a "source only" camera in AquaMVS), but the loading code must still handle it correctly.

### Pitfall 5: Coordinate System Z Convention

**What goes wrong:** The world frame uses +Z down (into water). Camera positions have Z near 0. Water surface is at `water_z > 0`. Underwater targets have `Z > water_z`. Getting this wrong causes "point above water" invalidity for all test points.

**Why it happens:** The +Z-down convention is inherited from AquaCal and is the opposite of the common +Z-up computer vision convention.

**How to avoid:** Use the reference geometry from AquaMVS's tests: camera at `(0.635, 0, 0)` with `water_z = 0.978`, test points at `(x, y, water_z + depth)` where `depth > 0`.

### Pitfall 6: gradcheck Requires float64

**What goes wrong:** `torch.autograd.gradcheck` requires double precision tensors to compute accurate finite differences. Passing float32 inputs will produce large gradient errors even for correct implementations.

**Why it happens:** gradcheck uses finite differences with step size ~1e-6, which is near float32 machine epsilon (~1.2e-7). Float64 has ~2.2e-16 machine epsilon, giving accurate numerical gradients.

**How to avoid:** In gradient tests, convert inputs to float64 with `.double()` before passing to gradcheck. The model will need a `.double()` method or reconstruction in float64.

---

## Code Examples

Verified patterns from AquaMVS source code:

### RefractiveProjectionModel.project() — Newton-Raphson Core

```python
# Source: AquaMVS src/aquamvs/projection/refractive.py (lines 176-227)
# Fixed 10 iterations — key for differentiability
for _ in range(10):
    d_air_sq = r_p * r_p + h_c * h_c
    d_air = torch.sqrt(d_air_sq)

    r_diff = r_q - r_p
    d_water_sq = r_diff * r_diff + h_q * h_q
    d_water = torch.sqrt(d_water_sq)

    sin_air = r_p / d_air
    sin_water = r_diff / d_water

    f = self.n_air * sin_air - self.n_water * sin_water
    f_prime = self.n_air * h_c * h_c / (
        d_air_sq * d_air
    ) + self.n_water * h_q * h_q / (d_water_sq * d_water)

    r_p = r_p - f / (f_prime + 1e-12)
    r_p = torch.clamp(r_p, min=0.0)      # not in-place: stays in autograd graph
    r_p = torch.minimum(r_p, r_q)        # not in-place
```

### AquaCal Cross-Validation Test Fixture

```python
# Source: AquaMVS tests/test_projection/test_cross_validation.py
# Reference geometry values (from AquaMVS CLAUDE.md):
{
    "fx": 1400.0, "fy": 1400.0,
    "cx": 800.0, "cy": 600.0,
    "image_size": (1600, 1200),
    "R": np.eye(3, dtype=np.float64),
    "t": np.array([-0.635, 0.0, 0.0], dtype=np.float64),
    "water_z": 0.978,
    "normal": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "n_air": 1.0, "n_water": 1.333,
    "dist_coeffs": np.zeros(5, dtype=np.float64),  # post-undistortion
}
```

### CalibrationData Loading and RefractiveProjectionModel Construction

```python
# Complete flow from raw JSON to projection model
from aquapose.calibration import load_calibration_data, compute_undistortion_maps, RefractiveProjectionModel

calib = load_calibration_data("calibration.json")

for cam_name in calib.ring_cameras:
    cam = calib.cameras[cam_name]
    undist = compute_undistortion_maps(cam)  # uses K_new, not K

    model = RefractiveProjectionModel(
        K=undist.K_new,                     # POST-undistortion intrinsics
        R=cam.R,
        t=cam.t,
        water_z=calib.water_z,
        normal=calib.interface_normal,
        n_air=calib.n_air,
        n_water=calib.n_water,
    )
    # Pre-undistort frames:
    # undistorted_frame = cv2.remap(frame, undist.map_x, undist.map_y, cv2.INTER_LINEAR)
```

### Autograd Gradient Check

```python
# Source: AquaMVS tests/test_projection/test_refractive.py (lines 406-429)
# Use torch.autograd.gradcheck for verification
import torch

points = torch.tensor(
    [[0.0, 0.0, 1.5], [0.2, 0.1, 1.3]],
    dtype=torch.float32,
    requires_grad=True,
)
pixels, valid = model.project(points)
loss = pixels[valid].sum()
loss.backward()  # must not raise; points.grad must be finite

# For rigorous numerical gradient check (use float64):
points_double = points.double().detach().requires_grad_(True)
model_double = RefractiveProjectionModel(
    K=model.K.double(), R=model.R.double(), t=model.t.double(),
    water_z=model.water_z, normal=model.normal.double(),
    n_air=model.n_air, n_water=model.n_water
)
from torch.autograd import gradcheck
gradcheck(lambda p: model_double.project(p)[0], (points_double,), atol=1e-4)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| AquaCal's NumPy Brent-search projection | AquaMVS's PyTorch Newton-Raphson (10 fixed iters) | AquaMVS implementation | ~50x faster, fully differentiable, same accuracy for flat interface |
| Per-camera `interface_distance` key | `water_z` key (with backward compat) | AquaCal serialization v1.0 | Must handle legacy key in deserialization; AquaMVS's loader already does |

**Deprecated/outdated in AquaCal:**
- `refractive_project_fast()` and `refractive_project_fast_batch()`: Deprecated in favor of `refractive_project()` and `refractive_project_batch()` which auto-select the fast Newton path. Don't reference these if cross-validating.

---

## Open Questions

1. **AquaCal as AquaPose dependency: how to manage it**
   - What we know: AquaCal is not on PyPI, installed locally with `pip install -e`
   - What's unclear: Whether `aquacal @ file:///...` in pyproject.toml is acceptable, or whether it should be documented as a manual prerequisite like pytorch3d
   - Recommendation: Follow the pytorch3d pattern — add a comment in pyproject.toml and a setup step in CLAUDE.md. Using `file://` paths breaks portability.

2. **Auxiliary (fisheye center) camera handling in projection**
   - What we know: In AquaMVS the center camera is "source only" — it provides photometric evidence but never produces a depth map itself
   - What's unclear: In AquaPose, should the center camera be excluded from the analysis-by-synthesis optimization entirely, or can it contribute silhouette evidence?
   - Recommendation: Load all 13 cameras but scope Phase 1 to proving the projection model works for all of them. Whether the center camera participates in optimization is a Phase 4 decision.

3. **Z-uncertainty report: analytical vs. ray-based calculation**
   - What we know: "Purely analytical — geometric ray intersection calculations" is specified. This means using the implemented ray casting and triangulation geometry, not sampling actual video frames.
   - What's unclear: Does "analytical" mean deriving a closed-form formula, or running the numerical ray-based simulation? For a 13-camera top-down geometry, the closed-form is tractable for Z-uncertainty but complex for X/Y.
   - Recommendation: Use the ray-simulation approach: for each depth, project ground-truth points, perturb by ±0.5px in each camera, cast rays, triangulate, measure error. This is "analytical" in the sense that it uses only geometry (no empirical data), and is more informative than a formula.

---

## Sources

### Primary (HIGH confidence — directly examined source code)

- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/projection/refractive.py` — Complete PyTorch RefractiveProjectionModel (project, cast_ray, to)
- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/calibration.py` — Complete AquaCal bridge (load_calibration_data, CameraData, CalibrationData, compute_undistortion_maps, undistort_image)
- `C:/Users/tucke/PycharmProjects/AquaMVS/tests/test_projection/test_refractive.py` — Unit tests for RefractiveProjectionModel (differentiability tests, Snell's law verification, roundtrip tests)
- `C:/Users/tucke/PycharmProjects/AquaMVS/tests/test_projection/test_cross_validation.py` — Cross-validation test pattern vs AquaCal NumPy (fixture geometry, tolerance values, validation strategy)
- `C:/Users/tucke/PycharmProjects/AquaMVS/tests/test_calibration.py` — Unit tests for calibration loading
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/io/serialization.py` — AquaCal JSON format, field names, backward compatibility
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/core/refractive_geometry.py` — NumPy reference implementations (Newton-Raphson, Brent-search, Snell's law)
- `C:/Users/tucke/PycharmProjects/AquaCal/src/aquacal/core/interface_model.py` — Interface class, water_z per-camera storage
- `C:/Users/tucke/PycharmProjects/AquaMVS/CLAUDE.md` — Reference geometry values (radius 0.635m, water_z=0.978m, n_water=1.333)
- `C:/Users/tucke/PycharmProjects/AquaPose/pyproject.toml` — Current AquaPose dependencies (torch, numpy, opencv-python, scipy, h5py)

### Secondary (MEDIUM confidence)

- None required — all critical information came from primary sources.

---

## Metadata

**Confidence breakdown:**
- What to port: HIGH — code fully examined, works in AquaMVS
- Validation strategy: HIGH — cross-validation test pattern directly available in AquaMVS
- Distortion handling: HIGH — AquaMVS pattern is clear and tested
- Z-uncertainty report approach: MEDIUM — methodology is clear, but exact output format is Claude's discretion
- AquaCal dependency management: MEDIUM — established pattern from AquaMVS but cross-machine portability is an open question

**Research date:** 2026-02-19
**Valid until:** Stable — sources are local repos; valid until AquaCal or AquaMVS APIs change
