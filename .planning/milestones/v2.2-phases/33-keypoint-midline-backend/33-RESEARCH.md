# Phase 33: Keypoint Midline Backend - Research

**Researched:** 2026-02-28
**Domain:** PyTorch inference, spline interpolation, weighted least squares, CLI extension
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Anatomical keypoints
- The 6 keypoints are: **nose, head, spine1, spine2, spine3, tail**
- Each keypoint has a fixed `t` value in [0, 1] representing its arc-length fraction along the body
- t-values are a **configurable field** in the keypoint backend config, not hardcoded
- A new CLI command `aquapose prep calibrate-keypoints` auto-calibrates t-values from training data
- The `prep` CLI group is new — will eventually also house manual LUT generation (deferred)
- Manual keypoint annotations already exist for training data

#### Partial visibility handling
- Two-tier approach: hard confidence floor (configurable, default 0.1) removes truly terrible inferences → NaN+conf=0
- Remaining points pass through with their actual confidence values for soft weighting in reconstruction
- Minimum observed keypoint count: **3 of 6** (configurable via `min_observed_keypoints`)
- Below the minimum → empty midline (same as degenerate mask in segment-then-extract)
- Output is always exactly `n_sample_points` — NaN+conf=0 outside `[t_min_observed, t_max_observed]`, never a shorter array
- The n_points correspondence contract from PITFALLS.md is respected: point index i always means the same anatomical position across cameras

#### Backend failure mode
- Degenerate keypoint output → flagged empty midline, consistent with segment-then-extract
- No cross-backend fallback (no falling back to segment-then-extract on failure)
- Trust the model's confidence — no separate sanity checks on keypoint spatial distribution
- Empty midlines produced silently (no warning logs); observers/diagnostics track via events

#### Confidence weighting in reconstruction
- Both triangulation and curve optimizer use **sqrt(confidence)** to scale observation weights
- Triangulation: scale each view's A-matrix rows by sqrt(conf) before SVD (standard weighted least squares)
- Curve optimizer: same sqrt weighting on fitting objective terms
- Points with NaN coordinates are **excluded entirely** from the DLT system (not passed as zero-weight rows)
- When confidence is None (segment-then-extract), uniform weights apply — no config override needed
- Identical output to previous version when confidence is `None` (backward compatibility)

### Claude's Discretion
- Regression head architecture (layers, activation, output format)
- Spline degree and fitting method for keypoints → N-point midline
- Internal structure of `aquapose prep calibrate-keypoints` implementation
- Config field names and default values (except those specified above)

### Deferred Ideas (OUT OF SCOPE)
- Manual LUT generation under `aquapose prep` CLI group — future phase
- Body-model extrapolation for partial midlines (QUAL-01) — future requirement
- Confidence calibration via temperature scaling (QUAL-02) — future requirement
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MID-01 | Pipeline supports a keypoint regression backend as a swappable alternative to segment-then-extract, selectable via config | `DirectPoseBackend` stub already exists in `core/midline/backends/direct_pose.py`; `get_backend()` registry already routes to it; `MidlineConfig.backend` field exists. This phase replaces the stub with a real implementation. |
| MID-02 | Keypoint backend produces N ordered midline points with per-point confidence from a learned regression model (U-Net encoder + regression head) | `_PoseModel` already trained and saved in `training/pose.py`; inference uses same `_PoseModel` class; output is 6*(x,y) sigmoid-normalized coords; spline → resample to n_sample_points needed. |
| MID-03 | Keypoint backend handles partial visibility by marking unobserved regions with NaN coordinates and zero confidence, always outputting exactly `n_sample_points` | Two-tier confidence floor → NaN+0 for low-confidence keypoints; spline evaluated only within `[t_min_observed, t_max_observed]`; NaN-filled outside observed arc-span; output shape always (n_sample_points, 2). |
| MID-04 | Both midline backends produce the same output structure (N-point Midline2D) so reconstruction is backend-agnostic | `Midline2D` dataclass already has `point_confidence: np.ndarray | None` field; `segment_then_extract` fills with uniform 1.0s; `direct_pose` fills with actual per-point values. No structural change to Midline2D needed. |
| RECON-01 | Triangulation backend weights per-point observations by confidence when available, falling back to uniform weights when confidence is None | Current `triangulate_midlines()` reads `midline.point_confidence` field; need to modify `_triangulate_body_point()` to accept per-view weights and apply `sqrt(conf)` scaling to DLT A-matrix rows before SVD. |
| RECON-02 | Curve optimizer backend weights observations by confidence when available, falling back to uniform weights when confidence is None | `_data_loss()` in `curve_optimizer.py` currently uses symmetric Chamfer; need confidence-weighted variant when `point_confidence` is present on any input `Midline2D`. |
</phase_requirements>

---

## Summary

Phase 33 implements the `DirectPoseBackend` that was stubbed in Phase 32-01. The stub already exists at `src/aquapose/core/midline/backends/direct_pose.py` — it just raises `NotImplementedError`. The backend registry (`get_backend()`) and the `MidlineConfig.backend` field are already wired. The `_PoseModel` (U-Net encoder + regression head) and `KeypointDataset` are already implemented in `training/pose.py`. The `Midline2D` dataclass already carries `point_confidence`. This phase is pure implementation: no new protocols, no new dataclasses at the structural level.

The three technical areas are: (1) `DirectPoseBackend` inference — load `_PoseModel`, extract affine crop (using the Phase 32 `extract_affine_crop` utility), run inference, convert 6 sigmoid keypoints to N-point midline via spline with NaN-padding; (2) reconstruction weighting — both `triangulate_midlines()` and `CurveOptimizer._data_loss()` need sqrt(confidence) weighting when `point_confidence` is present; (3) CLI — a new `prep` group with `calibrate-keypoints` subcommand, and extended `MidlineConfig` for `direct_pose` fields.

The largest risk is the triangulation weighting change. The DLT system in `_triangulate_body_point()` uses `triangulate_rays()` from `calibration/projection.py` which does not currently accept weights. Weighted DLT requires modifying the A-matrix rows before SVD, which must stay inside `reconstruction/triangulation.py` to respect the import boundary (core/ must not import engine/). The backward-compatibility guarantee (identical output when `confidence is None`) means the weighting is a conditional no-op path.

**Primary recommendation:** Implement `DirectPoseBackend` in three clean waves: (1) inference + spline + NaN-padding (MID-01, MID-02, MID-03, MID-04) + MidlineConfig extension + `prep calibrate-keypoints` CLI; (2) weighted triangulation (RECON-01); (3) weighted curve optimizer (RECON-02).

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | project default | Model inference, tensor ops | Already used for _PoseModel, training |
| scipy.interpolate | project default | CubicSpline / interp1d for keypoints → midline | Already used in reconstruction/midline.py for arc-length resampling |
| numpy | project default | Array ops, NaN handling | Already used throughout |
| click | project default | CLI group/subcommand | Already used for `train_group` pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cv2 | project default | `extract_affine_crop` back-projection | Used inside DirectPoseBackend to produce OBB-aligned crop for inference |
| torch.nn | project default | `_PoseModel` loading for inference | Import at construction time (lazy) to avoid engine/ import at module level |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy CubicSpline | scipy make_lsq_spline | CubicSpline is simpler for 6-to-N extrapolation at fixed t-values; make_lsq_spline is better for over-determined fits. With exactly 6 points and partial NaN, CubicSpline with sorted visible t-values is the right call. |
| sqrt(conf) weighting | conf^1 or conf^2 | sqrt is standard weighted least squares for standard-deviation-based weights; conf^1 over-penalizes; conf^2 concentrates too much on high-conf points. Locked decision. |

---

## Architecture Patterns

### Recommended Project Structure Changes

```
src/aquapose/
├── core/midline/backends/
│   ├── direct_pose.py          # Replace stub with real implementation
│   └── __init__.py             # No change needed (already routes to direct_pose)
├── reconstruction/
│   └── triangulation.py        # Add confidence weighting to _triangulate_body_point()
│   └── curve_optimizer.py      # Add confidence weighting to _data_loss()
├── engine/
│   └── config.py               # Extend MidlineConfig with direct_pose fields
└── cli.py                      # Add prep_group, register to cli
src/aquapose/training/
└── cli.py                      # Add calibrate-keypoints to a new prep group, OR in cli.py
tests/unit/core/midline/
└── test_direct_pose_backend.py # NEW - backend unit tests
tests/unit/core/reconstruction/
└── test_reconstruction_stage.py# EXTEND - confidence weighting tests
```

### Pattern 1: DirectPoseBackend Inference Flow

**What:** Load `_PoseModel` at construction (fail-fast), run inference per detection, convert 6 keypoints → spline → N-point midline.

**When to use:** Called by `MidlineStage` via `backend.process_frame()` when `backend="direct_pose"`.

**Implementation sketch:**

```python
# src/aquapose/core/midline/backends/direct_pose.py

class DirectPoseBackend:
    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cuda",
        n_points: int = 15,
        n_keypoints: int = 6,
        keypoint_t_values: list[float] | None = None,  # configurable t-values
        confidence_floor: float = 0.1,
        min_observed_keypoints: int = 3,
    ) -> None:
        # Fail-fast weights validation (same pattern as SegmentThenExtractBackend)
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(...)

        self._n_points = n_points
        self._n_keypoints = n_keypoints
        self._t_values = keypoint_t_values or _default_t_values(n_keypoints)
        self._conf_floor = confidence_floor
        self._min_observed = min_observed_keypoints
        self._device = device

        # Lazy import pattern (used in SegmentThenExtractBackend)
        import torch
        from aquapose.training.pose import _PoseModel
        model = _PoseModel(n_keypoints=n_keypoints, pretrained=False)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        self._model = model

    def process_frame(
        self,
        frame_idx: int,
        frame_dets: dict[str, list[Detection]],
        frames: dict[str, np.ndarray],
        camera_ids: list[str],
    ) -> dict[str, list[AnnotatedDetection]]:
        ...
```

**Key steps inside process_frame:**
1. For each camera/detection, extract affine crop using `extract_affine_crop(frame, center_xy, angle_math_rad, obb_w, obb_h, crop_size=(128,128))`. If detection has no `angle` (axis-aligned box), use `angle=0.0`.
2. Run `_PoseModel` inference → shape `(1, n_keypoints*2)` sigmoid output in [0,1] crop space.
3. Back-project keypoints to frame space using `invert_affine_points(crop_kps, affine_crop.M)`.
4. Apply confidence floor: model confidence must be inferred from a separate head OR estimated from sigmoid output distance from 0.5 (see Open Questions). For the Phase 33 locked plan: confidence floor removes keypoints below threshold → NaN+0.
5. Check min_observed_keypoints threshold.
6. Fit CubicSpline through visible t-values, evaluate at `np.linspace(0, 1, n_sample_points)`.
7. NaN-pad outside `[t_min_observed, t_max_observed]`.
8. Return `AnnotatedDetection(detection=det, mask=None, crop_region=None, midline=midline_2d, ...)`.

### Pattern 2: Confidence-Weighted DLT Triangulation (RECON-01)

**What:** Scale each camera's A-matrix rows by `sqrt(conf_i)` before SVD in `_triangulate_body_point()`.

**Background:** DLT triangulation solves `Ax = 0` via SVD of A. Each camera contributes 2 rows. Weighting row i by w_i makes the SVD minimize `||WAx||^2` — standard weighted least squares.

**Location:** `src/aquapose/reconstruction/triangulation.py`, specifically:
- `triangulate_midlines()` gathers `pixels: dict[str, torch.Tensor]` per body point
- `_triangulate_body_point()` calls `triangulate_rays(origs, dirs)` — this is the DLT call site
- `triangulate_rays()` is in `aquapose.calibration.projection` — **do NOT modify that function** (shared utility)

**The correct approach:** Modify `_triangulate_body_point()` to accept an optional `weights: dict[str, float]` parameter. When provided, scale origins/directions before passing to `triangulate_rays()`, OR implement a local weighted SVD instead of calling `triangulate_rays()` directly.

Actually, `triangulate_rays()` builds the A-matrix and solves via SVD. The safest approach is to duplicate the DLT solve locally within the weighting path, or pre-scale the ray representation. The cleanest solution: add a `_weighted_triangulate_rays(origins, directions, weights)` helper in `triangulation.py` that applies `w * A_rows` before SVD.

**Propagation path:**
```
triangulate_midlines(midline_set, ...)
  → reads midline.point_confidence[i] for body point i across cameras
  → passes weights dict to _triangulate_body_point(pixels, models, inlier_threshold, weights={cam_id: sqrt_conf})
  → _triangulate_body_point calls _weighted_triangulate_rays(origs, dirs, weights_tensor)
```

**Backward compatibility:** When `point_confidence is None` on all midlines in `midline_set`, weights default to uniform (no change). Output is identical when all weights are 1.0.

**NaN handling in triangulation:** Body points with NaN coordinates are already excluded via the `np.any(np.isnan(pt))` check at line 847 of `triangulation.py`. No change needed — NaN-coord points from the keypoint backend will simply be skipped (as intended by the CONTEXT.md decision: "Points with NaN coordinates are excluded entirely from the DLT system").

### Pattern 3: Confidence-Weighted Curve Optimizer (RECON-02)

**What:** Weight the chamfer distance contribution per camera observation point by `sqrt(confidence)`.

**Location:** `src/aquapose/reconstruction/curve_optimizer.py`, specifically `_data_loss()` and `_chamfer_distance_2d()`.

**Current behavior:** `_chamfer_distance_2d(proj_valid, obs_pts)` is symmetric: mean of `proj→obs` and `obs→proj` directed distances. All observed points have equal weight.

**Modified behavior:** When confidence weights are available, compute a weighted mean for the `obs→proj` direction: for each observed point i, its contribution is `w_i * min_dist(obs_i → proj)`. The `proj→obs` direction remains unweighted (the projected spline points have no per-point confidence). Average over cameras as before.

**Propagation path:**
```python
# In _data_loss(), for each fish/camera:
obs_pts_weighted = ...  # (M, 2) observed points
conf_weights = ...  # (M,) weights from midline.point_confidence, or None

# Pass conf_weights to a modified chamfer function
chamfer = _weighted_chamfer_distance_2d(proj_valid, obs_pts_weighted, conf_weights)
```

**MidlineSet access in curve optimizer:** `optimize_midlines()` receives `midline_set: MidlineSet` (dict[int, dict[str, Midline2D]]). The `Midline2D` objects have `point_confidence`. The `midlines_per_fish` tensor prep needs to also extract and carry weights alongside the `obs_pts` tensors.

**Backward compatibility:** When `midline.point_confidence is None`, use uniform weights (identical behavior). The `_chamfer_distance_2d()` function can remain unchanged; a new `_weighted_chamfer_distance_2d()` handles the confidence path.

### Pattern 4: MidlineConfig Extension for direct_pose

**What:** Add `direct_pose`-specific fields to `MidlineConfig` in `engine/config.py`.

**Existing MidlineConfig fields:** `confidence_threshold`, `weights_path`, `backend`, `n_points`, `min_area`, `detection_tolerance`, `speed_threshold`, orientation weights.

**New fields needed:**
```python
@dataclass(frozen=True)
class MidlineConfig:
    ...
    # direct_pose backend fields
    keypoint_weights_path: str | None = None      # Path to _PoseModel weights
    keypoint_t_values: list[float] | None = None   # 6 t-values, or None = uniform
    keypoint_confidence_floor: float = 0.1        # Hard floor below → NaN+0
    min_observed_keypoints: int = 3               # Minimum keypoints required
```

**IMPORTANT:** `_filter_fields()` is already applied to `MidlineConfig` in `load_config()` — new fields automatically load from YAML without error on old configs (they get their defaults).

**Note:** `weights_path` currently means U-Net weights for `segment_then_extract`. For `direct_pose`, use `keypoint_weights_path` to avoid ambiguity. Do NOT reuse `weights_path` for both backends.

**build_stages() wiring:** `MidlineStage.__init__()` currently passes `weights_path` to `get_backend()`. When `backend="direct_pose"`, it should pass `keypoint_weights_path` instead (and the other `direct_pose`-specific fields).

### Pattern 5: `prep` CLI Group and `calibrate-keypoints`

**What:** A new `prep` click group in `cli.py` (or a new `src/aquapose/training/prep_cli.py`) with a `calibrate-keypoints` subcommand.

**`calibrate-keypoints` purpose:** Given a COCO annotations.json with keypoint data, compute the mean normalized arc-length position of each keypoint across all training samples. Output: a list of 6 t-values in [0,1] for use as `keypoint_t_values` in config.

**Pattern mirrors `train_group`:**
```python
# src/aquapose/cli.py (or a new prep_cli.py)
@click.group("prep")
def prep_group() -> None:
    """Prepare calibration data and configuration."""

@prep_group.command("calibrate-keypoints")
@click.option("--annotations", required=True, type=click.Path(exists=True), ...)
@click.option("--output", required=True, type=click.Path(), ...)
def calibrate_keypoints(...) -> None:
    """Compute keypoint t-values from COCO keypoint annotations."""
    ...
    # Write a YAML snippet with t_values: [0.0, 0.1, ...]
```

**Register to CLI:** `cli.add_command(prep_group)` in `cli.py` (same as `cli.add_command(train_group)`).

### Anti-Patterns to Avoid

- **Importing `engine/` from `core/`:** The `DirectPoseBackend` is in `core/midline/backends/`. It must NOT import `engine/config.py` or anything from `engine/`. Config values are passed as constructor parameters. The import boundary is enforced by pre-commit AST lint.
- **Passing zero-weight NaN rows to DLT:** NaN-coordinate body points from the keypoint backend must be excluded from the `pixels` dict before calling `_triangulate_body_point()`. They are currently excluded by the `np.any(np.isnan(pt))` guard in `triangulate_midlines()` — this is already correct.
- **Mutating frozen MidlineConfig:** `MidlineConfig` is a frozen dataclass. `DirectPoseBackend` parameters come from the constructor call in `MidlineStage.__init__()`, not from config mutation. The stage extracts the fields from `midline_config` at construction time.
- **Reusing `weights_path` for two different models:** `weights_path` means U-Net weights for `segment_then_extract`. Use `keypoint_weights_path` for the pose model. Both fields can coexist in MidlineConfig; each backend reads only its own field.
- **Shorter output arrays for partial midlines:** The output of `DirectPoseBackend.process_frame()` must ALWAYS produce exactly `n_sample_points` points. Never return a shorter array. Use NaN+0 padding for unobserved regions. Returning fewer points breaks the n_points correspondence contract and causes index misalignment in triangulation.
- **Modifying `triangulate_rays()` in `calibration/projection.py`:** That function is a shared utility. Add a local `_weighted_triangulate_rays()` inside `reconstruction/triangulation.py` instead.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Keypoints → midline curve | Custom polynomial fit | `scipy.interpolate.CubicSpline` with visible-t-values only | CubicSpline correctly handles irregular t-spacings; available in project deps |
| Affine crop for inference | New crop utility | `extract_affine_crop()` in `segmentation/crop.py` | Already implemented in Phase 32; handles OBB angle correctly |
| Back-project crop keypoints to frame | Manual matrix math | `invert_affine_points()` in `segmentation/crop.py` | Numerically exact via `cv2.invertAffineTransform` |
| Model inference | Custom forward pass | `_PoseModel.forward()` + `torch.no_grad()` | Model class already in `training/pose.py`; import it for inference too |
| Weighted SVD | Third-party solver | Local `_weighted_triangulate_rays()` inside `triangulation.py` | torch already available; scale A-matrix rows then use `torch.linalg.svd` |

**Key insight:** The project already has ~80% of the plumbing for this phase. The implementation is mostly connecting existing pieces rather than inventing new ones.

---

## Common Pitfalls

### Pitfall 1: Skeleton-Squashing — Broken n_points Correspondence Contract
**What goes wrong:** If the keypoint backend returns fewer than `n_sample_points` points (e.g., only the 4 observed keypoints), triangulation maps point index 0 from camera A to a different body position than index 0 from camera B, producing zigzag 3D reconstructions.
**Why it happens:** Different partial visibility patterns per camera means different indices refer to different body positions.
**How to avoid:** ALWAYS output exactly `n_sample_points`. Use NaN + conf=0 for unobserved arc positions. This is the architectural contract documented in the CONTEXT.md `specifics` section.
**Warning signs:** 3D reconstructions look noisy/zigzag despite clean 2D midlines.

### Pitfall 2: Confidence Floor Applied to Wrong Quantity
**What goes wrong:** The `confidence_floor` is applied to per-point confidence values, but the `_PoseModel` output has no explicit confidence head — it outputs only (x,y) coords via Sigmoid. The confidence must be derived or treated as implicit.
**Why it happens:** The training pipeline (`train_pose`) uses MSE loss on coordinates; there is no explicit confidence regression.
**How to avoid:** For Phase 33, treat per-point confidence as a post-hoc heuristic: distance of sigmoid output from 0.5 is a proxy (output near 0.5 means uncertain). Alternatively, use the annotation visibility flags from the COCO JSON (visibility=0 → conf=0, visibility=1 → configurable partial conf, visibility=2 → full conf=1.0). The CONTEXT.md says "trust the model's confidence" but the model doesn't explicitly output it — this is an open question (see below).
**Warning signs:** Reconstruction quality unchanged between high-conf and low-conf keypoint inferences.

### Pitfall 3: Affine Crop with Axis-Aligned Detection
**What goes wrong:** `extract_affine_crop()` requires an OBB angle. If the upstream detection was axis-aligned YOLO (no `Detection.angle`), the backend errors on None angle.
**Why it happens:** `DirectPoseBackend` is designed for OBB detections from Phase 32, but may also receive detections from the standard YOLO backend.
**How to avoid:** In `DirectPoseBackend.process_frame()`, check `det.angle is not None`; if None, use `angle=0.0` (no rotation). This produces an axis-aligned crop identical to `compute_crop_region` + `extract_crop`.
**Warning signs:** `AttributeError` or `TypeError` on `None` angle values in `extract_affine_crop()`.

### Pitfall 4: Backward-Compatibility Break in triangulate_midlines()
**What goes wrong:** Adding `weights` parameter to `_triangulate_body_point()` breaks callers that pass positional arguments.
**Why it happens:** The function signature change.
**How to avoid:** Add `weights: dict[str, float] | None = None` as a keyword-only parameter with `None` defaulting to uniform weights. All existing calls that don't pass weights continue to work unchanged.
**Warning signs:** Test failures in existing `test_reconstruction_stage.py` or `test_triangulation.py`.

### Pitfall 5: CurveOptimizer midlines_per_fish tensor prep
**What goes wrong:** `optimize_midlines()` converts `midline_set` to a list of `dict[str, torch.Tensor]` (observed points). If `point_confidence` is not also extracted during this conversion, `_data_loss()` can't access it.
**Why it happens:** The obs_pts tensors don't carry confidence; `Midline2D` is the source of truth.
**How to avoid:** During the `midlines_per_fish` preparation loop, also build a parallel `confidence_per_fish: list[dict[str, torch.Tensor | None]]` structure. Pass both to `_data_loss()`.
**Warning signs:** Confidence weighting silently ignored in curve optimizer output.

### Pitfall 6: `training/` importing `core/` — Import Boundary Violation
**What goes wrong:** `DirectPoseBackend` is in `core/` and imports `_PoseModel` from `training/`. This direction is allowed (core/ can import training/ as a third-party-like module for model weights). BUT `training/` must not import `engine/` (pre-commit enforces this).
**Why it happens:** Confusion about the import boundary rules.
**How to avoid:** `core/midline/backends/direct_pose.py` imports `from aquapose.training.pose import _PoseModel` — this is fine. The boundary is `training/ must not import engine/`, not `core/ must not import training/`.
**Warning signs:** Pre-commit AST import boundary lint failure.

---

## Code Examples

Verified patterns from existing codebase:

### Loading _PoseModel for inference (from training/pose.py)
```python
# Source: src/aquapose/training/pose.py
import torch
from aquapose.training.pose import _PoseModel

model = _PoseModel(n_keypoints=6, pretrained=False)
state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Inference
with torch.no_grad():
    image_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    pred = model(image_tensor.unsqueeze(0).to(device))  # (1, 12)
    kps_norm = pred[0].cpu().numpy().reshape(6, 2)  # (6, 2) in [0,1] crop space
```

### Back-projecting keypoints to frame space (from segmentation/crop.py)
```python
# Source: src/aquapose/segmentation/crop.py
from aquapose.segmentation.crop import extract_affine_crop, invert_affine_points

affine = extract_affine_crop(
    frame,
    center_xy=(det_cx, det_cy),
    angle_math_rad=det.angle if det.angle is not None else 0.0,
    obb_w=det_w,
    obb_h=det_h,
    crop_size=(128, 128),
)
# kps_crop: (6, 2) crop-space pixel coords
kps_crop_px = kps_norm * np.array([128, 128])
kps_frame = invert_affine_points(kps_crop_px, affine.M)  # (6, 2) frame coords
```

### NaN-padded midline construction (locked contract)
```python
# After spline evaluation:
t_eval = np.linspace(0.0, 1.0, n_sample_points)
midline_pts = np.full((n_sample_points, 2), np.nan, dtype=np.float32)
midline_conf = np.zeros(n_sample_points, dtype=np.float32)

# Fit spline on visible keypoints
visible_t = t_values[visible_mask]   # subset of t_values
visible_kps = kps_frame[visible_mask]  # (M, 2)
spl = scipy.interpolate.CubicSpline(visible_t, visible_kps, extrapolate=False)

# Only evaluate within observed arc span
t_min, t_max = visible_t[0], visible_t[-1]
observed_mask = (t_eval >= t_min) & (t_eval <= t_max)
midline_pts[observed_mask] = spl(t_eval[observed_mask]).astype(np.float32)

# Confidence: interpolate from per-keypoint confidence for observed span
conf_spl = scipy.interpolate.interp1d(visible_t, visible_conf, bounds_error=False,
                                       fill_value=0.0)
midline_conf[observed_mask] = conf_spl(t_eval[observed_mask]).astype(np.float32)
```

### MidlineConfig extension pattern (from engine/config.py)
```python
# Following existing frozen dataclass pattern in engine/config.py
@dataclass(frozen=True)
class MidlineConfig:
    ...
    # Existing fields remain unchanged
    # New direct_pose fields:
    keypoint_weights_path: str | None = None
    keypoint_t_values: list[float] | None = None
    keypoint_confidence_floor: float = 0.1
    min_observed_keypoints: int = 3
```

### Weighted DLT triangulation sketch
```python
# In reconstruction/triangulation.py

def _weighted_triangulate_rays(
    origins: torch.Tensor,    # (N, 3)
    directions: torch.Tensor, # (N, 3)
    weights: torch.Tensor,    # (N,) — sqrt(confidence) values
) -> torch.Tensor:
    """Weighted DLT: minimize ||diag(w) A x||^2."""
    # Build DLT A-matrix (2N x 4 homogeneous)
    # Scale rows by weights before SVD
    # Return 3D point
    ...
```

### Confidence extraction in triangulate_midlines()
```python
# In triangulate_midlines(), per body point i:
weights: dict[str, float] = {}
for cam_id, midline in cam_midlines.items():
    if cam_id not in models:
        continue
    pt = midline.points[i]
    if np.any(np.isnan(pt)):
        continue
    # Extract per-point confidence for weighting
    if midline.point_confidence is not None:
        w = float(np.sqrt(midline.point_confidence[i]))
    else:
        w = 1.0  # uniform — backward compatible
    pixels[cam_id] = torch.from_numpy(pt).float()
    weights[cam_id] = w

result = _triangulate_body_point(
    pixels, models, inlier_threshold, water_z=water_z, weights=weights
)
```

### prep CLI group pattern (mirroring train_group in training/cli.py)
```python
# In src/aquapose/cli.py or a new prep_cli.py:
@click.group("prep")
def prep_group() -> None:
    """Prepare calibration data and derived configuration."""

@prep_group.command("calibrate-keypoints")
@click.option("--annotations", required=True, type=click.Path(exists=True),
              help="Path to COCO keypoint annotations JSON.")
@click.option("--output", required=True, type=click.Path(),
              help="Path to write t-values YAML snippet.")
@click.option("--n-keypoints", default=6, type=int)
def calibrate_keypoints(annotations: str, output: str, n_keypoints: int) -> None:
    """Compute keypoint t-values from COCO annotations."""
    ...
    # Compute mean arc-length fraction for each keypoint across all samples
    # Write YAML: keypoint_t_values: [0.0, 0.18, 0.36, ...]
```

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` (hatch test scripts) |
| Quick run command | `hatch run test` |
| Full suite command | `hatch run test-all` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MID-01 | `backend="direct_pose"` no longer raises NotImplementedError | unit | `hatch run test tests/unit/core/midline/test_midline_stage.py -x` | Extend existing |
| MID-02 | DirectPoseBackend.process_frame() returns AnnotatedDetection with Midline2D containing n_sample_points | unit | `hatch run test tests/unit/core/midline/test_direct_pose_backend.py -x` | Wave 0 |
| MID-03 | Partial visibility: output always n_sample_points, NaN+0 outside observed span | unit | `hatch run test tests/unit/core/midline/test_direct_pose_backend.py::test_partial_visibility -x` | Wave 0 |
| MID-04 | Both backends produce Midline2D with same shape/field structure | unit | `hatch run test tests/unit/core/midline/ -x` | Extend existing |
| RECON-01 | triangulate_midlines() with point_confidence produces different result than without | unit | `hatch run test tests/unit/core/reconstruction/test_reconstruction_stage.py -x` | Extend existing |
| RECON-02 | CurveOptimizer with point_confidence applies weighting | unit | `hatch run test tests/unit/core/reconstruction/test_reconstruction_stage.py -x` | Extend existing |

### Sampling Rate
- **Per task commit:** `hatch run test`
- **Per wave merge:** `hatch run test`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/core/midline/test_direct_pose_backend.py` — covers MID-02, MID-03 (stub test file, real backend)
- [ ] `tests/unit/core/midline/test_midline_stage.py` — extend: remove `raises NotImplementedError` assertion for direct_pose, add real test

*(If no gaps: existing test for `test_midline_stage.py` currently asserts `NotImplementedError` for `direct_pose` — that test must be updated, not just extended)*

---

## Open Questions

1. **Per-point confidence source in `_PoseModel` output**
   - What we know: `_PoseModel.forward()` returns `(B, n_keypoints*2)` normalized (x,y) coords via Sigmoid. No explicit confidence head was trained.
   - What's unclear: CONTEXT.md says "Two-tier approach: hard confidence floor removes truly terrible inferences → NaN+conf=0; remaining points pass through with their actual confidence values." But the trained model doesn't output confidence values.
   - Recommendation: Use annotation visibility from COCO JSON at inference time to decide training-time confidence, and at inference time use a heuristic: sigmoid output near 0.5 in either x OR y coordinate suggests uncertain prediction (model unsure). Alternatively, define confidence=1.0 for all keypoints above the floor, and only the hard floor (visibility=0 equivalent) produces NaN+0. This is the simpler interpretation that matches "trust the model's confidence." The planner should specify whether a confidence head needs to be added to `_PoseModel` or whether binary visible/not-visible is sufficient. **[LOW confidence — needs clarification in planning session]**

2. **OBB detection dependency: does `direct_pose` require Phase 32 OBB detections, or can it work with axis-aligned boxes?**
   - What we know: Phase 32 adds `extract_affine_crop()` and OBB detection. Phase 33 depends on Phase 32.
   - What's unclear: The MidlineStage currently builds axis-aligned crops from YOLO axis-aligned detections. If `direct_pose` backend is used with standard YOLO (no OBB), the crop will be axis-aligned (angle=0), which still works but may be suboptimal for tilted fish.
   - Recommendation: Handle gracefully — use `det.angle if det.angle is not None else 0.0`. This is already covered in Pitfall 3. **[HIGH confidence]**

3. **`_PoseModel` output vs crop size: 128x128 normalization**
   - What we know: `_PoseModel` expects 128x128 input (see `_INPUT_SIZE = 128` in `training/pose.py`). Output is normalized to [0,1] in crop space.
   - What's unclear: The affine crop utility produces crop of `crop_size=(128, 128)` by default. If `DetectionConfig.crop_size` is configured differently (e.g., [256, 128] for wide fish), the model was trained at 128x128 but inference uses 256x128.
   - Recommendation: `DirectPoseBackend` should always resize the crop to 128x128 before inference, regardless of `DetectionConfig.crop_size`. The keypoint coordinates are normalized to [0,1] in whatever crop space the model was trained on (128x128). Denormalization uses crop_size=128 for both dimensions. **[HIGH confidence]**

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection (all code read above) — verified actual implementation state

### Secondary (MEDIUM confidence)
- Standard weighted DLT formulation — well-established in multi-view geometry literature; scale A-matrix rows by weights before SVD
- scipy.interpolate.CubicSpline — standard Python spline library; appropriate for 6-point → N-point interpolation

### Tertiary (LOW confidence)
- Per-point confidence heuristic (sigmoid distance from 0.5) — not verified; training/pose.py has no confidence head; this is a proposed convention

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already present in project
- Architecture: HIGH - all integration points verified by reading actual source code
- Pitfalls: HIGH (most) / LOW (confidence heuristic question)

**Research date:** 2026-02-28
**Valid until:** 2026-03-30 (stable codebase, no external dependencies changing)
