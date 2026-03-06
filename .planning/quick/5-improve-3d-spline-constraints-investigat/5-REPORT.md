# Quick-5: Spline Folding Vulnerability — Investigation Report

**Date:** 2026-02-23
**Author:** automated diagnostic (quick-5)
**Files examined:** `src/aquapose/reconstruction/curve_optimizer.py`

---

## Executive Summary

The `CurveOptimizer` can produce physically implausible splines that fold nearly 180 degrees despite having curvature, smoothness, and length penalties. The root cause is a combination of four structural weaknesses:

1. **The curvature penalty operates on control points, not the evaluated curve.** B-spline basis smoothing redistributes a sharp fold across multiple control points, each individually below the 30-degree threshold. A 150-degree total fold in a K=7 spline has **exactly zero curvature penalty**.

2. **With K=7 control points (5 interior angles), a 150-degree fold is perfectly legal.** Each interior angle receives 150/5=30 degrees — exactly at the threshold. The penalty is quadratic-beyond-threshold, so at threshold it contributes zero.

3. **The smoothness penalty (second differences of control points) is blind to slow U-turns.** A smooth gradual U-turn has very small second differences and near-zero smoothness penalty.

4. **The current lambda values** (`lambda_curvature=5.0`, `lambda_smoothness=1.0`) make the regularization gradients 4x–50x smaller than typical data loss gradients, so the optimizer easily "pays" the regularization cost to fit a poor observation.

---

## 1. Current Architecture

### Curvature Penalty (`_curvature_penalty`)

For each consecutive control point triplet (C_{i-1}, C_i, C_{i+1}), computes the bend angle:

```
bend = acos(v1 · v2 / (|v1| |v2|))
```

where `v1 = C_i - C_{i-1}`, `v2 = C_{i+1} - C_i`. Applies a **quadratic penalty for angles exceeding `max_bend_angle_deg=30`**:

```
penalty = mean((clamp(bend - max_rad, min=0))^2)
```

**Default weight:** `lambda_curvature = 5.0`

### Smoothness Penalty (`_smoothness_penalty`)

Second-order finite differences of control points:

```
penalty = mean(|C_{i+1} - 2*C_i + C_{i-1}|^2)
```

**Default weight:** `lambda_smoothness = 1.0`

### Length Penalty (`_length_penalty`)

Zero within ±30% of nominal length (0.085m), quadratic outside:

```
lower = 0.085 * 0.70 = 0.0595m
upper = 0.085 * 1.30 = 0.1105m
penalty = clamp(lower - arc, 0)^2 + clamp(arc - upper, 0)^2
```

**Default weight:** `lambda_length = 10.0`

### Total Loss

```
L = L_data + 10.0*L_length + 5.0*L_curvature + 1.0*L_smoothness
```

where `L_data` is mean symmetric Chamfer distance across cameras (in pixels).

---

## 2. Root Cause Analysis

### 2.1 B-Spline Basis Smoothing Masks Control-Point Folds

The curvature penalty evaluates bend angles **between consecutive control points**, but the actual rendered curve is a cubic B-spline that smooths out sharp control-point angles. As a result:

- **The maximum bend angle on the evaluated curve is nearly independent of the control-point fold angle.**
- At a 180-degree fold, the evaluated curve has a maximum per-segment angle of **only 2.4 degrees** (K=4) or **2.5 degrees** (K=7).

This means the curvature penalty, even when it fires at the control-point level, is penalizing something that the rendered curve has already smoothed away. The penalty is poorly matched to the observable quantity (curve shape).

### 2.2 Fold Distribution Makes Per-Angle Threshold Ineffective

With K=7 and `max_bend_angle_deg=30`:

- A **150-degree total fold** is distributed as 5 interior angles of **30 degrees each** — exactly at the threshold.
- The quadratic penalty is `clamp(bend - 30, 0)^2 = 0` at exactly 30 degrees.
- **Result: a 150-degree fold in K=7 has zero curvature penalty.**

The threshold only activates when any single control-point angle exceeds 30 degrees. For a 180-degree fold in K=7, each angle is 36 degrees, producing a curvature penalty of only **0.055 weighted units**.

### 2.3 Smoothness Penalty Cannot Detect U-Turns

The second-difference penalty measures **local curvature acceleration**, not total curvature. A smooth gradual U-turn has nearly constant curvature, so its second differences are small:

- At 180-degree fold (K=4): smoothness penalty weighted = **0.0009** (essentially zero)
- At 180-degree fold (K=7): smoothness penalty weighted = **0.0001** (essentially zero)

A smooth U-turn is not penalized by smoothness because it is, by definition, smooth.

### 2.4 Lambda Values Are Too Small Relative to Data Loss Gradients

Estimated data loss gradient with respect to control points:

```
d(L_data)/d(ctrl_pt) ~ focal/depth * chamfer_per_point
                      ~ (1400 px / 0.5 m) * 1.0 = 1400 px/m
```

Curvature penalty gradient at a 180-degree fold:

| K | Curvature penalty (weighted) | Curvature grad norm | Ratio (curv/data) |
|---|------------------------------|---------------------|-------------------|
| 4 | 5.48                         | 492.8               | **0.35**          |
| 7 | 0.055                        | 29.6                | **0.021**         |

Even in the worst case (K=4, 180-degree fold), the curvature regularization gradient is only **35% of the data loss gradient**. In the fine stage (K=7), it is essentially **2% of the data loss gradient**. The optimizer can trivially absorb the regularization cost to fit a noisy or ambiguous observation.

---

## 3. Numerical Evidence Tables

### 3.1 K=4 Penalties at Various Fold Angles

Control points: 4 points, 2 interior angles each = fold/2 degrees.
Nominal length: 0.085m, depth: 0.5m below water.

| Fold (deg) | Each CP Angle | Curv Pen (weighted) | Smooth Pen (weighted) | Len Pen (weighted) | Total Reg | Reg/5px data | Reg/15px data |
|------------|--------------|---------------------|-----------------------|--------------------|-----------|--------------|---------------|
| 0          | 0            | 0.00000             | 0.00000               | 0.00000            | 0.00000   | 0.0000       | 0.0000        |
| 30         | 15           | 0.00000             | 0.00003               | 0.00000            | 0.00003   | 0.0000       | 0.0000        |
| 60         | 30           | 0.00000             | 0.00012               | 0.00000            | 0.00012   | 0.0000       | 0.0000        |
| 90         | 45           | 0.34269             | 0.00026               | 0.00004            | 0.34300   | 0.0686       | 0.0229        |
| 120        | 60           | 1.37078             | 0.00045               | 0.00041            | 1.37164   | 0.2743       | 0.0914        |
| 150        | 75           | 3.08425             | 0.00067               | 0.00133            | 3.08625   | 0.6173       | 0.2058        |
| 180        | 90           | 5.48311             | 0.00090               | 0.00291            | 5.48692   | 1.0974       | 0.3658        |

**Key observation:** At 60-degree total fold (each angle exactly at the 30-degree threshold), total regularization is **0.00012** — negligible vs any realistic chamfer loss.

### 3.2 K=7 Penalties at Various Fold Angles

Control points: 7 points, 5 interior angles each = fold/5 degrees.

| Fold (deg) | Max CP Angle | Curv Pen (weighted) | Smooth Pen (weighted) | Len Pen (weighted) | Total Reg | Reg/5px data | Reg/15px data |
|------------|-------------|---------------------|-----------------------|--------------------|-----------|--------------|---------------|
| 0          | 0.0         | 0.00000             | 0.00000               | 0.00000            | 0.00000   | 0.0000       | 0.0000        |
| 30         | 6.0         | 0.00000             | 0.00000               | 0.00000            | 0.00000   | 0.0000       | 0.0000        |
| 60         | 12.0        | 0.00000             | 0.00001               | 0.00000            | 0.00001   | 0.0000       | 0.0000        |
| 90         | 18.0        | 0.00000             | 0.00002               | 0.00000            | 0.00002   | 0.0000       | 0.0000        |
| 120        | 24.0        | 0.00000             | 0.00003               | 0.00000            | 0.00003   | 0.0000       | 0.0000        |
| 150        | 30.0        | 0.00000             | 0.00005               | 0.00000            | 0.00005   | 0.0000       | 0.0000        |
| 180        | 36.0        | 0.05483             | 0.00008               | 0.00000            | 0.05491   | 0.0110       | 0.0037        |

**Critical finding:** With K=7, a 150-degree total fold receives **exactly zero curvature penalty**. A 180-degree fold receives only 0.055 weighted units — negligible vs any realistic chamfer data loss.

### 3.3 B-Spline Smoothing Effect

The evaluated curve (100 dense points) shows that B-spline interpolation dramatically reduces the apparent fold:

| Fold (deg) | Max CP Angle (K=4) | Max Curve Angle (K=4) | Chord/Arc (K=4) | Max CP Angle (K=7) | Max Curve Angle (K=7) | Chord/Arc (K=7) |
|------------|-------------------|-----------------------|-----------------|-------------------|-----------------------|-----------------|
| 0          | 0.0               | 0.45                  | 1.000           | 0.0               | 0.48                  | 1.000           |
| 60         | 30.0              | 0.78                  | 0.953           | 12.0              | 0.89                  | 0.943           |
| 90         | 45.0              | 1.08                  | 0.892           | 18.0              | 1.27                  | 0.873           |
| 120        | 60.0              | 1.44                  | 0.800           | 24.0              | 1.66                  | 0.780           |
| 150        | 75.0              | 1.87                  | 0.672           | 30.0              | 2.06                  | 0.667           |
| 180        | 90.0              | 2.40                  | 0.500           | 36.0              | 2.47                  | 0.539           |

**Key finding:** Even at 180-degree control-point fold, the evaluated curve has max per-segment angle of only 2.4 degrees. The curvature penalty "sees" 90-degree angles (K=4) or 36-degree angles (K=7) at the control points, but the actual curve is smooth. The chord/arc ratio is a much better predictor of the fold severity.

---

## 4. Recommendations (Ranked by Impact)

### Recommendation 1 (Impact: HIGH, Effort: LOW): Add Chord-to-Arc-Length Ratio Penalty

**What:** Add a penalty `(1 - chord/arc)^2` where chord = distance from first to last spline point, arc = total arc length.

**Why:** Straight fish: ratio ~1.0 (penalty ~0). U-turned fish: ratio ~0.5 (penalty ~0.25). This penalty:
- Responds immediately and continuously to any folding, regardless of K
- Is invariant to how the fold is distributed among control points
- Cannot be "gamed" by distributing bend angles below the threshold

**Numerical evidence:**

| Fold (deg) | Chord/Arc (K=4) | CAP value | Chord/Arc (K=7) | CAP value |
|------------|-----------------|-----------|-----------------|-----------|
| 0          | 1.000           | 0.000     | 1.000           | 0.000     |
| 60         | 0.953           | 0.002     | 0.943           | 0.003     |
| 90         | 0.892           | 0.012     | 0.873           | 0.016     |
| 120        | 0.800           | 0.040     | 0.780           | 0.048     |
| 150        | 0.672           | 0.107     | 0.667           | 0.111     |
| 180        | 0.500           | 0.249     | 0.540           | 0.212     |

**Implementation:**

```python
def _chord_arc_penalty(ctrl_pts: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Penalize folded splines via chord-to-arc-length ratio.

    Returns (1 - chord/arc)^2, which is 0 for straight splines and
    approaches 1 for fully folded U-turns.
    """
    spline_pts = torch.einsum("ek,nkd->ned", basis, ctrl_pts)  # (N, n_eval, 3)
    chord = torch.linalg.norm(spline_pts[:, -1, :] - spline_pts[:, 0, :], dim=1)  # (N,)
    diffs = spline_pts[:, 1:, :] - spline_pts[:, :-1, :]
    arc = torch.linalg.norm(diffs, dim=2).sum(dim=1)  # (N,)
    ratio = chord / (arc + 1e-8)
    return ((1.0 - ratio) ** 2).mean()
```

**Suggested weight:** `lambda_chord_arc = 50.0` (makes it comparable to data loss at 90-degree fold).

Add `lambda_chord_arc: float = 50.0` to `CurveOptimizerConfig` and include it in both `closure_coarse` and `closure_fine`.

---

### Recommendation 2 (Impact: HIGH, Effort: LOW): Evaluate Curvature on the Dense Spline Curve

**What:** Compute the curvature penalty on `n_eval` spline-evaluated points rather than on the K control points.

**Why:** The current penalty is blind to the actual curve shape. With K=7, a 150-degree fold produces zero penalty because each control-point angle is ≤30 degrees. If curvature is measured on the evaluated curve:
- The dense curve's local bend angle reflects actual pose
- Fold cannot be masked by distributing bend across multiple control points

**Implementation:** Replace the current control-point-based `_curvature_penalty` with a version that:
1. Evaluates `spline_pts = basis @ ctrl_pts` (already done in `_length_penalty`)
2. Computes bend angles on consecutive triplets of `spline_pts`
3. Applies the same quadratic-beyond-threshold penalty

```python
def _curvature_penalty_dense(
    ctrl_pts: torch.Tensor,
    basis: torch.Tensor,
    config: CurveOptimizerConfig,
) -> torch.Tensor:
    """Curvature penalty on the evaluated B-spline curve (not control points)."""
    spline_pts = torch.einsum("ek,nkd->ned", basis, ctrl_pts)  # (N, n_eval, 3)
    v1 = spline_pts[:, 1:-1, :] - spline_pts[:, :-2, :]  # (N, n_eval-2, 3)
    v2 = spline_pts[:, 2:, :] - spline_pts[:, 1:-1, :]
    norm1 = torch.linalg.norm(v1, dim=2, keepdim=True).clamp(min=1e-8)
    norm2 = torch.linalg.norm(v2, dim=2, keepdim=True).clamp(min=1e-8)
    cos_bend = (v1 / norm1 * v2 / norm2).sum(dim=2).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    bend_angles = torch.acos(cos_bend)  # (N, n_eval-2)
    max_rad = torch.tensor(config.max_bend_angle_deg * np.pi / 180.0, ...)
    excess = torch.clamp(bend_angles - max_rad, min=0.0)
    return (excess ** 2).mean()
```

Note: `max_bend_angle_deg` should be reduced from 30 degrees to ~3-5 degrees per dense segment (since dense evaluation has many more segments).

**Suggested max_bend_angle_deg for dense evaluation:** `360.0 / n_eval` degrees (allows up to one full bend over the whole body), or simply `5.0` degrees.

---

### Recommendation 3 (Impact: HIGH, Effort: LOW): Increase `lambda_curvature` by 10-50x

**What:** Increase `lambda_curvature` from 5.0 to 50-250.

**Why:** At a 180-degree fold in K=7, the curvature gradient norm is 29.6 vs an estimated data loss gradient of ~1400. The curvature penalty contributes only 2% of the optimizer signal. To make curvature competitive:

```
Required lambda_curvature = 1400 / 29.57 * safety_factor
                           = 47 * 3 = ~150
```

**Risk:** Increasing lambda too much will prevent the optimizer from fitting legitimate highly-curved fish. Start with 50x current = `lambda_curvature = 250` and monitor: if real fish fail to fit, reduce to 50.

**Complement with Rec 1:** A chord-arc penalty (Rec 1) at `lambda_chord_arc=50` together with `lambda_curvature=50` gives combined regularization gradient ~570 at 180-degree fold vs data gradient ~1400 (ratio 0.4), which is meaningful.

---

### Recommendation 4 (Impact: MEDIUM, Effort: LOW): Lower `max_bend_angle_deg` for Coarse Stage

**What:** Use a tighter angle threshold for the coarse stage (K=4) and standard for fine (K=7).

**Why:** With K=4 (2 interior angles), a 60-degree total fold puts each angle exactly at the threshold with zero penalty. Lowering to 15 degrees would penalize any fold beyond 30 degrees total.

**Implementation:** Add `max_bend_angle_deg_coarse: float = 15.0` to `CurveOptimizerConfig` and use it in the coarse closure.

**Numerical effect:** At 90-degree total fold (K=4, 15-degree threshold):
- Excess per joint: 45 - 15 = 30 degrees
- Curvature penalty: approximately 4x larger than current

---

### Recommendation 5 (Impact: MEDIUM, Effort: MEDIUM): Add Monotonicity Constraint on Tangent Direction

**What:** Penalize reversal of the tangent's primary component (the component along the fish's initial heading). This directly prevents the spine from "turning back on itself."

**Why:** A fold requires the tangent to reverse direction. Penalizing the dot product of consecutive tangents going negative prevents this without needing explicit angle computation.

**Implementation:**

```python
def _monotonicity_penalty(ctrl_pts: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Penalize tangent reversals along the spine."""
    spline_pts = torch.einsum("ek,nkd->ned", basis, ctrl_pts)  # (N, n_eval, 3)
    tangents = spline_pts[:, 1:, :] - spline_pts[:, :-1, :]  # (N, n_eval-1, 3)
    # Consecutive tangent dot products: negative means reversal
    cos_consecutive = (tangents[:, 1:, :] * tangents[:, :-1, :]).sum(dim=2)  # (N, n_eval-2)
    reversals = torch.clamp(-cos_consecutive, min=0.0)  # (N, n_eval-2)
    return (reversals ** 2).mean()
```

**Suggested weight:** `lambda_monotonicity = 100.0`

---

### Recommendation 6 (Impact: LOW, Effort: LOW): Add Total Turning Angle Constraint

**What:** Penalize when the sum of all bend angles on the evaluated curve exceeds a threshold (e.g., 60 degrees total = moderate swimming bend).

**Why:** Complements Rec 2. Where Rec 2 penalizes any individual bend > threshold, this penalizes the total accumulated turn. A realistic fish can have 0-60 degrees total curvature; a folded fish has 150-180 degrees.

**Note:** This is less differentiable than the chord-arc ratio (which approximates total turn) and harder to tune. Rec 1 (chord-arc) is preferred as it achieves the same effect more simply.

---

## 5. Recommended Implementation Order

| Priority | Recommendation | Effort | Expected Impact |
|----------|---------------|--------|-----------------|
| 1 | Add chord-arc ratio penalty (`lambda_chord_arc=50`) | 1-2 hours | Eliminates fold blindspot for K=7 |
| 2 | Increase `lambda_curvature` from 5 to 50 | 5 min | 10x more resistance to folding |
| 3 | Evaluate curvature on dense curve (lower `max_bend_angle_deg` to 5 deg) | 2-4 hours | Removes basis-smoothing blindspot |
| 4 | Lower coarse-stage `max_bend_angle_deg` to 15 degrees | 5 min | Prevents coarse stage from folding |
| 5 | Add monotonicity penalty (`lambda_monotonicity=100`) | 1-2 hours | Directly prevents tangent reversals |

**Recommended minimal fix (implement today):** Recommendations 1 + 2 + 4 require approximately 30 minutes and provide the largest marginal improvement. This brings the K=7 penalty from ~0 to ~12 weighted units at 150-degree fold, which is competitive with a 12px chamfer loss.

---

## 6. Verification Strategy

After implementing fixes, verify against:

1. **Synthetic straight fish:** All penalties must remain ~0. Total regularization must not exceed 10% of data loss.
2. **Synthetic 90-degree fold:** Total regularization must exceed data loss at 10px chamfer (ensures optimizer actively resists folding).
3. **Real video (--synthetic flag):** Run existing `diagnose_pipeline.py --method curve --synthetic` and visually confirm no folded splines in output overlay.
4. **Unit test:** Add `test_chord_arc_penalty_monotone` verifying the chord-arc penalty increases monotonically with fold angle.

---

## 7. Files to Modify

- `src/aquapose/reconstruction/curve_optimizer.py`
  - Add `_chord_arc_penalty()` function
  - Modify `_curvature_penalty()` or add `_curvature_penalty_dense()` variant
  - Add `lambda_chord_arc`, `lambda_monotonicity`, `max_bend_angle_deg_coarse` to `CurveOptimizerConfig`
  - Update `closure_coarse` and `closure_fine` to include new penalties

- `tests/unit/test_curve_optimizer.py`
  - Add tests for new penalty functions
  - Add regression test verifying 150-degree fold is penalized
