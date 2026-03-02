# Proposal: Triangulation Reconstruction Rebuild

> Outside consultant proposal for rebuilding the 3D reconstruction pipeline,
> starting with a stripped-down triangulation backend and an evaluation harness
> to measure every subsequent change.

---

## Motivation

The current reconstruction system has two backends (triangulation and curve
optimizer) with ~3000 combined lines of code, different output characteristics,
and enough accumulated complexity that the contribution of individual steps
cannot be measured. The goal is to rebuild from a clean, minimal triangulation
baseline where every piece earns its place empirically.

## Strategic Decisions

1. **Start with triangulation, not curve optimization.** Triangulation is
   stateless, debuggable, and each step's contribution can be measured
   independently. The curve optimizer is deferred — it must demonstrably beat
   the triangulation baseline on the evaluation harness to justify its
   complexity.

2. **Pose estimation backend only.** The segmentation midline backend is
   deferred. Pose estimation provides anatomically ordered keypoints with
   confidence scores, which eliminates the need for correspondence and
   orientation machinery in reconstruction.

3. **Orientation lives in the midline stage.** Reconstruction trusts that input
   midlines are correctly oriented (head-to-tail). Orientation alignment is not
   reintroduced into reconstruction unless evaluation demonstrates a need.

4. **Half-widths are pass-through only.** Populated from upstream if available,
   filled with defaults if not. Reconstruction does not use half-widths for any
   decisions, weighting, or rejection logic.

---

## Triangulation v1 Design

### Input

`MidlineSet` — `fish_id → camera_id → Midline2D` from the pose estimation
backend. Correspondence is solved (ordered keypoints). Orientation is solved
upstream.

### Step 1: Per-body-point triangulation

For each of the N body points (default 15):

1. Gather all cameras that observe this point (non-NaN, non-zero confidence if
   available).
2. Triangulate using all available cameras via DLT normal equations. Use
   confidence-weighted DLT when `point_confidence` is available, unweighted
   otherwise.
3. Compute per-camera reprojection residuals.
4. Reject cameras with residuals beyond a threshold of the median residual (the
   specific threshold — e.g., median + 2σ or a fixed multiple — is determined
   during evaluation).
5. Re-triangulate with inlier cameras only.
6. Reject points above the water surface (Z ≤ water_z).

**One strategy regardless of camera count.** No branching at 2, 3-7, or 8+
cameras. With fewer cameras there is less redundancy for outlier rejection;
the response is to flag low confidence, not to switch algorithms.

### Step 2: B-spline fitting

1. Arc-length parameterization using original body-point indices:
   `u_i = i / (N - 1)` for valid (successfully triangulated) indices only.
2. Require a minimum number of valid points (default: `n_control_points + 2 =
   9`). Fail the fish if below threshold.
3. Fit cubic B-spline via `scipy.interpolate.make_lsq_spline` with 7 control
   points and clamped uniform knots.

### Step 3: Output assembly

- **Arc length:** Sum of segment lengths over dense spline evaluation (1000
  points).
- **Reprojection residuals:** Project spline samples back into each observing
  camera, compute mean point-to-point distance against input 2D midline.
  Standardized metric used by both reconstruction and evaluation.
- **Half-widths:** Pass through from upstream if present. Default values if not.
  Not used in reconstruction logic.
- **Confidence flag:** `is_low_confidence = True` when a configurable fraction
  of body points had fewer than 3 inlier cameras.

### Dropped from current triangulation backend

| Component | Reason |
|-----------|--------|
| Orientation alignment | Moved upstream to midline stage |
| Epipolar refinement | Expensive, fragile, solves a problem that ordered keypoints already address. Revisit only if evaluation shows correspondence errors are significant. |
| Camera-count branching (2 / 3-7 / 8+) | Replaced by single unified strategy |
| `refine_midline_lm` stub | Dead code — delete |
| Per-camera half-width conversion with single-camera focal length | Half-widths are pass-through; no reconstruction-side conversion needed for v1 |

---

## Evaluation Harness

### Principles

- Measure before and after every change.
- No full manual 3D annotation — the cost is prohibitive and the data would
  have low reuse value.
- Require calibration data (integration-level tests), not CI-portable unit
  tests.

### Test Fixtures

Frozen `MidlineSet` snapshots paired with corresponding calibration data.

**Source data:** 5-minute, 12-camera video of 9 schooling fish. Challenging
conditions (tight schooling, frequent tank-edge proximity) representative of
real operating scenarios.

**Fixture generation:** Run the existing pipeline (detection → tracking →
association → midline extraction) on a single chunk of manageable length —
push chunk duration until cross-camera association begins to degrade, then
sample from within that window. This avoids blocking on chunking infrastructure
while producing clean `MidlineSet` data.

**Frame selection (15-20 frames):** Select frames with diversity across the
axes that stress reconstruction differently:

- **Camera coverage per fish:** Include frames where most fish have 10+
  cameras and frames where some fish have only 3-5 (tank edges, occlusion).
- **Inter-fish distance:** Include both spread-out and tightly-packed
  configurations.
- **Posture:** Include straight fish and fish mid-turn with high curvature.
- **Temporal spacing:** Enforce minimum ~2-3 second gap between selected
  frames to avoid redundancy from inter-frame similarity.

Selection can be guided by pipeline metadata (per-fish camera counts, pairwise
centroid distances) or done by visual inspection. Additional fixtures can be
added later as chunking infrastructure matures and harder scenarios become
accessible.

### Tier 1: Reprojection self-consistency

**Cost:** Zero annotation. Automatic.

For each fish, each camera: project the reconstructed 3D spline back to 2D,
compute mean point-to-point distance against the input 2D midline. Report
per-fish mean, per-fish max, and overall aggregates.

This does not prove correctness (depth ambiguity), but detects regressions.

### Tier 2: Leave-one-out camera stability

**Cost:** Zero annotation. Automatic. ~N× the cost of Tier 1.

For each fish: reconstruct with all cameras, then reconstruct N times dropping
one camera each time. Report max control-point displacement across dropout
runs.

High displacement flags depth ambiguity or over-reliance on a single view.
Directly probes the weakness that reprojection error alone cannot detect.

### Tier 3: Synthetic ground truth (deferred)

**Cost:** One-time engineering, requires calibration data only.

Generate known 3D splines, project through the real calibration model, add
noise, reconstruct, measure 3D error. Validates geometric correctness of the
pipeline in isolation. Good candidate for lightweight unit tests once the
prototype is working — synthetic inputs can be precomputed and checked in
without requiring calibration at test time.

### Test integration

- Tier 1 and 2 are integration tests with the `@slow` marker, run via
  `hatch run test-all`.
- Output is a human-readable summary table and a machine-diffable regression
  gate.
- First run against the current triangulation backend establishes the baseline.

---

## What's Deferred

| Item | Status |
|------|--------|
| Curve optimizer rebuild | Must beat triangulation baseline to justify reintroduction |
| Segmentation midline backend support | Revisit after triangulation v1 is solid |
| Epipolar refinement | Reintroduce only if evaluation shows correspondence errors matter |
| Orientation logic in reconstruction | Reintroduce only if upstream orientation proves unreliable |
| Tier 3 synthetic evaluation | Build once prototype is working; good source of unit tests |
| Half-width reconstruction logic | Revisit if downstream consumers need accurate 3D half-widths |

---

## Suggested Execution Order

1. **Build evaluation harness** — fixture capture, Tier 1 and 2 metrics,
   summary output. Run against current backend to establish baseline.
2. **Implement triangulation v1** — stripped-down pipeline as described above.
   Run evaluation, compare to baseline.
3. **Tune outlier rejection threshold** — use evaluation harness to find the
   right residual threshold empirically.
4. **Delete dead code** — `refine_midline_lm` stub, unused orientation/epipolar
   code paths (after v1 is validated).
5. **Assess curve optimizer** — decide whether to rebuild based on where
   triangulation v1 falls short.
