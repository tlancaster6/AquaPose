# 3D Fish Midline Reconstruction: Medial Axis → Triangulation → Spline Pipeline

## Problem Statement

We have a multi-view 3D fish tracking pipeline with 12 synchronized cameras viewing fish through water (refractive interface). Detection (YOLOv8) and segmentation (U-Net on crops) work well but the downstream stages — differentiable mesh rendering via PyTorch3D with Adam optimization of a midline spline + swept ellipse model against silhouette IoU across all views — take 30+ minutes per second of data. We need to replace the reconstruction phase with a direct approach that avoids iterative rendering entirely, while preserving the ability to recover a continuous 3D midline curve (not just discrete keypoints).

**Existing code we keep:** YOLOv8 detection, U-Net segmentation, calibrated refractive camera model with ray intersection code that handles arbitrary numbers of points and is well-optimized.

**What we replace:** PCA-based keypoint extraction, epipolar initialization, and the entire differentiable mesh rendering + Adam optimization loop.

---

## Pipeline: Mask → 2D Medial Axis → Multi-View Triangulation → 3D Spline + LM Refinement

### Stage 1: 2D Medial Axis Extraction

**Input:** Binary mask per fish per camera (from U-Net).

**Process:**

- Run `skimage.morphology.medial_axis(mask, return_distance=True)` to get both the 1-pixel skeleton and the distance transform (local half-width at each skeleton pixel).
- Convert skeleton to a graph using `skan.csr.skeleton_to_csgraph`. Identify endpoints (degree-1 nodes) and junctions (degree-3+ nodes).
- Find the longest geodesic path between any pair of endpoints along the skeleton graph. This is the head-to-tail midline. Discard all other branches (fin spurs, noise artifacts).
- Traverse the pruned path to produce an ordered sequence of (x, y) pixel coordinates from one endpoint to the other, plus corresponding half-width values from the distance transform.

**Output per fish per camera:** Ordered array of 2D midline points `[(x₀,y₀,w₀), ..., (xₖ,yₖ,wₖ)]` where `w` is local half-width, ordered head-to-tail.

**Head-tail consistency across cameras:** Use the previous frame's 3D head position projected into each view to determine which endpoint is "head." On the first frame, use the epipolar consensus logic you already have (or a simple heuristic: the wider end is typically the head).

### Stage 2: Arc-Length Sampling

**Input:** Ordered 2D midline points per fish per camera (variable length per camera due to resolution/viewing angle).

**Process:**

- Compute cumulative arc length along the 2D midline in each view.
- Normalize to [0, 1] (0 = head, 1 = tail).
- Resample at N fixed normalized arc-length positions (e.g., N=15, sampling at t = 0.0, 1/14, 2/14, ..., 1.0) via linear interpolation along the 2D curve.
- Also interpolate the half-width at each sampled position.

**Output per fish per camera:** Fixed-size array of N points in 2D pixel coordinates plus half-widths, all at consistent normalized body positions. Point `i` in camera A corresponds to point `i` in camera B.

**Why this correspondence works:** Fish are slender bodies with a single dominant axis. The arc-length parameterization of the midline projection is approximately preserved across views. This is the same assumption underlying all midline-based fish tracking (Butail & Paley 2012, Voesenek et al. 2016).

### Stage 3: Multi-View Triangulation

**Input:** For each of the N body positions, multiple corresponding 2D points (one per camera that sees this fish).

**Process:**

- For each fish, determine which cameras have a valid mask (from the detection/segmentation stages). Require ≥ 2 cameras.
- For each of the N body positions independently: feed the 2D pixel coordinates from all visible cameras into the existing refractive ray intersection code → one 3D point.
- Optionally filter by triangulation residual: if a single camera contributes an outlier (e.g., from a segmentation error), exclude it and re-triangulate.

**Output per fish per frame:** N 3D points ordered head-to-tail, plus N half-width values (averaged across cameras or triangulated separately).

### Stage 4: 3D Spline Fitting

**Input:** N ordered 3D points along the fish body.

**Process:**

- Fit a cubic B-spline (`scipy.interpolate.splprep` with 3D points, or explicit B-spline with 5–8 control points) through the triangulated points.
- Smoothing parameter controls the tradeoff between fidelity to the triangulated points and midline smoothness. Tune once for your setup.
- Separately fit a 1D spline to the half-width values as a function of arc length → width profile.

**Output per fish per frame:** Spline control points defining a continuous 3D midline, plus a width-profile spline. These together define a "tube model" equivalent to your mesh model's midline + cross-sections.

### Stage 5: Reprojection-Based LM Refinement (Optional but Recommended)

**Input:** 3D spline control points (from Stage 4, used as initialization), 2D medial axis points from all cameras (from Stage 2), refractive projection model.

**Parameters to optimize:** The spline control points (5–8 points × 3 coordinates = 15–24 scalar parameters).

**Cost function (minimize via `scipy.optimize.least_squares`, method='lm'):**

```
E_reproj   = Σ_cameras Σ_i || refractive_project(spline(tᵢ), cam) - midline_2D(tᵢ, cam) ||²
E_smooth   = λ₁ * ∫ (dκ/ds)² · w(s) ds     [curvature gradient, weighted: higher near head]
E_temporal = λ₂ * || control_points_t - control_points_{t-1} ||²
E_width    = λ₃ * Σ_i (width_3D(tᵢ) - mean_triangulated_width(tᵢ))²

E_total = E_reproj + E_smooth + E_temporal + E_width
```

- `refractive_project` is your existing forward projection model (no rendering — just the geometric ray model applied to a 3D point).
- The Jacobian is analytic and sparse: each residual depends on at most a few control points (B-spline locality).
- Warm-start from Stage 4 output means this converges in 5–20 LM iterations.

**Output per fish per frame:** Refined 3D spline control points + width profile.

### Stage 6: Temporal Smoothing (Post-Processing)

**Input:** Per-frame spline control points across the full track.

**Process:**

- Apply a Kalman filter or Savitzky-Golay filter to control point trajectories over time.
- Alternatively, if you included E_temporal in Stage 5, this may be unnecessary.
- Interpolate missing frames (from detection gaps) using the temporal model.
- Extract derived quantities: centroid position, heading vector (tangent at s=0), total curvature, tail-beat frequency, etc.

---

## Performance Optimizations

- **Batch triangulations into a single vectorized call** rather than looping over N body points per fish independently. With ~15 points × ~8 fish = 120 triangulations per frame, this eliminates significant Python loop overhead.
- **Replace `skan` with a simple two-pass BFS** to find the longest path through the skeleton. Fish masks produce near-linear skeletons with at most a few short spurs — a full graph library is overkill and adds a dependency with numba warmup costs.
- **Cache camera visibility maps across frames** and update incrementally rather than recomputing from scratch. Fish move continuously, so the set of cameras that see each fish changes slowly.
- **Parallelize across fish, not pipeline stages.** All stages are independent across fish within a frame. Thread or process pool over fish for near-linear scaling with fish count.
- **Gate the LM refinement on triangulation residual.** Skip Stage 5 when the direct triangulation residual is already low — likely 70–80% of frames. Only refine when partial occlusion or segmentation errors produce noisy triangulations.
- **Use `skeletonize` instead of `medial_axis`** if width information isn't needed for your research questions. It skips the distance transform computation.

---

## Key Implementation Notes

- **Coordinate transforms for the 2D medial axis:** The masks are in 256×256 crop space. You need to map sampled midline points back to full-frame pixel coordinates before triangulation, using the bounding box from detection. This is just a scale + translate.
- **The distance transform half-widths are also in crop-pixel units** and need the same scaling.
- **Your refractive ray intersection code is the critical dependency.** It needs a callable interface: input = list of (camera_id, pixel_x, pixel_y) tuples for a single body point; output = 3D point + residual.
- **The refractive projection model** (3D point → 2D pixel, the forward direction) is needed for Stage 5. This should already exist as part of your differentiable renderer — you just need it as a standalone function.
- **`skan` install:** `pip install skan`. Dependencies: numpy, scipy, scikit-image, numba, pandas.
- **Head-tail ambiguity** is reduced but not eliminated by this approach. The arc-length parameterization doesn't know which end is the head. Use temporal continuity (project previous frame's head position) or your existing two-start flip test on the first frame only.
