# 3D Fish Midline Reconstruction: Medial Axis → Triangulation → Spline Pipeline

## Problem Statement

We have a multi-view 3D fish tracking pipeline with 12 synchronized cameras viewing fish through water (refractive interface). Detection (YOLOv8) and segmentation (U-Net on cropped detections) work well but the downstream stages — differentiable mesh rendering via PyTorch3D with Adam optimization of a midline spline + swept ellipse model against silhouette IoU across all views — take 30+ minutes per second of video. We need to replace everything after segmentation with a direct reconstruction approach that avoids iterative rendering entirely, while preserving the ability to recover a continuous 3D midline curve (not just discrete keypoints). We also lack a cross-view identity solution: persistent fish tracking was originally planned to follow 3D reconstruction.

**Existing code we keep:** YOLOv8 detection, U-Net segmentation producing binary masks (variable crop size), calibrated refractive camera model with ray intersection code that handles arbitrary numbers of points and is well-optimized, and a refractive forward projection model (3D point → 2D pixel).

**What we replace:** PCA-based keypoint extraction, epipolar initialization, and the entire differentiable mesh rendering + Adam optimization loop.

---

## Pipeline

### Stage 0: Cross-View Identity Association + 3D Tracking

**Prerequisite.** All downstream stages require knowing which mask in camera A corresponds to which mask in camera B. We do not have an existing cross-view identity solution — the previously used epipolar consensus is being replaced along with the rest of the optimization pipeline.

**Input:** Per-camera sets of 2D centroids (mask centroid or bounding box center) from detection/segmentation.

**Process:**

- For each detection in each camera, cast a single refractive ray from the 2D centroid into 3D space using the existing ray model.
- Cluster rays across all cameras that intersect consistently in 3D. Use RANSAC over camera subsets: sample minimal subsets (2–3 cameras), triangulate the centroid, check reprojection error against remaining cameras, keep the consensus set. Each cluster corresponds to one physical fish, giving both the cross-view identity assignment and an initial 3D centroid position.
- **Persistent tracking across frames:** With per-frame 3D centroids established, assign persistent fish IDs using Hungarian algorithm or nearest-neighbor matching in 3D space. 3D tracking is far more reliable than per-camera 2D tracking since fish that occlude each other in one view are typically well-separated in 3D.

**Output per frame:** A mapping from (camera_id, detection_id) → fish_id, plus a 3D centroid position per fish. These feed into Stage 1 (head-tail disambiguation) and Stage 3 (which masks to triangulate together).

**Error detection:** If a fish's centroid triangulation residual is abnormally high, the identity assignment is suspect. Flag for downstream quality checks.

### Stage 1: Mask Preprocessing + 2D Medial Axis Extraction

**Input:** Binary mask per fish per camera (variable crop size, from U-Net). Cross-view identity from Stage 0.

**Process:**

- **Mask smoothing (required at current U-Net IoU ~0.62):** Apply morphological closing then opening to the binary mask before skeletonization. Use an adaptive kernel radius proportional to the mask's minor axis width (e.g., `max(3, minor_axis_width // 8)` pixels). This removes boundary noise that causes skeleton wobble without eroding thin body regions like the caudal peduncle.
- Run `skimage.morphology.skeletonize(mask)` to get a 1-pixel-wide skeleton. (`skeletonize` produces more stable, less branchy results than `medial_axis` on noisy masks.)
- Separately compute the distance transform of the *smoothed* mask via `scipy.ndimage.distance_transform_edt(mask)`. Sample it at skeleton pixel locations to get local half-width values. This decouples width estimation from skeleton extraction, giving cleaner results than `medial_axis(return_distance=True)` when masks are noisy.
- **Longest-path pruning via two-pass BFS:** Find all endpoint pixels (degree-1 nodes in the skeleton's pixel connectivity graph). Run BFS from any endpoint to find the farthest endpoint, then BFS from that endpoint to find the true farthest endpoint — the path between these two is the head-to-tail midline. Discard all other branches (fin spurs, noise artifacts). This avoids the overhead of a full graph analysis library.
- Traverse the pruned path to produce an ordered sequence of (x, y) pixel coordinates from one endpoint to the other, plus corresponding half-width values from the distance transform.

**Output per fish per camera:** Ordered array of 2D midline points `[(x₀,y₀,w₀), ..., (xₖ,yₖ,wₖ)]` where `w` is local half-width, ordered head-to-tail.

**Head-tail consistency across cameras:** Project the 3D centroid from Stage 0 (or the previous frame's 3D head position, once tracking is established) into each view to determine which skeleton endpoint is "head." On the first frame, use a width heuristic (the wider end is typically the head) or the two-start flip test from the original pipeline.

### Stage 2: Arc-Length Sampling

**Input:** Ordered 2D midline points per fish per camera (variable length per camera due to resolution/viewing angle).

**Process:**

- Compute cumulative arc length along the 2D midline in each view.
- Normalize to [0, 1] (0 = head, 1 = tail).
- Resample at N fixed normalized arc-length positions (e.g., N=15, sampling at t = 0.0, 1/14, 2/14, ..., 1.0) via linear interpolation along the 2D curve.
- Also interpolate the half-width at each sampled position.

**Output per fish per camera:** Fixed-size array of N points in 2D pixel coordinates plus half-widths, all at consistent normalized body positions. Point `i` in camera A corresponds to point `i` in camera B.

**Why this correspondence works (and when it doesn't):** Fish are slender bodies with a single dominant axis. The arc-length parameterization of the midline projection is approximately preserved across views. This is the same assumption underlying all midline-based fish tracking (Butail & Paley 2012, Voesenek et al. 2016). However, the approximation breaks down for significantly curved fish viewed from very different angles — foreshortening compresses the arc-length mapping unevenly. Cameras viewing along the fish's body axis are the worst offenders. Stage 3 includes mitigations (RANSAC, view-angle weighting) to handle this.

### Stage 3: Multi-View Triangulation

**Input:** For each of the N body positions, up to 12 corresponding 2D points (one per camera that sees this fish). Cross-view identity from Stage 0.

**Process:**

- For each fish, determine which cameras have a valid mask (from detection/segmentation + Stage 0 identity). Require ≥ 2 cameras.
- **View-angle weighting:** For each body point *i* in each camera, compute the angle between the camera's viewing ray and the local tangent direction of the 2D midline at that point (available from finite differences of adjacent skeleton points). Cameras whose viewing ray is nearly parallel to the local tangent — i.e., looking along the fish's body axis at that point — suffer the worst arc-length correspondence errors and should be downweighted or excluded for that body point.
- **RANSAC per body point:** For each of the N body positions, rather than using all visible cameras naively: sample minimal subsets (2–3 cameras), triangulate via the refractive ray intersection code, score against all other cameras by reprojection error, keep the consensus set. This rejects cameras where arc-length correspondence produced a bad mapping for that particular body point. The refractive ray code already returns residuals, so the inlier/outlier signal is free.
- Triangulate each body point using the inlier camera set from RANSAC.

**Output per fish per frame:** N 3D points ordered head-to-tail, plus N half-width values (averaged across inlier cameras), plus per-point triangulation residuals (useful for gating Stage 5).

### Stage 4: 3D Spline Fitting

**Input:** N ordered 3D points along the fish body.

**Process:**

- Fit a cubic B-spline (`scipy.interpolate.splprep` with 3D points, or explicit B-spline with 5–8 control points) through the triangulated points.
- Smoothing parameter controls the tradeoff between fidelity to the triangulated points and midline smoothness. Tune once for your setup.
- Separately fit a 1D spline to the half-width values as a function of arc length → width profile.

**Output per fish per frame:** Spline control points defining a continuous 3D midline, plus a width-profile spline. These together define a "tube model" equivalent to your mesh model's midline + cross-sections.

### Stage 5: Reprojection-Based LM Refinement (Optional — Add Only If Baseline Insufficient)

**Stages 0–4 constitute the complete baseline pipeline. Implement and evaluate them first.** Only add Stage 5 if direct triangulation with RANSAC and mask smoothing produces insufficient midline quality for your research questions. If Stage 5 is needed, start with E_reproj only and add regularization terms one at a time, each justified by a concrete failure case it addresses.

**Motivation:** The LM refinement jointly optimizes the 3D spline against all 2D observations, which makes it robust to residual arc-length correspondence errors that RANSAC doesn't catch. It finds the 3D curve that best explains all views simultaneously, rather than treating each body point independently. However, with four loss terms and three hyperparameters, there is a real risk of rebuilding a simpler version of the optimization pipeline you're escaping.

**Input:** 3D spline control points (from Stage 4, used as initialization), 2D medial axis points from all cameras (from Stage 2), refractive projection model.

**Parameters to optimize:** The spline control points (5–8 points × 3 coordinates = 15–24 scalar parameters).

**Cost function (minimize via `scipy.optimize.least_squares`, method='lm'):**

Start with E_reproj only. Add terms incrementally if needed:

```
E_reproj   = Σ_cameras Σ_i || refractive_project(spline(tᵢ), cam) - midline_2D(tᵢ, cam) ||²

--- add only if needed ---
E_smooth   = λ₁ * ∫ (dκ/ds)² · w(s) ds     [curvature gradient, weighted: higher near head]
E_temporal = λ₂ * || control_points_t - control_points_{t-1} ||²
E_width    = λ₃ * Σ_i (width_3D(tᵢ) - mean_triangulated_width(tᵢ))²
```

- `refractive_project` is your existing forward projection model (no rendering — just the geometric ray model applied to a 3D point).
- The Jacobian is analytic and sparse: each residual depends on at most a few control points (B-spline locality).
- Warm-start from Stage 4 output means this converges in 5–20 LM iterations.

**Output per fish per frame:** Refined 3D spline control points + width profile.

### Stage 6: Temporal Smoothing (Post-Processing)

**Input:** Per-frame spline control points across the full track.

**Process:**

- Apply a Kalman filter or Savitzky-Golay filter to control point trajectories over time.
- Interpolate missing frames (from detection gaps) using the temporal model.
- Extract derived quantities: centroid position, heading vector (tangent at s=0), total curvature, tail-beat frequency, etc.

---

## Performance Optimizations

- **Batch triangulations into a single vectorized call** rather than looping over N body points per fish independently. With ~15 points × ~8 fish = 120 triangulations per frame, this eliminates significant Python loop overhead.
- **Cache camera visibility maps across frames** and update incrementally rather than recomputing from scratch. Fish move continuously, so the set of cameras that see each fish changes slowly.
- **Parallelize across fish, not pipeline stages.** All stages are independent across fish within a frame. Thread or process pool over fish for near-linear scaling with fish count.
- **Gate Stage 5 on triangulation residual** (if Stage 5 is used at all). Skip refinement when the mean per-point residual from Stage 3 is below a threshold. Only refine when partial occlusion or segmentation errors produce noisy triangulations.
- **Use `skeletonize` without the distance transform** if width information isn't needed for your research questions. Skip the `distance_transform_edt` call entirely.

---

## Key Implementation Notes

- **Coordinate transforms for the 2D medial axis:** The masks are in crop-pixel space (variable size per detection). Map sampled midline points back to full-frame pixel coordinates before triangulation, using the bounding box from detection. This is just a scale + translate. The distance transform half-widths also need this scaling.
- **Refractive ray intersection code is the critical dependency.** It needs a callable interface: input = list of (camera_id, pixel_x, pixel_y) tuples for a single body point; output = 3D point + residual. Must support batched calls for performance (see Performance Optimizations).
- **The refractive forward projection model** (3D point → 2D pixel) is needed for Stage 5 and for Stage 0's reprojection scoring. This should already exist as part of your differentiable renderer — extract it as a standalone function.
- **Stage 0 reuses your existing refractive ray code** for centroid triangulation. The RANSAC logic there is the same pattern used in Stage 3 for body points — factor it out as a shared utility.
- **Head-tail ambiguity** is reduced but not eliminated. The arc-length parameterization doesn't know which end is the head. Use the 3D centroid from Stage 0 plus temporal continuity (project previous frame's head position). On the first frame, use a width heuristic or the two-start flip test from the original pipeline.
- **Epipolar-guided correspondence refinement** (future upgrade): If arc-length correspondence proves too noisy on highly curved fish despite RANSAC and view-angle weighting, a more robust approach is to use arc-length as an initial guess then refine by finding the closest point on each camera's 2D midline to the epipolar line projected from the correspondence in another camera. This is more involved to implement but essentially solves the correspondence problem geometrically.
