# Reconstruction Backends: Detailed Step-by-Step Analysis

> Fact-finding report for high-level review of the triangulation and curve
> optimization reconstruction pathways. Covers input data flow from the
> midline stage through final Midline3D output.

---

## 1. Input: What Reconstruction Receives

Both backends receive the same input structure: a **MidlineSet** — a nested dict
`fish_id → camera_id → Midline2D`.

### Midline2D (defined in `reconstruction/midline.py`)

| Field | Shape / Type | Source |
|-------|-------------|--------|
| `points` | `(N, 2)` float32 | Full-frame pixel coords (x, y) |
| `half_widths` | `(N,)` float32 | Distance-transform half-widths in full-frame px |
| `fish_id` | int | From tracklet group |
| `camera_id` | str | Camera identifier |
| `frame_index` | int | Frame number |
| `is_head_to_tail` | bool | Orientation flag (may be False if unresolved) |
| `point_confidence` | `(N,)` float32 or None | Per-point confidence; None = uniform |

**N = 15 by default** (configurable via `n_points`).

### How Midline2D is produced (two backends)

**Segmentation backend** (`core/midline/backends/segmentation.py`):
1. Crop detection from undistorted frame using OBB region
2. Run U-Net to produce binary mask (128×64)
3. Morphological smoothing (adaptive closing+opening by fish minor-axis)
4. Largest connected component only
5. Skeletonize → distance transform
6. Two-pass BFS to find longest path (endpoint-to-endpoint)
7. Arc-length resample to N evenly-spaced points
8. Transform crop-space → full-frame coordinates
9. `point_confidence` = None (uniform)

**Pose estimation backend** (`core/midline/backends/pose_estimation.py`):
1. Crop detection from undistorted frame using OBB region
2. Run YOLO-pose to get K anatomical keypoints with confidence
3. Interpolate K keypoints to N points along the midline using configured t-values
4. `point_confidence` populated from model confidence scores

### Orientation resolution (Stage 4 post-processing)

When tracklet groups are available and the segmentation backend is used:
- Cross-camera geometric scoring via ForwardLUTs
- Velocity alignment signal
- Temporal prior (consistency with previous frame)
- Weighted vote determines head-to-tail direction
- Points flipped if needed before passing to reconstruction

For pose_estimation backend, orientation is inherent (nose→tail keypoint order).

---

## 2. Shared Entry Point: ReconstructionStage

File: `core/reconstruction/stage.py`

### Primary path (`_run_with_tracklet_groups`):

1. **For each TrackletGroup (fish):**
   - Build per-frame camera membership from tracklet `frames` and `frame_status`
   - Only use frames where `frame_status == "detected"` (skip coasted frames)

2. **For each frame:**
   - Check if `len(cameras) >= min_cameras` (default 3); drop if not
   - For each camera in the membership, look up the matching AnnotatedDetection by centroid proximity (tolerance 10px)
   - Extract `Midline2D` from the AnnotatedDetection
   - Re-check camera count after midline lookup (some may lack midlines)
   - Build `midline_set = {fish_id: {cam_id: Midline2D, ...}}`
   - Call `backend.reconstruct_frame(frame_idx, midline_set)`

3. **Gap interpolation per fish:**
   - Scan for consecutive missing frames bounded by valid frames
   - If gap ≤ `max_interp_gap` (default 5): linear interpolation of control points
   - Interpolated frames: `confidence=0`, `is_low_confidence=True`

4. **Assemble frame-major output:** `list[dict[fish_id, Midline3D]]`

### Legacy path (`_run_legacy`):
- No tracklet groups — assigns sequential fish IDs per camera per frame
- Each detection becomes a separate "fish" (no cross-camera identity)
- No gap interpolation

---

## 3. Backend A: Triangulation

File: `reconstruction/triangulation.py` (~1085 lines)

### Overview

Point-wise approach: triangulate each of the N body points independently from
multi-camera 2D observations, then fit a B-spline through the triangulated 3D
points.

### Step-by-step for `triangulate_midlines()`:

#### Step 1: Orientation alignment (`_align_midline_orientations`)
- **Purpose:** Ensure body point `i` refers to the same physical location across cameras
- Fix first camera (sorted order) as reference
- For each remaining camera, try both orientations (original and flipped)
- Triangulate 3 sample points (head, mid, tail) against reference for each orientation
- Pick orientation with shorter total chord length (smooth curve = correct alignment)
- **Complexity:** O(N_cameras) pairwise chord comparisons × 3 sample points each

#### Step 2: Epipolar correspondence refinement (`_refine_correspondences_epipolar`)
- **Purpose:** Snap each target camera's body points to epipolar-consistent positions
- Select reference camera = longest 2D arc length (least foreshortened)
- For each of the N body points on the reference camera:
  - Cast ray from reference pixel, sample 100 depths in [0.01, 2.0] metres
  - Project 3D samples into target camera → epipolar curve (up to 100 points)
  - For each target skeleton point, compute minimum distance to epipolar curve
  - Snap to nearest skeleton point if within `snap_threshold` (default 20px)
  - Points beyond threshold → NaN (excluded from triangulation)
- **Output:** Refined Midline2D per target camera; reference camera unchanged

#### Step 3: Per-body-point triangulation (`_triangulate_body_point`)
Dispatches based on camera count per body point:

**2 cameras:**
- Check ray angle ≥ 5° (skip near-parallel pairs)
- Triangulate single pair via DLT normal equations
- Reject if Z ≤ water_z (above water surface)

**3–7 cameras (exhaustive pairwise):**
- Try all C(N,2) camera pairs
- For each pair: check ray angle, triangulate, reject above-water
- Score by max reprojection error on held-out cameras
- Keep pair with lowest max held-out error
- Re-triangulate using ALL cameras within `inlier_threshold` (default 50px) of seed
- Final Z validation

**8+ cameras (residual rejection):**
- Triangulate using all cameras at once
- Compute per-camera reprojection residuals
- Reject cameras with residual > median + 2σ
- Re-triangulate with inliers only
- Final Z validation

**Weighted variant:** When `point_confidence` is available, uses `_weighted_triangulate_rays` (confidence-weighted DLT normal equations) instead of unweighted `triangulate_rays`.

#### Step 4: Layer 3 depth validation
- If `max_depth` is configured, reject points with Z > water_z + max_depth
- Catches deep outliers that survived Layer 1 (water surface) and Layer 2 (ray angle)

#### Step 5: B-spline fitting (`_fit_spline`)
- Arc-length parameterization: `u_i = original_index / (N-1)` for valid indices only
  - **Important:** uses original body-point indices, not renormalized positions
- Require ≥ `min_body_points` (default 9 = n_control_points + 2) valid triangulated points
- Fit cubic B-spline via `scipy.interpolate.make_lsq_spline`
  - Clamped uniform knot vector (4 copies at each end, 3 interior)
  - Default 7 control points → 11 knots
- Arc length: sum of 999 segment lengths over 1000 evaluation points

#### Step 6: Half-width conversion
- For each valid body point: convert pixel half-width to metres via pinhole approx
  - `hw_m = hw_px * depth_m / focal_px`
  - Uses focal length from first inlier camera
- Interpolate to all N body positions via `scipy.interpolate.interp1d`

#### Step 7: Spline reprojection residuals
- Evaluate fitted spline at N uniformly-spaced parameters
- Project all N 3D points into every observing camera
- Compare against observed 2D midline points
- Compute mean and max residual, per-camera residuals

#### Step 8: Confidence flagging
- `is_low_confidence = True` when >20% of body points had <3 inlier cameras

### Output: `Midline3D`
- `control_points`: (7, 3) float32
- `knots`: (11,) float32
- `degree`: 3
- `arc_length`: metres
- `half_widths`: (N,) float32 in metres
- `mean_residual`, `max_residual`: pixels (spline-based)
- `per_camera_residuals`: dict

---

## 4. Backend B: Curve Optimizer

File: `reconstruction/curve_optimizer.py` (~1883 lines)

### Overview

Correspondence-free approach: directly optimize 3D B-spline control points to
minimize chamfer distance between reprojected spline and observed 2D skeletons.
All fish batched into a single tensor and optimized in parallel on GPU.

### Step-by-step for `CurveOptimizer.optimize_midlines()`:

#### Step 1: Prepare observation tensors
- For each fish: gather per-camera 2D skeleton points, filter NaN rows
- Extract sqrt(confidence) weights when available
- Move everything to GPU

#### Step 2: Estimate reference orientations
- For each fish, pick camera with most observation points
- PCA on 2D skeleton → principal axis → 3D direction (z=0)

#### Step 3: Estimate 3D centroids
- If not provided: for each fish, cast rays from each camera's mean 2D point
- Multi-camera: triangulate ray intersection → centroid
- Single camera: cast ray at estimated depth (0.5m fallback)

#### Step 4: Triangulation-seeded cold start (optional, `tri_seed=False` by default)
- Run full `triangulate_midlines()` for cold-start fish
- Validate: majority of control points must be below water surface
- Evaluate initial chamfer loss; reject seeds > 100px
- **Disabled by default** with documented rationale: "triangulation pipeline rarely
  succeeds and a single bad seed can poison the shared L-BFGS Hessian"

#### Step 5: Coarse optimization (K=4 control points)

**Initialize:**
- Warm-start fish: use previous frame's control points (detach+clone), upsample if K differs
- Cold-start fish: straight line of `nominal_length_m` (default 0.085m) centered at centroid, oriented along PCA axis

**Loss function** (total = data + 5 regularizers):

| Term | Weight (default) | Description |
|------|---------|-------------|
| Data loss | 1.0 (implicit) | Mean symmetric chamfer distance (projected spline ↔ observed skeleton), averaged over cameras then fish. In pixel units. When no valid projections exist, uses depth penalty (100× distance above water). |
| Length penalty | λ=10.0 | Quadratic penalty on arc length outside [nominal×(1-tol), nominal×(1+tol)]. Default band: [0.0595m, 0.1105m]. |
| Curvature penalty | λ=10.0 | Quadratic penalty on inter-control-point bend angles exceeding 30°. |
| Chord-arc penalty | λ=10.0 | Quadratic penalty when chord/arc ratio drops below 0.75 (folded spline). |
| Z-variance penalty | λ=10.0 | Penalizes depth spread of spline around per-fish mean Z. |
| Smoothness penalty | λ=1.0 | Second-difference penalty on control points. |

All regularizers are **normalized** by reference-scale denominators (nominal_length², max_bend², threshold²) so lambdas express pixel-equivalent importance.

**Optimizer:** L-BFGS with strong Wolfe line search, `max_iter=20`, `history_size=10`

**Two constraint enforcement modes:**
- **Penalty-based (default, `use_alm=False`):** Single L-BFGS pass with all penalties in the loss
- **ALM (optional, `use_alm=True`):** Augmented Lagrangian outer loop (5 iters) wrapping L-BFGS inner loop, with explicit constraint violations for arc length, curvature, and chord-arc ratio

**Adaptive early stopping:** Per-fish convergence tracking; freeze gradient to zero for converged fish (delta < 0.5px for 3 consecutive steps).

#### Step 6: Upsample coarse → fine
- Evaluate coarse spline (K=4) at K_fine=7 parameter positions
- Uses basis matrix multiplication for shape-preserving upsampling

#### Step 7: Warm-start validation
- For each warm-started fish: compare current fine-init loss against previous frame's loss
- If current > 2× previous: revert to cold start for that fish

#### Step 8: Fine optimization (K=7 control points)
- Same loss function and optimizer as coarse, but `max_iter=40`
- Same two constraint enforcement modes
- Same per-fish adaptive early stopping
- Uses `max_iter=1` per L-BFGS step in the inner loop to enable per-fish convergence checking between steps

#### Step 9: Build Midline3D output
- Arc length via 1000-point numerical integration (same as triangulation)
- Reprojection residuals: chamfer distance per camera (consistent with optimizer metric)
- Half-widths: average all per-camera per-point pixel half-widths converted to metres
  - Uses same `_pixel_half_width_to_metres` as triangulation
  - **But:** assigns uniform hw across all sample points (single mean value), unlike triangulation which interpolates per-position

#### Step 10: Warm-start bookkeeping
- Store fine control points for next frame
- **Consistency flip:** if dot product of current vs. previous spline direction is negative, flip control points to maintain temporal consistency
- Store per-fish loss for warm-start validation next frame

---

## 5. Side-by-Side Comparison

| Aspect | Triangulation | Curve Optimizer |
|--------|--------------|-----------------|
| **Correspondence** | Explicit: point i in cam A ↔ point i in cam B (via orientation align + epipolar snap) | None: chamfer distance is order-independent |
| **Computation device** | CPU (numpy/scipy/torch on CPU) | GPU (torch CUDA) |
| **Per-frame cost** | N body points × C(N_cams, 2) pairwise triangulations + spline fit | L-BFGS iterations × (spline eval + refractive projection + chamfer) × 2 stages |
| **Warm-starting** | None (each frame independent) | Yes: previous frame's control points seed next frame |
| **Physical priors** | 3-layer Z validation, ray-angle filter | 5 regularization terms + optional ALM constraints |
| **Output control points** | 7 (configurable) | 7 (coarse=4, fine=7) |
| **Residual metric** | Point-wise reprojection of spline samples | Chamfer distance |
| **Half-width handling** | Per-position interpolation from valid points | Uniform mean across all positions |
| **State** | Stateless | Stateful (warm-starts, previous losses, snapshots) |
| **Code size** | ~1085 lines | ~1883 lines |

---

## 6. Potential Problems

### Triangulation backend

1. **Exhaustive pairwise search scales poorly.** For 7 cameras, C(7,2)=21 pairs per body point × 15 body points = 315 triangulations per fish per frame. Each involves ray casting (refractive Snell's law) + DLT solve + reprojection scoring. For 8+ cameras the code switches to a different strategy (all-camera + residual rejection), creating a behavioral discontinuity at the 7→8 camera boundary.

2. **Epipolar refinement is expensive and fragile.** For N=15 body points × (N_cams-1) target cameras, each requiring 100-depth ray sampling + projection + cdist against all N skeleton points. The snap can fail silently (NaN rows) if the fish is outside the [0.01, 2.0] depth range or if the skeleton is too sparse near the epipolar curve.

3. **Orientation alignment uses only 3 sample points.** The greedy pairwise alignment fixes the first camera (sorted order) as reference and aligns each other camera independently. If the reference camera has a poor view (e.g., looking nearly end-on at the fish), its midline is short and ambiguous, potentially causing all other cameras to align to a bad reference. The reference for orientation alignment (sorted first) differs from the reference for epipolar refinement (longest arc), which is inconsistent.

4. **50px inlier threshold is very permissive.** A 50-pixel reprojection error is large enough to admit outlier cameras while still "passing" inlier gating. The threshold is shared across the entire fish rather than being adaptive to body-point position or camera geometry.

5. **Half-width conversion uses only the first inlier camera's focal length.** For non-central body points, the depth varies significantly across cameras, and the pinhole approximation degrades away from the optical axis. Using a single camera's focal length introduces systematic bias.

### Curve optimizer backend

6. **Shared L-BFGS Hessian across all fish.** All fish are batched into one `(N_fish, K, 3)` tensor and share a single L-BFGS optimizer. A single fish with pathological observations (wrong identity, occluded, boundary-clipped) can corrupt the Hessian approximation and destabilize all other fish. The per-fish freezing (zero gradient) is a mitigation but doesn't prevent Hessian contamination from pre-freeze steps.

7. **Six regularization terms with hand-tuned weights.** The interaction between lambda_length, lambda_curvature, lambda_chord_arc, lambda_z_variance, and lambda_smoothness creates a complex energy landscape. Each is normalized differently. The comment "lambda=10 means trade up to 10px of chamfer" suggests the intent, but whether the normalizations actually produce this calibration in practice is unclear.

8. **Chamfer distance is Z-insensitive.** The projection through the refractive model compresses depth variation — a fish at depth Z and Z+5cm may project to nearly identical pixel locations. The Z-variance penalty partially addresses this, but it's a soft penalty competing with 5 other terms. This is acknowledged in comments but remains a fundamental limitation.

9. **Cold-start assumption of straight fish at 8.5cm.** The nominal_length_m=0.085m and straight-line initialization assumes fish of a specific size in a neutral pose. For fish that are significantly curved, longer/shorter than nominal, or at unusual depths, the cold start may be far from the true solution, risking convergence to a local minimum.

10. **Warm-start consistency flip uses simple dot product.** The head-tail direction check (dot product of first-to-last control point vectors between frames) can misfire if the fish makes a tight U-turn between frames. The flip is applied to the stored warm-start but not to the Midline3D output, so the output direction can still alternate frame-to-frame.

11. **Per-fish convergence check is expensive.** At each fine-stage step, evaluates `_data_loss` independently for every non-frozen fish (looping in Python, one at a time). This defeats the purpose of batched GPU optimization. For N_fish=10 and 40 fine steps, that's up to 400 individual loss evaluations.

### Shared concerns

12. **`_find_matching_annotated` uses centroid proximity (10px tolerance).** This brittle matching between tracklet centroids and AnnotatedDetections can fail when detections are close together (common in crowded frames) or when centroid definitions differ between the tracker and detector.

13. **Two separate implementations of the same matching helper.** `_find_matching_annotated` exists in both `core/reconstruction/stage.py:407` and `core/midline/stage.py:553` with identical logic but no shared code.

14. **Half-width handling differs between backends.** Triangulation produces per-position interpolated half-widths; curve optimizer produces a uniform mean. Downstream consumers (HDF5 output, visualization) receive different half-width profiles depending on which backend ran, with no indication of which method was used.

15. **`refine_midline_lm` is a stub.** The Levenberg-Marquardt refinement function exists in the public API (`__init__.py` exports it) but returns its input unchanged. Dead code in the public interface.
