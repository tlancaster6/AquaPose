# Multi-View Refractive Triangulation in AquaPose

How AquaPose reconstructs 3D fish midlines from 2D camera views through water.

---

## The Setup

A ring of 12 cameras sits above an aquarium, looking down into the water. Light from a fish travels **through water, crosses the air-water interface, then reaches the camera**. The interface bends (refracts) every light ray, so standard pinhole geometry doesn't work — we need Snell's law.

**Key parameters:**
- Water surface height: `water_z` (metres, in world coordinates)
- Interface normal: `n̂` — points from water toward air, typically `[0, 0, -1]`
- Refractive indices: `n_air = 1.0`, `n_water = 1.333`
- Per-camera: intrinsics `K`, rotation `R`, translation `t`

---

## Step 1: Refractive Projection (3D → 2D)

> *"Given a 3D point underwater, where does it appear in a camera image?"*

A 3D point **Q** underwater doesn't project through a straight line to the camera center **C**. Instead, the light path bends at the water surface. We need to find the **refraction point P** on the surface where Snell's law is satisfied.

### Geometry

```
     C (camera, in air)
     |  ↑ h_c (camera height above water)
     |
  ───P──────────── water surface (z = water_z)
     |\
     | \ ↓ h_q (fish depth below surface)
     |  \
     Q   (fish point, underwater)

  |--r_p--|         horizontal dist: camera → P
  |----r_q------|   horizontal dist: camera → Q
```

- **Air segment:** Camera C to surface point P — angle θ_air from vertical
- **Water segment:** Surface point P to fish Q — angle θ_water from vertical

### Snell's Law

At the refraction point P:

$$n_{air} \cdot \sin\theta_{air} = n_{water} \cdot \sin\theta_{water}$$

Expressed in terms of the horizontal radius `r_p` (distance from camera footprint to P):

$$\sin\theta_{air} = \frac{r_p}{\sqrt{r_p^2 + h_c^2}}, \qquad \sin\theta_{water} = \frac{r_q - r_p}{\sqrt{(r_q - r_p)^2 + h_q^2}}$$

### Solving for P: Newton-Raphson

We solve for `r_p` iteratively. Define the residual:

$$f(r_p) = n_{air} \cdot \frac{r_p}{\sqrt{r_p^2 + h_c^2}} - n_{water} \cdot \frac{r_q - r_p}{\sqrt{(r_q - r_p)^2 + h_q^2}}$$

Update rule (10 fixed iterations for autograd compatibility):

$$r_p \leftarrow r_p - \frac{f(r_p)}{f'(r_p)}$$

Once P is found, the air-side ray (C → P) is a straight line. Standard pinhole projection finishes the job:

$$\mathbf{p}_{cam} = R \cdot P_{world} + t, \qquad \text{pixel} = K \cdot \frac{\mathbf{p}_{cam}}{p_{cam,z}}$$

**Source:** `src/aquapose/calibration/projection.py` — `RefractiveProjectionModel.project()`

---

## Step 2: Ray Casting (2D → 3D)

> *"Given a pixel, cast a ray into the water."*

This is the inverse of projection. We need the ray's **origin** (on the water surface) and **direction** (into the water).

### Algorithm

1. **Back-project pixel to camera ray:**

$$\mathbf{d}_{cam} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \qquad \mathbf{d}_{world} = R^T \cdot \hat{\mathbf{d}}_{cam}$$

2. **Intersect with water surface** (z = water_z):

$$t = \frac{z_{water} - C_z}{d_{world,z}}, \qquad \text{origin} = C + t \cdot \mathbf{d}_{world}$$

3. **Apply Snell's law to get the refracted direction:**

$$\sin\theta_t = \frac{n_{air}}{n_{water}} \cdot \sin\theta_i$$

$$\cos\theta_t = \sqrt{1 - \sin^2\theta_t}$$

$$\mathbf{d}_{refracted} = \frac{n_{air}}{n_{water}} \mathbf{d}_{world} + \left(\cos\theta_t - \frac{n_{air}}{n_{water}} \cos\theta_i\right) \hat{\mathbf{n}}$$

The output is a ray: any underwater point at depth `d` along it is `origin + d · d_refracted`.

**Source:** `src/aquapose/calibration/projection.py` — `RefractiveProjectionModel.cast_ray()`

---

## Step 3: Triangulation (Multiple Rays → 3D Point)

> *"Given refracted rays from N cameras, find the 3D point they converge on."*

With perfect data, all rays intersect at one point. In practice they don't, so we find the point **minimizing perpendicular distance to all rays**.

### Least-Squares Formulation

For ray i with origin **o**_i and direction **d̂**_i, the perpendicular projector onto the plane normal to the ray is:

$$P_i = I - \hat{\mathbf{d}}_i \hat{\mathbf{d}}_i^T$$

The optimal point **p** solves:

$$\underbrace{\left(\sum_i P_i\right)}_{A} \mathbf{p} = \underbrace{\sum_i P_i \, \mathbf{o}_i}_{b}$$

Solved via `torch.linalg.lstsq(A, b)` which handles near-singular cases via SVD.

**Source:** `src/aquapose/calibration/projection.py` — `triangulate_rays()`

---

## Step 4: Robust Multi-Camera Assembly

> *"Not all cameras see the fish. Some views are noisy. How do we pick the right rays?"*

The `_triangulate_body_point()` function applies three layers of defense:

### Layer 1 — Physical Constraints
- Reject points above water (`z ≤ water_z`)
- Reject points below max depth (`z > water_z + max_depth`)

### Layer 2 — Ray-Angle Filter
- Skip camera pairs whose rays meet at < 5°
- Near-parallel rays produce wildly inaccurate depth estimates
- The ring rig guarantees ≥ 8° for any in-tank point, so this catches degenerate outliers

### Layer 3 — Camera-Count Strategy

| Cameras | Strategy |
|---------|----------|
| 1 | Cannot triangulate — skip |
| 2 | Direct pair triangulation + Z validation |
| 3–7 | **Exhaustive pairwise:** try all C(N,2) pairs, score each against held-out cameras, expand best seed to all inliers (reprojection error < 50 px) |
| 8+ | **All-camera + residual rejection:** triangulate with all, remove outliers beyond median + 2σ, re-triangulate with inliers |

**Source:** `src/aquapose/reconstruction/triangulation.py` — `_triangulate_body_point()`

---

## Step 5: Cross-Camera Midline Alignment

> *"Skeletonization gives arbitrary point ordering. Camera A might detect head→tail while Camera B detects tail→head."*

For each non-reference camera, try both orientations:
1. Triangulate sample points (head, mid, tail) with the reference camera
2. Compute total chord length of the resulting 3D segments
3. **Keep the orientation with shorter chord length** — correct alignment produces a smooth curve; flipped produces a zigzag

**Source:** `src/aquapose/reconstruction/triangulation.py` — `_align_midline_orientations()`

---

## Step 6: Epipolar Correspondence Refinement

> *"Body point #5 in Camera A might not correspond to body point #5 in Camera B."*

For each reference-camera body point:
1. Cast its refracted ray into the water
2. Sample 50 depth values (0.5 m to 3.0 m) along the ray
3. Project each sample into the target camera → traces an **epipolar curve**
4. Snap the target's closest skeleton point to the curve (if within 15 px)
5. Otherwise reject the correspondence

This enforces geometric consistency before triangulation.

**Source:** `src/aquapose/reconstruction/triangulation.py` — `_refine_correspondences_epipolar()`

---

## Step 7: B-Spline Fitting

> *"Smooth the triangulated 3D points into a continuous fish midline."*

After triangulating up to 15 body points per fish, fit a cubic B-spline:

- **Degree:** 3 (cubic)
- **Control points:** 7
- **Knot vector:** `[0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]`
- **Minimum data points:** 9 (7 controls + 2 for overdetermination)
- **Method:** `scipy.interpolate.make_lsq_spline()` — weighted least-squares

Arc length is computed by evaluating the spline at 1000 points and summing segment lengths.

**Source:** `src/aquapose/reconstruction/triangulation.py` — `_fit_spline()`

---

## Full Pipeline Flow

```
Video Frame
  │
  ▼
Detection (YOLO) ──→ Bounding boxes per camera
  │
  ▼
Segmentation (U-Net) ──→ Binary fish masks per camera
  │
  ▼
Midline Extraction ──→ 15-point 2D skeleton per detection
  │  (skeletonize → longest-path BFS → arc-length resample)
  │
  ▼
Cross-View Association (RANSAC centroid clustering)
  │  Cast rays from each detection → triangulate centroids
  │  Cluster centroids that land near each other → same fish
  │
  ▼
Temporal Tracking (Hungarian matching)
  │  Match fish identities frame-to-frame
  │
  ▼
3D Reconstruction
  │  1. Align midline orientations across cameras
  │  2. Refine correspondences via epipolar geometry
  │  3. Triangulate each of 15 body points (with RANSAC)
  │  4. Fit cubic B-spline (7 control points)
  │
  ▼
Midline3D: 3D fish midline with arc length, residuals, confidence
```

---

## Why Z Is Hard

All cameras look **down** into the tank. Their rays are nearly parallel in the Z direction. Result:

- **XY accuracy:** good (rays converge at steep angles horizontally)
- **Z accuracy:** 3–10× worse (small pixel errors → large depth errors)

This is intrinsic to the top-down rig geometry and is the primary source of reconstruction uncertainty.
