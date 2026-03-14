# AquaPose: Technical Summary for Journal Article

## 1. Purpose and Scope

AquaPose is a 3D fish pose estimation library that reconstructs time-series fish midlines from synchronized multi-view video captured by above-water cameras looking through a flat air-water interface. It is the pose estimation companion to AquaCal (refractive multi-camera calibration) and AquaMVS (refractive multi-view stereo), consuming AquaCal's calibration output and synchronized multi-camera video to produce per-fish 3D B-spline midlines as HDF5 time series.

The key distinguishing feature is **physically correct Snell's law refraction modeling** integrated into every geometric operation — ray casting, forward projection, lookup table construction, triangulation, and reprojection — combined with a fully automated pipeline from raw video to 3D midlines requiring no manual annotation after initial model training.

---

## 2. Input and Output

**Input:**

- AquaCal calibration JSON (intrinsics, extrinsics, distortion coefficients, water surface parameters, refractive indices)
- Synchronized multi-camera video (one video file per camera, any codec supported by OpenCV)
- YOLO OBB and Pose model weights (trained on manual or pseudo-label annotations)
- Pre-generated lookup tables (forward and inverse, cached to disk with hash-based invalidation)

**Output (per run):**

- HDF5 time series containing, per frame per fish:
  - B-spline control points (7 x 3, degree 3, clamped knots)
  - Arc length (metres)
  - Half-widths at sample points (metres)
  - Per-camera reprojection residuals (pixels)
  - Reconstruction confidence metadata (camera count, low-confidence flag)
  - Centroid depth and per-point Z offsets
- Serialized pipeline configuration (YAML)
- Optional per-chunk diagnostic caches (pickle) for pipeline replay

---

## 3. Reference Camera Geometry

The reference rig consists of 12 **ring cameras** (standard lens, pinhole model) arranged in a downward-looking circle above a flat-bottomed aquarium, plus 1 **center camera** (fisheye lens, auxiliary). All cameras observe through a flat water surface at a calibrated depth `water_z`. The coordinate system follows the OpenCV convention inherited from AquaCal: +X right, +Y forward, +Z down (into water), with origin at the reference camera's optical center. All internal values are in metres.

| Role | Count | Lens | Used for detection/pose | Used for triangulation |
|------|-------|------|------------------------|----------------------|
| Ring | 12 | Standard (pinhole) | Yes | Yes |
| Center | 1 | Fisheye | No | No |

The center camera is excluded from the pose pipeline because its compressed field of view after undistortion introduces geometric artifacts incompatible with the affine crop scheme used for pose estimation.

---

## 4. Refractive Projection Model

The `RefractiveProjectionModel` implements the core geometric primitive used throughout the pipeline. It provides two operations: `cast_ray` (pixel to world ray) and `project` (world point to pixel), both implemented as differentiable PyTorch functions.

### 4.1 Ray Casting (pixel -> world ray)

Given a pixel coordinate `(u, v)`:

1. **Pinhole back-projection:** Convert to a normalized camera-frame ray direction using `K_inv`.
2. **World-frame rotation:** Rotate into the world frame via `R^T`.
3. **Ray-surface intersection:** Intersect the camera ray with the flat water surface plane at `Z = water_z`, yielding the refraction **origin** on the interface.
4. **Snell's law refraction (vectorized):**

```
cos(theta_i) = |dot(d_incident, n)|
sin^2(theta_t) = (n_air / n_water)^2 * (1 - cos^2(theta_i))
cos(theta_t) = sqrt(1 - sin^2(theta_t))
d_refracted = (n_ratio * d_incident) + (cos(theta_t) - n_ratio * cos(theta_i)) * n_oriented
```

The result is a unit-length ray direction pointing from the interface into the water. A 3D point at ray depth `d` is recovered as `P = origin + d * direction`.

### 4.2 Forward Projection (world point -> pixel)

The inverse operation requires finding the interface point that satisfies Snell's law for the given camera center, underwater point, and surface normal. This has no closed-form solution.

AquaPose parameterizes the problem by the horizontal radial distance `r_p` from the camera footprint to the interface point and solves with **10 fixed Newton-Raphson iterations** using an analytical Jacobian:

```
Residual:  f(r_p) = n_air * sin(theta_air) - n_water * sin(theta_water)
```

The fixed iteration count makes the operation compatible with PyTorch autograd. After convergence, standard pinhole projection (`K * [R | t]`) maps the interface point to pixel coordinates. A validity mask flags points above the water surface or behind the camera.

### 4.3 Lens Distortion

Per-camera lens distortion is corrected as a preprocessing step using OpenCV's `remap` with precomputed tables. The loader dispatches between pinhole (5-8 coefficient) and fisheye (4 coefficient) undistortion models based on per-camera metadata from AquaCal. All downstream computation operates in undistorted image coordinates with a post-undistortion intrinsic matrix `K_new`.

---

## 5. Pipeline Architecture

### 5.1 Stage-Based Execution

The pipeline is organized as a five-stage sequential computation, with each stage reading structured input from a shared `PipelineContext` and writing structured output back to it:

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| 1. Detection | Undistorted frames | Per-camera detections | YOLO OBB fish detection |
| 2. Pose | Detections + frames | Enriched detections | YOLO Pose keypoint regression |
| 3. Tracking | Enriched detections | Per-camera 2D tracklets | Kalman filter tracking with gap interpolation |
| 4. Association | 2D tracklets + LUTs | Cross-view identity groups | Ray-based scoring + Leiden clustering |
| 5. Reconstruction | Identity groups + calibration | 3D B-spline midlines | DLT triangulation + spline fitting |

Stages are pure computation modules with no side effects. An observer system (console logging, diagnostic caching, timing) attaches to lifecycle events emitted by the engine without mutating pipeline state.

### 5.2 Temporal Chunking

Long videos are processed in fixed-size temporal chunks (default 200 frames) by the `ChunkOrchestrator`. State is carried across chunk boundaries via a `ChunkHandoff` object containing:

- Per-camera Kalman filter states (for seamless track continuation)
- Global identity map (local fish IDs -> persistent global IDs)
- Per-fish tracklet membership sets (for identity stitching)

**Identity stitching** between chunks uses greedy 1:1 bipartite matching: each fish identity in the new chunk is matched to the previous chunk's identity whose `(camera_id, track_id)` set has maximum overlap with the new group's constituent tracklets.

### 5.3 Truncation and Replay

The pipeline supports partial execution via `stop_after` (run only the first N stages) and replay from cached diagnostic snapshots, enabling iterative development of downstream stages without re-running upstream computation.

---

## 6. Detection (Stage 1)

Fish are detected using Ultralytics YOLOv8/v26 oriented bounding box (OBB) models. Raw OBB detections undergo polygon-based NMS via Shapely for exact intersection-over-union computation, replacing axis-aligned IoU approximations that perform poorly on elongated, rotated fish bodies. The backend converts Ultralytics OBB angles (clockwise radians) to math convention (counterclockwise, `[-pi, pi]`) and produces `Detection` objects carrying oriented bounding box corners in `[TL, TR, BR, BL]` order.

Batch inference uses automatic CUDA OOM recovery: if a batch exceeds GPU memory, the batch size is halved and retried without pipeline failure.

---

## 7. Pose Estimation (Stage 2)

Six-keypoint fish poses (nose, head, spine1, spine2, spine3, tail) are regressed using a YOLO-Pose model on stretch-fill affine crops. For each OBB detection:

1. A 3-point affine transform maps the OBB corners (TL, TR, BL) to canvas corners, producing a rotation-aligned, aspect-ratio-distorted crop that fills the entire input canvas.
2. The YOLO-Pose model predicts keypoint locations and per-keypoint confidence scores on the crop.
3. Predicted keypoints are back-projected to full-frame coordinates via the inverse affine transform.

The stretch-fill crop convention (as opposed to letterboxing or scale-to-fit) is critical: training data uses the same affine mapping, so any mismatch between training and inference crop geometry degrades keypoint localization. A minimum confidence floor is applied per keypoint; keypoints below this threshold are marked invisible.

---

## 8. Tracking (Stage 3)

Per-camera tracklet formation uses a 24-dimensional constant-velocity Kalman filter (6 keypoints x 2D position + 2D velocity). The measurement noise covariance is **confidence-scaled**: `R[2k, 2k] = base_R / max(conf_k, eps)`, so high-confidence keypoints exert stronger pull on the state estimate.

### 8.1 Frame-to-Frame Association

Each frame's detections are matched to predicted track states via the Hungarian algorithm, using a combined cost matrix:

```
cost = (1 - OKS) + lambda * (1 - OCM)
```

- **OKS (Object Keypoint Similarity)**: Spatial agreement between predicted and observed keypoints, weighted by per-keypoint visibility and keypoint-specific sigma values.
- **OCM (Orientation-Curvature Matching)**: Cosine similarity of the spine heading vector (spine1 -> spine3), penalizing heading disagreement.

Hard gating thresholds on maximum match distance and total cost prevent spurious associations.

### 8.2 Track Lifecycle

Tracks follow a state machine: **Probationary** (newly born, must accumulate hits) -> **Confirmed** (actively tracked) -> **Coasting** (unmatched, propagated by prediction only) -> **Dead** (coasted too long). Two recovery mechanisms operate during coasting:

- **ORU (Observation Recovery from Unmatched)**: Re-checks unmatched detections against coasting tracks with relaxed thresholds.
- **OCR (Observation-to-Coast Recovery)**: Recovers coasting tracks when a detection's OKS to the predicted state exceeds a recovery threshold.

### 8.3 Gap Interpolation

After the forward tracking pass completes, short temporal gaps (up to `max_gap_frames`) in confirmed tracks are filled by per-keypoint cubic spline interpolation over the gap's bounding frames.

---

## 9. Cross-View Association (Stage 4)

Fish identities are established across cameras in three stages.

### 9.1 Pairwise Ray-Based Scoring

For each pair of tracklets from different cameras that overlap temporally:

1. Rays are cast from each camera through each tracklet's keypoints at shared frames using precomputed forward lookup tables (bilinear interpolation over a regular pixel grid).
2. Per-keypoint ray-ray closest-point distances are computed in batch.
3. Per-frame mean distances are converted to soft affinity scores via a linear kernel: `score = 1 - dist / threshold` for distances below threshold, zero otherwise.
4. An early termination heuristic examines only the first `early_k` shared frames; if no inliers are found, the pair is skipped entirely.
5. The final pair score is the soft inlier fraction weighted by an overlap reliability factor.

Camera pairs with insufficient shared viewing volume (determined by the inverse LUT's voxel visibility overlap) are pruned before scoring.

### 9.2 Leiden Community Detection

Pairwise scores populate an undirected weighted graph whose nodes are tracklets. Leiden community detection partitions this graph into fish identity groups, subject to **must-not-link constraints** that prevent two tracklets from the same camera (with temporally overlapping frames) from being assigned to the same fish. Constraint violations are resolved by greedy eviction of the lowest-affinity tracklet from each violated cluster.

### 9.3 Multi-Keypoint Group Validation

After clustering, each group is validated by computing per-keypoint ray-to-centroid residuals across all member tracklets. Groups with excessively high centroid reprojection error are split or evicted.

### 9.4 Singleton Recovery

Tracklets not assigned to any multi-view group are recovered by computing per-keypoint ray residuals against existing group centroids. A greedy best-first assignment with a binary split-assign sweep merges singletons into groups when residuals fall below threshold, maximizing the number of cameras contributing to each fish's reconstruction.

---

## 10. Lookup Tables

Two precomputed lookup table structures accelerate the geometric operations in the association stage:

### 10.1 Forward LUT (pixel -> ray)

A regular pixel grid (configurable step size) is projected through the full refractive ray casting pipeline. At query time, ray origins and directions for arbitrary pixel coordinates are obtained by bilinear interpolation over the grid, avoiding per-query Newton-Raphson solves.

### 10.2 Inverse LUT (3D voxel -> pixel)

A cylindrical voxel grid spanning the tank volume is projected into each ring camera via the refractive forward projection model. The resulting structure stores per-voxel visibility masks and projected pixel coordinates for each camera. This enables:

- **Camera overlap graph construction**: Counting shared visible voxels per camera pair to prune scoring.
- **Ghost point lookup**: Fast 3D-to-pixel projection without per-query optimization.

Both LUTs use SHA-256 hash-based cache invalidation keyed on the calibration file contents and LUT configuration parameters.

---

## 11. 3D Reconstruction (Stage 5)

### 11.1 Confidence-Weighted DLT Triangulation

For each fish at each frame, corresponding 2D midline points across cameras are triangulated using a confidence-weighted Direct Linear Transform formulation. Each camera's ray direction `d_i` contributes to the normal equations weighted by `sqrt(confidence_i)`:

```
A = sum_i(w_i * (I - d_i * d_i^T))
b = sum_i(w_i * (I - d_i * d_i^T) * o_i)
P = lstsq(A, b)
```

where `o_i` is the ray origin on the water surface and `w_i` is the confidence weight.

**Iterative outlier rejection** (up to C-2 rounds, where C is the camera count): after initial triangulation, per-camera reprojection residuals are computed through the full refractive model. The single worst-contributing camera is removed if its residual exceeds a threshold and at least 3 cameras remain. The point is re-triangulated from the remaining inliers.

Points reconstructed above the water surface are rejected. A minimum camera count (default 3) is required for a valid reconstruction.

### 11.2 Midline Densification

The 6 keypoints from pose estimation are interpolated to a denser midline (default 15 points) via linear spline interpolation along arc-length-normalized `t` values before triangulation. This provides sufficient body coverage for B-spline fitting.

### 11.3 B-Spline Fitting

Triangulated 3D body points are fit with a cubic (degree 3) B-spline with 7 control points using clamped uniform knots (`[0,0,0,0, 0.25, 0.5, 0.75, 1,1,1,1]`) and least-squares fitting via `scipy.interpolate.make_lsq_spline`. A low-confidence flag is raised when fewer than the minimum required body points are successfully triangulated.

### 11.4 Z-Flattening

An optional post-triangulation step sets all body-point Z coordinates to the centroid Z before spline fitting. This suppresses frame-to-frame depth jitter arising from the inherent Z-anisotropy of the top-down camera geometry (cameras above the water have much weaker depth resolution than lateral resolution). Per-point Z offsets are preserved in the output for downstream analysis.

### 11.5 Temporal Z-Smoothing

Reconstructed centroid Z coordinates are smoothed across track segments using a Gaussian filter to further suppress depth jitter. Smoothing operates per-segment to respect track boundaries.

### 11.6 Gap Interpolation

Short temporal gaps (up to `max_interp_gap` frames) in reconstructed 3D midlines are filled by linear interpolation of the B-spline control points between bounding frames.

### 11.7 Half-Width Estimation

Per-sample-point half-widths are converted from pixel space to world metres using a pinhole depth approximation: `half_width_m = half_width_px * depth / focal_length`, using the depth of the corresponding triangulated body point.

---

## 12. Training Data Pipeline

### 12.1 Annotation Geometry

Training labels for YOLO OBB and Pose models are derived from keypoint annotations. PCA via SVD computes oriented bounding boxes from visible keypoints: the first two principal components define the OBB axes, and corner ordering is canonicalized to `[TL, TR, BR, BL]` by aligning the head-tail axis with the long axis. This produces training labels geometrically consistent with the stretch-fill affine crop used at inference.

### 12.2 Sample Store

A SQLite-backed sample store manages training data with content-hash (SHA-256) deduplication, source-priority upsert (manual annotations take precedence over pseudo-labels), provenance tracking, augmentation lineage, and tag-based querying. The store assembles YOLO-format dataset directories with proper `dataset.yaml` metadata including keypoint shape and flip indices.

### 12.3 TPS Elastic Augmentation

Pose training data is augmented using thin-plate spline (TPS) warping to synthesize curved fish from straight or mildly curved examples. Two deformation profiles are generated:

- **C-curve**: Parabolic lateral displacement `4 * t * (1-t) * chord * tan(angle)`, producing a uniform arc. Zero displacement at endpoints, maximum at midpoint.
- **S-curve**: Sinusoidal lateral displacement using `sin(2*pi*t)` for a full S-shape. Amplitude halved relative to C-curve for visual parity.

Flanking TPS control points at +/-25 px along keypoint normals prevent lateral blur artifacts. Only visible keypoints serve as TPS control points to avoid singular matrices. Default configuration: 4 variants per image (2 C-curve, 2 S-curve) with 5-15 degree angle range.

### 12.4 Pseudo-Label Generation

Semi-supervised training labels are generated from pipeline output. Reconstructed 3D B-spline midlines are evaluated at the canonical keypoint arc-length fractions and reprojected into each camera view via the refractive projection model, producing per-fish keypoint annotations with confidence scores derived from reconstruction quality (reprojection residual, camera count). Curvature is computed from spline control points as mean absolute bending angle between adjacent tangent vectors.

### 12.5 YOLO Training

A unified training wrapper drives Ultralytics YOLO model training for OBB and Pose tasks with model-type-specific defaults (image size, augmentation parameters). Training integrates with the pseudo-label pipeline for iterative self-training rounds, where each round produces new pseudo-labels from improved model predictions.

### 12.6 Label Studio Integration

Bidirectional YOLO-to-Label Studio JSON conversion enables human review and correction of pseudo-labels. The export path converts YOLO annotations to Label Studio task format with keypoint names sourced from `dataset.yaml`; the import path converts corrected Label Studio annotations back to YOLO format for retraining.

---

## 13. Triangulation Uncertainty Characterization

Reconstruction accuracy is characterized by Monte Carlo simulation: ground-truth 3D points are projected into all cameras via the refractive model, pixel coordinates are perturbed with Gaussian noise (+/-0.5 px), and re-triangulated. This reveals the depth-dependent Z-anisotropy inherent to the camera geometry, informing the Z-flattening and temporal smoothing strategies.

---

## 14. Synthetic Testing Framework

A synthetic data generation module creates controlled-ground-truth test scenarios by generating known 3D fish midlines (straight lines, circular arcs, sinusoidal S-curves) and projecting them into synthetic camera rigs. The synthetic pipeline replaces Stage 1 (Detection) with a synthetic data stage and runs the remaining pipeline stages, enabling end-to-end algorithm validation without real video data.

---

## 15. Software Architecture

- **Language:** Python 3.11+
- **Tensor library:** PyTorch for all geometric computation; NumPy at the AquaCal boundary and for array I/O
- **Key dependencies:** Ultralytics (YOLO OBB/Pose), OpenCV (undistortion, image I/O), SciPy (spline fitting, interpolation), igraph + leidenalg (community detection), Shapely (polygon NMS), h5py (HDF5 output), Pydantic (configuration validation)
- **Build system:** Hatch with pre-commit hooks (Ruff linting/formatting, basedpyright type checking)
- **Architecture:** Three-layer design with strict one-way dependencies:
  - **Layer 1 (Core):** Pure computation modules with no side effects
  - **Layer 2 (Engine):** Pipeline orchestration, stage sequencing, chunk management, event emission
  - **Layer 3 (Observers):** Passive diagnostic collection, console output, artifact writing
- **Device convention:** Pipeline modules receive device from configuration; low-level math follows input tensor device with no hardcoded device assumptions
- **Error handling:** Failed projections (behind camera, total internal reflection) return validity masks rather than raising exceptions; invalid data is represented as NaN with zero confidence

---

## Citations

### Detection and Pose Estimation

[1] G. Jocher, A. Chaurasia, and J. Qiu. *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics

### Tracking

[2] R. E. Kalman. "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering* 82(1), 1960.

[3] H. W. Kuhn. "The Hungarian Method for the Assignment Problem." *Naval Research Logistics Quarterly* 2(1-2), 1955.

[4] T.-Y. Lin et al. "Microsoft COCO: Common Objects in Context." *ECCV*, 2014. arXiv:1405.0312.

### Clustering

[5] V. A. Traag, L. Waltman, and N. J. van Eck. "From Louvain to Leiden: Guaranteeing Well-Connected Communities." *Scientific Reports* 9, 2019.

### Multi-View Geometry and Reconstruction

[6] R. Hartley and A. Zisserman. *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press, 2003.

[7] C. de Boor. *A Practical Guide to Splines*. Springer, 1978 (revised 2001).

### Refractive Geometry

[8] A. Agrawal, S. Ramalingam, Y. Taguchi, and V. Chari. "A Theory of Multi-Layer Flat Refractive Geometry." *CVPR*, 2012.

[9] A. Jordt-Sedlazeck and R. Koch. "Refractive Structure-from-Motion on Underwater Images." *ICCV*, 2013.

### Software Libraries

[10] G. Bradski. "The OpenCV Library." *Dr. Dobb's Journal* 25(11), 2000.

[11] P. Virtanen et al. "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." *Nature Methods* 17, 2020.

### Augmentation

[12] F. L. Bookstein. "Principal Warps: Thin-Plate Splines and the Decomposition of Deformations." *IEEE TPAMI* 11(6), 1989.
