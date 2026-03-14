# AquaPose: Key Techniques Report

This report documents the principal algorithms and methods comprising the AquaPose 3D fish pose estimation pipeline. AquaPose reconstructs fish 3D midlines from multi-view keypoint detections using a 12-camera aquarium rig with refractive calibration.

---

## 1. Refractive Multi-View Projection

All 3D-to-2D and 2D-to-3D transformations account for refraction at the flat air-water interface via Snell's law. The forward projection solves for the refraction point on the water surface using Newton-Raphson iteration (10 fixed steps), while the inverse (ray casting) traces a pinhole ray from the camera center, refracts it at the interface, and extends it into the water volume. Both operations are implemented as differentiable PyTorch functions. Lens distortion is corrected per-camera using OpenCV's pinhole (5-8 coefficient) and fisheye (4 coefficient) undistortion models with precomputed remap tables. Calibration parameters are loaded from AquaCal JSON files.

> **Citations:** [8], [9], [10]

**Files:** `src/aquapose/calibration/projection.py`, `src/aquapose/calibration/loader.py`

---

## 2. YOLO OBB Detection

Fish are detected using Ultralytics YOLOv8/v26 oriented bounding box (OBB) models [1]. Raw OBB detections undergo polygon-based NMS via Shapely for exact intersection-over-union computation, replacing axis-aligned IoU approximations. The backend negates Ultralytics OBB angles (clockwise radians) to math convention (counterclockwise, [-pi, pi]) and produces `Detection` objects with oriented bounding box geometry.

> **Citations:** [1]

**Files:** `src/aquapose/core/detection/backends/yolo_obb.py`

---

## 3. YOLO Pose Estimation

Six-keypoint fish poses are regressed using a YOLO-Pose model [1] on stretch-fill affine crops. For each OBB detection, a 3-point affine transform maps the OBB corners (TL, TR, BL) to canvas corners, producing a rotation-aligned crop without letterboxing. Predicted keypoints are back-projected to full-frame coordinates via the inverse affine transform.

> **Citations:** [1]

**Files:** `src/aquapose/core/pose/backends/pose_estimation.py`, `src/aquapose/core/pose/crop.py`

---

## 4. PCA-Based Oriented Bounding Boxes

Training label geometry is derived from visible keypoints using PCA via SVD. The first two principal components define the OBB axes; corner ordering is canonicalized to [TL, TR, BR, BL] by rotating the frame so the head-tail axis aligns with the long axis. This provides consistent OBB labels for YOLO OBB training, matching the stretch-fill crop convention used at inference.

> **Citations:** [6]

**Files:** `src/aquapose/training/geometry.py`

---

## 5. Keypoint Tracking

Per-camera tracklet formation uses a 24-dimensional constant-velocity Kalman filter [2] (6 keypoints x 2D position + velocity). Frame-to-frame association combines two cost terms via Hungarian assignment [3]:

- **OKS (Object Keypoint Similarity)** [4]: measures keypoint spatial agreement, weighted by per-keypoint visibility.
- **OCM (Orientation-Curvature Matching)**: penalizes heading and shape disagreement between predicted and observed poses.

Track lifecycle follows Probationary -> Confirmed -> Coasting -> Dead states, with ORU (observation recovery from unmatched) and OCR (observation-to-coast recovery) mechanisms for re-acquiring lost tracks. After forward tracking completes, gaps in confirmed tracks are filled by cubic spline interpolation of keypoint trajectories.

> **Citations:** [2], [3], [4]

**Files:** `src/aquapose/core/tracking/keypoint_tracker.py`

---

## 6. Cross-View Association

Fish identities are established across cameras in three stages:

### 6a. Pairwise Scoring

For each pair of tracklets from different cameras, refractive rays are cast from camera centers through tracklet centroids. The ray-ray closest-point distance and midpoint position are computed; pairs within a distance threshold contribute affinity scores. Forward and inverse lookup tables (LUTs) with bilinear interpolation accelerate the pixel-to-ray and voxel-to-pixel mappings.

### 6b. Leiden Clustering

Pairwise scores populate an undirected graph whose nodes are tracklets. Leiden community detection [5] partitions this graph into fish identity groups, subject to must-not-link constraints that prevent two tracklets from the same camera from being assigned to the same fish.

### 6c. Singleton Recovery

Tracklets not assigned to any multi-view group are recovered by computing per-keypoint ray-to-3D residuals against existing group centroids. A greedy best-first assignment with a binary split-assign sweep merges singletons into groups when residuals fall below threshold.

> **Citations:** [5], [8], [9]

**Files:** `src/aquapose/core/association/scoring.py`, `src/aquapose/core/association/clustering.py`, `src/aquapose/core/association/recovery.py`, `src/aquapose/calibration/luts.py`

---

## 7. Confidence-Weighted DLT Triangulation

Corresponding 2D keypoints across cameras are triangulated using a confidence-weighted DLT formulation [6]. The normal equations A = sum(w_i * (I - d_i * d_i^T)) weight each camera's ray direction d_i by keypoint confidence w_i. Outlier rejection iteratively removes the single worst-contributing camera when its reprojection residual exceeds a threshold (default 10 px), then re-triangulates from inliers. Points reconstructed above the water surface are z-flattened to the surface plane.

> **Citations:** [6]

**Files:** `src/aquapose/core/reconstruction/backends/dlt.py`

---

## 8. B-Spline Fitting

Triangulated 3D keypoints are fit with a cubic (k=3) B-spline with 7 control points using clamped uniform knots and least-squares fitting [7] via `scipy.interpolate.make_lsq_spline`. A confidence flag is raised when fewer than the minimum required body points (default 9) are successfully triangulated.

> **Citations:** [7], [11]

**Files:** `src/aquapose/core/reconstruction/utils.py`

---

## 9. Temporal Z-Smoothing

Reconstructed centroid z-coordinates are smoothed across track segments using a Gaussian filter [11] to suppress frame-to-frame depth jitter arising from the inherent z-anisotropy of the top-down camera geometry. Smoothing operates per-segment to respect track boundaries.

> **Citations:** [11]

**Files:** `src/aquapose/core/reconstruction/temporal_smoothing.py`

---

## 10. TPS Elastic Augmentation

Pose training data is augmented using thin-plate spline (TPS) warping [12] to synthesize curved fish from straight or mildly curved examples. Two deformation variants are generated:

- **C-curve**: Uniform circular arc via a parabolic displacement profile scaled by `tan(angle)`.
- **S-curve**: Sinusoidal displacement using `sin(2*pi*t)` for a full S-shape.

Flanking TPS control points at +/-25 px along keypoint normals prevent lateral blur. Only visible keypoints serve as control points to avoid singular TPS matrices. Default: 4 variants per image (2 C-curve, 2 S-curve) with 5-15 degree angle range.

> **Citations:** [12]

**Files:** `src/aquapose/training/elastic_deform.py`

---

## 11. Pseudo-Labeling

Semi-supervised training labels are generated from model predictions on unlabeled data. Reconstructed 3D spline midlines are reprojected into each camera view via the refractive projection model and inverse LUTs, producing per-fish keypoint annotations with confidence scores. Curvature is computed from spline control points as mean absolute bending angle. These pseudo-labels feed back into YOLO OBB and Pose model retraining.

**Files:** `src/aquapose/training/pseudo_labels.py`

---

## 12. YOLO Training

A unified training wrapper drives Ultralytics YOLO model training for OBB and Pose tasks [1]. Model-type-specific defaults (image size, augmentation parameters) are applied automatically. Supported augmentation includes mosaic, rotation, and scale. Training integrates with the pseudo-label pipeline for iterative self-training rounds.

> **Citations:** [1]

**Files:** `src/aquapose/training/yolo_training.py`

---

## 13. Triangulation Uncertainty Characterization

Reconstruction accuracy is characterized by Monte Carlo simulation: ground-truth 3D points are projected into all cameras, pixel coordinates are perturbed (+/-0.5 px Gaussian noise), and re-triangulated. This reveals the depth-dependent z-anisotropy inherent to the camera geometry (top-down cameras exhibit 3-5x worse z resolution than xy).

**Files:** `src/aquapose/calibration/uncertainty.py`

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
