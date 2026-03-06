# AquaPose: Key Techniques Report

This report documents the principal algorithms and methods comprising the AquaPose 3D fish pose estimation pipeline. AquaPose reconstructs fish 3D midlines from multi-view silhouettes using a 13-camera aquarium rig with refractive calibration. Two diverging reconstruction paths are supported: **direct triangulation** (geometric, per-frame) and **curve-based optimization** (differentiable, GPU-accelerated).

---

## 1. Refractive Multi-View Projection

All 3D-to-2D and 2D-to-3D transformations account for refraction at the flat air–water interface via Snell's law. The forward projection solves for the refraction point on the water surface using Newton–Raphson iteration (10 fixed steps), while the inverse (ray casting) traces a pinhole ray from the camera center, refracts it at the interface, and extends it into the water volume. Both operations are implemented as differentiable PyTorch functions, enabling gradient-based optimization of 3D geometry through the refractive model. Lens distortion is corrected per-camera using OpenCV's pinhole (5–8 coefficient) and fisheye (4 coefficient) undistortion models with precomputed remap tables. Calibration parameters are loaded from AquaCal JSON files.

**Files:** `src/aquapose/calibration/projection.py`, `src/aquapose/calibration/loader.py`

## 2. Object Detection

Two detection backends are supported:

- **MOG2 background subtraction** [1] with morphological cleanup (close then open, elliptical 5×5 kernel), connected-component labeling, and watershed splitting [2] to separate merged foreground blobs containing multiple fish.
- **YOLOv8** [3] with configurable confidence and NMS-IoU thresholds, providing bounding-box detections without requiring a learned background model.

Both produce interchangeable `Detection` objects (bounding box, optional mask).

**Files:** `src/aquapose/segmentation/detector.py`

## 3. Pseudo-Label Generation with SAM2

Training masks for the segmentation network are generated automatically using SAM2 [4] rather than manual annotation. Detection bounding boxes (with padding) are used as box prompts to SAM2's image predictor; when multiple masks are returned, the largest-area mask is selected. Quality filters reject detections below a confidence threshold, masks smaller than a minimum area, and masks with fill ratios outside a plausible range. The largest connected component is extracted to suppress stray pixels.

**Files:** `src/aquapose/segmentation/sam_labels.py`

## 4. U-Net Segmentation

Binary fish masks are predicted by a U-Net [5] with a MobileNetV3-Small [6] encoder pretrained on ImageNet. The decoder uses bilinear upsampling with skip connections across four resolution levels. Input crops are resized to 128×128. Training uses a combined BCE + Dice [7] loss, AdamW [8] with differential learning rates (encoder at 0.1× the decoder rate), cosine annealing [9], gradient clipping (max norm 5.0), and early stopping (patience 20 epochs). Data augmentation includes random flips, 90° rotations, and color jitter. Train–validation splitting is stratified by camera to maintain balanced representation.

**Files:** `src/aquapose/segmentation/model.py`, `src/aquapose/segmentation/training.py`

## 5. 2D Midline Extraction

Fish midlines are extracted from binary masks through a multi-step procedure:

1. **Morphological smoothing** with adaptive kernel size derived from the mask's minor axis length.
2. **Skeletonization** [10] via `skimage.morphology.skeletonize`, producing a one-pixel-wide skeleton.
3. **Longest-path extraction** by two-pass BFS on the 8-connected skeleton graph, yielding an ordered head-to-tail path.
4. **Arc-length resampling** at 15 uniformly spaced parameter values along the normalized arc length, with linear interpolation of pixel coordinates and half-widths (from the Euclidean distance transform).

Masks touching the frame boundary are rejected (clipped fish), and masks below a minimum area are skipped.

**Files:** `src/aquapose/reconstruction/midline.py`

## 6. Cross-View Fish Association

Fish identities are established across cameras using RANSAC [11] centroid clustering on refractive rays:

1. Centroids are computed from mask moments; refractive rays are cast from each camera.
2. Random pairs of rays are triangulated to candidate 3D positions.
3. Candidates are reprojected into all cameras; detections within a pixel threshold are classified as inliers.
4. The 3D centroid is refined by re-triangulating from all inlier cameras.
5. Duplicate clusters within 0.04 m and with >0.2 camera co-visibility ratio are merged.

Single-view detections fall back to a ray–depth heuristic at a default tank depth.

**Files:** `src/aquapose/tracking/associate.py`

## 7. Temporal Tracking

A multi-state tracker manages fish identity across frames:

- **States:** Probationary → Confirmed (after 5 consecutive hits) → Coasting (up to 7 missed frames) → Dead.
- **Motion model:** Constant velocity with 0.8× damping per frame during coasting.
- **Association:** Tracks claim detections within a 15-pixel reprojection threshold; claims with residuals exceeding 3× the running mean are rejected.

**Files:** `src/aquapose/tracking/tracker.py`

---

## 8. Reconstruction Path A: Direct Triangulation

The geometric reconstruction path operates per-frame without optimization:

### 8a. Multi-View Triangulation

Corresponding 2D body points (from arc-length-aligned midlines) are triangulated across cameras using SVD least-squares ray intersection [12]. Robustness is achieved through layered defenses:

- **Ray-angle filtering:** Pairs with inter-ray angle <5° are rejected to avoid ill-conditioned geometry.
- **Z-validity filtering:** Points above the water surface are discarded.
- **Adaptive strategy by camera count:** 2 cameras use direct pair triangulation; 3–7 cameras perform exhaustive pairwise search scored by held-out camera residuals; 8+ cameras use all-camera triangulation with median + 2σ outlier rejection.

### 8b. Cross-Camera Orientation Alignment

Midline head–tail orientation is aligned greedily against a reference camera (longest 2D arc length). Each non-reference camera is tested in both orientations; the orientation minimizing the triangulated chord length is selected, achieving O(N) complexity versus O(2^N) brute force.

### 8c. Epipolar Correspondence Refinement

Correspondences are refined using epipolar geometry [12]: body points from the reference camera are projected as epipolar curves into target cameras, and target midline points are snapped to the nearest curve within a 20-pixel threshold.

### 8d. B-Spline Fitting

Triangulated 3D body points are fit with a cubic (k=3) B-spline with 7 control points using clamped uniform knots and least-squares fitting [13] via `scipy.interpolate.make_lsq_spline`. A confidence flag is raised when >20% of body points are triangulated from fewer than 3 cameras.

**Files:** `src/aquapose/reconstruction/triangulation.py`

---

## 9. Reconstruction Path B: Curve-Based Optimization

The optimization path directly fits 3D B-spline midlines to 2D observations by minimizing a differentiable loss.

### 9a. Loss Function

The total loss combines six terms in pixel-equivalent units:

| Term | Description |
|------|-------------|
| **Chamfer data loss** [14] | Symmetric 2D chamfer distance between projected spline points and observed midline points, averaged per camera per fish. Invalid (above-water) projections incur a depth penalty. |
| **Arc-length penalty** | Quadratic penalty for spline arc length outside a ±30% tolerance band around the nominal fish length (0.085 m). |
| **Curvature penalty** | Quadratic penalty for per-joint bend angles exceeding 30° (a biomechanical constraint on fish body bending). |
| **Z-variance penalty** | Penalizes variance of Z-coordinates along the spline, encoding the observation that fish bodies are approximately depth-planar during normal swimming. |
| **Chord-arc ratio penalty** | Penalizes chord/arc-length ratios below 0.75, preventing the spline from folding back on itself. |
| **Smoothness penalty** | Penalizes the squared second-difference of control points, suppressing high-frequency oscillations. |

### 9b. Optimization

Optimization uses L-BFGS [15] in a coarse-to-fine schedule: 4 control points for 20 iterations, then 7 control points for 40 iterations, with convergence tolerance of 0.5 pixels over 3 patience steps. All fish in a frame are optimized in parallel as a batched (N_fish, K, 3) tensor. Warm-starting from the previous frame's solution is attempted; if the warm-started loss exceeds 2× the cold-start loss, the optimizer falls back to straight-line initialization.

### 9c. Augmented Lagrangian Method (Optional)

An optional ALM [16] mode enforces arc-length, curvature, and chord-arc constraints as hard constraints rather than soft penalties. The method runs 5 outer iterations, each solving an L-BFGS subproblem, with penalty parameter ρ increasing from 10 to 1000 (factor 2× per iteration) and constraint tolerance 0.001.

**Files:** `src/aquapose/reconstruction/curve_optimizer.py`

---

## 10. Parametric Fish Mesh

Reconstructed midlines are visualized as 3D meshes using a parametric model:

- **Spine:** A circular arc parameterized by position, heading (yaw), pitch, curvature, and scale, with Gram–Schmidt orthogonalization to produce a local coordinate frame (tangent, normal, binormal) at each cross-section.
- **Cross-sections:** Elliptical profiles at 7 arc-length positions with species-specific height and width ratios (default: cichlid proportions), tapering at snout and caudal peduncle.
- **Mesh:** N×M tube vertices plus head/tail apex caps, with counterclockwise face winding, assembled into `pytorch3d.structures.Meshes` [17] objects for GPU-compatible rendering.

**Files:** `src/aquapose/mesh/builder.py`, `src/aquapose/mesh/spine.py`, `src/aquapose/mesh/profiles.py`

---

## 11. Triangulation Uncertainty Characterization

Reconstruction accuracy is characterized by Monte Carlo simulation: ground-truth 3D points are projected into all cameras, pixel coordinates are perturbed (±0.5 px Gaussian noise), and re-triangulated. This reveals the depth-dependent Z-anisotropy inherent to the camera geometry (top-down cameras exhibit 3–5× worse Z resolution than XY).

**Files:** `src/aquapose/calibration/uncertainty.py`

---

## Citations

### Detection and Segmentation

[1] Z. Zivkovic. "Improved Adaptive Gaussian Mixture Model for Background Subtraction." *ICPR*, 2004. Z. Zivkovic and F. van der Heijden. "Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction." *Pattern Recognition Letters* 27(7), 2006.

[2] L. Vincent and P. Soille. "Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion Simulations." *IEEE TPAMI* 13(6), 1991.

[3] G. Jocher, A. Chaurasia, and J. Qiu. *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics

[4] N. Ravi et al. "SAM 2: Segment Anything in Images and Videos." *arXiv:2408.00714*, 2024.

[5] O. Ronneberger, P. Fischer, and T. Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015.

[6] A. Howard et al. "Searching for MobileNetV3." *ICCV*, 2019.

[7] F. Milletari, N. Navab, and S.-A. Ahmadi. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." *3DV*, 2016.

### Training

[8] I. Loshchilov and F. Hutter. "Decoupled Weight Decay Regularization." *ICLR*, 2019. arXiv:1711.05101.

[9] I. Loshchilov and F. Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*, 2017. arXiv:1608.03983.

### Midline and Shape Analysis

[10] H. Blum. "A Transformation for Extracting New Descriptors of Shape." In *Models for the Perception of Speech and Visual Form*, MIT Press, 1967.

### Multi-View Geometry and Reconstruction

[11] M. A. Fischler and R. C. Bolles. "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography." *Communications of the ACM* 24(6), 1981.

[12] R. Hartley and A. Zisserman. *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press, 2003.

[13] C. de Boor. *A Practical Guide to Splines*. Springer, 1978 (revised 2001).

### Optimization

[14] H. Fan, H. Su, and L. J. Guibas. "A Point Set Generation Network for 3D Object Reconstruction from a Single Image." *CVPR*, 2017.

[15] D. C. Liu and J. Nocedal. "On the Limited Memory BFGS Method for Large Scale Optimization." *Mathematical Programming* 45(1–3), 1989.

[16] M. R. Hestenes. "Multiplier and Gradient Methods." *Journal of Optimization Theory and Applications* 4, 1969. M. J. D. Powell. "A Method for Nonlinear Constraints in Minimization Problems." In *Optimization*, Academic Press, 1969.

### Refractive Geometry

[17] A. Agrawal, S. Ramalingam, Y. Taguchi, and V. Chari. "A Theory of Multi-Layer Flat Refractive Geometry." *CVPR*, 2012.

[18] A. Jordt-Sedlazeck and R. Koch. "Refractive Structure-from-Motion on Underwater Images." *ICCV*, 2013.

### Software Libraries

[19] N. Ravi et al. "Accelerating 3D Deep Learning with PyTorch3D." *arXiv:2007.08501*, 2020. https://github.com/facebookresearch/pytorch3d

[20] G. Bradski. "The OpenCV Library." *Dr. Dobb's Journal* 25(11), 2000.

[21] S. van der Walt et al. "scikit-image: Image Processing in Python." *PeerJ* 2:e453, 2014.

[22] P. Virtanen et al. "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." *Nature Methods* 17, 2020.
