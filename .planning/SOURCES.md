# Sources

Literature references for non-standard techniques used in AquaPose, organized by pipeline stage. Each entry documents the technique, its citation, and where it is implemented in the codebase.

## Entry Template

```
### [Short Identifier]

**Title:** Full paper title
**Authors:** Last, F., Last, F., ...
**Venue:** Conference/Journal, Year
**Link:** https://arxiv.org/abs/XXXX.XXXXX
**Inspired:** Brief description of what this paper influenced in AquaPose
**Where used:** Path or module where the inspired approach is implemented
```

---

## Calibration

### [Agrawal-2012-MultilayerRefractive]

**Title:** A Theory of Multi-Layer Flat Refractive Geometry
**Authors:** Agrawal, A., Ramalingam, S., Taguchi, Y., Chari, V.
**Venue:** CVPR, 2012
**Inspired:** Flat refractive interface model. Forward projection uses Newton-Raphson iteration (10 fixed steps) to find the refraction point on the water surface. Inverse ray casting traces a pinhole ray to the air-water interface and refracts it via Snell's law. Both paths are implemented as differentiable PyTorch functions.
**Where used:** `src/aquapose/calibration/projection.py` — `RefractiveProjectionModel.project()` (forward) and `RefractiveProjectionModel.cast_ray()` (inverse)

### [Hartley-2003-MVG]

**Title:** Multiple View Geometry in Computer Vision (2nd ed.)
**Authors:** Hartley, R., Zisserman, A.
**Venue:** Cambridge University Press, 2003
**Inspired:** SVD least-squares ray intersection for multi-view triangulation. The normal equation form `A = sum_i (I - d_i d_i^T)` is solved via `torch.linalg.lstsq`.
**Where used:** `src/aquapose/calibration/projection.py` — `triangulate_rays()`; `src/aquapose/core/reconstruction/utils.py` — `weighted_triangulate_rays()`

---

## Detection

### [Jocher-2023-YOLOv8]

**Title:** Ultralytics YOLOv8
**Authors:** Jocher, G., Chaurasia, A., Qiu, J.
**Venue:** Software release, 2023
**Link:** https://github.com/ultralytics/ultralytics
**Inspired:** Two detection backends: (1) standard YOLOv8 bounding-box detection with configurable confidence and NMS-IoU thresholds; (2) YOLOv8-OBB oriented bounding box detection, whose corner points drive downstream affine crop extraction for midline models. Also used for YOLO-seg (instance segmentation) and YOLO-pose (keypoint) inference in the midline stage.
**Where used:** `src/aquapose/core/detection/backends/yolo.py` — `YOLOBackend`; `src/aquapose/core/detection/backends/yolo_obb.py` — `YOLOOBBBackend`; `src/aquapose/core/midline/backends/segmentation.py` — `SegmentationBackend`; `src/aquapose/core/midline/backends/pose_estimation.py` — `PoseEstimationBackend`

---

## Tracking

### [Cao-2023-OC-SORT]

**Title:** Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking
**Authors:** Cao, J., Pang, J., Weng, X., Khirodkar, R., Kitani, K.
**Venue:** CVPR, 2023
**Link:** https://arxiv.org/abs/2203.14360
**Inspired:** Per-camera 2D multi-object tracking via OC-SORT (observation-centric SORT with Kalman filter). Wraps the boxmot library implementation. Tracks transition through detected/coasted states; Kalman-predicted positions fill gaps during occlusion.
**Where used:** `src/aquapose/core/tracking/ocsort_wrapper.py` — `OcSortTracker`

---

## Association

### [Traag-2019-Leiden]

**Title:** From Louvain to Leiden: Guaranteeing Well-Connected Communities
**Authors:** Traag, V. A., Waltman, L., van Eck, N. J.
**Venue:** Scientific Reports 9, 2019
**Link:** https://arxiv.org/abs/1810.08473
**Inspired:** Cross-view fish identity clustering. Pairwise cross-camera tracklet affinities (scored by ray-ray closest-point distance) are assembled into a weighted graph. Leiden community detection with RBConfigurationVertexPartition partitions the graph into identity groups, with must-not-link constraints enforcing same-camera exclusion.
**Where used:** `src/aquapose/core/association/clustering.py` — `cluster_tracklets()`; `src/aquapose/core/association/scoring.py` — `ray_ray_closest_point()`

---

## Midline Extraction

### [Blum-1967-MedialAxis]

**Title:** A Transformation for Extracting New Descriptors of Shape
**Authors:** Blum, H.
**Venue:** Models for the Perception of Speech and Visual Form, MIT Press, 1967
**Inspired:** Morphological skeletonization of binary fish masks to extract one-pixel-wide medial axis representations. Combined with Euclidean distance transform for half-width estimation, BFS longest-path extraction on the 8-connected skeleton graph, and arc-length resampling at uniform parameter values.
**Where used:** `src/aquapose/core/midline/midline.py` — `_skeleton_and_widths()`, `_longest_path_bfs()`, `_resample_arc_length()`, `MidlineExtractor`

---

## Reconstruction

### [deBoor-1978-Splines]

**Title:** A Practical Guide to Splines
**Authors:** de Boor, C.
**Venue:** Springer, 1978 (revised 2001)
**Inspired:** Cubic (k=3) B-spline fitting of triangulated 3D body points with 7 control points and clamped uniform knots. Uses `scipy.interpolate.make_lsq_spline` for least-squares fitting.
**Where used:** `src/aquapose/core/reconstruction/utils.py` — `build_spline_knots()`, `fit_spline()`; `src/aquapose/core/reconstruction/backends/dlt.py` — `DltBackend`

---

## Training Set Bootstrapping (Planned)

### [Redolfi-2023-MultiViewAL]

**Title:** Rethinking the Data Annotation Process for Multi-view 3D Pose Estimation with Active Learning and Self-Training
**Authors:** Redolfi, A., Einfalt, M., Berjawi, T., Ludwig, K., Lenz, R.
**Venue:** WACV, 2023
**Link:** https://arxiv.org/abs/2112.13709
**Inspired:** Iterative self-training loop design — using multi-view triangulation to generate pseudo-labels, pooling with a small manual annotation set, and retraining from pretrained base weights each round to limit error propagation.
**Where used:** Pseudo-labeling milestone (seed: `.planning/seed-pseudo-labeling.md`)

### [Biderman-2025-UncertaintyMVAnimal]

**Title:** An Uncertainty-Aware Framework for Data-Efficient Multi-View Animal Pose Estimation
**Authors:** Biderman, D., Whiteway, M., et al.
**Venue:** arXiv preprint, 2025
**Link:** https://arxiv.org/abs/2510.09903
**Inspired:** Two techniques: (1) using per-camera residual variance (not just mean) as a pseudo-label quality signal, analogous to their Mahalanobis distance filtering; (2) pose-diversity sampling via k-means clustering on 3D pose configurations instead of uniform temporal subsampling.
**Where used:** Pseudo-labeling milestone — confidence scoring and frame selection (seed: `.planning/seed-pseudo-labeling.md`)
