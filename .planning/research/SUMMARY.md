# Project Research Summary

**Project:** AquaPose
**Domain:** Multi-view 3D fish pose estimation via refractive multi-view triangulation (with shelved analysis-by-synthesis alternative)
**Researched:** 2026-02-21
**Confidence:** MEDIUM-HIGH (primary pipeline uses well-understood geometry; refractive camera model validated; shelved analysis-by-synthesis path retained as fallback)

## Executive Summary

AquaPose is a research-grade multi-view system for reconstructing the full 3D midline and body shape of cichlid fish from 12 synchronized overhead cameras (13th top-down camera excluded for poor mask quality). The domain has no direct comparators — existing tools (DLC+Anipose, DANNCE, SLEAP) handle multi-view reconstruction but assume standard pinhole cameras in air and output keypoints, not body shape. AquaPose's three distinguishing innovations are: (1) physically correct refractive projection through the air-water interface integrated into both ray casting and forward projection, (2) a 3D midline spline reconstructed from triangulated medial axes across views rather than discrete keypoints, and (3) centroid-based RANSAC cross-view identity assignment with Hungarian tracking in 3D space. The recommended build order is strict: calibration and refractive projection validation must precede any reconstruction work, and single-fish validation must precede multi-fish extension.

The primary reconstruction pipeline is direct triangulation: 2D medial axis extraction from segmentation masks, arc-length correspondence across views, RANSAC multi-view triangulation per body point, and 3D cubic B-spline fitting. An optional Levenberg-Marquardt refinement stage jointly optimizes the 3D spline against all 2D observations using the refractive forward projection model. This replaces the original analysis-by-synthesis approach (differentiable mesh rendering via PyTorch3D + Adam optimization), which took 30+ minutes per second of video and is now shelved but retained as a fallback.

The recommended stack centers on PyTorch 2.4.1 with CUDA 12.1 for calibration, refractive projection, and segmentation (U-Net). The reconstruction path itself is scipy/numpy/scikit-image — no differentiable rendering framework is required for the primary pipeline. PyTorch3D is retained only for the shelved analysis-by-synthesis path.

The most critical risk for scientific validity remains the camera geometry: 12 top-down cameras with nearly parallel optical axes create a degenerate Z-reconstruction problem. This has been quantified (see Z-uncertainty report) and must be tracked as a per-point quality metric. A secondary risk is medial axis instability on noisy masks — the current U-Net produces masks at IoU ~0.62, which requires morphological smoothing before skeletonization to avoid skeleton wobble.

## Key Findings

### Recommended Stack

The stack is organized around PyTorch for calibration, refractive projection, and segmentation inference. The reconstruction pipeline itself uses standard scientific Python (scipy, numpy, scikit-image) and does not require differentiable rendering. Detection uses YOLOv8 with U-Net binary segmentation on cropped detections; SAM2 is used offline for pseudo-label generation only and is not in the inference path.

**Core technologies:**
- Python 3.11 + PyTorch 2.4.1 + CUDA 12.1: For calibration, refractive projection, U-Net segmentation inference
- scikit-image >= 0.22: `skeletonize` for medial axis extraction, morphological operations for mask smoothing
- scipy >= 1.13: `splprep` / B-spline fitting for 3D midline, `least_squares` for optional LM refinement, `linear_sum_assignment` for Hungarian assignment, `distance_transform_edt` for half-width estimation
- OpenCV 4.13 (headless): MOG2 background subtraction, video I/O, 2D overlays, morphological preprocessing
- h5py >= 3.11: Primary output format for per-frame pose trajectories
- rerun-sdk >= 0.22: Primary debugging and QA visualization; synchronized multi-camera 2D + 3D
- YOLOv8 (ultralytics): Object detection for fish bounding boxes
- U-Net (custom, MobileNetV3-Small encoder): Binary mask segmentation on cropped detections

**Shelved-pipeline-only dependencies (not required for primary reconstruction):**
- PyTorch3D 0.7.9 (source install): Differentiable silhouette renderer for analysis-by-synthesis path
- kornia >= 0.7: Differentiable Lovasz-hinge IoU loss for analysis-by-synthesis path

**Critical version constraint:** PyTorch is pinned at 2.4.1 for compatibility with the shelved PyTorch3D path. If the shelved path is formally abandoned, this constraint can be relaxed.

### Expected Features

AquaPose is a research tool, not a product. "Users" are the research team itself and the downstream behavioral biology pipeline. Table stakes are features whose absence makes the system scientifically invalid; differentiators constitute the novel research contribution.

**Must have (table stakes):**
- Refractive camera model with physically correct Snell's law projection — the foundation; without this the multi-view geometry is invalid
- Direct 3D midline triangulation from multi-view medial axes — the primary reconstruction mechanism
- Per-fish per-frame 3D midline reconstruction — the v1 deliverable
- Multi-view silhouette extraction pipeline (YOLOv8 + U-Net) — produces the inputs to the reconstruction pipeline
- Cross-view holdout validation with reprojection IoU metric — required for scientific credibility
- Per-frame 3D trajectory output (position, orientation, curvature) in HDF5/CSV — enables downstream biological analysis
- Video I/O for multi-camera synchronized frame extraction

**Should have (competitive / v1.x):**
- Width-profile reconstruction (half-width spline from distance transforms) — enables body shape analysis
- Multi-fish detection and parallel per-fish reconstruction — v2 deliverable
- Centroid-based RANSAC identity with Hungarian 3D tracking — the cross-view identity mechanism
- Occlusion handling with temporal continuity in 3D — required for robust multi-fish tracking
- Optional LM refinement stage for joint reprojection optimization — adds robustness to arc-length correspondence errors

**Defer (v2+):**
- Full-day continuous tracking (hours-long recordings with identity persistence)
- Behavioral feature extraction library (tail-beat frequency, curvature, inter-fish distance, approach angle)
- Sex-differentiated shape model (requires labeled morphometric training data)
- Batch processing infrastructure for full experimental dataset
- Shape-signature identity via body plan decomposition (original v2 Re-ID mechanism)

**Explicit anti-features (do not build):**
- Real-time processing — batch offline processing; real-time is not a goal
- GUI annotation tool — use Label Studio + supervision for format conversion
- Monocular (single-camera) reconstruction — geometrically ill-posed; biases architecture away from multi-view
- Appearance-based Re-ID — centroid-based 3D identity is the primary mechanism; appearance Re-ID adds complexity without clear benefit

### Architecture Approach

The system is organized as a strict linear pipeline: Detection & Segmentation --> Cross-View Identity --> Midline Extraction --> 3D Triangulation & Spline Fitting --> (Optional LM Refinement) --> Output. The critical architectural decision is how arc-length correspondence enables cross-view triangulation without explicit feature matching: fish are slender bodies with a single dominant axis, so the normalized arc-length parameterization of the 2D medial axis projection is approximately preserved across views. This assumption breaks down for significantly curved fish viewed from very different angles, which is handled by RANSAC per body point and view-angle weighting.

**Major components:**
1. CalibrationLoader + RefractiveProjector — parses AquaCal JSON; exposes per-camera refractive ray casting (2D -> 3D ray) and forward projection (3D -> 2D pixel); PyTorch-based for differentiability where needed
2. YOLODetector + UNetSegmentor — YOLOv8 producing bounding boxes, U-Net producing binary masks per crop; shared crop utilities for coordinate transforms
3. CrossViewIdentifier — per-frame RANSAC over centroid rays across cameras; clusters rays into fish identities; produces (camera_id, detection_id) -> fish_id mapping plus 3D centroid per fish
4. HungarianTracker — frame-to-frame 3D centroid assignment via scipy `linear_sum_assignment`; persistent fish IDs across frames
5. MedialAxisExtractor — morphological smoothing, `skeletonize`, distance transform half-width, longest-path BFS pruning, head-tail disambiguation via 3D centroid projection
6. ArcLengthSampler — cumulative arc-length normalization to [0,1]; resampling at N fixed positions (e.g., N=15); produces cross-view point correspondences
7. RANSACTriangulator — per body point RANSAC over camera subsets; refractive ray intersection; view-angle weighting to downweight foreshortened views; outputs N 3D points + residuals per fish
8. SplineFitter — `scipy.interpolate.splprep` for 3D midline B-spline (5-8 control points); separate 1D spline for width profile
9. LMRefiner (optional) — `scipy.optimize.least_squares` with method='lm'; jointly optimizes spline control points against all 2D medial axis observations via refractive forward projection; warm-starts from SplineFitter output
10. TrajectoryWriter + Visualizer — h5py HDF5 output; rerun-sdk for live QA; OpenCV for 2D overlays

**Key patterns:**
- Warm-start every frame from the previous frame's spline (reduces LM iterations from ~20 to ~5 when used)
- Batch triangulations across body points into vectorized calls (~15 points x ~8 fish = 120 triangulations/frame)
- Design for N fish from day one — all function signatures accept lists; parallelize across fish within a frame
- Cross-view holdout: withhold 1-2 cameras from triangulation; evaluate reprojection error on them as a generalization metric
- Gate optional LM refinement on triangulation residual — skip when RANSAC residuals are below threshold

### Critical Pitfalls

1. **Non-differentiable refractive projection** — Implementing the refractive model using numpy or scipy breaks gradient flow for any component that needs it (LM Jacobians via autograd, forward projection in the optional refiner). The refractive projection has been reimplemented in PyTorch. This is less critical than in the shelved pipeline since the primary reconstruction path does not require differentiable gradients through the full chain, but the forward projection model must still be correct for reprojection scoring in RANSAC and the optional LM refiner.

2. **Depth-independent refraction model** — Using OpenCV's standard distortion model to approximate refraction produces systematic depth-dependent errors (fish near tank floor reproject worse than near-surface fish). The error is a consistent 3D bias, not noise, and invalidates reconstruction. Implement full per-ray Snell's law projection tracing through air-glass-water interface with correct refractive indices.

3. **Arc-length correspondence errors on curved fish** — The arc-length parameterization assumes the 2D medial axis projection preserves body-position correspondence across views. This breaks down when a curved fish is viewed from angles where foreshortening compresses the arc-length mapping unevenly. Cameras viewing along the fish's body axis are the worst offenders. Mitigations: RANSAC per body point rejects cameras with bad correspondence; view-angle weighting downweights foreshortened views; optional LM refinement jointly optimizes across all views. If these mitigations are insufficient, epipolar-guided correspondence refinement is a future upgrade path.

4. **Top-down camera Z-weakness not quantified** — 12 cameras with nearly parallel optical axes create degenerate Z reconstruction. 2D reprojection error can look excellent while Z is wrong by centimeters. This has been quantified analytically (see Z-uncertainty report). Report X, Y, Z errors separately — never report only aggregate reprojection error.

5. **Medial axis instability on noisy masks** — The current U-Net produces masks at IoU ~0.62, well below the 0.90 target. Noisy mask boundaries cause skeleton wobble, spurious branches, and shifted midline positions. Mitigations: morphological closing + opening with adaptive kernel before skeletonization; longest-path BFS pruning to discard spurious branches; RANSAC triangulation rejects outlier body points. Monitor skeleton quality as segmentation improves.

6. **Single-fish architecture blocking v2 extension** — Building v1 with global state (one mask, one optimizer, one mesh object) requires a full rewrite at v2. Design for N fish from day one: per-fish state, batch-first operations, detection returning a list of per-fish masks even when length is 1.

## Implications for Roadmap

Based on combined research, the build order is indicated by strict data dependencies and the need for validation gates before proceeding:

### Phase 1: Calibration and Refractive Geometry Foundation (Complete)
**Rationale:** Everything downstream depends on a working, validated RefractiveProjector. No reconstruction code is scientifically meaningful until the camera model is correct.
**Delivers:** CalibrationLoader parsing AquaCal JSON; RefractiveProjector implementing per-ray Snell's law in PyTorch; ray casting (2D -> 3D ray) and forward projection (3D -> 2D pixel); unit tests covering central rays and edge-field rays at 30-48 deg incidence; validation showing < 1px reprojection error on known 3D points; quantified Z-uncertainty bounds for the 12-camera top-down geometry.
**Status:** Complete. Refractive projection reimplemented in PyTorch. Z-uncertainty quantified analytically.

### Phase 2: Segmentation Pipeline (Complete)
**Rationale:** Segmentation produces the masks that drive every downstream phase. It was built and validated independently of the reconstruction pipeline.
**Delivers:** YOLOv8-based fish detection with bounding boxes; U-Net binary segmentation on cropped detections (MobileNetV3-Small encoder, ~2.5M params, 128x128 input); SAM2-based pseudo-label generation for training data; per-frame binary masks per camera per fish.
**Status:** Complete. Best val IoU 0.623 — below 0.90 target but accepted to unblock downstream phases. Morphological smoothing required before skeletonization.
**Uses:** YOLOv8 (ultralytics), custom U-Net, SAM2 (offline), OpenCV

### Phase 3: Fish Mesh Model (Complete, Shelved)
**Rationale:** The parametric fish mesh (midline spline + swept ellipse cross-sections) was built as part of the original analysis-by-synthesis pipeline. It is complete but shelved — the primary pipeline uses direct triangulation instead of differentiable mesh rendering.
**Delivers:** FishMeshBuilder producing watertight triangle meshes from FishState {p, psi, kappa, s}; midline spline + swept ellipses.
**Status:** Complete, shelved. Retained as fallback if direct triangulation proves insufficient.

### Phase 4: Per-Fish Reconstruction via Analysis-by-Synthesis (Shelved)
**Rationale:** The original v1 core — differentiable mesh rendering + Adam optimization against silhouette IoU. Shelved due to 30+ min/sec runtime and replaced by the direct triangulation pipeline (Phases 5-7+).
**Delivers:** (When shelved pipeline was active) RefractiveRenderer, SoftSilhouetteShader, multi-objective loss, PoseOptimizer.
**Status:** Shelved. Code retained. See `.planning/inbox/fish-reconstruction-pivot.md` for pivot rationale.

### Phase 5: Cross-View Identity & 3D Tracking (New)
**Rationale:** All downstream reconstruction stages require knowing which mask in camera A corresponds to which mask in camera B. This is a prerequisite for medial axis triangulation.
**Delivers:** CrossViewIdentifier using RANSAC over centroid rays across cameras; (camera_id, detection_id) -> fish_id mapping; 3D centroid per fish; HungarianTracker for persistent fish IDs across frames via 3D centroid assignment.
**Addresses features:** Cross-view identity, persistent 3D tracking [table stakes for multi-fish]
**Uses:** Existing refractive ray casting code, scipy (Hungarian assignment)

### Phase 6: 2D Medial Axis & Arc-Length Sampling (New)
**Rationale:** Extracts the 2D midline representation from segmentation masks and establishes cross-view point correspondences via normalized arc-length. This is the input to multi-view triangulation.
**Delivers:** MedialAxisExtractor (morphological smoothing, skeletonize, distance transform half-width, BFS pruning, head-tail disambiguation); ArcLengthSampler (cumulative arc-length normalization, fixed-N resampling).
**Addresses features:** 2D midline extraction, cross-view correspondence [table stakes for reconstruction]
**Uses:** scikit-image (skeletonize, morphology), scipy (distance_transform_edt)

### Phase 7+: 3D Triangulation, Spline Fitting, Output (TBD)
**Rationale:** The core reconstruction: triangulate corresponding body points across views, fit a 3D spline, optionally refine via LM. Detailed phase planning TBD.
**Delivers:** RANSACTriangulator, SplineFitter, optional LMRefiner, TrajectoryWriter, visualization and validation.
**Uses:** Existing refractive ray intersection code, scipy (splprep, least_squares), h5py, rerun-sdk

### Phase Ordering Rationale

- **Calibration first:** The RefractiveProjector is a shared dependency for every subsequent phase; errors here compound downstream.
- **Segmentation second:** Masks are required by all reconstruction stages; building and validating segmentation early saves calendar time.
- **Cross-view identity before midline extraction:** Medial axis extraction and arc-length correspondence require knowing which masks correspond to the same physical fish across cameras.
- **Midline extraction before triangulation:** The triangulation stage consumes cross-view point correspondences produced by arc-length sampling.
- **Single-fish before multi-fish:** The feature dependency graph is explicit — multi-fish tracking requires reliable single-fish reconstruction.
- **Output and validation integrated with reconstruction:** The 3D evaluation framework (cross-view holdout, per-axis error reporting) is built alongside reconstruction, not as a separate phase.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 5 (Cross-View Identity):** RANSAC clustering of refractive rays for identity assignment; tuning inlier thresholds for this rig's geometry.
- **Phase 6 (Medial Axis):** Skeleton stability at current mask quality (IoU ~0.62); adaptive morphological kernel sizing; head-tail disambiguation reliability.
- **Phase 7+ (Triangulation):** Arc-length correspondence accuracy on curved fish; view-angle weighting calibration; LM refinement convergence behavior.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Calibration):** Complete. Refractive projection reimplemented in PyTorch.
- **Phase 2 (Segmentation):** Complete. YOLOv8 + U-Net trained and validated.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Primary pipeline uses scipy/numpy/scikit-image — mature, well-documented, no version fragility. PyTorch3D version pinning only matters for shelved pipeline. U-Net + YOLOv8 are stable. |
| Features | MEDIUM-HIGH | Table stakes and anti-features are HIGH confidence. The novel contribution (refractive triangulation with arc-length correspondence) is well-grounded in geometry but untested on this rig at current mask quality. |
| Architecture | HIGH | Direct triangulation pipeline is well-understood geometry. Build order derives from clear data dependencies. The pivot proposal (`.planning/inbox/fish-reconstruction-pivot.md`) is the authoritative pipeline design document. |
| Pitfalls | MEDIUM | Core optics pitfalls (refractive distortion, Z-weakness) are HIGH confidence from peer-reviewed literature. Arc-length correspondence errors on curved fish are MEDIUM — mitigations exist (RANSAC, view-angle weighting) but need empirical validation. Medial axis instability at IoU ~0.62 is a known risk with known mitigations. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Z-uncertainty budget:** The theoretical Z reconstruction uncertainty for this specific rig has been quantified analytically. The practical impact on midline triangulation quality at operating depth needs validation against physical reference data.
- **Arc-length correspondence accuracy on curved fish:** The normalized arc-length parameterization assumes projection preserves body-position correspondence. This assumption degrades for curved fish viewed from different angles. The degree of degradation at this rig's camera geometry needs empirical characterization.
- **Medial axis stability at current mask quality (IoU ~0.62):** Noisy mask boundaries cause skeleton wobble and spurious branches. Morphological smoothing mitigates this, but the residual error propagated into triangulation needs quantification.
- **Shape-signature discriminability:** It is unknown whether shape parameters estimated for each fish are sufficiently distinct to serve as a biometric identifier. This is deferred to v2 but remains the core scientific bet for persistent identity.
- **Female detection under worst-case conditions:** YOLOv8/U-Net recall for stationary female fish has not been measured under all lighting conditions. This is the most likely operational failure mode for Phase 2.

## Sources

### Primary (HIGH confidence)
- PyTorch3D INSTALL.md — version compatibility matrix (PyTorch 2.4.1 + PyTorch3D 0.7.9 confirmed): https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- PyTorch3D renderer docs — SoftSilhouetteShader, MeshRasterizer, batched API: https://pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/rasterizer.html
- AquaPose proposed_pipeline.md — original project spec (see `.planning/inbox/proposed_pipeline.md`)
- **AquaPose reconstruction pivot proposal — authoritative pipeline design document (see `.planning/inbox/fish-reconstruction-pivot.md`)**
- Refractive Two-View Reconstruction for Underwater 3D Vision (IJCV 2019) — refractive model correctness requirements: https://link.springer.com/article/10.1007/s11263-019-01218-9
- Multi-animal pose estimation and tracking with DeepLabCut (Nature Methods 2022) — competitor baseline: https://www.nature.com/articles/s41592-022-01443-0
- WaterMask: Instance Segmentation for Underwater Imagery (ICCV 2023) — underwater segmentation precedent: https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf
- OpenCV MOG2 documentation: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

### Secondary (MEDIUM confidence)
- A Calibration Tool for Refractive Underwater Vision (arXiv 2024) — port tilt estimation, refractive calibration: https://arxiv.org/abs/2405.18018
- VoGE: Differentiable Volume Renderer for Analysis-by-Synthesis (OpenReview) — confirms analysis-by-synthesis pattern (shelved pipeline): https://openreview.net/forum?id=AdPJb9cud_Y
- SOD-SORT: Multi-fish tracking with EKF + Hungarian — confirms tracking pattern: https://arxiv.org/html/2507.06400v3
- vmTracking: multi-animal pose tracking (PLOS Biology 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC11845028/
- PyTorch3D GitHub issues #1626, #905, #1855 — soft rasterizer hyperparameter behavior (shelved pipeline): https://github.com/facebookresearch/pytorch3d/issues/1626
- Adventures with Differentiable Mesh Rendering (Andrew Chan blog) — practical implementation gotchas (shelved pipeline)

### Tertiary (LOW confidence)
- Shape-signature identity for fish Re-ID — no direct precedent found; extrapolated from SMAL-based shape Re-ID for quadrupeds. Deferred to v2.
- PyTorch3D source build stability against PyTorch 2.5-2.10 — community reports in GitHub issues; no official confirmation. Relevant only if shelved pipeline is reactivated.

---
*Research completed: 2026-02-19; updated for reconstruction pivot: 2026-02-21*
*Ready for roadmap: yes*
