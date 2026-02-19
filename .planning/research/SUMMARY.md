# Project Research Summary

**Project:** AquaPose
**Domain:** Multi-view 3D fish pose estimation via analysis-by-synthesis with differentiable refractive rendering
**Researched:** 2026-02-19
**Confidence:** MEDIUM (novel domain with no direct comparators; core primitives well-documented; refractive rendering + fish mesh combination is genuinely new)

## Executive Summary

AquaPose is a research-grade analysis-by-synthesis system for reconstructing the full 3D body shape and pose of cichlid fish from 13 synchronized overhead cameras. The domain has no direct comparators — existing tools (DLC+Anipose, DANNCE, SLEAP) handle multi-view reconstruction but assume standard pinhole cameras in air and output keypoints, not body shape. AquaPose's three distinguishing innovations are: (1) physically correct refractive projection through the air-water interface integrated into a differentiable renderer, (2) a parametric fish mesh model encoding midline curvature and cross-section geometry rather than discrete keypoints, and (3) shape-signature-based identity assignment enabling persistent tracking without appearance features. The recommended build order is strict: calibration and refractive projection validation must precede any optimization work, and single-fish validation must precede multi-fish extension.

The recommended stack centers on PyTorch 2.4.1 + PyTorch3D 0.7.9 (installed from source) with CUDA 12.1. This specific version combination is the only configuration confirmed to work by PyTorch3D's official installation guide. The rest of the stack (Detectron2, SAM2, OpenCV MOG2, kornia, filterpy, h5py) is well-established for this domain and presents low integration risk. The single largest installation risk is the 5-version gap between current PyTorch (2.10.0) and what PyTorch3D officially supports (2.4.x); the mitigation is to pin PyTorch at 2.4.1 for all development and not upgrade until PyTorch3D publishes a compatible release.

The most critical risk for scientific validity is the camera geometry: 13 top-down cameras with nearly parallel optical axes create a degenerate Z-reconstruction problem. Reprojection error in 2D can look excellent (< 2px) while Z estimates are wrong by centimeters. This must be quantified before any optimization code is written, by measuring 3D reconstruction accuracy on a physical reference object at multiple depths. A secondary risk is the refractive projection itself — implementing it as a depth-independent distortion correction (as standard OpenCV calibration does) will introduce systematic errors that invalidate the entire pipeline. The AquaCal library handles this correctly, but its differentiability must be verified before it is assumed to work with PyTorch autograd.

## Key Findings

### Recommended Stack

The stack is organized around PyTorch as the sole deep learning framework throughout; mixing frameworks is not feasible because Detectron2, PyTorch3D, SAM2, and kornia are all PyTorch-native and gradient flow cannot cross framework boundaries. The differentiable rendering primitive is PyTorch3D's `SoftSilhouetteShader` + `MeshRasterizer`, which provides the soft probabilistic blending necessary for silhouette-fitting from scratch (hard rasterizers like nvdiffrast cannot be used here). Detection uses OpenCV MOG2 as the primary foreground detector with Detectron2 Mask R-CNN as the segmentation backbone; SAM2 is used offline for pseudo-label generation only and is not in the inference path.

**Core technologies:**
- Python 3.11 + PyTorch 2.4.1 + CUDA 12.1: The only confirmed-compatible baseline for PyTorch3D 0.7.9
- PyTorch3D 0.7.9 (source install): The only production-grade differentiable silhouette renderer with PyTorch-native mesh structures
- Detectron2 (source install): Mask R-CNN for instance segmentation; PointRend head available if boundary quality is insufficient
- SAM2 (source install, offline only): Zero-shot pseudo-label generation with video propagation for annotation bootstrapping
- OpenCV 4.13 (headless): MOG2 background subtraction, video I/O, 2D overlays
- kornia >= 0.7: Differentiable Lovász-hinge IoU loss; avoids custom implementation of differentiable binary IoU
- scipy >= 1.13: Epipolar ray intersection (Phase II), Hungarian assignment (Phase IV)
- filterpy 1.4.4: Extended Kalman Filter for per-fish 3D tracking
- h5py >= 3.11: Primary output format for per-frame pose trajectories
- rerun-sdk >= 0.22: Primary debugging and QA visualization; synchronized multi-camera 2D + 3D

**Critical version constraint:** Do not use PyTorch > 2.4.x until PyTorch3D publishes a new release with confirmed compatibility. PyTorch3D source builds against 2.5–2.10 are possible but require manual patches and have caused community reports of instability.

### Expected Features

AquaPose is a research tool, not a product. "Users" are the research team itself and the downstream behavioral biology pipeline. Table stakes are features whose absence makes the system scientifically invalid; differentiators constitute the novel research contribution.

**Must have (table stakes):**
- Refractive differentiable renderer with physically correct Snell's law projection — the core mechanism; without this the novelty claim does not exist
- Parametric fish mesh model (midline spline + swept ellipse cross-sections) — required by the optimizer; defines the shape space
- Single-fish per-frame pose/shape optimization — the v1 deliverable
- Multi-view silhouette extraction pipeline (MOG2 + Mask R-CNN) — produces the inputs to the optimizer
- Cross-view holdout validation with reprojection IoU metric — required for scientific credibility
- Per-frame 3D trajectory output (position, orientation, curvature) in HDF5/CSV — enables downstream biological analysis

**Should have (competitive / v1.x):**
- Shape-pose decomposition separating identity-linked body plan from instantaneous pose — enables identity-by-shape
- Multi-fish detection and parallel per-fish optimization — v2 deliverable
- Identity assignment via shape signatures — the novel Re-ID mechanism; replaces appearance-based Re-ID
- Occlusion handling with warm-start identity recovery — required for robust multi-fish tracking

**Defer (v2+):**
- Full-day continuous tracking (hours-long recordings with identity persistence)
- Behavioral feature extraction library (tail-beat frequency, curvature, inter-fish distance, approach angle)
- Sex-differentiated shape model (requires labeled morphometric training data)
- Batch processing infrastructure for full experimental dataset

**Explicit anti-features (do not build):**
- Real-time processing — incompatible with analysis-by-synthesis optimization; batch offline
- GUI annotation tool — use Label Studio + supervision for format conversion
- Monocular (single-camera) reconstruction — geometrically ill-posed; biases architecture away from multi-view
- Appearance-based Re-ID — commit to shape-signature identity first; adding appearance Re-ID creates two competing identity systems

### Architecture Approach

The system is organized as a strict linear pipeline of five phases: Segmentation (Phase I) → 3D Initialization (Phase II) → Differentiable Refinement (Phase III) → Tracking and Identity (Phase IV) → Output and Visualization (Phase V). The critical architectural decision is how the refractive projection integrates with PyTorch3D: mesh vertices are pre-projected through the differentiable RefractiveProjector (Π_ref) before being handed to PyTorch3D's rasterizer, which then operates in distorted camera space. Gradients flow back from the silhouette loss through the rasterizer, through the pre-projected vertex positions, through Π_ref, and into the FishState parameters {p, ψ, κ, s}. This entire chain must be differentiable; breaking it at any point (e.g., using numpy inside Π_ref) silently breaks the optimizer without raising an error.

**Major components:**
1. CalibrationLoader + RefractiveProjector — parses AquaCal JSON; exposes differentiable per-camera Π_ref; must be built and validated first
2. InstanceSegmenter + KeypointExtractor — Detectron2 Mask R-CNN producing binary masks M_i^(j) per camera per fish per frame
3. EpipolarInitializer — refractive ray intersection via scipy.optimize.least_squares to estimate FishState {p, ψ, κ=0, s} on first frame only
4. FishMeshBuilder — pure PyTorch parametric mesh from FishState; midline spline + swept ellipses → watertight triangle mesh
5. RefractiveRenderer + LossComputer — PyTorch3D MeshRasterizer + SoftSilhouetteShader; multi-objective loss (silhouette IoU + gravity prior + shape prior + temporal smoothness)
6. PoseOptimizer — Adam with ~50–100 iterations per frame; warm-starts from previous frame's solution at 30fps
7. MotionPredictor + AssignmentSolver — filterpy EKF + scipy Hungarian algorithm for frame-to-frame identity assignment
8. TrajectoryWriter + Visualizer — h5py HDF5 output; rerun-sdk for live QA; pyvista for publication renders

**Key patterns:**
- Warm-start every frame from the previous frame's FishState (reduces iterations from ~500 to ~50–100)
- Batch all N fish into a single GPU call (PyTorch3D batched Meshes + Cameras)
- Design for N fish from day one — all function signatures accept lists; use `join_meshes_as_batch` not single-mesh APIs
- Cross-view holdout: withhold 1–2 cameras from gradient computation; evaluate IoU on them as a generalization metric

### Critical Pitfalls

1. **Non-differentiable refractive projection** — Implementing Π_ref using numpy or scipy breaks the gradient chain silently. The optimizer runs without error but physical gradients are lost. Implement Π_ref entirely in PyTorch (Newton-Raphson with fixed iterations using autograd-compatible operations). Verify AquaCal's differentiability before assuming it works with autograd.

2. **Depth-independent refraction model** — Using OpenCV's standard distortion model to approximate refraction produces systematic depth-dependent errors (fish near tank floor reproject worse than near-surface fish). The error is a consistent 3D bias, not noise, and invalidates reconstruction. Implement full per-ray Snell's law projection tracing through air-glass-water interface with correct refractive indices.

3. **Silhouette-only fitting converges to wrong local minimum** — Silhouette IoU loss is highly non-convex. Top-down cameras cannot disambiguate head-tail orientation (180° flip produces nearly identical silhouette). Must implement multi-start optimization for first frame of each track (4–8 orientation initializations, select lowest loss). Add coarse keypoint loss (head tip, tail tip) if detectable. Temporal smoothness regularization resists single-frame escapes.

4. **Top-down camera Z-weakness not quantified** — 13 cameras with nearly parallel optical axes create degenerate Z reconstruction. 2D reprojection error can look excellent while Z is wrong by centimeters. Quantify theoretical Z uncertainty bound before writing any optimization code. Validate 3D reconstruction accuracy on a physical reference at 3+ known depths. Report X, Y, Z errors separately — never report only aggregate reprojection error.

5. **Single-fish architecture blocking v2 extension** — Building v1 with global state (one mask, one optimizer, one mesh object) requires a full rewrite at v2. Design for N fish from day one: parameterized Fish class with per-instance state, batch-first PyTorch3D mesh operations, detection returning a list of per-fish masks even when length is 1.

## Implications for Roadmap

Based on combined research, a 5-phase build order is indicated by strict data dependencies and the need for validation gates before proceeding:

### Phase 1: Calibration and Refractive Geometry Foundation
**Rationale:** Everything downstream depends on a working, differentiable, validated RefractiveProjector. No optimization code is scientifically meaningful until the camera model is correct. Building this first prevents propagating a subtle calibration error through the entire system.
**Delivers:** CalibrationLoader parsing AquaCal JSON; differentiable RefractiveProjector (Π_ref) implementing per-ray Snell's law in PyTorch; unit tests covering central rays and edge-field rays at 30–48° incidence; validation showing < 1px reprojection error on known 3D points; quantified Z-uncertainty bounds for the 13-camera top-down geometry.
**Addresses features:** Camera calibration (refractive) [table stakes]
**Avoids pitfalls:** Non-differentiable Π_ref (Pitfall 1), depth-independent refraction model (Pitfall 2), port tilt unmodeled (Pitfall 6), Newton-Raphson edge instability (Pitfall 4), deferred calibration validation (Architecture Anti-Pattern 4)
**Research flag:** NEEDS RESEARCH — AquaCal's internal differentiability must be verified before assuming it integrates with autograd; the Newton-Raphson convergence behavior at near-critical angles needs empirical characterization on this rig's geometry.

### Phase 2: Segmentation Pipeline
**Rationale:** Segmentation is a prerequisite for both initialization and optimization; it produces the masks that drive every downstream phase. It can be built and validated independently of the rendering pipeline, which makes it an ideal early phase for parallelism with Phase 3 development.
**Delivers:** MOG2-based foreground detection with shadow suppression; Detectron2 Mask R-CNN trained on annotated frames (bootstrapped with SAM2); per-frame binary masks M_i^(j) per camera per fish; per-sex, per-behavior recall validation (males, females, stationary, edge-of-frame).
**Addresses features:** Multi-view silhouette extraction [table stakes]
**Avoids pitfalls:** MOG2 female fish dropout (Pitfall 5); missed per-sex validation
**Uses:** OpenCV 4.13 (MOG2), Detectron2, SAM2, supervision (format conversion), Label Studio (annotation QA)
**Research flag:** STANDARD PATTERNS — Detectron2 Mask R-CNN training is well-documented; SAM2 pseudo-label generation workflow is documented in Meta's release. The main unknowns are rig-specific (female contrast, lighting conditions) which require empirical tuning, not research.

### Phase 3: Single-Fish 3D Reconstruction (v1 Core)
**Rationale:** This is the core novelty and the v1 scientific deliverable. It must be built as a complete single-fish pipeline before any multi-fish extension. The build order within this phase follows the architectural dependency graph: FishMeshBuilder → RefractiveRenderer → LossComputer → EpipolarInitializer → PoseOptimizer.
**Delivers:** FishMeshBuilder producing watertight triangle meshes from FishState {p, ψ, κ, s}; RefractiveRenderer rendering per-camera silhouettes via Π_ref + PyTorch3D SoftSilhouetteShader; multi-objective LossComputer (L_sil + L_grav + L_shape + L_temp); EpipolarInitializer for first-frame cold start; PoseOptimizer (Adam, warm-start, multi-start for first frame); cross-view holdout validation showing IoU on held-out cameras.
**Addresses features:** Differentiable silhouette renderer, parametric fish mesh, single-fish optimization, cross-view holdout validation [all table stakes]
**Avoids pitfalls:** Silhouette local minima without multi-start (Pitfall 3), rotation gimbal lock (Pitfall 10), soft rasterizer hyperparameters (Pitfall 7), sequential camera rendering anti-pattern, Z-only reprojection validation
**Uses:** PyTorch3D 0.7.9, kornia (differentiable IoU), scipy (epipolar initialization)
**Research flag:** NEEDS RESEARCH — the specific sigma/gamma hyperparameters for PyTorch3D's soft rasterizer at this rig's fish pixel sizes are unknown and require empirical sweep. The interaction between the temporal loss term (L_temp) and warm-start stability at 30fps needs characterization. The head-tail disambiguation strategy (multi-start vs. keypoint loss) needs validation on actual footage.

### Phase 4: Trajectory Output and Validation
**Rationale:** Complete the v1 pipeline by adding output storage and establishing the evaluation framework with 3D (not just 2D) ground truth metrics. This phase makes the system scientifically publishable and validates the core claim before scaling to multi-fish.
**Delivers:** TrajectoryWriter with HDF5 output (per-fish, per-frame position, orientation, curvature, scale); 3D reconstruction accuracy metric validated on a physical reference object at 3+ depths; separate X/Y/Z error reporting; 2D overlay visualization via OpenCV; 3D QA visualization via rerun-sdk.
**Addresses features:** Per-frame trajectory output, reprojection error metric [table stakes]
**Avoids pitfalls:** Reprojection-only validation masking 3D failures (Pitfall 11); no ground truth measurement protocol
**Uses:** h5py, rerun-sdk, pyvista (publication renders), matplotlib
**Research flag:** STANDARD PATTERNS — HDF5 output and matplotlib analysis are fully documented. The main decision is the ground truth measurement protocol (physical target design), which is experimental design rather than software research.

### Phase 5: Multi-Fish Tracking and Identity (v2)
**Rationale:** Only after v1 single-fish reconstruction is validated (cross-view holdout IoU meets threshold) does multi-fish extension make sense. Scaling to 9 fish requires the tracking and identity layers that were deliberately deferred.
**Delivers:** Multi-fish batched optimization (9 fish per GPU call using PyTorch3D batched Meshes); MotionPredictor (filterpy EKF, 3D position + velocity state per fish); AssignmentSolver (scipy Hungarian with Mahalanobis cost + sex-penalty augmentation); InteractionHandler for merge-split events with N-fish topology constraint; shape-pose decomposition for shape-signature identity; identity persistence validation across simulated occlusion events.
**Addresses features:** Shape-pose decomposition, multi-fish tracking, identity via shape signatures [v1.x targets]
**Avoids pitfalls:** Single-fish architecture blocking v2 (Pitfall 9), shape/pose coupling (Pitfall 8)
**Uses:** filterpy, scipy (Hungarian), scikit-learn (sex classification), PyTorch3D batched rendering
**Research flag:** NEEDS RESEARCH — the shape-signature Re-ID approach has no direct precedent in the fish tracking literature. The clustering stability of shape parameters across individuals within the same sex class needs empirical validation before committing to it as the identity mechanism. The interaction handler logic for merge-split events in a 9-fish tank is complex and should be researched during phase planning.

### Phase Ordering Rationale

- **Calibration first:** The RefractiveProjector is a shared dependency for every subsequent phase; errors here compound downstream. Validating it first eliminates the largest source of systemic bias.
- **Segmentation second:** Masks are required by initialization and optimization; building and validating the segmentation pipeline in parallel with Phase 3 mesh/rendering development saves calendar time without creating blocking dependencies.
- **Single-fish before multi-fish:** The feature dependency graph is explicit — multi-fish tracking requires reliable single-fish reconstruction, and identity-by-shape requires shape-pose decomposition which requires a working single-fish mesh optimization.
- **Output and validation before scaling:** Establishing the 3D evaluation framework during v1 prevents inheriting unvalidated reconstruction into the more complex multi-fish system.
- **Multi-fish last:** The identity mechanism (shape signatures) is the highest-risk novel contribution. Deferring it until v1 is proven reduces the risk that identity failures are confused with reconstruction failures during debugging.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (Calibration):** AquaCal's internal PyTorch differentiability needs verification; Newton-Raphson convergence characterization at near-critical angles is rig-specific.
- **Phase 3 (Reconstruction Core):** PyTorch3D soft rasterizer hyperparameter tuning; head-tail disambiguation strategy; temporal loss stability with warm-start.
- **Phase 5 (Multi-Fish / Identity):** Shape-signature Re-ID stability across individuals has no direct literature precedent; merge-split interaction handling logic is complex.

Phases with standard patterns (skip research-phase):
- **Phase 2 (Segmentation):** Detectron2 + SAM2 workflow is well-documented; unknowns are empirical (rig-specific), not conceptual.
- **Phase 4 (Output and Validation):** HDF5 output and visualization are fully standard; the ground truth protocol is experimental design, not software research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | PyTorch/PyTorch3D version pinning is HIGH confidence (official docs confirmed). Detectron2, SAM2, kornia are HIGH confidence. The 5-version compatibility gap between PyTorch 2.4.1 and current (2.10.0) introduces real installation risk. |
| Features | MEDIUM | Table stakes and anti-features are HIGH confidence (competitor analysis is clear). Differentiators (refractive rendering, shape Re-ID) have no direct comparators, so feature scope is validated by domain logic rather than industry patterns. |
| Architecture | HIGH | System design is confirmed by the detailed project spec in `.planning/proposed_pipeline.md`. PyTorch3D rendering architecture verified via official docs. Build order derives from clear data dependencies. |
| Pitfalls | MEDIUM | Core optics and math pitfalls (refractive distortion, Z-weakness, silhouette ambiguity) are HIGH confidence from peer-reviewed literature. Multi-fish extension pitfalls are MEDIUM from analogous animal tracking work. Some implementation specifics (shape Re-ID stability) are LOW — flagged. |

**Overall confidence:** MEDIUM

### Gaps to Address

- **AquaCal differentiability:** It is not confirmed whether AquaCal's RefractiveProjector exposes a PyTorch-differentiable `project()` method or requires reimplementation in PyTorch. This must be verified before Phase 1 is planned. If AquaCal is numpy-based, the refractive projection must be reimplemented from scratch in PyTorch — a significant scope addition.
- **Z-uncertainty budget:** The theoretical Z reconstruction uncertainty for this specific rig (13 top-down cameras, baseline distances, operating depth) has not been quantified. This must be computed analytically in Phase 1 before committing to the analysis-by-synthesis approach for Z.
- **Shape-signature discriminability:** It is unknown whether the shape parameters {κ, s} estimated for each fish in the tank are sufficiently distinct to serve as a biometric identifier across a full-day recording. This is the core scientific bet of v2. The gap should be flagged in the roadmap and a validation experiment designed early in Phase 5.
- **PyTorch3D sigma/gamma for this rig:** Optimal soft rasterizer hyperparameters depend on fish apparent size in pixels, which depends on camera distance and focal length. These values must be swept empirically during Phase 3 development.
- **Female detection under worst-case conditions:** MOG2 recall for stationary female fish has not been measured. This is the most likely operational failure mode for Phase 2.

## Sources

### Primary (HIGH confidence)
- PyTorch3D INSTALL.md — version compatibility matrix (PyTorch 2.4.1 + PyTorch3D 0.7.9 confirmed): https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- PyTorch3D renderer docs — SoftSilhouetteShader, MeshRasterizer, batched API: https://pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/rasterizer.html
- AquaPose proposed_pipeline.md — authoritative project spec (internal document)
- Refractive Two-View Reconstruction for Underwater 3D Vision (IJCV 2019) — refractive model correctness requirements: https://link.springer.com/article/10.1007/s11263-019-01218-9
- Multi-animal pose estimation and tracking with DeepLabCut (Nature Methods 2022) — competitor baseline: https://www.nature.com/articles/s41592-022-01443-0
- WaterMask: Instance Segmentation for Underwater Imagery (ICCV 2023) — underwater segmentation precedent: https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf
- OpenCV MOG2 documentation: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

### Secondary (MEDIUM confidence)
- A Calibration Tool for Refractive Underwater Vision (arXiv 2024) — port tilt estimation, refractive calibration: https://arxiv.org/abs/2405.18018
- VoGE: Differentiable Volume Renderer for Analysis-by-Synthesis (OpenReview) — confirms analysis-by-synthesis pattern: https://openreview.net/forum?id=AdPJb9cud_Y
- SOD-SORT: Multi-fish tracking with EKF + Hungarian — confirms tracking pattern: https://arxiv.org/html/2507.06400v3
- vmTracking: multi-animal pose tracking (PLOS Biology 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC11845028/
- PyTorch3D GitHub issues #1626, #905, #1855 — soft rasterizer hyperparameter behavior: https://github.com/facebookresearch/pytorch3d/issues/1626
- Adventures with Differentiable Mesh Rendering (Andrew Chan blog) — practical implementation gotchas

### Tertiary (LOW confidence)
- Shape-signature identity for fish Re-ID — no direct precedent found; extrapolated from SMAL-based shape Re-ID for quadrupeds. Needs validation during Phase 5 planning.
- PyTorch3D source build stability against PyTorch 2.5–2.10 — community reports in GitHub issues; no official confirmation. Treat as LOW until PyTorch3D publishes updated release.

---
*Research completed: 2026-02-19*
*Ready for roadmap: yes*
