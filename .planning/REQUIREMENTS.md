# Requirements: AquaPose

**Defined:** 2026-02-19
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Calibration & Geometry

- [ ] **CALIB-01**: System can load AquaCal calibration data (intrinsics, extrinsics, refractive model params) from JSON
- [ ] **CALIB-02**: System provides differentiable refractive projection (3D world point → 2D pixel) in PyTorch, returning both pixel coordinates and depth
- [ ] **CALIB-03**: System provides differentiable ray casting (2D pixel → 3D underwater ray) in PyTorch
- [ ] **CALIB-04**: System quantifies Z-reconstruction uncertainty bounds for the 13-camera top-down geometry at multiple tank depths (one-time rig characterization, not per-frame)

### Segmentation

- [x] **SEG-01**: System detects fish via MOG2 background subtraction with shadow suppression, producing padded bounding boxes with ≥95% per-camera recall (including females). Without tracker safety net, this recall target is load-bearing.
- [x] **SEG-02**: System generates pseudo-label masks by feeding bounding boxes as prompts to SAM single-frame
- [x] **SEG-03**: System generates quality-filtered pseudo-label masks directly from SAM2 (box-only prompt), bypassing manual annotation — quality filtering replaces human correction
- [x] **SEG-04**: System segments fish on cropped patches from detection bounding boxes (variable crop dimensions matching bbox), not full 1600×1200 frames
- [x] **SEG-05**: System trains Mask R-CNN on pseudo-label crop annotations, achieving ≥0.90 mean mask IoU on validation split

### Fish Model

- [ ] **MESH-01**: System generates a parametric fish mesh from state vector {p, ψ, κ, s} via midline spline + swept cross-section ellipses, fully differentiable in PyTorch
- [ ] **MESH-02**: System supports free cross-section mode where per-section height/width are optimizable parameters for shape profile self-calibration
- [ ] **MESH-03**: System initializes 3D fish state via epipolar consensus from coarse keypoints (head, center, tail) using refractive ray intersection

### Reconstruction (Direct Triangulation Pipeline)

- [ ] **RECON-01**: System extracts 2D medial axis from binary masks via morphological smoothing + skeletonization + longest-path BFS pruning, producing an ordered head-to-tail midline with local half-widths from the distance transform
- [ ] **RECON-02**: System resamples 2D midlines at N fixed normalized arc-length positions (head=0, tail=1), producing consistent cross-view correspondences with coordinate transform from crop space to full-frame pixels
- [ ] **RECON-03**: System triangulates each of the N body positions across cameras via refractive ray intersection with per-point RANSAC and view-angle weighting to reject arc-length correspondence outliers
- [ ] **RECON-04**: System fits a cubic B-spline (5–8 control points) through the N triangulated 3D points, plus a 1D width-profile spline, producing a continuous 3D midline + tube model per fish per frame
- [ ] **RECON-05**: *(Optional, add only if baseline insufficient)* System refines 3D spline control points via Levenberg-Marquardt minimization of reprojection error against 2D medial axis observations across all cameras

### Reconstruction — Shelved (Analysis-by-Synthesis)

*Shelved with Phase 4. Code retained as optional advanced route.*

- [x] **RECON-ABS-01**: System renders differentiable silhouettes of the fish mesh into each camera view via refractive projection + PyTorch3D rasterizer, with per-camera weighting by angular diversity
- [x] **RECON-ABS-02**: System computes multi-objective loss: silhouette IoU + gravity prior + morphological constraint + temporal smoothness
- [x] **RECON-ABS-03**: System runs 2-initialization multi-start (forward + 180° flip) on first frame to resolve head-tail ambiguity
- [x] **RECON-ABS-04**: System optimizes per-frame fish pose via Adam with warm-start from previous frame
- [x] **RECON-ABS-05**: System validates reconstruction via cross-view holdout, achieving ≥0.80 mean holdout IoU

### Cross-View Identity and Tracking

- [x] **TRACK-01**: System associates detections across cameras to physical fish via RANSAC-based centroid ray clustering — casting refractive rays from 2D centroids, triangulating minimal camera subsets, and scoring consensus against remaining cameras
- [x] **TRACK-02**: System produces a 3D centroid per fish per frame with reprojection residual; high-residual associations are flagged for downstream quality checks
- [x] **TRACK-03**: System assigns persistent fish IDs across frames via Hungarian algorithm on 3D centroid distances, leveraging the fact that fish rarely swap positions in 3D even when they overlap in individual 2D views
- [x] **TRACK-04**: System enforces population constraint (exactly 9 fish at all times). If a track is lost and a new detection appears in the same frame window, they are linked

### Output

- [ ] **OUT-01**: System stores per-frame results in HDF5 including: fish_id, 3D spline control points, width profile, centroid position, heading, curvature, n_cameras, triangulation residual — readable by standard h5py without custom schemas
- [ ] **OUT-02**: System overlays reprojected 3D midline + width profile onto original camera views for 2D visual QA
- [ ] **OUT-03**: System renders 3D midline tube models in the tank volume via rerun-sdk with trajectory trails and identity coloring

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Sex Classification

- **SEX-01**: System classifies each detected fish as male or female from color histogram features extracted from masked crops, using a simple classifier trained on labeled examples
- **SEX-02**: System aggregates per-frame sex classifications across cameras and over time to assign a stable sex label per track
- **SEX-03**: System adds a sex-mismatch penalty to the Hungarian association cost matrix to prevent male-female track swaps

### Identity & Shape

- **SHAPE-01**: System decomposes mesh parameters into identity-linked shape (body plan, size) and instantaneous pose (position, orientation, bend)
- **SHAPE-02**: System assigns persistent identity via shape signatures across full-day recordings
- **SHAPE-03**: System fits sex-differentiated shape profiles (separate male/female cross-section templates)

### Robustness

- **ROBUST-01**: System handles merge-and-split interaction events when fish detections become inseparable
- **ROBUST-02**: System mines hard examples from production runs and feeds them back into segmentation training

### Scale

- **SCALE-01**: System processes full-day recordings (8+ hours) with streaming/checkpointing
- **SCALE-02**: System provides batch processing infrastructure for full experimental datasets

### Analysis

- **ANALYSIS-01**: System extracts behavioral features from 3D body state (tail-beat frequency, curvature, approach angle, inter-fish distance)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Real-time processing | Batch offline only; real-time not a research requirement |
| GUI annotation tool | Use Label Studio + supervision; building a GUI is orthogonal to reconstruction |
| Monocular reconstruction | Geometrically ill-posed; biases architecture away from multi-view |
| Appearance-based Re-ID | Commit to shape-signature identity; appearance Re-ID fails under view changes for visually similar cichlids |
| SMAL-based animal model | Fish are not quadrupeds; midline spline + cross-sections is the correct topology |
| Cloud/multi-user infrastructure | Single-lab research tool; share results as files |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CALIB-01 | Phase 1 | Complete |
| CALIB-02 | Phase 1 | Complete |
| CALIB-03 | Phase 1 | Complete |
| CALIB-04 | Phase 1 | Complete |
| SEG-01 | Phase 2 | Complete |
| SEG-02 | Phase 2 | Complete |
| SEG-03 | Phase 2 | Complete |
| SEG-04 | Phase 2 | Complete |
| SEG-05 | Phase 2 | Complete |
| MESH-01 | Phase 3 | Complete |
| MESH-02 | Phase 3 | Complete |
| MESH-03 | Phase 3 | Complete |
| RECON-ABS-01..05 | Phase 4 (shelved) | Complete |
| RECON-01 | Phase 6 | Pending |
| RECON-02 | Phase 6 | Pending |
| RECON-03 | Phase 7 (TBD) | Pending |
| RECON-04 | Phase 7 (TBD) | Pending |
| RECON-05 | Phase 7 (TBD) | Pending |
| TRACK-01 | Phase 5 | Complete |
| TRACK-02 | Phase 5 | Complete |
| TRACK-03 | Phase 5 | Complete |
| TRACK-04 | Phase 5 | Complete |
| OUT-01 | Phase 8 (TBD) | Pending |
| OUT-02 | Phase 8 (TBD) | Pending |
| OUT-03 | Phase 8 (TBD) | Pending |

**Coverage:**
- v1 requirements: 25 (excluding shelved RECON-ABS)
- Mapped to phases: 25 (Phase 7–8 TBD, will be added via /gsd:add-phase)
- Unmapped: 0
- Deferred to v2: SEX-01, SEX-02, SEX-03

---
*Requirements defined: 2026-02-19*
*Last updated: 2026-02-21 — Reconstruction pivot: shelved RECON-ABS (Phase 4), new RECON-01..05 for direct triangulation, rewritten TRACK-01..04 for 3D identity/tracking, SEX-* deferred to v2, OUT-* updated for spline output*
