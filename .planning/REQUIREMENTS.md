# Requirements: AquaPose

**Defined:** 2026-02-19
**Core Value:** Accurate single-fish 3D reconstruction from multi-view silhouettes via differentiable refractive rendering

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Calibration & Geometry

- [ ] **CALIB-01**: System can load AquaCal calibration data (intrinsics, extrinsics, refractive model params) from JSON
- [ ] **CALIB-02**: System provides differentiable refractive projection (3D world point → 2D pixel) in PyTorch, returning both pixel coordinates and depth
- [ ] **CALIB-03**: System provides differentiable ray casting (2D pixel → 3D underwater ray) in PyTorch
- [ ] **CALIB-04**: System quantifies Z-reconstruction uncertainty bounds for the 13-camera top-down geometry at multiple tank depths (one-time rig characterization, not per-frame)

### Segmentation

- [x] **SEG-01**: System detects fish via MOG2 background subtraction with shadow suppression, producing padded bounding boxes with ≥95% per-camera recall (including females). Without tracker safety net, this recall target is load-bearing.
- [ ] **SEG-02**: System generates pseudo-label masks by feeding bounding boxes as prompts to SAM single-frame
- [ ] **SEG-03**: System supports Label Studio annotation workflow (export images + pseudo-labels, import corrected masks)
- [x] **SEG-04**: System segments fish on cropped and resized patches (256×256) from detection bounding boxes, not full 1600×1200 frames
- [ ] **SEG-05**: System trains Mask R-CNN (Detectron2) on corrected crop annotations, achieving ≥0.90 mean mask IoU (≥0.85 for females)

### Fish Model

- [ ] **MESH-01**: System generates a parametric fish mesh from state vector {p, ψ, κ, s} via midline spline + swept cross-section ellipses, fully differentiable in PyTorch
- [ ] **MESH-02**: System supports free cross-section mode where per-section height/width are optimizable parameters for shape profile self-calibration
- [ ] **MESH-03**: System initializes 3D fish state via epipolar consensus from coarse keypoints (head, center, tail) using refractive ray intersection

### Reconstruction

- [ ] **RECON-01**: System renders differentiable silhouettes of the fish mesh into each camera view via refractive projection + PyTorch3D rasterizer, with per-camera weighting by angular diversity to prevent clustered ring cameras from dominating
- [ ] **RECON-02**: System computes multi-objective loss: silhouette IoU + gravity prior + morphological constraint first, then temporal smoothness once tracking provides frame-to-frame associations
- [ ] **RECON-03**: System runs 2-initialization multi-start (forward + 180° flip) on first frame of each track to resolve head-tail ambiguity
- [ ] **RECON-04**: System optimizes per-frame fish pose via Adam with warm-start from previous frame's solution
- [ ] **RECON-05**: System validates reconstruction via cross-view holdout (fit on N-k cameras, evaluate IoU on k held-out cameras), achieving ≥0.80 mean holdout IoU

### Tracking

- [ ] **TRACK-01**: System maintains per-fish 3D Extended Kalman Filter (position + velocity) with anisotropic process noise (higher Z uncertainty)
- [ ] **TRACK-02**: System associates detections to tracks per frame via Hungarian algorithm with Mahalanobis distance cost and gating threshold
- [ ] **TRACK-03**: System injects predicted bounding boxes from 3D tracker into detection stage for fish not detected by background subtraction (tracker safety net). Frame 1 runs without safety net; frame 2+ uses previous frame's track predictions.
- [ ] **TRACK-04**: System enforces population constraint (exactly 9 fish at all times). If a track is lost and a new detection appears in the same frame window, they are linked.

### Sex Classification

- [ ] **SEX-01**: System classifies each detected fish as male or female from color histogram features extracted from masked crops, using a simple classifier trained on labeled examples
- [ ] **SEX-02**: System aggregates per-frame sex classifications across cameras and over time to assign a stable sex label per track
- [ ] **SEX-03**: System adds a sex-mismatch penalty to the Hungarian association cost matrix to prevent male-female track swaps

### Output

- [ ] **OUT-01**: System stores per-frame pose trajectories in HDF5 including: fish_id, position, heading, midline, curvature, scale, sex, confidence, n_cameras, silhouette_loss
- [ ] **OUT-02**: System overlays projected 3D mesh onto original camera views for 2D visual QA
- [ ] **OUT-03**: System renders fish meshes in the tank volume in 3D via rerun-sdk with trajectory trails and identity coloring

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

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
| Real-time processing | Analysis-by-synthesis is iterative gradient descent; batch offline only |
| GUI annotation tool | Use Label Studio + supervision; building a GUI is orthogonal to reconstruction |
| Monocular reconstruction | Geometrically ill-posed; biases architecture away from multi-view |
| Appearance-based Re-ID | Commit to shape-signature identity; appearance Re-ID fails under view changes for visually similar cichlids |
| SMAL-based animal model | Fish are not quadrupeds; midline spline + cross-sections is the correct topology |
| Cloud/multi-user infrastructure | Single-lab research tool; share results as files |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CALIB-01 | Phase 1 | Pending |
| CALIB-02 | Phase 1 | Pending |
| CALIB-03 | Phase 1 | Pending |
| CALIB-04 | Phase 1 | Pending |
| SEG-01 | Phase 2 | Complete |
| SEG-02 | Phase 2 | Pending |
| SEG-03 | Phase 2 | Pending |
| SEG-04 | Phase 2 | Complete |
| SEG-05 | Phase 2 | Pending |
| MESH-01 | Phase 3 | Pending |
| MESH-02 | Phase 3 | Pending |
| MESH-03 | Phase 3 | Pending |
| RECON-01 | Phase 4 | Pending |
| RECON-02 | Phase 4 | Pending |
| RECON-03 | Phase 4 | Pending |
| RECON-04 | Phase 4 | Pending |
| RECON-05 | Phase 4 | Pending |
| TRACK-01 | Phase 5 | Pending |
| TRACK-02 | Phase 5 | Pending |
| TRACK-03 | Phase 5 | Pending |
| TRACK-04 | Phase 5 | Pending |
| SEX-01 | Phase 5 | Pending |
| SEX-02 | Phase 5 | Pending |
| SEX-03 | Phase 5 | Pending |
| OUT-01 | Phase 6 | Pending |
| OUT-02 | Phase 6 | Pending |
| OUT-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0

---
*Requirements defined: 2026-02-19*
*Last updated: 2026-02-19 — Traceability completed during roadmap creation*
