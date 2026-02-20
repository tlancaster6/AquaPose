# Roadmap: AquaPose

## Overview

AquaPose is built in strict dependency order: the refractive camera model must be validated before any optimization code is written, segmentation masks must exist before initialization is possible, a working parametric mesh model must precede differentiable rendering, and single-fish reconstruction must be validated before tracking and identity layers are added. Six phases follow this natural dependency chain — each phase delivers a coherent, testable capability that the next phase depends on.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Calibration and Refractive Geometry** - Differentiable refractive projection validated and ready for use by all downstream phases
- [ ] **Phase 2: Segmentation Pipeline** - Multi-view binary masks ready for every frame; annotation workflow established
- [ ] **Phase 3: Fish Mesh Model and 3D Initialization** - Parametric fish mesh differentiable in PyTorch; first-frame cold-start working from coarse keypoints
- [ ] **Phase 4: Single-Fish Reconstruction** - Per-frame pose optimization converges on real data; cross-view holdout IoU meets threshold
- [ ] **Phase 5: Tracking and Sex Classification** - Frame-to-frame track continuity with temporal smoothness loss active; population constraint enforced
- [ ] **Phase 6: Output and Visualization** - Per-frame trajectories written to HDF5; 2D overlay and 3D rerun visualization operational

## Phase Details

### Phase 1: Calibration and Refractive Geometry
**Goal**: A validated, differentiable refractive projection layer is available that downstream phases can import with confidence — errors here would silently corrupt every gradient in the system
**Depends on**: Nothing (first phase)
**Requirements**: CALIB-01, CALIB-02, CALIB-03, CALIB-04
**Success Criteria** (what must be TRUE):
  1. Given a known 3D point, the system reprojects it to pixel coordinates matching ground truth to within 1px for all 13 cameras across the full tank depth range
  2. Gradients flow through the refractive projection (autograd backward pass completes without error; numerical gradient check passes for depth, x, y inputs)
  3. Given a 2D pixel, ray casting produces a 3D underwater ray direction consistent with Snell's law at the air-water interface
  4. A one-time Z-uncertainty characterization report exists for the 13-camera top-down geometry, quantifying expected X/Y/Z reconstruction error at 3+ tank depths separately
**Plans**: 2 plans
- [x] 01-01-PLAN.md — Port calibration loader and refractive projection model from AquaMVS with cross-validation tests
- [x] 01-02-PLAN.md — Z-uncertainty analytical characterization report

### Phase 2: Segmentation Pipeline
**Goal**: The system can produce corrected binary fish masks for any input frame across all 13 cameras, achieving recall targets even for low-contrast females
**Depends on**: Phase 1
**Requirements**: SEG-01, SEG-02, SEG-03, SEG-04, SEG-05
**Success Criteria** (what must be TRUE):
  1. MOG2 detection produces padded bounding boxes with ≥95% per-camera recall measured on a held-out sample including female fish and stationary subjects
  2. SAM2 pseudo-labels are generated via box-only prompting with quality filtering, producing masks suitable for direct Mask R-CNN training without manual correction
  3. A trained Mask R-CNN model produces binary mask predictions on variable-size bbox crops achieving ≥0.90 mean mask IoU on the validation split
  4. The segmentation pipeline accepts N fish as input (returning a list of masks) even in the single-fish v1 operating mode
**Plans**: 4 plans
- [ ] 02-01-PLAN.md — Delete Label Studio code, remove debug scripts, clean dependencies
- [ ] 02-02-PLAN.md — Update pseudo-labeler (box-only SAM2, quality filtering) and dataset (variable crops, stratified split)
- [ ] 02-03-PLAN.md — Update Mask R-CNN (separate detect/crop/segment stages, crop-space output, variable crops)
- [ ] 02-04-PLAN.md — Integration wiring and human verification of full pipeline

### Phase 02.1: Segmentation Troubleshooting (INSERTED)

**Goal:** Systematically test each Phase 2 segmentation component (MOG2, SAM2, Mask R-CNN) on real data, diagnose failures, and fix until quality is sufficient to unblock Phase 4
**Depends on:** Phase 2
**Requirements:** SEG-1, SEG-2, SEG-3, SEG-4
**Plans:** 3/4 plans executed

Plans:
- [ ] 02.1-01-PLAN.md — Consolidate MOG2 diagnostic scripts, validate recall on all 13 cameras
- [ ] 02.1-02-PLAN.md — SAM2 pseudo-label evaluation against manual ground truth
- [ ] 02.1-03-PLAN.md — Mask R-CNN training on Label Studio corrections and evaluation

### Phase 02.1.1: Object-detection alternative to MOG2 (INSERTED)

**Goal:** Replace MOG2 background subtraction with a YOLOv8 object detector as a runtime-configurable alternative first-stage fish localizer, trained on 150 manually annotated frames, achieving higher recall than MOG2 on a stratified validation set
**Depends on:** Phase 02.1
**Requirements:** SEG-01, SEG-04
**Plans:** 1/3 plans complete

Plans:
- [x] 02.1.1-01-PLAN.md — MOG2-guided frame sampling and Label Studio bounding box annotation
- [x] 02.1.1-02-PLAN.md — YOLOv8 training, YOLODetector implementation, and comparative evaluation vs MOG2
- [x] 02.1.1-03-PLAN.md — Pipeline integration: wire YOLODetector into SAM2 pseudo-labeler and full segmentation workflow

### Phase 3: Fish Mesh Model and 3D Initialization
**Goal**: A fully differentiable parametric fish mesh model exists and can produce a plausible first-frame 3D state estimate from coarse keypoint detections, ready to be handed to the optimizer
**Depends on**: Phase 1
**Requirements**: MESH-01, MESH-02, MESH-03
**Success Criteria** (what must be TRUE):
  1. Given a fish state vector {p, ψ, κ, s}, the mesh builder produces a watertight triangle mesh and gradients flow back through all four state components
  2. Free cross-section mode allows per-section height and width to be optimizable parameters (shape profile self-calibration runs without error)
  3. Given coarse head/center/tail keypoints from at least 3 cameras, epipolar initialization produces a 3D state estimate within plausible range of the fish's actual position (visually reasonable when overlaid on camera views)
  4. All mesh and initialization APIs accept lists of fish states (batch-first design) even when called with a single-element list
**Plans**: 2 plans
Plans:
- [ ] 03-01-PLAN.md — Differentiable parametric fish mesh (FishState, spine, cross-sections, builder, PyTorch3D Meshes)
- [ ] 03-02-PLAN.md — PCA keypoint extraction and refractive triangulation for 3D initialization

### Phase 4: Single-Fish Reconstruction
**Goal**: The full analysis-by-synthesis loop works end-to-end on real data — a single fish's pose is recovered frame-by-frame with cross-view holdout IoU demonstrating the system generalizes beyond the cameras it was fit on
**Depends on**: Phase 2, Phase 3
**Requirements**: RECON-01, RECON-02, RECON-03, RECON-04, RECON-05
**Success Criteria** (what must be TRUE):
  1. Differentiable silhouettes of the fish mesh render correctly into each camera view via refractive projection + PyTorch3D rasterizer, with per-camera angular-diversity weighting applied
  2. The multi-objective loss computes silhouette IoU + gravity prior + morphological constraint terms; temporal smoothness term activates once tracking associations are available (see Phase 5), but the loss is architecturally ready for it
  3. First-frame optimization runs 2-start (forward + 180° flip) and selects the lower-loss result, resolving head-tail ambiguity on real footage
  4. Frame-by-frame warm-start optimization converges in ≤100 Adam iterations on frames after the first, producing visually plausible reconstructions
  5. Cross-view holdout validation achieves ≥0.80 mean IoU on held-out cameras across a representative clip
**Plans**: TBD

### Phase 5: Tracking and Sex Classification
**Goal**: Fish identities persist frame-to-frame, the temporal smoothness loss becomes active (completing the multi-objective loss), and the population constraint prevents track count from deviating from 9
**Depends on**: Phase 4
**Requirements**: TRACK-01, TRACK-02, TRACK-03, TRACK-04, SEX-01, SEX-02, SEX-03
**Success Criteria** (what must be TRUE):
  1. A 3D Extended Kalman Filter maintains position + velocity state per fish with anisotropic process noise; the filter's predicted bounding box is injected into detection for frame 2+ (tracker safety net active)
  2. Hungarian assignment with Mahalanobis distance cost successfully links detections to tracks across a test clip; the sex-mismatch cost penalty visibly reduces male-female swap events compared to baseline
  3. Temporal smoothness loss activates via the track associations provided by the EKF, and its gradient flows back through pose parameters without breaking the optimizer
  4. Population constraint logic links a lost track to a new detection in the same frame window, keeping active track count at 9 throughout a test clip with no persistent occlusions
  5. Each track carries a stable sex label derived from per-frame color histogram classifier votes aggregated across cameras and time
**Plans**: TBD

### Phase 6: Output and Visualization
**Goal**: Results are persisted in a machine-readable format and inspectable via 2D overlays and 3D visualization, making the system usable for downstream behavioral research
**Depends on**: Phase 5
**Requirements**: OUT-01, OUT-02, OUT-03
**Success Criteria** (what must be TRUE):
  1. An HDF5 file is written containing per-frame, per-fish records with all required fields: fish_id, position, heading, midline, curvature, scale, sex, confidence, n_cameras, silhouette_loss — and the file is readable by standard h5py code without custom schemas
  2. 2D overlay video (projected 3D mesh on original camera frames) can be generated for any clip and any subset of cameras for visual QA
  3. 3D rerun-sdk visualization shows fish meshes moving through the tank volume with trajectory trails and identity coloring, rendering in real time from the HDF5 output
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

Note: Phase 3 depends only on Phase 1 (not Phase 2), so Phases 2 and 3 can develop in parallel if needed. Phase 4 requires both Phase 2 and Phase 3 to be complete.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Calibration and Refractive Geometry | 2/2 | Complete | 2026-02-19 |
| 2. Segmentation Pipeline | 0/4 | Planning complete | - |
| 02.1 Segmentation Troubleshooting | 1/3 | In progress | - |
| 02.1.1 Object-detection alternative to MOG2 | 2/3 | Complete    | 2026-02-20 |
| 3. Fish Mesh Model and 3D Initialization | 0/2 | Planning complete | - |
| 4. Single-Fish Reconstruction | 0/TBD | Not started | - |
| 5. Tracking and Sex Classification | 0/TBD | Not started | - |
| 6. Output and Visualization | 0/TBD | Not started | - |
