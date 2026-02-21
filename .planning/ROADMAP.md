# Roadmap: AquaPose

## Overview

AquaPose is built in strict dependency order: the refractive camera model must be validated before any optimization code is written, segmentation masks must exist before initialization is possible, a working parametric mesh model must precede differentiable rendering, and per-fish reconstruction must be validated before tracking and identity layers are added. Six phases follow this natural dependency chain — each phase delivers a coherent, testable capability that the next phase depends on.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Calibration and Refractive Geometry** - Differentiable refractive projection validated and ready for use by all downstream phases
- [x] **Phase 2: Segmentation Pipeline** - Multi-view binary masks ready for every frame; annotation workflow established
- [x] **Phase 3: Fish Mesh Model and 3D Initialization** - Parametric fish mesh differentiable in PyTorch; first-frame cold-start working from coarse keypoints
- [x] **Phase 4: Per-Fish Reconstruction** - ~~Analysis-by-synthesis mesh optimization~~ *(shelved 2026-02-21 — primary workflow replaced by direct triangulation pipeline, Phases 5+)*
- [ ] **Phase 5: Cross-View Identity and 3D Tracking** - RANSAC-based cross-view identity association and persistent 3D fish tracking via Hungarian assignment
- [ ] **Phase 6: 2D Medial Axis and Arc-Length Sampling** - Skeleton-based midline extraction from masks with normalized arc-length correspondence across views

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
  3. A trained U-Net model produces binary mask predictions on resized bbox crops (best val IoU: 0.623 — below 0.90 target but accepted as sufficient to unblock Phase 4; can revisit with more training data)
  4. The segmentation pipeline accepts N fish as input (returning a list of masks) even in the single-fish v1 operating mode
**Plans**: 4 plans
- [x] 02-01-PLAN.md — Delete Label Studio code, remove debug scripts, clean dependencies
- [x] 02-02-PLAN.md — Update pseudo-labeler (box-only SAM2, quality filtering) and dataset (variable crops, stratified split)
- [x] 02-03-PLAN.md — Update Mask R-CNN (separate detect/crop/segment stages, crop-space output, variable crops)
- [x] 02-04-PLAN.md — Integration wiring and human verification of full pipeline

### Phase 02.1: Segmentation Troubleshooting (INSERTED)

**Goal:** Systematically test each Phase 2 segmentation component on real data, diagnose failures, and fix until quality is sufficient to unblock Phase 4
**Depends on:** Phase 2
**Requirements:** SEG-1, SEG-2, SEG-3, SEG-4
**Plans:** 3/3 plans executed

Plans:
- [x] 02.1-01-PLAN.md — Consolidate MOG2 diagnostic scripts, validate recall on all 13 cameras
- [x] 02.1-02-PLAN.md — SAM2 pseudo-label evaluation against manual ground truth
- [x] 02.1-03-PLAN.md — Replace Mask R-CNN with lightweight U-Net; train and evaluate on pseudo-labels (best val IoU: 0.623 — accepted as sufficient to unblock Phase 4)

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
- [x] 03-01-PLAN.md — Differentiable parametric fish mesh (FishState, spine, cross-sections, builder, PyTorch3D Meshes)
- [x] 03-02-PLAN.md — PCA keypoint extraction and refractive triangulation for 3D initialization

### Phase 4: Per-Fish Reconstruction *(Archived)*
**Status**: Shelved 2026-02-21. The analysis-by-synthesis approach (differentiable mesh rendering + Adam optimization) was functionally complete but took 30+ minutes per second of video, making it impractical as the primary pipeline. Replaced by a direct triangulation pipeline (Phases 5–7) for main workflow. Code from Phase 3 (mesh model, refractive projection) is retained; the renderer and optimizer remain available as an optional advanced route.
**Requirements**: RECON-ABS-01, RECON-ABS-02, RECON-ABS-03, RECON-ABS-04, RECON-ABS-05
**Plans**: 3 plans (not executed)
Plans:
- [x] 04-01-PLAN.md — *(shelved)* Differentiable silhouette renderer and multi-objective loss
- [x] 04-02-PLAN.md — *(shelved)* FishOptimizer with 2-start first-frame and warm-start optimization
- [x] 04-03-PLAN.md — *(shelved)* Cross-view holdout validation

### Phase 04.1: Isolate phase4-specific code post-archive (INSERTED)

**Goal:** Archive the analysis-by-synthesis codebase to a dedicated branch, then strip all Phase 4-specific code from the working branch so only shared infrastructure remains for Phase 5+
**Depends on:** Phase 4
**Requirements:** RECON-ABS-01, RECON-ABS-02, RECON-ABS-03, RECON-ABS-04, RECON-ABS-05
**Plans:** 1/1 plans complete

Plans:
- [x] 04.1-01-PLAN.md — Archive ABS code to branch, delete optimization module/tests/scripts, verify clean build

### Phase 5: Cross-View Identity and 3D Tracking
**Goal**: Given per-camera detections, determine which masks across cameras correspond to the same physical fish, and maintain persistent fish IDs across frames — providing the cross-view identity mapping that all downstream reconstruction stages depend on
**Depends on**: Phase 2 (segmentation), Phase 1 (refractive ray model)
**Requirements**: TRACK-01, TRACK-02, TRACK-03, TRACK-04
**Success Criteria** (what must be TRUE):
  1. For each frame, RANSAC-based centroid ray clustering correctly associates detections across cameras to physical fish, producing a (camera_id, detection_id) → fish_id mapping with ≥ 2 cameras per fish
  2. Triangulated 3D centroid per fish has a reprojection residual below a defined threshold; high-residual associations are flagged for downstream quality checks
  3. Hungarian assignment in 3D space maintains persistent fish IDs across frames on a test clip; track count remains stable at the known population size (9 fish) with no persistent ID swaps
  4. The identity module exposes a clean interface consumed by downstream stages: given a frame's detections, returns per-fish camera sets, 3D centroids, and persistent IDs
**Plans**: 3 plans

Plans:
- [ ] 05-01-PLAN.md — RANSAC centroid ray clustering for cross-view association (TRACK-01, TRACK-02)
- [ ] 05-02-PLAN.md — FishTracker with Hungarian assignment and track lifecycle (TRACK-03, TRACK-04)
- [ ] 05-03-PLAN.md — HDF5 serialization and tracking module public API

### Phase 6: 2D Medial Axis and Arc-Length Sampling
**Goal**: Extract stable 2D midlines from segmentation masks and produce fixed-size, arc-length-normalized point correspondences across cameras — the 2D input that multi-view triangulation consumes
**Depends on**: Phase 5 (cross-view identity), Phase 2 (segmentation masks)
**Requirements**: RECON-01, RECON-02
**Success Criteria** (what must be TRUE):
  1. Morphological smoothing + skeletonization produces a single clean head-to-tail skeleton from U-Net masks, with spurious branches pruned via longest-path BFS, on ≥90% of masks across a test clip
  2. Arc-length resampling produces N fixed-size 2D midline points (plus half-widths) per fish per camera, with consistent head-to-tail ordering across cameras verified by reprojecting Stage 0's 3D centroid
  3. Coordinate transforms correctly map crop-space midline points back to full-frame pixel coordinates using detection bounding boxes
  4. The module handles edge cases gracefully: masks too small to skeletonize, degenerate skeletons (no clear longest path), and single-camera fish (passes through without crashing, flagged for downstream)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 (shelved) → 5 → 6 → 7+ (TBD)

Note: Phase 5 depends on Phase 2 (segmentation) and Phase 1 (refractive ray model). Phase 6 depends on Phase 5. Phases 7–8 will be added via /gsd:add-phase.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Calibration and Refractive Geometry | 2/2 | Complete | 2026-02-19 |
| 2. Segmentation Pipeline | 0/4 | Planning complete | - |
| 02.1 Segmentation Troubleshooting | 3/3 | Complete | 2026-02-20 |
| 02.1.1 Object-detection alternative to MOG2 | 3/3 | Complete | 2026-02-20 |
| 3. Fish Mesh Model and 3D Initialization | 0/2 | Planning complete | - |
| 4. Per-Fish Reconstruction | 0/3 | Shelved    | 2026-02-21 |
| 5. Cross-View Identity and 3D Tracking | 2/3 | In Progress|  |
| 6. 2D Medial Axis and Arc-Length Sampling | 0/TBD | Not started | - |
