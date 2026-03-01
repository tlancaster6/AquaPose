# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- ✅ **v2.0 Alpha** — Phases 13-21 (shipped 2026-02-27)
- ✅ **v2.1 Identity** — Phases 22-28 (shipped 2026-02-28)
- 🚧 **v2.2 Backends** — Phases 29-33 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-9) — SHIPPED 2026-02-25</summary>

- [x] Phase 1: Calibration and Refractive Geometry (2/2 plans) — complete
- [x] Phase 2: Segmentation Pipeline (4/4 plans) — complete
- [x] Phase 02.1: Segmentation Troubleshooting (3/3 plans) — complete (INSERTED)
- [x] Phase 02.1.1: Object-detection alternative to MOG2 (3/3 plans) — complete (INSERTED)
- [x] Phase 3: Fish Mesh Model and 3D Initialization (2/2 plans) — complete
- [x] Phase 4: Per-Fish Reconstruction (3/3 plans) — shelved (ABS too slow)
- [x] Phase 04.1: Isolate phase4-specific code (1/1 plan) — complete (INSERTED)
- [x] Phase 5: Cross-View Identity and 3D Tracking (3/3 plans) — complete
- [x] Phase 6: 2D Medial Axis and Arc-Length Sampling (1/1 plan) — complete
- [x] Phase 7: Multi-View Triangulation (1/1 plan) — complete
- [x] Phase 8: End-to-End Integration Testing (3 plans, 2 summaries) — complete
- [x] Phase 9: Curve-Based Optimization (2/2 plans) — complete

**12 phases, 28 plans total**
Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v2.0 Alpha (Phases 13-21) — SHIPPED 2026-02-27</summary>

- [x] Phase 13: Engine Core (4/4 plans) — completed 2026-02-25
- [x] Phase 14: Golden Data and Verification Framework (2/2 plans) — completed 2026-02-25
- [x] Phase 14.1: Fix Critical Mismatch (2/2 plans) — completed 2026-02-25 (INSERTED)
- [x] Phase 15: Stage Migrations (5/5 plans) — completed 2026-02-26
- [x] Phase 16: Numerical Verification and Legacy Cleanup (2/2 plans) — completed 2026-02-26
- [x] Phase 17: Observers (5/5 plans) — completed 2026-02-26
- [x] Phase 18: CLI and Execution Modes (3/3 plans) — completed 2026-02-26
- [x] Phase 19: Alpha Refactor Audit (4/4 plans) — completed 2026-02-26
- [x] Phase 20: Post-Refactor Loose Ends (5/5 plans) — completed 2026-02-27
- [x] Phase 21: Retrospective, Prospective (2/2 plans) — completed 2026-02-27

**10 phases, 34 plans total**
Full details: `.planning/milestones/v2.0-ROADMAP.md`

</details>

<details>
<summary>✅ v2.1 Identity (Phases 22-28) — SHIPPED 2026-02-28</summary>

- [x] Phase 22: Pipeline Scaffolding (2/2 plans) — completed 2026-02-27
- [x] Phase 23: Refractive Lookup Tables (2/2 plans) — completed 2026-02-27
- [x] Phase 24: Per-Camera 2D Tracking (1/1 plan) — completed 2026-02-27
- [x] Phase 25: Association Scoring and Clustering (2/2 plans) — completed 2026-02-27
- [x] Phase 26: Association Refinement and Pipeline Wiring (3/3 plans) — completed 2026-02-27
- [x] Phase 27: Diagnostic Visualization (1/1 plan) — completed 2026-02-27
- [x] Phase 28: E2E Testing (1/1 plan) — completed 2026-02-27

**7 phases, 12 plans total**
Full details: `.planning/milestones/v2.1-ROADMAP.md`

</details>

### 🚧 v2.2 Backends (In Progress)

**Milestone Goal:** Add swappable detection and midline backends (YOLO-OBB, keypoint regression), build training infrastructure, and clean up the config system and guidebook.

- [x] **Phase 29: Guidebook Audit** — Audit and update GUIDEBOOK.md for v2.1 accuracy and v2.2 planned features (complete 2026-02-28)
- [x] **Phase 30: Config and Contracts** — Unify pipeline config, propagate device, extend Detection and Midline2D dataclasses (3/3 plans complete) (completed 2026-02-28)
- [x] **Phase 31: Training Infrastructure** — Build `aquapose train` CLI group and `src/aquapose/training/` package (early: start training while building integration) (completed 2026-02-28)
- [x] **Phase 32: YOLO-OBB Detection Backend** — Add OBB detector, affine crop utilities, and OBB overlay visualization (completed 2026-02-28)
- [ ] **Phase 33: Keypoint Midline Backend** — Implement DirectPoseBackend pipeline integration and confidence-weighted reconstruction
- [ ] **Phase 34: Stabilization** — Interactive QA, bug fixes, parameter tuning, real-data validation, cleanup

## Phase Details

### Phase 29: Guidebook Audit — COMPLETE (2026-02-28)
**Goal**: GUIDEBOOK.md accurately reflects the v2.1 codebase and documents v2.2 planned features, giving future Claude sessions a reliable architectural reference
**Depends on**: Nothing (documentation work, no code dependencies)
**Requirements**: DOCS-01, DOCS-02
**Plans**: 2/2 complete
Plans:
- [x] 29-01-PLAN.md — Audit and update GUIDEBOOK.md for v2.1 accuracy
- [x] 29-02-PLAN.md — Add v2.2 planned feature inline tags
**Summaries**: 29-01-SUMMARY.md, 29-02-SUMMARY.md

### Phase 30: Config and Contracts — COMPLETE (2026-02-28)
**Goal**: Pipeline config is unified and backward-compatible, device propagates to all stages from one top-level parameter, and Detection/Midline2D dataclasses carry the optional fields that v2.2 backends require
**Depends on**: Phase 29
**Requirements**: CFG-01, CFG-02, CFG-03, CFG-04, CFG-05, CFG-06, CFG-07, CFG-08, CFG-09, CFG-10, CFG-11, CFG-12
**Success Criteria** (what must be TRUE):
  1. Setting `device: cpu` at the top level of a pipeline config runs the full pipeline on CPU with no per-stage device overrides required; E2E tests cover both CPU and CUDA modes
  2. Changing `n_sample_points` in pipeline config changes the number of midline points produced by all stages — no hardcoded `15` literals remain in any module
  3. Existing v2.1 YAML config files load without error after config schema changes (backward compatibility via universal `_filter_fields()`)
  4. `aquapose init-config <name>` creates a ready-to-use project directory with correctly ordered YAML fields and optional `--synthetic` flag for synthetic config section
  5. `Detection` and `Midline2D` dataclasses carry their new optional fields (`angle`, `obb_points`, `point_confidence`) and all existing code paths treat absent fields as `None` without modification
**Plans**: 3/3 complete
Plans:
- [x] 30-01-PLAN.md — Dataclass extensions and strict config validation
- [x] 30-02-PLAN.md — Config field promotion and propagation
- [x] 30-03-PLAN.md — init-config rewrite and path resolution
**Summaries**: 30-01-SUMMARY.md, 30-02-SUMMARY.md, 30-03-SUMMARY.md

### Phase 31: Training Infrastructure
**Goal**: All model training is accessible through a single `aquapose train` CLI group with consistent conventions, replacing disconnected scripts — built early so model training can begin while pipeline integration proceeds
**Depends on**: Phase 30 (config conventions, device parameter)
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04
**Success Criteria** (what must be TRUE):
  1. `aquapose train --help` lists `unet`, `yolo-obb`, and `pose` subcommands; each subcommand accepts `--data-dir`, `--output-dir`, `--epochs`, `--device`, and `--val-split` with consistent semantics
  2. `aquapose train pose --backbone-weights <path>` loads U-Net encoder weights and freezes backbone for transfer learning; `--unfreeze` flag enables end-to-end fine-tuning
  3. Running `aquapose train unet` produces the same training behavior as the existing `segmentation/training.py` script, which is then superseded
  4. `src/aquapose/training/` module exists as a proper package with no imports from `engine/` (import boundary enforced by pre-commit hook)
**Plans**: 2 plans
Plans:
- [ ] 31-01-PLAN.md — Training package scaffold, shared utilities, datasets, U-Net subcommand
- [ ] 31-02-PLAN.md — YOLO-OBB and pose subcommands, migration cleanup

### Phase 32: YOLO-OBB Detection Backend
**Goal**: Pipeline supports YOLO-OBB as a selectable detection model that produces rotation-aligned affine crops and OBB polygon overlays in diagnostic mode
**Depends on**: Phase 30 (Detection.angle field), Phase 31 (training CLI for OBB model training)
**Requirements**: DET-01, DET-02, DET-03, VIZ-01, VIZ-02
**Success Criteria** (what must be TRUE):
  1. Running the pipeline with `detector_kind: yolo_obb` in config produces detections with non-None `angle` and `obb_points` fields; running with `detector_kind: yolo` produces unchanged behavior
  2. Affine crops extracted from OBB detections show fish bodies axis-aligned within the crop (orientation smoke test passes for known-angle detections)
  3. A point in crop coordinates can be back-projected to full-frame pixel coordinates via the inverse transform, with round-trip error under 1 pixel
  4. Diagnostic mode renders OBB polygon overlays on detection frames and bounding box overlays (both axis-aligned and OBB) on tracklet trail frames
**Plans**: TBD

### Phase 33: Keypoint Midline Backend
**Goal**: Pipeline supports a keypoint regression backend that produces N ordered midline points with per-point confidence, and both reconstruction backends weight observations by that confidence
**Depends on**: Phase 32 (affine crop utilities), Phase 30 (Midline2D.point_confidence field), Phase 31 (keypoint training)
**Requirements**: MID-01, MID-02, MID-03, MID-04, RECON-01, RECON-02
**Success Criteria** (what must be TRUE):
  1. Setting `midline_backend: direct_pose` in config runs the full pipeline end-to-end using the keypoint regression model; setting `midline_backend: segment_then_extract` restores the original behavior
  2. The keypoint backend always produces exactly `n_sample_points` midline points per fish per camera, with unobserved regions marked as NaN coordinates and zero confidence rather than a shorter array
  3. Both midline backends produce `Midline2D` instances with the same shape and field structure — the reconstruction stages require no backend-specific branching
  4. Triangulation backend uses per-point confidence as weights when confidence is present; reconstruction produces identical output to the previous version when confidence is `None`
  5. Curve optimizer backend uses per-point confidence as weights when confidence is present; reconstruction produces identical output to the previous version when confidence is `None`
**Plans**: TBD

### Phase 33.1: Keypoint Training Data Augmentation (INSERTED)

**Goal:** Add keypoint-aware data augmentation transforms to the pose regression training pipeline, masked MSE loss for partial visibility, and ConcatDataset training for 2x effective epoch size — critical for generalization from the 78-sample dataset
**Requirements**: AUG-01, AUG-02, AUG-03, AUG-04
**Depends on:** Phase 33
**Plans:** 1 plan

Plans:
- [ ] 33.1-01-PLAN.md — KeypointDataset augmentation, masked loss, updated train_pose, comprehensive tests

### Phase 34: Stabilization
**Goal**: All v2.2 features work correctly on real data with tuned parameters — milestone goals are complete in practice, not just in theory
**Depends on**: Phases 29-33
**Requirements**: (cross-cutting — validates all requirements on real data)
**Success Criteria** (what must be TRUE):
  1. A reusable QA skill/agent is drafted and iteratively refined through the stabilization process
  2. Full pipeline runs end-to-end on real video data with both detection backends (YOLO, YOLO-OBB) and both midline backends (segment-then-extract, direct-pose)
  3. All bugs discovered during feature phases are triaged and critical bugs are fixed
  4. Basic parameter tuning is complete (confidence thresholds, NMS parameters, keypoint model quality validation)
  5. Pending todos accumulated during v2.2 are triaged (resolved, deferred, or converted to future requirements)
**Notes**: This is an interactive process between agent and user. The first step is drafting a custom QA skill/agent that can be reused in future milestones.
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-9 | v1.0 | 28/28 | Complete | 2026-02-25 |
| 13-21 | v2.0 | 34/34 | Complete | 2026-02-27 |
| 22-28 | v2.1 | 12/12 | Complete | 2026-02-28 |
| 29. Guidebook Audit | v2.2 | 2/2 | Complete | 2026-02-28 |
| 30. Config and Contracts | v2.2 | 3/3 | Complete | 2026-02-28 |
| 31. Training Infrastructure | 2/2 | Complete    | 2026-02-28 | - |
| 32. YOLO-OBB Detection Backend | 2/2 | Complete    | 2026-02-28 | - |
| 33. Keypoint Midline Backend | 1/2 | In Progress|  | - |
| 34. Stabilization | v2.2 | 0/TBD | Not started | - |
