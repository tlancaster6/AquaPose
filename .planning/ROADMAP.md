# Roadmap: AquaPose

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- ✅ **v2.0 Alpha** — Phases 13-21 (shipped 2026-02-27)
- ✅ **v2.1 Identity** — Phases 22-28 (shipped 2026-02-28)
- ✅ **v2.2 Backends** — Phases 29-33.1 (shipped 2026-03-01)
- 🚧 **v3.0 Ultralytics Unification** — Phases 35-37 (in progress)

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

<details>
<summary>✅ v2.2 Backends (Phases 29-33.1) — SHIPPED 2026-03-01</summary>

- [x] **Phase 29: Guidebook Audit** — complete 2026-02-28
- [x] **Phase 30: Config and Contracts** — complete 2026-02-28
- [x] **Phase 31: Training Infrastructure** — complete 2026-02-28
- [x] **Phase 32: YOLO-OBB Detection Backend** — complete 2026-02-28
- [x] **Phase 33: Keypoint Midline Backend** — complete 2026-03-01
- [x] **Phase 33.1: Keypoint Training Data Augmentation** — complete 2026-03-01

**6 phases (including inserted 33.1), Phase 34 (Stabilization) deferred**
Full details: see Phase Details section below for v2.2

</details>

### 🚧 v3.0 Ultralytics Unification (In Progress)

**Milestone Goal:** Replace custom U-Net segmentation and keypoint regression models with Ultralytics-native YOLO26n-seg and YOLO26n-pose, unifying detection, segmentation, and midline extraction on one framework. Starts by stripping the failed custom code, then builds training wrappers for the new models, then integrates them as backends within the existing Stage architecture.

**⚠ Cross-cutting concern — Coordinate spaces:** Full-image ↔ crop-space conversions are a pervasive source of error, especially with OBB affine warps. Mismatches between training-time and inference-time crop preparation cause silent accuracy failures. Every phase must explicitly verify coordinate round-trips at each boundary (training labels, inference output, back-projection to full frame). Existing crop utilities should be reused with extreme care.

- [x] **Phase 35: Codebase Cleanup** — Remove custom U-Net, SAM2 pipeline, old midline backends, MOG2 backend, and legacy training CLI commands (completed 2026-03-01)
- [ ] **Phase 36: Training Wrappers** — Add NDJSON seg data converter and YOLO-seg/pose training wrappers following existing yolo_obb.py pattern
- [ ] **Phase 37: Pipeline Integration** — Implement YOLOSegBackend and YOLOPoseBackend as selectable midline backends with instance matching and config support

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

### Phase 31: Training Infrastructure — COMPLETE (2026-02-28)
**Goal**: All model training is accessible through a single `aquapose train` CLI group with consistent conventions, replacing disconnected scripts — built early so model training can begin while pipeline integration proceeds
**Depends on**: Phase 30 (config conventions, device parameter)
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04
**Success Criteria** (what must be TRUE):
  1. `aquapose train --help` lists `unet`, `yolo-obb`, and `pose` subcommands; each subcommand accepts `--data-dir`, `--output-dir`, `--epochs`, `--device`, and `--val-split` with consistent semantics
  2. `aquapose train pose --backbone-weights <path>` loads U-Net encoder weights and freezes backbone for transfer learning; `--unfreeze` flag enables end-to-end fine-tuning
  3. Running `aquapose train unet` produces the same training behavior as the existing `segmentation/training.py` script, which is then superseded
  4. `src/aquapose/training/` module exists as a proper package with no imports from `engine/` (import boundary enforced by pre-commit hook)
**Plans**: 2/2 complete
Plans:
- [x] 31-01-PLAN.md — Training package scaffold, shared utilities, datasets, U-Net subcommand
- [x] 31-02-PLAN.md — YOLO-OBB and pose subcommands, migration cleanup

### Phase 32: YOLO-OBB Detection Backend — COMPLETE (2026-02-28)
**Goal**: Pipeline supports YOLO-OBB as a selectable detection model that produces rotation-aligned affine crops and OBB polygon overlays in diagnostic mode
**Depends on**: Phase 30 (Detection.angle field), Phase 31 (training CLI for OBB model training)
**Requirements**: DET-01, DET-02, DET-03, VIZ-01, VIZ-02
**Success Criteria** (what must be TRUE):
  1. Running the pipeline with `detector_kind: yolo_obb` in config produces detections with non-None `angle` and `obb_points` fields; running with `detector_kind: yolo` produces unchanged behavior
  2. Affine crops extracted from OBB detections show fish bodies axis-aligned within the crop (orientation smoke test passes for known-angle detections)
  3. A point in crop coordinates can be back-projected to full-frame pixel coordinates via the inverse transform, with round-trip error under 1 pixel
  4. Diagnostic mode renders OBB polygon overlays on detection frames and bounding box overlays (both axis-aligned and OBB) on tracklet trail frames
**Plans**: 2/2 complete

### Phase 33: Keypoint Midline Backend — COMPLETE (2026-03-01)
**Goal**: Pipeline supports a keypoint regression backend that produces N ordered midline points with per-point confidence, and both reconstruction backends weight observations by that confidence
**Depends on**: Phase 32 (affine crop utilities), Phase 30 (Midline2D.point_confidence field), Phase 31 (keypoint training)
**Requirements**: MID-01, MID-02, MID-03, MID-04, RECON-01, RECON-02
**Success Criteria** (what must be TRUE):
  1. Setting `midline_backend: direct_pose` in config runs the full pipeline end-to-end using the keypoint regression model; setting `midline_backend: segment_then_extract` restores the original behavior
  2. The keypoint backend always produces exactly `n_sample_points` midline points per fish per camera, with unobserved regions marked as NaN coordinates and zero confidence rather than a shorter array
  3. Both midline backends produce `Midline2D` instances with the same shape and field structure — the reconstruction stages require no backend-specific branching
  4. Triangulation backend uses per-point confidence as weights when confidence is present; reconstruction produces identical output to the previous version when confidence is `None`
  5. Curve optimizer backend uses per-point confidence as weights when confidence is present; reconstruction produces identical output to the previous version when confidence is `None`
**Plans**: 2/2 complete

### Phase 33.1: Keypoint Training Data Augmentation — COMPLETE (2026-03-01)
**Goal**: Add keypoint-aware data augmentation transforms to the pose regression training pipeline, masked MSE loss for partial visibility, and ConcatDataset training for 2x effective epoch size
**Depends on**: Phase 33
**Requirements**: AUG-01, AUG-02, AUG-03, AUG-04
**Plans**: 1/1 complete
Plans:
- [x] 33.1-01-PLAN.md — KeypointDataset augmentation, masked loss, updated train_pose, comprehensive tests

---

### Phase 35: Codebase Cleanup
**Goal**: The codebase contains no custom U-Net, SAM2 pseudo-label, old midline backend, MOG2 detection, or legacy training CLI code — only Ultralytics-based models and the new training wrappers remain, leaving a clean foundation for v3.0 backends
**Depends on**: Nothing (cleanup precedes building)
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04, CLEAN-05
**Success Criteria** (what must be TRUE):
  1. `segmentation/model.py`, `_UNet`, `_PoseModel`, and `BinaryMaskDataset` are deleted; no import of these symbols exists anywhere in the codebase
  2. SAM2 pseudo-label generation code is removed; the only path from raw video to training data is COCO JSON → NDJSON conversion
  3. Custom model code (UNetSegmentor, _PoseModel) removed from `segment_then_extract` and `direct_pose` backends; both backends stubbed as no-ops pending Phase 37 YOLO model wiring
  4. MOG2 detection backend is removed; the only registered detection backends are `yolo` and `yolo_obb`
  5. `train_unet` and `train_pose` CLI commands are removed from the `aquapose train` group; `aquapose train --help` no longer lists them
**Plans**: 2 plans
Plans:
- [ ] 35-01-PLAN.md — Remove custom models, SAM2, MOG2, and old training CLI
- [ ] 35-02-PLAN.md — Stub midline backends as no-ops, correct planning docs

### Phase 36: Training Wrappers
**Goal**: A COCO-to-NDJSON segmentation data converter and training wrappers for YOLO26n-seg and YOLO26n-pose are available from the CLI, following the same pattern as the existing `yolo_obb.py` training wrapper
**Depends on**: Phase 35 (clean codebase, no legacy training commands to conflict)
**Requirements**: DATA-01, TRAIN-01, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. Running the seg data converter with a COCO segmentation JSON produces a directory of NDJSON files matching the schema that `scripts/build_yolo_training_data.py` produces for OBB and pose — `hatch run python scripts/build_yolo_training_data.py --mode seg` or equivalent
  2. `aquapose train seg --data-dir <path> --output-dir <path> --epochs <n>` launches a YOLO26n-seg training run and saves weights to the output directory
  3. `aquapose train pose --data-dir <path> --output-dir <path> --epochs <n>` launches a YOLO26n-pose training run and saves weights to the output directory
  4. Both training wrappers accept the same flags (`--epochs`, `--device`, `--imgsz`, `--batch`) with identical semantics to the existing `yolo-obb` subcommand
**Plans**: TBD

### Phase 37: Pipeline Integration
**Goal**: The pipeline supports `yolo_seg` and `yolo_pose` as selectable midline backends; running either end-to-end produces `Midline2D` objects compatible with the reconstruction stages, with fish identities correctly linked from tracked detections to model outputs
**Depends on**: Phase 36 (trained models exist), Phase 35 (custom model code removed; existing segment_then_extract and direct_pose backends are no-op stubs awaiting YOLO model wiring)
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04
**Success Criteria** (what must be TRUE):
  1. Setting `midline.backend: yolo_seg` in pipeline config runs the full pipeline end-to-end; the MidlineStage produces binary masks per detection that feed skeletonization the same way U-Net masks did
  2. Setting `midline.backend: yolo_pose` in pipeline config runs the full pipeline end-to-end; the MidlineStage produces `Midline2D` objects with 6-keypoint coordinates resampled to `n_sample_points` and per-point confidence scores
  3. Both backends produce `Midline2D` instances with identical shape and field structure — the reconstruction stages require no backend-specific branching
  4. IoU-based instance matching correctly links each YOLO model output (mask or keypoints) to its tracked fish identity; unmatched detections are dropped rather than causing index errors
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-9 | v1.0 | 28/28 | Complete | 2026-02-25 |
| 13-21 | v2.0 | 34/34 | Complete | 2026-02-27 |
| 22-28 | v2.1 | 12/12 | Complete | 2026-02-28 |
| 29. Guidebook Audit | v2.2 | 2/2 | Complete | 2026-02-28 |
| 30. Config and Contracts | v2.2 | 3/3 | Complete | 2026-02-28 |
| 31. Training Infrastructure | v2.2 | 2/2 | Complete | 2026-02-28 |
| 32. YOLO-OBB Detection Backend | v2.2 | 2/2 | Complete | 2026-02-28 |
| 33. Keypoint Midline Backend | v2.2 | 2/2 | Complete | 2026-03-01 |
| 33.1. Keypoint Training Data Augmentation | v2.2 | 1/1 | Complete | 2026-03-01 |
| 35. Codebase Cleanup | 2/2 | Complete   | 2026-03-01 | - |
| 36. Training Wrappers | v3.0 | 0/TBD | Not started | - |
| 37. Pipeline Integration | v3.0 | 0/TBD | Not started | - |
