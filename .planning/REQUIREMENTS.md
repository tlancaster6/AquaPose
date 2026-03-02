# Requirements: AquaPose

**Defined:** 2026-03-01
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.0 Requirements

Requirements for Ultralytics Unification milestone. Each maps to roadmap phases.

### Cleanup

- [x] **CLEAN-01**: All custom U-Net model code removed (segmentation/model.py, training/_UNet, _PoseModel, BinaryMaskDataset)
- [x] **CLEAN-02**: SAM2 pseudo-label pipeline removed (no longer needed — Ultralytics models trained on NDJSON datasets)
- [x] **CLEAN-03**: Custom model code removed from midline backends (segment_then_extract, direct_pose stubbed as no-ops) — awaiting YOLO-seg and YOLO-pose model wiring in Phase 37
- [x] **CLEAN-04**: MOG2 detection backend removed — YOLO and YOLO-OBB are the only detection backends
- [x] **CLEAN-05**: Old training CLI commands removed (train_unet, train_pose) — replaced by Ultralytics training wrappers

### Training Data

- [x] **DATA-01**: Segmentation training data converter takes COCO segmentation JSON (polygon annotations) and produces NDJSON-format YOLO-seg dataset (matching existing OBB/pose NDJSON pattern)

### Training

- [x] **TRAIN-01**: YOLO-seg training wrapper callable from CLI, following existing yolo_obb.py pattern
- [x] **TRAIN-02**: YOLO-pose training wrapper callable from CLI, following existing yolo_obb.py pattern

### Pipeline Integration

- [x] **PIPE-01**: YOLOSegBackend produces binary masks per detection for midline extraction via skeletonization
- [x] **PIPE-02**: YOLOPoseBackend produces keypoint coordinates with per-point confidence for direct midline construction
- [x] **PIPE-03**: Config system supports backend selection (yolo_seg, yolo_pose) via midline.backend field

### Stabilization

- [x] **STAB-01**: Training data script produces standard YOLO txt labels + dataset.yaml (not NDJSON); training wrappers consume txt+yaml
- [x] **STAB-02**: `weights_path` and `keypoint_weights_path` consolidated into single `weights_path` field
- [x] **STAB-03**: `init-config` generates correct defaults (YOLO-OBB detection, explicit backend selection, valid weights path)
- [ ] **STAB-04**: All stale docstrings referencing U-Net, no-op stubs, or Phase 37 pending status are updated

## Previous Milestone Requirements (v2.2)

All v2.2 requirements completed. See MILESTONES.md for details.

## Future Requirements

### CLI Formalization

- **CLI-01**: Training data preparation integrated as formal CLI subcommand (currently scripts/)
- **CLI-02**: Unified training CLI for all model types (detect, seg, pose)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Backwards compatibility with U-Net weights/configs | Explicitly dropped — fresh start with Ultralytics |
| SAM2 pseudo-label generation | Replaced by direct annotation conversion from COCO JSON |
| MOG2 detection fallback | YOLO is the only detection backend going forward |
| Custom model architectures | Ultralytics models only — no custom encoder/decoder code |
| Label Studio integration | Direct COCO JSON → NDJSON conversion, no annotation UI |
| Training data CLI formalization | Deferred — scripts/ workflow sufficient for now |
| Heatmap-based keypoint regression | Direct regression via YOLO-pose is superior |
| PyTorch Lightning for training | Ultralytics handles training loops internally |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLEAN-01 | Phase 35 | Complete |
| CLEAN-02 | Phase 35 | Complete |
| CLEAN-03 | Phase 35 | Complete |
| CLEAN-04 | Phase 35 | Complete |
| CLEAN-05 | Phase 35 | Complete |
| DATA-01 | Phase 36 | Complete |
| TRAIN-01 | Phase 36 | Complete |
| TRAIN-02 | Phase 36 | Complete |
| PIPE-01 | Phase 37 | Complete |
| PIPE-02 | Phase 37 | Complete |
| PIPE-03 | Phase 37 | Complete |
| STAB-01 | Phase 38 | Complete |
| STAB-02 | Phase 38 | Complete |
| STAB-03 | Phase 38 | Complete |
| STAB-04 | Phase 38 | Pending |

**Coverage:**
- v3.0 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-03-01*
*Last updated: 2026-03-02 after Phase 38 addition*
