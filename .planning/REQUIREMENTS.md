# Requirements: AquaPose

**Defined:** 2026-02-28
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v2.2 Requirements

Requirements for v2.2 Backends milestone. Each maps to roadmap phases.

### Documentation

- [ ] **DOCS-01**: Guidebook reflects current v2.1 codebase state (stale references removed, architectural descriptions accurate)
- [ ] **DOCS-02**: Guidebook documents v2.2 planned features (YOLO-OBB, keypoint midline, training CLI, project structure) in relevant sections

### Config and Contracts

- [ ] **CFG-01**: Pipeline accepts a single top-level `device` parameter that propagates to all stages, eliminating per-stage device configuration
- [ ] **CFG-02**: Midline sample point count (`n_sample_points`) is configurable via pipeline config with no hardcoded constants remaining in any module
- [ ] **CFG-03**: Fish count is a single unified parameter (no duplicate `expect_fish_count`/`n_animals` fields)
- [ ] **CFG-04**: `stop_frame` is a top-level pipeline parameter, not nested within a stage-specific config section
- [ ] **CFG-05**: All stage configs use `_filter_fields()` so that existing YAML files load without error after schema changes
- [ ] **CFG-06**: `init-config` creates a project directory scaffold under `~/aquapose/projects/<name>/` with sensible default paths and required subdirectories
- [ ] **CFG-07**: Config paths resolve relative to `project_dir`, allowing portable project layouts; absolute paths override when provided
- [ ] **CFG-08**: `init-config --synthetic` includes the synthetic config section; omitted by default
- [ ] **CFG-09**: `init-config` orders YAML fields by user relevance (paths and animal count first, stage-specific parameters after) rather than alphabetically
- [ ] **CFG-10**: Detection dataclass carries optional rotation angle so that OBB orientation flows to downstream stages
- [ ] **CFG-11**: Midline2D dataclass carries optional per-point confidence so that reconstruction can weight observations; None means uniform confidence
- [ ] **CFG-12**: Pipeline runs end-to-end in both CPU and CUDA device modes, verified by E2E tests

### Detection

- [ ] **DET-01**: Pipeline supports YOLO-OBB as a configurable detection model selectable via `detector_kind: yolo_obb` in config
- [ ] **DET-02**: OBB detections produce rotation-aligned affine crops suitable for downstream segmentation and keypoint models
- [ ] **DET-03**: Affine crop utilities support back-projection from crop coordinates to full-frame pixel coordinates via inverse transform

### Midline

- [ ] **MID-01**: Pipeline supports a keypoint regression backend as a swappable alternative to segment-then-extract, selectable via config
- [ ] **MID-02**: Keypoint backend produces N ordered midline points with per-point confidence from a learned regression model (U-Net encoder + regression head)
- [ ] **MID-03**: Keypoint backend handles partial visibility by marking unobserved regions with NaN coordinates and zero confidence, always outputting exactly `n_sample_points`
- [ ] **MID-04**: Both midline backends produce the same output structure (N-point Midline2D) so reconstruction is backend-agnostic

### Reconstruction

- [ ] **RECON-01**: Triangulation backend weights per-point observations by confidence when available, falling back to uniform weights when confidence is None
- [ ] **RECON-02**: Curve optimizer backend weights observations by confidence when available, falling back to uniform weights when confidence is None

### Training

- [ ] **TRAIN-01**: `aquapose train` CLI group provides discoverable subcommands for all trainable models (U-Net segmentation, YOLO-OBB, keypoint regression)
- [ ] **TRAIN-02**: Training subcommands follow consistent conventions for `--data-dir`, `--output-dir`, `--epochs`, `--device`, `--resume` across all model types
- [ ] **TRAIN-03**: Keypoint training supports frozen-backbone transfer learning from existing U-Net segmentation weights, with optional unfreeze for end-to-end fine-tuning
- [ ] **TRAIN-04**: Existing training scripts in `scripts/` are superseded by `src/aquapose/training/` module with shared utilities

### Visualization

- [ ] **VIZ-01**: Diagnostic mode renders OBB polygon overlays on detection frames
- [ ] **VIZ-02**: Tracklet trail visualization includes bounding box overlays (both axis-aligned and OBB when available)

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Quality Enhancements

- **QUAL-01**: Body-model extrapolation for partial midlines (infer missing keypoints from learned shape prior)
- **QUAL-02**: Confidence calibration via temperature scaling on keypoint head outputs
- **QUAL-03**: Joint segmentation + keypoint multi-task training loss

### Detection Enhancements

- **DET-04**: OBB model fine-tuned on fish-specific dataset (v2.2 uses DOTA-pretrained weights)

### Training Enhancements

- **TRAIN-05**: Keypoint pseudo-label auto-generation from skeletonizer output to bootstrap training data
- **TRAIN-06**: Integral regression alternative if direct regression accuracy is insufficient

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Heatmap-based keypoint regression | 4px quantization at 128x128 crop is ~8% body-length error; direct regression is superior at this resolution |
| Separate `aquapose-train` binary entrypoint | Splits help system; `aquapose train` subgroup is discoverable under unified CLI |
| Pydantic for config | Frozen dataclasses decided in v2.0; migration touches every config path with no scientific benefit |
| Real-time 13-camera inference | Batch mode only per project constraints; 5-30 min clips |
| PyTorch Lightning for training | Simple bare loops don't justify trainer abstraction overhead |
| albumentations dependency | torchvision.transforms.v2 covers all needed augmentations including keypoint-aware transforms |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DOCS-01 | — | Pending |
| DOCS-02 | — | Pending |
| CFG-01 | — | Pending |
| CFG-02 | — | Pending |
| CFG-03 | — | Pending |
| CFG-04 | — | Pending |
| CFG-05 | — | Pending |
| CFG-06 | — | Pending |
| CFG-07 | — | Pending |
| CFG-08 | — | Pending |
| CFG-09 | — | Pending |
| CFG-10 | — | Pending |
| CFG-11 | — | Pending |
| CFG-12 | — | Pending |
| DET-01 | — | Pending |
| DET-02 | — | Pending |
| DET-03 | — | Pending |
| MID-01 | — | Pending |
| MID-02 | — | Pending |
| MID-03 | — | Pending |
| MID-04 | — | Pending |
| RECON-01 | — | Pending |
| RECON-02 | — | Pending |
| TRAIN-01 | — | Pending |
| TRAIN-02 | — | Pending |
| TRAIN-03 | — | Pending |
| TRAIN-04 | — | Pending |
| VIZ-01 | — | Pending |
| VIZ-02 | — | Pending |

**Coverage:**
- v2.2 requirements: 29 total
- Mapped to phases: 0
- Unmapped: 29 (pending roadmap creation)

---
*Requirements defined: 2026-02-28*
*Last updated: 2026-02-28 after initial definition*
