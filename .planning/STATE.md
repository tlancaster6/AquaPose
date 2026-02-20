# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Accurate single-fish 3D reconstruction from multi-view silhouettes via differentiable refractive rendering
**Current focus:** Phase 02-segmentation-pipeline Plan 02 complete — box-only SAM2, quality filtering, variable-size CropDataset, stratified split, negative examples; 203 tests passing

## Current Position

Phase: 02-segmentation-pipeline
Plan: 02 of N complete
Status: Pseudo-labeling pipeline fully updated; CropDataset native-resolution; build_training_data.py generates train.json/val.json with negatives
Last activity: 2026-02-20 — 02-02 complete (commits 003e8de, 2766e7d)

Progress: [██████░░░░] 60% (12 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 10
- Average duration: 15 min
- Total execution time: ~2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-calibration-and-refractive-geometry | 2 | 50 min | 25 min |
| 02-segmentation-pipeline | 3 | 35 min | 12 min |
| 03-fish-mesh-model-and-3d-initialization | 2 | 19 min | 10 min |
| 02.1-segmentation-troubleshooting | 1 (of 3) | 8 min | 8 min |
| 02.1.1-object-detection-alternative-to-mog2 | 3 (of 3) | ~3 sessions | - |
| 02-segmentation-pipeline (new) | 2 (of N) | 31 min | 15 min |

**Recent Trend:**
- Last 5 plans: 8 min, 8 min, 12 min, 15 min, 10 min
- Trend: stable/fast

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: AquaCal is numpy-based — refractive projection must be reimplemented in PyTorch (AquaCal used only for loading calibration JSON, not for forward projection)
- [Init]: Phase 3 (Fish Mesh) depends only on Phase 1, not Phase 2 — can develop in parallel with segmentation if calendar time matters
- [Init]: Temporal smoothness loss (RECON-02) built in Phase 4 but only activates in Phase 5 when tracking provides associations — build the hook, wire it in Phase 5
- [Init]: All APIs are batch-first (list of fish states) from day one — even Phase 4 single-fish code uses single-element lists
- [01-01]: Cross-validation compares against AquaCal NumPy (not AquaMVS PyTorch) — AquaMVS not importable in hatch env due to missing open3d/lightglue; AquaCal NumPy is the actual ground truth
- [01-01]: K_inv float32 inversion tolerance set to atol=1e-4 — float32 with fx=1400 produces ~6e-5 error, expected float32 precision not a bug
- [01-02]: Z/XY anisotropy is 132x mean (30x-577x range) at 0.5px noise — Phase 4 optimizer should weight Z loss approximately 100x smaller than XY
- [01-02]: build_synthetic_rig uses water_z = height_above_water (0.75m) since AquaCal places cameras at world Z=0 with Z increasing downward into water
- [02-01]: Shadow exclusion via threshold at 254 (MOG2 outputs 127 for shadows, 255 for foreground)
- [02-01]: Detection.mask is full-frame sized (not cropped to bbox) to feed directly into SAM as mask prompt
- [02-02]: SAM2 predictor lazily loaded on first predict() call to avoid GPU allocation on import
- [02-02]: Label Studio uses its own RLE variant (mask2rle) not pycocotools RLE
- [02-03]: torchvision maskrcnn_resnet50_fpn_v2 instead of Detectron2 (unmaintained, Windows-incompatible)
- [02-03]: Custom collate_fn with tuple(zip(*batch)) for Mask R-CNN's list-of-dicts format
- [03-01]: torch.sinc used for sin(kappa*t*s)/kappa stability at kappa=0 — sinc(x/pi)=sin(x)/x, smooth everywhere
- [03-01]: miropsota pytorch3d-0.7.9+pt2.9.1cu128 works with torch 2.10+cu130 on Windows (CUDA mesh ops not needed in Phase 3)
- [03-01]: Watertight winding: tube (v0,v2,v1)+(v1,v2,v3), head cap (apex,j_next,j), tail cap (apex,j,j_next) — all edges shared exactly 2x
- [03-01]: Spine centered at t=0.5 midpoint; builder translates by state.p — keeps spine generation position-independent
- [03-02]: Canonical sign (endpoint_a = max projection) ensures deterministic PCA keypoint output across calls
- [03-02]: >=3 cameras enforced at triangulate_keypoint level (triangulate_rays technically needs only 2)
- [03-02]: kappa=0 at cold-start initialization; head/tail disambiguation deferred to Phase 4 optimizer
- [03-02]: Synthetic test cameras: t = (-cam_x, -cam_y, 0.0) with R=I, cameras at Z=0, water at Z=1.0 in world
- [02.1-01]: No automated PASS/FAIL in test_mog2.py — detection quality assessed visually from annotated stills
- [02.1-01]: 2-stage shadow fix in detector.py: Stage 1 merges shadow+foreground (fg_mask>=127); Stage 2 watershed-splits merged blobs using foreground-only (255) cores as seeds
- [02.1-02]: Default --pseudo-labels-dir is output/pseudo_labels (matches run_pseudo_labels.py YOLO output, not verify_pseudo_labels path)
- [02.1-02]: YOLO traceability note printed in test_sam2.py summary output for audit trail
- [02.1.1-01]: Frame diversity via MOG2 count bins (0, 1, 2, 3+); at least 1 frame per non-empty bin, surplus distributed proportional to bin size; ring cameras get 10 frames, center camera gets 30 (150 total)
- [02.1.1-02]: YOLODetector.detect() returns full-frame rectangular mask (padded bbox region = 255) for SAM2 compatibility — same format as MOG2Detector
- [02.1.1-02]: make_detector factory uses **kwargs forwarding; model_path is a required kwarg for 'yolo'
- [02.1.1-02]: eval script creates fresh MOG2Detector() per val image; optional video warmup via --video-dir for fair temporal comparison
- [02.1.1-02]: Trained YOLOv8n early-stopped at epoch 10: recall=0.780, mAP50=0.799, precision=0.760
- [02.1.1-02]: Formal YOLO vs MOG2 comparison deferred — MOG2 requires video warmup context unavailable for isolated val frames; YOLO val metrics accepted as sufficient validation evidence (user-approved)
- [02.1.1-03]: run_pseudo_labels.py validates YOLO weights path at startup before loading any models — fail-fast pattern for misconfigured CLI usage
- [02.1.1-03]: Integration tests mock SAMPseudoLabeler.predict() via monkeypatch rather than patching SAM2 internals — keeps tests GPU-free and fast while still exercising the full pipeline chain
- [02-01 new]: to_coco_dataset lives in pseudo_labeler.py — COCO conversion co-located with pseudo-label generation, not in Label Studio module
- [02-01 new]: Label Studio fully removed from segmentation module — label-studio-converter dependency deleted, no remaining references in src/
- [02-02]: filter_mask() lives in pseudo_labeler.py — quality filtering is a pseudo-label concern, not a build-script concern
- [02-02]: CropDataset drops crop_size parameter entirely — Mask R-CNN FPN+RoI handles variable inputs natively
- [02-02]: stratified_split groups by camera_id COCO field — each camera gets proportional val representation
- [02-02]: build_training_data.py writes train.json + val.json + coco_annotations.json — three files, one source of truth

### Roadmap Evolution

- Phase 02.1 inserted after Phase 02: Segmentation Troubleshooting (URGENT)
- Phase 02.1.1 inserted after Phase 02.1: Object-detection alternative to MOG2 (URGENT)

### Pending Todos

- **Consolidate scripts into CLI workflow** (tooling) — scripts/ now cleaned to 5 production scripts (build_training_data.py, eval_yolo_vs_mog2.py, organize_yolo_dataset.py, sample_yolo_frames.py, train_yolo.py)

### Blockers/Concerns

- [Phase 1 - RESOLVED]: Z-uncertainty budget quantified: Z error is 132x larger than XY for top-down 13-camera rig (see docs/reports/z_uncertainty_report.md)
- [Phase 02.1 - IN PROGRESS]: MOG2 validated on 2 cameras — e3v83eb frame 006765 shows 0 detections (possible stationary-fish failure); counts up to 18 suggest over-splitting; visual review of output/test_mog2/ stills needed before proceeding to SAM2 plan
- [Phase 4]: PyTorch3D sigma/gamma hyperparameters for this rig's fish pixel sizes unknown — empirical sweep needed during Phase 4 development

## Session Continuity

Last session: 2026-02-20
Stopped at: Completed 02-segmentation-pipeline 02-02-PLAN.md
Next action: Execute 02-segmentation-pipeline Plan 03 (Mask R-CNN training on generated dataset)
