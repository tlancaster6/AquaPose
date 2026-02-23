# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 9 Plan 01 complete — CurveOptimizer implemented with coarse-to-fine B-spline optimization

## Current Position

Phase: 09-curve-based-optimization-as-a-replacement-for-triangulation
Plan: 01 complete
Status: curve_optimizer.py implemented, 21 unit tests passing. Ready for Phase 9 Plan 02 (integration/validation against real data).
Last activity: 2026-02-23 - Completed quick-4: sinusoidal spine shapes and diverse fish configs for synthetic testing
Stopped at: Completed 09-01-PLAN.md

Progress: [█████████░] 93% (phases 1-3 complete, phase 4 shelved, phases 5-9 plan 01 complete)

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
| 02-segmentation-pipeline (new) | 3 (of N) | 36 min | 12 min |

**Recent Trend:**
- Last 5 plans: 8 min, 8 min, 12 min, 15 min, 5 min
- Trend: stable/fast

*Updated after each plan completion*
| Phase 04-per-fish-reconstruction P02 | 6 | 2 tasks | 3 files |
| Phase 04-per-fish-reconstruction P03 | 8 | 2 tasks | 4 files |
| Phase 04.1-isolate-phase-4-specific-code-post-archive P01 | 2 | 2 tasks | 15 files |
| Phase 05-cross-view-identity-and-3d-tracking P01 | 5 | 2 tasks | 4 files |
| Phase 05 P02 | 4 | 2 tasks | 3 files |
| Phase 05-cross-view-identity-and-3d-tracking P03 | 5 | 2 tasks | 3 files |
| Phase 07-multi-view-triangulation P01 | 9 | 2 tasks | 3 files |
| Phase 08 P01 | 10 | 2 tasks | 9 files |
| Phase 08 P02 | 9 | 2 tasks | 8 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: AquaCal is numpy-based — refractive projection must be reimplemented in PyTorch (AquaCal used only for loading calibration JSON, not for forward projection)
- [Init]: Phase 3 (Fish Mesh) depends only on Phase 1, not Phase 2 — can develop in parallel with segmentation if calendar time matters
- [Init]: All APIs are batch-first (list of fish states) from day one — even Phase 4 single-fish code uses single-element lists
- [01-01]: Cross-validation compares against AquaCal NumPy (not AquaMVS PyTorch) — AquaMVS not importable in hatch env due to missing open3d/lightglue; AquaCal NumPy is the actual ground truth
- [01-01]: K_inv float32 inversion tolerance set to atol=1e-4 — float32 with fx=1400 produces ~6e-5 error, expected float32 precision not a bug
- [01-02]: Z/XY anisotropy is 132x mean (30x-577x range) at 0.5px noise — downstream optimizer should weight Z loss approximately 100x smaller than XY
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
- [Phase 02-03]: segment() accepts crops+crop_regions, returns crop-space masks — callers reconstruct full-frame via paste_mask(result.mask, result.crop_region)
- [Phase 02-03]: SegmentationResult.mask_rle removed — raw ndarray mask is more useful downstream; callers encode RLE if needed
- [Phase 02-03]: predict() kept as backward-compat wrapper calling segment() with trivial CropRegion covering full image
- [Phase 02-03]: train() uses stratified_split by default; accepts train_json/val_json to consume build_training_data.py output directly
- [Pivot 2026-02-21]: Analysis-by-synthesis pipeline shelved — 30+ min/sec runtime impractical. Replaced by direct triangulation pipeline: medial axis → arc-length → RANSAC triangulation → spline fitting. See .planning/inbox/fish-reconstruction-pivot.md
- [Pivot 2026-02-21]: Sex classification (SEX-01..03) deferred to v2 — not part of direct triangulation pipeline
- [Pivot 2026-02-21]: Cross-view identity (Stage 0) promoted to Phase 5 — prerequisite for all reconstruction stages
- [Pivot 2026-02-21]: Old RECON-01..05 renamed to RECON-ABS-01..05 (shelved); new RECON-01..05 defined for direct triangulation
- [Phase 04.1-01]: archive/phase4-abs branch created before stripping — preserves full ABS codebase (renderer, optimizer, loss, validation)
- [Phase 04.1-01]: diagnose_real_frame.py and diagnose_synthetic.py deleted (imported from aquapose.optimization); visualize_val_predictions.py retained (segmentation-only imports)
- [Phase 04.1-01]: pytorch3d.structures retained in mesh/builder.py — shared infrastructure, not ABS-specific
- [Phase 05-01]: assigned_mask dict prevents double-assignment across both prior-guided and RANSAC passes
- [Phase 05-01]: Single-view detections use 0.5m default tank depth along refracted ray as centroid heuristic
- [Phase 05-02]: XY-only cost matrix prevents Z-noise induced ID swaps (Z uncertainty 132x larger than XY)
- [Phase 05-02]: TRACK-04 population constraint: dead track fish_id recycled to first unmatched observation in same frame
- [Phase 05]: cast(h5py.Dataset, ...) used for basedpyright narrowing of h5py subscript return types in writer.py
- [Phase 06-01]: axis_minor_length used (not deprecated minor_axis_length) for skimage regionprops minor axis
- [Phase 06-01]: skeletonize return wrapped in np.asarray(dtype=bool) for basedpyright type narrowing
- [Phase 06-01]: _orient_midline uses lazy torch import to call RefractiveProjectionModel.project for head position projection
- [Phase 07-01]: Exhaustive pairwise triangulation for <=7 cams: score by max held-out reprojection error; best pair seeds inlier re-triangulation at DEFAULT_INLIER_THRESHOLD=15px
- [Phase 07-01]: is_low_confidence=True when any body point has only 2-camera observation; fixed 7-control-point B-spline with SPLINE_KNOTS=[0,0,0,0,0.25,0.5,0.75,1,1,1,1]
- [Phase 07-01]: refine_midline_lm is no-op stub (RECON-05 deferred); half-widths converted via pinhole approx hw_px*depth_m/focal_px
- [Phase 08]: FishTracker and MidlineExtractor instantiated once in reconstruct() and passed to stages
- [Phase 08]: SPLINE_KNOTS and SPLINE_K stored as HDF5 group attributes per OUT-01 spec
- [Phase 08]: run_segmentation re-reads video from disk to avoid OOM on long videos
- [Phase 08]: FISH_COLORS defined as BGR tuples in overlay.py; plot3d.py converts BGR->RGB floats on import for matplotlib compatibility
- [Phase 08]: render_3d_animation checks FFMpegWriter.isAvailable() at runtime and falls back to PillowWriter (GIF) with UserWarning -- no hard FFMpeg dependency
- [Phase 08]: Diagnostic mode catches all visualization exceptions individually to avoid crashing the main pipeline on render failures
- [Phase 09-01]: Bend angle computed as acos(v1·v2) — 0 for straight spine, pi for U-turn. Clamped cos_bend to [-1+1e-6, 1-1e-6] before acos to prevent NaN gradients at collinear/antiparallel control points
- [Phase 09-01]: Huber delta fixed at 17.5px (midpoint of 15-20px range) for per-camera chamfer aggregation
- [Phase 09-01]: test_optimize_synthetic_fish marked @pytest.mark.slow (5-10s on CPU with L-BFGS)
- [Phase 09-02]: --method flag defaults to triangulation; curve opt-in via --method curve in diagnose_pipeline.py

### Phase 4 Shelved Decisions (Analysis-by-Synthesis)

*These decisions apply only to the shelved analysis-by-synthesis pipeline. Retained for reference.*

- [04-01]: Vertex pre-projection approach for RefractiveSilhouetteRenderer: pre-project world->NDC via RefractiveCamera, then render with FoVOrthographicCameras (identity)
- [04-01]: Camera image size derived from K matrix: H=round(2*cy), W=round(2*cx)
- [04-01]: torch downgraded 2.10+cu130 -> 2.9.1+cu128, torchvision 0.24.1+cu128 (to fix pytorch3d DLL ABI mismatch)
- [04-01]: Gravity prior uses theta^2 (pitch proxy); explicit roll parameter deferred
- [04-01]: Angular diversity temperature: higher T = MORE spread; temperature=0.5 default
- [Phase 04-02]: Per-parameter Adam LR groups: p uses lr*5 per RESEARCH.md Z-anisotropy note
- [Phase 04-02]: MockRenderer Gaussian blob from vertex mean: GPU-free differentiable test harness
- [Phase 04-03]: run_holdout_validation uses round-robin (frame_idx % n_cameras)
- [Phase 04-03]: evaluate_holdout_iou uses existing optimized states when provided
- [Phase 04-03]: render_overlay uses BGR convention: (0,255,0) = green, (0,0,255) = red

### Roadmap Evolution

- Phase 02.1 inserted after Phase 02: Segmentation Troubleshooting (URGENT)
- Phase 02.1.1 inserted after Phase 02.1: Object-detection alternative to MOG2 (URGENT)
- Phase 04 shelved (2026-02-21): Analysis-by-synthesis too slow (30+ min/sec). Replaced by direct triangulation pipeline.
- Phases 05-06 rewritten (2026-02-21): Phase 5 = Cross-View Identity & 3D Tracking; Phase 6 = 2D Medial Axis & Arc-Length Sampling
- Phases 07-08 pending addition via /gsd:add-phase: Phase 7 = Triangulation & Spline Fitting; Phase 8 = Output & Visualization
- Phase 04.1 inserted after Phase 04: Isolate phase-4 specific code post-archive (URGENT)
- Phase 7 added: Multi-View Triangulation
- Phase 8 added: End-to-end integration testing and benchmarking
- Phase 9 added: Curve-based optimization as a replacement for triangulation

### Pending Todos

- **Consolidate scripts into CLI workflow** (tooling) — scripts/ now cleaned to 5 production scripts (build_training_data.py, eval_yolo_vs_mog2.py, organize_yolo_dataset.py, sample_yolo_frames.py, train_yolo.py)
- **Integrate full-frame exclusion masks from AquaMVS** (calibration) — load optional per-camera masks to filter out invalid regions before detection/segmentation/triangulation
- **Add Phase 7 and Phase 8 to roadmap** via /gsd:add-phase

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Fix triangulation bugs: NaN contamination, coupled thresholds, greedy orientation | 2026-02-23 | 4d3d07e | [1-fix-triangulation-bugs-nan-contamination](./quick/1-fix-triangulation-bugs-nan-contamination/) |
| 2 | Add synthetic data module (FishConfig, MidlineSet generation, GT Midline3D, --synthetic flag) | 2026-02-23 | b9dd734 | [2-add-synthetic-data-module-as-drop-in-rep](./quick/2-add-synthetic-data-module-as-drop-in-rep/) |
| 3 | Add synthetic mode diagnostic visualizations (3D GT comparison, camera overlays, error distribution, markdown report) | 2026-02-23 | b3a4dd7 | [3-add-synthetic-mode-diagnostic-visualizat](./quick/3-add-synthetic-mode-diagnostic-visualizat/) |
| 4 | Add drift dynamics + sinusoidal/compound spine shapes + diverse fish configs | 2026-02-23 | 58385c1 | [4-add-per-frame-position-drift-and-heading](./quick/4-add-per-frame-position-drift-and-heading/) |
| 5 | Investigate spline folding: numerical evidence that K=7 allows 150-degree fold with zero penalty; 6 ranked recommendations | 2026-02-23 | cedbbe1 | [5-improve-3d-spline-constraints-investigat](./quick/5-improve-3d-spline-constraints-investigat/) |

### Blockers/Concerns

- [Phase 1 - RESOLVED]: Z-uncertainty budget quantified: Z error is 132x larger than XY for top-down 13-camera rig (see docs/reports/z_uncertainty_report.md)
- [Phase 02.1 - RESOLVED]: MOG2 validated; YOLO added as alternative detector
- [Phase 4 - RESOLVED via pivot]: PyTorch3D sigma/gamma hyperparameters moot — no longer using differentiable rendering in primary pipeline
- [Phase 5 - NEW]: Cross-view identity has no existing implementation to build on — RANSAC centroid clustering is new code

## Session Continuity

Last session: 2026-02-23
Stopped at: Completed quick-5 (improve-3d-spline-constraints-investigat)
Next action: Implement spline folding fixes from quick-5 report — priority: chord-arc penalty + increase lambda_curvature. Then continue Phase 09 Plan 02 (integration/validation against real data).
