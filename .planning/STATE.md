---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Backends
status: unknown
last_updated: "2026-03-01T01:50:20.589Z"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 12
  completed_plans: 11
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-28)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 33 complete — ready for Phase 34 (Stabilization)

## Current Position

Phase: 33 of 35 (Keypoint Midline Backend) — ALL PLANS COMPLETE
Plan: 33-02 complete
Status: In progress
Last activity: 2026-03-01 - Completed 33-02: Confidence-weighted triangulation and curve optimizer

Progress: [█████████░] 79% (11/14 plans complete — Phase 29 both plans done, Phase 30 Plans 01-04 done, Phase 31 Plans 01-02 done, Phase 32 Plans 01-02 done, Phase 33 Plans 01-02 done)

## Performance Metrics

**Velocity:**
- Total plans completed: 3 (this milestone)
- Average duration: 18 min
- Total execution time: 53 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 29-guidebook-audit | 2 | 5 min | 2.5 min |
| 30-config-and-contracts | 3/4 done | 45 min | 15 min |

| 31-training-infrastructure | 2 | 26 min | 13 min |
| 32-yolo-obb-detection-backend | 2 | 36 min | 18 min |
| 33-keypoint-midline-backend | 2 | 58 min | 29 min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Key decisions entering v2.2 (see PROJECT.md Key Decisions for full log):

- YOLO-OBB is a configurable model within the detection backend — OBB output extends Detection with optional angle/obb_points fields
- Keypoint midline is a swappable backend (direct_pose) alongside segment_then_extract — regression vs skeletonization
- U-Net encoder + regression head; frozen backbone initially, optional unfreeze for fine-tuning
- Partial midlines: NaN + confidence=0 for unobserved regions; always output exactly n_sample_points
- N_SAMPLE_POINTS moved to ReconstructionConfig.n_points (default 15) — no hardcoded literals anywhere
- device at top-level PipelineConfig, propagated through build_stages()
- training/ must not import engine/ (AST import boundary enforced by pre-commit)

From 29-01 execution:
- GUIDEBOOK.md Sections 16 (Definition of Done) and 18 (Discretionary Items) deleted — roadmap has per-phase success criteria; guidebook is not the right place for discretionary items

From 29-02 execution:
- Keypoint midline NaN policy locked as architectural contract: evaluate spline only in [t_min_observed, t_max_observed], NaN + confidence=0 outside — Phase 33 implementers must treat this as authoritative
- YOLO-OBB documented as configurable model with optional Detection fields (angle, obb_points) — no pipeline changes needed
- Confidence-weighted triangulation: per-point confidence flows from keypoint backend to Stage 5; uniform weights when confidence is None

From 30-01 execution:
- Detection.angle uses standard math radians [-pi, pi]; YOLO-OBB backend (Plan 32) handles angle convention conversion at the boundary
- segment_then_extract always fills point_confidence=1.0 — locked contract, skeletonization has no per-point uncertainty model
- _filter_fields() now applied to all 8 config types with strict reject; unknown fields raise ValueError

From 30-02 execution:
- device auto-detected via _default_device() using torch.cuda.is_available(); defaults to cuda:0 when GPU present
- n_sample_points default is 10 (not 15); propagates to midline.n_points in load_config() unless midline.n_points explicitly set
- n_animals sentinel is 0; load_config() raises ValueError when resolved_n_animals <= 0
- device and stop_frame removed from DetectionConfig; _RENAME_HINTS updated with did-you-mean hints
- N_SAMPLE_POINTS=15 kept as module-level fallback constant; all pipeline modules accept n_sample_points as constructor parameter

From 30-03 execution:
- init-config CLI uses positional <name> arg (not --output/-o); creates ~/aquapose/projects/<name>/ scaffold
- Generated YAML omits device and stop_frame; n_animals placeholder is string "SET_ME" (fails load_config validation)
- project_dir resolution in load_config() uses Path.resolve() — adds drive letter on Windows; tests use tmp_path to stay cross-platform
- synthetic section in generated YAML excludes fish_count — it propagates from n_animals at runtime
- [Phase 31-01]: train_unet() uses data_dir convention (annotations.json + images in same dir) instead of old coco_json + image_root pair for simpler CLI interface
- [Phase 31-01]: training/ package rewrites BinaryMaskDataset and stratified_split from segmentation/ as fresh implementations with identical public API

From 31-02 execution:
- _PoseModel borrows enc0-enc4 only from _UNet; decoder replaced by regression head (AdaptiveAvgPool2d → flatten → Linear → ReLU → Linear → Sigmoid)
- stratified_split broadened to _HasImages Protocol so KeypointDataset qualifies without circular import
- segmentation/training.py and segmentation/dataset.py deleted; segmentation/__init__.py cleaned of training re-exports
- test_training.py migrated to train_unet(data_dir=...) convention; old evaluate() not ported (no equivalent)

From 32-01 execution:
- crop_size stored as list[int] not tuple[int,int] — Python tuples serialize as !!python/tuple in PyYAML which yaml.safe_load cannot parse
- OBB angle conversion (negate) happens once in YOLOOBBBackend.detect() — Detection.angle is always standard math CCW radians after that
- extract_affine_crop() uses cv2.BORDER_CONSTANT=0 for zero-fill letterboxing; crop canvas size is always crop_size regardless of obb_w/obb_h
- invert_affine_point/invert_affine_points round-trip error < 1px confirmed for 6 angles including 0, pi/4, pi/2, -pi/3, pi/6, -pi
- [Phase 32-yolo-obb-detection-backend]: OBB polygon replaces AABB in both visualization observers; _match_detection uses centroid-to-bbox-center distance; detections accessed via getattr for graceful degradation

From 33-01 execution:
- Detection dataclass has no centroid field; DirectPoseBackend derives centroid from bbox via hasattr guard — same pattern as association/stage.py
- CubicSpline falls back to linear interp1d when < 4 unique t-values visible, preventing scipy ValueError on partial visibility
- _PoseModel and torch.load are lazy-imported inside __init__; tests patch at aquapose.training.pose._PoseModel and torch.load globally, then replace backend._model directly
- MidlineConfig extended with 4 new fields (all with defaults) — existing YAML configs load without error
- [Phase 33-02]: Weighted triangulation uses normal equations matching triangulate_rays() approach; _tri_rays() local helper dispatches weighted/unweighted; obs->proj weighted in chamfer, proj->obs unweighted; confidence_per_fish parallel structure propagated to all _data_loss() calls

### Pending Todos

12 pending todos from v2.1 — see .planning/todos/pending/

### Blockers/Concerns

- OBB angle convention risk RESOLVED: YOLOOBBBackend.detect() negates ultralytics CW angle to standard math CCW — verified by unit test

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 10 | Store 3D consensus centroids on TrackletGroup and write centroid correspondences to disk via DiagnosticObserver | 2026-02-28 | 10395eb | Verified | [10-store-3d-consensus-centroids-on-tracklet](./quick/10-store-3d-consensus-centroids-on-tracklet/) |

## Session Continuity

Last session: 2026-03-01
Stopped at: Completed 33-02-PLAN.md (Confidence-Weighted Reconstruction)
Resume file: None
