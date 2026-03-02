---
gsd_state_version: 1.0
milestone: v3.1
milestone_name: Reconstruction
status: unknown
last_updated: "2026-03-02T20:38:26.820Z"
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 5
  completed_plans: 4
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 41 — Evaluation Harness

## Current Position

Phase: 41 of 45 (Evaluation Harness)
Plan: 2 of N in current phase (41-02 complete)
Status: In progress
Last activity: 2026-03-02 — Completed 41-02: Evaluation harness with frame selection, Tier 1/Tier 2 metrics, ASCII + JSON output

Progress: [██░░░░░░░░] 10%

## Accumulated Context

### Decisions

Key decisions from v3.0 (carried forward):
- Backend names are "segmentation" and "pose_estimation" (not yolo_seg/yolo_pose)
- Standard YOLO txt+yaml format for all training data
- Single `weights_path` field in DetectionConfig and MidlineConfig
- core/types/ contains shared cross-stage types (MidlineSet, Midline2D, Midline3D, etc.)

v3.1 strategic decisions:
- Start with triangulation, not curve optimization — stateless, debuggable, measurable
- Pose estimation backend only — ordered keypoints eliminate correspondence/orientation machinery
- Half-widths are pass-through only — not used for decisions, weighting, or rejection
- Single triangulation strategy regardless of camera count (no camera-count branching)
- ~300 frames (10 sec) working dataset — chunk processing out of scope
- Outlier rejection threshold (RECON-02) requires empirical tuning via eval harness
- [Phase 40-diagnostic-capture]: NPZ key convention uses flat slash-separated keys for numpy.load compatibility
- [Phase 40-diagnostic-capture]: MidlineFixture is a data contract only (no loader) - loader deferred to Plan 02
- [Phase 40-diagnostic-capture]: No min_cameras filter on frame inclusion - all frames with at least one midline captured
- [Phase 40-diagnostic-capture]: load_midline_fixture derives frame_indices from parsed midline keys (not meta/frame_indices) to handle empty fixtures correctly
- [Phase 40-diagnostic-capture]: camera_ids in MidlineFixture loaded from meta/camera_ids to preserve original capture ordering
- [Phase 41-01-calib-bundle]: NPZ_VERSION updated to 2.0; both 1.0 and 2.0 accepted in loader via _SUPPORTED_VERSIONS
- [Phase 41-01-calib-bundle]: export_midline_fixtures writes 1.0 (no calib/) when models=None, 2.0 (with calib/) when models provided
- [Phase 41-01-calib-bundle]: CalibBundle is frozen dataclass; per-camera keys discovered dynamically by scanning NPZ key names
- [Phase 41-01-calib-bundle]: Shared calibration params (water_z, n_air, n_water, interface_normal) taken from first model in dict
- [Phase 41-02]: Harness calls triangulate_midlines() directly (not TriangulationBackend) to avoid calibration file dependency
- [Phase 41-02]: Tier 2 leave-one-out reduced midline_set excludes dropout_cam for ALL fish (not just current fish)

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Phase 43 outlier rejection threshold cannot be set a priori; depends on Phase 41 eval harness output
- Coordinate space conversions (full-image <-> crop-space) remain a cross-cutting concern for midline backends

## Session Continuity

Last session: 2026-03-02
Stopped at: Completed 41-evaluation-harness/41-02-PLAN.md
Resume file: None
