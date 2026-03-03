---
gsd_state_version: 1.0
milestone: v3.1
milestone_name: Reconstruction
status: unknown
last_updated: "2026-03-03T02:21:14.885Z"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 11
  completed_plans: 10
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 44 — Validation and Tuning

## Current Position

Phase: 44 of 45 (Validation and Tuning)
Plan: 1 of 2 in current phase (44-01 complete)
Status: In progress
Last activity: 2026-03-03 - Completed quick task 15: Replace binary inlier counting with soft linear scoring kernel and remove ghost penalty

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
- [Phase 42-01]: flag_outliers returns empty set for <2 values or std==0
- [Phase 42-01]: Baseline JSON augments eval_results.json with baseline_metadata key (not a new schema)
- [Phase 42-01]: No tests for measure_baseline.py - manual execution only per CONTEXT.md
- [Phase 43]: MIN_BODY_POINTS re-exported from triangulation.py via noqa F401 to preserve existing test imports without changing test files
- [Phase 43]: Backward-compat aliases kept in triangulation.py for zero-change backward compatibility with existing private-function imports
- [Phase 43]: DltBackend uses single-strategy DLT (triangulate → reject → re-triangulate) with no camera-count branching, no orientation alignment, no epipolar refinement

### Roadmap Evolution

- Phase 43.1 inserted after Phase 43: Association Tuning (URGENT)

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Phase 43 outlier rejection threshold cannot be set a priori; depends on Phase 41 eval harness output
- Coordinate space conversions (full-image <-> crop-space) remain a cross-cutting concern for midline backends

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 15 | Replace binary inlier counting with soft linear scoring kernel and remove ghost penalty | 2026-03-03 | df0e470 | [15-replace-binary-inlier-counting-with-soft](./quick/15-replace-binary-inlier-counting-with-soft/) |

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed quick task 15 — soft scoring kernel replacing binary inlier counting in association scoring
Resume file: None
