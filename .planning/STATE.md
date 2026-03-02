---
gsd_state_version: 1.0
milestone: v3.1
milestone_name: Reconstruction
status: unknown
last_updated: "2026-03-02T19:55:07.427Z"
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 40 — Diagnostic Capture

## Current Position

Phase: 40 of 45 (Diagnostic Capture)
Plan: 2 of 2 in current phase (40-01, 40-02 complete)
Status: In progress
Last activity: 2026-03-02 — Completed 40-02: load_midline_fixture NPZ deserializer with round-trip tests

Progress: [█░░░░░░░░░] 5%

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

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Phase 43 outlier rejection threshold cannot be set a priori; depends on Phase 41 eval harness output
- Coordinate space conversions (full-image <-> crop-space) remain a cross-cutting concern for midline backends

## Session Continuity

Last session: 2026-03-02
Stopped at: Completed 40-diagnostic-capture/40-02-PLAN.md
Resume file: None
