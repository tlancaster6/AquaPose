---
gsd_state_version: 1.0
milestone: v3.1
milestone_name: Reconstruction
status: ready_to_plan
last_updated: "2026-03-02T21:30:00Z"
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 40 — Diagnostic Capture

## Current Position

Phase: 40 of 45 (Diagnostic Capture)
Plan: 0 of ? in current phase
Status: Ready to plan
Last activity: 2026-03-02 — Roadmap created for v3.1 (6 phases, 19 requirements mapped)

Progress: [░░░░░░░░░░] 0%

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

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Phase 43 outlier rejection threshold cannot be set a priori; depends on Phase 41 eval harness output
- Coordinate space conversions (full-image <-> crop-space) remain a cross-cutting concern for midline backends

## Session Continuity

Last session: 2026-03-02
Stopped at: Roadmap created, Phase 40 ready to plan
Resume file: None
