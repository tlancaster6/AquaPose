---
gsd_state_version: 1.0
milestone: v3.1
milestone_name: Reconstruction
status: defining_requirements
last_updated: "2026-03-02T21:00:00Z"
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.1 Reconstruction — rebuild from minimal triangulation baseline with evaluation harness

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-02 — Milestone v3.1 started

## Accumulated Context

### Decisions

Key decisions from v3.0 (carried forward):
- Backend names are "segmentation" and "pose_estimation" (not yolo_seg/yolo_pose)
- Standard YOLO txt+yaml format for all training data (NDJSON was tried and reverted)
- Single `weights_path` field in both DetectionConfig and MidlineConfig
- core/types/ contains shared cross-stage types (Detection, CropRegion, AffineCrop, Midline2D, Midline3D, MidlineSet)
- Legacy reconstruction/, segmentation/, tracking/ dirs eliminated — all code under core/

v3.1 strategic decisions (from reconstruction rebuild proposal):
- Start with triangulation, not curve optimization — stateless, debuggable, measurable
- Pose estimation backend only — ordered keypoints eliminate correspondence/orientation machinery
- Orientation lives in midline stage, not reconstruction
- Half-widths are pass-through only — not used for decisions, weighting, or rejection
- Single triangulation strategy regardless of camera count (no 2/3-7/8+ branching)
- ~300 frames (10 sec) working dataset — chunk processing out of scope

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance to next milestone)

### Blockers/Concerns

**Coordinate space conversions remain a cross-cutting concern:**
Full-image ↔ crop-space conversions are a pervasive source of error. Both YOLO-seg and YOLO-pose backends use affine warps for crop extraction and back-projection. Unit tests mock YOLO inference — real end-to-end with trained weights needs human verification.

## Session Continuity

Last session: 2026-03-02
Stopped at: Defining v3.1 requirements
Resume file: None
