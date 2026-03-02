---
gsd_state_version: 1.0
milestone: null
milestone_name: null
status: between_milestones
last_updated: "2026-03-02T20:00:00Z"
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
**Current focus:** Between milestones — v3.0 Ultralytics Unification shipped 2026-03-02

## Current Position

Phase: None (between milestones)
Plan: N/A
Status: v3.0 milestone complete, all 16 requirements satisfied, audit passed
Last activity: 2026-03-02 — Completed v3.0 milestone (5 phases, 14 plans, 656 tests passing)

## Accumulated Context

### Decisions

Key decisions from v3.0 (carried forward for next milestone):
- Backend names are "segmentation" and "pose_estimation" (not yolo_seg/yolo_pose)
- Standard YOLO txt+yaml format for all training data (NDJSON was tried and reverted)
- Single `weights_path` field in both DetectionConfig and MidlineConfig
- core/types/ contains shared cross-stage types (Detection, CropRegion, AffineCrop, Midline2D, Midline3D, MidlineSet)
- Legacy reconstruction/, segmentation/, tracking/ dirs eliminated — all code under core/

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance to next milestone)

### Blockers/Concerns

**Coordinate space conversions remain a cross-cutting concern:**
Full-image ↔ crop-space conversions are a pervasive source of error. Both YOLO-seg and YOLO-pose backends use affine warps for crop extraction and back-projection. Unit tests mock YOLO inference — real end-to-end with trained weights needs human verification.

## Session Continuity

Last session: 2026-03-02
Stopped at: Completed v3.0 milestone archival
Resume file: None
