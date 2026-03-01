---
gsd_state_version: 1.0
milestone: v3.0
milestone_name: Ultralytics Unification
status: roadmap_ready
last_updated: "2026-03-01"
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 35 — Codebase Cleanup (ready to plan)

## Current Position

Phase: 35 of 37 (Codebase Cleanup)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-03-01 — v3.0 roadmap created (3 phases, 12 requirements mapped)

Progress: [░░░░░░░░░░] 0%

## Accumulated Context

### Decisions

Key decisions entering v3.0:

- v2.2 terminated early: custom U-Net segmentation (IoU 0.623) and custom keypoint regression both performed poorly — insufficient for production use
- Ultralytics unification chosen: YOLO11n-seg replaces U-Net segmentation, YOLO11n-pose replaces keypoint regression
- Backwards compatibility explicitly not a concern — strip old code, start fresh with Ultralytics only
- YOLO-OBB detection backend (Phase 32) is kept — already uses Ultralytics and works well
- Existing Stage Protocol architecture retained — new models plug in as backends within the 5-stage pipeline
- Training uses NDJSON format; existing `scripts/build_yolo_training_data.py` covers OBB/pose; only seg conversion is new (DATA-01)
- 6 anatomical keypoints: nose, head, spine1, spine2, spine3, tail

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance to v3.0)

### Blockers/Concerns

**CRITICAL — Coordinate space conversions (cross-cutting, all phases):**
Full-image ↔ crop-space conversions are a pervasive source of error, especially with OBB affine warps. Mismatches between how crops are prepared for training vs inference cause silent accuracy failures. Existing machinery (`extract_affine_crop`, `invert_affine_point`, `transform_keypoints`) should be reused but with extreme attention to coordinate normalization at every boundary. v2.2 development surfaced and fixed many such bugs, but confidence in correctness remains low. Every phase must explicitly verify coordinate round-trips.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 12 | Add tool script for YOLO-OBB and YOLO-pose training set generation from multi-animal pose annotations | 2026-03-01 | 304bc09 | [12-add-tool-script-for-yolo-obb-and-yolo-po](./quick/12-add-tool-script-for-yolo-obb-and-yolo-po/) |

## Session Continuity

Last session: 2026-03-01
Stopped at: v3.0 roadmap created — ready to plan Phase 35
Resume file: None
