---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Backends
status: in_progress
last_updated: "2026-02-28T19:55:00.000Z"
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 12
  completed_plans: 3
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-28)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 30 — Config/Contracts (in progress, Plan 01 complete)

## Current Position

Phase: 30 of 35 (Config/Contracts) — Plan 30-01 complete
Plan: 30-01 complete
Status: In progress
Last activity: 2026-02-28 — Completed 30-01: Extend Dataclasses and Universalize Config Validation

Progress: [██░░░░░░░░] 25% (3/12 plans complete — Phase 29 both plans done, Phase 30 Plan 01 done)

## Performance Metrics

**Velocity:**
- Total plans completed: 1 (this milestone)
- Average duration: 9 min
- Total execution time: 9 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 29-guidebook-audit | 2 | 5 min | 2.5 min |
| 30-config-and-contracts | 1/4 done | 9 min | 9 min |

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
- _RENAME_HINTS excludes detection.device and detection.stop_frame — those fields still exist on DetectionConfig and will be moved in Plan 02
- _filter_fields() now applied to all 8 config types with strict reject; unknown fields raise ValueError

### Pending Todos

12 pending todos from v2.1 — see .planning/todos/pending/

### Blockers/Concerns

- EVAL-01 deferred from v2.1: E2E regression tests skip due to pytestmark; rebuild with configurable N_SAMPLE_POINTS (CFG-02 in Phase 30 resolves this)
- OBB angle convention risk: ultralytics outputs radians clockwise in [-pi/4, 3pi/4), OpenCV uses degrees counter-clockwise — crop-orientation smoke test required before any keypoint training

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 30-01-PLAN.md (Extend Dataclasses and Universalize Config Validation)
Resume file: None
