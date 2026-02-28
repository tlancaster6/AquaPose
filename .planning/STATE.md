---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Backends
status: unknown
last_updated: "2026-02-28T18:06:59.927Z"
progress:
  total_phases: 1
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-28)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 30 — Config/Contracts (next)

## Current Position

Phase: 30 of 33 (Config/Contracts) — Phase 29 complete
Plan: 29-02 complete, Phase 29 fully done
Status: In progress
Last activity: 2026-02-28 — Completed 29-02: Add v2.2 Planned Features to GUIDEBOOK.md

Progress: [██░░░░░░░░] 17% (2/12 plans complete — Phase 29 both plans done)

## Performance Metrics

**Velocity:**
- Total plans completed: 0 (this milestone)
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 29-guidebook-audit | 2 | 5 min | 2.5 min |

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

### Pending Todos

12 pending todos from v2.1 — see .planning/todos/pending/

### Blockers/Concerns

- EVAL-01 deferred from v2.1: E2E regression tests skip due to pytestmark; rebuild with configurable N_SAMPLE_POINTS (CFG-02 in Phase 30 resolves this)
- OBB angle convention risk: ultralytics outputs radians clockwise in [-pi/4, 3pi/4), OpenCV uses degrees counter-clockwise — crop-orientation smoke test required before any keypoint training

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 29-02-PLAN.md (Add v2.2 Planned Features to GUIDEBOOK.md)
Resume file: None
