---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: Backends
status: in_progress
last_updated: "2026-02-28T17:51:11Z"
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 2
  completed_plans: 1
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-28)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 29 — Guidebook Audit (plan 01 complete, plan 02 remaining)

## Current Position

Phase: 29 of 33 (Guidebook Audit)
Plan: 29-01 complete, 29-02 next
Status: In progress
Last activity: 2026-02-28 — Completed 29-01: Audit and Fix Stale Content in GUIDEBOOK.md

Progress: [█░░░░░░░░░] 8% (1/12 plans complete — 1 phase * 2 plans in phase 29)

## Performance Metrics

**Velocity:**
- Total plans completed: 0 (this milestone)
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 29-guidebook-audit | 1 | 2 min | 2 min |

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

### Pending Todos

12 pending todos from v2.1 — see .planning/todos/pending/

### Blockers/Concerns

- EVAL-01 deferred from v2.1: E2E regression tests skip due to pytestmark; rebuild with configurable N_SAMPLE_POINTS (CFG-02 in Phase 30 resolves this)
- OBB angle convention risk: ultralytics outputs radians clockwise in [-pi/4, 3pi/4), OpenCV uses degrees counter-clockwise — crop-orientation smoke test required before any keypoint training

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 29-01-PLAN.md (Audit and Fix Stale Content in GUIDEBOOK.md)
Resume file: None
