---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: Backends
status: planning
last_updated: "2026-02-28T20:00:00.000Z"
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-28)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Defining requirements for v2.2 Backends

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-28 — Milestone v2.2 Backends started

## Accumulated Context

### Decisions

Key decisions entering v2.2:

- YOLO-OBB is a configurable model within detection backend (not a swappable backend) — OBB output is a superset of standard bboxes, affine crop handles rotation
- Keypoint midline is a swappable backend alternative to segment-then-extract — fundamentally different approach (regression vs segmentation+skeletonization)
- U-Net encoder + regression head for keypoint model — transfer learning from existing segmentation weights
- Two training modes: frozen-backbone → unfreeze fine-tuning, OR standard pretrained-only training
- 6 anatomical keypoints → fit 2D curve → resample to N points + per-point confidence — reconstruction backends see standard N-point midlines
- Partial midlines: NaN + confidence=0 for unobserved regions, no extrapolation (body model deferred)
- Per-point confidence added to Midline2D — backward compatible, segment-then-extract fills uniform confidence
- N_SAMPLE_POINTS must be configurable (not hardcoded to 15), keypoint count inferred from model or config
- Training infrastructure: src/aquapose/training/ with CLI entry points (aquapose train <model>), absorb existing scripts
- device as first-level pipeline config parameter, stages keep tensors on starting device
- stop_frame promoted to top-level config, init-config shows user-relevant fields first, --synthetic flag
- Guidebook audit first phase — consistency with v2.1 codebase + planned v2.2 features

### Roadmap Evolution

(v2.2 roadmap pending)

### Pending Todos

- 12 pending todos from v2.1 (see .planning/todos/pending/)

### Blockers/Concerns

- EVAL-01 deferred from v2.1: regression test suite skipped with pytestmark; rebuild with configurable N_SAMPLE_POINTS

## Session Continuity

Last session: 2026-02-28
Stopped at: Milestone v2.2 initialization — defining requirements
