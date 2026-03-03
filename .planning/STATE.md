---
gsd_state_version: 1.0
milestone: v3.2
milestone_name: Evaluation Ecosystem
status: defining_requirements
last_updated: "2026-03-03"
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-03)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.2 Evaluation Ecosystem — defining requirements

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-03 — Milestone v3.2 started

## Accumulated Context

### Decisions

Carried forward from v3.1:
- DLT is the sole reconstruction backend (old triangulation + curve optimizer removed)
- Outlier rejection threshold tuned to 10.0 (from 50.0)
- Association defaults accepted (sweep showed marginal gains; ~70% singleton rate is upstream bottleneck)
- NPZ fixture v2.0 format with CalibBundle for self-contained offline evaluation
- Pose estimation backend only for reconstruction — ordered keypoints eliminate correspondence machinery

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Coordinate space conversions (full-image <-> crop-space) remain a cross-cutting concern for midline backends
- ~70% singleton rate in association — upstream detection/tracking coverage is the bottleneck

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 15 | Replace binary inlier counting with soft linear scoring kernel and remove ghost penalty | 2026-03-03 | df0e470 | [15-replace-binary-inlier-counting-with-soft](./quick/15-replace-binary-inlier-counting-with-soft/) |
| 16 | Restructure tune_association.py with joint 2D grid sweep for ray_distance_threshold x score_min | 2026-03-03 | 4905671 | [16-restructure-tune-association-py-with-joi](./quick/16-restructure-tune-association-py-with-joi/) |
| 17 | Unify n_sample_points config: remove N_SAMPLE_POINTS constant, remove MidlineConfig.n_points, add ReconstructionConfig.n_sample_points, default to 15 | 2026-03-03 | 07a8314 | [17-unify-n-sample-points-config](./quick/17-unify-n-sample-points-config/) |

## Session Continuity

Last session: 2026-03-03
Stopped at: Milestone v3.2 started — defining requirements
Resume file: None
