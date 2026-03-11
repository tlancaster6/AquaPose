---
gsd_state_version: 1.0
milestone: v3.8
milestone_name: Improved Association
status: unknown
last_updated: "2026-03-11T20:30:18.225Z"
progress:
  total_phases: 15
  completed_phases: 13
  total_plans: 25
  completed_plans: 23
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-11)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.8 Improved Association — Phase 91: Singleton Recovery

## Current Position

Phase: 91 of 92 (Singleton Recovery)
Plan: 01 complete (1/1 plans done)
Status: Phase execution complete, ready for next phase
Last activity: 2026-03-11 — Phase 91 plan 01 executed

Progress: [██░░░░░░░░] 17%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

*Updated after each plan completion*
| Phase 89 P01 | 4min | 2 tasks | 6 files |
| Phase 91 P01 | 13min | 2 tasks | 5 files |

## Accumulated Context

### Decisions

Recent decisions affecting current work:
- v3.7: Custom OKS tracker with 24-dim KF stores keypoints internally in `_KptTrackletBuilder` but discards them in `to_tracklet2d()` — Phase 87 fixes this pass-through
- v3.7: ASSOC-01 keypoint centroid lost in Phase 85 BoxMot removal — superseded by full multi-keypoint scoring in Phase 88
- Research: Fragment merging removed (works against upstream fragmentation intent); refinement.py replaced by validation.py (richer signal, combines eviction + changepoint in one pass)
- Research: No new dependencies — ruptures rejected; custom O(n) prefix-sum changepoint sufficient
- [Phase 89]: Fragment merging removed permanently from association pipeline — works against upstream fragmentation intent; max_merge_gap removed from AssociationConfig and ClusteringConfigLike
- [Phase 91]: Module independence: recovery.py copies _point_to_ray_distance() standalone — no cross-module imports from validation.py or refinement.py
- [Phase 91]: Staleness invalidation: per_frame_confidence and consensus_centroids set to None on groups that gain new tracklets after recovery
- [Phase 91]: Split-assign requires both segments to match DIFFERENT groups — single-segment match leaves singleton unchanged

### Pending Todos

10 pending todos — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- P2 (LUT coordinate space): Round-trip unit test required in Phase 88 before any end-to-end eval — same bug caused 86% singleton rate in v3.7
- P7 (refinement removal): Grep audit of `per_frame_confidence` and `consensus_centroids` consumers must happen before `refinement.py` is deleted in Phase 90
- Phase 90 changepoint threshold needs calibration against confirmed-correct v3.7 benchmark tracklets; false positive target < 30%

## Session Continuity

Last session: 2026-03-11
Stopped at: Completed 91-01-PLAN.md
Resume file: None
