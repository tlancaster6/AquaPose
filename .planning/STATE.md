---
gsd_state_version: 1.0
milestone: v3.8
milestone_name: Improved Association
status: unknown
last_updated: "2026-03-11T19:25:55.023Z"
progress:
  total_phases: 13
  completed_phases: 12
  total_plans: 21
  completed_plans: 20
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-11)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.8 Improved Association — Phase 87: Tracklet2D Keypoint Propagation

## Current Position

Phase: 87 of 92 (Tracklet2D Keypoint Propagation)
Plan: 01 complete (1/1 plans done)
Status: Phase execution complete, pending verification
Last activity: 2026-03-11 — Phase 87 plan 01 executed

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

## Accumulated Context

### Decisions

Recent decisions affecting current work:
- v3.7: Custom OKS tracker with 24-dim KF stores keypoints internally in `_KptTrackletBuilder` but discards them in `to_tracklet2d()` — Phase 87 fixes this pass-through
- v3.7: ASSOC-01 keypoint centroid lost in Phase 85 BoxMot removal — superseded by full multi-keypoint scoring in Phase 88
- Research: Fragment merging removed (works against upstream fragmentation intent); refinement.py replaced by validation.py (richer signal, combines eviction + changepoint in one pass)
- Research: No new dependencies — ruptures rejected; custom O(n) prefix-sum changepoint sufficient
- [Phase 89]: Fragment merging removed permanently from association pipeline — works against upstream fragmentation intent; max_merge_gap removed from AssociationConfig and ClusteringConfigLike

### Pending Todos

10 pending todos — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- P2 (LUT coordinate space): Round-trip unit test required in Phase 88 before any end-to-end eval — same bug caused 86% singleton rate in v3.7
- P7 (refinement removal): Grep audit of `per_frame_confidence` and `consensus_centroids` consumers must happen before `refinement.py` is deleted in Phase 90
- Phase 90 changepoint threshold needs calibration against confirmed-correct v3.7 benchmark tracklets; false positive target < 30%

## Session Continuity

Last session: 2026-03-11
Stopped at: Roadmap created — ready to plan Phase 87
Resume file: None
