---
gsd_state_version: 1.0
milestone: v3.8
milestone_name: Improved Association
status: in_progress
last_updated: "2026-03-11T20:31:11Z"
progress:
  total_phases: 15
  completed_phases: 14
  total_plans: 27
  completed_plans: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-11)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.8 Improved Association — Phase 92: Parameter Tuning Pass

## Current Position

Phase: 92 of 92 (Parameter Tuning Pass)
Plan: 01 complete (1/2 plans done)
Status: Phase 92 plan 01 complete, ready for plan 02
Last activity: 2026-03-11 — Phase 92 plan 01 executed

Progress: [████░░░░░░] 40%

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
| Phase 91-singleton-recovery P02 | 5min | 1 tasks | 1 files |
| Phase 92 P01 | 9min | 2 tasks | 9 files |

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
- [Phase 91-singleton-recovery]: recovery_enabled guard added at stage.py call site (not just inside recover_singletons) to avoid lazy import overhead when disabled
- [Phase 92-01]: Centroid-only toggle placed BEFORE keypoints=None check in score_tracklet_pair so toggle is respected even when keypoints are populated
- [Phase 92-01]: keypoint_confidence_floor added to joint (Phase 1) grid not carry-forward — it interacts tightly with ray_dist and score_min
- [Phase 92-01]: 3D joint grid gives 27 combos acceptable for sweep time

### Pending Todos

10 pending todos — see .planning/todos/pending/ (review for relevance)

### Roadmap Evolution

- Phase 91.1 inserted after Phase 91: association bottleneck investigation and remediation (URGENT)

### Blockers/Concerns

- P2 (LUT coordinate space): Round-trip unit test required in Phase 88 before any end-to-end eval — same bug caused 86% singleton rate in v3.7
- P7 (refinement removal): Grep audit of `per_frame_confidence` and `consensus_centroids` consumers must happen before `refinement.py` is deleted in Phase 90
- Phase 90 changepoint threshold needs calibration against confirmed-correct v3.7 benchmark tracklets; false positive target < 30%

## Session Continuity

Last session: 2026-03-11
Stopped at: Completed 92-01-PLAN.md
Resume file: None
