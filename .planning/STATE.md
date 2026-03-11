---
gsd_state_version: 1.0
milestone: v3.7
milestone_name: Improved Tracking
status: unknown
last_updated: "2026-03-10T23:59:19.824Z"
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 4
  completed_plans: 4
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-10)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.7 Improved Tracking — Phase 80 (Baseline Metrics)

## Current Position

Phase: 80 of v3.7 (Baseline Metrics)
Plan: 01 complete
Status: Plan 80-01 complete — ready to plan Phase 81 or next phase
Last activity: 2026-03-10 — Phase 80-01 complete, OC-SORT baseline established (27 tracks vs 9-fish target)

Progress: [███░░░░░░░] 30%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: ~10min
- Total execution time: ~20min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 78 | 2 | ~20min | ~10min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
v3.6 milestone decisions archived to milestones/v3.6-ROADMAP.md.

Recent decisions affecting current work:
- [v3.7 seed]: KF state dimension (60-dim vs 24-dim) to be resolved during Phase 83 planning
- [v3.7 seed]: BoxMot dependency removal decision deferred to Phase 85
- [v3.7 seed]: TRACK-10 (BYTE-style secondary pass) conditional on INV-04 findings in Phase 78
- [Phase 78]: GO recommendation — no keypoint identity jumps, no OBB merging during occlusion
- [Phase 78]: Phase 79 (Occlusion Remediation) skipped per GO decision
- [Phase 78]: Confidence threshold can drop to 0.1 with polygon NMS
- [Phase 78]: White-wall detection dropout identified — Phase 78.1 inserted for production retrain
- [Phase 78]: Multi-instance pose output is Gaussian NMS artifact, not cross-fish detection
- [Phase 78.1-01]: Tagged split mode required — random split causes augmentation leakage (children split across train/val)
- [Phase 78.1-01]: Val filter changed to source != pseudo — corrected labels are human-reviewed, eligible for val by default
- [Phase 78.1-01]: 4:1 augmentation ratio confirmed important (0.926 mAP50-95 at 2:1 vs 0.972 at 4:1)
- [Phase 78.1-01]: Production models: OBB run_20260310_115419 (mAP50-95=0.781), Pose production_retrain_4aug run (mAP50-95=0.972)
- [Phase 78.1-02]: OBB production model approved: mAP50-95=0.781 (+7.4 pts vs Round 1 0.707)
- [Phase 78.1-02]: Pose production model approved: mAP50-95=0.974 (+0.6 pts on harder all-source 128-image val)
- [Phase 78.1-02]: White-wall recall deferred — no systematic dropout, marginal cases remain, not blocking Phase 80+
- [Phase 78.1-02]: False positive above tank (conf=0.618) accepted — filtered by triangulation geometry; tank ROI mask deferred
- [Phase 78.1-02]: Both models deployed to config.yaml as production defaults (run_20260310_115419 OBB, run_20260310_171543 Pose)
- [Phase 80-01]: OC-SORT baseline: 27 tracks vs 9-fish target (3x over-fragmented) — primary failure mode for Phase 84 to address
- [Phase 80-01]: min_hits=1 used for honest baseline — no warm-up penalty that would artificially inflate track count
- [Phase 80-01]: Single-pass architecture in measure_baseline_tracking.py — detection+tracking+video rendering in one loop, avoids memory duplication

### Roadmap Evolution

- Phase 78.1 inserted after Phase 78: OBB & Pose Production Retrain (URGENT) — detection dropout on white-wall background persists after pseudo-label round; retrain with corrected pseudo-labels in train/val split and more epochs

### Pending Todos

10 pending todos -- see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Algae domain shift between manual annotations (clean tank) and current conditions may cause false positives in pseudo-labels
- Phase 79 skipped (GO decision); Phase 86 conditional on Phase 85 findings

## Session Continuity

Last session: 2026-03-10
Stopped at: Completed 80-01-PLAN.md — OC-SORT baseline established, evaluate_fragmentation_2d added, 80-BASELINE.md created
Resume file: N/A
