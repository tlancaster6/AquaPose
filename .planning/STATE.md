---
gsd_state_version: 1.0
milestone: v3.7
milestone_name: Improved Tracking
status: roadmap
last_updated: "2026-03-10"
progress:
  total_phases: 9
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-10)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.7 Improved Tracking — Phase 78.1 (OBB & Pose Production Retrain)

## Current Position

Phase: 78.1 of v3.7 (OBB & Pose Production Retrain)
Plan: 01 complete, 02 pending
Status: In progress — Plan 01 complete, proceeding to Plan 02
Last activity: 2026-03-10 — Plan 78.1-01 complete (dataset assembly + production training runs done)

Progress: [█░░░░░░░░░] 10%

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

### Roadmap Evolution

- Phase 78.1 inserted after Phase 78: OBB & Pose Production Retrain (URGENT) — detection dropout on white-wall background persists after pseudo-label round; retrain with corrected pseudo-labels in train/val split and more epochs

### Pending Todos

10 pending todos -- see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Algae domain shift between manual annotations (clean tank) and current conditions may cause false positives in pseudo-labels
- Phase 79 skipped (GO decision); Phase 86 conditional on Phase 85 findings

## Session Continuity

Last session: 2026-03-10
Stopped at: Phase 78.1 Plan 01 complete — production OBB and Pose models trained; proceed to Plan 02
Resume file: N/A
