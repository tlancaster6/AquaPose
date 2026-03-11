---
gsd_state_version: 1.0
milestone: v3.7
milestone_name: Improved Tracking
status: unknown
last_updated: "2026-03-11T01:13:48.348Z"
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 7
  completed_plans: 5
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-10)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.7 Improved Tracking — Phase 81 (Pipeline Reorder)

## Current Position

Phase: 81 of v3.7 (Pipeline Reorder — Segmentation Removal)
Plan: 01 complete
Status: Plan 81-01 complete — v3.7 pipeline structure in place; ready for Phase 82 (keypoint-cost-association)
Last activity: 2026-03-11 — Phase 81-01 complete, core/midline renamed to core/pose, PoseStage rewritten for in-place Detection enrichment

Progress: [████░░░░░░] 40%

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
- [Phase 81-01]: PoseStage enriches Detection objects in-place (no AnnotatedDetection wrapper, no annotated_detections context field)
- [Phase 81-01]: process_batch() returns (kpts_xy, kpts_conf) tuples; PoseStage handles back-projection to full-frame coordinates
- [Phase 81-01]: Segmentation backend retained in codebase but removed from backend registry (not user-facing)
- [Phase 81-01]: ReconstructionStage now raises ValueError if tracklet_groups is None (removed stale annotated_detections fallback)

### Roadmap Evolution

- Phase 78.1 inserted after Phase 78: OBB & Pose Production Retrain (URGENT) — detection dropout on white-wall background persists after pseudo-label round; retrain with corrected pseudo-labels in train/val split and more epochs

### Pending Todos

10 pending todos -- see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Algae domain shift between manual annotations (clean tank) and current conditions may cause false positives in pseudo-labels
- Phase 79 skipped (GO decision); Phase 86 conditional on Phase 85 findings

## Session Continuity

Last session: 2026-03-11
Stopped at: Completed 81-01-PLAN.md — core/midline renamed to core/pose, PoseStage enriches Detection in-place, pipeline reordered to Detection->Pose->Tracking->Association->Reconstruction
Resume file: N/A
