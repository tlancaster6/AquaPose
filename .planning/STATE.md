---
gsd_state_version: 1.0
milestone: v3.7
milestone_name: Improved Tracking
status: unknown
last_updated: "2026-03-11T15:47:50.452Z"
progress:
  total_phases: 9
  completed_phases: 8
  total_plans: 16
  completed_plans: 15
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-10)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.7 Improved Tracking — Phase 83 (Custom Keypoint Tracker)

## Current Position

Phase: 86 of v3.7 (Cleanup Conditional)
Plan: 01 complete (cross-chunk handoff fix + dead code removal)
Status: Executing Phase 86 — Plan 01 done, Plan 02 in progress
Last activity: 2026-03-11 — Plan 86-01 complete; handoff bug fixed, dead types removed

Progress: [█████████░] 90%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: ~8min
- Total execution time: ~25min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 78 | 2 | ~20min | ~10min |
| 82 | 1 | ~5min | ~5min |
| Phase 83-custom-tracker-implementation P01 | 9 | 1 tasks | 5 files |
| Phase 83 P02 | 501 | 2 tasks | 5 files |
| Phase 84-integration-evaluation P01 | 45 | 2 tasks | 6 files |

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
- [Phase 81-02]: _keypoints_to_midline t_values=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0] as module-level constant — avoids cross-module coupling with PoseConfig
- [Phase 81-02]: tuning.py/runner.py use dual-path (detections v3.7 + annotated_detections legacy fallback) for backward compat with old diagnostic runs
- [Phase 82-01]: spine1 (index 2) selected as default centroid keypoint — mid-body, most stable under frame clipping and occlusion
- [Phase 82-01]: centroid_confidence_floor=0.3 matches pose backend default; interior keypoints reliably exceed threshold in production
- [Phase 83-01]: KF state dimension is 24 (6 kpts x 2D x 2), not 60 — CONTEXT.md 60-dim was conceptual shorthand
- [Phase 83-01]: _KptTrackletBuilder independent from ocsort_wrapper._TrackletBuilder — stores keypoints+confs for Plan 02 bidirectional merge
- [Phase 83]: KeypointTracker.get_tracklets() uses _collect_merged_builders() returning raw builders for mutable gap interpolation before Tracklet2D conversion
- [Phase 83]: oks_sigmas not in TrackingConfig — loaded from DEFAULT_SIGMAS at construction time to avoid config/sigma coupling
- [Phase 84-integration-evaluation]: keypoint_bidi produces 44 tracks vs OC-SORT 30 vs target 9 — both over-fragment at occlusion; root cause is identity-breaking at occlusion events, not temporal gaps (both trackers have 0 gaps, continuity=1.000)
- [Phase 84-integration-evaluation]: BYTE-style pass (TRACK-10) triggered by coverage=0.898 < 0.90; deferred — root cause is occlusion reacquisition, not low-confidence detection misses; a BYTE pass would not fix 44-track fragmentation
- [Phase 84-integration-evaluation]: Bidi merge was already stripped before plan 84-02 ran; only stale comments remained
- [Phase 84-integration-evaluation]: Single-pass keypoint_bidi: 42 tracks 93.6% coverage; bidi was 44 tracks 89.8%; BYTE trigger no longer fires

### Roadmap Evolution

- Phase 78.1 inserted after Phase 78: OBB & Pose Production Retrain (URGENT) — detection dropout on white-wall background persists after pseudo-label round; retrain with corrected pseudo-labels in train/val split and more epochs
- Phase 84.1 inserted after Phase 84: tracker tuning (URGENT)

### Pending Todos

10 pending todos -- see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

- Algae domain shift between manual annotations (clean tank) and current conditions may cause false positives in pseudo-labels
- Phase 79 skipped (GO decision); Phase 86 conditional on Phase 85 findings

## Session Continuity

Last session: 2026-03-11
Stopped at: Phase 84-02 complete — all 4 tasks done, human-verify approved. Ready for Phase 85.
Resume file: N/A
