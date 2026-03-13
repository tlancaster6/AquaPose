---
phase: 84-integration-evaluation
plan: 01
subsystem: tracking
tags: [keypoint-tracker, ocsort, evaluation, pipeline-wiring, metrics]

# Dependency graph
requires:
  - phase: 83-custom-tracker-implementation
    provides: KeypointTracker (keypoint_bidi backend) — bidirectional merge, gap interpolation
  - phase: 80-baseline-tracking-measurement
    provides: Phase 80 OC-SORT baseline (27 tracks, 93.1% coverage) for comparison
provides:
  - TrackingConfig extended with base_r, lambda_ocm, max_gap_frames; default is keypoint_bidi
  - TrackingStage dispatches to KeypointTracker or OcSortTracker based on tracker_kind
  - KeypointTracker exported in tracking package __all__
  - Evaluation script running both trackers on same clip with side-by-side metrics
  - 84-EVALUATION.md: quantitative comparison with interpretation
affects: [85-e2e-pipeline-validation, tracker-tuning]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Dual-backend tracker dispatch via tracker_kind config field (ocsort | keypoint_bidi)
    - Single-pass detection+pose caching before feeding to multiple trackers

key-files:
  created:
    - scripts/evaluate_custom_tracker.py
    - .planning/phases/84-integration-evaluation/84-EVALUATION.md
    - eval_tracker_output/comparison_metrics.json
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/core/tracking/stage.py
    - src/aquapose/core/tracking/__init__.py

key-decisions:
  - "keypoint_bidi produces 44 tracks vs OC-SORT 30 vs target 9 — both over-fragment at occlusion; custom tracker does NOT improve fragmentation under default params"
  - "BYTE-style trigger fired (coverage 0.898 < 0.90); TRACK-10 deferred because root cause is occlusion reacquisition (identity break), not low-confidence detection misses"
  - "TrackingConfig.tracker_kind default remains keypoint_bidi per phase plan; OC-SORT available as ocsort fallback"
  - "Evaluation confirms pipeline wiring is correct: both trackers run end-to-end on the same clip without errors"

patterns-established:
  - "Evaluation pattern: cache all Detection objects (OBB + pose) in one pass, replay to multiple trackers for fair comparison"

requirements-completed: [INTEG-01, INTEG-02]

# Metrics
duration: 45min
completed: 2026-03-11
---

# Phase 84 Plan 01: Integration and Evaluation Summary

**KeypointTracker wired into TrackingStage (keypoint_bidi default), evaluated against OC-SORT on Phase 80 clip — both trackers over-fragment at occlusion; BYTE-style secondary pass recommended (coverage 0.898)**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-03-11T03:30:00Z
- **Completed:** 2026-03-11T04:15:00Z
- **Tasks:** 2 auto (+ 1 checkpoint pending human review)
- **Files modified:** 6

## Accomplishments

- TrackingStage now dispatches to KeypointTracker when tracker_kind="keypoint_bidi" (default) and OcSortTracker when tracker_kind="ocsort" — full backward compatibility maintained
- TrackingConfig extended with base_r, lambda_ocm, max_gap_frames; default tracker_kind switched to keypoint_bidi
- Evaluation script runs OBB detection + pose estimation once, caches detections, feeds to both trackers — fair apples-to-apples comparison
- Results: keypoint_bidi produces 44 tracks (vs OC-SORT 30 vs target 9); both over-fragment at occlusion; coverage 0.898 triggers BYTE recommendation

## Task Commits

1. **Task 1: Pipeline wiring** — `d131a2e` (feat) [committed in Phase 83-02 + 84-01 default-change commit]
2. **Task 2: Evaluation script and metrics comparison** — `0d5cde1` (feat)

## Files Created/Modified

- `src/aquapose/engine/config.py` — TrackingConfig extended with base_r, lambda_ocm, max_gap_frames; tracker_kind default = keypoint_bidi
- `src/aquapose/core/tracking/stage.py` — TrackingStage.run() dispatches by tracker_kind to KeypointTracker or OcSortTracker
- `src/aquapose/core/tracking/__init__.py` — KeypointTracker added to exports and __all__
- `scripts/evaluate_custom_tracker.py` — 847-line standalone evaluation script
- `.planning/phases/84-integration-evaluation/84-EVALUATION.md` — side-by-side metrics comparison with interpretation
- `eval_tracker_output/comparison_metrics.json` — raw metrics JSON from evaluation run

## Decisions Made

- **keypoint_bidi fragmentation**: Custom tracker produces 44 tracks vs OC-SORT 30 — worse, not better. Root cause is stricter OKS matching rejecting noisy occlusion reacquisitions that OC-SORT's IoU matching accepts.
- **BYTE-style secondary pass (TRACK-10)**: Coverage trigger fired (0.898). Deferred because the root issue is identity-breaking at occlusion (a different problem from low-confidence detection misses). A BYTE pass would not fix 44-track fragmentation.
- **Tracker default**: Kept as keypoint_bidi per the phase plan. Users can revert to ocsort via `tracking.tracker_kind: ocsort` in config YAML.
- **Parameter tuning deferred**: Results are informative as-is. Tuning directions documented in 84-EVALUATION.md for future work.

## Deviations from Plan

**None** — plan executed exactly as specified, including the BYTE coverage trigger evaluation.

## Issues Encountered

None. The evaluation script ran cleanly on first attempt. All 1200 frames processed with detection + pose estimation. Both tracker backends produced valid metrics.

## Next Phase Readiness

- Pipeline wiring complete — both tracker backends operational
- Evaluation data available at `eval_tracker_output/comparison_metrics.json`
- Annotated video at `eval_tracker_output/keypoint_bidi_tracking.mp4`
- Task 3 (checkpoint:human-verify) pending user review of 84-EVALUATION.md and annotated video
- Phase 85 (E2E pipeline validation) can proceed after review, with the option to tune keypoint_bidi params or use ocsort as default

---
*Phase: 84-integration-evaluation*
*Completed: 2026-03-11*
