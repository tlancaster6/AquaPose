---
phase: 80-baseline-metrics
plan: 01
subsystem: evaluation
tags: [ocsort, tracking, fragmentation, metrics, baseline, evaluation]

# Dependency graph
requires:
  - phase: 78.1-obb-pose-production-retrain
    provides: Production OBB and Pose models deployed to config.yaml
provides:
  - evaluate_fragmentation_2d() public function in fragmentation.py for 2D tracklet gap analysis
  - measure_baseline_tracking.py standalone script for OC-SORT baseline measurement
  - 80-BASELINE.md with quantitative OC-SORT baseline (27 tracks vs 9 target)
affects: [84-custom-tracker, 85-boxmot-removal]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-pass detection+tracking+video rendering loop to avoid memory duplication"
    - "evaluate_fragmentation_2d() adapts 3D fragmentation logic for 2D Tracklet2D objects"

key-files:
  created:
    - src/aquapose/evaluation/stages/fragmentation.py (evaluate_fragmentation_2d added)
    - tests/unit/evaluation/test_stage_fragmentation.py
    - scripts/measure_baseline_tracking.py
    - .planning/phases/80-baseline-metrics/80-BASELINE.md
    - baseline_tracking_output/baseline_metrics.json
  modified:
    - src/aquapose/evaluation/stages/__init__.py (export evaluate_fragmentation_2d)
    - .gitignore (added *.mp4)

key-decisions:
  - "OC-SORT baseline: 27 tracks vs 9 target (3x over-fragmented) — primary failure mode for Phase 84 to address"
  - "min_hits=1 used for honest baseline — no warm-up penalty that would artificially inflate track count"
  - "Single-pass architecture chosen over 2-pass to avoid holding all frames in memory"

patterns-established:
  - "evaluate_fragmentation_2d: builds fish_frames dict from Tracklet2D.frames tuples, then reuses same gap/continuity math as 3D version"

requirements-completed: [INV-03]

# Metrics
duration: ~30min
completed: 2026-03-10
---

# Phase 80 Plan 01: Baseline Metrics Summary

**OC-SORT baseline established: 27 tracks vs 9-fish target (3x over-fragmented), 93.1% detection coverage, zero within-track gaps — fragmentation script and annotated video produced**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-03-10
- **Completed:** 2026-03-10
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added `evaluate_fragmentation_2d()` as tested public function in fragmentation.py, adapting 3D fragmentation logic to work with 2D `Tracklet2D` objects
- Built `scripts/measure_baseline_tracking.py` — standalone OC-SORT pipeline with single-pass detection+tracking+video rendering, polygon NMS, and JSON metrics output
- Ran script on e3v83eb frames 3300-4500 and recorded baseline in 80-BASELINE.md: 27 tracks, 93.1% coverage, 0 gaps, births=18/deaths=17

## Task Commits

Each task was committed atomically:

1. **Task 1: Add evaluate_fragmentation_2d() and test it** - `88b7173` (feat + test, TDD)
2. **Task 2: Build measure_baseline_tracking.py script** - `5a6e75f` (feat)
3. **Task 3: Run script, verify output, record baseline in 80-BASELINE.md** - `a891a1d` (feat)

## Files Created/Modified

- `src/aquapose/evaluation/stages/fragmentation.py` - Added `evaluate_fragmentation_2d()` for Tracklet2D gap analysis
- `src/aquapose/evaluation/stages/__init__.py` - Exported new function
- `tests/unit/evaluation/test_stage_fragmentation.py` - TDD tests for evaluate_fragmentation_2d
- `scripts/measure_baseline_tracking.py` - Standalone OC-SORT baseline measurement script
- `.planning/phases/80-baseline-metrics/80-BASELINE.md` - Structured baseline metrics with gap-to-target analysis
- `baseline_tracking_output/baseline_metrics.json` - Raw metrics JSON from the baseline run
- `.gitignore` - Added `*.mp4` to prevent accidental tracking of large video files

## Decisions Made

- `min_hits=1` used for the baseline (no warm-up penalty) — gives an honest view of OC-SORT's identity-breaking behavior without artificially inflating track count via discarded early detections
- Single-pass architecture in the script: detection+tracking+video rendering in one loop, avoids duplicating ~1200 frame reads and eliminates per-frame storage of detection/track data
- `evaluate_fragmentation_2d` builds `fish_frames` directly from `Tracklet2D.frames` tuples rather than calling the existing 3D `evaluate_fragmentation` (which expects `Midline3D.frame_index`) — avoids type mismatch, keeps logic clean

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Merged 2-pass video rendering into single-pass**
- **Found during:** Task 3 (script execution)
- **Issue:** Task 2 commit implemented a 2-pass approach (detect+track first pass, re-read video for annotation second pass), which consumed ~2x memory and required storing all per-frame detection data in a list
- **Fix:** Merged into single-pass: render each frame to VideoWriter immediately after detection+tracking in the same loop
- **Files modified:** scripts/measure_baseline_tracking.py
- **Verification:** Script ran successfully, produced baseline_metrics.json and annotated video
- **Committed in:** a891a1d (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug/inefficiency)
**Impact on plan:** Fix reduced memory footprint and eliminated need to store all detections in RAM. No scope creep.

## Issues Encountered

None beyond the single-pass merge deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 80-BASELINE.md provides the quantitative target for Phase 84: reduce track count from 27 to <=9, maintain zero gaps and >=93.1% detection coverage
- `evaluate_fragmentation_2d` is ready for use in Phase 84 tracker evaluation
- `measure_baseline_tracking.py` can be adapted as a template for Phase 84 evaluation scripts

---
*Phase: 80-baseline-metrics*
*Completed: 2026-03-10*
