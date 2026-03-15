---
phase: 99-reconstruction-quality-metrics
plan: 01
subsystem: evaluation
tags: [reconstruction, reprojection-error, camera-visibility, evaluation, metrics]

# Dependency graph
requires:
  - phase: 97-full-pipeline-run
    provides: "32-chunk diagnostic run caches (run_20260314_200051) used as eval input"
provides:
  - "p99 reprojection error percentile in ReconstructionMetrics"
  - "camera visibility statistics in ReconstructionMetrics"
  - "Reconstruction Quality section in performance-accuracy.md with full-run numbers"
  - "reconstruction_quality_full_run.csv with all raw metrics"
affects:
  - 98-performance-metrics
  - publication

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "camera_visibility dict holds mean/median/min/max/distribution for cameras-per-fish"
    - "p99 computed alongside p50/p90/p95 in a single np.percentile call"

key-files:
  created:
    - .planning/results/data/reconstruction_quality_full_run.csv
    - .planning/results/performance-accuracy.md
  modified:
    - src/aquapose/evaluation/stages/reconstruction.py
    - src/aquapose/evaluation/output.py

key-decisions:
  - "camera_visibility stored as dict[str, float|int] with nested 'distribution' sub-dict for JSON compatibility"
  - "n_cameras field on Midline3D is the source for camera visibility (minimum cameras across body points)"
  - "All metrics derived directly from aquapose eval on run_20260314_200051; no estimates"

patterns-established:
  - "ReconstructionMetrics new optional fields use None defaults for backward compatibility"
  - "format_eval_report() camera visibility section placed between per_camera_error and per_point_error"

requirements-completed: [RECON-01, RECON-02, RECON-03]

# Metrics
duration: 35min
completed: 2026-03-15
---

# Phase 99 Plan 01: Reconstruction Quality Metrics Summary

**p99 reproj error (14.41px) and camera visibility (mean 3.60 cameras/fish) added to evaluator and recorded from 9,450-frame full run**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-03-15T08:18:00Z
- **Completed:** 2026-03-15T08:53:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added `p99_reprojection_error` and `camera_visibility` fields to `ReconstructionMetrics` dataclass with full serialization
- Computed both metrics inside `evaluate_reconstruction()` from `frame_results` — no runner changes needed
- Updated `format_eval_report()` to display p99 and a full camera visibility section with distribution
- Ran full eval on Phase 97 run (9,450 frames, 12 cameras, 9 fish, 32 chunks): mean reproj error 3.41px, p99 14.41px
- Recorded per-keypoint breakdown (head 5.35px mean vs mid-body 2.73–3.63px) and camera distribution (63% of fish-frames at 4 cameras)
- Created `reconstruction_quality_full_run.csv` and Section 9 in `performance-accuracy.md`

## Task Commits

Each task was committed atomically:

1. **Task 1: Add p99 percentile and camera visibility stats to reconstruction evaluator** - `64887db` (feat)
2. **Task 2: Run full evaluation and record reconstruction quality metrics** - `6df506c` (feat)

## Files Created/Modified

- `src/aquapose/evaluation/stages/reconstruction.py` - Added p99_reprojection_error, camera_visibility fields; updated evaluate_reconstruction() and to_dict()
- `src/aquapose/evaluation/output.py` - Added p99 and camera visibility display in format_eval_report()
- `.planning/results/performance-accuracy.md` - New Section 9 with full-run reconstruction quality metrics
- `.planning/results/data/reconstruction_quality_full_run.csv` - Raw metrics CSV (32 rows)

## Decisions Made

- `camera_visibility` stored as `dict[str, float | int]` with a nested `"distribution"` key containing the count-per-n_cameras map. This keeps the top-level dict JSON-serializable as-is (distribution is already a plain dict).
- Source for camera count: `Midline3D.n_cameras` (minimum cameras across body points), already set by the reconstruction backend.
- All reported numbers derived from `aquapose eval` on run_20260314_200051 — no fabrication.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The eval CLI uses positional `[RUN]` argument, not `--run-dir` flag as shown in the plan's example. Discovered from `-h` output; adjusted command accordingly. No code fix needed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- RECON-01, RECON-02, RECON-03 requirements satisfied with real numbers from the full 9,450-frame run
- performance-accuracy.md now has complete reconstruction quality section ready for publication
- Camera visibility data reveals 63% of fish-frames observed by 4 cameras, 28% by 3; head/tail keypoints have higher reproj error than mid-body

---
*Phase: 99-reconstruction-quality-metrics*
*Completed: 2026-03-15*

## Self-Check: PASSED

All key files verified present and both task commits found in git history.
