---
phase: 97-full-pipeline-run
plan: 01
subsystem: pipeline
tags: [aquapose, pipeline, diagnostic, hdf5, chunked-processing]

requires:
  - phase: 96-z-denoising-and-documentation
    provides: v3.9 reconstruction modernization (raw keypoints, spline toggle, z-denoising)
provides:
  - Complete 32-chunk diagnostic run with per-chunk caches for metric extraction
  - HDF5 output with 9,450 frames x 9 fish (points + control_points)
  - Per-chunk timing data for all 5 stages
affects: [98-performance-metrics, 99-reconstruction-quality-metrics, 100-tracking-and-association-metrics, 101-results-document]

tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified: []

key-decisions:
  - "Split into two runs: full 32-chunk for evaluation data, separate 6-chunk for clean-machine timing"
  - "Full run timing not suitable for publication — dedicated timing run on idle machine needed"
  - "Used nohup for 3+ hour run instead of Claude Code background tasks"

patterns-established: []

requirements-completed: [RUN-01]

duration: ~3h (overnight nohup run)
completed: 2026-03-15
---

# Phase 97: Full Pipeline Run Summary

**32-chunk diagnostic pipeline run completed over full 5-minute (9,450-frame) recording with production models, producing per-chunk caches and 6.2MB HDF5 output**

## Performance

- **Duration:** ~3 hours (nohup overnight run)
- **Started:** 2026-03-14 ~20:00
- **Completed:** 2026-03-15 ~00:00 (estimated)
- **Tasks:** 2
- **Files modified:** 0 (execution-only phase)

## Accomplishments
- Full 9,450-frame diagnostic pipeline run completed without crashes across all 32 chunks
- Per-chunk cache.pkl files written for all 32 chunks in diagnostics/ directory
- HDF5 output: 9,450 frames x 9 fish with points (6x3), control_points (7x3), plus metadata (residuals, confidence, camera counts, z-offsets)
- Per-chunk timing reports for all 5 stages (detection, pose, tracking, association, reconstruction)
- Production models confirmed: OBB (run_20260311_174552), Pose (run_20260310_171543)

## Run Details

- **Run directory:** `~/aquapose/projects/YH/runs/run_20260314_200051/`
- **Chunk size:** 300 frames
- **Chunks:** 32 (31 full + 1 partial with 150 frames)
- **Per-chunk wall-time:** 221-334s (mean ~265s), first chunk slower due to warmup
- **Total wall-time:** ~2h 21m compute (plus overhead)

## Task Commits

No code changes — execution-only phase.

1. **Task 1: Verify config and launch pipeline** - config verified, run launched via nohup
2. **Task 2: Verify pipeline completion and outputs** - all success criteria confirmed

## Files Created/Modified

No source files modified. Run artifacts:
- `~/aquapose/projects/YH/runs/run_20260314_200051/midlines.h5` - 3D reconstruction output
- `~/aquapose/projects/YH/runs/run_20260314_200051/diagnostics/chunk_*/cache.pkl` - 32 per-chunk caches
- `~/aquapose/projects/YH/runs/run_20260314_200051/timing.txt` - per-chunk stage timing
- `~/aquapose/projects/YH/runs/run_20260314_200051/config.yaml` - frozen config snapshot

## Decisions Made

- Split into two runs: full 32-chunk run (this phase) for evaluation data, separate 6-chunk timing run for publication-quality per-stage breakdown. Reason: full run executed alongside other workloads (~5.5 min/chunk overhead), timing needs clean-machine conditions.
- Timing run feeds Phase 98, not this phase. This phase's success criteria are about data availability.

## Deviations from Plan

None - plan executed as written.

## Issues Encountered

None - pipeline completed all 32 chunks without errors.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 32 chunk caches available for metric extraction (Phases 99-100)
- 6-chunk timing run in progress on clean machine (Phase 98)
- Phases 98-101 can proceed: timing extraction, reconstruction quality, tracking/association metrics, results document

---
*Phase: 97-full-pipeline-run*
*Completed: 2026-03-15*
