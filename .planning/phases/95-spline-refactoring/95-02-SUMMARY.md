---
phase: 95-spline-refactoring
plan: "02"
subsystem: reconstruction
tags: [refactor, io, hdf5, evaluation, spline, raw-keypoint]

dependency_graph:
  requires:
    - phase: 95-01
      provides: Midline3D with optional points/control_points fields, spline_enabled config toggle
  provides:
    - HDF5 writer serializing raw-keypoint Midline3D to points dataset
    - Evaluation compute_per_point_error() handling both spline and raw-keypoint modes
    - Full end-to-end pipeline: spline_enabled flows from config through all stages
  affects: [reconstruction pipeline, io writer, evaluation metrics, downstream HDF5 readers]

tech-stack:
  added: []
  patterns: [NaN-fill dual-dataset HDF5 layout, backward-compat optional dataset reader, spline-vs-raw branch in evaluation]

key-files:
  created: []
  modified:
    - src/aquapose/io/midline_writer.py
    - src/aquapose/evaluation/stages/reconstruction.py

key-decisions:
  - "Both points and control_points datasets always present in HDF5; unused one filled with NaN for backward compat"
  - "read_midline3d_results() returns points=None for legacy HDF5 files without the dataset"
  - "compute_per_point_error() skips raw-keypoint midlines whose point count does not match n_body_points"
  - "pipeline.py spline_enabled wiring was already complete from 95-01 — no changes needed"

patterns-established:
  - "Dual-dataset NaN-fill pattern: create both datasets, write whichever is populated, leave the other as NaN"
  - "Backward-compat optional dataset reader: check if key in grp before reading, return None if absent"

requirements-completed: [SPL-01, SPL-03]

duration: ~8min
completed: 2026-03-13
---

# Phase 95 Plan 02: Spline Refactoring — Downstream Consumers Summary

**HDF5 writer now serializes raw-keypoint Midline3D to a new points dataset; evaluation per-point error branches on spline vs raw-keypoint mode; full pipeline wired end-to-end.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-13T22:10:06Z
- **Completed:** 2026-03-13T22:18:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `Midline3DWriter` now writes raw 3D keypoints to a `points` (N, max_fish, n_sample_points, 3) HDF5 dataset. Both `points` and `control_points` datasets are always present; the one not in use is filled with NaN. Reader returns `points=None` for legacy files.
- `compute_per_point_error()` in evaluation now branches: spline mode evaluates B-spline at uniform parameters; raw-keypoint mode uses `midline.points` directly when count matches.
- `pipeline.py` was already wired from 95-01 — `spline_enabled=config.reconstruction.spline_enabled` was confirmed complete.

## Task Commits

Each task was committed atomically:

1. **Task 1: Update Midline3DWriter for raw-keypoint mode** - `0c3dcae` (feat)
2. **Task 2: Wire spline_enabled through pipeline.py and update evaluation** - `e94ffde` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/aquapose/io/midline_writer.py` - Added points dataset, buffer, flush, and reader support
- `src/aquapose/evaluation/stages/reconstruction.py` - Added raw-keypoint branch in compute_per_point_error

## Decisions Made

- Both `points` and `control_points` HDF5 datasets are always created and present in every output file, even when only one is populated. This avoids conditional dataset existence checks in downstream readers and preserves backward compatibility.
- The reader returns `points=None` for legacy HDF5 files that predate this change, matching the existing pattern for `centroid_z` and `z_offsets`.
- `compute_per_point_error()` skips raw-keypoint midlines whose `len(m3d.points) != n_body_points` rather than resampling, to avoid introducing interpolation artifacts in per-point error computation.

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written. The pipeline.py wiring noted as a task was already complete from 95-01 (confirmed by reading the file); only the evaluation branch and writer updates were needed.

## Issues Encountered

- The plan's `type="checkpoint"` verification script used `tempfile.NamedTemporaryFile` which deletes the file before the read call completes on Linux. Rewrote to use a fixed `/tmp` path for verification. Functionality is correct.

## Next Phase Readiness

- Full spline refactoring complete: raw-keypoint mode works end-to-end from config through reconstruction, HDF5 serialization, and evaluation.
- Phase 96 (next in v3.9 milestone) can proceed.
- Existing HDF5 files written before this change will read `points=None` (backward compatible).

## Self-Check: PASSED

All key files exist. Both commits (0c3dcae, e94ffde) found. 1203 tests pass. 0 typecheck errors.

---
*Phase: 95-spline-refactoring*
*Completed: 2026-03-13*
