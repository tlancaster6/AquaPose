---
phase: 05-cross-view-identity-and-3d-tracking
plan: 03
subsystem: tracking
tags: [hdf5, h5py, serialization, chunked-io, round-trip]
dependency_graph:
  requires:
    - aquapose.tracking.tracker (FishTrack, FishTracker from Plan 02)
    - aquapose.tracking.associate (AssociationResult, FrameAssociations from Plan 01)
    - h5py
  provides:
    - aquapose.tracking.TrackingWriter
    - aquapose.tracking.read_tracking_results
  affects:
    - Phase 06 (medial axis) — consumes HDF5 output for per-fish camera sets and bboxes
tech_stack:
  added: []
  patterns:
    - Buffer-and-flush pattern buffers chunk_frames rows in numpy arrays before writing to HDF5
    - cast(h5py.Dataset, ...) for basedpyright narrowing of h5py subscript return types
    - Context manager protocol (__enter__/__exit__) wrapping close() for safe cleanup
key_files:
  created:
    - src/aquapose/tracking/writer.py
    - tests/unit/tracking/test_writer.py
  modified:
    - src/aquapose/tracking/__init__.py
key_decisions:
  - "cast(h5py.Dataset, grp[name]) used throughout _flush() and read_tracking_results() — basedpyright cannot narrow h5py subscript return from Group|Dataset|Datatype to Dataset"
  - "cast(list[str], grp.attrs[...]) used for camera_names attribute — h5py attr return includes Empty which is not iterable according to basedpyright"
  - "dtype='bool' string used for is_confirmed dataset — type[bool] is not assignable to dtype: str parameter"
patterns_established:
  - "Buffer-and-flush: pre-allocate numpy arrays to chunk_frames; fill row-by-row; resize HDF5 on flush"
  - "Slots sorted by fish_id for deterministic slot ordering across frames"
  - "Fill-values: fish_id=-1, centroid_3d=NaN, confidence=-1.0, n_cameras=0, is_confirmed=False, cam_assignments=-1, bboxes=-1"
requirements_completed:
  - TRACK-01
  - TRACK-02
  - TRACK-03
  - TRACK-04
duration: 5min
completed: 2026-02-21
---

# Phase 05 Plan 03: HDF5 Tracking Writer Summary

Chunked HDF5 serialization for tracking results with buffer-and-flush pattern, per-camera bbox sub-groups, and round-trip read function for Phase 6 consumption.

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-21T20:59:45Z
- **Completed:** 2026-02-21T21:04:30Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `TrackingWriter` creates a structured HDF5 file under `/tracking/` with 7 main datasets plus per-camera `camera_assignments` and `bboxes` sub-groups
- Buffer-and-flush pattern avoids per-frame I/O overhead; `chunk_frames=1000` default batches a 1000-frame window before flushing
- `read_tracking_results()` reads all datasets back into numpy arrays as a convenience function for Phase 6 and round-trip verification
- 8 unit tests covering round-trip, chunked flush timing, context manager, camera assignments/bboxes, fill-values, camera_names attribute, empty file, and tracker integration
- Complete tracking module API: all 7 public symbols importable from `aquapose.tracking`

## Task Commits

Each task was committed atomically:

1. **Task 1: HDF5 tracking writer with chunked datasets** - `ae33c38` (feat)
2. **Task 2: Unit tests for HDF5 writer round-trip and integration** - `57888c4` (test)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `src/aquapose/tracking/writer.py` - TrackingWriter class and read_tracking_results function
- `src/aquapose/tracking/__init__.py` - Added TrackingWriter, read_tracking_results to exports
- `tests/unit/tracking/test_writer.py` - 8 unit tests for HDF5 write/read round-trips

## Decisions Made

- `cast(h5py.Dataset, ...)` used throughout — basedpyright cannot narrow h5py subscript return type from `Group | Dataset | Datatype` to `Dataset`, requiring explicit casts
- `cast(list[str], grp.attrs["camera_names"])` — h5py attribute access returns a type including `Empty` which is not iterable; cast required for basedpyright narrowing
- `dtype="bool"` (string) not `bool` (type) for is_confirmed dataset — basedpyright requires str for h5py dtype parameter
- Slots sorted by `fish_id` for deterministic ordering across frames — reader can reliably map slot index to fish identity

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Three basedpyright type errors in writer.py**
- **Found during:** Task 1 (typecheck)
- **Issue:** (a) `type[bool]` not assignable to `dtype: str`; (b) `Group | Dataset | Datatype` not subscriptable as Dataset in _flush(); (c) h5py attr `"camera_names"` returns `Empty | ndarray | ...` not iterable
- **Fix:** Changed `bool` dtype to `"bool"` string; added `cast(h5py.Dataset, ...)` in `_flush()` and `read_tracking_results()`; used `cast(list[str], grp.attrs["camera_names"])` then `list(...)`
- **Files modified:** `src/aquapose/tracking/writer.py`
- **Verification:** `hatch run check` passes with 0 errors in tracking/
- **Committed in:** `ae33c38` (Task 1 commit)

**2. [Rule 1 - Lint] Unsorted import block in test_writer.py**
- **Found during:** Task 2 pre-commit hook
- **Issue:** Ruff I001 — import block unsorted
- **Fix:** `ruff check --fix` auto-sorted imports
- **Files modified:** `tests/unit/tracking/test_writer.py`
- **Verification:** `hatch run check` passes
- **Committed in:** `57888c4` (Task 2 commit, ruff reformatted before commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** All auto-fixes necessary for type safety and lint compliance. No scope creep.

## Issues Encountered

None beyond the auto-fixed type issues above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Complete tracking module API ready: `from aquapose.tracking import FishTracker, TrackingWriter, ransac_centroid_cluster`
- Phase 6 (medial axis) can consume HDF5 output by calling `read_tracking_results(path)` to get per-fish camera sets and bboxes
- HDF5 schema documented in `TrackingWriter` class docstring

---
*Phase: 05-cross-view-identity-and-3d-tracking*
*Completed: 2026-02-21*
