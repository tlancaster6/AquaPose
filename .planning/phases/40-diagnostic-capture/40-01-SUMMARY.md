---
phase: 40-diagnostic-capture
plan: "01"
subsystem: io
tags: [numpy, npz, serialization, midline, diagnostic, observer]

# Dependency graph
requires: []
provides:
  - MidlineFixture frozen dataclass defining the per-frame MidlineSet data contract
  - NPZ_VERSION constant and flat key convention for midline fixture files
  - DiagnosticObserver.export_midline_fixtures() assembling MidlineSets from snapshot data
  - DiagnosticObserver._match_annotated_by_centroid() centroid-proximity matching helper
  - Auto-export of midline_fixtures.npz in _on_pipeline_complete when output_dir is set
affects: [41-eval-harness, 42-reconstruction]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Flat NPZ key convention: meta/ for metadata, midline/{frame}/{fish}/{cam}/{field} for midline data"
    - "TDD: failing tests written before implementation for each task"
    - "Centroid-proximity matching (5px tolerance) to link Tracklet2D centroids to AnnotatedDetections"
    - "DiagnosticObserver follows existing export pattern: method + auto-export in _on_pipeline_complete"

key-files:
  created:
    - src/aquapose/io/midline_fixture.py
  modified:
    - src/aquapose/engine/diagnostic_observer.py
    - src/aquapose/io/__init__.py
    - tests/unit/engine/test_diagnostic_observer.py

key-decisions:
  - "NPZ key convention uses flat slash-separated keys (not nested groups) for numpy.load compatibility"
  - "No min_cameras filter on frame inclusion - all frames with at least one midline are captured"
  - "MidlineFixture is a frozen dataclass (data contract only, no loader logic) - loader deferred to Plan 02"
  - "point_confidence defaults to uniform 1.0 float32 array when None (preserves key shape invariant)"
  - "centroid matching tolerance set to 5px Euclidean distance (mirrors ReconstructionStage logic)"

patterns-established:
  - "NPZ fixture pattern: meta/ keys + data/{dim1}/{dim2}/{dim3}/{field} hierarchy for multi-dimensional indexed data"
  - "Layer discipline: centroid matching reimplemented in engine/ rather than importing from core/reconstruction/"

requirements-completed: [DIAG-01]

# Metrics
duration: 6min
completed: 2026-03-02
---

# Phase 40 Plan 01: Diagnostic Capture - Midline Fixture Serialization Summary

**MidlineFixture frozen dataclass + DiagnosticObserver NPZ export assembling per-frame MidlineSets via centroid-proximity matching from Association and Midline stage snapshots**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-02T19:41:39Z
- **Completed:** 2026-03-02T19:47:47Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created `MidlineFixture` frozen dataclass in `aquapose.io.midline_fixture` with `frames`, `frame_indices`, `camera_ids`, and `metadata` fields, plus `NPZ_VERSION = "1.0"` and documented flat NPZ key convention
- Implemented `DiagnosticObserver.export_midline_fixtures()` that assembles `MidlineSet` data from captured snapshots using centroid-proximity matching (5px tolerance) and writes compressed NPZ files
- Extended `_on_pipeline_complete()` to auto-export `midline_fixtures.npz` alongside `centroid_correspondences.npz` when `output_dir` is configured
- Added 6 new tests (TDD style: RED then GREEN) covering importability, frozen dataclass enforcement, NPZ key content, error handling, no-min-cameras inclusion, and PipelineComplete auto-export

## Task Commits

Each task was committed atomically:

1. **Task 1: Create MidlineFixture dataclass and NPZ key convention** - `10d7b5f` (feat)
2. **Task 2: Add MidlineSet assembly and NPZ export to DiagnosticObserver** - `e61e002` (feat)

_Note: TDD tasks have test + implementation in a single commit per task._

## Files Created/Modified

- `src/aquapose/io/midline_fixture.py` - MidlineFixture frozen dataclass and NPZ_VERSION constant with full key convention documentation
- `src/aquapose/engine/diagnostic_observer.py` - Added export_midline_fixtures(), _match_annotated_by_centroid(), extended _on_pipeline_complete()
- `src/aquapose/io/__init__.py` - Added MidlineFixture and NPZ_VERSION to public API
- `tests/unit/engine/test_diagnostic_observer.py` - Added 6 new tests plus helper functions _make_midline2d(), _make_annotated_detection(), _fire_midline_stage()

## Decisions Made

- NPZ key convention uses flat slash-separated keys rather than nested HDF5-style groups, since `numpy.load` returns flat keys and the loader (Plan 02) must parse them by splitting on `/`
- No `min_cameras` filter on frame inclusion: all frames with at least one midline are captured, giving Plan 41's eval harness maximum data
- `MidlineFixture` is a pure data contract (no serialization/deserialization logic) — Plan 02 adds the loader
- `point_confidence` serialized as uniform 1.0 float32 when the field is `None`, preserving the NPZ key shape invariant for the loader
- Centroid matching tolerance of 5px matches the logic in `ReconstructionStage` but is reimplemented in `engine/` to preserve the layer boundary (no import from `core/reconstruction/`)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Pre-commit ruff lint flagged use of Unicode multiplication sign `×` in docstring (RUF002) — replaced with ASCII `x`
- Pre-commit ruff lint flagged `pytest.raises(Exception)` as too broad (B017/PT011) — replaced with `dataclasses.FrozenInstanceError`
- Pre-commit ruff lint flagged `zip()` without `strict=` (B905) — added `strict=False`
- All lint issues resolved before commit; no scope impact

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `midline_fixtures.npz` serialization side of DIAG-01 is complete
- Plan 02 can now implement the loader (`MidlineFixtureLoader`) that reads the NPZ and reconstructs `MidlineFixture`
- The NPZ key convention is documented in `midline_fixture.py` module docstring and enforced by tests

---
*Phase: 40-diagnostic-capture*
*Completed: 2026-03-02*
