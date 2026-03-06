---
phase: 08-end-to-end-integration-testing-and-benchmarking
plan: 01
subsystem: pipeline
tags: [pipeline, orchestrator, hdf5, io, stages, testing]
dependency_graph:
  requires:
    - src/aquapose/tracking/tracker.py
    - src/aquapose/reconstruction/midline.py
    - src/aquapose/reconstruction/triangulation.py
    - src/aquapose/segmentation/detector.py
    - src/aquapose/segmentation/model.py
    - src/aquapose/calibration/loader.py
    - src/aquapose/calibration/projection.py
  provides:
    - src/aquapose/pipeline/orchestrator.py
    - src/aquapose/pipeline/stages.py
    - src/aquapose/io/midline_writer.py
  affects:
    - src/aquapose/io/__init__.py
tech_stack:
  added: []
  patterns:
    - Chunked HDF5 resizable datasets (same pattern as TrackingWriter)
    - Internal lazy imports inside stage functions to defer heavy dependencies
    - TYPE_CHECKING guard for type-only imports to avoid circular deps
    - FishTracker and MidlineExtractor instantiated once at top of reconstruct()
key_files:
  created:
    - src/aquapose/pipeline/__init__.py
    - src/aquapose/pipeline/orchestrator.py
    - src/aquapose/pipeline/stages.py
    - src/aquapose/io/midline_writer.py
    - tests/unit/io/__init__.py
    - tests/unit/io/test_midline_writer.py
    - tests/unit/pipeline/__init__.py
    - tests/unit/pipeline/test_stages.py
  modified:
    - src/aquapose/io/__init__.py
decisions:
  - "FishTracker and MidlineExtractor are created once in reconstruct() and passed to stage functions — stage functions MUST NOT create new instances internally"
  - "SPLINE_KNOTS and SPLINE_K stored as HDF5 group attributes (not datasets) — consistent with OUT-01 spec and Midline3D struct semantics"
  - "run_segmentation re-reads video from disk instead of caching frames — avoids OOM on hours-long videos with 13 cameras"
  - "diagnostic mode saves detection_counts as .npz (not raw frames) to keep diagnostic overhead low"
  - "test_run_triangulation patches aquapose.reconstruction.triangulation.triangulate_midlines because stages.py uses a deferred local import"
metrics:
  duration: 10 min
  completed: 2026-02-21
  tasks_completed: 2
  files_created: 9
---

# Phase 8 Plan 01: Pipeline Orchestrator and HDF5 Midline Writer Summary

One-liner: `reconstruct()` entry point chains 5 stages with timing, writes HDF5 Midline3D output following the TrackingWriter chunked-append pattern.

## What Was Built

**`src/aquapose/pipeline/`** — new package with three files:

- **`stages.py`**: Five independently callable stage functions (`run_detection`, `run_segmentation`, `run_tracking`, `run_midline_extraction`, `run_triangulation`) each with typed signatures and INFO-level timing logs.
- **`orchestrator.py`**: `reconstruct()` callable API that discovers camera videos, loads calibration, creates stateful objects once, chains all 5 stages with `time.perf_counter()` timing, writes HDF5 output, and returns `ReconstructResult`. Supports `mode="production"|"diagnostic"` and `stop_frame` for short runs.
- **`__init__.py`**: Exports `reconstruct`, `ReconstructResult`.

**`src/aquapose/io/midline_writer.py`** — `Midline3DWriter` class:

- Chunked HDF5 resizable datasets under `/midlines/`: `frame_index`, `fish_id`, `control_points (N, max_fish, 7, 3)`, `arc_length`, `half_widths (N, max_fish, 15)`, `n_cameras`, `mean_residual`, `max_residual`, `is_low_confidence`.
- `SPLINE_KNOTS` and `SPLINE_K` stored as group attributes per OUT-01 spec.
- Context manager protocol (`__enter__`/`__exit__`).
- `read_midline3d_results()` reader for downstream consumers.

**Tests** — 9 tests total, all fast and GPU-free:
- 6 midline writer tests (round-trip, chunk flush, context manager, fill-values, attributes, empty frame)
- 3 stage function tests (run_triangulation with mock, run_tracking stateful preservation x2)

## Verification

- `hatch run check` passes (lint + typecheck; 4 pre-existing errors in detector.py only)
- `hatch run test tests/unit/io/test_midline_writer.py tests/unit/pipeline/test_stages.py` — 9/9 pass
- `python -c "from aquapose.pipeline import reconstruct, ReconstructResult; from aquapose.io import Midline3DWriter, read_midline3d_results"` — succeeds

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1: Orchestrator, stages, writer | f85f02b | feat(08-01): pipeline orchestrator, stage functions, and HDF5 midline writer |
| Task 2: Unit tests | d0f4ff7 | test(08-01): unit tests for Midline3DWriter and pipeline stage functions |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Wrong parameter name for RefractiveProjectionModel constructor**
- **Found during:** Task 1 typecheck
- **Issue:** `interface_normal=calib.interface_normal` used but actual param is `normal`
- **Fix:** Changed to `normal=calib.interface_normal` in orchestrator.py
- **Files modified:** `src/aquapose/pipeline/orchestrator.py`
- **Commit:** f85f02b

**2. [Rule 1 - Bug] `crop_image` does not exist in segmentation.crop**
- **Found during:** Task 1 implementation
- **Issue:** Planned to call `crop_image()` but the function is named `extract_crop()`
- **Fix:** Changed to `extract_crop()` with correct signature
- **Files modified:** `src/aquapose/pipeline/stages.py`
- **Commit:** f85f02b

**3. [Rule 1 - Bug] `compute_crop_region` takes `(x, y, w, h)` bbox not `(x1, y1, x2, y2)`**
- **Found during:** Task 1 implementation
- **Issue:** Initial code passed `(bx, by, bx+bw, by+bh)` but signature is `(bbox: tuple[int, int, int, int])` in `(x, y, w, h)` format
- **Fix:** Changed to pass `det.bbox` directly
- **Files modified:** `src/aquapose/pipeline/stages.py`
- **Commit:** f85f02b

**4. [Rule 1 - Bug] test_stages.py patch target was wrong for lazy import**
- **Found during:** Task 2 test execution
- **Issue:** `patch("aquapose.pipeline.stages.triangulate_midlines")` failed because `triangulate_midlines` is lazily imported inside the function body, not at module level
- **Fix:** Changed patch target to `aquapose.reconstruction.triangulation.triangulate_midlines`
- **Files modified:** `tests/unit/pipeline/test_stages.py`
- **Commit:** d0f4ff7

**5. [Rule 1 - Bug] Mock model cast_ray needs proper return type**
- **Found during:** Task 2 test execution
- **Issue:** Default MagicMock for `cast_ray` returns a MagicMock, not a `(origins, directions)` tuple, causing `ValueError: not enough values to unpack`
- **Fix:** Configured mock to return `(torch.zeros(N,3), torch.zeros(N,3))` tensors
- **Files modified:** `tests/unit/pipeline/test_stages.py`
- **Commit:** d0f4ff7

## Self-Check: PASSED

All created files verified to exist. Commits f85f02b and d0f4ff7 confirmed in git log.
