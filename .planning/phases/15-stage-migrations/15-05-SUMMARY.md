---
phase: 15-stage-migrations
plan: 05
subsystem: core
tags: [reconstruction, triangulation, curve-optimizer, b-spline, stage-protocol, pipeline]

# Dependency graph
requires:
  - phase: 15-04
    provides: TrackingStage, FishTrack.camera_detections for MidlineSet assembly
  - phase: 15-02
    provides: MidlineStage, AnnotatedDetection with Midline2D for MidlineSet assembly
  - phase: 13-engine-core
    provides: Stage Protocol, PipelineContext, PosePipeline, PipelineConfig
provides:
  - ReconstructionStage in core/reconstruction/ satisfying Stage Protocol
  - TriangulationBackend delegating to triangulate_midlines()
  - CurveOptimizerBackend wrapping stateful CurveOptimizer for warm-starting
  - MidlineSet assembly logic bridging Stage 2 and Stage 4 outputs
  - build_stages(config) factory in engine/pipeline.py wiring all 5 stages
  - 15-BUG-LEDGER.md documenting all preserved v1.0 quirks
  - 16 interface tests verifying Stage Protocol, backend delegation, import boundary
affects: [16-verification, 17-observers, 18-cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Calibration loaded at backend construction — fail-fast on missing file"
    - "Stateful CurveOptimizer persists across reconstruct_frame() calls for warm-starting"
    - "MidlineSet assembly from FishTrack.camera_detections + AnnotatedDetection.midline"
    - "build_stages(config) factory as canonical pipeline wiring entrypoint"
    - "TYPE_CHECKING guard for PipelineContext annotation — no runtime engine/ imports"

key-files:
  created:
    - src/aquapose/core/reconstruction/__init__.py
    - src/aquapose/core/reconstruction/types.py
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/core/reconstruction/backends/__init__.py
    - src/aquapose/core/reconstruction/backends/triangulation.py
    - src/aquapose/core/reconstruction/backends/curve_optimizer.py
    - tests/unit/core/reconstruction/__init__.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py
    - .planning/phases/15-stage-migrations/15-BUG-LEDGER.md
  modified:
    - src/aquapose/core/__init__.py
    - src/aquapose/engine/config.py
    - src/aquapose/engine/pipeline.py

key-decisions:
  - "TriangulationBackend loads calibration at __init__ and delegates reconstruct_frame() entirely to triangulate_midlines() — no reimplementation"
  - "CurveOptimizerBackend is stateful — single CurveOptimizer instance persists across frames for warm-starting, matching v1.0 behavior"
  - "MidlineSet assembly is new bridging code: FishTrack.camera_detections maps cam_id to det_idx in annotated_detections; this bridges Stage 2 and Stage 4"
  - "ReconstructionConfig extended with inlier_threshold, snap_threshold, max_depth — all previously hardcoded in v1.0"
  - "build_stages(config) lives in engine/pipeline.py alongside PosePipeline — orchestration logic belongs in engine/, not core/"
  - "Coasting fish (empty camera_detections) skipped during reconstruction — matches v1.0 behavior where coasting fish received no update"

patterns-established:
  - "Backend registry pattern: get_backend(kind, **kwargs) with lazy imports to avoid circular deps"
  - "_load_models() static method shared pattern across all stage backends"
  - "build_stages() as single factory for constructing all 5 stages from PipelineConfig"

requirements-completed: [STG-05]

# Metrics
duration: 35min
completed: 2026-02-26
---

# Phase 15 Plan 05: Reconstruction Stage Summary

**ReconstructionStage (Stage 5) with triangulation/curve_optimizer backends + build_stages() factory wiring all 5 stages into PosePipeline**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-02-26T01:00:00Z
- **Completed:** 2026-02-26T01:35:00Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments

- Created `core/reconstruction/` package with `ReconstructionStage` satisfying the Stage Protocol via structural typing
- Implemented `TriangulationBackend` and `CurveOptimizerBackend` as thin wrappers around v1.0 `triangulate_midlines()` and `CurveOptimizer.optimize_midlines()` — port behavior, not rewrite
- Implemented new `_assemble_midline_set()` bridging logic that constructs `dict[fish_id, dict[cam_id, Midline2D]]` from `FishTrack.camera_detections` + `AnnotatedDetection.midline`
- Added `build_stages(config)` factory to `engine/pipeline.py` as the canonical way to construct all 5 stages from a `PipelineConfig`
- Extended `ReconstructionConfig` with `inlier_threshold`, `snap_threshold`, `max_depth` fields
- Exported all 5 stage classes from `core/__init__.py`
- Created `15-BUG-LEDGER.md` documenting all preserved v1.0 quirks
- 16 interface tests pass: Protocol conformance, MidlineSet assembly, backend delegation, import boundary (ENG-07), all 5 stages importable, `build_stages()` smoke test, `PosePipeline` instantiation smoke test

## Task Commits

Each task was committed atomically:

1. **Task 1: Create core/reconstruction/ module with both backends and stage** - `392b6d2` (feat)
2. **Task 2: Interface tests for ReconstructionStage and full pipeline smoke test** - `6b37aae` (test)

## Files Created/Modified

- `src/aquapose/core/reconstruction/__init__.py` — Exports ReconstructionStage, Midline3D
- `src/aquapose/core/reconstruction/types.py` — Re-exports Midline2D, Midline3D, MidlineSet
- `src/aquapose/core/reconstruction/stage.py` — ReconstructionStage (Stage Protocol implementor)
- `src/aquapose/core/reconstruction/backends/__init__.py` — Backend registry for "triangulation" and "curve_optimizer"
- `src/aquapose/core/reconstruction/backends/triangulation.py` — TriangulationBackend wrapping triangulate_midlines()
- `src/aquapose/core/reconstruction/backends/curve_optimizer.py` — CurveOptimizerBackend wrapping stateful CurveOptimizer
- `src/aquapose/core/__init__.py` — Updated to export all 5 stage classes
- `src/aquapose/engine/config.py` — Extended ReconstructionConfig with inlier_threshold, snap_threshold, max_depth
- `src/aquapose/engine/pipeline.py` — Added build_stages(config) factory
- `tests/unit/core/reconstruction/__init__.py` — Package init
- `tests/unit/core/reconstruction/test_reconstruction_stage.py` — 16 interface tests
- `.planning/phases/15-stage-migrations/15-BUG-LEDGER.md` — v1.0 quirks ledger

## Decisions Made

- **TriangulationBackend is stateless** — no warm-starting; each frame is independent. CurveOptimizerBackend is stateful — single `CurveOptimizer` instance persists for warm-starting across frames.
- **MidlineSet assembly bridges Stage 2 + Stage 4** — `FishTrack.camera_detections` (cam_id → det_idx) is used to look up `annotated_detections[frame_idx][cam_id][det_idx].midline`. Coasting fish with empty `camera_detections` are skipped, matching v1.0.
- **`build_stages()` in engine/pipeline.py** — belongs in engine/ alongside PosePipeline since it is orchestration logic. Engine/ imports core/, never the reverse.
- **ReconstructionConfig extended** — `inlier_threshold`, `snap_threshold`, `max_depth` extracted from v1.0 hardcoded defaults to config fields for tuning without code changes.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- **Ruff lint: RUF043** — `pytest.raises(match=...)` patterns containing `.` needed to be raw strings (e.g., `r"context\.tracks"`). Fixed automatically by pre-commit hook identification; two patterns corrected.
- **Ruff format** — Minor formatting auto-fixed by pre-commit hooks (ternary expression reformatting in `curve_optimizer.py`).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 15 complete: all 5 stage migrations done (Detection, Midline, Association, Tracking, Reconstruction)
- `build_stages(config)` factory provides canonical pipeline wiring
- `PosePipeline(stages=build_stages(config), config=config)` instantiation proven
- Ready for Phase 16 (Verification): golden data comparison, numerical equivalence tests
- Bug ledger at `.planning/phases/15-stage-migrations/15-BUG-LEDGER.md` documents all preserved v1.0 quirks for reference during verification

---
*Phase: 15-stage-migrations*
*Completed: 2026-02-26*
