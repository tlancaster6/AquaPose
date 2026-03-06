---
phase: 20-post-refactor-loose-ends
plan: 05
subsystem: io, visualization, testing
tags: [refactoring, cleanup, discovery, diagnostics, regression-tests, environment-variables]

requires:
  - phase: 20-02
    provides: dead module deletion enabling clean slate for audit remediations
  - phase: 20-03
    provides: skip_camera removal and observer factory extraction

provides:
  - Shared camera-video discovery utility in io/discovery.py
  - Split diagnostics.py into midline_viz.py and triangulation_viz.py
  - Regression tests using environment variables for data paths

affects:
  - any future stage that discovers camera videos (use discover_camera_videos)
  - any caller of visualization/diagnostics (backward compat maintained)
  - CI/CD setup requiring regression test environment variables

tech-stack:
  added: []
  patterns:
    - "Shared discovery utility: discover_camera_videos() in io/discovery.py instead of inline glob logic in each stage"
    - "Thin re-export shim: diagnostics.py stays as 29-LOC backward-compat shim after split"
    - "Env-var-gated regression tests: pytest.skip() with clear message when AQUAPOSE_VIDEO_DIR not set"

key-files:
  created:
    - src/aquapose/io/discovery.py
    - src/aquapose/visualization/midline_viz.py
    - src/aquapose/visualization/triangulation_viz.py
  modified:
    - src/aquapose/io/__init__.py
    - src/aquapose/core/detection/stage.py
    - src/aquapose/core/midline/stage.py
    - src/aquapose/visualization/__init__.py
    - src/aquapose/visualization/diagnostics.py
    - tests/regression/conftest.py

key-decisions:
  - "diagnostics.py kept as 29-LOC backward-compat re-export shim — any existing imports from diagnostics still work"
  - "triangulation_viz.py is 1617 LOC (above ~800 guideline) — all content is cohesive triangulation/3D/synthetic viz; synthetic functions belong here not in a 4th module"
  - "MaskRCNNSegmentor backward-compat test retained — module still exists in segmentation/model.py, test is valid"
  - "Regression conftest weight defaults changed to relative paths (not machine-specific absolute paths)"

requirements-completed: [REMEDIATE]

duration: 35min
completed: 2026-02-27
---

# Phase 20 Plan 05: Remaining Audit Remediations Summary

**Camera-video discovery deduplicated into shared io/discovery.py utility, 2200-LOC diagnostics.py split into focused modules, regression tests switched to environment-variable-based data paths**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-02-27
- **Completed:** 2026-02-27
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Created `io/discovery.py` with `discover_camera_videos()` — eliminates duplicated glob logic from DetectionStage and MidlineStage
- Split 2203-LOC `diagnostics.py` into `midline_viz.py` (642 LOC, detection/tracking/midline viz) and `triangulation_viz.py` (1617 LOC, triangulation/synthetic/optimizer viz); diagnostics.py reduced to 29-LOC backward-compat shim
- Updated `tests/regression/conftest.py` to use `AQUAPOSE_VIDEO_DIR` and `AQUAPOSE_CALIBRATION_PATH` environment variables — no hardcoded machine-specific paths remain
- All 514 unit tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract shared camera-video discovery, split diagnostics.py** - `a3ebff5` (feat)
2. **Task 2: Fix regression test paths** - `3fa5c2f` (fix)

## Files Created/Modified

- `src/aquapose/io/discovery.py` - New: `discover_camera_videos(video_dir)` shared utility
- `src/aquapose/io/__init__.py` - Added `discover_camera_videos` to exports
- `src/aquapose/core/detection/stage.py` - Uses `discover_camera_videos` instead of inline glob
- `src/aquapose/core/midline/stage.py` - Uses `discover_camera_videos` instead of inline glob
- `src/aquapose/visualization/midline_viz.py` - New: detection, tracking, midline viz functions (642 LOC)
- `src/aquapose/visualization/triangulation_viz.py` - New: triangulation, synthetic, optimizer viz (1617 LOC)
- `src/aquapose/visualization/diagnostics.py` - Reduced to 29-LOC backward-compat re-export shim
- `src/aquapose/visualization/__init__.py` - Updated exports to use new focused modules
- `tests/regression/conftest.py` - Replaced hardcoded paths with environment variables

## Decisions Made

- `diagnostics.py` retained as thin re-export shim to preserve backward compatibility with any existing imports
- `triangulation_viz.py` is 1617 LOC vs the ~800 LOC guideline — all content is cohesively triangulation/3D/reconstruction visualization; splitting into 4 modules was not warranted by the plan's 3-module design
- Regression conftest weight defaults changed from absolute machine-specific paths to relative repo-based paths

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None — ruff auto-fixed minor style issues on first commit attempt (trailing whitespace, formatting), resolved automatically by re-staging.

## User Setup Required

Regression tests now require environment variables:
```bash
export AQUAPOSE_VIDEO_DIR=/path/to/core_videos
export AQUAPOSE_CALIBRATION_PATH=/path/to/calibration.json
# Optional:
export AQUAPOSE_YOLO_WEIGHTS=/path/to/yolo/best.pt
export AQUAPOSE_UNET_WEIGHTS=/path/to/unet/best_model.pth
```
Without these set, regression tests skip gracefully with a clear message.

## Next Phase Readiness

All Phase 20 audit remediations (AUD-004, AUD-012, AUD-013, AUD-014, AUD-017, AUD-018) are complete. Phase 20 is now fully done.
Phase 21 (Retrospective, Prospective) is ready to begin.

---
*Phase: 20-post-refactor-loose-ends*
*Completed: 2026-02-27*
