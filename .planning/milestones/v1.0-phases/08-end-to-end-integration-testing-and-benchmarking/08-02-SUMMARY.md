---
phase: 08-end-to-end-integration-testing-and-benchmarking
plan: 02
subsystem: visualization
tags: [visualization, overlay, 3d-animation, diagnostics, report, testing]
dependency_graph:
  requires:
    - src/aquapose/reconstruction/triangulation.py
    - src/aquapose/calibration/projection.py
    - src/aquapose/pipeline/orchestrator.py
  provides:
    - src/aquapose/visualization/__init__.py
    - src/aquapose/visualization/overlay.py
    - src/aquapose/visualization/plot3d.py
    - src/aquapose/pipeline/report.py
  affects:
    - src/aquapose/pipeline/__init__.py
    - src/aquapose/pipeline/orchestrator.py
tech_stack:
  added: []
  patterns:
    - scipy.interpolate.BSpline evaluation for spline-to-pixel rendering
    - cv2.polylines + cv2.circle for 2D overlay drawing
    - matplotlib FuncAnimation with FFMpegWriter/PillowWriter fallback
    - Pinhole approximation for width-circle radius (hw_m * focal_px / depth_m)
    - Shared BGR FISH_COLORS palette converted to RGB for matplotlib
    - Markdown report generation via string list joining (no template engine)
key_files:
  created:
    - src/aquapose/visualization/__init__.py
    - src/aquapose/visualization/overlay.py
    - src/aquapose/visualization/plot3d.py
    - src/aquapose/pipeline/report.py
    - tests/unit/visualization/__init__.py
    - tests/unit/visualization/test_overlay.py
  modified:
    - src/aquapose/pipeline/__init__.py
    - src/aquapose/pipeline/orchestrator.py
decisions:
  - "FISH_COLORS defined as BGR tuples in overlay.py (OpenCV convention); plot3d.py converts BGR->RGB floats on import for matplotlib compatibility"
  - "render_3d_animation checks FFMpegWriter.isAvailable() at runtime and falls back to PillowWriter (GIF) with a UserWarning -- no hard FFMpeg dependency"
  - "draw_midline_overlay modifies frame in-place AND returns it -- consistent with cv2 convention and allows chaining"
  - "Diagnostic mode catches all visualization exceptions individually to avoid crashing the main pipeline on render failures"
  - "report.py uses datetime.UTC (Python 3.11+) per UP017 ruff rule"
metrics:
  duration: 9 min
  completed: 2026-02-21
  tasks_completed: 2
  files_created: 6
  files_modified: 2
---

# Phase 8 Plan 02: Visualization and Diagnostic Report Summary

One-liner: 2D reprojection overlay via cv2.polylines + refractive projection, 3D FuncAnimation with FFMpeg/GIF fallback, and Markdown diagnostic report with timing and residual statistics.

## What Was Built

**`src/aquapose/visualization/` — new package:**

- **`overlay.py`**: `draw_midline_overlay()` evaluates the B-spline at `n_eval` points using `scipy.interpolate.BSpline`, projects via `RefractiveProjectionModel.project()`, and draws a cv2 polyline. With `draw_widths=True`, circles are drawn at every 5th point using the pinhole formula `radius_px = hw_m * focal_px / depth_m`. Color is assigned from `FISH_COLORS[fish_id % 10]`. `render_overlay_video()` wraps `cv2.VideoCapture` + `cv2.VideoWriter` to produce per-camera overlay MP4s.

- **`plot3d.py`**: `plot_3d_frame()` plots 3D lines for each fish in a single frame using matplotlib Axes3D with equal-aspect-ratio approximation. `render_3d_animation()` uses `FuncAnimation` to produce MP4 (FFMpegWriter) or GIF (PillowWriter fallback), with centroid trail dots fading over the last 10 frames.

- **`__init__.py`**: Exports `FISH_COLORS`, `draw_midline_overlay`, `render_overlay_video`, `plot_3d_frame`, `render_3d_animation`.

**`src/aquapose/pipeline/report.py`** — `write_diagnostic_report()`:
- Stage timing table with percentage of total
- Reconstruction summary: unique fish tracked, mean/max residuals, low-confidence midline percentage
- Per-frame fish count statistics (min, max, mean, std)
- Embedded figure references (relative paths, only for files that exist)
- Returns path to written `report.md`

**`src/aquapose/pipeline/orchestrator.py` — diagnostic mode additions:**
- Saves `figures/midline_3d_sample.png` from first non-empty frame
- Calls `render_3d_animation()` to produce `midlines_3d.mp4`
- Calls `render_overlay_video()` per camera to produce `overlays/{cam_id}.mp4`
- Calls `write_diagnostic_report()` to produce `report.md`
- All wrapped in individual `try/except` to prevent visualization failures from blocking HDF5 output

**Tests (3 total, all fast, all GPU-free):**
- `test_draw_midline_overlay_adds_polyline`: verifies non-zero pixels after overlay via mocked `project()`
- `test_draw_midline_overlay_with_widths`: verifies width circles add >= as many pixels as polyline-only
- `test_fish_color_palette_has_10_colors`: verifies FISH_COLORS has >= 10 distinct BGR int-tuples

## Verification

- `hatch run check` passes (lint + typecheck; 4 pre-existing errors in detector.py only)
- `hatch run test tests/unit/visualization/test_overlay.py` — 3/3 pass
- `python -c "from aquapose.pipeline import reconstruct, ReconstructResult, write_diagnostic_report"` — succeeds
- `python -c "from aquapose.visualization import draw_midline_overlay, render_overlay_video, render_3d_animation, plot_3d_frame"` — succeeds

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1: Visualization package | be039b6 | feat(08-02): 2D overlay renderer and 3D midline animation |
| Task 2: Report, tests, orchestrator | f214b80 | feat(08-02): diagnostic report generator, orchestrator diagnostic mode, and overlay unit tests |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `cv2.VideoWriter_fourcc` not a known attribute of `cv2` (basedpyright)**
- **Found during:** Task 1 typecheck
- **Issue:** `cv2.VideoWriter_fourcc` flagged as unknown attribute by basedpyright; correct form is `cv2.VideoWriter.fourcc`
- **Fix:** Changed to `cv2.VideoWriter.fourcc(*codec)`
- **Files modified:** `src/aquapose/visualization/overlay.py`
- **Commit:** be039b6

**2. [Rule 1 - Bug] `Figure | SubFigure | None` return from `ax.get_figure()` not compatible with `Figure` return type**
- **Found during:** Task 1 typecheck
- **Issue:** `ax.get_figure()` returns `Figure | SubFigure | None`; `plot_3d_frame` is declared to return `Figure`
- **Fix:** Added `isinstance(raw_fig, Figure)` check with `raise ValueError` for non-Figure cases
- **Files modified:** `src/aquapose/visualization/plot3d.py`
- **Commit:** be039b6

**3. [Rule 1 - Bug] FuncAnimation callback must return `Iterable[Artist]`, not `None`**
- **Found during:** Task 1 typecheck
- **Issue:** basedpyright flagged `_update` returning `None` as incompatible with FuncAnimation's expected `(...) -> Iterable[Artist]`
- **Fix:** Changed return type to `list[Artist]` and returned `[]` at end of `_update`
- **Files modified:** `src/aquapose/visualization/plot3d.py`
- **Commit:** be039b6

## Self-Check: PASSED

All created files verified to exist. Commits be039b6 and f214b80 confirmed in git log.
