# Phase 17: Observers - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

All diagnostic, export, and visualization side effects implemented as Observers subscribing to pipeline events. Five observers: timing, HDF5 export, 2D reprojection overlay, 3D midline animation, and diagnostic capture. Observers produce outputs independently of stage logic. The CLI entrypoint and execution modes are Phase 18.

</domain>

<decisions>
## Implementation Decisions

### HDF5 Export Format
- Two separate files: `outputs.h5` (lean, always written) and `snapshot.h5` (heavy, diagnostic mode only)
- `outputs.h5` contains final 3D spline control points only, frame-major layout (`/frames/0001/fish_0`, etc.)
- `outputs.h5` metadata is minimal: run ID, config hash, frame count, fish ID list
- `snapshot.h5` contains full pipeline state — all stage outputs (detections, 2D midlines, bundles, tracks, 3D splines)
- `snapshot.h5` metadata is rich: camera names, calibration reference, per-fish drop reasons, lifecycle states, frame timestamps
- The full-snapshot HDF5 is written by a separate observer (or observer mode) that only activates in diagnostic mode

### 2D Reprojection Overlay Visualization
- Output format: MP4 video (not image sequences)
- Default overlay content: reprojected 3D midline + original 2D midline (two colors for comparison)
- Bounding box + fish ID overlay machinery built in, available via config flag but not shown by default
- Camera layout: mosaic grid (all cameras tiled) is the default; per-camera individual videos available via config
- Video encoding via OpenCV VideoWriter or ffmpeg

### 3D Midline Animation Visualization
- Interactive HTML viewer using Plotly
- Self-contained HTML file with rotate/zoom/frame-scrub controls
- Uses Plotly `Frames` animation API with `go.Scatter3d` traces per fish
- Adds plotly as a dependency

### Diagnostic Capture
- Captures all stage outputs (not selective) — complete snapshot of PipelineContext after each stage
- In-memory only — no automatic persistence to disk
- Dict-like access API: `observer.stages["detection"][frame_idx]` returns the captured output
- Stores references (no deep copy) — relies on PipelineContext's freeze-on-populate invariant

### Observer Activation
- Timing observer is always-on (attached to every run regardless of mode)
- Mode presets defined in `engine/config.py`:
  - **production**: timing + HDF5 export (`outputs.h5`)
  - **diagnostic**: timing + HDF5 export + diagnostic capture + full-snapshot HDF5 + 2D reprojection overlay + 3D animation
  - **benchmark**: timing only
- Additive `--observers` CLI flag to add observers on top of the mode preset
- No removal/subtraction — to get fewer observers, pick a lighter mode
- Observer names for the CLI flag: `timing`, `hdf5`, `snapshot`, `diagnostic`, `overlay-2d`, `animation-3d`

### Claude's Discretion
- Timing report format and output destination (log, file, or both)
- Exact mosaic grid layout (e.g. 4x3 vs 3x4) based on camera count
- Video encoding parameters (codec, FPS, resolution)
- Plotly trace styling (colors, line widths, marker sizes)
- Internal structure of the snapshot HDF5 groups

</decisions>

<specifics>
## Specific Ideas

- 2D overlay should show 3D reprojected midline vs original 2D midline side-by-side in different colors — the primary use case is visually assessing reconstruction quality
- Mosaic grid as default because it gives a quick overview; per-camera videos for deep-diving specific views
- Diagnostic observer's dict-like API should feel natural in Jupyter notebooks for post-hoc analysis

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 17-observers*
*Context gathered: 2026-02-26*
