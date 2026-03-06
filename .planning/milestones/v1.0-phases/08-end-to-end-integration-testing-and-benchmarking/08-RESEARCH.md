# Phase 8: End-to-End Integration Testing and Benchmarking - Research

**Researched:** 2026-02-21
**Domain:** Pipeline orchestration, HDF5 output, visualization (matplotlib/opencv), diagnostic reporting
**Confidence:** HIGH (based on direct codebase inspection; no speculative library lookups needed — all required libraries are already project dependencies)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Test data & scenarios**
- Use a 5-minute video clip as the primary test data
- Include a `stop_frame` argument throughout — set low during development for fast iteration, run full 5 minutes once pipeline is working
- Test with all 9 fish (full population) from the start, no single-fish stepping stone
- e3v8250 excluded automatically (lives in separate folder from core videos)
- Evaluation is visual inspection only — no manual ground truth annotations

**Pipeline entry point**
- Python API call (not CLI): a function like `reconstruct(video_dir, calibration, ..., stop_frame=None)` that returns results programmatically
- Composable stages: each stage (detect, segment, track, extract midlines, triangulate) is independently callable; the E2E function chains them together
- Stage-by-stage batch processing: run each stage on all frames before moving to the next
- Within-stage sub-batching/chunking to support long videos without exhausting memory

**Pipeline modes**
- Single `mode` argument accepting `"diagnostic"` or `"production"`
- **Diagnostic mode**: all visualizations enabled, intermediate results saved to disk, timing stats logged, and a synthesized Markdown report with embedded figures produced at the end
- **Production mode**: no visualizations, only critical artifacts saved (final HDF5), minimal logging
- Additional customization levels deferred to future work

**Intermediate storage**
- Optional disk caching: in-memory by default, with a flag (enabled automatically in diagnostic mode) to persist intermediate results for debugging/resumption

**Output format**
- Primary output: HDF5 file with 3D midline results (consistent with Phase 5 tracking output pattern)
- Configurable output directory: user passes an output path, all results written there in organized structure

**Visualizations (diagnostic mode)**
- 3D midline overlay reprojected onto each camera's video frames
- 3D scatter/plot of reconstructed midlines in tank coordinates
- Per-stage diagnostic images (detection boxes, segmentation masks, skeletons, etc.)
- 3D animation of midlines moving through space, saved as MP4 via Matplotlib

**Timing & logging**
- Per-stage wall time logged, summary table printed at pipeline completion
- No real-time progress bars — summary at end is sufficient

### Claude's Discretion
- Exact HDF5 schema for 3D midlines (building on Phase 5 patterns)
- Sub-batch size defaults and chunking strategy
- Diagnostic Markdown report layout and figure arrangement
- Stage interface contracts (argument/return types between stages)
- Error handling strategy for partial failures (e.g., some frames fail triangulation)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 8 is the integration and output phase: wire the complete direct triangulation pipeline into a single callable function, implement the two output requirements (HDF5 3D midlines + visualization overlays), and create diagnostic tooling for visual inspection. There is no new algorithmic work; this phase is about plumbing existing modules together and building the output infrastructure.

The five pipeline stages are fully implemented: MOG2/YOLO detection (`segmentation/detector.py`), U-Net segmentation (`segmentation/model.py`), FishTracker cross-view identity (`tracking/tracker.py`), MidlineExtractor (`reconstruction/midline.py`), and triangulate_midlines (`reconstruction/triangulation.py`). The HDF5 writer pattern is established in `tracking/writer.py` (TrackingWriter) — the 3D midline writer follows the same chunked-append pattern. Visualization uses OpenCV (already a dependency) for 2D overlays on video frames and Matplotlib (standard library in hatch env, already importable) for 3D plots and MP4 animation.

The primary planning challenge is designing the stage interface contracts — what each batch stage returns and what the next stage consumes — and the HDF5 schema extension for `Midline3D` data. Both are "Claude's discretion" items. The diagnostic Markdown report is a file-writing task (write figures, embed as relative paths) with no new library dependencies.

**Primary recommendation:** Build Phase 8 as two plans: (1) pipeline orchestrator + HDF5 midline writer + unit tests; (2) visualization overlays + diagnostic report generator + end-to-end test on real data.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| h5py | >=3.9 (project dep) | HDF5 3D midline storage | Already used in Phase 5 TrackingWriter; chunked-append pattern established |
| opencv-python | >=4.8 (project dep) | Video reading, 2D overlay rendering | Already used throughout codebase; cv2.VideoCapture for frame reading |
| matplotlib | stdlib in hatch env | 3D scatter plots, MP4 animation | Standard scientific Python; `mpl_toolkits.mplot3d`, `FuncAnimation`, `FFMpegWriter` |
| numpy | >=1.24 (project dep) | Array manipulation throughout | Already ubiquitous |
| scipy | >=1.11 (project dep) | Spline evaluation for overlay | Already used in triangulation module |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib.Path | stdlib | Output directory management | Use throughout for cross-platform path handling |
| time / dataclasses | stdlib | Timing stats, per-stage result containers | Use `time.perf_counter()` for wall-time measurement |
| logging | stdlib | Structured pipeline logging | Production mode uses WARNING level; diagnostic uses DEBUG |
| contextlib.ExitStack | stdlib | Resource cleanup for writers | Manage HDF5 file + video writers together |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib MP4 | imageio, cv2.VideoWriter | matplotlib FuncAnimation integrates naturally with 3D plots already being built; imageio has no advantages here |
| h5py direct write | zarr | h5py is already established in the codebase; zarr would introduce new dependency for no benefit |
| Markdown string building | Jinja2 templating | Jinja2 is overkill for a single diagnostic report; plain f-string construction is sufficient |

**Installation:** No new dependencies needed. All required libraries are already in `pyproject.toml` or Python stdlib.

---

## Architecture Patterns

### Recommended Project Structure

```
src/aquapose/
├── pipeline/           # NEW: E2E orchestration
│   ├── __init__.py     # exports reconstruct(), Stage result types
│   ├── orchestrator.py # reconstruct() function, stage dispatch
│   ├── stages.py       # run_detection(), run_segmentation(), run_tracking(),
│   │                   #   run_midline_extraction(), run_triangulation()
│   └── report.py       # DiagnosticReport: figure writing + Markdown generation
├── io/
│   ├── __init__.py     # export Midline3DWriter, read_midline3d_results
│   └── midline_writer.py  # HDF5 writer for Midline3D (modeled on TrackingWriter)
└── visualization/      # NEW: visual output utilities
    ├── __init__.py
    ├── overlay.py      # reproject 3D midline onto camera frames (2D overlay)
    └── plot3d.py       # 3D scatter + MP4 animation via matplotlib
```

### Pattern 1: Stage-by-Stage Batch Processing

**What:** Each stage consumes the full frame list output of the previous stage before the next stage begins. Stages are independently callable functions that accept a list of frames' worth of data.

**When to use:** Memory-bounded long videos — each stage can sub-batch internally before returning the accumulated result list.

**Example interface contracts:**

```python
# Stage 1 output: per-camera, per-frame detection lists
DetectionsPerFrame = list[dict[str, list[Detection]]]
# frame_idx -> camera_id -> [Detection, ...]

# Stage 2 output: per-camera, per-frame segmentation mask lists
MasksPerFrame = list[dict[str, list[np.ndarray]]]
# frame_idx -> camera_id -> [mask_uint8, ...]  (parallel to detections)

# Stage 3 output: per-frame list of confirmed FishTracks
TracksPerFrame = list[list[FishTrack]]
# frame_idx -> [FishTrack, ...]

# Stage 4 output: per-frame MidlineSet
MidlineSetsPerFrame = list[MidlineSet]
# frame_idx -> {fish_id: {camera_id: Midline2D}}

# Stage 5 output: per-frame dict[fish_id, Midline3D]
Midlines3DPerFrame = list[dict[int, Midline3D]]
# frame_idx -> {fish_id: Midline3D}
```

**Key detail:** FishTracker is stateful (maintains track history). It must NOT be re-instantiated between sub-batches. The tracker instance spans the full video; only frame data is chunked.

### Pattern 2: HDF5 Midline Writer (extending TrackingWriter pattern)

**What:** Chunked-append HDF5 writer for `Midline3D` data, following the exact same pattern as `TrackingWriter` in `tracking/writer.py`.

**Schema (Claude's discretion — recommended):**
```
/midlines/
    frame_index         (N,)              int64    fillvalue=0
    fish_id             (N, max_fish)     int32    fillvalue=-1
    control_points      (N, max_fish, 7, 3) float32  fillvalue=NaN
    knots               (11,)             float32  — stored once as attr
    degree              scalar attr       int
    arc_length          (N, max_fish)     float32  fillvalue=NaN
    half_widths         (N, max_fish, 15) float32  fillvalue=NaN
    n_cameras           (N, max_fish)     int32    fillvalue=0
    mean_residual       (N, max_fish)     float32  fillvalue=-1.0
    max_residual        (N, max_fish)     float32  fillvalue=-1.0
    is_low_confidence   (N, max_fish)     bool     fillvalue=False
```

**Why this schema:** Matches OUT-01 requirement exactly. Same fish slot / fill-value conventions as TrackingWriter so downstream readers need only one reader pattern. `knots` and `degree` stored as attrs (constant across all frames).

### Pattern 3: Reprojection Overlay (OUT-02)

**What:** For each camera, evaluate the fitted B-spline at N points, project to 2D via `RefractiveProjectionModel.project()`, draw on the video frame with OpenCV.

```python
# Evaluate spline at N points using scipy
from scipy.interpolate import BSpline
spl = BSpline(midline_3d.knots, midline_3d.control_points, midline_3d.degree)
pts_3d = spl(np.linspace(0, 1, 30))  # (30, 3)

# Project via refractive model
pts_tensor = torch.from_numpy(pts_3d).float()
pixels, valid = model.project(pts_tensor)  # (30, 2), (30,)

# Draw polyline on frame
valid_px = pixels[valid].numpy().astype(int)
cv2.polylines(frame, [valid_px.reshape(-1, 1, 2)], False, (0, 255, 0), 2)
```

**Width tube overlay:** evaluate half_widths at same arc positions, draw ellipses or filled circles for body cross-section width indication.

### Pattern 4: 3D Animation (OUT-03 via matplotlib)

**What:** Accumulate `Midline3D.control_points` per frame, animate via `matplotlib.animation.FuncAnimation`, save with `FFMpegWriter` or `PillowWriter`.

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame_idx):
    ax.cla()
    for fish_id, midline in frames_data[frame_idx].items():
        spl = BSpline(midline.knots, midline.control_points, midline.degree)
        pts = spl(np.linspace(0, 1, 30))
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.set_title(f"Frame {frame_idx}")

anim = FuncAnimation(fig, update, frames=len(frames_data))
writer = FFMpegWriter(fps=15)
anim.save("midlines_3d.mp4", writer=writer)
```

**Note:** FFMpeg must be installed on the system (usually available on Windows via conda/choco). If unavailable, fallback: `PillowWriter` produces animated GIF.

### Pattern 5: Diagnostic Markdown Report

**What:** After a diagnostic mode run, write a Markdown file that embeds figure paths and a timing table. No templating library needed.

```python
def write_diagnostic_report(output_dir: Path, timing: dict[str, float], ...) -> None:
    lines = ["# AquaPose Diagnostic Run\n"]
    lines.append(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n")
    lines.append("## Stage Timing\n")
    lines.append("| Stage | Time (s) |")
    lines.append("|-------|----------|")
    for stage, t in timing.items():
        lines.append(f"| {stage} | {t:.1f} |")
    lines.append("\n## Detection Samples\n")
    for fig_path in detection_figure_paths:
        lines.append(f"![detection]({fig_path.name})\n")
    (output_dir / "report.md").write_text("\n".join(lines))
```

### Anti-Patterns to Avoid

- **Re-instantiating FishTracker per sub-batch:** FishTracker is stateful; it must be created once before the frame loop and passed across sub-batches. The batch stage functions should accept an existing tracker instance, not create a new one.
- **Loading all video frames into RAM at once:** With 5-minute clips at ~9000 frames, use cv2.VideoCapture frame-by-frame reads inside the stage loop. Do not `list(gen)` on a video reader up front.
- **Saving MP4 animation all at once for long videos:** For visualization, write individual overlay frames directly to `cv2.VideoWriter` during the overlay stage rather than accumulating all frames and using FuncAnimation for 2D overlays. FuncAnimation memory model only suitable for 3D animation (fewer frames, smaller data).
- **Using `hatch run` in scripts that import aquapose:** Stage functions are library code; scripts/tests call them via normal `import aquapose`. Keep CLI concerns out of the library.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| B-spline evaluation for overlay | Custom DeCasteljau loop | `scipy.interpolate.BSpline(knots, ctrl_pts, k)(t_array)` | Already used in triangulation; consistent knot/ctrl_pt convention |
| 2D polyline overlay | Custom pixel drawing | `cv2.polylines()` / `cv2.circle()` | One-liner; anti-aliased; already used in codebase |
| MP4 video writing | Raw frame byte assembly | `cv2.VideoWriter` (2D overlays) or `matplotlib FFMpegWriter` (3D) | Both patterns are well-supported by existing deps |
| Frame-by-frame video reading | Custom codec wrapping | `cv2.VideoCapture` with `.read()` loop | Mature; handles all codec formats |
| HDF5 chunked append | Custom binary file | `h5py` — same as TrackingWriter | Already validated in Phase 5; re-use exact pattern |
| Hungarian assignment (already done) | Any matching code | `FishTracker.update()` in tracking/tracker.py | Already handles this |

**Key insight:** Every building block exists in the codebase or in already-imported libraries. Phase 8 is almost entirely plumbing and output formatting, not new algorithms.

---

## Common Pitfalls

### Pitfall 1: FishTracker State Lost Between Sub-Batches
**What goes wrong:** If the pipeline creates a new `FishTracker` instance per sub-batch (e.g., per chunk of 100 frames), track IDs reset, coasting tracks die, and the population constraint breaks at chunk boundaries.
**Why it happens:** Natural to think "run detection on batch, then tracking on batch" as independent operations.
**How to avoid:** Pass the tracker instance as a parameter to the tracking stage function. The tracker persists across the full video; only the frame data is chunked.
**Warning signs:** Track count drops to 0 at chunk boundaries in diagnostic logs.

### Pitfall 2: MidlineExtractor Orientation Buffer Cleared Between Chunks
**What goes wrong:** `MidlineExtractor` maintains per-fish orientation state (`_orientations` dict) and back-correction buffers. If a new instance is created per chunk, orientation inheritance breaks in the first ~30 frames of each chunk.
**Why it happens:** Same pattern as FishTracker — stateful objects hidden inside "functionally-styled" stage functions.
**How to avoid:** Instantiate `MidlineExtractor` once before the frame loop and pass it as a parameter.
**Warning signs:** Head-tail flips appearing consistently at chunk boundaries.

### Pitfall 3: e3v8250 Camera Included
**What goes wrong:** The center camera `e3v8250` is wide-angle, top-down, and produces poor-quality masks. It lives in a separate folder. If the pipeline uses a glob to discover cameras, it may accidentally include it.
**Why it happens:** `video_dir` may contain mixed camera types.
**How to avoid:** Auto-exclude by detecting the e3v8250 folder structure at startup, or by checking against an explicit camera allowlist from the CalibrationData.
**Warning signs:** Significantly degraded triangulation quality; masks appear distorted/fish not visible.

### Pitfall 4: Diagnostic Report Figure Paths Break
**What goes wrong:** Markdown report embeds figure paths like `/absolute/path/to/fig.png`, which only renders on the machine that generated the report.
**Why it happens:** Lazy path construction.
**How to avoid:** Always embed figures using paths relative to the output directory. The report and all its figures live in the same output directory.
**Warning signs:** Broken image links in the generated Markdown.

### Pitfall 5: FFMpeg Not Available for 3D Animation
**What goes wrong:** `matplotlib.animation.FFMpegWriter` raises `FileNotFoundError` if `ffmpeg` is not on PATH.
**Why it happens:** FFMpeg is a system dependency, not a Python package.
**How to avoid:** Check for FFMpeg availability at startup; if unavailable, fall back to `PillowWriter` (GIF) and log a warning.
**Warning signs:** Pipeline crashes at animation save step despite all reconstruction succeeding.

### Pitfall 6: HDF5 File Left Open on Pipeline Exception
**What goes wrong:** If the pipeline crashes mid-run, the HDF5 file may be left in an inconsistent (partially-written) state.
**Why it happens:** `h5py.File` not used as context manager.
**How to avoid:** Use `Midline3DWriter` as a context manager (same as `TrackingWriter`). Wrap the pipeline loop in `with Midline3DWriter(...) as writer:`.
**Warning signs:** HDF5 read errors on partially-written output files.

### Pitfall 7: MOG2 Detector Needs Background Warm-Up
**What goes wrong:** MOG2 background subtraction requires ~500 frames to establish a stable background model. On the first ~500 frames, detection recall is low, with most fish undetected.
**Why it happens:** This is known behavior documented in `STATE.md` Phase decisions.
**How to avoid:** Either (a) run MOG2 on a warm-up clip before `stop_frame` counting starts, or (b) document in diagnostic report that first N frames have expected low recall. The `stop_frame` parameter should count from the start of recording, not post-warmup.
**Warning signs:** Very low track count for first 500 frames, improving sharply afterward.

---

## Code Examples

### Stage 1: Detection (per-frame batch)

```python
# Source: existing segmentation/detector.py interface
def run_detection_batch(
    video_paths: dict[str, Path],
    stop_frame: int | None,
    chunk_size: int = 100,
) -> DetectionsPerFrame:
    """Run MOG2 detection on all cameras, returning per-frame detection lists."""
    detectors = {cam: MOG2Detector() for cam in video_paths}
    caps = {cam: cv2.VideoCapture(str(p)) for cam, p in video_paths.items()}
    results: DetectionsPerFrame = []
    frame_idx = 0
    while True:
        if stop_frame is not None and frame_idx >= stop_frame:
            break
        frame_dets: dict[str, list[Detection]] = {}
        any_frame = False
        for cam, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue
            any_frame = True
            dets = detectors[cam].detect(frame)
            frame_dets[cam] = dets
        if not any_frame:
            break
        results.append(frame_dets)
        frame_idx += 1
    for cap in caps.values():
        cap.release()
    return results
```

### Stage 5: Triangulation (per-frame)

```python
# Source: reconstruction/triangulation.py
from aquapose.reconstruction.triangulation import triangulate_midlines

def run_triangulation_batch(
    midline_sets: MidlineSetsPerFrame,
    models: dict[str, RefractiveProjectionModel],
) -> Midlines3DPerFrame:
    results = []
    for frame_idx, midline_set in enumerate(midline_sets):
        midlines_3d = triangulate_midlines(midline_set, models, frame_index=frame_idx)
        results.append(midlines_3d)
    return results
```

### HDF5 Writer Skeleton

```python
# Source: based on tracking/writer.py pattern
class Midline3DWriter:
    def __init__(self, output_path: Path, max_fish: int = 9, chunk_frames: int = 1000):
        self._file = h5py.File(output_path, "w")
        grp = self._file.require_group("midlines")
        grp.attrs["knots"] = SPLINE_KNOTS.astype(np.float32)
        grp.attrs["degree"] = SPLINE_K
        # create chunked datasets for: frame_index, fish_id, control_points,
        # arc_length, half_widths, n_cameras, mean_residual, max_residual,
        # is_low_confidence  (same _make() pattern as TrackingWriter)

    def write_frame(self, frame_index: int, midlines: dict[int, Midline3D]) -> None:
        ...  # fill buffer row, flush when full

    def close(self) -> None:
        self._flush()
        self._file.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
```

### Spline Evaluation for Overlay

```python
# Source: scipy.interpolate docs — BSpline interface
from scipy.interpolate import BSpline
import numpy as np

def eval_midline_3d(midline: Midline3D, n_eval: int = 30) -> np.ndarray:
    """Evaluate the 3D B-spline at n_eval uniform arc positions.

    Returns shape (n_eval, 3) float32.
    """
    spl = BSpline(
        midline.knots.astype(np.float64),
        midline.control_points.astype(np.float64),
        midline.degree,
    )
    t_eval = np.linspace(0.0, 1.0, n_eval)
    return spl(t_eval).astype(np.float32)
```

### Timing Pattern

```python
import time

stage_times: dict[str, float] = {}

t0 = time.perf_counter()
detections = run_detection_batch(...)
stage_times["detection"] = time.perf_counter() - t0

t0 = time.perf_counter()
masks = run_segmentation_batch(...)
stage_times["segmentation"] = time.perf_counter() - t0

# ... etc ...

# Summary table
print("\n=== Pipeline Timing ===")
for stage, elapsed in stage_times.items():
    print(f"  {stage:20s}: {elapsed:6.1f}s")
print(f"  {'TOTAL':20s}: {sum(stage_times.values()):6.1f}s")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Analysis-by-synthesis (differentiable rendering) | Direct triangulation pipeline | 2026-02-21 pivot | 5 stages → this phase wires them |
| Manual HDF5 schema per phase | Consistent chunked-append writer pattern | Phase 5 (TrackingWriter) | Midline3DWriter follows same pattern exactly |
| Mask R-CNN segmentation | U-Net (MobileNetV3-Small encoder, 128x128 crops) | Phase 02.1 | Use `UNetSegmentor` (or `make_detector("yolo", ...)` for detection) |

---

## Open Questions

1. **MOG2 warm-up handling**
   - What we know: MOG2 needs ~500 frames to stabilize; `stop_frame` counts from frame 0
   - What's unclear: Should warm-up frames run before `stop_frame` counting (i.e., warm-up is free), or does the user just accept degraded quality in the first 500 frames?
   - Recommendation: Implement an optional `warmup_frames: int = 0` parameter. Default 0 (no warmup, consistent with stop_frame counting). Document the tradeoff.

2. **FFMpeg availability check**
   - What we know: `matplotlib.animation.FFMpegWriter` requires system ffmpeg
   - What's unclear: Whether ffmpeg is already available in the user's hatch env
   - Recommendation: Check via `matplotlib.animation.FFMpegWriter.isAvailable()` at diagnostic startup; fall back to `PillowWriter` (GIF) with a logged warning.

3. **Video writer codec for 2D overlays**
   - What we know: `cv2.VideoWriter` requires a codec fourcc; `mp4v` works on Windows
   - What's unclear: Whether the output MP4 will be playable in standard players with `mp4v`
   - Recommendation: Use `cv2.VideoWriter_fourcc(*'mp4v')` as default; document that `avc1` (H.264) may require additional codec installation.

4. **Stage result caching format for disk persistence (diagnostic mode)**
   - What we know: Intermediate results should be cached to disk in diagnostic mode for resumption
   - What's unclear: Format — numpy `.npz`, pickle, or HDF5 for each stage?
   - Recommendation: Use `.npz` for numpy-serializable stage results (detections as bbox arrays, masks as arrays). Keep it simple; this is for debugging, not production archival.

5. **Exact `reconstruct()` return type**
   - What we know: Returns results programmatically
   - What's unclear: Should it return the full `Midlines3DPerFrame` list in memory, or just the path to the output HDF5?
   - Recommendation: Return a `ReconstructResult` dataclass with `output_dir: Path`, `midlines_3d: Midlines3DPerFrame` (in memory), and `stage_timing: dict[str, float]`. Caller decides whether to process in-memory or reload from HDF5.

---

## Phase Requirements Traceability

| ID | Description | Research Support |
|----|-------------|-----------------|
| OUT-01 | HDF5 per-frame results: fish_id, 3D spline control points, width profile, centroid, heading, curvature, n_cameras, triangulation residual | Midline3DWriter pattern (Pattern 2 above); schema covers all fields except heading/curvature — derive heading from control_points[1]-control_points[0], curvature from spline second derivative |
| OUT-02 | Reprojected 3D midline + width profile overlay on camera views for 2D visual QA | Pattern 3 above; BSpline eval → RefractiveProjectionModel.project() → cv2.polylines |
| OUT-03 | 3D midline tube models in tank volume via rerun-sdk with trajectory trails | CONTEXT.md decided to use Matplotlib MP4 instead of rerun-sdk; Pattern 4 above covers this |

---

## Sources

### Primary (HIGH confidence)
- Codebase direct inspection: `src/aquapose/reconstruction/triangulation.py` — Midline3D dataclass fields, SPLINE_KNOTS, N_SAMPLE_POINTS constants
- Codebase direct inspection: `src/aquapose/tracking/writer.py` — TrackingWriter chunked-append pattern (HDF5 schema to mirror)
- Codebase direct inspection: `src/aquapose/reconstruction/midline.py` — MidlineExtractor stateful design (orientation buffers)
- Codebase direct inspection: `src/aquapose/tracking/tracker.py` — FishTracker stateful design (track lifecycle)
- `pyproject.toml` — confirmed existing dependencies (h5py, opencv-python, scipy, numpy, matplotlib not listed but stdlib in Python scientific stack)
- `.planning/phases/08-end-to-end-integration-testing-and-benchmarking/08-CONTEXT.md` — all locked decisions above

### Secondary (MEDIUM confidence)
- `scipy.interpolate.BSpline` — standard interface, consistent with make_lsq_spline output format already used in triangulation module
- `matplotlib.animation.FuncAnimation` + `FFMpegWriter` — standard matplotlib API, stable since matplotlib 1.1

### Tertiary (LOW confidence)
- FFMpeg system availability on Windows — not verified in this environment; flagged as Open Question

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in pyproject.toml; no new deps needed
- Architecture: HIGH — directly derived from existing module interfaces; stage contracts follow established patterns
- HDF5 schema: HIGH — mirrors TrackingWriter exactly; field set derived from Midline3D dataclass
- Pitfalls: HIGH — stateful object pitfalls derived from code inspection; others (FFMpeg, MOG2 warmup) from documented project decisions in STATE.md
- Visualization patterns: MEDIUM — matplotlib FuncAnimation API is stable but FFMpeg system dep is unverified

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (stable libraries; no fast-moving APIs involved)
