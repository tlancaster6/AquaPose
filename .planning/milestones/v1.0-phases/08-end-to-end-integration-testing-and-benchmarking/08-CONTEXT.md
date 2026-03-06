# Phase 8: End-to-End Integration Testing and Benchmarking - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the complete direct triangulation pipeline (detection → segmentation → cross-view identity → 2D midline extraction → 3D triangulation) into a single callable API, validate it end-to-end on real 13-camera video data, and establish benchmarking infrastructure for reconstruction quality assessment via visual inspection.

</domain>

<decisions>
## Implementation Decisions

### Test data & scenarios
- Use a 5-minute video clip as the primary test data
- Include a `stop_frame` argument throughout — set low during development for fast iteration, run full 5 minutes once pipeline is working
- Test with all 9 fish (full population) from the start, no single-fish stepping stone
- e3v8250 excluded automatically (lives in separate folder from core videos)
- Evaluation is visual inspection only — no manual ground truth annotations

### Pipeline entry point
- Python API call (not CLI): a function like `reconstruct(video_dir, calibration, ..., stop_frame=None)` that returns results programmatically
- Composable stages: each stage (detect, segment, track, extract midlines, triangulate) is independently callable; the E2E function chains them together
- Stage-by-stage batch processing: run each stage on all frames before moving to the next
- Within-stage sub-batching/chunking to support long videos without exhausting memory

### Pipeline modes
- Single `mode` argument accepting `"diagnostic"` or `"production"`
- **Diagnostic mode**: all visualizations enabled, intermediate results saved to disk, timing stats logged, and a synthesized Markdown report with embedded figures produced at the end
- **Production mode**: no visualizations, only critical artifacts saved (final HDF5), minimal logging
- Additional customization levels deferred to future work

### Intermediate storage
- Optional disk caching: in-memory by default, with a flag (enabled automatically in diagnostic mode) to persist intermediate results for debugging/resumption

### Output format
- Primary output: HDF5 file with 3D midline results (consistent with Phase 5 tracking output pattern)
- Configurable output directory: user passes an output path, all results written there in organized structure

### Visualizations (diagnostic mode)
- 3D midline overlay reprojected onto each camera's video frames
- 3D scatter/plot of reconstructed midlines in tank coordinates
- Per-stage diagnostic images (detection boxes, segmentation masks, skeletons, etc.)
- 3D animation of midlines moving through space, saved as MP4 via Matplotlib

### Timing & logging
- Per-stage wall time logged, summary table printed at pipeline completion
- No real-time progress bars — summary at end is sufficient

### Claude's Discretion
- Exact HDF5 schema for 3D midlines (building on Phase 5 patterns)
- Sub-batch size defaults and chunking strategy
- Diagnostic Markdown report layout and figure arrangement
- Stage interface contracts (argument/return types between stages)
- Error handling strategy for partial failures (e.g., some frames fail triangulation)

</decisions>

<specifics>
## Specific Ideas

- The `stop_frame` pattern is key to the development workflow: iterate quickly on a handful of frames, then scale up to the full clip once things work
- Diagnostic mode should produce a self-contained Markdown document with figures — a single artifact to review the full pipeline run
- Stage composability means you can re-run just triangulation if midline extraction changes, without re-running detection/segmentation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-end-to-end-integration-testing-and-benchmarking*
*Context gathered: 2026-02-21*
