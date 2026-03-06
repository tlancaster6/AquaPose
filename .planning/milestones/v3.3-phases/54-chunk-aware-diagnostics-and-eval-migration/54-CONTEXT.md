# Phase 54: Chunk-Aware Diagnostics and Eval Migration - Context

**Gathered:** 2026-03-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Adapt diagnostic mode for multi-chunk support with per-chunk cache directories, simplify stage caching to a single cache file per chunk, and migrate all plotting/visualization functionality (overlay2d, animation3d, tracklet trails) out of pipeline observers into the eval suite where they operate on cached data post-run via CLI subcommands.

</domain>

<decisions>
## Implementation Decisions

### Cache directory layout
- Chunk subdirectories: `diagnostics/chunk_000/cache.pkl`, `chunk_001/cache.pkl`, etc.
- Single-chunk (non-chunked) runs use the same layout — always `chunk_000/`
- `manifest.json` at `diagnostics/manifest.json` with: run_id, total_frames, chunk_size, per-chunk entries (index, start_frame, end_frame, stages_cached), version_fingerprint
- No legacy flat-layout support — existing caches from pre-phase-54 runs are unsupported; users re-run

### Selective caching
- Single `cache.pkl` per chunk containing the full PipelineContext after reconstruction — drop the per-stage cache concept entirely
- Diagnostic mode always caches (no config toggle to disable caching)
- Version fingerprint mismatch: warn but load anyway (no StaleCacheError rejection)

### Visualization migration
- Remove OverlayObserver, Animation3DObserver, and TrackletTrailObserver from engine/ entirely
- Remove these observers from observer_factory.py diagnostic mode auto-enable
- New CLI group: `aquapose viz overlay|animation|trails|all <run_dir>`
- `aquapose viz all` attempts every visualization, skips gracefully on failure (e.g. missing video for overlay)
- Code lives in `src/aquapose/evaluation/viz/` — overlay.py, animation.py, trails.py
- Output to `{run_dir}/viz/` by default

### Cross-chunk continuity
- Overlay and trail videos span the full recording (load all chunk caches, stitch frame ranges, one continuous output)
- Video frames resolved from cached `config.yaml` in the run directory automatically (no extra --video flag needed)
- 3D Plotly animation: unified scrubber timeline across all chunks, chunk boundaries invisible to viewer
- Fish color assignment: deterministic from global fish ID — `palette[fish_id % palette_length]`. Same fish always gets same color across chunks.

### Claude's Discretion
- Manifest schema details beyond the agreed fields
- Internal cache loading/merging implementation
- Video writer configuration (codec, fps defaults)
- Fallback behavior when video path from config.yaml is unreachable (black frames)
- Observer factory cleanup details

</decisions>

<specifics>
## Specific Ideas

- The reconstruction cache already contains the full PipelineContext with all accumulated fields from every stage — this is why a single cache per chunk is sufficient
- `aquapose viz all` should be the primary command users run; individual subcommands are for selective re-rendering
- The existing `src/aquapose/visualization/` package has utility functions (overlay.py, plot3d.py, frames.py) that the new eval/viz modules may reuse

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/aquapose/visualization/overlay.py`: Existing overlay drawing utilities (reproject + draw)
- `src/aquapose/visualization/plot3d.py`: 3D spline rendering for Plotly
- `src/aquapose/visualization/frames.py`: `synthetic_frame_iter()` for black-frame fallback
- `src/aquapose/evaluation/tuning.py`: TuningOrchestrator already loads stage caches — cache loading patterns to reuse
- `src/aquapose/engine/diagnostic_observer.py`: Current cache writing logic (atomic temp-file + rename, envelope format) to adapt

### Established Patterns
- Observer factory (`observer_factory.py`): currently auto-enables overlay/animation/trail observers in diagnostic mode — must be updated
- ChunkOrchestrator (`orchestrator.py`): currently raises ValueError if diagnostic mode + chunk_size > 0 — this constraint must be removed
- ChunkHandoff: maintains global fish IDs across chunks via identity stitching — visualization can rely on these IDs being consistent

### Integration Points
- `cli.py`: New `aquapose viz` CLI group alongside existing `aquapose run` and `aquapose train`
- `engine/config.py`: DiagnosticConfig may need updates for new cache structure
- `evaluation/runner.py`: EvalRunner loads from diagnostics/ — must be updated for new chunk layout

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 54-chunk-aware-diagnostics-and-eval-migration*
*Context gathered: 2026-03-04*
