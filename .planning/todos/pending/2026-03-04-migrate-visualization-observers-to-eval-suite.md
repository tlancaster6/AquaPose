---
created: 2026-03-04T02:17:05.029Z
title: Migrate visualization observers to eval suite
area: visualization
files:
  - src/aquapose/engine/overlay_observer.py
  - src/aquapose/engine/animation_observer.py
  - src/aquapose/engine/tracklet_trail_observer.py
  - src/aquapose/engine/observer_factory.py
  - src/aquapose/evaluation/runner.py
---

## Problem

The three visualization observers (Overlay2DObserver, Animation3DObserver, TrackletTrailObserver) are implemented as PipelineContext observers that hook into pipeline events. This design is incompatible with chunked runs via ChunkOrchestrator because:

1. **Fresh observers per chunk** — the orchestrator creates new observer instances for each chunk, so no cross-chunk state is maintained.
2. **Output overwriting** — each chunk writes to the same output paths, so only the last chunk's output survives.
3. **Chunk-local frame indices** — observers receive frame indices relative to the chunk start, not global video positions.
4. **No cross-chunk continuity** — track IDs, fish identities, and animation state don't carry across chunk boundaries.

These observers already depend on the completed PipelineContext (they only act on PipelineComplete), so they are effectively post-hoc consumers of pipeline results — a natural fit for the eval suite.

## Solution

Migrate visualization generation from observer-based event handlers to eval-suite functions that consume diagnostic stage caches:

- Extract the rendering logic from each observer into standalone eval functions (similar to existing `evaluate_detection()`, `evaluate_reconstruction()`, etc.)
- These functions would load from `diagnostics/<stage>_cache.pkl` files, just like existing evaluators
- Support stitching across chunks by accepting a list of cache files (one per chunk) with global frame offsets
- Expose via `aquapose eval --visualize overlay2d,animation3d,trails` or similar CLI flag
- Deprecate and eventually remove the observer-based visualization classes

Related todo: "Add per-stage diagnostic visualizations" covers adding *new* visualizations; this todo covers migrating *existing* ones to a compatible architecture.
