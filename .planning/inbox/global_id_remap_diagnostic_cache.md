# Plan: Remap fish IDs to global before diagnostic cache write

## Context

The overlay visualization shows all fish changing color at chunk boundaries (every 300 frames / 10 seconds). Root cause: the diagnostic cache (`cache.pkl`) stores `context.midlines_3d` and `context.tracklet_groups` with **local per-chunk fish IDs**. Identity stitching (local → global ID remapping) happens in the orchestrator *after* `pipeline.run()` returns, but `DiagnosticObserver` writes the cache inside `pipeline.run()` on `PipelineComplete` — so the cache never sees global IDs.

The HDF5 output (`midlines.h5`) is correct because the orchestrator remaps before writing to HDF5. But all viz tools (overlay, trails, animation) and evaluation code read from the diagnostic cache and see inconsistent local IDs across chunks.

## Architectural alignment

Per the guidebook:
- "Stitching and aggregation across batches lives outside the pipeline loop" (Section 5) — identity remapping belongs in the orchestrator, which already owns it
- The engine layer "coordinates observers" (Section 2) — the orchestrator controlling cache write timing is its job
- "Observers are passive consumers" (Section 10) — the observer doesn't decide when to write, the orchestrator tells it

The fix moves the cache write from `PipelineComplete` (inside pipeline) to an explicit orchestrator call (after stitching), which better reflects the architectural boundary: the pipeline produces local results, the orchestrator stitches them into global results, then coordinates the cache write.

## Approach

Defer the diagnostic cache write until after identity stitching, then remap `context.midlines_3d` and `context.tracklet_groups` in-place before writing.

### Step 1: DiagnosticObserver — defer cache write

**File:** `src/aquapose/engine/diagnostic_observer.py`

- In `_on_pipeline_complete`: stop writing cache/manifest immediately. Just capture the context reference.
- Add a public `flush_cache()` method that writes the cache and manifest. The orchestrator calls this after remapping.

### Step 2: Orchestrator — remap in-place, then flush

**File:** `src/aquapose/engine/orchestrator.py`

After `_stitch_identities()` (line ~320):

1. **Remap `context.midlines_3d` in-place**: Replace each frame's `{local_id: Midline3D}` dict with `{global_id: Midline3D}`. Also update each `Midline3D.fish_id` attribute.
2. **Remap `context.tracklet_groups` fish_ids in-place**: Rebuild each group with `dataclasses.replace(group, fish_id=global_id)` since `TrackletGroup` is frozen.
3. **Write HDF5**: Use the already-remapped `midlines_3d` directly (remove the existing remap loop — it's now redundant).
4. **Flush diagnostic observer**: Find `DiagnosticObserver` in the observers list and call `flush_cache()`.

### Dataclass mutability (already verified)

- **`TrackletGroup`** (`src/aquapose/core/association/types.py:21`): `frozen=True`. Must rebuild each group with `dataclasses.replace(group, fish_id=global_id)`.
- **`Midline3D`** (`src/aquapose/core/types/reconstruction.py:16`): **Not frozen**. Can update `midline.fish_id = global_id` directly.

## Files to modify

1. `src/aquapose/engine/diagnostic_observer.py` — defer cache write, add `flush_cache()`
2. `src/aquapose/engine/orchestrator.py` — remap in-place, call `flush_cache()`, simplify HDF5 write

## Verification

1. `hatch run test` — no regressions
2. Run 2-chunk diagnostic: `cd ~/aquapose/projects/YH && aquapose run --max-chunks 2 --add-observer diagnostic`
3. Inspect cache: verify chunk 0 and chunk 1 midlines_3d have consistent global fish IDs at the boundary (frame 299 vs frame 0)
4. Generate overlay: `aquapose viz --overlay` and confirm fish colors are stable across the chunk boundary
