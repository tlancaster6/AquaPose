---
phase: 26-global-id-remap-for-diagnostic-cache
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/engine/diagnostic_observer.py
  - src/aquapose/engine/orchestrator.py
  - tests/unit/engine/test_diagnostic_observer.py
  - tests/unit/engine/test_chunk_orchestrator.py
autonomous: true
must_haves:
  truths:
    - "Diagnostic cache.pkl contains globally-remapped fish IDs, not chunk-local IDs"
    - "Fish colors in overlay visualization are stable across chunk boundaries"
    - "HDF5 output remains correct (already was, just simplified)"
  artifacts:
    - path: "src/aquapose/engine/diagnostic_observer.py"
      provides: "flush_cache() method for deferred cache write"
    - path: "src/aquapose/engine/orchestrator.py"
      provides: "In-place context remapping + observer flush after stitching"
  key_links:
    - from: "src/aquapose/engine/orchestrator.py"
      to: "src/aquapose/engine/diagnostic_observer.py"
      via: "isinstance check + flush_cache() call after identity remap"
      pattern: "flush_cache"
---

<objective>
Defer diagnostic cache write until after identity stitching so cache.pkl contains globally-consistent fish IDs instead of chunk-local IDs. Currently DiagnosticObserver writes on PipelineComplete (inside pipeline.run()), before the orchestrator remaps local-to-global fish IDs. This causes fish color changes at chunk boundaries in overlay visualizations.

Purpose: Fix fish identity discontinuity in diagnostic cache at chunk boundaries.
Output: Modified diagnostic_observer.py and orchestrator.py with deferred write + in-place remap.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/engine/diagnostic_observer.py
@src/aquapose/engine/orchestrator.py
@src/aquapose/core/association/types.py (TrackletGroup — frozen=True, needs dataclasses.replace)
@src/aquapose/core/types/reconstruction.py (Midline3D — not frozen, can mutate fish_id directly)
@tests/unit/engine/test_diagnostic_observer.py
@tests/unit/engine/test_chunk_orchestrator.py
@.planning/inbox/global_id_remap_diagnostic_cache.md

<interfaces>
From src/aquapose/core/association/types.py:
```python
@dataclass(frozen=True)
class TrackletGroup:
    fish_id: int
    tracklets: tuple[Tracklet2D, ...]
    # ... other fields
```

From src/aquapose/core/types/reconstruction.py:
```python
@dataclass
class Midline3D:
    fish_id: int
    frame_index: int
    # ... other fields (not frozen, fish_id is mutable)
```

From src/aquapose/engine/diagnostic_observer.py:
```python
class DiagnosticObserver:
    def _write_chunk_cache(self, context: object) -> None: ...
    def _write_manifest(self, context: object) -> None: ...
    def _on_pipeline_complete(self, event: PipelineComplete) -> None: ...
    # _last_context already tracked via StageComplete handler
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Defer DiagnosticObserver cache write and add flush_cache()</name>
  <files>src/aquapose/engine/diagnostic_observer.py, tests/unit/engine/test_diagnostic_observer.py</files>
  <action>
In `diagnostic_observer.py`:

1. Modify `_on_pipeline_complete()` (lines 271-286): Remove the calls to `_write_chunk_cache(context)` and `_write_manifest(context)`. Keep the output_dir guard and the context resolution logic (event context or _last_context fallback), but instead of writing, just store the resolved context: `self._last_context = context`. The method becomes a no-op for file I/O — it only updates the context reference.

2. Add a public `flush_cache()` method:
```python
def flush_cache(self) -> None:
    """Write chunk cache and manifest using the stored context.

    Called by the orchestrator after identity stitching has remapped
    fish IDs in the context. No-op if output_dir is None or no context
    is available.
    """
    if self._output_dir is None:
        return
    context = self._last_context
    if context is None:
        return
    self._write_chunk_cache(context)
    self._write_manifest(context)
```

In `test_diagnostic_observer.py`:

3. Update tests that expect cache/manifest to be written on PipelineComplete. The key tests affected:
   - `test_chunk_000_cache_written`: After `_fire_pipeline(observer)`, cache should NOT exist yet. Call `observer.flush_cache()` then assert it exists.
   - `test_chunk_002_cache_written`: Same pattern — add `observer.flush_cache()` before assert.
   - `test_chunk_cache_contains_full_context`: Add `observer.flush_cache()` before reading pickle.
   - `test_chunk_cache_envelope_schema`: Add `observer.flush_cache()` before reading pickle.
   - `test_only_one_cache_per_chunk_written_on_pipeline_complete`: After PipelineComplete, assert NO cache (deferred). Then call `flush_cache()` and assert exactly one cache.pkl.
   - `test_no_per_stage_cache_files_written`: Add `observer.flush_cache()` before glob.
   - `test_inmemory_stages_dict_still_works`: No change needed (tests in-memory dict, not disk).
   - `test_manifest_written_on_pipeline_complete`: Add `observer.flush_cache()` before assert.
   - `test_manifest_schema`: Add `observer.flush_cache()` before reading manifest.
   - `test_manifest_chunk_entry_schema`: Add `observer.flush_cache()` before reading manifest.
   - `test_manifest_appended_for_multiple_chunks`: Add `observer.flush_cache()` after each `_fire_pipeline` call.
   - `test_load_chunk_cache_loads_new_format`: Add `observer.flush_cache()` before load.
   - `test_load_stage_cache_still_works_with_new_format`: Add `observer.flush_cache()` before load.

4. Add a new test `test_pipeline_complete_does_not_write_cache` that fires the full pipeline event sequence WITHOUT calling flush_cache() and asserts no cache.pkl or manifest.json exist on disk. This validates the deferral.

5. Add a new test `test_flush_cache_writes_cache_and_manifest` that fires pipeline events, calls flush_cache(), and asserts both cache.pkl and manifest.json are created.

6. Update the `_fire_pipeline` helper's return type — it should also return the observer for convenience, OR callers can just call flush_cache on the observer they already have (current pattern is fine, callers hold observer reference).

Also update `test_manifest_start_frame` in `test_chunk_orchestrator.py`: add `observer.flush_cache()` after the PipelineComplete event and before checking manifest.json.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test -- tests/unit/engine/test_diagnostic_observer.py tests/unit/engine/test_chunk_orchestrator.py::test_manifest_start_frame -x</automated>
  </verify>
  <done>DiagnosticObserver no longer writes on PipelineComplete. flush_cache() is the only way to trigger disk writes. All existing tests updated and passing plus new deferral test.</done>
</task>

<task type="auto">
  <name>Task 2: Remap context in-place in orchestrator and flush observer</name>
  <files>src/aquapose/engine/orchestrator.py, tests/unit/engine/test_chunk_orchestrator.py</files>
  <action>
In `orchestrator.py`, in `ChunkOrchestrator.run()`, after `_stitch_identities()` returns `identity_map` (line ~320):

1. **Remap `context.midlines_3d` in-place** (insert before the HDF5 write loop at line 330):
```python
# Remap midlines_3d fish IDs: local -> global
for frame_midlines in midlines_3d:
    remapped_frame: dict[int, object] = {}
    for lid, ml in frame_midlines.items():
        global_id = identity_map.get(int(lid), int(lid))
        ml.fish_id = global_id  # Midline3D is not frozen
        remapped_frame[global_id] = ml
    frame_midlines.clear()
    frame_midlines.update(remapped_frame)
```

2. **Remap `context.tracklet_groups` in-place** (insert after midlines remap):
```python
# Remap tracklet_groups fish IDs: local -> global
from dataclasses import replace as dc_replace
context.tracklet_groups = [
    dc_replace(group, fish_id=identity_map.get(group.fish_id, group.fish_id))
    for group in tracklet_groups
]
```
Note: `dataclasses.replace` import can go at top of the method or use inline import. Since it's stdlib, a top-of-file import is fine — add `from dataclasses import replace as dc_replace` at the module level (after existing imports).

3. **Simplify HDF5 write loop** (lines 330-336): Replace the existing remap-and-write loop with direct iteration since midlines_3d is already remapped:
```python
for local_idx, frame_midlines in enumerate(midlines_3d):
    global_frame_idx = chunk_start + local_idx
    writer.write_frame(global_frame_idx, frame_midlines)
```

4. **Flush DiagnosticObserver** (insert after HDF5 write, before handoff building):
```python
# Flush diagnostic cache with globally-remapped IDs
from aquapose.engine.diagnostic_observer import DiagnosticObserver
for obs in observers:
    if isinstance(obs, DiagnosticObserver):
        obs.flush_cache()
```
Use a direct import inside the loop body (not TYPE_CHECKING) since this runs once per chunk, not in a hot path. This avoids any circular import concerns at module level.

5. **Simplify handoff building** (lines 342-347): Since tracklet_groups are already remapped, replace:
```python
gid: int = identity_map.get(group.fish_id, group.fish_id)
```
with just:
```python
gid = group.fish_id
```

In `test_chunk_orchestrator.py`:

6. Add a test `test_diagnostic_observer_flushed_after_remap` that:
   - Creates a ChunkOrchestrator with a mock pipeline returning a context with midlines_3d and tracklet_groups
   - Patches `build_observers` to return a DiagnosticObserver(output_dir=tmp_path)
   - Runs the orchestrator
   - Loads cache.pkl and verifies the context inside has global fish IDs (not local)
   This is the integration test proving the full flow works.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test -- tests/unit/engine/test_chunk_orchestrator.py tests/unit/engine/test_diagnostic_observer.py -x</automated>
  </verify>
  <done>Orchestrator remaps context.midlines_3d and context.tracklet_groups in-place with global IDs before writing HDF5 or flushing diagnostic cache. HDF5 write loop simplified (no redundant remap). DiagnosticObserver.flush_cache() called after remap. All tests pass.</done>
</task>

</tasks>

<verification>
```bash
# Full test suite — no regressions
cd /home/tlancaster6/Projects/AquaPose && hatch run test

# Type check
cd /home/tlancaster6/Projects/AquaPose && hatch run typecheck
```
</verification>

<success_criteria>
- DiagnosticObserver.flush_cache() is the only path to write cache.pkl and manifest.json
- PipelineComplete event no longer triggers disk writes
- Orchestrator remaps context.midlines_3d dict keys and Midline3D.fish_id to global IDs
- Orchestrator remaps context.tracklet_groups fish_id to global IDs via dataclasses.replace
- HDF5 write loop uses already-remapped midlines (no double remap)
- Handoff building uses already-remapped group.fish_id
- DiagnosticObserver.flush_cache() called after remap in orchestrator loop
- All existing tests updated and passing, new integration test validates end-to-end
</success_criteria>

<output>
After completion, create `.planning/quick/26-global-id-remap-for-diagnostic-cache/26-01-SUMMARY.md`
</output>
