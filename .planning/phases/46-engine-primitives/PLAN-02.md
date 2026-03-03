# Plan 46.2: Per-Stage Pickle Cache Writing and Stage-Skip Logic

---
wave: 2
depends_on:
  - PLAN-01
files_modified:
  - src/aquapose/engine/diagnostic_observer.py
  - src/aquapose/engine/pipeline.py
  - tests/unit/engine/test_stage_cache_write.py
  - tests/unit/engine/test_stage_skip.py
requirements:
  - INFRA-01
  - INFRA-02
autonomous: true
---

## Goal

DiagnosticObserver writes per-stage pickle cache files on each StageComplete event (INFRA-01), and PosePipeline.run() accepts a pre-populated `initial_context` to skip stages whose outputs are already populated (INFRA-02).

## Locked Decisions (from CONTEXT.md)

- Pickle caching is automatic in diagnostic mode -- DiagnosticObserver writes cache files whenever `output_dir` is set, no extra config flag needed
- Each cache stores the **full PipelineContext snapshot** at that stage -- any single cache is self-contained through that stage
- Cache files wrapped in a metadata envelope dict containing: `run_id`, `timestamp`, `stage_name`, `version_fingerprint`, and the PipelineContext data
- Files stored in `diagnostics/<stage>_cache.pkl` alongside existing outputs
- PosePipeline.run() accepts an `initial_context` parameter
- Auto-detect which stages to skip by inspecting PipelineContext fields -- if a stage's output fields are already populated (non-None), skip it
- Skipped stages still emit StageComplete events with `elapsed_seconds=0` and `summary={'skipped': True}` -- observers see a complete timeline
- CarryForward state is included in the tracking stage cache via `context.carry_forward`
- Strict validation on initial_context: verify that required upstream fields are populated before each stage runs; fail fast with a clear message if not. **Only when initial_context is provided** (Pitfall 2 from RESEARCH.md)
- Store `_run_id` from PipelineStart event in DiagnosticObserver.on_event() -- avoids changing `__init__` signature

## Tasks

<task id="46.2.1">
<title>Add pickle cache writing to DiagnosticObserver</title>
<details>

**File:** `src/aquapose/engine/diagnostic_observer.py`

1. Add imports at top of file:
   ```python
   import pickle
   from aquapose.core.context import context_fingerprint
   ```

2. Add a `_run_id: str` instance variable to `__init__`, initialized to `""`.

3. In `on_event()`, capture run_id from PipelineStart:
   ```python
   if isinstance(event, PipelineStart):
       self._run_id = event.run_id
       return
   ```
   This must be added **before** the existing `PipelineComplete` check. Import `PipelineStart` from `aquapose.engine.events`.

4. In `on_event()`, after the existing snapshot capture for StageComplete (after `self.stages[event.stage_name] = snapshot`), add the cache write:
   ```python
   if self._output_dir is not None:
       self._write_stage_cache(event, context)
   ```

5. Implement `_write_stage_cache()` as a private method:
   ```python
   def _write_stage_cache(self, event: StageComplete, context: object) -> None:
       """Write a pickle cache file for the completed stage."""
       import datetime

       diagnostics_dir = self._output_dir / "diagnostics"
       diagnostics_dir.mkdir(parents=True, exist_ok=True)

       # Normalize stage name for filename: "DetectionStage" -> "detection"
       stage_key = event.stage_name.removesuffix("Stage").lower()
       cache_path = diagnostics_dir / f"{stage_key}_cache.pkl"

       envelope = {
           "run_id": self._run_id,
           "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
           "stage_name": event.stage_name,
           "version_fingerprint": context_fingerprint(context),
           "context": context,
       }

       cache_path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
       logger.info("Stage cache written: %s", cache_path)
   ```

   **Note on stage_key mapping:** The CONTEXT.md says `detection_cache.pkl`, `tracking_cache.pkl`, etc. Use `stage_name.removesuffix("Stage").lower()` to derive the filename key (e.g., `DetectionStage` -> `detection`, `SyntheticDataStage` -> `syntheticdata`). This is Claude's discretion per CONTEXT.md.

</details>
</task>

<task id="46.2.2">
<title>Add initial_context and stage-skip logic to PosePipeline.run()</title>
<details>

**File:** `src/aquapose/engine/pipeline.py`

1. Define the stage-to-output-fields mapping as a module-level constant (in `pipeline.py`, per RESEARCH.md Open Question 3):
   ```python
   _STAGE_OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
       "DetectionStage": ("frame_count", "camera_ids", "detections"),
       "SyntheticDataStage": ("frame_count", "camera_ids", "detections", "annotated_detections"),
       "TrackingStage": ("tracks_2d",),
       "AssociationStage": ("tracklet_groups",),
       "MidlineStage": ("annotated_detections",),
       "ReconstructionStage": ("midlines_3d",),
   }
   ```

2. Modify `PosePipeline.run()` signature to accept `initial_context`:
   ```python
   def run(self, initial_context: PipelineContext | None = None) -> PipelineContext:
   ```

3. Change the context initialization (currently `context = PipelineContext()`) to:
   ```python
   context = initial_context if initial_context is not None else PipelineContext()
   ```

4. If `initial_context` is provided, extract carry from it:
   ```python
   carry: CarryForward | None = None
   if initial_context is not None:
       carry = initial_context.carry_forward
   ```
   (replaces the existing `carry: CarryForward | None = None` line)

5. In the stage loop, **before** the existing `StageStart` emit, add skip detection:
   ```python
   stage_name = type(stage).__name__
   output_fields = _STAGE_OUTPUT_FIELDS.get(stage_name, ())
   already_populated = (
       bool(output_fields)
       and all(getattr(context, f, None) is not None for f in output_fields)
   )

   if already_populated:
       logger.info("Skipping %s -- outputs already populated in context", stage_name)
       self._bus.emit(StageStart(stage_name=stage_name, stage_index=i))
       self._bus.emit(
           StageComplete(
               stage_name=stage_name,
               stage_index=i,
               elapsed_seconds=0.0,
               summary={"skipped": True},
               context=context,
           ),
       )
       continue
   ```

6. After TrackingStage runs (existing `context, carry = stage.run(context, carry)` line), inject carry into context before StageComplete emission:
   ```python
   if isinstance(stage, TrackingStage):
       context, carry = stage.run(context, carry)
       context.carry_forward = carry
   ```

7. Update the docstring of `run()` to document the `initial_context` parameter and stage-skip behavior.

</details>
</task>

<task id="46.2.3">
<title>Write unit tests for cache writing and stage-skip logic</title>
<details>

**File:** `tests/unit/engine/test_stage_cache_write.py` (new file)

Tests for DiagnosticObserver cache writing:

1. **test_diagnostic_observer_writes_cache_on_stage_complete**: Create a `DiagnosticObserver(output_dir=tmp_path)`. Send it a `PipelineStart` event with `run_id="test_run"`. Then send a `StageComplete` event with `stage_name="DetectionStage"`, a mock PipelineContext with `frame_count=5`, `detections=[{} for _ in range(5)]`. Assert that `tmp_path / "diagnostics" / "detection_cache.pkl"` exists. Load the pickle file and verify the envelope keys (`run_id`, `timestamp`, `stage_name`, `version_fingerprint`, `context`). Verify `envelope["run_id"] == "test_run"` and `envelope["stage_name"] == "DetectionStage"`.

2. **test_diagnostic_observer_no_cache_without_output_dir**: Create a `DiagnosticObserver()` (no output_dir). Send StageComplete. Assert no files written (no exceptions, no side effects).

3. **test_diagnostic_observer_cache_round_trips_with_load_stage_cache**: Write a cache via DiagnosticObserver, then load it with `load_stage_cache()`. Assert the loaded context matches the original.

4. **test_diagnostic_observer_captures_run_id_from_pipeline_start**: Send PipelineStart with run_id="abc", then StageComplete. Load the cache and verify `envelope["run_id"] == "abc"`.

**File:** `tests/unit/engine/test_stage_skip.py` (new file)

Tests for PosePipeline stage-skip logic:

1. **test_pipeline_skips_populated_stages**: Create a minimal PosePipeline with 2 stub stages (StubStageA populates `detections`, StubStageB populates `tracks_2d`). Create a PipelineContext with `detections` already populated. Run with `initial_context`. Assert StubStageA's `run()` was NOT called, and StubStageB's `run()` WAS called.

2. **test_pipeline_emits_skipped_stage_complete**: Same setup as above. Collect events via a test observer. Assert StageComplete for the skipped stage has `summary={"skipped": True}` and `elapsed_seconds == 0.0`.

3. **test_pipeline_no_skip_without_initial_context**: Run pipeline normally (no initial_context). Assert all stages execute.

4. **test_pipeline_extracts_carry_from_initial_context**: Create a PipelineContext with `carry_forward=CarryForward(tracks_2d_state={"cam1": {}})`. Verify the carry is available to the pipeline (this validates the carry extraction logic).

5. **test_carry_forward_injected_after_tracking_stage**: Create a pipeline with a stub TrackingStage. Run the pipeline. Assert `context.carry_forward` is not None after TrackingStage runs.

For stub stages, use simple classes that satisfy the Stage protocol:
```python
class StubDetection:
    def __init__(self): self.called = False
    def run(self, context):
        self.called = True
        context.frame_count = 5
        context.camera_ids = ["cam1"]
        context.detections = [{} for _ in range(5)]
        return context
```

These tests must NOT import from GPU/data-dependent modules. Use only stub stages.

All tests must be runnable with `hatch run test`.

Run `hatch run test` after writing tests to verify they pass.

</details>
</task>

## Verification

```bash
hatch run test tests/unit/engine/test_stage_cache_write.py tests/unit/engine/test_stage_skip.py -v
hatch run check
```

- [ ] `diagnostics/<stage>_cache.pkl` files are written by DiagnosticObserver on StageComplete when output_dir is set
- [ ] Cache envelope contains run_id, timestamp, stage_name, version_fingerprint, context
- [ ] No cache files are written when output_dir is None
- [ ] `PosePipeline.run(initial_context=ctx)` skips stages whose output fields are non-None
- [ ] Skipped stages emit StageComplete with `{"skipped": True}` and `elapsed_seconds=0.0`
- [ ] `carry_forward` is injected into context after TrackingStage runs
- [ ] Initial carry is extracted from `initial_context.carry_forward`
- [ ] All unit tests pass
- [ ] `hatch run check` passes

## must_haves

- DiagnosticObserver writes `diagnostics/<stage>_cache.pkl` on each StageComplete when output_dir is set
- Cache envelope includes run_id, timestamp, stage_name, version_fingerprint, context
- PosePipeline.run() accepts `initial_context: PipelineContext | None = None`
- Stage-skip logic auto-detects populated fields and skips stages
- Skipped stages emit StageComplete with `{"skipped": True}` and `elapsed_seconds=0.0`
- TrackingStage carry_forward is stored on PipelineContext after run
- No cache files written in non-diagnostic mode (output_dir is None)
