# Phase 46: Engine Primitives - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

The pipeline emits per-stage pickle cache files on each StageComplete event, and PosePipeline accepts a pre-populated context to skip upstream stages during sweeps. Requirements: INFRA-01, INFRA-02, INFRA-03, INFRA-04.

</domain>

<decisions>
## Implementation Decisions

### Cache file format & layout
- One pickle file per stage (e.g. `detection_cache.pkl`, `tracking_cache.pkl`) — not cumulative
- Each cache stores the **full PipelineContext snapshot** at that stage — any single cache is self-contained through that stage
- Files stored in `diagnostics/<stage>_cache.pkl` alongside existing outputs (pipeline_diagnostics.npz, midline_fixtures.npz)
- Cache files wrapped in a metadata envelope dict containing: run_id, timestamp, stage_name, version fingerprint, and the PipelineContext data

### Stage-skipping behavior
- PosePipeline.run() accepts an `initial_context` parameter
- Auto-detect which stages to skip by inspecting PipelineContext fields — if a stage's output fields are already populated (non-None), skip it
- Skipped stages still emit StageComplete events with `elapsed_seconds=0` and `summary={'skipped': True}` — observers see a complete timeline
- CarryForward state is included in the tracking stage cache — enables multi-batch resumption
- Strict validation on initial_context: verify that required upstream fields are populated before each stage runs; fail fast with a clear message if not

### Staleness detection
- Staleness detected via pickle load failure: catch AttributeError, ModuleNotFoundError, etc. during deserialization and raise StaleCacheError
- StaleCacheError message includes: cache file path, suggestion to re-run the pipeline, and the original exception for debugging
- StaleCacheError defined in `core/context.py` alongside PipelineContext and ContextLoader
- Basic shape validation after successful deserialization (e.g., frame_count == len(detections))

### Scope of caching trigger
- Pickle caching is automatic in diagnostic mode — DiagnosticObserver writes cache files whenever output_dir is set, no extra config flag needed
- Single-file loading only — since each cache is a full context snapshot, one file is sufficient to resume from any stage

### Loader API
- Standalone function `load_stage_cache(path) -> PipelineContext` in `core/context.py` — usable from scripts, notebooks, and pipeline
- CLI flag `--resume-from path/to/cache.pkl` on `aquapose run` that loads context and skips upstream stages

### Claude's Discretion
- Exact metadata envelope structure (dict keys, version fingerprint format)
- Internal implementation of field-based skip detection (mapping from stage class to output field names)
- Whether to log skipped stages at INFO or DEBUG level

</decisions>

<specifics>
## Specific Ideas

- The primary use case is parameter sweeps: run the full pipeline once in diagnostic mode, then reload from association or midline cache to sweep reconstruction parameters without re-running upstream stages
- Success criterion explicitly references `diagnostics/<stage>_cache.pkl` naming convention

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DiagnosticObserver` (`engine/diagnostic_observer.py`): Already captures StageSnapshot on each StageComplete event and writes artifacts on PipelineComplete. Natural place to add pickle serialization logic.
- `StageSnapshot` dataclass: Already mirrors PipelineContext fields — can be used as the serialization intermediary or bypassed in favor of direct PipelineContext pickling.
- `PipelineContext.get()` method: Existing field validation pattern (raises ValueError for None fields) — similar pattern for initial_context validation.

### Established Patterns
- Observer-driven artifact writing: DiagnosticObserver._on_pipeline_complete() already writes pipeline_diagnostics.npz and midline_fixtures.npz. Cache writing should follow the same pattern but trigger on StageComplete rather than PipelineComplete.
- Config serialization as first artifact: Pipeline writes config.yaml before any stage runs (ENG-08). Cache metadata should reference this for reproducibility.
- Stage name resolution: `type(stage).__name__` used throughout pipeline.py for stage identification.

### Integration Points
- `PosePipeline.run()` at line 120 of pipeline.py: Must add `initial_context` parameter and skip logic in the stage execution loop (lines 164-184)
- `DiagnosticObserver.on_event()` at line 136: Add pickle writing on StageComplete events
- `cli.py`: Add `--resume-from` flag to the `run` command
- `core/context.py`: Add StaleCacheError exception and load_stage_cache() function
- TrackingStage special case (pipeline.py:170-171): CarryForward must be serialized alongside context in the tracking cache

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 46-engine-primitives*
*Context gathered: 2026-03-03*
