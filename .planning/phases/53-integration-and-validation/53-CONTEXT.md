# Phase 53: Integration and Validation - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire ChunkOrchestrator into `aquapose run` as the universal production path and validate output equivalence against non-chunked baseline. Delete redundant HDF5Observer. Add `--max-chunks` CLI flag for early stopping.

</domain>

<decisions>
## Implementation Decisions

### CLI dispatch strategy
- ChunkOrchestrator is the universal code path for `aquapose run` — chunk_size=null is the degenerate single-chunk case
- CLI creates PipelineConfig and hands everything to ChunkOrchestrator (config-only handoff); orchestrator handles stage/observer construction internally
- Drop `--resume-from` CLI flag but keep `load_stage_cache` and underlying machinery for programmatic use (evaluation sweeps)
- Keep `--stop-after` flag — ChunkOrchestrator passes it through to PosePipeline

### Output file unification
- `midlines.h5` via Midline3DWriter is the single canonical HDF5 output path
- Delete HDF5ExportObserver entirely (hdf5_observer.py, its tests, factory registration, all references)
- No downstream consumers reference either `outputs.h5` or `midlines.h5` in evaluation/tuning code — deletion is clean

### Validation test approach
- E2E validation with real data (YH project), human-reviewed comparison rather than automated assertion
- Compare 2×100-frame chunks vs 1×200-frame single chunk — association differences across chunk boundaries are expected and non-trivial
- Add `--max-chunks` as a CLI-only flag (not in PipelineConfig) passed directly to ChunkOrchestrator constructor
- `--max-chunks 1` with `chunk_size=200` is equivalent to the old `stop_frame=200` behavior

### Mode conflict handling
- Validate in ChunkOrchestrator constructor (fail-fast before I/O or model loading)
- Error condition: `chunk_size > 0 AND (max_chunks is None or max_chunks > 1) AND mode == diagnostic`
- Allowed: diagnostic + chunk_size=null (degenerate single chunk), diagnostic + max_chunks=1 (single chunk early stop)
- Error message includes suggestion: "Chunk mode and diagnostic mode are mutually exclusive. Use chunk_size=null for diagnostic runs on short clips, or set --max-chunks 1."

### Claude's Discretion
- Exact refactoring of cli.py internals to achieve config-only handoff
- How to wire --stop-after through orchestrator to pipeline
- Test structure for the HDF5Observer deletion (remove test file vs keep stubs)
- Whether to add a unit test for the diagnostic mode conflict validation

</decisions>

<specifics>
## Specific Ideas

- `--max-chunks` replaces the old `stop_frame` concept at the orchestrator level — cleaner separation of concerns
- Validation is a human check because cross-view association over 200 frames vs two 100-frame chunks with stitching produces legitimately different groupings — this is expected, not a bug

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ChunkOrchestrator` (engine/orchestrator.py): Already handles chunk loop, identity stitching, HDF5 writing, observer filtering — most of the work is done
- `Midline3DWriter` (io/midline_writer.py): Already handles per-chunk HDF5 append with global frame offset
- `build_observers()` (engine/observer_factory.py): Orchestrator already calls this and filters out HDF5ExportObserver
- `load_stage_cache` (core/context.py): Keep for programmatic use even though CLI flag is dropped

### Established Patterns
- Config validation: chunk_size < 100 warning already in engine/config.py (lines 675-677)
- Observer filtering: Orchestrator already removes HDF5ExportObserver and optionally ConsoleObserver
- CLI uses click decorators with --set for config overrides

### Integration Points
- cli.py:run() (lines 82-171): Main integration point — currently builds stages/observers/pipeline directly, needs to delegate to ChunkOrchestrator
- engine/__init__.py: Already exports ChunkOrchestrator, ChunkHandoff, write_handoff
- observer_factory.py: HDF5ExportObserver registration to be removed

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 53-integration-and-validation*
*Context gathered: 2026-03-03*
