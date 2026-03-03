# Phase 50: Cleanup and Replacement - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove the old evaluation machinery (monolithic NPZ export, midline fixture NPZ export, and standalone harness) from the codebase. The per-stage pickle cache system is now the sole evaluation data source. This is a deletion/cleanup phase — no new functionality.

</domain>

<decisions>
## Implementation Decisions

### NPZ removal scope
- Delete BOTH `export_pipeline_diagnostics` (monolithic NPZ) AND `export_midline_fixtures` (midline NPZ) from DiagnosticObserver
- Delete all 5 collector helper methods (`_collect_tracking_section`, `_collect_groups_section`, `_collect_correspondences_section`, `_collect_detection_counts_section`, `_collect_midline_counts_section`)
- Delete `_match_annotated_by_centroid`, `_build_projection_models`, and the module-level `_write_calib_arrays` helper
- Delete `io/midline_fixture.py` entirely (CalibBundle, load_midline_fixture, NPZ_VERSION)
- Keep `_on_pipeline_complete` as an empty no-op hook for future extensibility
- Remove the `NPZ_VERSION` import and `_NPZ_VERSION_V1` constant from DiagnosticObserver

### harness.py deletion
- Delete `evaluation/harness.py` entirely — `run_evaluation`, `generate_fixture`, and `EvalResults` are all superseded by EvalRunner + per-stage evaluators + TuningOrchestrator
- Audit `evaluation/metrics.py` and `evaluation/output.py` for orphaned code after harness deletion — remove anything that's now unreachable
- Delete `tests/unit/evaluation/test_harness.py` and any other test files whose only subject is deleted code

### Public API cleanup
- Clean break — remove `EvalResults`, `generate_fixture`, `run_evaluation` from `evaluation/__init__.py` and `__all__`. No deprecation shims.
- Remove `calibration_path` parameter from DiagnosticObserver constructor signature
- Full cleanup of `calibration_path` through the config hierarchy: observer_factory.py, config.py, and any YAML config references

### Claude's Discretion
- Order of deletions (which files to clean first)
- Whether to consolidate any surviving utility functions during cleanup
- Test file discovery for orphaned tests beyond test_harness.py

</decisions>

<specifics>
## Specific Ideas

No specific requirements — straightforward deletion-and-cleanup phase guided by the success criteria.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `StageSnapshot` dataclass: Survives cleanup — still used for in-memory stage capture
- `_write_stage_cache` method: Survives — pickle cache is the replacement for NPZ
- `EvalRunner` (runner.py): The replacement for `run_evaluation`
- `TuningOrchestrator` (tuning.py): The replacement for `generate_fixture` + sweep logic

### Established Patterns
- DiagnosticObserver pattern: on_event dispatches to handlers by event type. After cleanup, only StageComplete (snapshot + pickle) and PipelineStart (run_id capture) handlers remain active, plus empty PipelineComplete hook.
- evaluation/__init__.py: Re-exports from submodules with explicit `__all__`. Must be updated after deletions.

### Integration Points
- `observer_factory.py`: Constructs DiagnosticObserver — must remove calibration_path arg
- `engine/config.py`: May define calibration_path in observer config — must clean up
- `evaluation/__init__.py`: Must remove harness.py re-exports
- `io/__init__.py`: Must remove midline_fixture.py re-exports if any

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 50-cleanup-and-replacement*
*Context gathered: 2026-03-03*
