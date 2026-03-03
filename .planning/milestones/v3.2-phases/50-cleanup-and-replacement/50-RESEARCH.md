# Phase 50: Cleanup and Replacement - Research

**Researched:** 2026-03-03
**Domain:** Python module deletion, test cleanup, public API pruning
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**NPZ removal scope:**
- Delete BOTH `export_pipeline_diagnostics` (monolithic NPZ) AND `export_midline_fixtures` (midline NPZ) from DiagnosticObserver
- Delete all 5 collector helper methods (`_collect_tracking_section`, `_collect_groups_section`, `_collect_correspondences_section`, `_collect_detection_counts_section`, `_collect_midline_counts_section`)
- Delete `_match_annotated_by_centroid`, `_build_projection_models`, and the module-level `_write_calib_arrays` helper
- Delete `io/midline_fixture.py` entirely (CalibBundle, load_midline_fixture, NPZ_VERSION)
- Keep `_on_pipeline_complete` as an empty no-op hook for future extensibility
- Remove the `NPZ_VERSION` import and `_NPZ_VERSION_V1` constant from DiagnosticObserver

**harness.py deletion:**
- Delete `evaluation/harness.py` entirely — `run_evaluation`, `generate_fixture`, and `EvalResults` are all superseded by EvalRunner + per-stage evaluators + TuningOrchestrator
- Audit `evaluation/metrics.py` and `evaluation/output.py` for orphaned code after harness deletion — remove anything that's now unreachable
- Delete `tests/unit/evaluation/test_harness.py` and any other test files whose only subject is deleted code

**Public API cleanup:**
- Clean break — remove `EvalResults`, `generate_fixture`, `run_evaluation` from `evaluation/__init__.py` and `__all__`. No deprecation shims.
- Remove `calibration_path` parameter from DiagnosticObserver constructor signature
- Full cleanup of `calibration_path` through the config hierarchy: observer_factory.py, config.py, and any YAML config references

### Claude's Discretion

- Order of deletions (which files to clean first)
- Whether to consolidate any surviving utility functions during cleanup
- Test file discovery for orphaned tests beyond test_harness.py

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLEAN-04 | Monolithic `pipeline_diagnostics.npz` machinery removed or fully integrated into per-stage cache system | Delete `export_pipeline_diagnostics`, its 5 collector helpers, `_write_calib_arrays`, and auto-export call in `_on_pipeline_complete`. The per-stage pickle cache (`_write_stage_cache`) is the sole replacement. |
| CLEAN-05 | `evaluation/harness.py` removed — functionality consolidated into reconstruction stage evaluator | Delete harness.py, then audit and prune orphaned code from metrics.py and output.py. The reconstruction evaluator in `stages/reconstruction.py` already has `compute_tier1` and `Tier2Result` consumption via its own `evaluate_reconstruction()` function. |
</phase_requirements>

## Summary

Phase 50 is a pure deletion and cleanup phase. The new evaluation system (per-stage pickle caches, EvalRunner, TuningOrchestrator, per-stage evaluators) was built in Phases 46–49 and is already in use. This phase removes the old system's code and tests.

The work divides naturally into three clusters: (1) DiagnosticObserver surgery — remove all NPZ export machinery while preserving StageSnapshot capture and pickle cache writing; (2) harness.py deletion with downstream orphan removal from metrics.py, output.py, and their tests; (3) public API surface cleanup across `evaluation/__init__.py`, `io/__init__.py`, and the DiagnosticObserver constructor.

The primary risk is unintentional breakage: some symbols from the deleted modules (`Tier1Result`, `Tier2Result`, `select_frames`, `compute_tier1`) are still in active use by `stages/reconstruction.py`, `runner.py`, `tuning.py`, and `output.py`. These must be retained. Only the subset of functions exclusively consumed by `harness.py` (`compute_tier2`, `format_summary_table`, `write_regression_json`) and the NPZ-specific helpers become truly orphaned.

**Primary recommendation:** Delete files and methods in dependency order (harness first, then its callers in __init__.py, then io/midline_fixture.py, then DiagnosticObserver NPZ methods), then run `hatch run test` after each deletion cluster to catch breakage early.

## Standard Stack

### Core

No new libraries needed. All work uses:

| Component | Version | Purpose |
|-----------|---------|---------|
| Python `pathlib` / `os` | stdlib | File deletion |
| `hatch run test` | project standard | Test suite validation after each change |
| `hatch run check` | project standard | Lint + typecheck gate |

This phase does not introduce any new dependencies.

## Architecture Patterns

### What Survives in DiagnosticObserver

After all deletions, `DiagnosticObserver` retains exactly these elements:

```python
class DiagnosticObserver:
    def __init__(self, output_dir: str | Path | None = None) -> None:
        # calibration_path parameter REMOVED
        self.stages: dict[str, StageSnapshot] = {}
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._run_id: str = ""

    def on_event(self, event: Event) -> None:
        # PipelineStart handler: capture run_id
        # PipelineComplete handler: call _on_pipeline_complete() (no-op)
        # StageComplete handler: build StageSnapshot + call _write_stage_cache()

    def _write_stage_cache(self, event: StageComplete, context: object) -> None:
        # SURVIVES: pickle envelope writer

    def _on_pipeline_complete(self) -> None:
        # SURVIVES but becomes no-op (body cleared, docstring updated)
```

**Removed from DiagnosticObserver:**
- `calibration_path` constructor parameter and `self._calibration_path` attribute
- `_NPZ_VERSION_V1` constant and `_NPZ_VERSION_LATEST` import (`from aquapose.io.midline_fixture import NPZ_VERSION as _NPZ_VERSION_LATEST`)
- `export_pipeline_diagnostics()` method
- `export_midline_fixtures()` method
- `_match_annotated_by_centroid()` static method
- `_build_projection_models()` method
- `_collect_tracking_section()`, `_collect_groups_section()`, `_collect_correspondences_section()`, `_collect_detection_counts_section()`, `_collect_midline_counts_section()` methods
- Module-level `_write_calib_arrays()` function
- Constants: `_CENTROID_MATCH_TOLERANCE_PX`, `_DETECTION_STAGE_NAME` (check — only used by deleted methods?), `_PER_FRAME_FIELDS`, `_SCALAR_FIELDS`

**Caution on constants:** `_DETECTION_STAGE_NAME`, `_TRACKING_STAGE_NAME`, `_ASSOCIATION_STAGE_NAME`, `_MIDLINE_STAGE_NAME` are used by the deleted collector methods AND by `_build_projection_models`/`export_midline_fixtures`. After deletion, check whether any of these constants remain referenced by surviving code. The `_PER_FRAME_FIELDS` tuple is used by `StageSnapshot.__getitem__` so it must be kept. `_SCALAR_FIELDS` appears only in docstrings/comments and can be removed.

### What Survives in metrics.py

**Keep:** `Tier1Result`, `Tier2Result`, `select_frames`, `compute_tier1`
- `stages/reconstruction.py` imports `Tier2Result` and `compute_tier1`
- `runner.py` and `tuning.py` import `select_frames`
- `output.py` imports `Tier1Result` and `Tier2Result`
- `evaluation/__init__.py` re-exports all of these

**Delete:** `compute_tier2`
- Only caller is `harness.py` (being deleted)
- No other src file imports it
- Its test coverage is in `test_metrics.py`

### What Survives in output.py

**Keep:** `flag_outliers`, `format_baseline_report`, `format_eval_report`, `format_eval_json`, `_NumpySafeEncoder`
- `format_eval_report` and `format_eval_json` are used by the CLI (`eval` command)
- `flag_outliers` and `format_baseline_report` are tested and still in `__all__`

**Delete:** `format_summary_table`, `write_regression_json`
- Only callers are `harness.py` (being deleted)
- After harness deletion these are unreachable from production code
- Their tests are in `test_output.py` (covering both harness-era and EvalRunner-era output)

**Note:** `test_output.py` tests both the old (`format_summary_table`, `write_regression_json`) and new (`format_eval_report`, `format_eval_json`, `flag_outliers`, `format_baseline_report`) functions. Only the old function tests become dead after cleanup. The file itself must NOT be deleted — it retains value for the surviving functions. Delete only the tests for `format_summary_table` and `write_regression_json` from `test_output.py`.

**Separately check:** `test_eval_output.py` in the same directory — read it to determine if it covers only new or mixed functions.

### What Survives in __init__.py (evaluation)

**Remove from `evaluation/__init__.py`:**
- `from aquapose.evaluation.harness import EvalResults, generate_fixture, run_evaluation`
- `"EvalResults"`, `"generate_fixture"`, `"run_evaluation"` from `__all__`

**Remove from `evaluation/__init__.py` (orphaned metrics):**
- `"Tier1Result"`, `"Tier2Result"` — these are still imported by `output.py` and used internally, but check if they should remain in the public API. The user said clean break on `harness` exports; `Tier1Result`/`Tier2Result`/`select_frames` may still be worth keeping in `__all__` since `stages/reconstruction.py` uses them. **Decision (Claude's discretion):** Keep `Tier1Result`, `Tier2Result`, and `select_frames` in `__init__.py` and `__all__` — they are legitimately public types still used by the reconstruction evaluator.

**Remove from `evaluation/__init__.py` after orphan audit:**
- `"select_frames"` — still used internally; decide whether to keep in public API. Recommendation: keep.
- Remove `format_summary_table`, `write_regression_json` from `output.py` imports if they were re-exported (check current `__init__.py` — they are not currently in `__all__`, so no change needed there).

### What Survives in io/__init__.py

**Remove from `io/__init__.py`:**
- `from .midline_fixture import NPZ_VERSION, CalibBundle, MidlineFixture, load_midline_fixture`
- All four names from `__all__`

### observer_factory.py Changes

In `build_observers()`, the `DiagnosticObserver` construction currently passes `calibration_path=config.calibration_path`. After removing `calibration_path` from the constructor:

```python
# Before:
DiagnosticObserver(
    output_dir=config.output_dir, calibration_path=config.calibration_path
)

# After:
DiagnosticObserver(output_dir=config.output_dir)
```

This appears in two places in `observer_factory.py`:
1. In the `mode == "diagnostic"` branch
2. In the additive `--add-observer` `DiagnosticObserver` construction

### config.py Scope — calibration_path NOT removed

The `calibration_path` in `PipelineConfig` is not being removed. It is used by stages (DetectionStage, MidlineStage, AssociationStage, ReconstructionStage) as a core pipeline parameter. Only the `DiagnosticObserver` constructor stops accepting it.

## Test Files to Delete Entirely

| File | Why Delete |
|------|-----------|
| `tests/unit/evaluation/test_harness.py` | Only subject is `harness.py` which is being deleted |
| `tests/unit/io/test_midline_fixture.py` | Only subject is `io/midline_fixture.py` which is being deleted |

## Test Files to Partially Edit

| File | What to Remove |
|------|---------------|
| `tests/unit/engine/test_diagnostic_observer.py` | All tests for `export_pipeline_diagnostics`, `export_midline_fixtures`, their helpers, and the `test_on_pipeline_complete_exports_midline_fixtures` test. Keep tests for `StageSnapshot`, `on_event`, `_write_stage_cache`, and the surviving observer behavior. |
| `tests/unit/evaluation/test_output.py` | Tests for `format_summary_table` and `write_regression_json`. Keep tests for `flag_outliers`, `format_baseline_report`, `format_eval_report`, `format_eval_json`. |
| `tests/unit/evaluation/test_metrics.py` | Tests for `compute_tier2`. Keep tests for `select_frames`, `compute_tier1`, `Tier1Result`, `Tier2Result`. |

**Also investigate:** `tests/unit/evaluation/test_eval_output.py` — determine if this file covers only new functions or contains legacy test subjects.

## Test Files That Survive Unchanged

All other test files survive unchanged. The `calibration_path` references in other test files (e.g., `test_dlt_backend.py`, `test_reconstruction_stage.py`, `test_synthetic.py`) refer to the stage-level `calibration_path` parameter which is not being removed.

## Don't Hand-Roll

| Problem | Use Instead |
|---------|-------------|
| Finding all test lines to delete | Read the test file, identify ranges by function name, edit with precision |
| Checking for remaining references after deletion | `grep -rn <symbol> src/ tests/ --include="*.py"` |

## Common Pitfalls

### Pitfall 1: Removing calibration_path from PipelineConfig
**What goes wrong:** The context decisions mention "remove `calibration_path` parameter from DiagnosticObserver constructor" and "config hierarchy: observer_factory.py, config.py". This does NOT mean removing it from `PipelineConfig.calibration_path` — that field is a core pipeline parameter used by all stages.
**How to avoid:** Only remove `calibration_path` from `DiagnosticObserver.__init__` and the two `DiagnosticObserver(...)` calls in `observer_factory.py`.
**Confidence:** HIGH — verified by reading `observer_factory.py` and `config.py`. The `config.py` reference at line 605 is for path resolution, not observer config.

### Pitfall 2: Deleting symbols still used by reconstruction evaluator
**What goes wrong:** Removing `Tier1Result`, `Tier2Result`, `compute_tier1`, `Tier2Result` from `metrics.py` because "harness is deleted".
**How to avoid:** `stages/reconstruction.py` imports `Tier2Result` and `compute_tier1` directly from `metrics`. Only `compute_tier2` becomes orphaned.
**Confidence:** HIGH — verified by grep.

### Pitfall 3: Deleting all of test_output.py
**What goes wrong:** `test_output.py` tests both legacy (`format_summary_table`, `write_regression_json`) and current (`format_eval_report`, `format_eval_json`, etc.) functions. Deleting the file removes tests for surviving code.
**How to avoid:** Edit `test_output.py` — delete only the legacy function tests, preserve the rest.
**Confidence:** HIGH — verified by reading test file.

### Pitfall 4: Missing the NPZ_VERSION import in diagnostic_observer.py
**What goes wrong:** Leaving `from aquapose.io.midline_fixture import NPZ_VERSION as _NPZ_VERSION_LATEST` after deleting `midline_fixture.py`, causing an ImportError.
**How to avoid:** Remove this import and the `_NPZ_VERSION_V1 = "1.0"` constant from `diagnostic_observer.py` as part of the NPZ method removal.
**Confidence:** HIGH — confirmed in the file header.

### Pitfall 5: Missing stage-name constants used only by deleted methods
**What goes wrong:** Leaving `_DETECTION_STAGE_NAME`, `_TRACKING_STAGE_NAME`, `_ASSOCIATION_STAGE_NAME`, `_MIDLINE_STAGE_NAME` constants in `diagnostic_observer.py` after deleting all methods that used them.
**How to avoid:** After removing all NPZ methods, grep for each constant — if no surviving code references it, remove it.
**Note:** `_CENTROID_MATCH_TOLERANCE_PX` is only used by `_match_annotated_by_centroid` (deleted). The four stage-name constants may all be orphaned after deletion.
**Confidence:** HIGH — verified by reading the file.

### Pitfall 6: forgetting _on_pipeline_complete body cleanup
**What goes wrong:** Leaving the `_on_pipeline_complete` body with calls to `export_pipeline_diagnostics` and `export_midline_fixtures` after those methods are removed.
**How to avoid:** Replace `_on_pipeline_complete` body with `pass` and update its docstring to indicate it is a no-op hook kept for future extensibility.
**Confidence:** HIGH — explicit decision in CONTEXT.md.

### Pitfall 7: Leaving `generate_fixture` in harness.py scope in tests
**What goes wrong:** After deleting `evaluation/harness.py`, imports in scripts or config YAML files that reference it cause ImportError at runtime.
**How to avoid:** The legacy scripts (`tune_association.py`, `tune_threshold.py`, `measure_baseline.py`) were already deleted in Phase 49. No other source files import from `harness.py` (confirmed by grep).
**Confidence:** HIGH — verified: only `evaluation/__init__.py` imports from harness.py, and test_harness.py.

## Code Examples

### DiagnosticObserver __init__ after cleanup (HIGH confidence)

```python
def __init__(
    self,
    output_dir: str | Path | None = None,
) -> None:
    self.stages: dict[str, StageSnapshot] = {}
    self._output_dir = Path(output_dir) if output_dir is not None else None
    self._run_id: str = ""
```

### _on_pipeline_complete after cleanup (HIGH confidence)

```python
def _on_pipeline_complete(self) -> None:
    """No-op hook called when the pipeline completes.

    Reserved for future extensibility.
    """
    pass
```

### evaluation/__init__.py after cleanup (HIGH confidence)

Remove these lines:
```python
from aquapose.evaluation.harness import EvalResults, generate_fixture, run_evaluation
```

And remove from `__all__`:
```python
"EvalResults",
"generate_fixture",
"run_evaluation",
```

Also remove from `__all__` if `format_summary_table` and `write_regression_json` were present (they are not in current `__all__`, confirmed).

### io/__init__.py after cleanup (HIGH confidence)

Remove:
```python
from .midline_fixture import (
    NPZ_VERSION,
    CalibBundle,
    MidlineFixture,
    load_midline_fixture,
)
```

And from `__all__`:
```python
"NPZ_VERSION",
"CalibBundle",
"MidlineFixture",
"load_midline_fixture",
```

### observer_factory.py DiagnosticObserver construction (HIGH confidence)

```python
# Diagnostic mode branch
DiagnosticObserver(output_dir=config.output_dir)

# Additive observer branch
DiagnosticObserver(output_dir=config.output_dir)
```

## Deletion Checklist (Ordered)

**Recommended order (minimizes cascading import errors during development):**

1. Delete `evaluation/harness.py`
2. Update `evaluation/__init__.py` — remove harness imports and `__all__` entries
3. Audit and edit `evaluation/metrics.py` — remove `compute_tier2`
4. Audit and edit `evaluation/output.py` — remove `format_summary_table`, `write_regression_json`
5. Delete `io/midline_fixture.py`
6. Update `io/__init__.py` — remove midline_fixture imports
7. Edit `engine/diagnostic_observer.py` — remove NPZ imports, constants, methods, body of `_on_pipeline_complete`, remove `calibration_path` from constructor
8. Edit `engine/observer_factory.py` — remove `calibration_path` from DiagnosticObserver calls
9. Delete `tests/unit/evaluation/test_harness.py`
10. Delete `tests/unit/io/test_midline_fixture.py`
11. Edit `tests/unit/engine/test_diagnostic_observer.py` — remove tests for deleted methods
12. Edit `tests/unit/evaluation/test_output.py` — remove tests for deleted functions
13. Edit `tests/unit/evaluation/test_metrics.py` — remove tests for `compute_tier2`
14. Run `hatch run test` — verify full pass
15. Run `hatch run check` — verify lint + typecheck

## Open Questions

1. **test_eval_output.py content** — RESOLVED
   - Covers only `format_eval_report` and `format_eval_json` (current-era functions)
   - Uses `DetectionMetrics`, `TrackingMetrics`, etc. — no legacy types
   - This file survives entirely unchanged

2. **Stage name constants in diagnostic_observer.py**
   - What we know: `_DETECTION_STAGE_NAME`, `_TRACKING_STAGE_NAME`, `_ASSOCIATION_STAGE_NAME`, `_MIDLINE_STAGE_NAME` are currently only referenced by methods being deleted
   - What's unclear: Whether any test or docstring references them by name
   - Recommendation: Remove them during DiagnosticObserver cleanup; grep confirms no surviving method uses them

3. **`_PER_FRAME_FIELDS` and `_SCALAR_FIELDS` constants**
   - `_PER_FRAME_FIELDS` is referenced by `StageSnapshot.__getitem__` (survives)
   - `_SCALAR_FIELDS` appears only in comments — can be removed
   - Recommendation: Keep `_PER_FRAME_FIELDS`, remove `_SCALAR_FIELDS`

## Sources

### Primary (HIGH confidence)

- Direct code inspection of all files listed above — findings are verified from current source
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/diagnostic_observer.py` — full file read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/harness.py` — full file read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/__init__.py` — full file read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/io/__init__.py` — full file read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/engine/observer_factory.py` — full file read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/metrics.py` — full file read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/evaluation/output.py` — full file read
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/io/midline_fixture.py` — full file read
- `/home/tlancaster6/Projects/AquaPose/tests/unit/evaluation/test_harness.py` — full file read
- `/home/tlancaster6/Projects/AquaPose/tests/unit/engine/test_diagnostic_observer.py` — partial + grep
- `/home/tlancaster6/Projects/AquaPose/tests/unit/io/test_midline_fixture.py` — header read
- Grep sweeps for cross-references: `calibration_path`, `compute_tier2`, `format_summary_table`, `write_regression_json`, `Tier1Result`, `Tier2Result`, `select_frames`

## Metadata

**Confidence breakdown:**
- File inventory and deletion scope: HIGH — all relevant files inspected directly
- Symbol cross-reference analysis: HIGH — grep-verified
- Test file split (delete vs. edit): HIGH — test files read and mapped
- calibration_path scope: HIGH — confirmed config.py and stage code are unaffected

**Research date:** 2026-03-03
**Valid until:** This is a one-time cleanup; valid indefinitely (no external dependencies)
