---
phase: 48-evalrunner-and-aquapose-eval-cli
verified: 2026-03-03T19:45:00Z
status: passed
score: 15/15 must-haves verified
re_verification: false
gaps: []
human_verification: []
---

# Phase 48: EvalRunner and Aquapose Eval CLI Verification Report

**Phase Goal:** Users can evaluate any diagnostic run directory and receive a multi-stage quality report in human-readable or JSON format, replacing the functionality of scripts/measure_baseline.py
**Verified:** 2026-03-03T19:45:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Plan 01)

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | EvalRunner discovers per-stage pickle caches from a run directory and loads them via load_stage_cache() | VERIFIED | `_discover_caches()` probes `<run_dir>/diagnostics/<stage>_cache.pkl` for all 5 stage keys; calls `load_stage_cache(cache_path)` |
| 2  | EvalRunner calls Phase 47 stage evaluators with correctly unpacked PipelineContext data | VERIFIED | `run()` dispatches each present stage to the correct evaluator with properly unpacked fields (detections, tracks_2d, tracklet_groups, annotated_detections, midlines_3d) |
| 3  | EvalRunner silently skips stages whose cache files are missing | VERIFIED | `_discover_caches()` checks `cache_path.exists()` before loading; missing files produce no exception; `test_detection_only_cache` and `test_empty_run_dir_returns_all_none` both pass |
| 4  | EvalRunnerResult frozen dataclass holds Optional metric results for all 5 stages plus run metadata | VERIFIED | `@dataclass(frozen=True) class EvalRunnerResult` with `run_id`, `stages_present`, `detection`, `tracking`, `association`, `midline`, `reconstruction`, `frames_evaluated`, `frames_available` |
| 5  | EvalRunner reads n_animals from config.yaml in the run directory when association cache is present | VERIFIED | `_read_n_animals()` uses inline `from aquapose.engine.config import load_config`; reads `self._run_dir / "config.yaml"`; called in `run()` when `"association" in caches` |
| 6  | EvalRunner supports n_frames sampling via select_frames() when n_frames is not None | VERIFIED | `run(n_frames=...)` calls `select_frames(tuple(range(frame_count)), n_frames)`; `frames_evaluated = len(sampled_indices)`; `test_n_frames_sampling` passes (frames_evaluated=3, frames_available=10) |
| 7  | EvalRunner can run from synthetic test data without a real pipeline run or GPU | VERIFIED | 10 unit tests use synthetic pickle fixtures; all pass in 13.88s with no GPU dependency |

### Observable Truths (Plan 02)

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 8  | format_eval_report(result) produces multi-stage ASCII report with header summary + one section per present stage | VERIFIED | Lines 354-513 of output.py; header block, summary lines, per-stage detail sections with `_row()` and `_header()` helpers; 6 tests pass |
| 9  | format_eval_json(result) produces JSON string with run_metadata and stages dict, using _NumpySafeEncoder | VERIFIED | Lines 516-530 of output.py; delegates to `result.to_dict()` + `json.dumps(..., cls=_NumpySafeEncoder, indent=2)`; 6 tests pass including numpy encoding test |
| 10 | 'aquapose eval <run-dir>' prints human-readable report to stdout and writes eval_results.json to run directory | VERIFIED | `eval_cmd` in cli.py (line 232) calls `format_eval_report(result)` for `report=="text"` (default), writes `Path(run_dir) / "eval_results.json"` unconditionally |
| 11 | 'aquapose eval <run-dir> --report json' prints JSON to stdout | VERIFIED | `eval_cmd` calls `format_eval_json(result)` when `report == "json"`; `--report [text\|json]` option confirmed via `aquapose eval --help` |
| 12 | 'aquapose eval <run-dir> --n-frames 10' passes n_frames to EvalRunner | VERIFIED | `--n-frames` option in `eval_cmd`; `runner.run(n_frames=n_frames)` called with the value |
| 13 | StaleCacheError is caught in CLI and re-raised as click.ClickException | VERIFIED | cli.py lines 247-248: `except StaleCacheError as exc: raise click.ClickException(str(exc)) from exc` |
| 14 | scripts/measure_baseline.py is deleted from the repository | VERIFIED | `ls scripts/measure_baseline.py` returns DELETED; confirmed by bash check |
| 15 | evaluation/__init__.py exports EvalRunner and EvalRunnerResult | VERIFIED | `__init__.py` lines 17-18: `from aquapose.evaluation.runner import EvalRunner, EvalRunnerResult`; both in `__all__`; `from aquapose.evaluation import EvalRunner, EvalRunnerResult, format_eval_report, format_eval_json` import succeeds |

**Score:** 15/15 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/evaluation/runner.py` | EvalRunner class and EvalRunnerResult frozen dataclass | VERIFIED | 402 lines; substantive implementation with _discover_caches, _read_n_animals, _build_midline_sets, _match_annotated_by_centroid; exports EvalRunner, EvalRunnerResult |
| `tests/unit/evaluation/test_runner.py` | Unit tests for EvalRunner with synthetic cache fixtures | VERIFIED | 446 lines; 10 tests covering empty run dir, detection-only, all stages, n_frames sampling, JSON serialization, missing config.yaml, StaleCacheError propagation, tracking flat list, to_dict absent stages, stages_present sorted |
| `src/aquapose/evaluation/output.py` | format_eval_report() and format_eval_json() functions | VERIFIED | format_eval_report at line 354 (160 lines), format_eval_json at line 516 (15 lines); both exported; TYPE_CHECKING guard for EvalRunnerResult |
| `src/aquapose/evaluation/__init__.py` | Updated exports including EvalRunner, EvalRunnerResult, format_eval_report, format_eval_json | VERIFIED | Imports EvalRunner, EvalRunnerResult from runner; format_eval_report, format_eval_json from output; all 4 in __all__ |
| `src/aquapose/cli.py` | New 'eval' command in Click group | VERIFIED | `@cli.command("eval")` at line 217; function `eval_cmd` with run_dir argument, --report and --n-frames options |
| `tests/unit/evaluation/test_eval_output.py` | Unit tests for format_eval_report and format_eval_json | VERIFIED | 322 lines; 12 tests covering full/partial/empty results, JSON validity, schema, stage keys, numpy encoding |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `runner.py` | `aquapose.core.context.load_stage_cache` | top-level import | VERIFIED | Line 10: `from aquapose.core.context import load_stage_cache` (top-level, not inline — acceptable; plan said "inline import" but top-level works correctly and passes all plan tests) |
| `runner.py` | `aquapose.evaluation.stages` (all 5 evaluators) | top-level import | VERIFIED | Lines 14-25: all 5 evaluators and 5 metric types imported from `aquapose.evaluation.stages` |
| `runner.py` | `aquapose.engine.config.load_config` | inline import in _read_n_animals | VERIFIED | Line 281 inside `_read_n_animals()`: `from aquapose.engine.config import load_config` |
| `cli.py` | `aquapose.evaluation.runner.EvalRunner` | inline import in eval_cmd | VERIFIED | Line 242 inside `eval_cmd`: `from aquapose.evaluation.runner import EvalRunner` |
| `cli.py` | `aquapose.evaluation.output.format_eval_report, format_eval_json` | inline import in eval_cmd | VERIFIED | Lines 238-240 inside `eval_cmd`: `from aquapose.evaluation.output import (_NumpySafeEncoder, format_eval_json, format_eval_report)` |
| `output.py` | `aquapose.evaluation.runner.EvalRunnerResult` | TYPE_CHECKING guard | VERIFIED | Lines 19-20: `if TYPE_CHECKING: from aquapose.evaluation.runner import EvalRunnerResult` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| EVAL-06 | 48-01, 48-02 | `aquapose eval <run-dir>` CLI produces multi-stage human-readable report to stdout | SATISFIED | `eval_cmd` calls `format_eval_report(result)` by default; CLI help confirms; 6 format_eval_report tests pass |
| EVAL-07 | 48-01, 48-02 | `aquapose eval <run-dir> --report json` produces machine-readable JSON output | SATISFIED | `eval_cmd` calls `format_eval_json(result)` when `--report json`; 6 format_eval_json tests pass |
| CLEAN-03 | 48-02 | `scripts/measure_baseline.py` retired after `aquapose eval` achieves feature parity | SATISFIED | File confirmed deleted; `aquapose eval` covers all measure_baseline.py functionality (run evaluation, format report, save results) |

No orphaned requirements found — all 3 requirement IDs assigned to Phase 48 in REQUIREMENTS.md are covered by the plans.

### Anti-Patterns Found

None detected in phase 48 files.

Scanned files:
- `src/aquapose/evaluation/runner.py` — no TODO/FIXME/placeholder; no empty implementations; no return null/empty stubs
- `src/aquapose/evaluation/output.py` (new functions only) — no placeholder patterns
- `src/aquapose/cli.py` (eval_cmd only) — no placeholder patterns
- `tests/unit/evaluation/test_runner.py` — substantive synthetic fixtures
- `tests/unit/evaluation/test_eval_output.py` — substantive assertions against known metric values

### Human Verification Required

None. All phase goals are verifiable programmatically:
- CLI invocation (`aquapose eval --help`): confirmed working
- Imports: confirmed with `python -c "from aquapose.evaluation import ..."`
- Test suite: 22 tests pass without GPU or real pipeline data
- Lint: all checks passed
- Type errors: zero errors in phase 48 files (pre-existing errors in unrelated files unaffected)

### Gaps Summary

No gaps. All 15 observable truths verified. All 6 required artifacts exist with substantive implementations. All 6 key links confirmed wired. All 3 requirement IDs (EVAL-06, EVAL-07, CLEAN-03) satisfied.

One minor deviation from plan wording noted but not a gap: `runner.py` imports `load_stage_cache` at the top level rather than inline. The plan said "inline import for cache loading" but the implementation uses a top-level import. This is architecturally sound (not an engine coupling concern), all plan tests pass, and lint/typecheck are clean. The engine coupling concern in the plan only applied to `aquapose.engine.config` imports, which correctly remain inline in `_read_n_animals()`.

---

_Verified: 2026-03-03T19:45:00Z_
_Verifier: Claude (gsd-verifier)_
