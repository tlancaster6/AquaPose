---
phase: 48-evalrunner-and-aquapose-eval-cli
plan: 02
subsystem: evaluation
tags: [click, cli, ascii-report, json, output-formatting]

# Dependency graph
requires:
  - phase: 48-01
    provides: EvalRunner, EvalRunnerResult, per-stage Metrics dataclasses with to_dict()

provides:
  - format_eval_report() — multi-stage ASCII report formatter in output.py
  - format_eval_json() — JSON formatter using _NumpySafeEncoder in output.py
  - aquapose eval CLI command with --report and --n-frames options
  - evaluation/__init__.py exports EvalRunner, EvalRunnerResult, format_eval_report, format_eval_json
  - scripts/measure_baseline.py deleted (CLEAN-03)

affects: [phase-49-tuning-orchestrator, users running aquapose eval]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CLI command named eval_cmd to avoid shadowing Python built-in eval, registered as @cli.command('eval')"
    - "Inline imports inside CLI commands (StaleCacheError, EvalRunner, formatters) to avoid top-level coupling"
    - "TYPE_CHECKING guard for EvalRunnerResult in output.py to prevent circular imports"
    - "format_eval_json delegates to result.to_dict() + json.dumps with _NumpySafeEncoder"

key-files:
  created:
    - tests/unit/evaluation/test_eval_output.py
  modified:
    - src/aquapose/evaluation/output.py
    - src/aquapose/evaluation/__init__.py
    - src/aquapose/cli.py
  deleted:
    - scripts/measure_baseline.py

key-decisions:
  - "eval_cmd function name avoids shadowing Python built-in eval; registered as @cli.command('eval')"
  - "format_eval_json simply calls result.to_dict() and json.dumps — no duplication of to_dict() logic"
  - "TYPE_CHECKING guard used for EvalRunnerResult import in output.py (safe: runner.py does not import output.py)"
  - "eval_results.json always written to run_dir on every eval invocation regardless of --report flag"

patterns-established:
  - "Inline imports inside CLI command functions to avoid top-level coupling with evaluation subsystem"
  - "format_eval_report uses nested _row() and _header() helpers for consistent column alignment"

requirements-completed: [EVAL-06, EVAL-07, CLEAN-03]

# Metrics
duration: 3min
completed: 2026-03-03
---

# Phase 48 Plan 02: EvalRunner Output Formatters and CLI Summary

**ASCII and JSON multi-stage report formatters wired to `aquapose eval <run-dir>` CLI with deletion of legacy measure_baseline.py**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-03T19:21:37Z
- **Completed:** 2026-03-03T19:24:37Z
- **Tasks:** 2 (TDD: Task 1 with RED/GREEN cycles; Task 2 direct)
- **Files modified:** 4 modified, 1 created, 1 deleted

## Accomplishments
- format_eval_report() produces multi-stage ASCII report with header block, one-line summary per stage, and per-stage detail sections (Tier 1/Tier 2 sub-structure for reconstruction)
- format_eval_json() produces valid JSON via result.to_dict() + _NumpySafeEncoder, handles partial/empty results and numpy scalars
- `aquapose eval <run-dir>` CLI command prints human-readable or JSON report, writes eval_results.json to run directory, catches StaleCacheError as ClickException
- evaluation/__init__.py updated to export format_eval_report, format_eval_json
- scripts/measure_baseline.py deleted (CLEAN-03): all functionality replaced by `aquapose eval`

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for format_eval_report and format_eval_json** - `9bfbf7a` (test)
2. **Task 1 (GREEN): format_eval_report and format_eval_json implementation** - `c93fc82` (feat)
3. **Task 2: CLI eval command, __init__.py exports, delete measure_baseline.py** - `053a653` (feat)

**Plan metadata:** (docs commit — pending)

_Note: TDD task had two commits (RED test then GREEN implementation); lint fix applied inline before GREEN commit._

## Files Created/Modified
- `src/aquapose/evaluation/output.py` - Added format_eval_report(), format_eval_json(), TYPE_CHECKING import guard
- `src/aquapose/evaluation/__init__.py` - Added format_eval_report, format_eval_json, updated __all__
- `src/aquapose/cli.py` - Added eval_cmd registered as @cli.command("eval")
- `tests/unit/evaluation/test_eval_output.py` - New: 11 unit tests for format_eval_report and format_eval_json
- `scripts/measure_baseline.py` - Deleted (CLEAN-03)

## Decisions Made
- eval_cmd function name avoids shadowing Python built-in eval; registered as @cli.command("eval")
- format_eval_json simply calls result.to_dict() and json.dumps — no duplication of to_dict() logic
- TYPE_CHECKING guard used for EvalRunnerResult import in output.py (no circular import risk, but cleaner)
- eval_results.json always written to run_dir on every eval invocation regardless of --report flag

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Ruff lint pre-commit hook failed on `if isinstance(value, float): ... else: ...` (SIM108 rule) — converted to ternary operator inline before committing GREEN phase.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 48 complete: EvalRunner (Plan 01) + formatters + CLI (Plan 02) all done
- Phase 49 (TuningOrchestrator) can import from aquapose.evaluation: EvalRunner, format_eval_report, format_eval_json
- aquapose eval command is live and ready for user testing against diagnostic run directories
- Blocker to verify before Phase 49: stop_after field presence in PipelineConfig (noted in STATE.md)

---
*Phase: 48-evalrunner-and-aquapose-eval-cli*
*Completed: 2026-03-03*

## Self-Check: PASSED

- src/aquapose/evaluation/output.py: FOUND
- src/aquapose/cli.py: FOUND
- tests/unit/evaluation/test_eval_output.py: FOUND
- scripts/measure_baseline.py: VERIFIED DELETED
- Commit 9bfbf7a (RED tests): FOUND
- Commit c93fc82 (GREEN implementation): FOUND
- Commit 053a653 (CLI + __init__ + delete): FOUND
