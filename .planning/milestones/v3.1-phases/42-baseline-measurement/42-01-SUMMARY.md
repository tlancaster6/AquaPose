---
phase: 42-baseline-measurement
plan: "01"
subsystem: evaluation
tags: [numpy, json, argparse, outlier-detection, baseline, regression]

# Dependency graph
requires:
  - phase: 41-evaluation-harness
    provides: run_evaluation(), EvalResults, format_summary_table, write_regression_json
provides:
  - flag_outliers helper in evaluation/output.py
  - format_baseline_report with 2-sigma outlier annotation
  - scripts/measure_baseline.py standalone baseline measurement script
  - baseline_results.json schema with baseline_metadata (timestamp, fixture_path, backend_identifier)
affects:
  - 44-regression-testing (reads baseline_results.json as regression reference)
  - any future phase using evaluation/output.py

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Statistical outlier detection using numpy mean/std with configurable threshold
    - Augmenting an existing JSON file with metadata before saving under a new name

key-files:
  created:
    - scripts/measure_baseline.py
  modified:
    - src/aquapose/evaluation/output.py
    - src/aquapose/evaluation/__init__.py

key-decisions:
  - "flag_outliers returns empty set for <2 values or std==0 (not detectable, not an error)"
  - "format_baseline_report uses unicode escape for sigma in legend line to pass ruff RUF002"
  - "Baseline JSON is the existing eval JSON augmented with baseline_metadata key (not a new schema)"
  - "No tests for measure_baseline.py — manual execution only per CONTEXT.md decision"

patterns-established:
  - "Outlier flagging: flag_outliers(dict[str, float]) -> set[str] with threshold_std parameter"
  - "Baseline reports annotate outliers inline with ' *' suffix and trailing legend line"

requirements-completed: [EVAL-06]

# Metrics
duration: 5min
completed: 2026-03-02
---

# Phase 42 Plan 01: Baseline Measurement Summary

**Outlier-annotated baseline measurement script writing baseline_results.json + baseline_report.txt next to fixture via flag_outliers and format_baseline_report built on Phase 41 harness**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-02T20:41:14Z
- **Completed:** 2026-03-02T20:46:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `flag_outliers()` helper to evaluation/output.py detecting entries exceeding mean + 2*std
- Added `format_baseline_report()` producing asterisk-marked outlier rows in Tier 1 per-camera, Tier 1 per-fish, and Tier 2 dropout sections with legend
- Created `scripts/measure_baseline.py` as a self-contained argparse script that calls run_evaluation(), prints outlier-flagged report, and saves baseline_results.json (with baseline_metadata) and baseline_report.txt next to the fixture

## Task Commits

Each task was committed atomically:

1. **Task 1: Add outlier flagging to evaluation output module** - `b0d7999` (feat)
2. **Task 2: Create measure_baseline.py script with argparse, metadata, and persistence** - `fb1fe2f` (feat)

## Files Created/Modified

- `src/aquapose/evaluation/output.py` - Added flag_outliers helper and format_baseline_report function
- `src/aquapose/evaluation/__init__.py` - Exported flag_outliers and format_baseline_report in __all__
- `scripts/measure_baseline.py` - New standalone baseline measurement script

## Decisions Made

- Used unicode escape `\u03c3` in the legend string to satisfy ruff RUF002 (no ambiguous Greek characters in source), but kept `>2σ` in the displayed output string.
- Actually used plain text "* = outlier (>2 std from mean)" in the docstring and unicode in the emitted string — the docstring got simplified to "2 std" to pass ruff.
- Baseline JSON reuses the eval_results.json written by run_evaluation() and adds a `baseline_metadata` block before saving as baseline_results.json — avoids duplicating serialization logic.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ruff RUF002 ambiguous Greek sigma in docstring**
- **Found during:** Task 1 commit (pre-commit hook failure)
- **Issue:** Docstring contained `σ` (Greek small letter sigma) which ruff flags as RUF002 (ambiguous character)
- **Fix:** Replaced `>2σ from mean` with `>2 std from mean` in the docstring text; kept `\u03c3` unicode escape in the runtime string constant
- **Files modified:** src/aquapose/evaluation/output.py
- **Verification:** Pre-commit ruff hook passed on second commit attempt
- **Committed in:** b0d7999 (Task 1 commit, second attempt)

---

**Total deviations:** 1 auto-fixed (Rule 1 - lint compliance)
**Impact on plan:** Minor docstring wording change; runtime output unaffected.

## Issues Encountered

- ruff RUF002 fires on ambiguous Greek characters in docstrings. Workaround: use ASCII description in docstrings, unicode escape in runtime strings.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 42 baseline tooling is complete. The script can be run against any midline fixture NPZ to produce baseline_results.json for Phase 44 regression testing.
- Phase 44 (regression testing) needs to read baseline_results.json and compare against new results; the `baseline_metadata.backend_identifier` field identifies the triangulation backend used.

---
*Phase: 42-baseline-measurement*
*Completed: 2026-03-02*

## Self-Check: PASSED

- FOUND: scripts/measure_baseline.py
- FOUND: src/aquapose/evaluation/output.py
- FOUND: 42-01-SUMMARY.md
- FOUND commit b0d7999 (Task 1)
- FOUND commit fb1fe2f (Task 2)
