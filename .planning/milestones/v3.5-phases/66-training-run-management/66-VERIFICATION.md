---
phase: 66-training-run-management
verified: 2026-03-05T21:30:00Z
status: passed
score: 8/8 must-haves verified
re_verification: false
---

# Phase 66: Training Run Management Verification Report

**Phase Goal:** Users can track, compare, and iterate on training runs with full provenance of which pseudo-label round and thresholds produced each model
**Verified:** 2026-03-05T21:30:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Each training run outputs to a unique timestamped directory under projects/{project}/training/{model_type}/run_{timestamp}/ | VERIFIED | `create_run_dir` in run_manager.py L50-53 constructs `{project_dir}/training/{model_type}/run_{timestamp}/`; 2 tests confirm directory structure |
| 2 | Run directory contains frozen config YAML, and post-training summary.json with best metrics and dataset provenance | VERIFIED | `snapshot_config` writes config.yaml + dataset sidecars; `write_summary` writes summary.json with metrics from parse_best_metrics and provenance from extract_dataset_provenance; 6 tests cover these |
| 3 | summary.json records parent weights path, dataset path, source breakdown from confidence.json, and tag | VERIFIED | write_summary L190-212 populates run_id, tag, model_type, model_variant, parent_weights, dataset_path, dataset_sources, training_config, metrics, training_duration_seconds, created; test confirms all keys present |
| 4 | CLI commands accept --config (required) and --tag (optional) instead of --output-dir | VERIFIED | cli.py yolo_obb (L23-28), seg (L191-200), pose (L309-320) all have --config and --tag; 3 help tests confirm flags present and --output-dir absent |
| 5 | Post-training output prints suggested next steps (compare command, best weights path) | VERIFIED | print_next_steps in run_manager.py L219-242 prints run_dir, best_weights, and compare/retrain commands; called from all 3 CLI commands |
| 6 | User can run 'aquapose train compare --config config.yaml --model-type obb' to see a comparison table of all runs | VERIFIED | compare command in cli.py L133-181 wired with discover_runs, load_run_summaries, format_comparison_table; help test confirms --config, --model-type, --csv flags |
| 7 | Best values per metric column are highlighted in bold green | VERIFIED | format_comparison_table L134-139 wraps best values with click.style(fg="green", bold=True); test confirms ANSI escape codes present |
| 8 | Comparison table shows source type breakdown (consensus %, gap %) from provenance data | VERIFIED | _format_sources L63-74 renders "X% cons / Y% gap" in Sources column; test confirms "80%" appears in table output |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/training/run_manager.py` | Run directory creation, config snapshot, summary.json write, results.csv parsing, provenance extraction | VERIFIED | 242 lines, 7 public functions, all exported in __init__.py |
| `src/aquapose/training/compare.py` | Run discovery, summary loading, table formatting, CSV export | VERIFIED | 219 lines, 4 public functions, all exported in __init__.py |
| `src/aquapose/training/cli.py` | Modified yolo-obb/seg/pose commands with --config/--tag, compare subcommand | VERIFIED | All 3 training commands + compare command present with full run management integration |
| `tests/unit/training/test_run_manager.py` | Unit tests for run_manager functions | VERIFIED | 14 tests across 7 test classes |
| `tests/unit/training/test_compare.py` | Unit tests for comparison functions | VERIFIED | 9 tests across 4 test classes |
| `tests/unit/training/test_training_cli.py` | CLI help and import boundary tests | VERIFIED | 9 tests including compare help and import boundary |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| cli.py | run_manager.py | `from .run_manager import create_run_dir, snapshot_config, write_summary, print_next_steps` | WIRED | All 3 training commands import and call run_manager functions (L79-84, L248-253, L366-371) |
| cli.py | compare.py | `from .compare import discover_runs, load_run_summaries, format_comparison_table, write_comparison_csv` | WIRED | compare command imports and calls all 4 functions (L161-166) |
| run_manager.py | yaml.safe_load | Config reading without engine imports | WIRED | L28: `yaml.safe_load(f)` in resolve_project_dir; import boundary test passes |
| compare.py | summary.json | `json.load` to read run summaries | WIRED | L52: `json.load(f)` in load_run_summaries |
| __init__.py | run_manager.py | Public API exports | WIRED | L41-48: all 6 run_manager functions imported and in __all__ |
| __init__.py | compare.py | Public API exports | WIRED | L11-16: all 4 compare functions imported and in __all__ |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TRAIN-01 | 66-01 | Each training run outputs to a unique timestamped directory with config snapshot and metric summary | SATISFIED | create_run_dir creates timestamped dirs; snapshot_config writes config.yaml; write_summary writes summary.json with metrics |
| TRAIN-02 | 66-02 | User can run `aquapose train compare` to generate cross-run comparison (Ultralytics training metrics + aquapose eval pipeline metrics) | SATISFIED | compare CLI command with auto-discovery, terminal table, CSV export; metrics include mAP50, mAP50-95, precision, recall |
| TRAIN-03 | 66-01 | Comparison tracks which pseudo-label round and confidence thresholds were used per run | SATISFIED | summary.json stores full dataset_sources from confidence.json (consensus_threshold, gap_threshold, pipeline_run); compare table shows source breakdown fractions |

No orphaned requirements found.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No anti-patterns found |

No TODOs, FIXMEs, placeholders, or stub implementations detected in any phase files.

### Human Verification Required

None required. All observable truths are verifiable through code inspection and automated tests. The CLI commands integrate with Ultralytics training wrappers which require GPU/data to run end-to-end, but the wiring and structure are fully verified.

### Notes

The comparison table's "Sources" column shows consensus/gap fractions (e.g., "80% cons / 20% gap") rather than raw threshold values. The full provenance data (consensus_threshold, gap_threshold, pipeline_run) is preserved in each run's summary.json and is accessible for inspection. The table design is a reasonable UX choice -- showing fractions at a glance with full details available in the run directory.

All 987 tests pass (including 32 training-specific tests). Import boundary test confirms no engine imports in training/ modules. Lint and typecheck were reported clean in the SUMMARYs.

---

_Verified: 2026-03-05T21:30:00Z_
_Verifier: Claude (gsd-verifier)_
