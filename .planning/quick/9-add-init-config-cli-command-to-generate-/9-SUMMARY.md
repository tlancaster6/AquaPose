---
phase: quick
plan: 9
subsystem: cli
tags: [cli, config, init-config, click]
dependency_graph:
  requires: [aquapose.engine.config.serialize_config, aquapose.engine.config.PipelineConfig]
  provides: [aquapose init-config subcommand]
  affects: [src/aquapose/cli.py]
tech_stack:
  added: []
  patterns: [Click subcommand, isolated_filesystem testing]
key_files:
  created: []
  modified:
    - src/aquapose/cli.py
    - tests/unit/engine/test_cli.py
decisions:
  - Used click.ClickException for overwrite refusal — integrates cleanly with Click's error output and non-zero exit codes
  - PipelineConfig() constructed with no args to get pure defaults; serialize_config() produces the YAML directly
metrics:
  duration: 10
  completed_date: "2026-02-26"
  tasks_completed: 1
  files_modified: 2
---

# Quick Task 9: Add init-config CLI Subcommand Summary

**One-liner:** `aquapose init-config` subcommand generating default YAML scaffold via PipelineConfig() + serialize_config() with --output and --force flags.

## What Was Built

Added an `init-config` Click subcommand to the existing AquaPose CLI. The command generates a complete default YAML configuration file by instantiating a default `PipelineConfig()` and serializing it with `serialize_config()`. This gives users a starting-point config with all pipeline sections and their defaults, without needing to read source code.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add init-config subcommand and tests | 1d6eaf9 | src/aquapose/cli.py, tests/unit/engine/test_cli.py |

## Changes Made

### src/aquapose/cli.py
- Added `serialize_config` to the existing `aquapose.engine` import
- Added `init_config` Click subcommand registered on the `cli` group with `--output/-o` and `--force` flags
- Command writes YAML to output path; refuses overwrite unless `--force` given

### tests/unit/engine/test_cli.py
- Added `TestInitConfig` class with 5 tests:
  - `test_init_config_creates_default_file`: verifies default `aquapose.yaml` is created with all expected sections
  - `test_init_config_custom_output`: verifies `-o custom.yaml` writes to specified path
  - `test_init_config_refuses_overwrite`: verifies exit code != 0 and file unchanged when overwriting without --force
  - `test_init_config_force_overwrites`: verifies `--force` allows overwrite with exit code 0
  - `test_init_config_help`: verifies `--help` shows `--output` option

## Verification

All 23 CLI tests pass (18 existing + 5 new). Lint clean. Pre-existing typecheck errors (13) in unrelated files are out of scope.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `src/aquapose/cli.py` exists and contains `def init_config`
- `tests/unit/engine/test_cli.py` exists and contains `TestInitConfig`
- Commit `1d6eaf9` verified: `feat(quick-9): add init-config CLI subcommand`
- All 23 CLI tests pass
