---
phase: 18-cli-and-execution-modes
plan: 01
status: complete
---

## Summary

Created the `aquapose run` CLI entrypoint as a thin Click-based wrapper over PosePipeline, with ConsoleObserver for stage-level progress output to stderr.

## What was built

- **CLI module** (`src/aquapose/cli.py`): Click group with `run` command accepting `--config`, `--mode`, `--set`, `--add-observer`, and `--verbose` flags. Contains zero pipeline computation logic.
- **ConsoleObserver** (`src/aquapose/engine/console_observer.py`): Prints `[1/5] StageName... done (12.3s)` on StageComplete, summary on PipelineComplete, and error on PipelineFailed. Verbose mode adds per-frame detail.
- **pyproject.toml**: Added `click>=8.1` dependency and `[project.scripts] aquapose = "aquapose.cli:main"` entry point.
- **Observer assembly**: `_build_observers()` helper builds mode-specific observer lists. Production mode: ConsoleObserver + TimingObserver + HDF5ExportObserver.

## Key files

### Created
- `src/aquapose/cli.py`
- `src/aquapose/engine/console_observer.py`
- `tests/unit/engine/test_cli.py`
- `tests/unit/engine/test_console_observer.py`

### Modified
- `pyproject.toml`
- `src/aquapose/engine/__init__.py`

## Verification

- 19 unit tests pass (12 CLI + 7 ConsoleObserver)
- CLI imports cleanly, no computation logic in module
- Entry point registered in pyproject.toml

## Commits

- `c611205` feat(18-01): add CLI entrypoint and ConsoleObserver
