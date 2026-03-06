---
phase: 18-cli-and-execution-modes
status: passed
verified: 2026-02-26
---

## Phase 18: CLI and Execution Modes -- Verification

### Phase Goal
`aquapose run` is a working CLI entrypoint that accepts a config path and mode flag, assembles the correct observer set, and runs the pipeline -- with no pipeline logic living in the CLI layer.

### Requirement Verification

| Req ID | Description | Status | Evidence |
|--------|-------------|--------|----------|
| CLI-01 | `aquapose run` CLI entrypoint as thin wrapper over PosePipeline | PASS | `from aquapose.cli import cli` succeeds; `[project.scripts] aquapose = "aquapose.cli:main"` in pyproject.toml; 12 CLI unit tests pass |
| CLI-02 | Production mode -- standard pipeline execution | PASS | Production mode assembles ConsoleObserver + TimingObserver + HDF5ExportObserver; default mode is production |
| CLI-03 | Diagnostic mode -- activates all observers | PASS | `--mode diagnostic` assembles all 6 observers (console, timing, hdf5, overlay2d, animation3d, diagnostic) |
| CLI-04 | Synthetic mode -- stage adapter injects synthetic data | PASS | SyntheticDataStage satisfies Stage protocol; build_stages returns 4-stage list for synthetic mode; configurable via SyntheticConfig |
| CLI-05 | Benchmark mode -- timing-focused, minimal observers | PASS | `--mode benchmark` assembles ConsoleObserver + TimingObserver only; no HDF5, no visualization |

### Must-Have Verification

**Plan 18-01:**
- `aquapose run --config path.yaml` parses args, builds config, assembles observers, calls PosePipeline.run(): PASS
- Production mode attaches TimingObserver + HDF5ExportObserver: PASS
- CLI contains zero reconstruction logic: PASS (only _build_observers, cli, run, main functions)
- Console output shows stage-level progress: PASS (ConsoleObserver prints `[1/5] Stage... done (12.3s)`)
- `--set key=val` feeds dot-notation overrides into load_config(): PASS
- Exit code 1 on failure: PASS (tested with mocked exception)

**Plan 18-02:**
- Diagnostic mode activates ALL 5 observers plus console: PASS
- Benchmark mode activates TimingObserver ONLY plus console: PASS
- Diagnostic mode produces extra artifacts without code change to stages: PASS (observer-only assembly)
- `--add-observer` augments any mode additively: PASS

**Plan 18-03:**
- `aquapose run --mode synthetic` runs pipeline using synthetic data: PASS
- SyntheticDataStage replaces Detection + Midline stages: PASS (build_stages returns 4-stage list)
- Synthetic mode still requires real calibration file: PASS (calibration_path is required)
- Synthetic data is configurable (fish count, frame count, noise): PASS (SyntheticConfig defaults 3/30/0)
- SyntheticDataStage satisfies Stage protocol: PASS

### Automated Test Summary

- 576 total unit tests pass
- 35 tests specific to Phase 18 (CLI, ConsoleObserver, build_stages, SyntheticDataStage)
- No computation logic in CLI module (verified via AST inspection)

### Score

**5/5 requirements verified. All must-haves satisfied.**
