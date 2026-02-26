# Phase 18: CLI and Execution Modes - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

`aquapose run` CLI entrypoint as a thin wrapper over PosePipeline. Parses args, builds config, assembles the correct observer set per mode, and calls `PosePipeline.run()`. No pipeline logic lives in the CLI layer. Four execution modes: production, diagnostic, synthetic, benchmark.

</domain>

<decisions>
## Implementation Decisions

### Mode behavior
- **Production mode** (default): Attaches timing + HDF5 export observers. Standard pipeline execution with artifacts.
- **Diagnostic mode**: Attaches ALL 5 observers (timing, HDF5, 2D overlay, 3D animation, diagnostic capture). Full introspection.
- **Benchmark mode**: Attaches timing observer ONLY. No HDF5, no visualization. Measures pure computation time.
- **Synthetic mode**: Replaces Detection + Midline stages with a synthetic data adapter. Association, tracking, and reconstruction run as normal. Still requires real calibration file.
- Additive observer flags only (`--add-observer`). No removal flags. Users pick a simpler mode and build up.

### CLI flag design
- **Library**: Click (decorator-based argument parsing)
- **Required**: `--config path.yaml` is always required. Video dir and calibration path live in the config file.
- **Optional**: `--mode` (defaults to "production"), `--set key=val` for dot-notation config overrides (e.g. `--set detection.detector_kind=mog2`), `--add-observer` for additive observer attachment, `--verbose` for frame-level detail
- **Entry point**: `aquapose run` as the single command (no subcommands per mode)

### Synthetic mode data
- Generate known-geometry 3D fish splines as ground truth
- Project to 2D views via the real refractive calibration model
- Feed as synthetic detections with pre-computed 2D midlines (replaces Detection + Midline stages)
- Real calibration file required — tests synthetic fish in actual camera rig geometry
- Configurable via config file: fish count, frame count, noise level
- Defaults: 3 fish, 30 frames, no noise

### Output and logging
- **Default console**: Stage-level progress as each starts/completes with timing: `[1/5] Detection... done (12.3s)`
- **--verbose**: Adds per-frame counts (detections found, midlines extracted, associations made, tracks updated)
- **Exit codes**: 0 on success, 1 on any failure. Error details to stderr.
- **Completion summary**: Print run output directory path + total elapsed time: `Run complete: ~/aquapose/runs/run_20260226_143022/ (47.2s)`

### Claude's Discretion
- Console observer implementation (how stage progress and verbose output are printed)
- Click group/command structure details
- `--set` flag parsing implementation (Click callback vs custom type)
- `--add-observer` flag value names and how they map to observer classes
- Synthetic data generation internals (spline shapes, swimming motion patterns)

</decisions>

<specifics>
## Specific Ideas

- CLI should be CI-pipeline friendly — no interactive prompts, clean exit codes, errors to stderr
- The existing `load_config()` already supports YAML + CLI overrides via dot-notation — `--set` should feed directly into that
- `build_stages()` in pipeline.py is the canonical stage factory — synthetic mode should swap stages at this level
- The guidebook says: "No subcommands per mode. No script may call stage functions directly."

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 18-cli-and-execution-modes*
*Context gathered: 2026-02-26*
