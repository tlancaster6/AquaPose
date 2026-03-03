# Phase 49: TuningOrchestrator and `aquapose tune` CLI - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

CLI command (`aquapose tune --stage association|reconstruction`) that sweeps stage-specific parameters using upstream stage caches, validates top-N candidates at higher frame counts, and outputs a before/after comparison with a YAML config diff block. Replaces `scripts/tune_association.py` and `scripts/tune_threshold.py`, which are deleted after feature parity is achieved.

</domain>

<decisions>
## Implementation Decisions

### Sweep strategy
- **Association:** Joint 2D grid over `ray_distance_threshold x score_min` (these interact via the soft scoring kernel), then sequential carry-forward for remaining params (`eviction_reproj_threshold`, `leiden_resolution`, `early_k`)
- **Reconstruction:** 1D grid per param, sequential carry-forward (`outlier_threshold` first, then `n_points`)
- **No early stopping:** Always sweep all parameter stages regardless of intermediate results
- **Grids are hardcoded defaults:** Use `ASSOCIATION_DEFAULT_GRID` and `RECONSTRUCTION_DEFAULT_GRID` from `evaluation/stages/`. No CLI flags for custom ranges — researchers edit the code if needed

### Two-tier validation
- **Fast sweep:** Tier 1 only (`skip_tier2=True`) at `--n-frames` count (default 30)
- **Top-N validation:** Full Tier 1 + Tier 2 at `--n-frames-validate` count (default 100) for the top candidates
- **Re-run target stage only** during validation — upstream stage caches are reused, only the swept stage is re-executed with candidate params at the higher frame count
- **Default `--top-n` is 3**

### Output and config diff
- **Winner vs baseline only:** One clean before/after comparison table (yield, mean error, max error, singleton rate, tier 2 stability) with deltas
- **2D yield matrix:** Print the joint grid yield % matrix for association sweeps (shows parameter interaction patterns)
- **Progress lines always:** Print per-combo result line as each completes (yield, error, etc.) — no quiet mode needed
- **Config diff as YAML snippet:** Print a ready-to-paste YAML block showing only changed keys and new values, matching the config.yaml structure

### CLI design
- **Command:** `aquapose tune --stage association` or `aquapose tune --stage reconstruction`
- **`--config` / `-c` required:** Path to the run-generated config YAML (exhaustive, not the minimalist user config). The config's parent directory IS the run directory
- **No --run or --resume-from:** The orchestrator infers the run directory from the config file's parent path and auto-discovers stage cache pickles from the `stages/` subdirectory
- **Frame count flags:** `--n-frames` (fast sweep, default 30), `--n-frames-validate` (thorough validation, default 100)
- **`--top-n`** (default 3): Number of candidates for full validation

### Claude's Discretion
- Internal TuningOrchestrator class design and method decomposition
- How stage cache discovery works (glob pattern, naming convention)
- Scoring function design for ranking candidates
- Console formatting details (column widths, separators)

</decisions>

<specifics>
## Specific Ideas

- The config path provided to `aquapose tune` is always the auto-generated exhaustive config from a pipeline run (saved in the run directory), not the minimalist user-facing config. The config's parent path is the run directory itself.
- Preserve the 2D yield matrix from `tune_association.py` — it's valuable for understanding parameter interactions
- Association sweep scoring should prioritize fish yield (as primary) and reprojection error (as secondary), matching the existing `_compute_score` pattern in `tune_association.py`

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluation/stages/association.py`: `ASSOCIATION_DEFAULT_GRID`, `AssociationMetrics`, `evaluate_association()` — grid definitions and metrics computation already exist
- `evaluation/stages/reconstruction.py`: `RECONSTRUCTION_DEFAULT_GRID`, `ReconstructionMetrics`, `evaluate_reconstruction()` — grid definitions and metrics computation already exist
- `evaluation/harness.py`: `generate_fixture()`, `run_evaluation()`, `EvalResults` — fixture generation and evaluation orchestration
- `core/context.py`: `load_stage_cache()`, `StaleCacheError` — stage cache loading infrastructure already in place
- `cli.py`: Click-based CLI with `@cli.command()` pattern, `--config` flag pattern, config override parsing

### Established Patterns
- CLI commands are thin wrappers: `cli.py` parses args, calls into engine/core modules
- Stage caches are pickle files in `stages/` subdirectory of run directory
- Config loading: `load_config(yaml_path, cli_overrides)` handles merging
- Evaluation flow: load fixture/cache -> run stage with params -> compute metrics -> compare

### Integration Points
- New `tune` command registered on the `cli` Click group via `@cli.command("tune")`
- TuningOrchestrator lives in a new module (likely `engine/tuning.py` or `evaluation/tuning.py`)
- Uses `load_stage_cache()` to skip upstream stages
- Uses stage evaluators from `evaluation/stages/` for metrics
- Retirement of `scripts/tune_association.py` and `scripts/tune_threshold.py` after feature parity

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 49-tuningorchestrator-and-aquapose-tune-cli*
*Context gathered: 2026-03-03*
