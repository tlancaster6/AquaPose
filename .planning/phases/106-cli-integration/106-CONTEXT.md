# Phase 106: CLI Integration - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire all ReID capabilities into an `aquapose reid` command group with subcommands (`embed`, `repair`, `mine-crops`, `fine-tune`) following existing CLI patterns. Migrate the existing top-level `mine-reid-crops` command into the group. No new ReID algorithms — this phase is purely CLI surface.

</domain>

<decisions>
## Implementation Decisions

### Command group structure
- Create `aquapose reid` as a Click group with subcommands: `embed`, `repair`, `mine-crops`, `fine-tune`
- Group definition lives in `core/reid/cli.py`, imported and registered in main `cli.py` via `cli.add_command()`
- Project context inherited from parent `cli` group — each subcommand uses `get_project_dir(ctx)` and `resolve_run()` individually (same pattern as other groups)

### Existing command migration
- Move `mine-reid-crops` (currently top-level at `cli.py:705`) into the `reid` group as `reid mine-crops`
- Remove the old top-level `mine-reid-crops` command entirely — no backward-compatibility shim
- Subcommand name: `mine-crops` (preserves the mining metaphor)

### Fine-tune CLI surface
- Add `aquapose reid fine-tune -p YH` as a proper CLI subcommand
- Implements the full workflow from the standalone script: train → evaluate AUC → if gate passes, re-embed with fine-tuned model
- Reports AUC and gate pass/fail status
- Delete `scripts/train_reid_head.py` after CLI command is working — it's fully superseded

### Embed model selection
- `reid embed` accepts `--weights PATH` flag to use a fine-tuned model; omit for default MegaDescriptor-T (zero-shot)
- `--overwrite` flag required to overwrite existing `embeddings.npz` (fail with message if exists without flag)
- After embedding, automatically compute and print zero-shot ReID metrics (within/between similarity, rank-1, mAP) using existing `compute_reid_metrics()` from `core/reid/eval.py`

### Repair subcommand
- `reid repair` wraps Phase 105's swap detection and repair module (not yet built)
- Follows same patterns: resolve run, load data, run detection/repair, write `midlines_reid.h5`, print summary

### Claude's Discretion
- Exact CLI option names and help text for `fine-tune` parameters (epochs, lr, AUC gate threshold)
- Whether `fine-tune` exposes all `ReidTrainingConfig` fields as CLI options or just the commonly-changed ones
- Progress reporting style (click.echo vs progress bar)
- Error handling patterns for missing prerequisites (e.g., no `embeddings.npz` when running `fine-tune`)

</decisions>

<specifics>
## Specific Ideas

- Command naming: `mine-crops` preserves the "mining" metaphor rather than the more generic `train-data`
- The full fine-tune workflow (train → AUC gate → conditional re-embed) should be a single command, not composable steps — matches the standalone script's ergonomics
- `embed` should always show eval metrics after running — zero overhead, good sanity check

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `EmbedRunner` (`core/reid/runner.py`): Complete batch embedding logic, just needs CLI wiring
- `TrainingDataMiner` + `MinerConfig` (`core/reid/miner.py`): Already fully functional, currently wired as top-level command
- `compute_reid_metrics()` (`core/reid/eval.py`): Zero-shot eval, ready to call after embedding
- `ReidTrainingConfig`, `build_feature_cache`, `train_reid_head` (`training/reid_training.py`): Full training module
- `ReidConfig` (`engine/config.py:379`): Embedding config dataclass (model_name, batch_size, crop_size, device, embedding_dim)
- `resolve_run()`, `get_project_dir()` (`cli_utils.py`): Standard run/project resolution

### Established Patterns
- Click groups defined in separate modules, imported via `cli.add_command()` (see `train_group`, `data_group`, `prep_group`, `pseudo_label_group`)
- Lazy imports inside command functions for fast CLI startup
- `click.echo()` for progress output
- `@click.pass_context` + `ctx.obj` for project context propagation
- `--overwrite/--no-overwrite` pattern used by `mine-reid-crops`

### Integration Points
- `cli.py`: Import and register `reid_group` via `cli.add_command()`
- Remove `mine_reid_crops_cmd` command definition (lines 705-807) from `cli.py`
- `scripts/train_reid_head.py`: Delete after CLI `fine-tune` is working

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 106-cli-integration*
*Context gathered: 2026-03-25*
