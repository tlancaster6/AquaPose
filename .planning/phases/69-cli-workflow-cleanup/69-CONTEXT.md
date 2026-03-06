# Phase 69: CLI Workflow Cleanup - Context

**Gathered:** 2026-03-06
**Status:** Partial — needs further discussion

<domain>
## Phase Boundary

Reorganize the AquaPose CLI from its current module-oriented structure into a workflow-oriented structure with project-aware path resolution. Reduce the number of explicit path arguments by adopting a canonical project root convention.

This phase depends on Phase 68 (training data storage and tracking) — data organization decisions will directly affect CLI commands and their defaults.

</domain>

<decisions>
## Implementation Decisions

### Project resolution
- Canonical project root: `~/aquapose/projects/{name}/` (hardcoded, no env var override for now)
- `aquapose init <name>` creates `~/aquapose/projects/<name>/` with scaffold
- `aquapose --project <name>` resolves to `~/aquapose/projects/<name>/` for all other commands
- Auto-detection from CWD: walk up looking for `config.yaml` if `--project` not provided
- `--project` accepts a project name (not a path), e.g. `--project YH`

### Command reorganization
- Current structure organized by code module (training/, prep/); proposed structure organized by workflow stage
- `init` is exempt from `--project` resolution (it creates the project)
- `prep` commands (generate-luts, calibrate-keypoints) are both pipeline run prerequisites — keep them grouped
- `smooth-z` is a post-processing utility; may move under a subgroup or stay top-level (TBD)
- `train augment-elastic` is data augmentation, not model training — placement TBD (depends on Phase 68 data org)
- `train compare` evaluates training runs; `eval` evaluates pipeline runs — two "evaluate" commands in different namespaces

### Run shorthand
- Commands that take a run directory should accept shorthand: timestamp suffix (`20260306_075925`), or omit for latest
- Negative indexing (`-1` for latest, `-2` for second-latest) under consideration

### Default path conventions
- `--config` replaced by project resolution on most commands
- `--data-dir` on train commands defaults to `training_data/{model_type}/` within project
- `pseudo-label generate` defaults to latest run if no run specified

</decisions>

<specifics>
## Specific Ideas

### Proposed command tree
```
aquapose [--project NAME]
├── init <name>                    # Create project (no --project needed)
├── run                            # Run pipeline
├── eval [RUN]                     # Evaluate pipeline run
├── tune [RUN]                     # Tune params from run
├── viz [RUN]                      # Visualize run
├── prep
│   ├── generate-luts              # Generate LUTs from project calibration
│   └── calibrate-keypoints        # Compute keypoint t-values
├── pseudo-label
│   ├── generate [RUN]             # Generate from run diagnostics
│   ├── assemble                   # Assemble training dataset
│   └── inspect [DATASET]          # Visualize labels
├── train
│   ├── obb [--data-dir DATASET]   # Train OBB (defaults to training_data/obb)
│   ├── seg [--data-dir DATASET]   # Train seg
│   ├── pose [--data-dir DATASET]  # Train pose
│   ├── augment-elastic            # Elastic deformation augmentation
│   └── compare                    # Compare training runs
└── smooth-z [RUN]                 # Post-process z
```

### Current pain points
- Almost every command takes `--config path/to/config.yaml` redundantly
- Users must stitch together output paths from one command as input to the next
- Commands organized by code module rather than workflow stage
- No concept of "current project" — every invocation is standalone

</specifics>

<code_context>
## Existing Code Insights

### CLI files
- `src/aquapose/cli.py` — main Click group, top-level commands (run, init-config, eval, tune, viz, smooth-z)
- `src/aquapose/training/cli.py` — `train` subgroup (yolo-obb, seg, pose, augment-elastic, compare)
- `src/aquapose/training/prep.py` — `prep` subgroup (calibrate-keypoints, generate-luts)
- `src/aquapose/training/pseudo_label_cli.py` — `pseudo-label` subgroup (generate, inspect, assemble)

### Entry point
- `pyproject.toml`: `aquapose` console script -> `aquapose.cli:main`

### Framework
- All CLI commands use Click exclusively
- Subgroups attached via `cli.add_command()` in main cli.py

### Established patterns
- Config-centric: most commands read `config.yaml` for path resolution
- Paths in config are relative to config file's parent directory
- `init-config` currently takes a NAME argument and creates scaffold at `~/aquapose/projects/{name}/`
- Run directories are timestamped: `runs/run_{YYYYMMDD_HHMMSS}/`

</code_context>

<deferred>
## Deferred Ideas

- `AQUAPOSE_HOME` env var override for non-canonical project locations — add only if someone needs it
- Interactive project selector (TUI) when `--project` omitted and CWD is not inside a project
- `aquapose projects list` command to show all projects

</deferred>

---

*Phase: 69-cli-workflow-cleanup*
*Context gathered: 2026-03-06*
