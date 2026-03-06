# Phase 69: CLI Workflow Cleanup - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Reorganize the AquaPose CLI from its current module-oriented structure into a workflow-oriented structure with project-aware path resolution. Reduce the number of explicit path arguments by adopting a canonical project root convention. Consolidate overlapping commands after Phase 68's data store landed.

This phase depends on Phase 68 (training data storage and tracking).

</domain>

<decisions>
## Implementation Decisions

### Project resolution
- Canonical project root: `~/aquapose/projects/{name}/` (hardcoded, no env var override for now)
- `aquapose init <name>` creates `~/aquapose/projects/<name>/` with scaffold (renamed from `init-config`)
- `aquapose --project <name>` resolves to `~/aquapose/projects/<name>/` for all other commands
- Auto-detection from CWD: walk up looking for `config.yaml` if `--project` not provided
- `--project` accepts a project name (not a path), e.g. `--project YH`
- CWD detection only — no hidden state file, no "last used project" memory
- `init` is exempt from `--project` resolution (it creates the project)

### Init scaffold
- Rename `init-config` to `init`
- Scaffold includes `training_data/obb/` and `training_data/pose/` directories (Phase 68 store convention)
- Full scaffold: config.yaml, runs/, models/, geometry/, videos/, training_data/obb/, training_data/pose/

### Data + pseudo-label consolidation
- `pseudo-label assemble` is removed — `data assemble` is the single assembly path
- `dataset_assembly.py` module is deleted entirely (clean break, store-based assembly replaces it)
- `pseudo-label generate` and `pseudo-label inspect` stay in the `pseudo-label` group (they produce labels from pipeline runs, separate concern from the store)
- `pseudo-label generate` does NOT auto-import into store — user runs `data import` as a separate step (transparency, user controls what gets imported)
- Workflow: `pseudo-label generate [RUN]` -> `data import --store pose --source pseudo` -> `data assemble`

### augment-elastic
- Remove the standalone `train augment-elastic` CLI command (redundant with `data import --augment`)
- Keep `elastic_deform.py` module with `generate_variants()` as library code (used by `data import --augment`)
- Drop preview grid functionality from CLI (was for A/B experiment, not needed in store workflow)

### Run shorthand
- Commands that take run dirs (`eval`, `viz`, `tune`, `smooth-z`) accept: timestamp suffix (`20260306_075925`), `latest` keyword, negative index (`-1` for latest, `-2` for second-latest), or full path
- No argument = latest run (resolved relative to project's `runs/` dir)
- Run identifier is a positional argument: `aquapose eval 20260306`, `aquapose eval latest`, `aquapose eval -2`
- `tune` unified with eval/viz pattern: `aquapose tune --stage association latest` (resolves run dir from project, finds config inside run dir)

### Command reorganization
- `prep` commands (generate-luts, calibrate-keypoints) stay grouped under `prep`
- `smooth-z` stays top-level (post-processing utility, takes run shorthand)
- `train` keeps: obb, seg, pose, compare (augment-elastic removed)
- `data` keeps all Phase 68 commands: import, convert, assemble, status, list, exclude, include, remove
- `pseudo-label` keeps: generate, inspect (assemble removed)

### Default path conventions
- `--config` replaced by project resolution on most commands
- `--data-dir` on train commands defaults to `training_data/{model_type}/` within project
- `pseudo-label generate` defaults to latest run if no run specified

### Claude's Discretion
- Implementation details of the run resolution utility function
- How to handle Click argument parsing for negative indices (may conflict with flags)
- Whether to use Click's `context_settings` or a custom decorator for project resolution
- Error messages when project resolution fails

</decisions>

<specifics>
## Specific Ideas

### Proposed command tree (updated post-Phase 68)
```
aquapose [--project NAME]
+-- init <name>                    # Create project (no --project needed)
+-- run                            # Run pipeline
+-- eval [RUN]                     # Evaluate pipeline run (latest if omitted)
+-- tune --stage STAGE [RUN]       # Tune params from run
+-- viz [RUN]                      # Visualize run
+-- smooth-z [RUN]                 # Post-process z
+-- prep
|   +-- generate-luts              # Generate LUTs from project calibration
|   +-- calibrate-keypoints        # Compute keypoint t-values
+-- pseudo-label
|   +-- generate [RUN]             # Generate from run diagnostics
|   +-- inspect [DATASET]          # Visualize labels
+-- data
|   +-- import                     # Ingest YOLO dirs into store
|   +-- convert                    # COCO -> YOLO format
|   +-- assemble                   # Build training dataset (symlinks)
|   +-- status                     # Cross-store summary
|   +-- list                       # Per-store listing
|   +-- exclude                    # Soft-delete samples
|   +-- include                    # Reverse exclude
|   +-- remove --purge             # Hard-delete samples
+-- train
    +-- obb [--data-dir DATASET]   # Train OBB (defaults to training_data/obb)
    +-- seg [--data-dir DATASET]   # Train seg
    +-- pose [--data-dir DATASET]  # Train pose
    +-- compare                    # Compare training runs
```

### Current pain points being solved
- Almost every command takes `--config path/to/config.yaml` redundantly
- Users must stitch together output paths from one command as input to the next
- Commands organized by code module rather than workflow stage
- No concept of "current project" — every invocation is standalone
- Two `assemble` commands with overlapping functionality
- `augment-elastic` CLI redundant with store-based augmentation

</specifics>

<code_context>
## Existing Code Insights

### CLI files
- `src/aquapose/cli.py` — main Click group, top-level commands (run, init-config, eval, tune, viz, smooth-z)
- `src/aquapose/training/cli.py` — `train` subgroup (yolo-obb, seg, pose, augment-elastic, compare)
- `src/aquapose/training/prep.py` — `prep` subgroup (calibrate-keypoints, generate-luts)
- `src/aquapose/training/pseudo_label_cli.py` — `pseudo-label` subgroup (generate, inspect, assemble)
- `src/aquapose/training/data_cli.py` — `data` subgroup (import, convert, assemble, status, list, exclude, include, remove) — NEW from Phase 68
- `src/aquapose/training/dataset_assembly.py` — pseudo-label assembly module (TO BE DELETED)

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
- `run_manager.py` has `resolve_project_dir(config_path)` — existing pattern for project dir resolution

### Integration points
- Project resolution utility needs to be shared across all CLI modules
- Run resolution utility needs to scan `{project_dir}/runs/` and sort by timestamp
- `data_cli.py` commands already take `--config` — these need migration to `--project` resolution

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
