# Phase 66: Training Run Management - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Run organization, cross-run comparison, and iterative retraining support for YOLO model training. Users can track, compare, and iterate on training runs with full provenance of which pseudo-label round and thresholds produced each model.

</domain>

<decisions>
## Implementation Decisions

### Run directory structure
- Runs live under the project directory: `~/aquapose/projects/{project}/training/{model_type}/run_{timestamp}/`
- Model types get their own subdirectory level: `training/obb/`, `training/seg/`, `training/pose/`
- Run directories named with timestamp only: `run_20260305_143022` (matches pipeline run naming convention)
- Each run contains: frozen training config YAML, Ultralytics results.csv, and a pretty-printed summary.json with best metrics, dataset path, and training duration

### Comparison report
- `aquapose train compare` command with two modes: auto-discover all runs under `training/{model_type}/` by default, or accept explicit run paths as positional args
- Primary metrics: Ultralytics training metrics (mAP50, mAP50-95, precision, recall, best epoch, training time)
- Additional metrics: per-validation-subgroup breakdowns (manual vs pseudo-label, consensus vs gap-fill) when subgroup info is available
- Output: terminal table with Click ANSI styling (bold/green for best values per column), plus `--csv` flag for CSV export
- No new dependencies — use Click's built-in `click.style()` for highlighting

### Provenance tracking
- Record dataset path in run's config snapshot
- Copy or reference confidence.json sidecar and dataset.yaml from the pseudo-label dataset so the exact round, thresholds, and source mix are preserved
- Record parent weights path when `--weights` is specified (transfer learning lineage: round 1 -> round 2 -> round 3)
- Comparison table shows source pipeline run + source type breakdown (e.g. "80% consensus, 20% gap-fill")
- summary.json is pretty-printed for human readability

### Retraining workflow
- Same existing train commands (`aquapose train yolo-obb`, `seg`, `pose`) — no special "retrain" command
- `--config` flag (required) replaces `--output-dir` — training commands read the project config to auto-generate timestamped run directory
- Optional `--tag` flag for human-readable notes (e.g. "round2-high-conf"), stored in summary.json
- After training completes, print suggested next steps (compare command, path to best weights)

### Claude's Discretion
- Exact summary.json schema
- How to extract and aggregate Ultralytics metrics from results.csv
- Terminal table column widths and formatting details
- How subgroup validation metrics are computed from dataset metadata

</decisions>

<specifics>
## Specific Ideas

- Directory structure mirrors how pipeline runs already live under `projects/{project}/runs/`
- The confidence.json sidecars from pseudo-label generation (Phase 63-64) already contain per-sample provenance — leverage these directly
- Post-training output should guide the user through the iterative workflow with actionable commands

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `training/cli.py`: Existing Click commands for `yolo-obb`, `seg`, `pose` — these will be modified to add `--config` and `--tag` flags
- `training/common.py`: `MetricsLogger` (CSV writer) — used by custom training, not by Ultralytics wrappers
- `training/yolo_obb.py` / `yolo_seg.py` / `yolo_pose.py`: Ultralytics wrappers that produce `_ultralytics/train/` directories with `results.csv`, `args.yaml`, weights
- `training/pseudo_label_cli.py`: Generates `confidence.json` sidecars and `dataset.yaml` per subset (consensus/gap)

### Established Patterns
- CLI uses Click with `@click.group` / `@click.command` decorators
- Pipeline runs use `run_{timestamp}` naming convention
- Project configs at `~/aquapose/projects/{project}/config.yaml`
- Ultralytics training outputs to `_ultralytics/train/` subdirectory with standardized file layout

### Integration Points
- Training CLI commands in `training/cli.py` — add `--config`, `--tag` flags, replace `--output-dir`
- New `compare` subcommand under the `train` Click group
- `engine/config.py` `load_config()` for reading project config to resolve project directory
- Pseudo-label confidence.json and dataset.yaml files as provenance sources

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 66-training-run-management*
*Context gathered: 2026-03-05*
