# Phase 66: Training Run Management - Research

**Researched:** 2026-03-05
**Domain:** Training run organization, metrics comparison, provenance tracking
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Runs live under the project directory: `~/aquapose/projects/{project}/training/{model_type}/run_{timestamp}/`
- Model types get their own subdirectory level: `training/obb/`, `training/seg/`, `training/pose/`
- Run directories named with timestamp only: `run_20260305_143022` (matches pipeline run naming convention)
- Each run contains: frozen training config YAML, Ultralytics results.csv, and a pretty-printed summary.json with best metrics, dataset path, and training duration
- `aquapose train compare` command with two modes: auto-discover all runs under `training/{model_type}/` by default, or accept explicit run paths as positional args
- Primary metrics: Ultralytics training metrics (mAP50, mAP50-95, precision, recall, best epoch, training time)
- Additional metrics: per-validation-subgroup breakdowns (manual vs pseudo-label, consensus vs gap-fill) when subgroup info is available
- Output: terminal table with Click ANSI styling (bold/green for best values per column), plus `--csv` flag for CSV export
- No new dependencies -- use Click's built-in `click.style()` for highlighting
- Record dataset path in run's config snapshot
- Copy or reference confidence.json sidecar and dataset.yaml from the pseudo-label dataset so the exact round, thresholds, and source mix are preserved
- Record parent weights path when `--weights` is specified (transfer learning lineage: round 1 -> round 2 -> round 3)
- Comparison table shows source pipeline run + source type breakdown (e.g. "80% consensus, 20% gap-fill")
- summary.json is pretty-printed for human readability
- Same existing train commands (`aquapose train yolo-obb`, `seg`, `pose`) -- no special "retrain" command
- `--config` flag (required) replaces `--output-dir` -- training commands read the project config to auto-generate timestamped run directory
- Optional `--tag` flag for human-readable notes (e.g. "round2-high-conf"), stored in summary.json
- After training completes, print suggested next steps (compare command, path to best weights)

### Claude's Discretion
- Exact summary.json schema
- How to extract and aggregate Ultralytics metrics from results.csv
- Terminal table column widths and formatting details
- How subgroup validation metrics are computed from dataset metadata

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | Each training run outputs to a unique timestamped directory with config snapshot and metric summary | Run directory structure pattern, summary.json schema, config snapshot logic via YAML dump |
| TRAIN-02 | User can run `aquapose train compare` to generate cross-run comparison (Ultralytics training metrics + aquapose eval pipeline metrics) | Ultralytics results.csv column format (verified from actual file), Click styling for tables, CSV export pattern |
| TRAIN-03 | Comparison tracks which pseudo-label round and confidence thresholds were used per run | Provenance via dataset.yaml + confidence.json references in summary.json, parent weights lineage tracking |
</phase_requirements>

## Summary

This phase modifies the existing training CLI commands (`yolo-obb`, `seg`, `pose`) to output to structured, timestamped run directories under the project tree, and adds a new `compare` subcommand for cross-run analysis. The existing training wrappers in `training/yolo_obb.py`, `yolo_seg.py`, and `yolo_pose.py` currently take an `output_dir` parameter and write to `_ultralytics/train/` subdirectories. The key change is replacing the `--output-dir` CLI flag with `--config` (project config path), from which the run directory is auto-generated.

The Ultralytics results.csv format has been verified from actual project files. For OBB training, columns include: `epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,train/angle_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,val/angle_loss,lr/pg0,lr/pg1,lr/pg2`. The compare command needs to parse this CSV, extract best-epoch metrics, and format a terminal table. No external dependencies are needed -- Click's `click.style()` provides ANSI color/bold for highlighting best values.

Provenance tracking leverages existing artifacts: pseudo-label generation already writes `confidence.json` sidecars and `dataset.yaml` per subset. The training run's summary.json references these by path, enabling full lineage from run -> dataset -> pseudo-label round -> pipeline run -> confidence thresholds. Parent weights tracking enables transfer learning lineage across iterative training rounds.

**Primary recommendation:** Modify existing `train_yolo_*` wrapper functions to accept a run-management context (run directory, config snapshot, tag), write summary.json post-training, and add a `compare` command to the `train_group` Click group.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| click | (existing) | CLI commands, `click.style()` for ANSI table formatting | Already used throughout project CLI |
| pyyaml | (existing) | Read/write config YAML snapshots | Already used in engine/config.py and pseudo_label_cli.py |
| csv (stdlib) | N/A | Parse Ultralytics results.csv | Standard library, no extra deps |
| json (stdlib) | N/A | Write/read summary.json and confidence.json | Standard library, no extra deps |
| datetime (stdlib) | N/A | Timestamp generation for run directories | Standard library |
| shutil (stdlib) | N/A | Copy confidence.json and dataset.yaml into run dir | Already used in training wrappers |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib (stdlib) | N/A | Path manipulation for run directory structure | Throughout |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Click ANSI styling | rich library | Rich gives prettier tables but adds a dependency; Click styling is sufficient and already available |
| csv stdlib | pandas | Pandas is overkill for reading a single CSV; csv module is simpler and already sufficient |

**Installation:**
No new dependencies required.

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/training/
├── cli.py              # Modified: --config replaces --output-dir, adds --tag, adds compare command
├── run_manager.py      # NEW: Run directory creation, config snapshot, summary.json write/read
├── yolo_obb.py         # Modified: accept run_dir instead of output_dir (or adapter in cli.py)
├── yolo_seg.py         # Modified: same pattern
├── yolo_pose.py        # Modified: same pattern
└── ...
```

### Pattern 1: Run Manager Module
**What:** Centralize run directory creation, config snapshotting, and summary generation in a dedicated module (`run_manager.py`).
**When to use:** Called by CLI commands before and after training.
**Why:** Avoids duplicating run-management logic across three training wrappers. The CLI commands call `create_run_dir()` to set up the directory, pass it to the existing `train_yolo_*` functions as the output_dir, then call `write_summary()` post-training.

```python
# Source: Project patterns (verified from codebase)
import json
import shutil
from datetime import datetime
from pathlib import Path

def create_run_dir(
    project_config_path: Path,
    model_type: str,  # "obb", "seg", "pose"
) -> Path:
    """Create timestamped run directory under project training tree.

    Returns the run directory path:
    ~/aquapose/projects/{project}/training/{model_type}/run_{timestamp}/
    """
    import yaml
    with open(project_config_path) as f:
        config = yaml.safe_load(f)

    project_dir = Path(config["project_dir"]).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_dir / "training" / model_type / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config(
    run_dir: Path,
    cli_args: dict,
    dataset_dir: Path | None = None,
) -> None:
    """Write frozen training config YAML to run directory."""
    import yaml
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        yaml.dump(cli_args, default_flow_style=False, sort_keys=False)
    )

    # Copy provenance files from dataset directory
    if dataset_dir is not None:
        for sidecar in ("confidence.json", "dataset.yaml"):
            src = dataset_dir / sidecar
            if src.exists():
                shutil.copy2(src, run_dir / f"dataset_{sidecar}")


def write_summary(
    run_dir: Path,
    results_csv_path: Path,
    training_args: dict,
    tag: str | None = None,
) -> None:
    """Parse Ultralytics results.csv and write summary.json."""
    best_metrics = parse_best_metrics(results_csv_path)
    summary = {
        "tag": tag,
        "dataset_path": str(training_args.get("data_dir", "")),
        "parent_weights": str(training_args.get("weights", "")),
        "model_variant": training_args.get("model", ""),
        "training_duration_seconds": best_metrics.get("total_time", 0),
        "best_epoch": best_metrics.get("best_epoch", -1),
        "metrics": {
            "mAP50": best_metrics.get("mAP50", 0),
            "mAP50-95": best_metrics.get("mAP50-95", 0),
            "precision": best_metrics.get("precision", 0),
            "recall": best_metrics.get("recall", 0),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
```

### Pattern 2: Ultralytics results.csv Parsing
**What:** Parse results.csv to extract best-epoch metrics. Best epoch = epoch with highest mAP50-95.
**When to use:** Post-training summary generation and compare command.

```python
# Source: Verified from actual results.csv in ~/aquapose/projects/YH/models/obb3/
import csv
from pathlib import Path

def parse_best_metrics(results_csv: Path) -> dict:
    """Extract best-epoch metrics from Ultralytics results.csv.

    The CSV has columns with leading spaces in headers (Ultralytics quirk).
    Best epoch is determined by highest metrics/mAP50-95(B).
    """
    rows = []
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys (Ultralytics adds leading spaces)
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(cleaned)

    if not rows:
        return {}

    # Find best epoch by mAP50-95
    best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0)))

    return {
        "best_epoch": int(float(best_row.get("epoch", 0))),
        "mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(best_row.get("metrics/mAP50-95(B)", 0)),
        "precision": float(best_row.get("metrics/precision(B)", 0)),
        "recall": float(best_row.get("metrics/recall(B)", 0)),
        "total_time": sum(float(r.get("time", 0)) for r in rows),
    }
```

### Pattern 3: Compare Command Table Formatting
**What:** Terminal table with Click ANSI styling. Bold/green for best value per column.
**When to use:** `aquapose train compare` output.

```python
# Source: Click docs for click.style()
import click

def format_comparison_table(runs: list[dict]) -> str:
    """Format runs as a terminal table with best values highlighted."""
    columns = ["Run", "Tag", "mAP50", "mAP50-95", "Prec", "Recall", "Epoch", "Dataset"]

    # Find best value indices per metric column
    metric_cols = ["mAP50", "mAP50-95", "Prec", "Recall"]
    best_indices = {}
    for col in metric_cols:
        values = [r.get(col, 0) for r in runs]
        if values:
            best_indices[col] = values.index(max(values))

    # Format rows
    lines = []
    header = "  ".join(f"{c:>10}" for c in columns)
    lines.append(click.style(header, bold=True))

    for i, run in enumerate(runs):
        cells = []
        for col in columns:
            val = run.get(col, "")
            formatted = f"{val:>10}" if not isinstance(val, float) else f"{val:>10.4f}"
            if col in best_indices and best_indices[col] == i:
                formatted = click.style(formatted, fg="green", bold=True)
            cells.append(formatted)
        lines.append("  ".join(cells))

    return "\n".join(lines)
```

### Pattern 4: Import Boundary Compliance
**What:** The training module must not import from `aquapose.engine` or `aquapose.cli` at the module level. Use `importlib.import_module` for dynamic imports when loading project config.
**When to use:** When the `--config` flag needs to read project config to resolve project_dir.
**Precedent:** `pseudo_label_cli.py` already uses this pattern: `_engine_config = importlib.import_module("aquapose.engine.config")`.

```python
# For run_manager.py, use pyyaml directly (no engine import needed):
import yaml
with open(config_path) as f:
    config = yaml.safe_load(f)
project_dir = Path(config.get("project_dir", "")).expanduser().resolve()
```

**Key insight:** The run_manager module can read project config with raw pyyaml (`yaml.safe_load`) to extract `project_dir`, avoiding engine imports entirely. It does not need the full `PipelineConfig` dataclass -- only the `project_dir` field.

### Anti-Patterns to Avoid
- **Modifying train_yolo_* function signatures**: Keep the existing wrapper functions as-is. The CLI layer handles run directory creation and summary writing, then passes the run_dir as the existing `output_dir` parameter. This minimizes changes to tested code.
- **Importing engine.config in training/**: Use raw yaml.safe_load to read project config. The test `test_training_modules_do_not_import_engine` enforces this boundary.
- **Storing absolute paths in summary.json**: Store paths relative to the project directory where possible, or absolute paths that are clearly labeled. The existing codebase uses absolute paths (see args.yaml), so follow that convention.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ANSI terminal colors | Custom escape codes | `click.style(text, fg="green", bold=True)` | Click handles terminal compatibility |
| CSV parsing | Manual string splitting | `csv.DictReader` with header stripping | Handles quoting, commas in values |
| YAML serialization | Custom formatters | `yaml.dump(data, default_flow_style=False)` | Already used throughout project |
| Timestamp formatting | Custom format strings | `datetime.now().strftime("%Y%m%d_%H%M%S")` | Matches existing `run_*` naming convention |

**Key insight:** This phase is largely "plumbing" -- connecting existing Ultralytics outputs to a structured directory layout and comparison interface. The individual pieces (YAML, CSV, JSON, Click CLI) are all standard library or already-used dependencies.

## Common Pitfalls

### Pitfall 1: Ultralytics results.csv Header Whitespace
**What goes wrong:** Column headers in Ultralytics results.csv have leading whitespace (e.g., `"                   epoch"` not `"epoch"`).
**Why it happens:** Ultralytics formatting for aligned terminal display.
**How to avoid:** Strip whitespace from DictReader keys: `{k.strip(): v.strip() for k, v in row.items()}`.
**Warning signs:** `KeyError: 'epoch'` when accessing parsed CSV rows.

### Pitfall 2: Ultralytics `train()` Appends Suffix to Directory Name
**What goes wrong:** If a `_ultralytics/train/` directory already exists, Ultralytics creates `_ultralytics/train2/`, `_ultralytics/train3/`, etc.
**Why it happens:** Ultralytics auto-increments the `name` parameter to avoid overwriting.
**How to avoid:** Each training run gets a fresh timestamped directory, so `_ultralytics/train/` will always be new. Use the `results.save_dir` return value (as existing code already does) rather than hardcoding the path.
**Warning signs:** Summary pointing to wrong results.csv.

### Pitfall 3: Import Boundary Violation
**What goes wrong:** Tests fail with "Import boundary violations found" if training modules import from `aquapose.engine`.
**Why it happens:** `test_training_modules_do_not_import_engine` in `test_training_cli.py` enforces the boundary via AST analysis.
**How to avoid:** Use `yaml.safe_load` to read project config directly. Only use `importlib.import_module` in CLI functions (not module-level imports), following the `pseudo_label_cli.py` pattern.
**Warning signs:** Test failures in `test_training_cli.py::test_training_modules_do_not_import_engine`.

### Pitfall 4: CLI Flag Backwards Compatibility
**What goes wrong:** Replacing `--output-dir` with `--config` breaks existing workflows.
**Why it happens:** Users may have scripts using `--output-dir`.
**How to avoid:** The decision is to replace (not add alongside). Ensure the help text is clear. Consider making `--data-dir` optional too since the config can specify the dataset path. However, `--data-dir` should remain required since it points to the assembled dataset, not a project config field.
**Warning signs:** User confusion about which flags to use.

### Pitfall 5: results.csv Column Names Vary by Task Type
**What goes wrong:** OBB has `train/angle_loss`, seg has `train/seg_loss`, pose has `train/pose_loss`. Hardcoding column names breaks for some model types.
**Why it happens:** Ultralytics uses task-specific loss columns.
**How to avoid:** Focus on the common metric columns (`metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `metrics/precision(B)`, `metrics/recall(B)`) which are consistent across tasks. Ignore task-specific loss columns in the comparison table.
**Warning signs:** KeyError when comparing runs of different model types.

## Code Examples

### Reading Project Config Without Engine Imports
```python
# Source: Pattern from training/pseudo_label_cli.py (verified in codebase)
import yaml
from pathlib import Path

def resolve_project_dir(config_path: Path) -> Path:
    """Extract project_dir from project config YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    project_dir = config.get("project_dir", "")
    if not project_dir:
        raise ValueError(f"project_dir not set in {config_path}")
    return Path(project_dir).expanduser().resolve()
```

### Discovering Runs for Comparison
```python
# Source: Pattern from codebase (run_* naming convention)
from pathlib import Path

def discover_runs(training_type_dir: Path) -> list[Path]:
    """Find all run_* directories under a training type directory.

    Args:
        training_type_dir: e.g. ~/aquapose/projects/YH/training/obb/

    Returns:
        Sorted list of run directories (by timestamp in name).
    """
    runs = sorted(
        d for d in training_type_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
    return runs
```

### Post-Training Next Steps Output
```python
# Source: CLI pattern from codebase
import click

def print_next_steps(run_dir: Path, model_type: str, best_weights: Path) -> None:
    """Print suggested next steps after training completes."""
    click.echo("")
    click.echo(click.style("Training complete!", bold=True))
    click.echo(f"  Run directory: {run_dir}")
    click.echo(f"  Best weights:  {best_weights}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  Compare runs:  aquapose train compare --config {run_dir.parent.parent.parent / 'config.yaml'} --model-type {model_type}")
    click.echo(f"  Use weights:   aquapose train {model_type} --config ... --weights {best_weights}")
```

### summary.json Schema (Claude's Discretion)
```python
# Recommended schema
summary_schema = {
    "run_id": "run_20260305_143022",           # directory name
    "tag": "round2-high-conf",                  # optional human note
    "model_type": "obb",                        # obb/seg/pose
    "model_variant": "yolo26n-obb",             # YOLO model variant
    "parent_weights": "/path/to/weights.pt",    # transfer learning lineage (null if from scratch)
    "dataset_path": "/path/to/assembled/dataset", # assembled dataset used
    "dataset_sources": {                         # from dataset's confidence.json/metadata
        "n_consensus": 1200,
        "n_gap": 300,
        "n_manual": 150,
        "consensus_fraction": 0.8,
        "gap_fraction": 0.2,
        "consensus_threshold": 0.5,
        "gap_threshold": 0.3,
        "pipeline_run": "run_20260304_120854",
    },
    "training_config": {                         # CLI args snapshot
        "epochs": 300,
        "batch_size": 8,
        "imgsz": 640,
        "patience": 50,
        "mosaic": 0.0,
    },
    "metrics": {                                 # best-epoch metrics from results.csv
        "best_epoch": 157,
        "mAP50": 0.923,
        "mAP50-95": 0.671,
        "precision": 0.891,
        "recall": 0.876,
    },
    "training_duration_seconds": 3420.5,
    "created": "2026-03-05T14:30:22",
}
```

### Extracting Dataset Provenance
```python
# Source: Pattern from pseudo_label_cli.py confidence.json output
import json
from pathlib import Path

def extract_dataset_provenance(dataset_dir: Path) -> dict:
    """Extract source breakdown from assembled dataset metadata.

    Reads confidence.json and pseudo_val_metadata.json from the dataset
    directory to determine what pseudo-label round and thresholds were used.
    """
    provenance: dict = {}

    # Read dataset.yaml for basic info
    dataset_yaml = dataset_dir / "dataset.yaml"
    if dataset_yaml.exists():
        import yaml
        with open(dataset_yaml) as f:
            provenance["dataset_config"] = yaml.safe_load(f)

    # Count images by source from directory structure
    train_images = list((dataset_dir / "images" / "train").glob("*.jpg"))
    provenance["n_train_images"] = len(train_images)

    # Read pseudo_val_metadata.json if exists (from Phase 65 assembly)
    meta_path = dataset_dir / "pseudo_val_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        sources = [entry.get("source", "unknown") for entry in meta.values()]
        provenance["n_consensus"] = sources.count("consensus")
        provenance["n_gap"] = sources.count("gap")

    return provenance
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `--output-dir` flag (manual path) | `--config` flag (auto-generate timestamped dir) | This phase | Consistent run organization, no manual path management |
| No cross-run comparison | `aquapose train compare` CLI command | This phase | Enables iterative training workflow |
| No provenance tracking | summary.json with dataset lineage | This phase | Full traceability from model to pseudo-label round |

## Open Questions

1. **Pose results.csv column names**
   - What we know: OBB has `train/angle_loss` and `metrics/mAP50(B)`. Seg likely has `train/seg_loss`.
   - What's unclear: Exact pose results.csv columns (may include `train/pose_loss`, `train/kobj_loss`).
   - Recommendation: Parse dynamically -- read whatever columns exist in the CSV, focus on the common `metrics/*` columns for comparison. Task-specific loss columns can be shown but not compared across model types.

2. **Subgroup validation metrics computation**
   - What we know: The assembled dataset from Phase 65 has a `pseudo_val_metadata.json` sidecar with per-image source breakdown.
   - What's unclear: How to run Ultralytics validation on specific subgroups (manual vs pseudo-label val). Ultralytics val mode takes a single dataset.yaml.
   - Recommendation: For v1, show only overall Ultralytics metrics in the comparison table. Subgroup breakdowns would require running `model.val()` with separate dataset.yaml files pointing to each subgroup -- defer to implementation if feasible.

## Sources

### Primary (HIGH confidence)
- Actual Ultralytics results.csv from `/home/tlancaster6/aquapose/projects/YH/models/obb3/_ultralytics/train/results.csv` -- verified column format
- Actual Ultralytics args.yaml from `/home/tlancaster6/aquapose/projects/YH/models/obb3/_ultralytics/train/args.yaml` -- verified config snapshot format
- Project codebase: `training/cli.py`, `training/yolo_obb.py`, `training/pseudo_label_cli.py` -- verified existing patterns

### Secondary (MEDIUM confidence)
- [Ultralytics training docs](https://docs.ultralytics.com/modes/train/) -- results.csv format documentation
- [Ultralytics GitHub issue #17555](https://github.com/ultralytics/ultralytics/issues/17555) -- results.csv column details

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new dependencies, all stdlib or existing
- Architecture: HIGH - Straightforward CLI + file I/O, patterns well-established in codebase
- Pitfalls: HIGH - Verified from actual Ultralytics output files in project

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (stable domain, standard patterns)
