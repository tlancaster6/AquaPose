# Phase 69: CLI Workflow Cleanup - Research

**Researched:** 2026-03-06
**Domain:** Click CLI refactoring, project resolution, command reorganization
**Confidence:** HIGH

## Summary

Phase 69 restructures the AquaPose CLI from a module-oriented layout with repetitive `--config` path arguments into a workflow-oriented design with project-aware path resolution. The core work involves: (1) a shared project resolution utility that all commands use, (2) a run resolution utility for shorthand run references, (3) removing redundant commands (`pseudo-label assemble`, `train augment-elastic`), (4) deleting the `dataset_assembly.py` module, (5) renaming `init-config` to `init` with an expanded scaffold, and (6) renaming `train yolo-obb` to `train obb`.

The codebase already uses Click 8.3.1 exclusively, with well-established patterns for CLI groups and subcommands. The existing `resolve_project_dir()` in `run_manager.py` provides a starting point, but needs to be generalized from config-path-based to name-based resolution with CWD fallback.

**Primary recommendation:** Implement as 4 plans -- (1) project resolution utility + `init` rename, (2) run resolution utility, (3) migrate all commands to use project/run resolution, (4) remove deprecated commands and module.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Canonical project root: `~/aquapose/projects/{name}/` (hardcoded, no env var override for now)
- `aquapose init <name>` creates `~/aquapose/projects/<name>/` with scaffold (renamed from `init-config`)
- `aquapose --project <name>` resolves to `~/aquapose/projects/<name>/` for all other commands
- Auto-detection from CWD: walk up looking for `config.yaml` if `--project` not provided
- `--project` accepts a project name (not a path), e.g. `--project YH`
- CWD detection only -- no hidden state file, no "last used project" memory
- `init` is exempt from `--project` resolution (it creates the project)
- Scaffold includes `training_data/obb/` and `training_data/pose/` directories
- Full scaffold: config.yaml, runs/, models/, geometry/, videos/, training_data/obb/, training_data/pose/
- `pseudo-label assemble` is removed -- `data assemble` is the single assembly path
- `dataset_assembly.py` module is deleted entirely
- `pseudo-label generate` and `pseudo-label inspect` stay in the `pseudo-label` group
- `pseudo-label generate` does NOT auto-import into store
- Workflow: `pseudo-label generate [RUN]` -> `data import --store pose --source pseudo` -> `data assemble`
- Remove standalone `train augment-elastic` CLI command
- Keep `elastic_deform.py` module with `generate_variants()` as library code
- Drop preview grid functionality from CLI
- Run shorthand accepts: timestamp suffix, `latest` keyword, negative index, or full path
- No argument = latest run
- Run identifier is a positional argument
- `tune` unified with eval/viz pattern: `aquapose tune --stage association latest`
- `prep` commands stay grouped under `prep`
- `smooth-z` stays top-level
- `train` keeps: obb, seg, pose, compare (augment-elastic removed)
- `data` keeps all Phase 68 commands
- `pseudo-label` keeps: generate, inspect (assemble removed)
- `--config` replaced by project resolution on most commands
- `--data-dir` on train commands defaults to `training_data/{model_type}/` within project
- `pseudo-label generate` defaults to latest run if no run specified

### Claude's Discretion
- Implementation details of the run resolution utility function
- How to handle Click argument parsing for negative indices (may conflict with flags)
- Whether to use Click's `context_settings` or a custom decorator for project resolution
- Error messages when project resolution fails

### Deferred Ideas (OUT OF SCOPE)
- `AQUAPOSE_HOME` env var override for non-canonical project locations
- Interactive project selector (TUI) when `--project` omitted and CWD is not inside a project
- `aquapose projects list` command to show all projects
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Click | 8.3.1 | CLI framework | Already used exclusively in all CLI modules |
| PyYAML | (bundled) | Config parsing | Already used for config.yaml resolution |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| click.testing.CliRunner | 8.3.1 | CLI test runner | All CLI tests use this pattern already |

No new dependencies needed. This is pure refactoring of existing Click code.

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/
├── cli.py                      # Main CLI group with --project option
├── cli_utils.py                # NEW: resolve_project(), resolve_run()
├── training/
│   ├── cli.py                  # train group (obb, seg, pose, compare)
│   ├── data_cli.py             # data group (import, convert, assemble, etc.)
│   ├── prep.py                 # prep group (generate-luts, calibrate-keypoints)
│   ├── pseudo_label_cli.py     # pseudo-label group (generate, inspect)
│   ├── run_manager.py          # Training run management (keeps resolve_project_dir)
│   ├── elastic_deform.py       # KEPT as library code (generate_variants)
│   └── dataset_assembly.py     # DELETED
```

### Pattern 1: Project Resolution via Click Context

**What:** A Click callback on the top-level group that resolves the project directory and stores it in `click.Context.obj`.

**When to use:** Every command that needs project_dir (all except `init`).

**Example:**
```python
# src/aquapose/cli_utils.py
from pathlib import Path
import click

AQUAPOSE_HOME = Path("~/aquapose/projects").expanduser()

def resolve_project(name: str | None) -> Path:
    """Resolve project directory from name or CWD.

    Args:
        name: Project name (e.g. 'YH'). If None, walks CWD upward
              looking for config.yaml.

    Returns:
        Resolved absolute path to project directory.

    Raises:
        click.ClickException: If project cannot be resolved.
    """
    if name is not None:
        project_dir = AQUAPOSE_HOME / name
        if not project_dir.exists():
            raise click.ClickException(
                f"Project '{name}' not found at {project_dir}"
            )
        return project_dir

    # Walk CWD upward looking for config.yaml
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "config.yaml").exists():
            return parent
        # Stop at home directory
        if parent == Path.home():
            break

    raise click.ClickException(
        "Could not resolve project. Use --project NAME or "
        "run from within a project directory."
    )
```

```python
# In cli.py, the main group:
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--project", "-p", default=None, help="Project name.")
@click.pass_context
def cli(ctx: click.Context, project: str | None) -> None:
    """AquaPose -- 3D fish pose estimation."""
    ctx.ensure_object(dict)
    # Don't resolve for init command (it creates the project)
    # Resolution is lazy -- only called when ctx.obj["project_dir"] is accessed
    ctx.obj["project_name"] = project
```

**Rationale for custom decorator over context_settings:** Click's `context_settings` does not support lazy resolution or command-specific exemptions. A decorator/callback approach gives full control over when resolution happens and what error messages look like.

### Pattern 2: Run Resolution Utility

**What:** A utility function that resolves run shorthand (timestamp, `latest`, negative index, full path) to a run directory path.

**When to use:** Commands that accept a RUN positional argument (eval, viz, tune, smooth-z, pseudo-label generate).

**Example:**
```python
import re

def resolve_run(
    run_ref: str | None,
    project_dir: Path,
    subdir: str = "runs",
) -> Path:
    """Resolve a run reference to an absolute path.

    Args:
        run_ref: One of:
            - None or "latest": most recent run
            - Timestamp suffix (e.g. "20260306_075925" or "20260306")
            - Negative index ("-1" = latest, "-2" = second-latest)
            - Full/relative path to run directory
        project_dir: Project root for relative resolution.
        subdir: Subdirectory under project_dir containing runs.

    Returns:
        Resolved absolute path to run directory.
    """
    runs_dir = project_dir / subdir

    if run_ref is None or run_ref == "latest":
        return _latest_run(runs_dir)

    # Negative index
    if run_ref.startswith("-") and run_ref[1:].isdigit():
        idx = int(run_ref)  # e.g. -1, -2
        sorted_runs = _sorted_runs(runs_dir)
        if abs(idx) > len(sorted_runs):
            raise click.ClickException(
                f"Only {len(sorted_runs)} runs exist, cannot use index {run_ref}"
            )
        return sorted_runs[idx]

    # Timestamp suffix (partial match)
    if re.match(r"^\d{8}", run_ref):
        matches = [d for d in runs_dir.iterdir() if d.is_dir() and run_ref in d.name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise click.ClickException(f"No run matching '{run_ref}' in {runs_dir}")
        raise click.ClickException(
            f"Ambiguous: {len(matches)} runs match '{run_ref}'"
        )

    # Full path
    path = Path(run_ref)
    if path.is_dir():
        return path.resolve()
    raise click.ClickException(f"Run directory not found: {run_ref}")


def _sorted_runs(runs_dir: Path) -> list[Path]:
    """List run directories sorted by name (timestamp order)."""
    if not runs_dir.exists():
        raise click.ClickException(f"Runs directory not found: {runs_dir}")
    runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )
    if not runs:
        raise click.ClickException(f"No runs found in {runs_dir}")
    return runs


def _latest_run(runs_dir: Path) -> Path:
    """Return the most recent run directory."""
    return _sorted_runs(runs_dir)[-1]
```

### Pattern 3: Click Negative Index Handling

**What:** Click interprets arguments starting with `-` as options. Negative indices like `-2` will fail argument parsing.

**How to handle:** Use `click.argument("run", default=None)` combined with `context_settings={"ignore_unknown_options": True}` on the command, OR accept negative indices as strings by quoting (`aquapose eval -- -2`). The simplest approach is to accept the run argument as a regular string and handle `-` prefixed values in `resolve_run()`.

**Recommendation:** Set `type=click.UNPROCESSED` on the RUN argument and handle all parsing in `resolve_run()`. This avoids Click interpreting `-2` as a flag. Alternatively, use `context_settings={"ignore_unknown_options": True}` on commands that accept run arguments.

```python
@cli.command("eval")
@click.argument("run", default=None, required=False)
@click.pass_context
def eval_cmd(ctx: click.Context, run: str | None) -> None:
    # Click may misinterpret -2 as a flag.
    # Workaround: use click.Context.args for unrecognized args
    ...
```

The simplest reliable approach: document that negative indices require `--` separator: `aquapose eval -- -2`. This is standard Unix convention and avoids fragile Click hacks.

### Pattern 4: Lazy Project Resolution Helper

**What:** A helper function that commands call to get project_dir from context, triggering resolution only when needed.

```python
def get_project_dir(ctx: click.Context) -> Path:
    """Get resolved project directory from Click context.

    Resolves lazily on first access.
    """
    if "project_dir" not in ctx.obj:
        from aquapose.cli_utils import resolve_project
        ctx.obj["project_dir"] = resolve_project(ctx.obj.get("project_name"))
    return ctx.obj["project_dir"]


def get_config_path(ctx: click.Context) -> Path:
    """Get config.yaml path from resolved project."""
    return get_project_dir(ctx) / "config.yaml"
```

### Anti-Patterns to Avoid
- **Importing engine.config in training modules:** The project has a strict training->engine import boundary. Use `yaml.safe_load` for config parsing in training code.
- **Eager project resolution on `init`:** The `init` command creates the project, so it must not try to resolve an existing one.
- **Breaking backward compatibility silently:** Commands that used to take `--config` should still work if `--config` is provided (graceful deprecation period or immediate removal per user preference -- CONTEXT says replace).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI argument parsing | Custom argparse | Click groups/commands | Already standard in this codebase |
| Run directory discovery | Custom file walker | `sorted(Path.iterdir())` with `run_` prefix filter | Run dirs follow `run_{timestamp}` convention |
| Config resolution | Complex path logic | `yaml.safe_load` + `project_dir` key | Existing pattern in `run_manager.py` |

## Common Pitfalls

### Pitfall 1: Click Negative Index Argument Conflict
**What goes wrong:** `aquapose eval -2` is interpreted by Click as an unknown option `-2`, not as an argument value.
**Why it happens:** Click's argument parser treats anything starting with `-` as an option/flag.
**How to avoid:** Either use `context_settings={"ignore_unknown_options": True}` on the specific command, or document `--` separator usage. The `ignore_unknown_options` approach requires collecting unprocessed args carefully.
**Warning signs:** `Error: No such option: -2` in Click output.

### Pitfall 2: Breaking Existing Tests
**What goes wrong:** Tests that invoke CLI commands with `--config` arguments will break when `--config` is removed.
**Why it happens:** Many test fixtures create `project_config` paths used in `--config` options.
**How to avoid:** Update all test invocations to use `--project` or CWD-based resolution. The existing test fixtures already create project directory structures, so they mainly need the `--project` or context-based approach.
**Warning signs:** Mass test failures on `Error: No such option: --config`.

### Pitfall 3: `dataset_assembly.py` Still Referenced
**What goes wrong:** Deleting `dataset_assembly.py` breaks imports in `training/__init__.py` and any test files.
**Why it happens:** The module is imported in `__init__.py` and its `assemble_dataset` function is in `__all__`.
**How to avoid:** Remove from `__init__.py`, `__all__`, and grep for all imports before deleting.
**Warning signs:** `ImportError: cannot import name 'assemble_dataset'`.

### Pitfall 4: `pseudo-label assemble` Command Still Registered
**What goes wrong:** After removing the `assemble` subcommand from `pseudo_label_group`, the `pseudo_label_cli.py` file still has the function and its imports.
**Why it happens:** Only the Click decorator is what registers the command; the function itself can remain as dead code.
**How to avoid:** Remove the entire `assemble` function and all its associated imports from `pseudo_label_cli.py`.

### Pitfall 5: Train Command Naming -- `yolo-obb` vs `obb`
**What goes wrong:** The CONTEXT says `train` keeps `obb, seg, pose, compare` but the current command is named `yolo-obb`. This rename could break user scripts.
**Why it happens:** The original name reflected the implementation (YOLO OBB) but the CONTEXT uses the shorter workflow name.
**How to avoid:** Rename `yolo-obb` to `obb` in the Click command decorator. This is a clean break (user-requested).

### Pitfall 6: `smooth-z` Currently Takes `--input` Not Run Shorthand
**What goes wrong:** `smooth-z` currently takes `--input path/to/midlines.h5` but the new design wants it to accept a RUN positional argument.
**Why it happens:** Different design era -- it was built before run resolution existed.
**How to avoid:** Change to accept RUN argument, resolve to `{run_dir}/midlines.h5` automatically.

## Code Examples

### Current vs New Command Signatures

#### Before (current):
```bash
aquapose run --config ~/aquapose/projects/YH/config.yaml
aquapose eval ~/aquapose/projects/YH/runs/run_20260304_120854
aquapose tune --stage association --config ~/aquapose/projects/YH/runs/run_20260304_120854/config_exhaustive.yaml
aquapose data import --config ~/aquapose/projects/YH/config.yaml --store pose --source pseudo --input-dir ...
aquapose train yolo-obb --config ~/aquapose/projects/YH/config.yaml --data-dir ...
```

#### After (new):
```bash
aquapose --project YH run
aquapose --project YH eval latest
aquapose --project YH tune --stage association latest
aquapose --project YH data import --store pose --source pseudo --input-dir ...
aquapose --project YH train obb  # defaults to training_data/obb/
```

### Project Resolution Integration Pattern
```python
# In any command that needs project_dir:
@data_group.command("status")
@click.pass_context
def status_cmd(ctx: click.Context) -> None:
    """Show cross-store summary."""
    project_dir = get_project_dir(ctx)
    # ... use project_dir instead of parsing --config
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `--config` on every command | `--project NAME` on top-level group | Phase 69 | Eliminates ~15 redundant `--config` options |
| `init-config` command name | `init` command name | Phase 69 | Cleaner CLI |
| `pseudo-label assemble` + `data assemble` | `data assemble` only | Phase 69 | Single assembly path |
| `train augment-elastic` CLI | `data import --augment` | Phase 68/69 | Store-based augmentation |
| Explicit run dir paths | Run shorthand (latest, -1, timestamp) | Phase 69 | Faster workflow |

## Open Questions

1. **Negative index argument parsing**
   - What we know: Click treats `-2` as an option flag, not an argument value.
   - What's unclear: Whether `ignore_unknown_options` is sufficient or introduces other parsing issues.
   - Recommendation: Use `ignore_unknown_options=True` on commands with RUN argument. If that causes issues, fall back to `--` separator convention. Test both approaches during implementation.

2. **`--config` backward compatibility**
   - What we know: CONTEXT says replace `--config` with project resolution.
   - What's unclear: Whether any external scripts depend on `--config`.
   - Recommendation: Clean break -- remove `--config` entirely (user has decided). No deprecation warning period.

3. **`tune` command config resolution**
   - What we know: `tune` currently takes `--config` pointing to the run's `config_exhaustive.yaml`. In the new design, it takes a RUN shorthand and finds the config inside the run dir.
   - What's unclear: Whether the config file name is always `config_exhaustive.yaml`.
   - Recommendation: Resolve `{run_dir}/config_exhaustive.yaml` automatically after run resolution.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection of all CLI files (cli.py, training/cli.py, data_cli.py, pseudo_label_cli.py, prep.py, run_manager.py, dataset_assembly.py)
- Click 8.3.1 installed in project (verified via `hatch run python -c "import click; print(click.__version__)"`)
- CONTEXT.md decisions from user discussion

### Secondary (MEDIUM confidence)
- Click documentation for `ignore_unknown_options` behavior with negative numbers (known Click behavior from training data, consistent with Click 8.x)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, pure refactoring of existing Click code
- Architecture: HIGH - patterns directly derived from existing codebase structure and user decisions
- Pitfalls: HIGH - identified from direct code inspection of current implementations and test files

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable domain, no external dependencies changing)
