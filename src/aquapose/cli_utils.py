"""Project and run resolution utilities for the AquaPose CLI."""

from __future__ import annotations

import re
from pathlib import Path

import click

AQUAPOSE_HOME: Path = Path("~/aquapose/projects").expanduser()
"""Default root directory for AquaPose projects."""


def resolve_project(name: str | None) -> Path:
    """Resolve a project directory by name or by walking CWD upward.

    Args:
        name: Project name to look up under AQUAPOSE_HOME, or None to
            auto-detect from the current working directory.

    Returns:
        Absolute path to the project directory.

    Raises:
        click.ClickException: If the project is not found.
    """
    if name is not None:
        project_dir = AQUAPOSE_HOME / name
        if not project_dir.is_dir():
            raise click.ClickException(f"Project '{name}' not found at {project_dir}")
        return project_dir

    # Walk CWD upward looking for config.yaml
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    current = cwd
    while True:
        if (current / "config.yaml").exists():
            return current
        if current == home or current == current.parent:
            break
        current = current.parent

    raise click.ClickException(
        f"No config.yaml found walking upward from {cwd}. "
        "Use --project NAME or cd into a project directory."
    )


def _sorted_runs(runs_dir: Path) -> list[Path]:
    """Return run_* directories sorted by name (ascending).

    Args:
        runs_dir: Directory containing run subdirectories.

    Returns:
        List of run directories sorted by name.
    """
    if not runs_dir.is_dir():
        return []
    return sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda p: p.name,
    )


def _latest_run(runs_dir: Path) -> Path:
    """Return the most recent run_* directory.

    Args:
        runs_dir: Directory containing run subdirectories.

    Returns:
        Path to the most recent run directory.

    Raises:
        click.ClickException: If no runs exist.
    """
    runs = _sorted_runs(runs_dir)
    if not runs:
        raise click.ClickException(f"No runs found in {runs_dir}")
    return runs[-1]


def resolve_run(run_ref: str | None, project_dir: Path, subdir: str = "runs") -> Path:
    """Resolve a run directory reference within a project.

    Args:
        run_ref: Run reference -- None or ``"latest"`` for most recent,
            a timestamp prefix (e.g. ``"20260306"`` or ``"20260306_075925"``),
            or a full/relative path.
        project_dir: Path to the project directory.
        subdir: Subdirectory under project_dir containing runs.

    Returns:
        Absolute path to the resolved run directory.

    Raises:
        click.ClickException: If the run cannot be resolved.
    """
    runs_dir = project_dir / subdir

    # None or "latest" -> most recent
    if run_ref is None or run_ref == "latest":
        return _latest_run(runs_dir)

    # Timestamp pattern: starts with 8+ digits
    if re.match(r"^\d{8}", run_ref):
        runs = _sorted_runs(runs_dir)
        # Match against the timestamp portion of run_YYYYMMDD_HHMMSS
        matches = [r for r in runs if r.name.removeprefix("run_").startswith(run_ref)]
        if not matches:
            raise click.ClickException(
                f"No run matching timestamp '{run_ref}' in {runs_dir}"
            )
        # Return the latest match (in case of multiple partial matches)
        return matches[-1]

    # Treat as path
    run_path = Path(run_ref)
    if not run_path.is_absolute():
        run_path = runs_dir / run_ref
    if not run_path.is_dir():
        raise click.ClickException(f"Run path does not exist: {run_path}")
    return run_path


def get_project_dir(ctx: click.Context) -> Path:
    """Lazily resolve and cache the project directory in ctx.obj.

    Reads ``ctx.obj["project_name"]`` and calls :func:`resolve_project`.
    The result is cached in ``ctx.obj["_project_dir"]`` for subsequent calls.

    Args:
        ctx: Click context with ``project_name`` in obj dict.

    Returns:
        Resolved project directory path.
    """
    ctx.ensure_object(dict)
    cached = ctx.obj.get("_project_dir")
    if cached is not None:
        return cached
    project_dir = resolve_project(ctx.obj.get("project_name"))
    ctx.obj["_project_dir"] = project_dir
    return project_dir


def get_config_path(ctx: click.Context) -> Path:
    """Return the project config.yaml path.

    Args:
        ctx: Click context (delegates to :func:`get_project_dir`).

    Returns:
        Path to the project's config.yaml file.
    """
    return get_project_dir(ctx) / "config.yaml"


__all__ = [
    "AQUAPOSE_HOME",
    "get_config_path",
    "get_project_dir",
    "resolve_project",
    "resolve_run",
]
