"""Unit tests for CLI utility functions (project and run resolution)."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from aquapose.cli_utils import (
    get_config_path,
    get_project_dir,
    resolve_project,
    resolve_run,
)


class TestResolveProject:
    """Tests for resolve_project."""

    def test_named_project_returns_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_project('YH') returns correct path when dir exists."""
        home = tmp_path / "aquapose" / "projects"
        home.mkdir(parents=True)
        (home / "YH").mkdir()
        monkeypatch.setattr("aquapose.cli_utils.AQUAPOSE_HOME", home)

        result = resolve_project("YH")
        assert result == home / "YH"

    def test_nonexistent_project_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_project('nonexistent') raises ClickException."""
        home = tmp_path / "aquapose" / "projects"
        home.mkdir(parents=True)
        monkeypatch.setattr("aquapose.cli_utils.AQUAPOSE_HOME", home)

        with pytest.raises(click.ClickException, match="not found"):
            resolve_project("nonexistent")

    def test_none_from_inside_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_project(None) from CWD inside project returns that project dir."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / "config.yaml").write_text("n_animals: 3\n")
        sub = project_dir / "runs" / "run_001"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)

        result = resolve_project(None)
        assert result == project_dir

    def test_none_from_outside_project_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_project(None) from CWD outside any project raises ClickException."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(click.ClickException, match=r"config\.yaml"):
            resolve_project(None)


class TestResolveRun:
    """Tests for resolve_run."""

    @pytest.fixture
    def runs_dir(self, tmp_path: Path) -> Path:
        """Create a mock runs/ directory with several run dirs."""
        rd = tmp_path / "runs"
        rd.mkdir()
        (rd / "run_20260304_112501").mkdir()
        (rd / "run_20260305_080000").mkdir()
        (rd / "run_20260306_075925").mkdir()
        return rd

    def test_none_returns_latest(self, runs_dir: Path, tmp_path: Path) -> None:
        """resolve_run(None, project_dir) returns most recent run."""
        result = resolve_run(None, tmp_path)
        assert result == runs_dir / "run_20260306_075925"

    def test_latest_returns_latest(self, runs_dir: Path, tmp_path: Path) -> None:
        """resolve_run('latest', project_dir) returns most recent run."""
        result = resolve_run("latest", tmp_path)
        assert result == runs_dir / "run_20260306_075925"

    def test_full_timestamp_match(self, runs_dir: Path, tmp_path: Path) -> None:
        """resolve_run with full timestamp returns matching run dir."""
        result = resolve_run("20260306_075925", tmp_path)
        assert result == runs_dir / "run_20260306_075925"

    def test_partial_timestamp_match(self, runs_dir: Path, tmp_path: Path) -> None:
        """resolve_run with partial timestamp prefix returns matching run dir."""
        result = resolve_run("20260306", tmp_path)
        assert result == runs_dir / "run_20260306_075925"

    def test_nonexistent_timestamp_raises(self, runs_dir: Path, tmp_path: Path) -> None:
        """resolve_run with non-matching timestamp raises ClickException."""
        with pytest.raises(click.ClickException, match="No run matching"):
            resolve_run("20261231", tmp_path)

    def test_absolute_path_exists(self, runs_dir: Path, tmp_path: Path) -> None:
        """resolve_run with existing absolute path returns that path."""
        target = runs_dir / "run_20260304_112501"
        result = resolve_run(str(target), tmp_path)
        assert result == target

    def test_absolute_path_nonexistent_raises(self, tmp_path: Path) -> None:
        """resolve_run with non-existent absolute path raises ClickException."""
        with pytest.raises(click.ClickException, match="does not exist"):
            resolve_run("/nonexistent/path/to/run", tmp_path)

    def test_no_runs_raises(self, tmp_path: Path) -> None:
        """resolve_run(None) raises ClickException when no runs exist."""
        (tmp_path / "runs").mkdir()
        with pytest.raises(click.ClickException, match="No runs found"):
            resolve_run(None, tmp_path)


class TestGetProjectDir:
    """Tests for get_project_dir and get_config_path."""

    def test_get_project_dir_caches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_project_dir lazily resolves and caches in ctx.obj."""
        home = tmp_path / "aquapose" / "projects"
        home.mkdir(parents=True)
        (home / "TestProj").mkdir()
        monkeypatch.setattr("aquapose.cli_utils.AQUAPOSE_HOME", home)

        ctx = click.Context(click.Command("test"))
        ctx.ensure_object(dict)
        ctx.obj["project_name"] = "TestProj"

        result1 = get_project_dir(ctx)
        result2 = get_project_dir(ctx)
        assert result1 == home / "TestProj"
        assert result1 is result2  # same cached object

    def test_get_config_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_config_path returns project_dir / 'config.yaml'."""
        home = tmp_path / "aquapose" / "projects"
        home.mkdir(parents=True)
        (home / "TestProj").mkdir()
        monkeypatch.setattr("aquapose.cli_utils.AQUAPOSE_HOME", home)

        ctx = click.Context(click.Command("test"))
        ctx.ensure_object(dict)
        ctx.obj["project_name"] = "TestProj"

        result = get_config_path(ctx)
        assert result == home / "TestProj" / "config.yaml"
