"""Unit tests for AquaPose CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click.testing
import pytest

from aquapose.cli import cli


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a Click CliRunner for isolated CLI testing."""
    return click.testing.CliRunner()


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a minimal project directory with config.yaml."""
    proj = tmp_path / "test_project"
    proj.mkdir()
    (proj / "config.yaml").write_text("mode: production\n")
    (proj / "runs").mkdir()
    return proj


@pytest.fixture
def monkeypatch_project(monkeypatch: pytest.MonkeyPatch, project_dir: Path) -> Path:
    """Patch resolve_project so --project test resolves to project_dir."""
    monkeypatch.setattr(
        "aquapose.cli_utils.resolve_project",
        lambda name: project_dir,
    )
    return project_dir


@pytest.fixture
def mock_pipeline(tmp_path: Path, monkeypatch_project: Path):
    """Mock load_config and ChunkOrchestrator for CLI tests.

    Yields a dict with the mocks for introspection.
    """
    mock_config = MagicMock()
    mock_config.output_dir = str(tmp_path / "output")
    mock_config.video_dir = str(tmp_path / "videos")
    mock_config.calibration_path = str(tmp_path / "cal.json")
    mock_config.mode = "production"
    mock_config.chunk_size = None  # degenerate single-chunk

    def _load_config_side_effect(**kwargs: object) -> MagicMock:
        cli_overrides = kwargs.get("cli_overrides", {})
        if isinstance(cli_overrides, dict) and "mode" in cli_overrides:
            mock_config.mode = cli_overrides["mode"]
        return mock_config

    mock_orchestrator_instance = MagicMock()

    with (
        patch(
            "aquapose.cli.load_config", side_effect=_load_config_side_effect
        ) as mock_lc,
        patch(
            "aquapose.cli.ChunkOrchestrator", return_value=mock_orchestrator_instance
        ) as mock_co,
    ):
        yield {
            "load_config": mock_lc,
            "ChunkOrchestrator": mock_co,
            "orchestrator_instance": mock_orchestrator_instance,
            "config": mock_config,
            "project_dir": monkeypatch_project,
        }


class TestCLIHelp:
    """Tests for CLI help and argument discovery."""

    def test_run_help_exits_zero(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0

    def test_run_help_shows_options(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert "--mode" in result.output
        assert "--set" in result.output
        assert "--add-observer" in result.output
        assert "--verbose" in result.output

    def test_run_without_project_fails(self, runner: click.testing.CliRunner) -> None:
        """run without --project (and no CWD detection) fails."""
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0


class TestCLIExecution:
    """Tests for CLI run command execution."""

    def test_run_success_exit_zero(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        result = runner.invoke(cli, ["--project", "test", "run"])
        assert result.exit_code == 0

    def test_run_failure_exit_one(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        mock_pipeline["orchestrator_instance"].run.side_effect = RuntimeError("boom")
        result = runner.invoke(cli, ["--project", "test", "run"])
        assert result.exit_code == 1

    def test_mode_defaults_to_production(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        runner.invoke(cli, ["--project", "test", "run"])
        # When --mode is not passed, mode should NOT be injected into overrides
        lc_kwargs = mock_pipeline["load_config"].call_args
        overrides = lc_kwargs.kwargs.get("cli_overrides", {})
        assert "mode" not in overrides

    def test_mode_diagnostic_passed_to_config(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        runner.invoke(cli, ["--project", "test", "run", "--mode", "diagnostic"])
        overrides = mock_pipeline["load_config"].call_args.kwargs.get(
            "cli_overrides", {}
        )
        assert overrides.get("mode") == "diagnostic"

    def test_set_overrides_parsed(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        runner.invoke(
            cli,
            [
                "--project",
                "test",
                "run",
                "--set",
                "detection.detector_kind=yolo",
                "--set",
                "tracking.max_age=10",
            ],
        )
        overrides = mock_pipeline["load_config"].call_args.kwargs.get(
            "cli_overrides", {}
        )
        assert overrides.get("detection.detector_kind") == "yolo"
        assert overrides.get("tracking.max_age") == 10

    def test_orchestrator_constructed_with_config(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        runner.invoke(cli, ["--project", "test", "run"])
        co_call = mock_pipeline["ChunkOrchestrator"].call_args
        assert co_call is not None
        assert co_call.kwargs["config"] is mock_pipeline["config"]

    def test_add_observer_passed_to_orchestrator(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        runner.invoke(
            cli,
            ["--project", "test", "run", "--add-observer", "timing"],
        )
        co_call = mock_pipeline["ChunkOrchestrator"].call_args
        extra = co_call.kwargs.get("extra_observers", ())
        assert "timing" in extra

    def test_verbose_flag_passed_to_orchestrator(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        runner.invoke(cli, ["--project", "test", "run", "--verbose"])
        co_call = mock_pipeline["ChunkOrchestrator"].call_args
        assert co_call.kwargs.get("verbose") is True

    def test_max_chunks_passed_to_orchestrator(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        """--max-chunks 2 passes max_chunks=2 to ChunkOrchestrator."""
        runner.invoke(cli, ["--project", "test", "run", "--max-chunks", "2"])
        co_call = mock_pipeline["ChunkOrchestrator"].call_args
        assert co_call.kwargs.get("max_chunks") == 2


class TestInitConfig:
    """Tests for the init subcommand."""

    @pytest.fixture(autouse=False)
    def patch_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect ~ to tmp_path by patching HOME/USERPROFILE env vars."""
        home_str = str(tmp_path)
        monkeypatch.setenv("HOME", home_str)
        monkeypatch.setenv("USERPROFILE", home_str)
        return tmp_path

    def test_init_config_creates_project_directory(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """init creates project dir with config.yaml and 4 subdirectories."""
        home_str = str(tmp_path)
        monkeypatch.setenv("HOME", home_str)
        monkeypatch.setenv("USERPROFILE", home_str)
        result = runner.invoke(cli, ["init", "test1"])
        assert result.exit_code == 0, result.output
        project_dir = tmp_path / "aquapose" / "projects" / "test1"
        assert project_dir.exists()
        assert (project_dir / "config.yaml").exists()
        assert (project_dir / "runs").is_dir()
        assert (project_dir / "models").is_dir()
        assert (project_dir / "geometry").is_dir()
        assert (project_dir / "videos").is_dir()
        assert (project_dir / "training_data" / "obb").is_dir()
        assert (project_dir / "training_data" / "pose").is_dir()

    def test_init_config_yaml_field_order(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Generated config.yaml has user-relevant field ordering."""
        home_str = str(tmp_path)
        monkeypatch.setenv("HOME", home_str)
        monkeypatch.setenv("USERPROFILE", home_str)
        runner.invoke(cli, ["init", "test1"])
        project_dir = tmp_path / "aquapose" / "projects" / "test1"
        import yaml as _yaml

        content = (project_dir / "config.yaml").read_text()
        parsed = _yaml.safe_load(content)
        keys = list(parsed.keys())
        assert keys.index("project_dir") < keys.index("video_dir")
        assert keys.index("video_dir") < keys.index("n_animals")
        assert keys.index("n_animals") < keys.index("detection")

    def test_init_config_with_synthetic_flag(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--synthetic flag adds synthetic section without fish_count."""
        home_str = str(tmp_path)
        monkeypatch.setenv("HOME", home_str)
        monkeypatch.setenv("USERPROFILE", home_str)
        result = runner.invoke(cli, ["init", "test2", "--synthetic"])
        assert result.exit_code == 0, result.output
        project_dir = tmp_path / "aquapose" / "projects" / "test2"
        import yaml as _yaml

        content = (project_dir / "config.yaml").read_text()
        parsed = _yaml.safe_load(content)
        assert "synthetic" in parsed
        assert "fish_count" not in parsed["synthetic"]

    def test_init_config_refuses_existing_directory(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """init fails with non-zero exit if project directory already exists."""
        home_str = str(tmp_path)
        monkeypatch.setenv("HOME", home_str)
        monkeypatch.setenv("USERPROFILE", home_str)
        target = tmp_path / "aquapose" / "projects" / "test1"
        target.mkdir(parents=True)
        result = runner.invoke(cli, ["init", "test1"])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_init_config_no_synthetic_by_default(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """By default (no --synthetic), synthetic key is absent from config.yaml."""
        home_str = str(tmp_path)
        monkeypatch.setenv("HOME", home_str)
        monkeypatch.setenv("USERPROFILE", home_str)
        runner.invoke(cli, ["init", "test3"])
        project_dir = tmp_path / "aquapose" / "projects" / "test3"
        import yaml as _yaml

        content = (project_dir / "config.yaml").read_text()
        parsed = _yaml.safe_load(content)
        assert "synthetic" not in parsed

    def test_init_config_help(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "NAME" in result.output


class TestSyntheticMode:
    """Tests for --mode synthetic CLI behavior."""

    def test_synthetic_mode_calls_orchestrator(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        runner.invoke(cli, ["--project", "test", "run", "--mode", "synthetic"])
        mock_pipeline["ChunkOrchestrator"].assert_called_once()

    def test_synthetic_mode_passes_mode_in_config(
        self,
        runner: click.testing.CliRunner,
        mock_pipeline: dict,
    ) -> None:
        runner.invoke(cli, ["--project", "test", "run", "--mode", "synthetic"])
        overrides = mock_pipeline["load_config"].call_args.kwargs.get(
            "cli_overrides", {}
        )
        assert overrides.get("mode") == "synthetic"
