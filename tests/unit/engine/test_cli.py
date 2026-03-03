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
def mock_pipeline(tmp_path: Path):
    """Mock load_config, build_stages, and PosePipeline for CLI tests.

    Yields a dict with the mocks for introspection.
    """
    mock_config = MagicMock()
    mock_config.output_dir = str(tmp_path / "output")
    mock_config.video_dir = str(tmp_path / "videos")
    mock_config.calibration_path = str(tmp_path / "cal.json")
    mock_config.mode = "production"

    def _load_config_side_effect(**kwargs: object) -> MagicMock:
        cli_overrides = kwargs.get("cli_overrides", {})
        if isinstance(cli_overrides, dict) and "mode" in cli_overrides:
            mock_config.mode = cli_overrides["mode"]
        return mock_config

    mock_stages = [MagicMock() for _ in range(5)]
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.return_value = MagicMock()

    with (
        patch(
            "aquapose.cli.load_config", side_effect=_load_config_side_effect
        ) as mock_lc,
        patch("aquapose.cli.build_stages", return_value=mock_stages) as mock_bs,
        patch(
            "aquapose.cli.PosePipeline", return_value=mock_pipeline_instance
        ) as mock_pp,
        patch(
            "aquapose.cli.VideoFrameSource",
        ) as mock_vfs,
    ):
        yield {
            "load_config": mock_lc,
            "build_stages": mock_bs,
            "PosePipeline": mock_pp,
            "pipeline_instance": mock_pipeline_instance,
            "config": mock_config,
            "stages": mock_stages,
            "VideoFrameSource": mock_vfs,
        }


class TestCLIHelp:
    """Tests for CLI help and argument discovery."""

    def test_run_help_exits_zero(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0

    def test_run_help_shows_options(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert "--config" in result.output
        assert "--mode" in result.output
        assert "--set" in result.output
        assert "--add-observer" in result.output
        assert "--verbose" in result.output

    def test_run_without_config_fails(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0


class TestCLIExecution:
    """Tests for CLI run command execution."""

    def test_run_success_exit_zero(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("mode: production\n")
        result = runner.invoke(cli, ["run", "--config", str(config_file)])
        assert result.exit_code == 0

    def test_run_failure_exit_one(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        mock_pipeline["pipeline_instance"].run.side_effect = RuntimeError("boom")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("mode: production\n")
        result = runner.invoke(cli, ["run", "--config", str(config_file)])
        assert result.exit_code == 1

    def test_mode_defaults_to_production(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(cli, ["run", "--config", str(config_file)])
        call_kwargs = mock_pipeline["load_config"].call_args
        cli_overrides = call_kwargs[1].get(
            "cli_overrides", call_kwargs[0][0] if call_kwargs[0] else {}
        )
        if isinstance(cli_overrides, dict):
            pass  # cli_overrides checked via load_config kwargs
        # When --mode is not passed, mode should NOT be injected into overrides
        # (defers to YAML config value or PipelineConfig default of "production")
        lc_kwargs = mock_pipeline["load_config"].call_args
        overrides = lc_kwargs.kwargs.get("cli_overrides", {})
        assert "mode" not in overrides

    def test_mode_diagnostic_passed_to_config(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(
            cli, ["run", "--config", str(config_file), "--mode", "diagnostic"]
        )
        overrides = mock_pipeline["load_config"].call_args.kwargs.get(
            "cli_overrides", {}
        )
        assert overrides.get("mode") == "diagnostic"

    def test_set_overrides_parsed(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(
            cli,
            [
                "run",
                "--config",
                str(config_file),
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

    def test_pipeline_constructed_with_stages_and_observers(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(cli, ["run", "--config", str(config_file)])
        pp_call = mock_pipeline["PosePipeline"].call_args
        assert pp_call.kwargs["stages"] == mock_pipeline["stages"]
        assert len(pp_call.kwargs["observers"]) >= 1  # At least ConsoleObserver

    def test_production_mode_has_timing_and_hdf5(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        from aquapose.engine import ConsoleObserver, HDF5ExportObserver, TimingObserver

        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(cli, ["run", "--config", str(config_file)])
        pp_call = mock_pipeline["PosePipeline"].call_args
        observers = pp_call.kwargs["observers"]
        types = [type(o) for o in observers]
        assert ConsoleObserver in types
        assert TimingObserver in types
        assert HDF5ExportObserver in types

    def test_add_observer_augments_list(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        from aquapose.engine import TimingObserver

        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(
            cli,
            ["run", "--config", str(config_file), "--add-observer", "timing"],
        )
        pp_call = mock_pipeline["PosePipeline"].call_args
        observers = pp_call.kwargs["observers"]
        timing_count = sum(1 for o in observers if isinstance(o, TimingObserver))
        # Production already has TimingObserver, plus --add-observer adds another
        assert timing_count >= 2

    def test_verbose_flag_passed_to_console_observer(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        from aquapose.engine import ConsoleObserver

        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(cli, ["run", "--config", str(config_file), "--verbose"])
        pp_call = mock_pipeline["PosePipeline"].call_args
        observers = pp_call.kwargs["observers"]
        console_obs = [o for o in observers if isinstance(o, ConsoleObserver)]
        assert len(console_obs) >= 1
        assert console_obs[0]._verbose is True


class TestDiagnosticMode:
    """Tests for --mode diagnostic observer assembly."""

    def test_diagnostic_mode_assembles_all_observers(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        from aquapose.engine import (
            Animation3DObserver,
            ConsoleObserver,
            DiagnosticObserver,
            HDF5ExportObserver,
            Overlay2DObserver,
            TimingObserver,
        )

        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(
            cli, ["run", "--config", str(config_file), "--mode", "diagnostic"]
        )
        pp_call = mock_pipeline["PosePipeline"].call_args
        observers = pp_call.kwargs["observers"]
        types = [type(o) for o in observers]
        assert ConsoleObserver in types
        assert TimingObserver in types
        assert HDF5ExportObserver in types
        assert Overlay2DObserver in types
        assert Animation3DObserver in types
        assert DiagnosticObserver in types


class TestBenchmarkMode:
    """Tests for --mode benchmark observer assembly."""

    def test_benchmark_mode_assembles_timing_only(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        from aquapose.engine import (
            Animation3DObserver,
            ConsoleObserver,
            DiagnosticObserver,
            HDF5ExportObserver,
            Overlay2DObserver,
            TimingObserver,
        )

        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(cli, ["run", "--config", str(config_file), "--mode", "benchmark"])
        pp_call = mock_pipeline["PosePipeline"].call_args
        observers = pp_call.kwargs["observers"]
        types = [type(o) for o in observers]
        assert ConsoleObserver in types
        assert TimingObserver in types
        # These should NOT be present in benchmark mode
        assert HDF5ExportObserver not in types
        assert Overlay2DObserver not in types
        assert Animation3DObserver not in types
        assert DiagnosticObserver not in types

    def test_add_observer_augments_benchmark_mode(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        from aquapose.engine import HDF5ExportObserver

        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(
            cli,
            [
                "run",
                "--config",
                str(config_file),
                "--mode",
                "benchmark",
                "--add-observer",
                "hdf5",
            ],
        )
        pp_call = mock_pipeline["PosePipeline"].call_args
        observers = pp_call.kwargs["observers"]
        types = [type(o) for o in observers]
        assert HDF5ExportObserver in types


class TestInitConfig:
    """Tests for the init-config subcommand."""

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
        """init-config creates project dir with config.yaml and 4 subdirectories."""
        home_str = str(tmp_path)
        monkeypatch.setenv("HOME", home_str)
        monkeypatch.setenv("USERPROFILE", home_str)
        result = runner.invoke(cli, ["init-config", "test1"])
        assert result.exit_code == 0, result.output
        project_dir = tmp_path / "aquapose" / "projects" / "test1"
        assert project_dir.exists()
        assert (project_dir / "config.yaml").exists()
        assert (project_dir / "runs").is_dir()
        assert (project_dir / "models").is_dir()
        assert (project_dir / "geometry").is_dir()
        assert (project_dir / "videos").is_dir()

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
        runner.invoke(cli, ["init-config", "test1"])
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
        result = runner.invoke(cli, ["init-config", "test2", "--synthetic"])
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
        """init-config fails with non-zero exit if project directory already exists."""
        home_str = str(tmp_path)
        monkeypatch.setenv("HOME", home_str)
        monkeypatch.setenv("USERPROFILE", home_str)
        target = tmp_path / "aquapose" / "projects" / "test1"
        target.mkdir(parents=True)
        result = runner.invoke(cli, ["init-config", "test1"])
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
        runner.invoke(cli, ["init-config", "test3"])
        project_dir = tmp_path / "aquapose" / "projects" / "test3"
        import yaml as _yaml

        content = (project_dir / "config.yaml").read_text()
        parsed = _yaml.safe_load(content)
        assert "synthetic" not in parsed

    def test_init_config_help(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["init-config", "--help"])
        assert result.exit_code == 0
        assert "NAME" in result.output


class TestSyntheticMode:
    """Tests for --mode synthetic CLI behavior."""

    def test_synthetic_mode_calls_build_stages(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(cli, ["run", "--config", str(config_file), "--mode", "synthetic"])
        mock_pipeline["build_stages"].assert_called_once()

    def test_synthetic_mode_uses_production_observers(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        from aquapose.engine import ConsoleObserver, HDF5ExportObserver, TimingObserver

        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(cli, ["run", "--config", str(config_file), "--mode", "synthetic"])
        pp_call = mock_pipeline["PosePipeline"].call_args
        observers = pp_call.kwargs["observers"]
        types = [type(o) for o in observers]
        assert ConsoleObserver in types
        assert TimingObserver in types
        assert HDF5ExportObserver in types

    def test_synthetic_mode_passes_mode_in_config(
        self,
        runner: click.testing.CliRunner,
        tmp_path: Path,
        mock_pipeline: dict,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_dir: /tmp/test\n")
        runner.invoke(cli, ["run", "--config", str(config_file), "--mode", "synthetic"])
        overrides = mock_pipeline["load_config"].call_args.kwargs.get(
            "cli_overrides", {}
        )
        assert overrides.get("mode") == "synthetic"
