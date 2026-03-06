"""Tests for generate-luts CLI, AssociationStage fail-fast, and build_stages LUT check."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from aquapose.cli import cli


class TestGenerateLutsCli:
    """Tests for the generate-luts CLI command."""

    @pytest.fixture
    def project_dir(self, tmp_path: Path) -> Path:
        """Create a project dir with config.yaml containing calibration_path."""
        proj = tmp_path / "test_project"
        proj.mkdir()
        # Create calibration file so path resolution works
        cal_dir = proj / "geometry"
        cal_dir.mkdir()
        (cal_dir / "calibration.json").write_text("{}")
        config_text = f"calibration_path: {cal_dir / 'calibration.json'}\n"
        (proj / "config.yaml").write_text(config_text)
        return proj

    @pytest.fixture
    def monkeypatch_project(
        self, monkeypatch: pytest.MonkeyPatch, project_dir: Path
    ) -> Path:
        """Patch resolve_project for --project test."""
        monkeypatch.setattr(
            "aquapose.cli_utils.resolve_project",
            lambda name: project_dir,
        )
        return project_dir

    @patch("aquapose.calibration.luts.save_inverse_luts")
    @patch("aquapose.calibration.luts.save_forward_luts")
    @patch("aquapose.calibration.luts.generate_inverse_lut")
    @patch("aquapose.calibration.luts.generate_forward_luts")
    @patch("aquapose.calibration.loader.compute_undistortion_maps")
    @patch("aquapose.calibration.loader.load_calibration_data")
    @patch("aquapose.calibration.luts.load_inverse_luts", return_value=None)
    @patch("aquapose.calibration.luts.load_forward_luts", return_value=None)
    def test_generate_luts_calls_generation(
        self,
        mock_load_fwd: MagicMock,
        mock_load_inv: MagicMock,
        mock_load_cal: MagicMock,
        mock_compute_undist: MagicMock,
        mock_gen_fwd: MagicMock,
        mock_gen_inv: MagicMock,
        mock_save_fwd: MagicMock,
        mock_save_inv: MagicMock,
        monkeypatch_project: Path,
    ) -> None:
        """generate-luts loads config and calls LUT generation functions."""
        mock_cal = MagicMock()
        mock_cal.ring_cameras = ["cam1", "cam2"]
        mock_cal.cameras = {
            "cam1": MagicMock(),
            "cam2": MagicMock(),
        }
        mock_load_cal.return_value = mock_cal
        mock_compute_undist.return_value = MagicMock()

        mock_gen_fwd.return_value = {"cam1": MagicMock(), "cam2": MagicMock()}
        mock_gen_inv.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--project", "test", "prep", "generate-luts"],
        )
        assert result.exit_code == 0, result.output
        mock_gen_fwd.assert_called_once()
        mock_gen_inv.assert_called_once()
        mock_save_fwd.assert_called_once()
        mock_save_inv.assert_called_once()

    @patch("aquapose.calibration.luts.load_inverse_luts")
    @patch("aquapose.calibration.luts.load_forward_luts")
    def test_generate_luts_skips_when_exist(
        self,
        mock_load_fwd: MagicMock,
        mock_load_inv: MagicMock,
        monkeypatch_project: Path,
    ) -> None:
        """generate-luts skips when LUTs already exist on disk."""
        # LUTs already exist
        mock_load_fwd.return_value = {"cam1": MagicMock()}
        mock_load_inv.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--project", "test", "prep", "generate-luts"],
        )
        assert result.exit_code == 0, result.output
        assert "already exist" in result.output

    @patch("aquapose.calibration.luts.save_inverse_luts")
    @patch("aquapose.calibration.luts.save_forward_luts")
    @patch("aquapose.calibration.luts.generate_inverse_lut")
    @patch("aquapose.calibration.luts.generate_forward_luts")
    @patch("aquapose.calibration.loader.compute_undistortion_maps")
    @patch("aquapose.calibration.loader.load_calibration_data")
    def test_generate_luts_force_regenerates(
        self,
        mock_load_cal: MagicMock,
        mock_compute_undist: MagicMock,
        mock_gen_fwd: MagicMock,
        mock_gen_inv: MagicMock,
        mock_save_fwd: MagicMock,
        mock_save_inv: MagicMock,
        monkeypatch_project: Path,
    ) -> None:
        """generate-luts --force regenerates even when LUTs exist."""
        mock_cal = MagicMock()
        mock_cal.ring_cameras = ["cam1"]
        mock_cal.cameras = {"cam1": MagicMock()}
        mock_load_cal.return_value = mock_cal
        mock_compute_undist.return_value = MagicMock()
        mock_gen_fwd.return_value = {"cam1": MagicMock()}
        mock_gen_inv.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--project", "test", "prep", "generate-luts", "--force"],
        )
        assert result.exit_code == 0, result.output
        mock_gen_fwd.assert_called_once()
        mock_gen_inv.assert_called_once()


class TestBuildStagesLutCheck:
    """Tests for build_stages early LUT existence check."""

    @patch("aquapose.engine.pipeline.load_inverse_luts", return_value=None)
    @patch("aquapose.engine.pipeline.load_forward_luts", return_value=None)
    @patch("aquapose.core.ReconstructionStage")
    @patch("aquapose.core.SyntheticDataStage")
    def test_build_stages_raises_when_luts_missing(
        self,
        mock_syn: MagicMock,
        mock_rec: MagicMock,
        mock_load_fwd: MagicMock,
        mock_load_inv: MagicMock,
    ) -> None:
        """build_stages raises FileNotFoundError when association runs and LUTs missing."""
        from aquapose.engine.config import PipelineConfig, SyntheticConfig

        config = PipelineConfig(
            mode="synthetic",
            calibration_path="/fake/cal.json",
            synthetic=SyntheticConfig(),
        )

        from aquapose.engine.pipeline import build_stages

        with pytest.raises(FileNotFoundError, match="generate-luts"):
            build_stages(config)

    @patch("aquapose.engine.pipeline.load_inverse_luts", return_value=None)
    @patch("aquapose.engine.pipeline.load_forward_luts", return_value=None)
    @patch("aquapose.core.ReconstructionStage")
    @patch("aquapose.core.SyntheticDataStage")
    def test_build_stages_skips_lut_check_before_association(
        self,
        mock_syn: MagicMock,
        mock_rec: MagicMock,
        mock_load_fwd: MagicMock,
        mock_load_inv: MagicMock,
    ) -> None:
        """build_stages does NOT check LUTs when stop_after is before association."""
        from aquapose.engine.config import PipelineConfig, SyntheticConfig

        config = PipelineConfig(
            mode="synthetic",
            calibration_path="/fake/cal.json",
            stop_after="tracking",
            synthetic=SyntheticConfig(),
        )

        from aquapose.engine.pipeline import build_stages

        # Should not raise even though LUTs are missing
        stages = build_stages(config)
        assert len(stages) > 0
        # load_forward_luts should not have been called
        mock_load_fwd.assert_not_called()
