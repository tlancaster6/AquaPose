"""Unit tests for build_stages mode-aware stage factory."""

from __future__ import annotations

from unittest.mock import patch

from aquapose.engine.config import PipelineConfig, SyntheticConfig
from aquapose.engine.pipeline import build_stages


class TestBuildStagesProductionMode:
    """Tests for build_stages with production mode."""

    def test_production_mode_returns_five_stages(self) -> None:
        config = PipelineConfig(
            mode="production",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch("aquapose.core.DetectionStage") as mock_det,
            patch("aquapose.core.MidlineStage") as mock_mid,
            patch("aquapose.core.AssociationStage") as mock_assoc,
            patch("aquapose.core.TrackingStage") as mock_trk,
            patch("aquapose.core.ReconstructionStage") as mock_rec,
            patch("aquapose.core.SyntheticDataStage"),
        ):
            stages = build_stages(config)
            assert len(stages) == 5
            mock_det.assert_called_once()
            mock_mid.assert_called_once()
            mock_assoc.assert_called_once()
            mock_trk.assert_called_once()
            mock_rec.assert_called_once()

    def test_production_mode_first_stage_is_detection(self) -> None:
        config = PipelineConfig(
            mode="production",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch("aquapose.core.DetectionStage") as mock_det,
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.AssociationStage"),
            patch("aquapose.core.TrackingStage"),
            patch("aquapose.core.ReconstructionStage"),
            patch("aquapose.core.SyntheticDataStage"),
        ):
            stages = build_stages(config)
            assert stages[0] is mock_det.return_value


class TestBuildStagesSyntheticMode:
    """Tests for build_stages with synthetic mode."""

    def test_synthetic_mode_returns_four_stages(self) -> None:
        config = PipelineConfig(
            mode="synthetic",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
            synthetic=SyntheticConfig(fish_count=2, frame_count=10),
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.SyntheticDataStage") as mock_syn,
            patch("aquapose.core.AssociationStage") as mock_assoc,
            patch("aquapose.core.TrackingStage") as mock_trk,
            patch("aquapose.core.ReconstructionStage") as mock_rec,
        ):
            stages = build_stages(config)
            assert len(stages) == 4
            mock_syn.assert_called_once()
            mock_assoc.assert_called_once()
            mock_trk.assert_called_once()
            mock_rec.assert_called_once()

    def test_synthetic_mode_first_stage_is_synthetic(self) -> None:
        config = PipelineConfig(
            mode="synthetic",
            calibration_path="/fake/cal.json",
            synthetic=SyntheticConfig(),
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.SyntheticDataStage") as mock_syn,
            patch("aquapose.core.AssociationStage"),
            patch("aquapose.core.TrackingStage"),
            patch("aquapose.core.ReconstructionStage"),
        ):
            stages = build_stages(config)
            assert stages[0] is mock_syn.return_value

    def test_synthetic_mode_preserves_downstream_stages(self) -> None:
        config = PipelineConfig(
            mode="synthetic",
            calibration_path="/fake/cal.json",
            synthetic=SyntheticConfig(),
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.SyntheticDataStage"),
            patch("aquapose.core.AssociationStage") as mock_assoc,
            patch("aquapose.core.TrackingStage") as mock_trk,
            patch("aquapose.core.ReconstructionStage") as mock_rec,
        ):
            stages = build_stages(config)
            assert stages[1] is mock_assoc.return_value
            assert stages[2] is mock_trk.return_value
            assert stages[3] is mock_rec.return_value
