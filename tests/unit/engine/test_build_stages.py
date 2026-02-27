"""Unit tests for build_stages mode-aware stage factory (v2.1).

v2.1: TrackingStage and AssociationStage stubs are added in Phase 22-02.
Until then, build_stages returns 3 stages (production) or 2 stages (synthetic).
These tests verify the current stage list structure.
"""

from __future__ import annotations

from unittest.mock import patch

from aquapose.engine.config import PipelineConfig, SyntheticConfig


class TestBuildStagesProductionMode:
    """Tests for build_stages with production mode."""

    def test_production_mode_returns_stages(self) -> None:
        config = PipelineConfig(
            mode="production",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch("aquapose.core.DetectionStage") as mock_det,
            patch("aquapose.core.MidlineStage") as mock_mid,
            patch("aquapose.core.ReconstructionStage") as mock_rec,
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert len(stages) >= 3
            mock_det.assert_called_once()
            mock_mid.assert_called_once()
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
            patch("aquapose.core.ReconstructionStage"),
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert stages[0] is mock_det.return_value


class TestBuildStagesSyntheticMode:
    """Tests for build_stages with synthetic mode."""

    def test_synthetic_mode_returns_stages(self) -> None:
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
            patch("aquapose.core.ReconstructionStage") as mock_rec,
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert len(stages) >= 2
            mock_syn.assert_called_once()
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
            patch("aquapose.core.ReconstructionStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert stages[0] is mock_syn.return_value

    def test_synthetic_mode_last_stage_is_reconstruction(self) -> None:
        config = PipelineConfig(
            mode="synthetic",
            calibration_path="/fake/cal.json",
            synthetic=SyntheticConfig(),
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.SyntheticDataStage"),
            patch("aquapose.core.ReconstructionStage") as mock_rec,
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert stages[-1] is mock_rec.return_value
