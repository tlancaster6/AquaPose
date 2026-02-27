"""Unit tests for build_stages mode-aware stage factory (v2.1).

v2.1: build_stages returns 5 stages in production mode and 4 in synthetic mode.
Stage order: Detection -> TrackingStage -> AssociationStubStage -> Midline -> Reconstruction
Synthetic order: SyntheticDataStage -> TrackingStage -> AssociationStubStage -> Reconstruction

TrackingStage is imported from aquapose.core.tracking.
AssociationStubStage is inline in engine/pipeline.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aquapose.engine.config import PipelineConfig, SyntheticConfig
from aquapose.engine.pipeline import AssociationStubStage


class TestBuildStagesProductionMode:
    """Tests for build_stages with production mode."""

    def test_production_mode_returns_5_stages(self) -> None:
        """build_stages returns exactly 5 stages in production mode."""
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
            assert len(stages) == 5, f"Expected 5 stages, got {len(stages)}"
            mock_det.assert_called_once()
            mock_mid.assert_called_once()
            mock_rec.assert_called_once()

    def test_production_mode_stage_order(self) -> None:
        """Production mode: Detection -> TrackingStage -> AssociationStub -> Midline -> Reconstruction."""
        from aquapose.core.tracking import TrackingStage

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
            assert stages[0] is mock_det.return_value, "Stage 0 must be DetectionStage"
            assert isinstance(stages[1], TrackingStage), "Stage 1 must be TrackingStage"
            assert isinstance(stages[2], AssociationStubStage), (
                "Stage 2 must be AssociationStubStage"
            )
            assert stages[3] is mock_mid.return_value, "Stage 3 must be MidlineStage"
            assert stages[4] is mock_rec.return_value, (
                "Stage 4 must be ReconstructionStage"
            )

    def test_production_mode_first_stage_is_detection(self) -> None:
        """First stage in production mode is DetectionStage."""
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

    def test_production_mode_last_stage_is_reconstruction(self) -> None:
        """Last stage in production mode is ReconstructionStage."""
        config = PipelineConfig(
            mode="production",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.ReconstructionStage") as mock_rec,
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert stages[-1] is mock_rec.return_value


class TestBuildStagesSyntheticMode:
    """Tests for build_stages with synthetic mode."""

    def test_synthetic_mode_returns_4_stages(self) -> None:
        """build_stages returns exactly 4 stages in synthetic mode."""
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
            assert len(stages) == 4, f"Expected 4 stages, got {len(stages)}"
            mock_syn.assert_called_once()
            mock_rec.assert_called_once()

    def test_synthetic_mode_stage_order(self) -> None:
        """Synthetic mode: SyntheticData -> TrackingStage -> AssociationStub -> Reconstruction."""
        from aquapose.core.tracking import TrackingStage

        config = PipelineConfig(
            mode="synthetic",
            calibration_path="/fake/cal.json",
            synthetic=SyntheticConfig(),
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.SyntheticDataStage") as mock_syn,
            patch("aquapose.core.ReconstructionStage") as mock_rec,
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert stages[0] is mock_syn.return_value, (
                "Stage 0 must be SyntheticDataStage"
            )
            assert isinstance(stages[1], TrackingStage), "Stage 1 must be TrackingStage"
            assert isinstance(stages[2], AssociationStubStage), (
                "Stage 2 must be AssociationStubStage"
            )
            assert stages[3] is mock_rec.return_value, (
                "Stage 3 must be ReconstructionStage"
            )

    def test_synthetic_mode_first_stage_is_synthetic(self) -> None:
        """First stage in synthetic mode is SyntheticDataStage."""
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
        """Last stage in synthetic mode is ReconstructionStage."""
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


class TestTrackingStageDirectly:
    """Tests for TrackingStage behavior and AssociationStubStage."""

    def test_tracking_stage_produces_tracks_2d(self) -> None:
        """TrackingStage sets context.tracks_2d (not None)."""
        from aquapose.core.context import PipelineContext
        from aquapose.core.tracking import TrackingStage
        from aquapose.engine.config import TrackingConfig

        ctx = PipelineContext()
        ctx.camera_ids = []
        ctx.detections = []
        stage = TrackingStage(config=TrackingConfig())
        result_ctx, _carry = stage.run(ctx)
        assert result_ctx.tracks_2d is not None
        assert isinstance(result_ctx.tracks_2d, dict)

    def test_tracking_stage_creates_default_carry_if_none(self) -> None:
        """TrackingStage creates a valid CarryForward when carry=None."""
        from aquapose.core.context import CarryForward, PipelineContext
        from aquapose.core.tracking import TrackingStage
        from aquapose.engine.config import TrackingConfig

        ctx = PipelineContext()
        ctx.camera_ids = []
        ctx.detections = []
        stage = TrackingStage(config=TrackingConfig())
        _, carry = stage.run(ctx)
        assert isinstance(carry, CarryForward)
        assert isinstance(carry.tracks_2d_state, dict)

    def test_association_stub_produces_empty_tracklet_groups(self) -> None:
        """AssociationStubStage sets context.tracklet_groups to an empty list."""
        from aquapose.core.context import PipelineContext

        ctx = PipelineContext()
        stub = AssociationStubStage()
        result_ctx = stub.run(ctx)
        assert result_ctx.tracklet_groups == []

    def test_tracking_stage_present_in_production_build(self) -> None:
        """TrackingStage appears at position 1 in production build."""

        config = PipelineConfig(
            mode="production",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.ReconstructionStage"),
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            stage_names = [type(s).__name__ for s in stages]
            assert "TrackingStage" in stage_names
            assert "AssociationStubStage" in stage_names
            assert stage_names.index("TrackingStage") == 1
            assert stage_names.index("AssociationStubStage") == 2

    def test_no_tracking_stub_stage_in_build(self) -> None:
        """TrackingStubStage is gone — not present in any build output."""
        config = PipelineConfig(
            mode="production",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.ReconstructionStage"),
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            stage_names = [type(s).__name__ for s in stages]
            assert "TrackingStubStage" not in stage_names

    @pytest.mark.parametrize("mode", ["production", "diagnostic", "benchmark"])
    def test_non_synthetic_modes_include_tracking_stage(self, mode: str) -> None:
        """All non-synthetic modes include TrackingStage at position 1."""
        from aquapose.core.tracking import TrackingStage

        config = PipelineConfig(
            mode=mode,
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.MidlineStage"),
            patch("aquapose.core.ReconstructionStage"),
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            stage_types = [type(s) for s in stages]
            assert TrackingStage in stage_types
            assert AssociationStubStage in stage_types
