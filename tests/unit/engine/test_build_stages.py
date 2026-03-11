"""Unit tests for build_stages mode-aware stage factory (v3.7).

v3.7: build_stages returns 5 stages in production mode and 4 in synthetic mode.
Stage order: Detection -> PoseStage -> TrackingStage -> AssociationStage -> Reconstruction
Synthetic order: SyntheticDataStage -> TrackingStage -> AssociationStage -> Reconstruction

TrackingStage is imported from aquapose.core.tracking.
AssociationStage is inline in engine/pipeline.py.

v3.7: PoseStage (Stage 2) runs before tracking, enriching Detection objects with
raw anatomical keypoints in-place. Tests patch VideoFrameSource to avoid real I/O.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aquapose.core.association import AssociationStage
from aquapose.engine.config import PipelineConfig, SyntheticConfig

# Shared patch target for VideoFrameSource (created in build_stages for non-synthetic modes)
_VFS_PATCH = "aquapose.core.types.frame_source.VideoFrameSource.__init__"


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
            patch(_VFS_PATCH, return_value=None),
            patch("aquapose.core.DetectionStage") as mock_det,
            patch("aquapose.core.PoseStage") as mock_pose,
            patch("aquapose.core.ReconstructionStage") as mock_rec,
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert len(stages) == 5, f"Expected 5 stages, got {len(stages)}"
            mock_det.assert_called_once()
            mock_pose.assert_called_once()
            mock_rec.assert_called_once()

    def test_production_mode_stage_order(self) -> None:
        """Production mode: Detection -> PoseStage -> TrackingStage -> AssociationStage -> Reconstruction."""
        from aquapose.core.tracking import TrackingStage

        config = PipelineConfig(
            mode="production",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch(_VFS_PATCH, return_value=None),
            patch("aquapose.core.DetectionStage") as mock_det,
            patch("aquapose.core.PoseStage") as mock_pose,
            patch("aquapose.core.ReconstructionStage") as mock_rec,
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert stages[0] is mock_det.return_value, "Stage 0 must be DetectionStage"
            assert stages[1] is mock_pose.return_value, "Stage 1 must be PoseStage"
            assert isinstance(stages[2], TrackingStage), "Stage 2 must be TrackingStage"
            assert isinstance(stages[3], AssociationStage), (
                "Stage 3 must be AssociationStage"
            )
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
            patch(_VFS_PATCH, return_value=None),
            patch("aquapose.core.DetectionStage") as mock_det,
            patch("aquapose.core.PoseStage"),
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
            patch(_VFS_PATCH, return_value=None),
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.PoseStage"),
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
            patch("aquapose.core.PoseStage"),
            patch("aquapose.core.SyntheticDataStage") as mock_syn,
            patch("aquapose.core.ReconstructionStage") as mock_rec,
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert len(stages) == 4, f"Expected 4 stages, got {len(stages)}"
            mock_syn.assert_called_once()
            mock_rec.assert_called_once()

    def test_synthetic_mode_stage_order(self) -> None:
        """Synthetic mode: SyntheticData -> TrackingStage -> AssociationStage -> Reconstruction."""
        from aquapose.core.tracking import TrackingStage

        config = PipelineConfig(
            mode="synthetic",
            calibration_path="/fake/cal.json",
            synthetic=SyntheticConfig(),
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.PoseStage"),
            patch("aquapose.core.SyntheticDataStage") as mock_syn,
            patch("aquapose.core.ReconstructionStage") as mock_rec,
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert stages[0] is mock_syn.return_value, (
                "Stage 0 must be SyntheticDataStage"
            )
            assert isinstance(stages[1], TrackingStage), "Stage 1 must be TrackingStage"
            assert isinstance(stages[2], AssociationStage), (
                "Stage 2 must be AssociationStage"
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
            patch("aquapose.core.PoseStage"),
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
            patch("aquapose.core.PoseStage"),
            patch("aquapose.core.SyntheticDataStage"),
            patch("aquapose.core.ReconstructionStage") as mock_rec,
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert stages[-1] is mock_rec.return_value


class TestTrackingStageDirectly:
    """Tests for TrackingStage behavior and AssociationStage."""

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
        """TrackingStage creates a valid ChunkHandoff when carry=None."""
        from aquapose.core.context import ChunkHandoff, PipelineContext
        from aquapose.core.tracking import TrackingStage
        from aquapose.engine.config import TrackingConfig

        ctx = PipelineContext()
        ctx.camera_ids = []
        ctx.detections = []
        stage = TrackingStage(config=TrackingConfig())
        _, carry = stage.run(ctx)
        assert isinstance(carry, ChunkHandoff)
        assert isinstance(carry.tracks_2d_state, dict)

    def test_tracking_stage_present_in_production_build(self) -> None:
        """TrackingStage appears at position 2 in production build (after PoseStage)."""

        config = PipelineConfig(
            mode="production",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch(_VFS_PATCH, return_value=None),
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.PoseStage"),
            patch("aquapose.core.ReconstructionStage"),
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            stage_names = [type(s).__name__ for s in stages]
            assert "TrackingStage" in stage_names
            assert "AssociationStage" in stage_names
            assert stage_names.index("TrackingStage") == 2
            assert stage_names.index("AssociationStage") == 3

    def test_no_tracking_stub_stage_in_build(self) -> None:
        """TrackingStubStage is gone — not present in any build output."""
        config = PipelineConfig(
            mode="production",
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch(_VFS_PATCH, return_value=None),
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.PoseStage"),
            patch("aquapose.core.ReconstructionStage"),
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            stage_names = [type(s).__name__ for s in stages]
            assert "TrackingStubStage" not in stage_names

    @pytest.mark.parametrize("mode", ["production", "diagnostic", "benchmark"])
    def test_non_synthetic_modes_include_tracking_stage(self, mode: str) -> None:
        """All non-synthetic modes include TrackingStage at position 2."""
        from aquapose.core.tracking import TrackingStage

        config = PipelineConfig(
            mode=mode,
            calibration_path="/fake/cal.json",
            video_dir="/fake/videos",
        )
        with (
            patch(_VFS_PATCH, return_value=None),
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.PoseStage"),
            patch("aquapose.core.ReconstructionStage"),
            patch("aquapose.core.SyntheticDataStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            stage_types = [type(s) for s in stages]
            assert TrackingStage in stage_types
            assert AssociationStage in stage_types


class TestBuildStagesConfigDevice:
    """Tests for build_stages config.device propagation."""

    def test_build_stages_uses_config_device(self) -> None:
        """build_stages with device='cpu' constructs without error."""
        config = PipelineConfig(
            mode="synthetic",
            calibration_path="/fake/cal.json",
            device="cpu",
        )
        with (
            patch("aquapose.core.DetectionStage"),
            patch("aquapose.core.PoseStage"),
            patch("aquapose.core.SyntheticDataStage") as mock_syn,
            patch("aquapose.core.ReconstructionStage"),
        ):
            from aquapose.engine.pipeline import build_stages

            stages = build_stages(config)
            assert len(stages) == 4
            mock_syn.assert_called_once()
            # Verify n_points (config.n_sample_points=10) was passed to SyntheticDataStage
            call_kwargs = mock_syn.call_args[1]
            assert call_kwargs.get("n_points") == config.n_sample_points
