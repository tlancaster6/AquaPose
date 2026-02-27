"""Unit tests for build_stages mode-aware stage factory (v2.1).

v2.1: build_stages returns 5 stages in production mode and 4 in synthetic mode.
Stage order: Detection -> TrackingStubStage -> AssociationStubStage -> Midline -> Reconstruction
Synthetic order: SyntheticDataStage -> TrackingStubStage -> AssociationStubStage -> Reconstruction

TrackingStubStage and AssociationStubStage are inline in engine/pipeline.py (not patched).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aquapose.engine.config import PipelineConfig, SyntheticConfig
from aquapose.engine.pipeline import AssociationStubStage, TrackingStubStage


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
        """Production mode: Detection -> TrackingStub -> AssociationStub -> Midline -> Reconstruction."""
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
            assert isinstance(stages[1], TrackingStubStage), (
                "Stage 1 must be TrackingStubStage"
            )
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
        """Synthetic mode: SyntheticData -> TrackingStub -> AssociationStub -> Reconstruction."""
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
            assert isinstance(stages[1], TrackingStubStage), (
                "Stage 1 must be TrackingStubStage"
            )
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


class TestStubStagesDirectly:
    """Tests for TrackingStubStage and AssociationStubStage behavior."""

    def test_tracking_stub_produces_empty_tracks_2d(self) -> None:
        """TrackingStubStage sets context.tracks_2d to an empty dict."""
        from aquapose.core.context import PipelineContext

        ctx = PipelineContext()
        stub = TrackingStubStage()
        result_ctx, _carry = stub.run(ctx)
        assert result_ctx.tracks_2d == {}

    def test_tracking_stub_creates_default_carry_if_none(self) -> None:
        """TrackingStubStage creates a default CarryForward when carry=None."""
        from aquapose.core.context import CarryForward, PipelineContext

        ctx = PipelineContext()
        stub = TrackingStubStage()
        _, carry = stub.run(ctx)
        assert isinstance(carry, CarryForward)
        assert carry.tracks_2d_state == {}

    def test_tracking_stub_passes_carry_through_unchanged(self) -> None:
        """TrackingStubStage returns carry unchanged when explicitly passed."""
        from aquapose.core.context import CarryForward, PipelineContext

        ctx = PipelineContext()
        custom_carry = CarryForward(tracks_2d_state={"cam1": {"some": "state"}})
        stub = TrackingStubStage()
        _, returned_carry = stub.run(ctx, carry=custom_carry)
        assert returned_carry is custom_carry
        assert returned_carry.tracks_2d_state == {"cam1": {"some": "state"}}

    def test_association_stub_produces_empty_tracklet_groups(self) -> None:
        """AssociationStubStage sets context.tracklet_groups to an empty list."""
        from aquapose.core.context import PipelineContext

        ctx = PipelineContext()
        stub = AssociationStubStage()
        result_ctx = stub.run(ctx)
        assert result_ctx.tracklet_groups == []

    def test_stub_stages_present_in_production_build(self) -> None:
        """Both stub stages appear in production build at positions 1 and 2."""
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
            assert "TrackingStubStage" in stage_names
            assert "AssociationStubStage" in stage_names
            assert stage_names.index("TrackingStubStage") == 1
            assert stage_names.index("AssociationStubStage") == 2

    @pytest.mark.parametrize("mode", ["production", "diagnostic", "benchmark"])
    def test_non_synthetic_modes_include_stubs(self, mode: str) -> None:
        """All non-synthetic modes include both stub stages."""
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
            assert TrackingStubStage in stage_types
            assert AssociationStubStage in stage_types
