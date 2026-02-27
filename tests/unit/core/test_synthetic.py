"""Unit tests for SyntheticDataStage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from aquapose.core.context import PipelineContext, Stage
from aquapose.core.synthetic import SyntheticDataStage
from aquapose.engine.config import SyntheticConfig


def _make_mock_calibration(
    camera_names: list[str] | None = None,
) -> MagicMock:
    """Create a mock CalibrationData with ring cameras."""
    if camera_names is None:
        camera_names = ["cam1", "cam2", "cam3"]

    cal = MagicMock()
    cameras = {}
    for name in camera_names:
        cam = MagicMock()
        cam.name = name
        cam.K = torch.eye(3, dtype=torch.float32) * 500
        cam.K[0, 2] = 320.0
        cam.K[1, 2] = 240.0
        cam.R = torch.eye(3, dtype=torch.float32)
        cam.t = torch.zeros(3, dtype=torch.float32)
        cam.image_size = (640, 480)
        cam.is_auxiliary = False
        cam.is_fisheye = False
        cameras[name] = cam

    cal.cameras = cameras
    cal.ring_cameras = sorted(camera_names)
    cal.water_z = 0.0
    cal.interface_normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    cal.n_air = 1.0
    cal.n_water = 1.333
    return cal


def _make_mock_projection_model() -> MagicMock:
    """Create a mock projection model that returns valid pixel coordinates."""
    model = MagicMock()

    def mock_project(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = points.shape[0]
        # Project to center-ish of image with some spread
        pixels = torch.zeros((n, 2), dtype=torch.float32)
        pixels[:, 0] = 320.0 + points[:, 0] * 1000  # x -> u
        pixels[:, 1] = 240.0 + points[:, 1] * 1000  # y -> v
        valid = torch.ones(n, dtype=torch.bool)
        return pixels, valid

    model.project = mock_project
    return model


class TestSyntheticDataStageProtocol:
    """Tests that SyntheticDataStage satisfies Stage protocol."""

    def test_satisfies_stage_protocol(self) -> None:
        stage = SyntheticDataStage.__new__(SyntheticDataStage)
        assert isinstance(stage, Stage)

    def test_has_run_method(self) -> None:
        assert hasattr(SyntheticDataStage, "run")


class TestSyntheticDataStageRun:
    """Tests for SyntheticDataStage.run() output."""

    def _run_stage(self, config: SyntheticConfig | None = None) -> PipelineContext:
        if config is None:
            config = SyntheticConfig(fish_count=2, frame_count=5, seed=42)

        mock_cal = _make_mock_calibration()
        mock_model = _make_mock_projection_model()

        with (
            patch(
                "aquapose.core.synthetic.load_calibration_data",
                return_value=mock_cal,
            ),
            patch(
                "aquapose.core.synthetic.RefractiveProjectionModel",
                return_value=mock_model,
            ),
        ):
            stage = SyntheticDataStage(
                calibration_path="/fake/cal.json",
                synthetic_config=config,
            )
            context = PipelineContext()
            return stage.run(context)

    def test_populates_frame_count(self) -> None:
        context = self._run_stage()
        assert context.frame_count == 5

    def test_populates_camera_ids(self) -> None:
        context = self._run_stage()
        assert context.camera_ids is not None
        assert len(context.camera_ids) == 3
        assert "cam1" in context.camera_ids

    def test_populates_detections(self) -> None:
        context = self._run_stage()
        assert context.detections is not None
        assert len(context.detections) == 5  # frame_count

    def test_populates_annotated_detections(self) -> None:
        context = self._run_stage()
        assert context.annotated_detections is not None
        assert len(context.annotated_detections) == 5

    def test_detections_have_camera_keys(self) -> None:
        context = self._run_stage()
        for frame_dets in context.detections:
            assert "cam1" in frame_dets
            assert "cam2" in frame_dets
            assert "cam3" in frame_dets

    def test_some_cameras_have_detections(self) -> None:
        context = self._run_stage()
        has_any = False
        for frame_dets in context.detections:
            for _cam_id, dets in frame_dets.items():
                if len(dets) > 0:
                    has_any = True
                    break
        assert has_any, "No cameras received any detections"

    def test_annotated_detections_have_midlines(self) -> None:
        from aquapose.reconstruction.midline import Midline2D

        context = self._run_stage()
        found_midline = False
        for frame_annot in context.annotated_detections:
            for _cam_id, annots in frame_annot.items():
                for annot in annots:
                    if annot.midline is not None:
                        assert isinstance(annot.midline, Midline2D)
                        assert annot.midline.points.shape == (15, 2)
                        found_midline = True
        assert found_midline, "No annotated detections have midlines"

    def test_deterministic_with_same_seed(self) -> None:
        config = SyntheticConfig(fish_count=2, frame_count=3, seed=123)
        ctx1 = self._run_stage(config)
        ctx2 = self._run_stage(config)

        # Check same number of detections per frame per camera
        for f in range(3):
            for cam in ctx1.camera_ids:
                assert len(ctx1.detections[f][cam]) == len(ctx2.detections[f][cam])

    def test_noise_changes_output(self) -> None:
        config_no_noise = SyntheticConfig(
            fish_count=2, frame_count=3, noise_std=0.0, seed=42
        )
        config_noise = SyntheticConfig(
            fish_count=2, frame_count=3, noise_std=5.0, seed=42
        )
        ctx1 = self._run_stage(config_no_noise)
        ctx2 = self._run_stage(config_noise)

        # With noise, midline points should differ
        diffs_found = False
        for f in range(3):
            for cam in ctx1.camera_ids:
                for a1, a2 in zip(
                    ctx1.annotated_detections[f][cam],
                    ctx2.annotated_detections[f][cam],
                    strict=False,
                ):
                    if (
                        a1.midline is not None
                        and a2.midline is not None
                        and not np.allclose(a1.midline.points, a2.midline.points)
                    ):
                        diffs_found = True
        assert diffs_found, "Noise had no effect on midline points"

    def test_frame_count_matches_config(self) -> None:
        config = SyntheticConfig(fish_count=1, frame_count=10, seed=42)
        context = self._run_stage(config)
        assert context.frame_count == 10
        assert len(context.detections) == 10
        assert len(context.annotated_detections) == 10
