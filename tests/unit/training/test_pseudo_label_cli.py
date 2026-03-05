"""Tests for pseudo-label CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import yaml
from click.testing import CliRunner

from aquapose.core.context import PipelineContext
from aquapose.core.types.reconstruction import Midline3D
from aquapose.training.pseudo_label_cli import pseudo_label_group


def _make_midline(
    fish_id: int = 1,
    frame_index: int = 0,
    per_camera_residuals: dict[str, float] | None = None,
) -> Midline3D:
    """Create a synthetic Midline3D for CLI testing."""
    control_points = np.zeros((7, 3), dtype=np.float32)
    control_points[:, 0] = np.linspace(0.0, 0.1, 7)
    control_points[:, 2] = 0.05

    knots = np.array([0, 0, 0, 0, 0.333, 0.5, 0.667, 1, 1, 1, 1], dtype=np.float32)

    if per_camera_residuals is None:
        per_camera_residuals = {"cam0": 2.0, "cam1": 3.0}

    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=control_points,
        knots=knots,
        degree=3,
        arc_length=0.1,
        half_widths=np.full(13, 0.005, dtype=np.float32),
        n_cameras=len(per_camera_residuals),
        mean_residual=2.0,
        max_residual=4.0,
        is_low_confidence=False,
        per_camera_residuals=per_camera_residuals,
    )


def _make_context(n_frames: int = 2) -> PipelineContext:
    """Create a synthetic PipelineContext with midlines_3d."""
    midlines_3d = []
    for frame_idx in range(n_frames):
        fish_dict = {
            1: _make_midline(
                fish_id=1,
                frame_index=frame_idx,
                per_camera_residuals={"cam0": 2.0, "cam1": 3.0},
            )
        }
        midlines_3d.append(fish_dict)

    ctx = PipelineContext()
    ctx.frame_count = n_frames
    ctx.camera_ids = ["cam0", "cam1"]
    ctx.midlines_3d = midlines_3d
    return ctx


def _mock_projection_project(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mock projection that returns linearly spaced pixels."""
    n = points.shape[0]
    pixels = torch.zeros(n, 2)
    pixels[:, 0] = torch.linspace(450, 550, n)
    pixels[:, 1] = torch.full((n,), 300.0)
    valid = torch.ones(n, dtype=torch.bool)
    return pixels, valid


def _make_frozen_config(
    tmp_path: Path, *, keypoint_t_values: list[float] | None = None
) -> Path:
    """Create a minimal frozen config YAML in tmp_path."""
    config_data = {
        "video_dir": str(tmp_path / "videos"),
        "calibration_path": str(tmp_path / "calib.json"),
        "output_dir": str(tmp_path),
        "n_animals": 3,
        "midline": {
            "backend": "pose_estimation",
            "weights_path": "fake.pt",
            "keypoint_t_values": keypoint_t_values,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    return config_path


def _setup_mock_calibration() -> MagicMock:
    """Create a mock CalibrationData with two cameras."""
    mock_cal = MagicMock()
    mock_cal.ring_cameras = ["cam0", "cam1"]

    mock_cam0 = MagicMock()
    mock_cam0.image_size = (1920, 1080)
    mock_cam1 = MagicMock()
    mock_cam1.image_size = (1920, 1080)
    mock_cal.cameras = {"cam0": mock_cam0, "cam1": mock_cam1}

    return mock_cal


class TestGenerateCommand:
    """Tests for the `pseudo-label generate` CLI command."""

    def test_fails_when_keypoint_t_values_is_none(self, tmp_path: Path) -> None:
        """Command fails with clear error when keypoint_t_values is None."""
        config_path = _make_frozen_config(tmp_path, keypoint_t_values=None)

        runner = CliRunner()
        result = runner.invoke(
            pseudo_label_group, ["generate", "--config", str(config_path)]
        )

        assert result.exit_code != 0
        assert "keypoint_t_values" in result.output

    def test_fails_when_no_diagnostic_caches(self, tmp_path: Path) -> None:
        """Command fails when no diagnostic caches found."""
        config_path = _make_frozen_config(
            tmp_path, keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0]
        )

        with patch("aquapose.evaluation.runner.load_run_context") as mock_load:
            mock_load.return_value = (None, {})

            runner = CliRunner()
            result = runner.invoke(
                pseudo_label_group, ["generate", "--config", str(config_path)]
            )

        assert result.exit_code != 0
        assert "No diagnostic caches" in result.output

    def test_produces_output_structure(self, tmp_path: Path) -> None:
        """Command produces expected directory structure with labels."""
        config_path = _make_frozen_config(
            tmp_path, keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0]
        )

        # Create mock video dir
        (tmp_path / "videos").mkdir()

        context = _make_context(n_frames=2)
        mock_cal = _setup_mock_calibration()

        # Mock projection model
        mock_proj = MagicMock()
        mock_proj.project.side_effect = _mock_projection_project

        # Mock frame source
        mock_frame_source = MagicMock()
        fake_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_frame_source.read_frame.return_value = {
            "cam0": fake_image,
            "cam1": fake_image,
        }
        mock_frame_source.__enter__ = MagicMock(return_value=mock_frame_source)
        mock_frame_source.__exit__ = MagicMock(return_value=False)

        # Mock undistortion maps
        mock_undist = MagicMock()
        mock_undist.K_new = torch.eye(3, dtype=torch.float32)

        with (
            patch("aquapose.evaluation.runner.load_run_context") as mock_load_ctx,
            patch("aquapose.calibration.loader.load_calibration_data") as mock_load_cal,
            patch(
                "aquapose.calibration.loader.compute_undistortion_maps"
            ) as mock_compute_undist,
            patch(
                "aquapose.calibration.projection.RefractiveProjectionModel"
            ) as mock_proj_cls,
            patch("aquapose.core.types.frame_source.VideoFrameSource") as mock_vfs_cls,
        ):
            mock_load_ctx.return_value = (context, {})
            mock_load_cal.return_value = mock_cal
            mock_compute_undist.return_value = mock_undist
            mock_proj_cls.return_value = mock_proj
            mock_vfs_cls.return_value = mock_frame_source

            runner = CliRunner()
            result = runner.invoke(
                pseudo_label_group, ["generate", "--config", str(config_path)]
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = tmp_path / "pseudo_labels"

        # Check directory structure
        assert (pseudo_dir / "obb" / "images" / "train").is_dir()
        assert (pseudo_dir / "obb" / "labels" / "train").is_dir()
        assert (pseudo_dir / "pose" / "images" / "train").is_dir()
        assert (pseudo_dir / "pose" / "labels" / "train").is_dir()

        # Check dataset.yaml files
        assert (pseudo_dir / "obb" / "dataset.yaml").exists()
        assert (pseudo_dir / "pose" / "dataset.yaml").exists()

        obb_ds = yaml.safe_load((pseudo_dir / "obb" / "dataset.yaml").read_text())
        assert obb_ds["nc"] == 1
        assert obb_ds["names"] == {0: "fish"}

        pose_ds = yaml.safe_load((pseudo_dir / "pose" / "dataset.yaml").read_text())
        assert pose_ds["nc"] == 1
        assert pose_ds["kpt_shape"] == [5, 3]
        assert "flip_idx" in pose_ds

        # Check confidence sidecar
        assert (pseudo_dir / "confidence.json").exists()
        sidecar = json.loads((pseudo_dir / "confidence.json").read_text())
        assert len(sidecar) > 0

        # Check label files exist
        obb_labels = list((pseudo_dir / "obb" / "labels" / "train").glob("*.txt"))
        pose_labels = list((pseudo_dir / "pose" / "labels" / "train").glob("*.txt"))
        assert len(obb_labels) > 0
        assert len(pose_labels) > 0

        # Check label file content
        label_content = obb_labels[0].read_text().strip()
        parts = label_content.split()
        # Should have 9 values per line (cls + 4 corners x 2)
        assert len(parts) == 9

    def test_help_text(self) -> None:
        """generate --help shows expected options."""
        runner = CliRunner()
        result = runner.invoke(pseudo_label_group, ["generate", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--lateral-pad" in result.output
        assert "--max-camera-residual" in result.output
