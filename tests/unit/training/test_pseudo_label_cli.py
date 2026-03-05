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


def _make_context_with_detections_and_tracks(n_frames: int = 2) -> PipelineContext:
    """Create a PipelineContext with midlines_3d, detections, and tracks_2d."""
    ctx = _make_context(n_frames)
    # Empty detections per frame (no detections for gap cameras)
    ctx.detections = [{} for _ in range(n_frames)]
    # Empty tracks_2d per camera
    ctx.tracks_2d = {"cam0": [], "cam1": []}
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
        "lut": {
            "tank_diameter": 1.0,
            "tank_height": 0.5,
            "voxel_resolution_m": 0.01,
            "margin_fraction": 0.1,
            "forward_grid_step": 4,
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


def _standard_patches():
    """Return context manager with standard mock patches for CLI tests."""
    return (
        patch("aquapose.evaluation.runner.load_run_context"),
        patch("aquapose.calibration.loader.load_calibration_data"),
        patch("aquapose.calibration.loader.compute_undistortion_maps"),
        patch("aquapose.calibration.projection.RefractiveProjectionModel"),
        patch("aquapose.core.types.frame_source.VideoFrameSource"),
    )


def _setup_standard_mocks(
    mock_load_ctx: MagicMock,
    mock_load_cal: MagicMock,
    mock_compute_undist: MagicMock,
    mock_proj_cls: MagicMock,
    mock_vfs_cls: MagicMock,
    context: PipelineContext,
    tmp_path: Path,
) -> MagicMock:
    """Configure standard mocks and return the mock projection model."""
    mock_cal = _setup_mock_calibration()
    mock_undist = MagicMock()
    mock_undist.K_new = torch.eye(3, dtype=torch.float32)

    mock_proj = MagicMock()
    mock_proj.project.side_effect = _mock_projection_project

    mock_frame_source = MagicMock()
    fake_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock_frame_source.read_frame.return_value = {
        "cam0": fake_image,
        "cam1": fake_image,
    }
    mock_frame_source.__enter__ = MagicMock(return_value=mock_frame_source)
    mock_frame_source.__exit__ = MagicMock(return_value=False)

    mock_load_ctx.return_value = (context, {})
    mock_load_cal.return_value = mock_cal
    mock_compute_undist.return_value = mock_undist
    mock_proj_cls.return_value = mock_proj
    mock_vfs_cls.return_value = mock_frame_source

    return mock_proj


class TestGenerateCommand:
    """Tests for the `pseudo-label generate` CLI command."""

    def test_fails_when_neither_flag_specified(self, tmp_path: Path) -> None:
        """Command fails when neither --consensus nor --gaps is provided."""
        config_path = _make_frozen_config(
            tmp_path, keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0]
        )

        runner = CliRunner()
        result = runner.invoke(
            pseudo_label_group, ["generate", "--config", str(config_path)]
        )

        assert result.exit_code != 0
        assert "At least one of --consensus or --gaps" in result.output

    def test_fails_when_keypoint_t_values_is_none(self, tmp_path: Path) -> None:
        """Command fails with clear error when keypoint_t_values is None."""
        config_path = _make_frozen_config(tmp_path, keypoint_t_values=None)

        runner = CliRunner()
        result = runner.invoke(
            pseudo_label_group,
            ["generate", "--consensus", "--config", str(config_path)],
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
                pseudo_label_group,
                ["generate", "--consensus", "--config", str(config_path)],
            )

        assert result.exit_code != 0
        assert "No diagnostic caches" in result.output

    def test_consensus_produces_output_structure(self, tmp_path: Path) -> None:
        """--consensus produces expected directory structure with labels."""
        config_path = _make_frozen_config(
            tmp_path, keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0]
        )
        (tmp_path / "videos").mkdir()

        context = _make_context(n_frames=2)

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
            _setup_standard_mocks(
                mock_load_ctx,
                mock_load_cal,
                mock_compute_undist,
                mock_proj_cls,
                mock_vfs_cls,
                context,
                tmp_path,
            )

            runner = CliRunner()
            result = runner.invoke(
                pseudo_label_group,
                ["generate", "--consensus", "--config", str(config_path)],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = tmp_path / "pseudo_labels"

        # Check consensus directory structure
        assert (pseudo_dir / "consensus" / "obb" / "images" / "train").is_dir()
        assert (pseudo_dir / "consensus" / "obb" / "labels" / "train").is_dir()
        assert (pseudo_dir / "consensus" / "pose" / "images" / "train").is_dir()
        assert (pseudo_dir / "consensus" / "pose" / "labels" / "train").is_dir()

        # Check dataset.yaml files
        assert (pseudo_dir / "consensus" / "obb" / "dataset.yaml").exists()
        assert (pseudo_dir / "consensus" / "pose" / "dataset.yaml").exists()

        obb_ds = yaml.safe_load(
            (pseudo_dir / "consensus" / "obb" / "dataset.yaml").read_text()
        )
        assert obb_ds["nc"] == 1
        assert obb_ds["names"] == {0: "fish"}

        pose_ds = yaml.safe_load(
            (pseudo_dir / "consensus" / "pose" / "dataset.yaml").read_text()
        )
        assert pose_ds["nc"] == 1
        assert pose_ds["kpt_shape"] == [5, 3]
        assert "flip_idx" in pose_ds

        # Check confidence sidecar
        assert (pseudo_dir / "consensus" / "confidence.json").exists()
        sidecar = json.loads((pseudo_dir / "consensus" / "confidence.json").read_text())
        assert len(sidecar) > 0

        # Check OBB label files exist
        obb_labels = list(
            (pseudo_dir / "consensus" / "obb" / "labels" / "train").glob("*.txt")
        )
        assert len(obb_labels) > 0

        # Check OBB label content format
        label_content = obb_labels[0].read_text().strip()
        parts = label_content.split()
        assert len(parts) == 9  # cls + 4 corners x 2

        # Check pose files have fish-index suffix pattern (crop-based)
        pose_images = list(
            (pseudo_dir / "consensus" / "pose" / "images" / "train").glob("*.jpg")
        )
        assert len(pose_images) > 0
        # Filename pattern: {frame:06d}_{cam}_{fish:03d}.jpg
        for img_path in pose_images:
            stem = img_path.stem
            parts_name = stem.split("_")
            assert len(parts_name) >= 3, f"Expected fish-index suffix: {stem}"
            # Last part should be zero-padded fish index (e.g. '000')
            assert parts_name[-1].isdigit()

        # Check pose label content has crop-normalized coordinates
        pose_labels = list(
            (pseudo_dir / "consensus" / "pose" / "labels" / "train").glob("*.txt")
        )
        assert len(pose_labels) > 0
        pose_content = pose_labels[0].read_text().strip()
        pose_values = [float(v) for v in pose_content.split()]
        # bbox values (indices 1-4) should be in [0, 1]
        for v in pose_values[1:5]:
            assert 0.0 <= v <= 1.0

    def test_gaps_produces_output_structure(self, tmp_path: Path) -> None:
        """--gaps produces expected directory structure with gap labels."""
        config_path = _make_frozen_config(
            tmp_path, keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0]
        )
        (tmp_path / "videos").mkdir()

        context = _make_context_with_detections_and_tracks(n_frames=2)

        # Mock detect_gaps to return a gap camera
        mock_gap_result = {
            "obb_line": "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
            "pose_line": "0 0.5 0.5 0.1 0.1 " + " ".join(["0.5 0.5 2"] * 5),
            "confidence": 0.8,
            "raw_metrics": {"mean_residual": 2.0, "n_cameras": 3},
            "keypoints_2d": np.zeros((5, 2)),
            "visibility": np.ones(5, dtype=bool),
        }

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
            patch("aquapose.training.pseudo_label_cli.detect_gaps") as mock_detect_gaps,
            patch(
                "aquapose.training.pseudo_label_cli.generate_gap_fish_labels"
            ) as mock_gen_gap,
            patch("aquapose.calibration.luts.load_inverse_luts") as mock_load_luts,
        ):
            _setup_standard_mocks(
                mock_load_ctx,
                mock_load_cal,
                mock_compute_undist,
                mock_proj_cls,
                mock_vfs_cls,
                context,
                tmp_path,
            )
            mock_detect_gaps.return_value = [("cam1", "no-detection")]
            mock_gen_gap.return_value = mock_gap_result
            mock_load_luts.return_value = MagicMock()

            runner = CliRunner()
            result = runner.invoke(
                pseudo_label_group,
                ["generate", "--gaps", "--config", str(config_path)],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = tmp_path / "pseudo_labels"

        # Check gap directory structure
        assert (pseudo_dir / "gap" / "obb" / "images" / "train").is_dir()
        assert (pseudo_dir / "gap" / "obb" / "labels" / "train").is_dir()
        assert (pseudo_dir / "gap" / "pose" / "images" / "train").is_dir()
        assert (pseudo_dir / "gap" / "pose" / "labels" / "train").is_dir()

        # Check gap dataset.yaml
        assert (pseudo_dir / "gap" / "obb" / "dataset.yaml").exists()
        assert (pseudo_dir / "gap" / "pose" / "dataset.yaml").exists()

        # Check gap confidence sidecar
        assert (pseudo_dir / "gap" / "confidence.json").exists()

    def test_gap_sidecar_contains_gap_fields(self, tmp_path: Path) -> None:
        """Gap confidence sidecar entries contain gap_reason and n_source_cameras."""
        config_path = _make_frozen_config(
            tmp_path, keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0]
        )
        (tmp_path / "videos").mkdir()

        context = _make_context_with_detections_and_tracks(n_frames=1)

        mock_gap_result = {
            "obb_line": "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
            "pose_line": "0 0.5 0.5 0.1 0.1 " + " ".join(["0.5 0.5 2"] * 5),
            "confidence": 0.8,
            "raw_metrics": {"mean_residual": 2.0, "n_cameras": 3},
            "keypoints_2d": np.zeros((5, 2)),
            "visibility": np.ones(5, dtype=bool),
        }

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
            patch("aquapose.training.pseudo_label_cli.detect_gaps") as mock_detect_gaps,
            patch(
                "aquapose.training.pseudo_label_cli.generate_gap_fish_labels"
            ) as mock_gen_gap,
            patch("aquapose.calibration.luts.load_inverse_luts") as mock_load_luts,
        ):
            _setup_standard_mocks(
                mock_load_ctx,
                mock_load_cal,
                mock_compute_undist,
                mock_proj_cls,
                mock_vfs_cls,
                context,
                tmp_path,
            )
            mock_detect_gaps.return_value = [("cam1", "no-detection")]
            mock_gen_gap.return_value = mock_gap_result
            mock_load_luts.return_value = MagicMock()

            runner = CliRunner()
            result = runner.invoke(
                pseudo_label_group,
                ["generate", "--gaps", "--config", str(config_path)],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = tmp_path / "pseudo_labels"
        sidecar = json.loads((pseudo_dir / "gap" / "confidence.json").read_text())

        # Check that gap sidecar entries have gap_reason and n_source_cameras
        assert len(sidecar) > 0
        for _image_name, image_data in sidecar.items():
            for label_entry in image_data["labels"]:
                assert "gap_reason" in label_entry
                assert "n_source_cameras" in label_entry
                assert label_entry["gap_reason"] == "no-detection"
                assert isinstance(label_entry["n_source_cameras"], int)

    def test_both_flags_together(self, tmp_path: Path) -> None:
        """Both --consensus and --gaps can run together."""
        config_path = _make_frozen_config(
            tmp_path, keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0]
        )
        (tmp_path / "videos").mkdir()

        context = _make_context_with_detections_and_tracks(n_frames=1)

        mock_gap_result = {
            "obb_line": "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
            "pose_line": "0 0.5 0.5 0.1 0.1 " + " ".join(["0.5 0.5 2"] * 5),
            "confidence": 0.8,
            "raw_metrics": {"mean_residual": 2.0, "n_cameras": 3},
            "keypoints_2d": np.zeros((5, 2)),
            "visibility": np.ones(5, dtype=bool),
        }

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
            patch("aquapose.training.pseudo_label_cli.detect_gaps") as mock_detect_gaps,
            patch(
                "aquapose.training.pseudo_label_cli.generate_gap_fish_labels"
            ) as mock_gen_gap,
            patch("aquapose.calibration.luts.load_inverse_luts") as mock_load_luts,
        ):
            _setup_standard_mocks(
                mock_load_ctx,
                mock_load_cal,
                mock_compute_undist,
                mock_proj_cls,
                mock_vfs_cls,
                context,
                tmp_path,
            )
            mock_detect_gaps.return_value = [("cam1", "no-tracklet")]
            mock_gen_gap.return_value = mock_gap_result
            mock_load_luts.return_value = MagicMock()

            runner = CliRunner()
            result = runner.invoke(
                pseudo_label_group,
                [
                    "generate",
                    "--consensus",
                    "--gaps",
                    "--config",
                    str(config_path),
                ],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = tmp_path / "pseudo_labels"
        # Both consensus and gap directories should exist
        assert (pseudo_dir / "consensus" / "obb" / "labels" / "train").is_dir()
        assert (pseudo_dir / "gap" / "obb" / "labels" / "train").is_dir()

        # Both should have confidence sidecars
        assert (pseudo_dir / "consensus" / "confidence.json").exists()
        assert (pseudo_dir / "gap" / "confidence.json").exists()

    def test_help_text(self) -> None:
        """generate --help shows expected options."""
        runner = CliRunner()
        result = runner.invoke(pseudo_label_group, ["generate", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--lateral-pad" in result.output
        assert "--max-camera-residual" in result.output
        assert "--consensus" in result.output
        assert "--gaps" in result.output
        assert "--min-cameras" in result.output
        assert "--crop-width" in result.output
        assert "--crop-height" in result.output
