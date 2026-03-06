"""Tests for pseudo-label CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import yaml
from click.testing import CliRunner

from aquapose.cli import cli
from aquapose.core.context import PipelineContext
from aquapose.core.types.reconstruction import Midline3D


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
    run_dir: Path, *, keypoint_t_values: list[float] | None = None
) -> Path:
    """Create a minimal frozen config YAML in run_dir."""
    config_data = {
        "video_dir": str(run_dir / "videos"),
        "calibration_path": str(run_dir / "calib.json"),
        "output_dir": str(run_dir),
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
    config_path = run_dir / "config.yaml"
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


def _setup_run_dir(tmp_path: Path, *, keypoint_t_values: list[float] | None) -> Path:
    """Create a run directory with frozen config and return its path."""
    run_dir = tmp_path / "runs" / "run_20260306_120000"
    run_dir.mkdir(parents=True)
    _make_frozen_config(run_dir, keypoint_t_values=keypoint_t_values)
    (run_dir / "videos").mkdir(exist_ok=True)
    return run_dir


def _setup_project_with_run(
    tmp_path: Path,
    monkeypatch: MagicMock,
    *,
    keypoint_t_values: list[float] | None,
) -> tuple[Path, Path]:
    """Create a project dir with a run, patch resolve_project, return (project, run)."""
    proj = tmp_path / "test_project"
    proj.mkdir()
    (proj / "config.yaml").write_text("project_dir: .\n")
    runs_dir = proj / "runs"
    runs_dir.mkdir()

    run_dir = runs_dir / "run_20260306_120000"
    run_dir.mkdir()
    _make_frozen_config(run_dir, keypoint_t_values=keypoint_t_values)
    (run_dir / "videos").mkdir(exist_ok=True)

    monkeypatch.setattr(
        "aquapose.cli_utils.resolve_project",
        lambda name: proj,
    )
    return proj, run_dir


class TestGenerateCommand:
    """Tests for the `pseudo-label generate` CLI command."""

    def test_fails_when_keypoint_t_values_is_none(
        self, tmp_path: Path, monkeypatch: MagicMock
    ) -> None:
        """Command fails with clear error when keypoint_t_values is None."""
        _proj, _run_dir = _setup_project_with_run(
            tmp_path, monkeypatch, keypoint_t_values=None
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--project", "test", "pseudo-label", "generate"],
        )

        assert result.exit_code != 0
        assert "keypoint_t_values" in result.output

    def test_fails_when_no_diagnostic_caches(
        self, tmp_path: Path, monkeypatch: MagicMock
    ) -> None:
        """Command fails when no diagnostic caches found."""
        _proj, _run_dir = _setup_project_with_run(
            tmp_path,
            monkeypatch,
            keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        )

        with patch("aquapose.evaluation.runner.load_run_context") as mock_load:
            mock_load.return_value = (None, {})

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["--project", "test", "pseudo-label", "generate"],
            )

        assert result.exit_code != 0
        assert "No diagnostic caches" in result.output

    def test_generates_merged_obb_and_separate_pose(
        self, tmp_path: Path, monkeypatch: MagicMock
    ) -> None:
        """Default run produces merged OBB dir and separate pose dirs."""
        _proj, run_dir = _setup_project_with_run(
            tmp_path,
            monkeypatch,
            keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        )

        context = _make_context_with_detections_and_tracks(n_frames=2)

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
                cli,
                ["--project", "test", "pseudo-label", "generate"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = run_dir / "pseudo_labels"

        # Check merged OBB directory structure
        assert (pseudo_dir / "obb" / "images" / "train").is_dir()
        assert (pseudo_dir / "obb" / "labels" / "train").is_dir()

        # Check consensus pose directory
        assert (pseudo_dir / "pose" / "consensus" / "images" / "train").is_dir()
        assert (pseudo_dir / "pose" / "consensus" / "labels" / "train").is_dir()

        # Check gap pose directory
        assert (pseudo_dir / "pose" / "gap" / "images" / "train").is_dir()
        assert (pseudo_dir / "pose" / "gap" / "labels" / "train").is_dir()

        # Check dataset.yaml files
        assert (pseudo_dir / "obb" / "dataset.yaml").exists()
        assert (pseudo_dir / "pose" / "consensus" / "dataset.yaml").exists()
        assert (pseudo_dir / "pose" / "gap" / "dataset.yaml").exists()

        obb_ds = yaml.safe_load((pseudo_dir / "obb" / "dataset.yaml").read_text())
        assert obb_ds["nc"] == 1
        assert obb_ds["names"] == {0: "fish"}

        pose_ds = yaml.safe_load(
            (pseudo_dir / "pose" / "consensus" / "dataset.yaml").read_text()
        )
        assert pose_ds["nc"] == 1
        assert pose_ds["kpt_shape"] == [5, 3]
        assert "flip_idx" in pose_ds

        # Check confidence sidecars
        assert (pseudo_dir / "obb" / "confidence.json").exists()
        assert (pseudo_dir / "pose" / "consensus" / "confidence.json").exists()
        assert (pseudo_dir / "pose" / "gap" / "confidence.json").exists()

        # Check OBB label files exist
        obb_labels = list((pseudo_dir / "obb" / "labels" / "train").glob("*.txt"))
        assert len(obb_labels) > 0

        # Check OBB label content format
        label_content = obb_labels[0].read_text().strip()
        parts = label_content.split()
        assert len(parts) == 9  # cls + 4 corners x 2

        # Check pose files have fish-index suffix pattern (crop-based)
        pose_images = list(
            (pseudo_dir / "pose" / "consensus" / "images" / "train").glob("*.jpg")
        )
        assert len(pose_images) > 0
        for img_path in pose_images:
            stem = img_path.stem
            parts_name = stem.split("_")
            assert len(parts_name) >= 3, f"Expected fish-index suffix: {stem}"
            assert parts_name[-1].isdigit()

    def test_generates_with_gaps(self, tmp_path: Path, monkeypatch: MagicMock) -> None:
        """Default run (gaps enabled) produces gap pose labels."""
        _proj, run_dir = _setup_project_with_run(
            tmp_path,
            monkeypatch,
            keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        )

        context = _make_context_with_detections_and_tracks(n_frames=2)

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
                cli,
                ["--project", "test", "pseudo-label", "generate"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = run_dir / "pseudo_labels"

        # Gap pose directory should exist
        assert (pseudo_dir / "pose" / "gap" / "images" / "train").is_dir()
        assert (pseudo_dir / "pose" / "gap" / "labels" / "train").is_dir()
        assert (pseudo_dir / "pose" / "gap" / "dataset.yaml").exists()
        assert (pseudo_dir / "pose" / "gap" / "confidence.json").exists()

    def test_obb_sidecar_contains_source_field(
        self, tmp_path: Path, monkeypatch: MagicMock
    ) -> None:
        """Merged OBB confidence sidecar has source and gap fields per fish."""
        _proj, run_dir = _setup_project_with_run(
            tmp_path,
            monkeypatch,
            keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        )

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
                cli,
                ["--project", "test", "pseudo-label", "generate"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = run_dir / "pseudo_labels"
        sidecar = json.loads((pseudo_dir / "obb" / "confidence.json").read_text())

        assert len(sidecar) > 0
        for _image_name, image_data in sidecar.items():
            assert "tracked_fish_count" in image_data
            assert "complete" in image_data
            assert image_data["complete"] is True
            for label_entry in image_data["labels"]:
                assert "source" in label_entry
                if label_entry["source"] == "gap":
                    assert "gap_reason" in label_entry
                    assert "n_source_cameras" in label_entry

    def test_skip_gaps_omits_gap_labels(
        self, tmp_path: Path, monkeypatch: MagicMock
    ) -> None:
        """--skip-gaps produces only consensus OBB and pose, no gap output."""
        _proj, run_dir = _setup_project_with_run(
            tmp_path,
            monkeypatch,
            keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        )

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
                cli,
                ["--project", "test", "pseudo-label", "generate", "--skip-gaps"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = run_dir / "pseudo_labels"

        # OBB and consensus pose should exist
        assert (pseudo_dir / "obb" / "images" / "train").is_dir()
        assert (pseudo_dir / "pose" / "consensus" / "images" / "train").is_dir()

        # Gap pose should NOT exist
        assert not (pseudo_dir / "pose" / "gap" / "images" / "train").exists()

    def test_completeness_filter_skips_incomplete(
        self, tmp_path: Path, monkeypatch: MagicMock
    ) -> None:
        """OBB image is skipped when not all tracked fish have labels."""
        _proj, run_dir = _setup_project_with_run(
            tmp_path,
            monkeypatch,
            keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        )

        # 2 tracked fish, but only fish_id=1 reaches reconstruction
        context = _make_context(n_frames=1)

        # Create tracks_2d with 2 tracklets both "detected" at frame 0
        mock_tracklet_1 = MagicMock()
        mock_tracklet_1.frames = [0]
        mock_tracklet_1.frame_status = ["detected"]

        mock_tracklet_2 = MagicMock()
        mock_tracklet_2.frames = [0]
        mock_tracklet_2.frame_status = ["detected"]

        context.tracks_2d = {
            "cam0": [mock_tracklet_1, mock_tracklet_2],
            "cam1": [mock_tracklet_1, mock_tracklet_2],
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
                cli,
                ["--project", "test", "pseudo-label", "generate", "--skip-gaps"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = run_dir / "pseudo_labels"

        # OBB images should be skipped (1 label < 2 tracked)
        obb_images = list((pseudo_dir / "obb" / "images" / "train").glob("*.jpg"))
        assert len(obb_images) == 0

        # Verify "skipped (incomplete)" appears in output
        assert "skipped (incomplete)" in result.output

    def test_completeness_filter_passes_complete(
        self, tmp_path: Path, monkeypatch: MagicMock
    ) -> None:
        """OBB image is written when all tracked fish have labels."""
        _proj, run_dir = _setup_project_with_run(
            tmp_path,
            monkeypatch,
            keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        )

        # 1 tracked fish, 1 reconstructed fish -> complete
        context = _make_context(n_frames=1)

        mock_tracklet = MagicMock()
        mock_tracklet.frames = [0]
        mock_tracklet.frame_status = ["detected"]

        context.tracks_2d = {
            "cam0": [mock_tracklet],
            "cam1": [mock_tracklet],
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
                cli,
                ["--project", "test", "pseudo-label", "generate", "--skip-gaps"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        pseudo_dir = run_dir / "pseudo_labels"
        obb_images = list((pseudo_dir / "obb" / "images" / "train").glob("*.jpg"))
        assert len(obb_images) > 0

    def test_help_text(self) -> None:
        """generate --help shows expected options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pseudo-label", "generate", "--help"])

        assert result.exit_code == 0
        assert "--lateral-pad" in result.output
        assert "--max-camera-residual" in result.output
        assert "--skip-gaps" in result.output
        assert "--min-cameras" in result.output
        assert "--crop-width" in result.output
        assert "--crop-height" in result.output
        assert "--config" not in result.output


class TestAssembleCommand:
    """Tests for the `pseudo-label assemble` CLI command."""

    def test_assemble_help_text(self) -> None:
        """assemble --help shows expected options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pseudo-label", "assemble", "--help"])

        assert result.exit_code == 0
        assert "--run-dir" in result.output
        assert "--manual-dir" in result.output
        assert "--output-dir" in result.output
        assert "--model-type" in result.output
        assert "--consensus-threshold" in result.output
        assert "--gap-threshold" in result.output
        assert "--exclude-gap-reason" in result.output
        assert "--seed" in result.output
        # Frame selection options
        assert "--temporal-step" in result.output
        assert "--diversity-bins" in result.output
        assert "--diversity-max-per-bin" in result.output

    def test_assemble_produces_output(self, tmp_path: Path) -> None:
        """assemble creates YOLO dataset from synthetic pseudo-labels."""
        from aquapose.training.pseudo_label_cli import pseudo_label_group

        # Set up synthetic run directory with merged OBB pseudo-labels
        run_dir = tmp_path / "run_001"
        img_dir = run_dir / "pseudo_labels" / "obb" / "images" / "train"
        lbl_dir = run_dir / "pseudo_labels" / "obb" / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(3):
            stem = f"00000{i}_cam0"
            (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8")
            (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        confidence = {
            f"00000{i}_cam0": {
                "labels": [
                    {
                        "fish_id": 1,
                        "confidence": 0.8,
                        "raw_metrics": {},
                        "source": "consensus",
                    }
                ],
                "tracked_fish_count": 1,
                "complete": True,
            }
            for i in range(3)
        }
        conf_dir = run_dir / "pseudo_labels" / "obb"
        (conf_dir / "confidence.json").write_text(json.dumps(confidence))

        output_dir = tmp_path / "assembled"
        runner = CliRunner()
        result = runner.invoke(
            pseudo_label_group,
            [
                "assemble",
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
                "--model-type",
                "obb",
                "--consensus-threshold",
                "0.5",
                "--pseudo-val-fraction",
                "0.0",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Dataset assembly complete" in result.output
        assert (output_dir / "dataset.yaml").exists()
        assert (output_dir / "images" / "train").is_dir()
        assert len(list((output_dir / "images" / "train").glob("*.jpg"))) == 3
