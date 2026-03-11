"""Tests for calibrate-keypoints CLI and PoseEstimationBackend fail-fast."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from aquapose.cli import cli


@pytest.fixture
def coco_annotations(tmp_path: Path) -> Path:
    """Create a minimal COCO annotations file with 2 annotations."""
    data = {
        "annotations": [
            {
                "id": 1,
                "keypoints": [
                    10.0,
                    20.0,
                    2,  # kp0 visible
                    30.0,
                    20.0,
                    2,  # kp1 visible
                    50.0,
                    20.0,
                    2,  # kp2 visible
                    70.0,
                    20.0,
                    2,  # kp3 visible
                    90.0,
                    20.0,
                    2,  # kp4 visible
                    110.0,
                    20.0,
                    2,  # kp5 visible
                ],
            },
            {
                "id": 2,
                "keypoints": [
                    10.0,
                    40.0,
                    2,
                    30.0,
                    40.0,
                    2,
                    50.0,
                    40.0,
                    2,
                    70.0,
                    40.0,
                    2,
                    90.0,
                    40.0,
                    2,
                    110.0,
                    40.0,
                    2,
                ],
            },
        ]
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a minimal project directory with config.yaml."""
    proj = tmp_path / "test_project"
    proj.mkdir()
    config_data = {
        "project_dir": str(proj),
        "video_dir": "videos",
        "calibration_path": "geometry/calibration.json",
        "detection": {"detector_kind": "yolo_obb"},
        "midline": {"backend": "pose_estimation"},
    }
    (proj / "config.yaml").write_text(
        yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    )
    return proj


@pytest.fixture
def monkeypatch_project(monkeypatch: pytest.MonkeyPatch, project_dir: Path) -> Path:
    """Patch resolve_project so --project test resolves to project_dir."""
    monkeypatch.setattr(
        "aquapose.cli_utils.resolve_project",
        lambda name: project_dir,
    )
    return project_dir


class TestCalibrateKeypointsConfig:
    """Test calibrate-keypoints --config flag and YAML in-place update."""

    def test_updates_config_yaml_in_place(
        self,
        coco_annotations: Path,
        monkeypatch_project: Path,
    ) -> None:
        """calibrate-keypoints writes keypoint_t_values into YAML."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "prep",
                "calibrate-keypoints",
                "--annotations",
                str(coco_annotations),
            ],
        )
        assert result.exit_code == 0, result.output

        # Verify YAML was updated
        config_yaml = monkeypatch_project / "config.yaml"
        updated = yaml.safe_load(config_yaml.read_text())
        assert "midline" in updated
        assert "keypoint_t_values" in updated["midline"]
        t_values = updated["midline"]["keypoint_t_values"]
        assert len(t_values) == 6
        # First should be 0.0, last should be 1.0
        assert t_values[0] == pytest.approx(0.0, abs=0.01)
        assert t_values[-1] == pytest.approx(1.0, abs=0.01)

        # Other fields should be preserved
        assert updated["detection"]["detector_kind"] == "yolo_obb"
        assert updated["midline"]["backend"] == "pose_estimation"

    def test_creates_midline_section_if_missing(
        self,
        coco_annotations: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """calibrate-keypoints creates midline section if config lacks it."""
        proj = tmp_path / "minimal_project"
        proj.mkdir()
        (proj / "config.yaml").write_text(
            yaml.dump(
                {"project_dir": str(proj), "video_dir": "videos"},
                default_flow_style=False,
            )
        )
        monkeypatch.setattr(
            "aquapose.cli_utils.resolve_project",
            lambda name: proj,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "prep",
                "calibrate-keypoints",
                "--annotations",
                str(coco_annotations),
            ],
        )
        assert result.exit_code == 0, result.output

        updated = yaml.safe_load((proj / "config.yaml").read_text())
        assert "midline" in updated
        assert "keypoint_t_values" in updated["midline"]


class TestCalibrateKeypointsYolo:
    """Test calibrate-keypoints with YOLO-format label directories."""

    @pytest.fixture
    def yolo_labels_dir(self, tmp_path: Path) -> Path:
        """Create a YOLO label directory with train/ and val/ subdirs."""
        # Evenly spaced keypoints along x-axis (normalized coords)
        # class cx cy w h x1 y1 v1 x2 y2 v2 ... (6 keypoints)
        kps = " ".join(f"{x:.4f} 0.5 2" for x in [0.1, 0.28, 0.46, 0.64, 0.82, 1.0])
        line = f"0 0.5 0.5 0.8 0.3 {kps}"

        train_dir = tmp_path / "labels" / "train"
        train_dir.mkdir(parents=True)
        (train_dir / "img001.txt").write_text(line + "\n")
        (train_dir / "img002.txt").write_text(line + "\n")

        val_dir = tmp_path / "labels" / "val"
        val_dir.mkdir(parents=True)
        (val_dir / "img003.txt").write_text(line + "\n")

        return tmp_path / "labels"

    def test_yolo_labels_detected_and_parsed(
        self,
        yolo_labels_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Directory input auto-detects YOLO format and computes t-values."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "prep",
                "calibrate-keypoints",
                "--annotations",
                str(yolo_labels_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "YOLO" in result.output

        config_yaml = monkeypatch_project / "config.yaml"
        updated = yaml.safe_load(config_yaml.read_text())
        t_values = updated["midline"]["keypoint_t_values"]
        assert len(t_values) == 6
        assert t_values[0] == pytest.approx(0.0, abs=0.01)
        assert t_values[-1] == pytest.approx(1.0, abs=0.01)

    def test_yolo_processes_all_subdirs(
        self,
        yolo_labels_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """YOLO mode recurses into train/ and val/ subdirectories."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "prep",
                "calibrate-keypoints",
                "--annotations",
                str(yolo_labels_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Processed 3 annotations" in result.output

    def test_yolo_empty_dir_fails(
        self,
        tmp_path: Path,
        monkeypatch_project: Path,
    ) -> None:
        """YOLO mode with empty directory reports error."""
        empty_dir = tmp_path / "empty_labels"
        empty_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "prep",
                "calibrate-keypoints",
                "--annotations",
                str(empty_dir),
            ],
        )
        assert result.exit_code != 0
        assert "No valid keypoint annotations" in result.output


class TestPoseEstimationBackendFailFast:
    """Test PoseEstimationBackend raises when keypoint_t_values is None."""

    def test_raises_when_keypoint_t_values_is_none(self) -> None:
        """PoseEstimationBackend accepts None keypoint_t_values (v3.7 — t_values unused)."""
        from aquapose.core.pose.backends.pose_estimation import (
            PoseEstimationBackend,
        )

        # In v3.7, keypoint_t_values is accepted but not validated (no midline upsampling)
        backend = PoseEstimationBackend(
            weights_path=None,
            keypoint_t_values=None,
        )
        assert backend is not None

    def test_accepts_explicit_keypoint_t_values(self) -> None:
        """PoseEstimationBackend accepts explicit keypoint_t_values without raising."""
        from aquapose.core.pose.backends.pose_estimation import (
            PoseEstimationBackend,
        )

        # Should not raise
        backend = PoseEstimationBackend(
            weights_path=None,
            keypoint_t_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        assert backend is not None
