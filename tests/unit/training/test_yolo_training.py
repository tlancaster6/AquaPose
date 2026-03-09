"""Unit tests for the consolidated YOLO training wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a minimal data directory with dataset.yaml."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "dataset.yaml").write_text("train: images/train\nval: images/val\n")
    return d


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create an output directory."""
    d = tmp_path / "output"
    d.mkdir()
    return d


def _make_mock_yolo(save_dir: Path) -> MagicMock:
    """Build a mock YOLO class whose train() returns a results object."""
    mock_results = MagicMock()
    mock_results.save_dir = str(save_dir)

    mock_instance = MagicMock()
    mock_instance.train.return_value = mock_results

    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


class TestWeightCopying:
    """Tests for weight-copying logic after training completes."""

    def test_both_best_and_last_exist(
        self, data_dir: Path, output_dir: Path, tmp_path: Path
    ) -> None:
        """Both best.pt and last.pt exist -> both copied to output_dir."""
        save_dir = tmp_path / "ultralytics_save"
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"best-weights")
        (weights_dir / "last.pt").write_bytes(b"last-weights")

        mock_yolo = _make_mock_yolo(save_dir)

        with (
            patch("ultralytics.YOLO", mock_yolo),
            patch("aquapose.training.yolo_training.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            from aquapose.training.yolo_training import train_yolo

            result = train_yolo(data_dir, output_dir, "obb")

        assert (output_dir / "best_model.pt").exists()
        assert (output_dir / "last_model.pt").exists()
        assert (output_dir / "best_model.pt").read_bytes() == b"best-weights"
        assert (output_dir / "last_model.pt").read_bytes() == b"last-weights"
        assert result == output_dir / "best_model.pt"

    def test_only_best_exists(
        self, data_dir: Path, output_dir: Path, tmp_path: Path
    ) -> None:
        """Only best.pt exists -> best_model.pt copied, no last_model.pt."""
        save_dir = tmp_path / "ultralytics_save"
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"best-weights")

        mock_yolo = _make_mock_yolo(save_dir)

        with (
            patch("ultralytics.YOLO", mock_yolo),
            patch("aquapose.training.yolo_training.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            from aquapose.training.yolo_training import train_yolo

            train_yolo(data_dir, output_dir, "obb")

        assert (output_dir / "best_model.pt").exists()
        assert not (output_dir / "last_model.pt").exists()

    def test_only_last_exists_fallback(
        self, data_dir: Path, output_dir: Path, tmp_path: Path
    ) -> None:
        """Only last.pt exists -> last copied AND best_model.pt created (fallback)."""
        save_dir = tmp_path / "ultralytics_save"
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "last.pt").write_bytes(b"last-weights")

        mock_yolo = _make_mock_yolo(save_dir)

        with (
            patch("ultralytics.YOLO", mock_yolo),
            patch("aquapose.training.yolo_training.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            from aquapose.training.yolo_training import train_yolo

            train_yolo(data_dir, output_dir, "obb")

        assert (output_dir / "last_model.pt").exists()
        assert (output_dir / "best_model.pt").exists()
        # Fallback: best_model.pt should have last.pt contents
        assert (output_dir / "best_model.pt").read_bytes() == b"last-weights"

    def test_neither_exists(
        self, data_dir: Path, output_dir: Path, tmp_path: Path
    ) -> None:
        """Neither best.pt nor last.pt -> no weight files in output_dir."""
        save_dir = tmp_path / "ultralytics_save"
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True)

        mock_yolo = _make_mock_yolo(save_dir)

        with (
            patch("ultralytics.YOLO", mock_yolo),
            patch("aquapose.training.yolo_training.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            from aquapose.training.yolo_training import train_yolo

            train_yolo(data_dir, output_dir, "obb")

        assert not (output_dir / "best_model.pt").exists()
        assert not (output_dir / "last_model.pt").exists()


class TestImportability:
    """Tests for public API importability."""

    def test_train_yolo_importable(self) -> None:
        """train_yolo is importable from the training package."""
        from aquapose.training import train_yolo

        assert callable(train_yolo)

    def test_train_yolo_obb_importable(self) -> None:
        """train_yolo_obb is importable from the training package."""
        from aquapose.training import train_yolo_obb

        assert callable(train_yolo_obb)

    def test_train_yolo_seg_importable(self) -> None:
        """train_yolo_seg is importable from the training package."""
        from aquapose.training import train_yolo_seg

        assert callable(train_yolo_seg)

    def test_train_yolo_pose_importable(self) -> None:
        """train_yolo_pose is importable from the training package."""
        from aquapose.training import train_yolo_pose

        assert callable(train_yolo_pose)


class TestValidation:
    """Tests for input validation."""

    def test_missing_dataset_yaml(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when dataset.yaml is missing."""
        from aquapose.training.yolo_training import train_yolo

        with pytest.raises(FileNotFoundError, match=r"dataset\.yaml"):
            train_yolo(tmp_path, tmp_path / "out", "obb")

    def test_invalid_model_type(self, data_dir: Path, tmp_path: Path) -> None:
        """ValueError raised for unknown model_type."""
        from aquapose.training.yolo_training import train_yolo

        with pytest.raises(ValueError, match="Unknown model_type"):
            train_yolo(data_dir, tmp_path / "out", "invalid_type")
