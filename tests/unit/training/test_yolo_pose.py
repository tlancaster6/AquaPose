"""Unit tests for the YOLO-pose training wrapper."""

from __future__ import annotations


def test_train_yolo_pose_importable() -> None:
    """train_yolo_pose is importable from the training package."""
    from aquapose.training.yolo_pose import train_yolo_pose

    assert callable(train_yolo_pose)


def test_train_yolo_pose_missing_yaml(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """train_yolo_pose raises FileNotFoundError when dataset.yaml is absent."""
    import pytest

    from aquapose.training.yolo_pose import train_yolo_pose

    with pytest.raises(FileNotFoundError, match=r"dataset\.yaml"):
        train_yolo_pose(
            data_dir=tmp_path,
            output_dir=tmp_path / "out",
        )
