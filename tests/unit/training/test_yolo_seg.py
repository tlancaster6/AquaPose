"""Unit tests for the YOLO-seg training wrapper."""

from __future__ import annotations


def test_train_yolo_seg_importable() -> None:
    """train_yolo_seg is importable from the training package."""
    from aquapose.training.yolo_seg import train_yolo_seg

    assert callable(train_yolo_seg)


def test_train_yolo_seg_missing_yaml(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """train_yolo_seg raises FileNotFoundError when dataset.yaml is absent."""
    import pytest

    from aquapose.training.yolo_seg import train_yolo_seg

    with pytest.raises(FileNotFoundError, match=r"dataset\.yaml"):
        train_yolo_seg(
            data_dir=tmp_path,
            output_dir=tmp_path / "out",
        )
