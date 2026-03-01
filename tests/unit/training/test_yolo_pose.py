"""Unit tests for YOLO-pose NDJSON-to-YOLO.txt conversion and data.yaml rewrite."""

from __future__ import annotations

import json
from pathlib import Path

from aquapose.training.yolo_pose import (
    _convert_pose_ndjson_to_txt,
    _rewrite_data_yaml_pose,
)


def _write_ndjson(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as NDJSON to path."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


_SAMPLE_KEYPOINTS = [
    [0.1, 0.2, 2],
    [0.2, 0.3, 2],
    [0.3, 0.4, 2],
    [0.4, 0.5, 2],
    [0.5, 0.6, 2],
    [0.6, 0.7, 2],
]


def test_convert_pose_ndjson_to_txt(tmp_path: Path) -> None:
    """NDJSON-to-YOLO.txt pose conversion creates one .txt per image with correct format."""
    ndjson_path = tmp_path / "train.ndjson"
    records = [
        {
            "image": "images/train/fish_001.jpg",
            "width": 128,
            "height": 64,
            "annotations": [
                {
                    "class_id": 0,
                    "bbox": [0.5, 0.5, 0.8, 0.6],
                    "keypoints": _SAMPLE_KEYPOINTS,
                }
            ],
        },
        {
            "image": "images/train/fish_002.jpg",
            "width": 128,
            "height": 64,
            "annotations": [
                {
                    "class_id": 0,
                    "bbox": [0.3, 0.4, 0.6, 0.5],
                    "keypoints": _SAMPLE_KEYPOINTS,
                },
                {
                    "class_id": 0,
                    "bbox": [0.7, 0.6, 0.4, 0.3],
                    "keypoints": _SAMPLE_KEYPOINTS,
                },
            ],
        },
    ]
    _write_ndjson(ndjson_path, records)
    labels_dir = tmp_path / "labels" / "train"

    _convert_pose_ndjson_to_txt(ndjson_path, labels_dir)

    assert (labels_dir / "fish_001.txt").exists()
    assert (labels_dir / "fish_002.txt").exists()

    # fish_001: 1 annotation
    lines_001 = (
        (labels_dir / "fish_001.txt").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines_001) == 1
    parts = lines_001[0].split()

    # Format: class_id cx cy w h x1 y1 v1 x2 y2 v2 ... = 1 + 4 + 3*N_kp
    n_kp = len(_SAMPLE_KEYPOINTS)
    expected_len = 1 + 4 + 3 * n_kp
    assert len(parts) == expected_len, (
        f"Expected {expected_len} values, got {len(parts)}"
    )

    # class_id
    assert parts[0] == "0"

    # bbox values are valid floats
    for val_str in parts[1:5]:
        float(val_str)  # should not raise

    # Visibility values are integers (0 or 2)
    for i in range(n_kp):
        vis_idx = 5 + i * 3 + 2
        vis_val = parts[vis_idx]
        assert vis_val in ("0", "1", "2"), f"Unexpected visibility value: {vis_val!r}"

    # fish_002: 2 annotations
    lines_002 = (
        (labels_dir / "fish_002.txt").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines_002) == 2


def test_rewrite_data_yaml_pose_preserves_kpt_shape(tmp_path: Path) -> None:
    """Pose data.yaml rewrite preserves kpt_shape and kpt_names fields."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        (
            "train: train.ndjson\n"
            "val: val.ndjson\n"
            "nc: 1\n"
            "names: ['fish']\n"
            "kpt_shape: [6, 3]\n"
            "kpt_names: [nose, head, spine1, spine2, spine3, tail]\n"
            "flip_idx: [0, 1, 2, 3, 4, 5]\n"
        ),
        encoding="utf-8",
    )

    output_yaml = _rewrite_data_yaml_pose(tmp_path, data_yaml)

    assert output_yaml.exists()
    content = output_yaml.read_text(encoding="utf-8")

    assert "train: images/train" in content
    assert "val: images/val" in content
    assert "nc: 1" in content

    # path: must be absolute
    path_line = next(line for line in content.splitlines() if line.startswith("path:"))
    abs_path_str = path_line.split("path:", 1)[1].strip()
    assert Path(abs_path_str).is_absolute()

    # Pose-specific fields preserved
    assert "kpt_shape" in content
    assert "[6, 3]" in content
    assert "kpt_names" in content
    assert "nose" in content
    assert "flip_idx" in content


def test_rewrite_data_yaml_pose_without_kpt_fields(tmp_path: Path) -> None:
    """Pose data.yaml rewrite works even when optional kpt fields are absent."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "train: train.ndjson\nval: val.ndjson\nnc: 1\nnames: ['fish']\n",
        encoding="utf-8",
    )

    output_yaml = _rewrite_data_yaml_pose(tmp_path, data_yaml)

    content = output_yaml.read_text(encoding="utf-8")
    assert "train: images/train" in content
    assert "val: images/val" in content
    # Should not crash — kpt fields simply absent from output
    assert "kpt_shape" not in content


def test_empty_ndjson_produces_empty_labels_pose(tmp_path: Path) -> None:
    """An empty NDJSON file should produce no label files and not crash."""
    ndjson_path = tmp_path / "train.ndjson"
    ndjson_path.write_text("", encoding="utf-8")
    labels_dir = tmp_path / "labels" / "train"

    _convert_pose_ndjson_to_txt(ndjson_path, labels_dir)

    assert labels_dir.exists()
    assert list(labels_dir.glob("*.txt")) == []
