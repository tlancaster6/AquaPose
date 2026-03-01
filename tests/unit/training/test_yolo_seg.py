"""Unit tests for YOLO-seg NDJSON-to-YOLO.txt conversion and data.yaml rewrite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aquapose.training.yolo_seg import (
    _convert_seg_ndjson_to_txt,
    _rewrite_data_yaml_seg,
)


def _write_ndjson(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as NDJSON to path."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_convert_seg_ndjson_to_txt(tmp_path: Path) -> None:
    """NDJSON-to-YOLO.txt seg conversion creates one .txt per image with correct format."""
    ndjson_path = tmp_path / "train.ndjson"
    records = [
        {
            "image": "images/train/fish_001.jpg",
            "width": 128,
            "height": 64,
            "annotations": [
                {
                    "class_id": 0,
                    "polygon": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.2]],
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
                    "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                },
                {
                    "class_id": 0,
                    "polygon": [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4]],
                },
            ],
        },
    ]
    _write_ndjson(ndjson_path, records)
    labels_dir = tmp_path / "labels" / "train"

    _convert_seg_ndjson_to_txt(ndjson_path, labels_dir)

    # One .txt file per image
    assert (labels_dir / "fish_001.txt").exists()
    assert (labels_dir / "fish_002.txt").exists()

    # fish_001: 1 annotation, 3 polygon points → 1 + 2*3 = 7 values
    lines_001 = (
        (labels_dir / "fish_001.txt").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines_001) == 1
    parts = lines_001[0].split()
    assert parts[0] == "0"  # class_id
    assert len(parts) == 1 + 2 * 3  # class_id + 2*N_vertices

    # fish_002: 2 annotations
    lines_002 = (
        (labels_dir / "fish_002.txt").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines_002) == 2

    # First annotation: 4 polygon points → 1 + 2*4 = 9 values
    parts2 = lines_002[0].split()
    assert parts2[0] == "0"
    assert len(parts2) == 1 + 2 * 4

    # Second annotation: 3 polygon points → 1 + 2*3 = 7 values
    parts3 = lines_002[1].split()
    assert len(parts3) == 1 + 2 * 3

    # All coordinate values should be valid floats in [0, 1]
    for part in parts[1:]:
        val = float(part)
        assert 0.0 <= val <= 1.0


def test_rewrite_data_yaml_seg(tmp_path: Path) -> None:
    """Rewrite data.yaml produces absolute path and images/ train/val entries."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "train: train.ndjson\nval: val.ndjson\nnc: 1\nnames: ['fish']\n",
        encoding="utf-8",
    )

    output_yaml = _rewrite_data_yaml_seg(tmp_path, data_yaml)

    assert output_yaml.exists()
    content = output_yaml.read_text(encoding="utf-8")

    assert "train: images/train" in content
    assert "val: images/val" in content
    assert "nc: 1" in content

    # path: must be absolute
    path_line = next(line for line in content.splitlines() if line.startswith("path:"))
    abs_path_str = path_line.split("path:", 1)[1].strip()
    assert Path(abs_path_str).is_absolute()


def test_empty_ndjson_produces_empty_labels(tmp_path: Path) -> None:
    """An empty NDJSON file should produce no label files and not crash."""
    ndjson_path = tmp_path / "train.ndjson"
    ndjson_path.write_text("", encoding="utf-8")
    labels_dir = tmp_path / "labels" / "train"

    _convert_seg_ndjson_to_txt(ndjson_path, labels_dir)

    # labels_dir is created but has no .txt files
    assert labels_dir.exists()
    assert list(labels_dir.glob("*.txt")) == []


@pytest.mark.parametrize(
    "polygon",
    [
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.2]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ],
)
def test_convert_seg_polygon_coord_count(tmp_path: Path, polygon: list) -> None:
    """Each label line has exactly 1 + 2*N values (class_id + flat polygon coords)."""
    ndjson_path = tmp_path / "train.ndjson"
    records = [
        {
            "image": "images/train/img_001.jpg",
            "width": 64,
            "height": 64,
            "annotations": [{"class_id": 0, "polygon": polygon}],
        }
    ]
    _write_ndjson(ndjson_path, records)
    labels_dir = tmp_path / "labels" / "train"

    _convert_seg_ndjson_to_txt(ndjson_path, labels_dir)

    label_file = labels_dir / "img_001.txt"
    assert label_file.exists()
    parts = label_file.read_text(encoding="utf-8").strip().splitlines()[0].split()
    expected = 1 + 2 * len(polygon)
    assert len(parts) == expected
