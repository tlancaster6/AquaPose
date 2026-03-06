"""Tests for elastic deformation library functions."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestParsePoseLabel:
    """Tests for parse_pose_label."""

    def test_parse_valid_label(self, tmp_path: Path) -> None:
        from aquapose.training.elastic_deform import parse_pose_label

        label_path = tmp_path / "test.txt"
        # cls cx cy w h x1 y1 v1 x2 y2 v2 ... (6 keypoints)
        parts = ["0", "0.5", "0.5", "0.8", "0.6"]
        for i in range(6):
            parts.extend([str(0.1 + i * 0.15), "0.5", "2"])
        label_path.write_text(" ".join(parts) + "\n")

        coords, visible = parse_pose_label(label_path, 100, 60)
        assert coords.shape == (6, 2)
        assert visible.shape == (6,)
        assert visible.all()
        # Coords should be denormalized to pixel space
        assert coords[0, 0] == pytest.approx(10.0, abs=0.1)  # 0.1 * 100
