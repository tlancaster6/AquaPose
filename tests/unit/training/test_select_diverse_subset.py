"""Unit tests for diversity-maximizing subset selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aquapose.training.select_diverse_subset import (
    select_obb_subset,
    select_pose_subset,
)


def _make_obb_pseudo_dir(
    tmp_path: Path,
    entries: list[tuple[str, int, int]],
) -> Path:
    """Create a fake OBB pseudo-label directory.

    Args:
        tmp_path: Pytest tmp_path fixture.
        entries: List of (cam_id, frame_idx, fish_count) tuples.

    Returns:
        Path to the pseudo_dir root.
    """
    pseudo_dir = tmp_path / "pseudo"
    obb_dir = pseudo_dir / "obb"
    img_dir = obb_dir / "images" / "train"
    lbl_dir = obb_dir / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    confidence: dict[str, dict] = {}
    for cam_id, frame_idx, fish_count in entries:
        stem = f"{frame_idx:06d}_{cam_id}"
        (img_dir / f"{stem}.png").write_bytes(b"")
        (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1 0.0\n")
        confidence[stem] = {
            "tracked_fish_count": fish_count,
            "labels": [{"confidence": 0.9}] * fish_count,
        }

    (obb_dir / "confidence.json").write_text(json.dumps(confidence))
    return pseudo_dir


def _make_pose_pseudo_dir(
    tmp_path: Path,
    entries: list[tuple[str, int, int, float]],
) -> Path:
    """Create a fake pose pseudo-label directory.

    Args:
        tmp_path: Pytest tmp_path fixture.
        entries: List of (cam_id, frame_idx, fish_idx, curvature) tuples.

    Returns:
        Path to the pseudo_dir root.
    """
    pseudo_dir = tmp_path / "pseudo"
    pose_dir = pseudo_dir / "pose" / "consensus"
    img_dir = pose_dir / "images" / "train"
    lbl_dir = pose_dir / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Group entries by (frame_idx, cam_id) to build confidence.json
    confidence: dict[str, dict] = {}
    for cam_id, frame_idx, fish_idx, curvature in entries:
        conf_key = f"{frame_idx:06d}_{cam_id}"
        crop_stem = f"{frame_idx:06d}_{cam_id}_{fish_idx}"
        (img_dir / f"{crop_stem}.png").write_bytes(b"")
        (lbl_dir / f"{crop_stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        if conf_key not in confidence:
            confidence[conf_key] = {"labels": []}
        # Ensure labels list is long enough for fish_idx
        while len(confidence[conf_key]["labels"]) <= fish_idx:
            confidence[conf_key]["labels"].append({"curvature_2d": 0.0})
        confidence[conf_key]["labels"][fish_idx] = {"curvature_2d": curvature}

    (pose_dir / "confidence.json").write_text(json.dumps(confidence))
    return pseudo_dir


class TestObbSelection:
    """Tests for select_obb_subset."""

    def test_proportional_allocation_equal_cameras(self, tmp_path: Path) -> None:
        """3 cameras, 10 entries each, target=9 -> 3 per camera."""
        entries = []
        for cam in ["cam1", "cam2", "cam3"]:
            for frame in range(10):
                entries.append((cam, frame * 100, 3))

        pseudo_dir = _make_obb_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_obb_subset(
            pseudo_dir, output_dir, target_count=9, val_fraction=0.0
        )

        assert stats["total_selected"] == 9
        assert stats["per_camera"]["cam1"] == 3
        assert stats["per_camera"]["cam2"] == 3
        assert stats["per_camera"]["cam3"] == 3

    def test_proportional_allocation_unequal_cameras(self, tmp_path: Path) -> None:
        """Unequal camera counts -> proportional per-camera allocation."""
        entries = []
        for frame in range(20):
            entries.append(("cam1", frame * 100, 2))
        for frame in range(5):
            entries.append(("cam2", frame * 100, 2))
        for frame in range(5):
            entries.append(("cam3", frame * 100, 2))

        pseudo_dir = _make_obb_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_obb_subset(
            pseudo_dir, output_dir, target_count=9, val_fraction=0.0
        )

        # per_camera = 9 // 3 = 3, flex = 0
        # Each camera gets 3 (or fewer if not enough entries)
        assert stats["total_selected"] == 9
        total_from_cameras = sum(stats["per_camera"].values())
        assert total_from_cameras == 9

    def test_temporal_spread(self, tmp_path: Path) -> None:
        """Selected entries span the time range, not clustered at one end."""
        entries = []
        for frame in range(100):
            entries.append(("cam1", frame, 2))

        pseudo_dir = _make_obb_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_obb_subset(
            pseudo_dir, output_dir, target_count=10, val_fraction=0.0
        )

        assert stats["total_selected"] == 10

        # Read confidence.json to get selected stems
        selected_conf = json.loads((output_dir / "confidence.json").read_text())
        frames = sorted(int(stem[:6]) for stem in selected_conf)

        # Selections should span the range: first < 15, last > 85
        assert frames[0] < 15, f"First frame {frames[0]} should be near start"
        assert frames[-1] > 85, f"Last frame {frames[-1]} should be near end"

    def test_target_exceeds_available(self, tmp_path: Path) -> None:
        """target > available -> selects all available."""
        entries = [("cam1", i * 100, 2) for i in range(5)]

        pseudo_dir = _make_obb_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_obb_subset(
            pseudo_dir, output_dir, target_count=100, val_fraction=0.0
        )

        assert stats["total_selected"] == 5
        assert stats["total_available"] == 5

    def test_single_camera(self, tmp_path: Path) -> None:
        """Single camera -> all picks from that camera."""
        entries = [("cam1", i * 100, 2) for i in range(20)]

        pseudo_dir = _make_obb_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_obb_subset(
            pseudo_dir, output_dir, target_count=10, val_fraction=0.0
        )

        assert stats["total_selected"] == 10
        assert stats["per_camera"]["cam1"] == 10
        assert stats["n_cameras"] == 1

    def test_val_fraction_zero(self, tmp_path: Path) -> None:
        """val_fraction=0 -> no val split, all go to train."""
        entries = [("cam1", i * 100, 2) for i in range(10)]

        pseudo_dir = _make_obb_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_obb_subset(
            pseudo_dir, output_dir, target_count=5, val_fraction=0.0
        )

        assert stats["val_count"] == 0
        assert stats["train_count"] == 5
        assert not (output_dir / "images" / "val").exists() or not any(
            (output_dir / "images" / "val").iterdir()
        )

    def test_output_files_exist(self, tmp_path: Path) -> None:
        """Selected files are actually copied to output split directories."""
        entries = [("cam1", i * 100, 2) for i in range(10)]

        pseudo_dir = _make_obb_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_obb_subset(
            pseudo_dir, output_dir, target_count=6, val_fraction=0.2
        )

        train_imgs = list((output_dir / "images" / "train").iterdir())
        train_lbls = list((output_dir / "labels" / "train").iterdir())
        assert len(train_imgs) == stats["train_count"]
        assert len(train_lbls) == stats["train_count"]

        if stats["val_count"] > 0:
            val_imgs = list((output_dir / "images" / "val").iterdir())
            val_lbls = list((output_dir / "labels" / "val").iterdir())
            assert len(val_imgs) == stats["val_count"]
            assert len(val_lbls) == stats["val_count"]


class TestPoseSelection:
    """Tests for select_pose_subset."""

    def test_curvature_stratification(self, tmp_path: Path) -> None:
        """Known curvature values -> selections cover all quartile bins."""
        entries = []
        # Create entries with curvatures spanning 0-1 range across 4 quartiles
        for i in range(40):
            curvature = i / 39.0  # 0.0 to 1.0
            entries.append(("cam1", i * 100, 0, curvature))

        pseudo_dir = _make_pose_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_pose_subset(
            pseudo_dir, output_dir, target_count=20, val_fraction=0.0
        )

        assert stats["total_selected"] == 20
        # All 4 curvature bins should be represented
        assert len(stats["per_curvature_bin"]) == 4
        for bin_id in range(4):
            assert stats["per_curvature_bin"].get(bin_id, 0) > 0

    def test_camera_curvature_cross_product(self, tmp_path: Path) -> None:
        """Camera + curvature cross-product grouping works."""
        entries = []
        for cam in ["cam1", "cam2"]:
            for i in range(20):
                curvature = i / 19.0
                entries.append((cam, i * 100, 0, curvature))

        pseudo_dir = _make_pose_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_pose_subset(
            pseudo_dir, output_dir, target_count=16, val_fraction=0.0
        )

        assert stats["total_selected"] == 16
        # Both cameras should be represented
        assert "cam1" in stats["per_camera"]
        assert "cam2" in stats["per_camera"]

    def test_fewer_entries_than_target(self, tmp_path: Path) -> None:
        """Fewer entries than target -> selects all."""
        entries = [("cam1", i * 100, 0, 0.5) for i in range(5)]

        pseudo_dir = _make_pose_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_pose_subset(
            pseudo_dir, output_dir, target_count=100, val_fraction=0.0
        )

        assert stats["total_selected"] == 5
        assert stats["total_available"] == 5

    def test_val_fraction_splits_later_frames(self, tmp_path: Path) -> None:
        """Val split takes later frames."""
        entries = []
        for i in range(20):
            entries.append(("cam1", i * 100, 0, i / 19.0))

        pseudo_dir = _make_pose_pseudo_dir(tmp_path, entries)
        output_dir = tmp_path / "output"

        stats = select_pose_subset(
            pseudo_dir, output_dir, target_count=10, val_fraction=0.2
        )

        assert stats["val_count"] >= 1
        assert stats["train_count"] + stats["val_count"] == stats["total_selected"]

        # Val images should be later frames than train images
        train_imgs = sorted((output_dir / "images" / "train").iterdir())
        val_imgs = sorted((output_dir / "images" / "val").iterdir())
        if train_imgs and val_imgs:
            last_train_frame = int(train_imgs[-1].stem[:6])
            first_val_frame = int(val_imgs[0].stem[:6])
            assert first_val_frame >= last_train_frame

    def test_missing_confidence_raises(self, tmp_path: Path) -> None:
        """Missing confidence.json raises FileNotFoundError."""
        pseudo_dir = tmp_path / "pseudo"
        (pseudo_dir / "pose" / "consensus").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match=r"confidence\.json"):
            select_pose_subset(pseudo_dir, tmp_path / "output")

    def test_empty_entries(self, tmp_path: Path) -> None:
        """No crop files -> returns zero counts."""
        pseudo_dir = tmp_path / "pseudo"
        pose_dir = pseudo_dir / "pose" / "consensus"
        img_dir = pose_dir / "images" / "train"
        img_dir.mkdir(parents=True)
        (pose_dir / "confidence.json").write_text("{}")

        stats = select_pose_subset(pseudo_dir, tmp_path / "output")

        assert stats["total_selected"] == 0
        assert stats["total_available"] == 0
