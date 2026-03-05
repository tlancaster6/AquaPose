"""Tests for dataset assembly module."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from aquapose.training.dataset_assembly import (
    assemble_dataset,
    collect_pseudo_labels,
    filter_by_confidence,
    filter_by_gap_reason,
    split_manual_val,
    split_pseudo_val,
)


def _make_obb_pseudo_label_dir(
    run_dir: Path,
    stems: list[str],
    confidence_map: dict[str, dict] | None = None,
) -> None:
    """Create a synthetic merged OBB pseudo-label directory.

    Args:
        run_dir: Root run directory.
        stems: List of image stems to create.
        confidence_map: Optional confidence.json content.
    """
    img_dir = run_dir / "pseudo_labels" / "obb" / "images" / "train"
    lbl_dir = run_dir / "pseudo_labels" / "obb" / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8")
        (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    conf_dir = run_dir / "pseudo_labels" / "obb"
    if confidence_map is None:
        confidence_map = {
            stem: {
                "labels": [
                    {
                        "fish_id": 1,
                        "confidence": 0.8,
                        "raw_metrics": {"mean_residual": 2.0},
                        "source": "consensus",
                    }
                ],
                "tracked_fish_count": 1,
                "complete": True,
            }
            for stem in stems
        }
    (conf_dir / "confidence.json").write_text(json.dumps(confidence_map))


def _make_pose_pseudo_label_dir(
    run_dir: Path,
    source: str,
    stems: list[str],
    confidence_map: dict[str, dict] | None = None,
) -> None:
    """Create a synthetic pose pseudo-label directory.

    Args:
        run_dir: Root run directory.
        source: "consensus" or "gap".
        stems: List of image stems to create.
        confidence_map: Optional confidence.json content.
    """
    img_dir = run_dir / "pseudo_labels" / "pose" / source / "images" / "train"
    lbl_dir = run_dir / "pseudo_labels" / "pose" / source / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8")
        (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    conf_dir = run_dir / "pseudo_labels" / "pose" / source
    if confidence_map is None:
        confidence_map = {
            stem: {
                "labels": [
                    {
                        "fish_id": 1,
                        "confidence": 0.8,
                        "raw_metrics": {"mean_residual": 2.0},
                    }
                ]
            }
            for stem in stems
        }
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "confidence.json").write_text(json.dumps(confidence_map))


def _make_manual_dir(
    manual_dir: Path,
    stems: list[str],
) -> None:
    """Create a synthetic manual YOLO annotation directory.

    Image stems should follow pattern: {frame}_{cam_id}
    """
    img_train = manual_dir / "images" / "train"
    lbl_train = manual_dir / "labels" / "train"
    img_train.mkdir(parents=True, exist_ok=True)
    lbl_train.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        (img_train / f"{stem}.jpg").write_bytes(b"\xff\xd8")
        (lbl_train / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    dataset_yaml = {
        "path": str(manual_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "fish"},
    }
    (manual_dir / "dataset.yaml").write_text(yaml.dump(dataset_yaml))


class TestCollectPseudoLabels:
    """Tests for collect_pseudo_labels."""

    def test_collects_from_single_run(self, tmp_path: Path) -> None:
        """Collects labels from a single run directory."""
        run_dir = tmp_path / "run_001"
        _make_obb_pseudo_label_dir(run_dir, ["000001_cam0", "000002_cam1"])

        result = collect_pseudo_labels([run_dir], "obb", "merged")

        assert len(result) == 2
        assert all(r["source"] == "merged" for r in result)
        assert all(r["run_id"] == "run_001" for r in result)

    def test_collects_from_multiple_runs(self, tmp_path: Path) -> None:
        """Collects and merges labels from multiple run directories."""
        run1 = tmp_path / "run_001"
        run2 = tmp_path / "run_002"
        _make_obb_pseudo_label_dir(run1, ["000001_cam0"])
        _make_obb_pseudo_label_dir(run2, ["000001_cam0"])

        result = collect_pseudo_labels([run1, run2], "obb", "merged")

        assert len(result) == 2
        run_ids = {r["run_id"] for r in result}
        assert run_ids == {"run_001", "run_002"}

    def test_computes_mean_confidence(self, tmp_path: Path) -> None:
        """Image-level confidence is the mean across fish."""
        run_dir = tmp_path / "run_001"
        conf = {
            "000001_cam0": {
                "labels": [
                    {
                        "fish_id": 1,
                        "confidence": 0.6,
                        "raw_metrics": {},
                        "source": "consensus",
                    },
                    {
                        "fish_id": 2,
                        "confidence": 0.8,
                        "raw_metrics": {},
                        "source": "consensus",
                    },
                ],
                "tracked_fish_count": 2,
                "complete": True,
            }
        }
        _make_obb_pseudo_label_dir(run_dir, ["000001_cam0"], conf)

        result = collect_pseudo_labels([run_dir], "obb", "merged")

        assert len(result) == 1
        assert abs(result[0]["confidence"] - 0.7) < 1e-6

    def test_handles_missing_confidence_json(self, tmp_path: Path) -> None:
        """Skips run directories without confidence.json gracefully."""
        run_dir = tmp_path / "run_001"
        _make_obb_pseudo_label_dir(run_dir, ["000001_cam0"])
        # Remove confidence.json
        (run_dir / "pseudo_labels" / "obb" / "confidence.json").unlink()

        result = collect_pseudo_labels([run_dir], "obb", "merged")

        assert len(result) == 0

    def test_returns_metadata_dict(self, tmp_path: Path) -> None:
        """Each result includes a metadata dict from confidence.json."""
        run_dir = tmp_path / "run_001"
        conf = {
            "000001_cam0": {
                "labels": [
                    {
                        "fish_id": 1,
                        "confidence": 0.9,
                        "raw_metrics": {"mean_residual": 1.5},
                        "source": "gap",
                        "gap_reason": "no-detection",
                    }
                ],
                "tracked_fish_count": 1,
                "complete": True,
            }
        }
        _make_obb_pseudo_label_dir(run_dir, ["000001_cam0"], conf)

        result = collect_pseudo_labels([run_dir], "obb", "merged")

        assert len(result) == 1
        assert "metadata" in result[0]
        assert result[0]["metadata"]["labels"][0]["gap_reason"] == "no-detection"

    def test_collects_pose_labels(self, tmp_path: Path) -> None:
        """Collects pose labels from pose/consensus directory."""
        run_dir = tmp_path / "run_001"
        _make_pose_pseudo_label_dir(
            run_dir, "consensus", ["000001_cam0", "000002_cam1"]
        )

        result = collect_pseudo_labels([run_dir], "pose/consensus", "consensus")

        assert len(result) == 2
        assert all(r["source"] == "consensus" for r in result)


class TestFilterByConfidence:
    """Tests for filter_by_confidence."""

    def test_filters_below_threshold(self) -> None:
        """Removes labels with confidence below min_confidence."""
        labels = [
            {"stem": "a", "confidence": 0.3},
            {"stem": "b", "confidence": 0.7},
            {"stem": "c", "confidence": 0.5},
        ]
        result = filter_by_confidence(labels, 0.5)
        stems = [r["stem"] for r in result]
        assert stems == ["b", "c"]

    def test_keeps_all_above_threshold(self) -> None:
        """Keeps all labels when all are above threshold."""
        labels = [{"stem": "a", "confidence": 0.9}, {"stem": "b", "confidence": 0.8}]
        result = filter_by_confidence(labels, 0.5)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        """Returns empty list for empty input."""
        assert filter_by_confidence([], 0.5) == []


class TestFilterByGapReason:
    """Tests for filter_by_gap_reason."""

    def test_excludes_matching_reasons(self) -> None:
        """Removes images where ALL fish match excluded reasons."""
        labels = [
            {
                "stem": "a",
                "metadata": {
                    "labels": [
                        {"gap_reason": "no-tracklet"},
                        {"gap_reason": "no-tracklet"},
                    ]
                },
            },
            {
                "stem": "b",
                "metadata": {
                    "labels": [
                        {"gap_reason": "no-detection"},
                        {"gap_reason": "no-tracklet"},
                    ]
                },
            },
        ]
        result = filter_by_gap_reason(labels, ["no-tracklet"])
        # "a" has ALL fish with no-tracklet -> excluded
        # "b" has one no-detection fish -> kept
        assert len(result) == 1
        assert result[0]["stem"] == "b"

    def test_keeps_all_when_no_exclusions(self) -> None:
        """Keeps all labels when exclude_reasons is empty."""
        labels = [
            {"stem": "a", "metadata": {"labels": [{"gap_reason": "no-tracklet"}]}},
        ]
        result = filter_by_gap_reason(labels, [])
        assert len(result) == 1

    def test_handles_missing_gap_reason(self) -> None:
        """Labels without gap_reason are never excluded."""
        labels = [
            {"stem": "a", "metadata": {"labels": [{"fish_id": 1}]}},
        ]
        result = filter_by_gap_reason(labels, ["no-tracklet"])
        assert len(result) == 1


class TestSplitManualVal:
    """Tests for split_manual_val."""

    def test_splits_per_camera_stratified(self, tmp_path: Path) -> None:
        """Splits manual data per-camera with correct fraction."""
        manual_dir = tmp_path / "manual"
        # 5 images from cam0, 5 from cam1
        stems = [f"{i:06d}_cam0" for i in range(5)] + [
            f"{i:06d}_cam1" for i in range(5)
        ]
        _make_manual_dir(manual_dir, stems)

        train_stems, val_stems = split_manual_val(manual_dir, val_fraction=0.4, seed=42)

        # With 5 per camera and 0.4 fraction, expect 2 val per camera = 4 total val
        assert len(val_stems) == 4
        assert len(train_stems) == 6
        # No overlap
        assert set(train_stems) & set(val_stems) == set()

    def test_deterministic_with_seed(self, tmp_path: Path) -> None:
        """Same seed produces same split."""
        manual_dir = tmp_path / "manual"
        stems = [f"{i:06d}_cam0" for i in range(10)]
        _make_manual_dir(manual_dir, stems)

        train1, val1 = split_manual_val(manual_dir, val_fraction=0.3, seed=42)
        train2, val2 = split_manual_val(manual_dir, val_fraction=0.3, seed=42)

        assert val1 == val2
        assert train1 == train2


class TestSplitPseudoVal:
    """Tests for split_pseudo_val."""

    def test_splits_correct_fraction(self) -> None:
        """Holds out approximately val_fraction of pseudo-labels."""
        labels = [{"stem": f"img_{i}"} for i in range(100)]
        train, val = split_pseudo_val(labels, val_fraction=0.1, seed=42)

        assert len(val) == 10
        assert len(train) == 90

    def test_deterministic(self) -> None:
        """Same seed produces same split."""
        labels = [{"stem": f"img_{i}"} for i in range(50)]
        _, val1 = split_pseudo_val(labels, val_fraction=0.2, seed=42)
        _, val2 = split_pseudo_val(labels, val_fraction=0.2, seed=42)
        assert [v["stem"] for v in val1] == [v["stem"] for v in val2]

    def test_empty_input(self) -> None:
        """Returns empty lists for empty input."""
        train, val = split_pseudo_val([], val_fraction=0.1, seed=42)
        assert train == []
        assert val == []


class TestAssembleDataset:
    """Tests for assemble_dataset end-to-end."""

    def test_full_assembly_obb(self, tmp_path: Path) -> None:
        """Assembles a complete YOLO OBB dataset from manual + pseudo-labels."""
        output_dir = tmp_path / "assembled"
        manual_dir = tmp_path / "manual"
        run_dir = tmp_path / "run_001"

        # Create manual data (4 images, 2 per camera)
        manual_stems = ["000001_cam0", "000002_cam0", "000001_cam1", "000002_cam1"]
        _make_manual_dir(manual_dir, manual_stems)

        # Create merged OBB pseudo-labels
        pseudo_stems = ["000010_cam0", "000011_cam0", "000012_cam1"]
        _make_obb_pseudo_label_dir(run_dir, pseudo_stems)

        result = assemble_dataset(
            output_dir=output_dir,
            manual_dir=manual_dir,
            run_dirs=[run_dir],
            model_type="obb",
            consensus_threshold=0.5,
            gap_threshold=0.3,
            exclude_gap_reasons=[],
            manual_val_fraction=0.5,
            pseudo_val_fraction=0.0,
            seed=42,
        )

        # Check output structure
        assert (output_dir / "images" / "train").is_dir()
        assert (output_dir / "images" / "val").is_dir()
        assert (output_dir / "labels" / "train").is_dir()
        assert (output_dir / "labels" / "val").is_dir()
        assert (output_dir / "dataset.yaml").exists()

        ds = yaml.safe_load((output_dir / "dataset.yaml").read_text())
        assert ds["val"] == "images/val"

        assert "manual_train" in result
        assert "manual_val" in result
        assert "pseudo_train" in result

    def test_full_assembly_pose(self, tmp_path: Path) -> None:
        """Assembles a complete YOLO pose dataset from pseudo-labels."""
        output_dir = tmp_path / "assembled"
        run_dir = tmp_path / "run_001"

        pseudo_stems = ["000010_cam0_000", "000011_cam0_000"]
        _make_pose_pseudo_label_dir(run_dir, "consensus", pseudo_stems)

        result = assemble_dataset(
            output_dir=output_dir,
            manual_dir=None,
            run_dirs=[run_dir],
            model_type="pose",
            consensus_threshold=0.5,
            gap_threshold=0.3,
            exclude_gap_reasons=[],
            manual_val_fraction=0.0,
            pseudo_val_fraction=0.0,
            seed=42,
        )

        assert result["consensus_train"] == 2

    def test_multi_run_collision_avoidance(self, tmp_path: Path) -> None:
        """Pseudo-label filenames are prefixed with run_id to avoid collisions."""
        output_dir = tmp_path / "assembled"
        run1 = tmp_path / "run_001"
        run2 = tmp_path / "run_002"

        _make_obb_pseudo_label_dir(run1, ["000001_cam0"])
        _make_obb_pseudo_label_dir(run2, ["000001_cam0"])

        assemble_dataset(
            output_dir=output_dir,
            manual_dir=None,
            run_dirs=[run1, run2],
            model_type="obb",
            consensus_threshold=0.0,
            gap_threshold=0.0,
            exclude_gap_reasons=[],
            manual_val_fraction=0.0,
            pseudo_val_fraction=0.0,
            seed=42,
        )

        train_images = list((output_dir / "images" / "train").glob("*.jpg"))
        assert len(train_images) == 2
        names = {img.stem for img in train_images}
        assert "run_001_000001_cam0" in names
        assert "run_002_000001_cam0" in names

    def test_manual_bypasses_confidence_filter(self, tmp_path: Path) -> None:
        """Manual annotations are always included regardless of thresholds."""
        output_dir = tmp_path / "assembled"
        manual_dir = tmp_path / "manual"
        _make_manual_dir(manual_dir, ["000001_cam0", "000002_cam0"])

        result = assemble_dataset(
            output_dir=output_dir,
            manual_dir=manual_dir,
            run_dirs=[],
            model_type="obb",
            consensus_threshold=0.99,
            gap_threshold=0.99,
            exclude_gap_reasons=[],
            manual_val_fraction=0.0,
            pseudo_val_fraction=0.0,
            seed=42,
        )

        assert result["manual_train"] == 2

    def test_pseudo_val_metadata_sidecar(self, tmp_path: Path) -> None:
        """Pseudo-label val metadata sidecar records source, confidence."""
        output_dir = tmp_path / "assembled"
        run_dir = tmp_path / "run_001"

        stems = [f"00000{i}_cam0" for i in range(10)]
        _make_obb_pseudo_label_dir(run_dir, stems)

        assemble_dataset(
            output_dir=output_dir,
            manual_dir=None,
            run_dirs=[run_dir],
            model_type="obb",
            consensus_threshold=0.0,
            gap_threshold=0.0,
            exclude_gap_reasons=[],
            manual_val_fraction=0.0,
            pseudo_val_fraction=0.2,
            seed=42,
        )

        sidecar_path = output_dir / "pseudo_val_metadata.json"
        assert sidecar_path.exists()
        sidecar = json.loads(sidecar_path.read_text())
        assert len(sidecar) == 2  # 20% of 10
        for entry in sidecar:
            assert "source" in entry
            assert "confidence" in entry
            assert "stem" in entry

    def test_obb_uses_min_threshold(self, tmp_path: Path) -> None:
        """OBB assembly uses min(consensus_threshold, gap_threshold)."""
        output_dir = tmp_path / "assembled"
        run_dir = tmp_path / "run_001"

        # Create OBB labels with confidence 0.4
        conf = {
            "000001_cam0": {
                "labels": [
                    {
                        "fish_id": 1,
                        "confidence": 0.4,
                        "raw_metrics": {},
                        "source": "consensus",
                    }
                ],
                "tracked_fish_count": 1,
                "complete": True,
            }
        }
        _make_obb_pseudo_label_dir(run_dir, ["000001_cam0"], conf)

        # consensus_threshold=0.5, gap_threshold=0.3 -> min=0.3
        result = assemble_dataset(
            output_dir=output_dir,
            manual_dir=None,
            run_dirs=[run_dir],
            model_type="obb",
            consensus_threshold=0.5,
            gap_threshold=0.3,
            exclude_gap_reasons=[],
            manual_val_fraction=0.0,
            pseudo_val_fraction=0.0,
            seed=42,
        )

        # 0.4 >= min(0.5, 0.3) = 0.3 -> passes
        assert result["pseudo_train"] == 1

    def test_pose_threshold_independent(self, tmp_path: Path) -> None:
        """Pose assembly applies independent thresholds to consensus/gap."""
        output_dir = tmp_path / "assembled"
        run_dir = tmp_path / "run_001"

        # Consensus with high confidence
        cons_conf = {
            "000001_cam0_000": {
                "labels": [{"fish_id": 1, "confidence": 0.9, "raw_metrics": {}}]
            }
        }
        _make_pose_pseudo_label_dir(
            run_dir, "consensus", ["000001_cam0_000"], cons_conf
        )

        # Gap with low confidence
        gap_conf = {
            "000002_cam0_000": {
                "labels": [
                    {
                        "fish_id": 1,
                        "confidence": 0.2,
                        "raw_metrics": {},
                        "gap_reason": "no-detection",
                    }
                ]
            }
        }
        _make_pose_pseudo_label_dir(run_dir, "gap", ["000002_cam0_000"], gap_conf)

        result = assemble_dataset(
            output_dir=output_dir,
            manual_dir=None,
            run_dirs=[run_dir],
            model_type="pose",
            consensus_threshold=0.5,
            gap_threshold=0.1,  # Low threshold -> gap passes
            exclude_gap_reasons=[],
            manual_val_fraction=0.0,
            pseudo_val_fraction=0.0,
            seed=42,
        )

        assert result["consensus_train"] == 1
        assert result["gap_train"] == 1

    def test_gap_reason_in_pseudo_val_metadata(self, tmp_path: Path) -> None:
        """Gap-source pseudo-labels include gap_reason in sidecar metadata."""
        output_dir = tmp_path / "assembled"
        run_dir = tmp_path / "run_001"

        gap_conf = {
            "000001_cam0_000": {
                "labels": [
                    {
                        "fish_id": 1,
                        "confidence": 0.8,
                        "raw_metrics": {},
                        "gap_reason": "no-detection",
                    },
                    {
                        "fish_id": 2,
                        "confidence": 0.7,
                        "raw_metrics": {},
                        "gap_reason": "no-detection",
                    },
                ]
            },
            "000002_cam0_000": {
                "labels": [
                    {
                        "fish_id": 1,
                        "confidence": 0.6,
                        "raw_metrics": {},
                        "gap_reason": "no-tracklet",
                    },
                ]
            },
        }
        _make_pose_pseudo_label_dir(
            run_dir, "gap", ["000001_cam0_000", "000002_cam0_000"], gap_conf
        )

        assemble_dataset(
            output_dir=output_dir,
            manual_dir=None,
            run_dirs=[run_dir],
            model_type="pose",
            consensus_threshold=0.0,
            gap_threshold=0.0,
            exclude_gap_reasons=[],
            manual_val_fraction=0.0,
            pseudo_val_fraction=1.0,  # All go to val for easy checking
            seed=42,
        )

        sidecar_path = output_dir / "pseudo_val_metadata.json"
        assert sidecar_path.exists()
        sidecar = json.loads(sidecar_path.read_text())
        assert len(sidecar) == 2

        by_stem = {}
        for e in sidecar:
            original_stem = e["stem"].removeprefix(f"{e['run_id']}_")
            by_stem[original_stem] = e
        assert by_stem["000001_cam0_000"]["gap_reason"] == "no-detection"
        assert by_stem["000002_cam0_000"]["gap_reason"] == "no-tracklet"

    def test_no_manual_dir(self, tmp_path: Path) -> None:
        """Assembly works without manual directory."""
        output_dir = tmp_path / "assembled"
        run_dir = tmp_path / "run_001"
        _make_obb_pseudo_label_dir(run_dir, ["000001_cam0"])

        result = assemble_dataset(
            output_dir=output_dir,
            manual_dir=None,
            run_dirs=[run_dir],
            model_type="obb",
            consensus_threshold=0.0,
            gap_threshold=0.0,
            exclude_gap_reasons=[],
            manual_val_fraction=0.0,
            pseudo_val_fraction=0.0,
            seed=42,
        )

        assert result["manual_train"] == 0
        assert result["manual_val"] == 0
