"""CLI integration tests for data import and convert commands."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from aquapose.cli import cli
from aquapose.training.store import SampleStore


@pytest.fixture
def runner() -> CliRunner:
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def yolo_dir(tmp_path: Path) -> Path:
    """Create a minimal YOLO-format directory with images and labels."""
    img_dir = tmp_path / "yolo" / "images" / "train"
    lbl_dir = tmp_path / "yolo" / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for i in range(3):
        # Create a small dummy image
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        img[10 + i, 20 + i] = 255  # unique pixel for unique hash
        cv2.imwrite(str(img_dir / f"sample_{i}.jpg"), img)

        # Create a matching OBB label
        lbl_dir.joinpath(f"sample_{i}.txt").write_text(
            "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        )

    return tmp_path / "yolo"


@pytest.fixture
def pose_yolo_dir(tmp_path: Path) -> Path:
    """Create a YOLO-format directory with pose labels (keypoints)."""
    img_dir = tmp_path / "pose_yolo" / "images" / "train"
    lbl_dir = tmp_path / "pose_yolo" / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create a single pose sample with 6 keypoints
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[30, 60] = 255
    cv2.imwrite(str(img_dir / "pose_sample.jpg"), img)

    # YOLO pose format: cls cx cy w h x1 y1 v1 x2 y2 v2 ... (6 keypoints)
    # Place keypoints along a horizontal line in normalized coords
    kps = []
    for k in range(6):
        x = 0.1 + k * 0.15
        y = 0.5
        v = 2  # visible
        kps.extend([x, y, v])
    kp_str = " ".join(str(v) for v in kps)
    lbl_dir.joinpath("pose_sample.txt").write_text(f"0 0.5 0.5 0.8 0.6 {kp_str}\n")

    return tmp_path / "pose_yolo"


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a minimal project directory with config.yaml."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yaml").write_text(f"project_dir: {project}\n")
    (project / "training_data" / "obb").mkdir(parents=True)
    (project / "training_data" / "pose").mkdir(parents=True)
    return project


@pytest.fixture
def monkeypatch_project(monkeypatch: pytest.MonkeyPatch, project_dir: Path) -> Path:
    """Patch resolve_project so --project test resolves to project_dir."""
    monkeypatch.setattr(
        "aquapose.cli_utils.resolve_project",
        lambda name: project_dir,
    )
    return project_dir


@pytest.fixture
def coco_json(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal COCO keypoints JSON and matching images directory."""
    images_dir = tmp_path / "coco_images"
    images_dir.mkdir()

    # Create 5 images with 1 fish annotation each
    images = []
    annotations = []
    for i in range(5):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[50 + i * 10, 100] = 255
        fname = f"img_{i:03d}.jpg"
        cv2.imwrite(str(images_dir / fname), img)

        images.append(
            {
                "id": i + 1,
                "file_name": fname,
                "width": 300,
                "height": 200,
            }
        )

        # 6 keypoints along a horizontal line
        kps = []
        for k in range(6):
            x = 50 + k * 30
            y = 100
            v = 2
            kps.extend([x, y, v])
        annotations.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "keypoints": kps,
                "num_keypoints": 6,
                "bbox": [30, 80, 180, 40],
                "area": 7200,
                "iscrowd": 0,
            }
        )

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "fish",
                "keypoints": [f"kp_{i}" for i in range(6)],
                "skeleton": [],
            }
        ],
    }

    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(json.dumps(coco))
    return coco_path, images_dir


class TestDataImport:
    """Tests for `aquapose data import` command."""

    def test_import_ingests_yolo_directory(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
        tmp_path: Path,
    ) -> None:
        """Import YOLO directory and verify samples appear in store."""
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "pseudo",
                "--input-dir",
                str(yolo_dir),
            ],
        )
        assert result.exit_code == 0, result.output

        # Verify samples in store
        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        assert store_db.exists()

        with SampleStore(store_db) as store:
            assert store.count() == 3

    def test_import_with_augment_creates_variants(
        self,
        runner: CliRunner,
        pose_yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import pose sample with --augment, verify augmented children exist."""
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "pose",
                "--source",
                "pseudo",
                "--input-dir",
                str(pose_yolo_dir),
                "--augment",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "augmented" in result.output.lower()

        store_db = monkeypatch_project / "training_data" / "pose" / "store.db"

        with SampleStore(store_db) as store:
            all_samples = store.query()
            # 1 original + 4 augmented variants
            assert len(all_samples) == 5
            # Verify parent_id linkage
            parents = [s for s in all_samples if s["parent_id"] is None]
            children = [s for s in all_samples if s["parent_id"] is not None]
            assert len(parents) == 1
            assert len(children) == 4

    def test_import_augment_skipped_for_obb(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import with --augment --store obb prints skip message."""
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "pseudo",
                "--input-dir",
                str(yolo_dir),
                "--augment",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (
            "skipping" in result.output.lower()
            or "only applies" in result.output.lower()
        )

        # No augmented children
        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            assert store.count() == 3  # only originals

    def test_import_reports_counts(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import shows imported/upserted/skipped counts."""
        # First import
        runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "pseudo",
                "--input-dir",
                str(yolo_dir),
            ],
        )

        # Second import (same files) -- should upsert or skip
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "manual",
                "--input-dir",
                str(yolo_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        # Should report upserted count
        assert "upserted" in result.output.lower()

    def test_import_with_batch_id(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import with --batch-id, verify import_batch_id in sample."""
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "pseudo",
                "--input-dir",
                str(yolo_dir),
                "--batch-id",
                "batch_001",
            ],
        )
        assert result.exit_code == 0, result.output

        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            assert all(s["import_batch_id"] == "batch_001" for s in samples)

    def test_import_with_metadata(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import with --metadata-json, verify metadata stored."""
        metadata = json.dumps({"confidence": 0.85, "run_id": "run_123"})
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "pseudo",
                "--input-dir",
                str(yolo_dir),
                "--metadata-json",
                metadata,
            ],
        )
        assert result.exit_code == 0, result.output

        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            for s in samples:
                meta = json.loads(s["metadata"])
                assert meta["confidence"] == 0.85
                assert meta["run_id"] == "run_123"

    def test_import_upsert_warns_about_deleted_augmentations(
        self,
        runner: CliRunner,
        pose_yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Re-importing after augment warns about cascade-deleted augmented variants."""
        # First import with augment
        runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "pose",
                "--source",
                "pseudo",
                "--input-dir",
                str(pose_yolo_dir),
                "--augment",
            ],
        )

        # Second import (higher priority) triggers upsert + cascade delete
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "pose",
                "--source",
                "manual",
                "--input-dir",
                str(pose_yolo_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (
            "cascade-deleted" in result.output.lower()
            or "augmented variants" in result.output.lower()
        )


class TestDataConvert:
    """Tests for `aquapose data convert` command."""

    def test_convert_coco_to_yolo_obb(
        self,
        runner: CliRunner,
        coco_json: tuple[Path, Path],
        tmp_path: Path,
    ) -> None:
        """Convert COCO to YOLO-OBB format."""
        coco_path, images_dir = coco_json
        output_dir = tmp_path / "yolo_output"

        result = runner.invoke(
            cli,
            [
                "data",
                "convert",
                "--coco-file",
                str(coco_path),
                "--images-dir",
                str(images_dir),
                "--output-dir",
                str(output_dir),
                "--type",
                "obb",
            ],
        )
        assert result.exit_code == 0, result.output

        obb_dir = output_dir / "obb"
        assert (obb_dir / "images" / "train").exists()
        assert (obb_dir / "dataset.yaml").exists()

    def test_convert_coco_to_yolo_pose(
        self,
        runner: CliRunner,
        coco_json: tuple[Path, Path],
        tmp_path: Path,
    ) -> None:
        """Convert COCO to YOLO-Pose format."""
        coco_path, images_dir = coco_json
        output_dir = tmp_path / "yolo_output"

        result = runner.invoke(
            cli,
            [
                "data",
                "convert",
                "--coco-file",
                str(coco_path),
                "--images-dir",
                str(images_dir),
                "--output-dir",
                str(output_dir),
                "--type",
                "pose",
            ],
        )
        assert result.exit_code == 0, result.output

        pose_dir = output_dir / "pose"
        assert (pose_dir / "images" / "train").exists()
        assert (pose_dir / "dataset.yaml").exists()


def _import_samples(
    runner: CliRunner,
    yolo_dir: Path,
    store: str,
    source: str,
) -> None:
    """Helper to import samples into a store via CLI."""
    runner.invoke(
        cli,
        [
            "--project",
            "test",
            "data",
            "import",
            "--store",
            store,
            "--source",
            source,
            "--input-dir",
            str(yolo_dir),
        ],
    )


@pytest.fixture
def yolo_dir_large(tmp_path: Path) -> Path:
    """Create a YOLO directory with 10 distinct samples for assembly tests."""
    img_dir = tmp_path / "yolo_large" / "images" / "train"
    lbl_dir = tmp_path / "yolo_large" / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for i in range(10):
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        img[i, i] = 255
        cv2.imwrite(str(img_dir / f"s_{i}.jpg"), img)
        lbl_dir.joinpath(f"s_{i}.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")

    return tmp_path / "yolo_large"


@pytest.fixture
def yolo_dir_pseudo(tmp_path: Path) -> Path:
    """Create a YOLO dir with pseudo-label samples (unique from yolo_dir_large)."""
    img_dir = tmp_path / "yolo_pseudo" / "images" / "train"
    lbl_dir = tmp_path / "yolo_pseudo" / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for i in range(5):
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        img[50 + i, 50 + i] = 128
        cv2.imwrite(str(img_dir / f"p_{i}.jpg"), img)
        lbl_dir.joinpath(f"p_{i}.txt").write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

    return tmp_path / "yolo_pseudo"


class TestDataAssemble:
    """Tests for `aquapose data assemble` command."""

    def test_assemble_creates_dataset_directory(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import samples, run assemble, verify YOLO directory with symlinks exists."""
        _import_samples(runner, yolo_dir, "obb", "manual")

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "assemble",
                "--store",
                "obb",
                "--name",
                "test_ds",
            ],
        )
        assert result.exit_code == 0, result.output

        ds_dir = monkeypatch_project / "training_data" / "obb" / "datasets" / "test_ds"
        assert ds_dir.exists()
        assert (ds_dir / "images" / "train").is_dir()
        assert (ds_dir / "dataset.yaml").exists()

    def test_assemble_with_source_filter(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        yolo_dir_large: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Assemble with --source pseudo, verify only pseudo samples included."""
        _import_samples(runner, yolo_dir, "obb", "manual")
        _import_samples(runner, yolo_dir_large, "obb", "pseudo")

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "assemble",
                "--store",
                "obb",
                "--name",
                "pseudo_only",
                "--source",
                "pseudo",
                "--val-fraction",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output

        ds_dir = (
            monkeypatch_project / "training_data" / "obb" / "datasets" / "pseudo_only"
        )
        train_imgs = list((ds_dir / "images" / "train").iterdir())
        assert len(train_imgs) == 10  # only pseudo samples

    def test_assemble_with_min_confidence(
        self,
        runner: CliRunner,
        monkeypatch_project: Path,
        tmp_path: Path,
    ) -> None:
        """Assemble with --min-confidence 0.5, verify low-confidence excluded."""
        # Import with metadata
        img_dir = tmp_path / "conf_yolo" / "images" / "train"
        lbl_dir = tmp_path / "conf_yolo" / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(4):
            img = np.zeros((64, 128, 3), dtype=np.uint8)
            img[i * 5, i * 5] = 200
            cv2.imwrite(str(img_dir / f"c_{i}.jpg"), img)
            lbl_dir.joinpath(f"c_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        # Import with high confidence
        runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "pseudo",
                "--input-dir",
                str(tmp_path / "conf_yolo"),
                "--metadata-json",
                json.dumps({"confidence": 0.9}),
            ],
        )

        # Import low-confidence samples with different content
        img_dir2 = tmp_path / "low_yolo" / "images" / "train"
        lbl_dir2 = tmp_path / "low_yolo" / "labels" / "train"
        img_dir2.mkdir(parents=True)
        lbl_dir2.mkdir(parents=True)

        for i in range(2):
            img = np.zeros((64, 128, 3), dtype=np.uint8)
            img[30 + i, 30 + i] = 50
            cv2.imwrite(str(img_dir2 / f"low_{i}.jpg"), img)
            lbl_dir2.joinpath(f"low_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "pseudo",
                "--input-dir",
                str(tmp_path / "low_yolo"),
                "--metadata-json",
                json.dumps({"confidence": 0.2}),
            ],
        )

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "assemble",
                "--store",
                "obb",
                "--name",
                "conf_ds",
                "--min-confidence",
                "0.5",
                "--val-fraction",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output

        ds_dir = monkeypatch_project / "training_data" / "obb" / "datasets" / "conf_ds"
        train_imgs = list((ds_dir / "images" / "train").iterdir())
        assert len(train_imgs) == 4  # only high-confidence


class TestDataStatus:
    """Tests for `aquapose data status` command."""

    def test_status_shows_both_stores(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import into both stores, verify status output."""
        _import_samples(runner, yolo_dir, "obb", "manual")
        _import_samples(runner, yolo_dir, "pose", "pseudo")

        result = runner.invoke(
            cli,
            ["--project", "test", "data", "status"],
        )
        assert result.exit_code == 0, result.output
        assert "obb" in result.output.lower()
        assert "pose" in result.output.lower()
        assert "3" in result.output  # 3 samples in each

    def test_status_handles_missing_store(
        self,
        runner: CliRunner,
        monkeypatch_project: Path,
    ) -> None:
        """Run status when no store exists, verify graceful handling."""
        result = runner.invoke(
            cli,
            ["--project", "test", "data", "status"],
        )
        assert result.exit_code == 0, result.output
        assert "no data" in result.output.lower()


class TestDataList:
    """Tests for `aquapose data list` command."""

    def test_list_shows_summary(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import samples, run list, verify output contains source counts."""
        _import_samples(runner, yolo_dir, "obb", "manual")

        result = runner.invoke(
            cli,
            ["--project", "test", "data", "list", "--store", "obb"],
        )
        assert result.exit_code == 0, result.output
        assert "manual" in result.output.lower()
        assert "3" in result.output


class TestDataExcludeInclude:
    """Tests for `aquapose data exclude` and `aquapose data include` commands."""

    def test_exclude_soft_deletes_samples(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import, run exclude with sample IDs, verify excluded tag added."""
        _import_samples(runner, yolo_dir, "obb", "manual")

        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            sid = samples[0]["id"]

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "exclude",
                "--store",
                "obb",
                "--ids",
                sid,
            ],
        )
        assert result.exit_code == 0, result.output

        with SampleStore(store_db) as store:
            row = store.get(sid)
            tags = json.loads(row["tags"])
            assert "excluded" in tags

    def test_include_reverses_exclude(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Exclude then include, verify excluded tag removed."""
        _import_samples(runner, yolo_dir, "obb", "manual")

        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            sid = samples[0]["id"]

        runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "exclude",
                "--store",
                "obb",
                "--ids",
                sid,
            ],
        )

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "include",
                "--store",
                "obb",
                "--ids",
                sid,
            ],
        )
        assert result.exit_code == 0, result.output

        with SampleStore(store_db) as store:
            row = store.get(sid)
            tags = json.loads(row["tags"])
            assert "excluded" not in tags

    def test_exclude_by_source(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Run exclude with --source filter instead of explicit IDs."""
        _import_samples(runner, yolo_dir, "obb", "manual")

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "exclude",
                "--store",
                "obb",
                "--source",
                "manual",
            ],
        )
        assert result.exit_code == 0, result.output

        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            # All should be excluded
            visible = store.query()
            assert len(visible) == 0


class TestDataRemove:
    """Tests for `aquapose data remove` command."""

    def test_remove_purge_hard_deletes(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Import, run remove --purge, verify files and DB rows gone."""
        _import_samples(runner, yolo_dir, "obb", "manual")

        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            sid = samples[0]["id"]

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "remove",
                "--store",
                "obb",
                "--ids",
                sid,
                "--purge",
            ],
        )
        assert result.exit_code == 0, result.output

        with SampleStore(store_db) as store:
            assert store.get(sid) is None
            assert store.count() == 2  # 3 - 1


class TestExcludeWithReasonCli:
    """Tests for exclude --reason CLI option."""

    def test_exclude_cmd_with_reason(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Exclude with --reason adds both 'excluded' and reason tags."""
        _import_samples(runner, yolo_dir, "obb", "manual")

        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            sid = samples[0]["id"]

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "exclude",
                "--store",
                "obb",
                "--ids",
                sid,
                "--reason",
                "bad_crop",
            ],
        )
        assert result.exit_code == 0, result.output

        with SampleStore(store_db) as store:
            row = store.get(sid)
            tags = json.loads(row["tags"])
            assert "excluded" in tags
            assert "bad_crop" in tags

    def test_status_shows_reason_breakdown(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        monkeypatch_project: Path,
    ) -> None:
        """Status command shows exclusion breakdown by reason."""
        _import_samples(runner, yolo_dir, "obb", "manual")

        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            # Exclude 2 with different reasons
            store.exclude([samples[0]["id"]], reason="bad_crop")
            store.exclude([samples[1]["id"]], reason="occluded")

        result = runner.invoke(
            cli,
            ["--project", "test", "data", "status"],
        )
        assert result.exit_code == 0, result.output
        assert "bad_crop" in result.output
        assert "occluded" in result.output


class TestParseFrameIndex:
    """Tests for parse_frame_index and temporal_split functions."""

    def test_parse_frame_index_valid(self) -> None:
        """Parse frame index from standard AquaPose filename."""
        from aquapose.training.coco_convert import parse_frame_index

        assert parse_frame_index("e3v82e0-20241005T130000_657000.png") == 657000

    def test_parse_frame_index_valid_jpg(self) -> None:
        """Parse frame index from .jpg filename."""
        from aquapose.training.coco_convert import parse_frame_index

        assert parse_frame_index("camera1_42.jpg") == 42

    def test_parse_frame_index_invalid(self) -> None:
        """Raise ValueError for filename without underscore-separated integer."""
        from aquapose.training.coco_convert import parse_frame_index

        with pytest.raises(ValueError, match="Cannot parse frame index"):
            parse_frame_index("no_number_here.png")

    def test_parse_frame_index_no_underscore(self) -> None:
        """Raise ValueError for filename with no underscore."""
        from aquapose.training.coco_convert import parse_frame_index

        with pytest.raises(ValueError, match="Cannot parse frame index"):
            parse_frame_index("filename.png")

    def test_temporal_split_groups_by_frame(self) -> None:
        """Temporal split keeps all cameras from same frame in same split."""
        from aquapose.training.coco_convert import temporal_split

        # 3 cameras x 5 frames = 15 images
        image_lookup = {}
        image_ids = []
        for frame_idx in range(5):
            for cam in range(3):
                img_id = frame_idx * 3 + cam + 1
                image_lookup[img_id] = {"file_name": f"cam{cam}_{frame_idx}.jpg"}
                image_ids.append(img_id)

        train_ids, val_ids = temporal_split(image_ids, image_lookup, val_fraction=0.2)

        # val should be last 1 frame (20% of 5 = 1), so 3 images
        assert len(val_ids) == 3
        assert len(train_ids) == 12

        # All val images should have the same frame index (the last one = 4)
        val_frames = {
            int(image_lookup[i]["file_name"].rsplit("_", 1)[-1].split(".")[0])
            for i in val_ids
        }
        assert val_frames == {4}


class TestConvertTemporalSplit:
    """Tests for convert command with --split-mode temporal."""

    def test_convert_temporal_split(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Convert with --split-mode temporal splits by frame index."""
        images_dir = tmp_path / "temporal_images"
        images_dir.mkdir()

        # Create 10 images: 2 cameras x 5 frames
        images = []
        annotations = []
        for frame_idx in range(5):
            for cam_idx in range(2):
                img_id = frame_idx * 2 + cam_idx + 1
                fname = f"cam{cam_idx}_{frame_idx}.jpg"
                img = np.zeros((200, 300, 3), dtype=np.uint8)
                img[50 + frame_idx * 10, 100 + cam_idx * 10] = 255
                cv2.imwrite(str(images_dir / fname), img)

                images.append(
                    {"id": img_id, "file_name": fname, "width": 300, "height": 200}
                )

                kps = []
                for k in range(6):
                    x = 50 + k * 30
                    y = 100
                    v = 2
                    kps.extend([x, y, v])
                annotations.append(
                    {
                        "id": img_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "keypoints": kps,
                        "num_keypoints": 6,
                        "bbox": [30, 80, 180, 40],
                        "area": 7200,
                        "iscrowd": 0,
                    }
                )

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 1, "name": "fish", "keypoints": [f"kp_{i}" for i in range(6)]}
            ],
        }
        coco_path = tmp_path / "temporal_coco.json"
        coco_path.write_text(json.dumps(coco))

        output_dir = tmp_path / "temporal_output"
        result = runner.invoke(
            cli,
            [
                "data",
                "convert",
                "--coco-file",
                str(coco_path),
                "--images-dir",
                str(images_dir),
                "--output-dir",
                str(output_dir),
                "--type",
                "obb",
                "--split-mode",
                "temporal",
            ],
        )
        assert result.exit_code == 0, result.output

        obb_dir = output_dir / "obb"
        val_imgs = list((obb_dir / "images" / "val").iterdir())
        train_imgs = list((obb_dir / "images" / "train").iterdir())
        # Last 1 frame (20% of 5) = 2 images in val
        assert len(val_imgs) == 2
        assert len(train_imgs) == 8


class TestImportValTagging:
    """Tests for val tagging on import."""

    def test_import_val_tagging(
        self,
        runner: CliRunner,
        monkeypatch_project: Path,
        tmp_path: Path,
    ) -> None:
        """Import from directory with val/ subdirectory tags samples with 'val'."""
        # Create YOLO dir with train/ and val/ subdirectories
        yolo_dir = tmp_path / "val_yolo"
        for split in ("train", "val"):
            img_dir = yolo_dir / "images" / split
            lbl_dir = yolo_dir / "labels" / split
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

        # 2 train images
        for i in range(2):
            img = np.zeros((64, 128, 3), dtype=np.uint8)
            img[i, i] = 255
            cv2.imwrite(str(yolo_dir / "images" / "train" / f"train_{i}.jpg"), img)
            (yolo_dir / "labels" / "train" / f"train_{i}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n"
            )

        # 1 val image
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        img[10, 10] = 128
        cv2.imwrite(str(yolo_dir / "images" / "val" / "val_0.jpg"), img)
        (yolo_dir / "labels" / "val" / "val_0.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "manual",
                "--input-dir",
                str(yolo_dir),
            ],
        )
        assert result.exit_code == 0, result.output

        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            all_samples = store.query(exclude_excluded=False)
            val_tagged = [s for s in all_samples if "val" in json.loads(s["tags"])]
            non_val = [s for s in all_samples if "val" not in json.loads(s["tags"])]
            assert len(val_tagged) == 1
            assert len(non_val) == 2


class TestAssembleSplitMode:
    """Tests for assemble command with --split-mode and --val-candidates."""

    def test_assemble_tagged_split_cli(
        self,
        runner: CliRunner,
        monkeypatch_project: Path,
        tmp_path: Path,
    ) -> None:
        """Assemble with --split-mode tagged uses val tags for splitting."""
        # Create and import samples with val/ structure
        yolo_dir = tmp_path / "tagged_yolo"
        for split in ("train", "val"):
            (yolo_dir / "images" / split).mkdir(parents=True)
            (yolo_dir / "labels" / split).mkdir(parents=True)

        for i in range(3):
            img = np.zeros((64, 128, 3), dtype=np.uint8)
            img[i, i] = 255
            cv2.imwrite(str(yolo_dir / "images" / "train" / f"t_{i}.jpg"), img)
            (yolo_dir / "labels" / "train" / f"t_{i}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n"
            )

        img = np.zeros((64, 128, 3), dtype=np.uint8)
        img[20, 20] = 128
        cv2.imwrite(str(yolo_dir / "images" / "val" / "v_0.jpg"), img)
        (yolo_dir / "labels" / "val" / "v_0.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        # Import (val/ samples get "val" tag)
        runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "manual",
                "--input-dir",
                str(yolo_dir),
            ],
        )

        # Assemble with tagged split
        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "assemble",
                "--store",
                "obb",
                "--name",
                "tagged_ds",
                "--split-mode",
                "tagged",
            ],
        )
        assert result.exit_code == 0, result.output

        ds_dir = (
            monkeypatch_project / "training_data" / "obb" / "datasets" / "tagged_ds"
        )
        val_imgs = list((ds_dir / "images" / "val").iterdir())
        train_imgs = list((ds_dir / "images" / "train").iterdir())
        assert len(val_imgs) == 1
        assert len(train_imgs) == 3

    def test_assemble_val_candidates_cli(
        self,
        runner: CliRunner,
        monkeypatch_project: Path,
        tmp_path: Path,
    ) -> None:
        """Assemble with --val-candidates filters val-eligible samples."""
        yolo_dir = tmp_path / "cand_yolo"
        (yolo_dir / "images" / "train").mkdir(parents=True)
        (yolo_dir / "labels" / "train").mkdir(parents=True)

        for i in range(6):
            img = np.zeros((64, 128, 3), dtype=np.uint8)
            img[i * 2, i * 2] = 200
            cv2.imwrite(str(yolo_dir / "images" / "train" / f"c_{i}.jpg"), img)
            (yolo_dir / "labels" / "train" / f"c_{i}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n"
            )

        runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "import",
                "--store",
                "obb",
                "--source",
                "manual",
                "--input-dir",
                str(yolo_dir),
            ],
        )

        # Tag 2 samples as "curated"
        store_db = monkeypatch_project / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            conn = store._connect()
            for s in samples[:2]:
                conn.execute(
                    "UPDATE samples SET tags = ? WHERE id = ?",
                    (json.dumps(["curated"]), s["id"]),
                )
            conn.commit()

        result = runner.invoke(
            cli,
            [
                "--project",
                "test",
                "data",
                "assemble",
                "--store",
                "obb",
                "--name",
                "cand_ds",
                "--val-candidates",
                "curated",
                "--val-fraction",
                "0.5",
            ],
        )
        assert result.exit_code == 0, result.output

        ds_dir = monkeypatch_project / "training_data" / "obb" / "datasets" / "cand_ds"
        val_imgs = list((ds_dir / "images" / "val").iterdir())
        train_imgs = list((ds_dir / "images" / "train").iterdir())
        # val = 50% of 2 curated = 1
        assert len(val_imgs) == 1
        # train = remaining 1 curated + 4 non-curated = 5
        assert len(train_imgs) == 5


class TestDataGroup:
    """Tests for the data CLI group registration."""

    def test_data_group_registered(self, runner: CliRunner) -> None:
        """Verify `aquapose data --help` shows import and convert."""
        result = runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0, result.output
        assert "import" in result.output
        assert "convert" in result.output
