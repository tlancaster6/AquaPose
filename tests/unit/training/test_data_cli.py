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
def project_config(tmp_path: Path) -> Path:
    """Create a minimal project config YAML."""
    import yaml

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config = {
        "project_dir": str(project_dir),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


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
        project_config: Path,
        tmp_path: Path,
    ) -> None:
        """Import YOLO directory and verify samples appear in store."""
        result = runner.invoke(
            cli,
            [
                "data",
                "import",
                "--config",
                str(project_config),
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
        project_dir = Path(
            __import__("yaml").safe_load(project_config.read_text())["project_dir"]
        )
        store_db = project_dir / "training_data" / "obb" / "store.db"
        assert store_db.exists()

        with SampleStore(store_db) as store:
            assert store.count() == 3

    def test_import_with_augment_creates_variants(
        self,
        runner: CliRunner,
        pose_yolo_dir: Path,
        project_config: Path,
    ) -> None:
        """Import pose sample with --augment, verify augmented children exist."""
        result = runner.invoke(
            cli,
            [
                "data",
                "import",
                "--config",
                str(project_config),
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

        project_dir = Path(
            __import__("yaml").safe_load(project_config.read_text())["project_dir"]
        )
        store_db = project_dir / "training_data" / "pose" / "store.db"

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
        project_config: Path,
    ) -> None:
        """Import with --augment --store obb prints skip message."""
        result = runner.invoke(
            cli,
            [
                "data",
                "import",
                "--config",
                str(project_config),
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
        project_dir = Path(
            __import__("yaml").safe_load(project_config.read_text())["project_dir"]
        )
        store_db = project_dir / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            assert store.count() == 3  # only originals

    def test_import_reports_counts(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        project_config: Path,
    ) -> None:
        """Import shows imported/upserted/skipped counts."""
        # First import
        runner.invoke(
            cli,
            [
                "data",
                "import",
                "--config",
                str(project_config),
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
                "data",
                "import",
                "--config",
                str(project_config),
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
        project_config: Path,
    ) -> None:
        """Import with --batch-id, verify import_batch_id in sample."""
        result = runner.invoke(
            cli,
            [
                "data",
                "import",
                "--config",
                str(project_config),
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

        project_dir = Path(
            __import__("yaml").safe_load(project_config.read_text())["project_dir"]
        )
        store_db = project_dir / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            assert all(s["import_batch_id"] == "batch_001" for s in samples)

    def test_import_with_metadata(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        project_config: Path,
    ) -> None:
        """Import with --metadata-json, verify metadata stored."""
        metadata = json.dumps({"confidence": 0.85, "run_id": "run_123"})
        result = runner.invoke(
            cli,
            [
                "data",
                "import",
                "--config",
                str(project_config),
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

        project_dir = Path(
            __import__("yaml").safe_load(project_config.read_text())["project_dir"]
        )
        store_db = project_dir / "training_data" / "obb" / "store.db"
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
        project_config: Path,
    ) -> None:
        """Re-importing after augment warns about cascade-deleted augmented variants."""
        # First import with augment
        runner.invoke(
            cli,
            [
                "data",
                "import",
                "--config",
                str(project_config),
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
                "data",
                "import",
                "--config",
                str(project_config),
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
    project_config: Path,
    yolo_dir: Path,
    store: str,
    source: str,
) -> None:
    """Helper to import samples into a store via CLI."""
    runner.invoke(
        cli,
        [
            "data",
            "import",
            "--config",
            str(project_config),
            "--store",
            store,
            "--source",
            source,
            "--input-dir",
            str(yolo_dir),
        ],
    )


def _get_project_dir(project_config: Path) -> Path:
    """Resolve project_dir from config yaml."""
    import yaml

    return Path(yaml.safe_load(project_config.read_text())["project_dir"])


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
        project_config: Path,
    ) -> None:
        """Import samples, run assemble, verify YOLO directory with symlinks exists."""
        _import_samples(runner, project_config, yolo_dir, "obb", "manual")

        result = runner.invoke(
            cli,
            [
                "data",
                "assemble",
                "--config",
                str(project_config),
                "--store",
                "obb",
                "--name",
                "test_ds",
            ],
        )
        assert result.exit_code == 0, result.output

        project_dir = _get_project_dir(project_config)
        ds_dir = project_dir / "training_data" / "obb" / "datasets" / "test_ds"
        assert ds_dir.exists()
        assert (ds_dir / "images" / "train").is_dir()
        assert (ds_dir / "dataset.yaml").exists()

    def test_assemble_with_source_filter(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        yolo_dir_large: Path,
        project_config: Path,
    ) -> None:
        """Assemble with --source pseudo, verify only pseudo samples included."""
        _import_samples(runner, project_config, yolo_dir, "obb", "manual")
        _import_samples(runner, project_config, yolo_dir_large, "obb", "pseudo")

        result = runner.invoke(
            cli,
            [
                "data",
                "assemble",
                "--config",
                str(project_config),
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

        project_dir = _get_project_dir(project_config)
        ds_dir = project_dir / "training_data" / "obb" / "datasets" / "pseudo_only"
        train_imgs = list((ds_dir / "images" / "train").iterdir())
        assert len(train_imgs) == 10  # only pseudo samples

    def test_assemble_with_min_confidence(
        self,
        runner: CliRunner,
        project_config: Path,
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
                "data",
                "import",
                "--config",
                str(project_config),
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
                "data",
                "import",
                "--config",
                str(project_config),
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
                "data",
                "assemble",
                "--config",
                str(project_config),
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

        project_dir = _get_project_dir(project_config)
        ds_dir = project_dir / "training_data" / "obb" / "datasets" / "conf_ds"
        train_imgs = list((ds_dir / "images" / "train").iterdir())
        assert len(train_imgs) == 4  # only high-confidence


class TestDataStatus:
    """Tests for `aquapose data status` command."""

    def test_status_shows_both_stores(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        project_config: Path,
    ) -> None:
        """Import into both stores, verify status output."""
        _import_samples(runner, project_config, yolo_dir, "obb", "manual")
        _import_samples(runner, project_config, yolo_dir, "pose", "pseudo")

        result = runner.invoke(
            cli,
            ["data", "status", "--config", str(project_config)],
        )
        assert result.exit_code == 0, result.output
        assert "obb" in result.output.lower()
        assert "pose" in result.output.lower()
        assert "3" in result.output  # 3 samples in each

    def test_status_handles_missing_store(
        self,
        runner: CliRunner,
        project_config: Path,
    ) -> None:
        """Run status when no store exists, verify graceful handling."""
        result = runner.invoke(
            cli,
            ["data", "status", "--config", str(project_config)],
        )
        assert result.exit_code == 0, result.output
        assert "no data" in result.output.lower()


class TestDataList:
    """Tests for `aquapose data list` command."""

    def test_list_shows_summary(
        self,
        runner: CliRunner,
        yolo_dir: Path,
        project_config: Path,
    ) -> None:
        """Import samples, run list, verify output contains source counts."""
        _import_samples(runner, project_config, yolo_dir, "obb", "manual")

        result = runner.invoke(
            cli,
            ["data", "list", "--config", str(project_config), "--store", "obb"],
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
        project_config: Path,
    ) -> None:
        """Import, run exclude with sample IDs, verify excluded tag added."""
        _import_samples(runner, project_config, yolo_dir, "obb", "manual")

        project_dir = _get_project_dir(project_config)
        store_db = project_dir / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            sid = samples[0]["id"]

        result = runner.invoke(
            cli,
            [
                "data",
                "exclude",
                "--config",
                str(project_config),
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
        project_config: Path,
    ) -> None:
        """Exclude then include, verify excluded tag removed."""
        _import_samples(runner, project_config, yolo_dir, "obb", "manual")

        project_dir = _get_project_dir(project_config)
        store_db = project_dir / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            sid = samples[0]["id"]

        runner.invoke(
            cli,
            [
                "data",
                "exclude",
                "--config",
                str(project_config),
                "--store",
                "obb",
                "--ids",
                sid,
            ],
        )

        result = runner.invoke(
            cli,
            [
                "data",
                "include",
                "--config",
                str(project_config),
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
        project_config: Path,
    ) -> None:
        """Run exclude with --source filter instead of explicit IDs."""
        _import_samples(runner, project_config, yolo_dir, "obb", "manual")

        result = runner.invoke(
            cli,
            [
                "data",
                "exclude",
                "--config",
                str(project_config),
                "--store",
                "obb",
                "--source",
                "manual",
            ],
        )
        assert result.exit_code == 0, result.output

        project_dir = _get_project_dir(project_config)
        store_db = project_dir / "training_data" / "obb" / "store.db"
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
        project_config: Path,
    ) -> None:
        """Import, run remove --purge, verify files and DB rows gone."""
        _import_samples(runner, project_config, yolo_dir, "obb", "manual")

        project_dir = _get_project_dir(project_config)
        store_db = project_dir / "training_data" / "obb" / "store.db"
        with SampleStore(store_db) as store:
            samples = store.query()
            sid = samples[0]["id"]

        result = runner.invoke(
            cli,
            [
                "data",
                "remove",
                "--config",
                str(project_config),
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


class TestDataGroup:
    """Tests for the data CLI group registration."""

    def test_data_group_registered(self, runner: CliRunner) -> None:
        """Verify `aquapose data --help` shows import and convert."""
        result = runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0, result.output
        assert "import" in result.output
        assert "convert" in result.output
