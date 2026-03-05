"""Tests for the training run manager module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from aquapose.training.run_manager import (
    create_run_dir,
    extract_dataset_provenance,
    parse_best_metrics,
    print_next_steps,
    resolve_project_dir,
    snapshot_config,
    write_summary,
)


def _make_project_config(tmp_path: Path) -> Path:
    """Create a minimal project config YAML for testing."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump({"project_dir": str(project_dir)}), encoding="utf-8"
    )
    return config_path


def _make_results_csv(path: Path) -> None:
    """Write a sample Ultralytics results.csv with whitespace-padded headers."""
    headers = [
        "                   epoch",
        "                    time",
        "       metrics/precision(B)",
        "          metrics/recall(B)",
        "           metrics/mAP50(B)",
        "        metrics/mAP50-95(B)",
    ]
    rows = [
        ["0", "10.5", "0.5", "0.4", "0.6", "0.3"],
        ["1", "11.0", "0.7", "0.6", "0.8", "0.5"],
        ["2", "10.8", "0.65", "0.55", "0.75", "0.45"],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


class TestResolveProjectDir:
    """Tests for resolve_project_dir."""

    def test_extracts_project_dir(self, tmp_path: Path) -> None:
        config_path = _make_project_config(tmp_path)
        result = resolve_project_dir(config_path)
        assert result == (tmp_path / "project").resolve()

    def test_raises_on_missing_project_dir(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"other_key": "value"}), encoding="utf-8")
        import pytest

        with pytest.raises(ValueError, match="project_dir"):
            resolve_project_dir(config_path)


class TestCreateRunDir:
    """Tests for create_run_dir."""

    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        config_path = _make_project_config(tmp_path)
        run_dir = create_run_dir(config_path, "obb")
        assert run_dir.exists()
        assert run_dir.is_dir()
        # Should be under project_dir/training/obb/run_YYYYMMDD_HHMMSS
        assert run_dir.parent.parent.name == "training"
        assert run_dir.parent.name == "obb"
        assert run_dir.name.startswith("run_")

    def test_raises_on_missing_project_dir_key(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"other": "stuff"}), encoding="utf-8")
        import pytest

        with pytest.raises(ValueError, match="project_dir"):
            create_run_dir(config_path, "obb")


class TestSnapshotConfig:
    """Tests for snapshot_config."""

    def test_writes_config_yaml(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        cli_args = {"epochs": 100, "batch_size": 16, "model": "yolo26n-obb"}
        snapshot_config(run_dir, cli_args)
        config_file = run_dir / "config.yaml"
        assert config_file.exists()
        loaded = yaml.safe_load(config_file.read_text())
        assert loaded["epochs"] == 100
        assert loaded["model"] == "yolo26n-obb"

    def test_copies_dataset_sidecars(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "confidence.json").write_text(
            json.dumps({"threshold": 0.5}), encoding="utf-8"
        )
        (dataset_dir / "dataset.yaml").write_text(
            yaml.dump({"path": "/data"}), encoding="utf-8"
        )

        snapshot_config(run_dir, {"epochs": 50}, dataset_dir=dataset_dir)

        assert (run_dir / "dataset_confidence.json").exists()
        assert (run_dir / "dataset_dataset.yaml").exists()
        loaded = json.loads((run_dir / "dataset_confidence.json").read_text())
        assert loaded["threshold"] == 0.5

    def test_skips_missing_sidecars(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        # No confidence.json or dataset.yaml in dataset_dir
        snapshot_config(run_dir, {"epochs": 50}, dataset_dir=dataset_dir)
        assert not (run_dir / "dataset_confidence.json").exists()
        assert not (run_dir / "dataset_dataset.yaml").exists()


class TestParseBestMetrics:
    """Tests for parse_best_metrics."""

    def test_parses_best_epoch_by_map50_95(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        _make_results_csv(csv_path)
        result = parse_best_metrics(csv_path)
        assert result["best_epoch"] == 1  # epoch 1 has highest mAP50-95 (0.5)
        assert result["mAP50-95"] == 0.5
        assert result["mAP50"] == 0.8
        assert result["precision"] == 0.7
        assert result["recall"] == 0.6
        assert result["total_time"] > 0

    def test_returns_empty_dict_for_empty_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_path.write_text("epoch,time,metrics/mAP50-95(B)\n", encoding="utf-8")
        result = parse_best_metrics(csv_path)
        assert result == {}


class TestWriteSummary:
    """Tests for write_summary."""

    def test_produces_valid_json_with_expected_keys(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_20260305_143022"
        run_dir.mkdir()
        csv_path = run_dir / "results.csv"
        _make_results_csv(csv_path)

        training_args = {
            "epochs": 100,
            "batch_size": 16,
            "model": "yolo26n-obb",
            "data_dir": "/data/dataset",
            "weights": "/models/best.pt",
        }
        write_summary(
            run_dir,
            csv_path,
            training_args=training_args,
            model_type="obb",
            tag="test-run",
        )

        summary_path = run_dir / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["tag"] == "test-run"
        assert summary["model_type"] == "obb"
        assert summary["model_variant"] == "yolo26n-obb"
        assert summary["parent_weights"] == "/models/best.pt"
        assert summary["dataset_path"] == "/data/dataset"
        assert "metrics" in summary
        assert summary["metrics"]["mAP50-95"] == 0.5
        assert "run_id" in summary
        assert "created" in summary
        assert "training_config" in summary
        assert "training_duration_seconds" in summary


class TestExtractDatasetProvenance:
    """Tests for extract_dataset_provenance."""

    def test_reads_confidence_json(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "confidence.json").write_text(
            json.dumps(
                {
                    "consensus_threshold": 0.5,
                    "gap_threshold": 0.3,
                    "pipeline_run": "run_20260304_120854",
                }
            ),
            encoding="utf-8",
        )
        result = extract_dataset_provenance(dataset_dir)
        assert result["consensus_threshold"] == 0.5
        assert result["gap_threshold"] == 0.3
        assert result["pipeline_run"] == "run_20260304_120854"

    def test_reads_pseudo_val_metadata(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        images_dir = dataset_dir / "images" / "train"
        images_dir.mkdir(parents=True)
        (images_dir / "img1.jpg").touch()
        (images_dir / "img2.jpg").touch()
        (dataset_dir / "pseudo_val_metadata.json").write_text(
            json.dumps(
                {
                    "img1": {"source": "consensus"},
                    "img2": {"source": "gap"},
                }
            ),
            encoding="utf-8",
        )
        result = extract_dataset_provenance(dataset_dir)
        assert result["n_consensus"] == 1
        assert result["n_gap"] == 1
        assert result["n_train_images"] == 2

    def test_handles_missing_files(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        result = extract_dataset_provenance(dataset_dir)
        assert isinstance(result, dict)


class TestPrintNextSteps:
    """Tests for print_next_steps."""

    def test_does_not_raise(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "training" / "obb" / "run_20260305_143022"
        run_dir.mkdir(parents=True)
        best_weights = run_dir / "best_model.pt"
        best_weights.touch()
        # Just verify it doesn't raise
        print_next_steps(run_dir, "obb", best_weights)
