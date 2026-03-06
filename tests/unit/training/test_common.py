"""Tests for training/common.py shared utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from aquapose.training.common import EarlyStopping, MetricsLogger, save_best_and_last

# ---------------------------------------------------------------------------
# EarlyStopping tests
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Tests for EarlyStopping utility class."""

    def test_max_mode_improves(self) -> None:
        """Higher metric value registers as improvement in max mode."""
        es = EarlyStopping(patience=3, mode="max")
        assert not es.step(0.5)
        assert es.best == 0.5

    def test_max_mode_no_improvement(self) -> None:
        """No improvement increments counter; returns True after patience."""
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.8)  # First step — improves from -inf
        assert not es.step(0.7)  # Worse — count = 1
        assert es.step(0.6)  # Worse — count = 2, should stop

    def test_min_mode_improves(self) -> None:
        """Lower metric value registers as improvement in min mode."""
        es = EarlyStopping(patience=3, mode="min")
        assert not es.step(0.9)
        assert es.best == 0.9
        assert not es.step(0.5)
        assert es.best == 0.5

    def test_min_mode_patience_exceeded(self) -> None:
        """Returns True once patience is exhausted in min mode."""
        es = EarlyStopping(patience=1, mode="min")
        es.step(0.5)  # Improvement — count stays 0
        # patience=1: one epoch without improvement triggers stop
        assert es.step(0.6)  # No improvement — count = 1 >= patience=1 → stop

    def test_patience_zero_never_stops(self) -> None:
        """Patience=0 disables early stopping (never returns True)."""
        es = EarlyStopping(patience=0, mode="max")
        for _ in range(100):
            assert not es.step(0.0)

    def test_best_tracks_maximum(self) -> None:
        """best property always holds the maximum metric seen in max mode."""
        es = EarlyStopping(patience=10, mode="max")
        es.step(0.3)
        es.step(0.9)
        es.step(0.5)
        assert es.best == 0.9

    def test_invalid_mode_raises(self) -> None:
        """Constructor raises ValueError for invalid mode."""
        import pytest

        with pytest.raises(ValueError, match="mode"):
            EarlyStopping(patience=5, mode="invalid")


# ---------------------------------------------------------------------------
# MetricsLogger tests
# ---------------------------------------------------------------------------


class TestMetricsLogger:
    """Tests for MetricsLogger utility class."""

    def test_creates_csv_with_header(self, tmp_path: Path) -> None:
        """Constructor creates metrics.csv with correct header row."""
        fields = ["train_loss", "val_iou"]
        MetricsLogger(tmp_path, fields)
        csv_path = tmp_path / "metrics.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "epoch" in content
        assert "train_loss" in content
        assert "val_iou" in content

    def test_log_appends_row(self, tmp_path: Path) -> None:
        """log() writes a row to the CSV file."""
        logger = MetricsLogger(tmp_path, ["train_loss", "val_iou"])
        logger.log(1, train_loss=0.8, val_iou=0.3)
        content = (tmp_path / "metrics.csv").read_text()
        assert "1" in content
        assert "0.8" in content or "0.800" in content

    def test_log_multiple_rows(self, tmp_path: Path) -> None:
        """log() appends one row per call without overwriting."""
        logger = MetricsLogger(tmp_path, ["loss"])
        logger.log(1, loss=1.0)
        logger.log(2, loss=0.9)
        logger.log(3, loss=0.8)
        lines = (tmp_path / "metrics.csv").read_text().strip().splitlines()
        # header + 3 rows
        assert len(lines) == 4

    def test_log_console_format(self, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """log() prints epoch summary with expected format to stdout."""
        logger = MetricsLogger(tmp_path, ["val_iou"])
        logger.set_total_epochs(10)
        logger.log(1, val_iou=0.5)
        captured = capsys.readouterr()
        assert "Epoch 1/10" in captured.out
        assert "val_iou" in captured.out
        assert "time:" in captured.out


# ---------------------------------------------------------------------------
# save_best_and_last tests
# ---------------------------------------------------------------------------


class TestSaveBestAndLast:
    """Tests for save_best_and_last checkpoint helper."""

    def _make_model(self) -> nn.Module:
        """Build a tiny linear model for testing."""
        return nn.Linear(2, 1)

    def test_always_saves_last(self, tmp_path: Path) -> None:
        """last_model.pth is always written."""
        model = self._make_model()
        save_best_and_last(model, tmp_path, 0.5, 0.0, "val_iou")
        assert (tmp_path / "last_model.pth").exists()

    def test_saves_best_when_metric_improves(self, tmp_path: Path) -> None:
        """best_model.pth is written when metric is higher than best_metric."""
        model = self._make_model()
        best_path, new_best = save_best_and_last(model, tmp_path, 0.7, 0.0, "val_iou")
        assert best_path.exists()
        assert new_best == 0.7

    def test_does_not_save_best_when_no_improvement(self, tmp_path: Path) -> None:
        """best_metric is returned unchanged when metric does not improve."""
        model = self._make_model()
        # First call establishes best
        save_best_and_last(model, tmp_path, 0.9, 0.0, "val_iou")
        # Second call — worse metric
        _, new_best = save_best_and_last(model, tmp_path, 0.5, 0.9, "val_iou")
        assert new_best == 0.9

    def test_loss_metric_minimizes(self, tmp_path: Path) -> None:
        """For loss metrics, lower values are treated as better."""
        model = self._make_model()
        _, new_best = save_best_and_last(model, tmp_path, 0.3, 0.5, "train_loss")
        assert new_best == 0.3  # Lower loss is improvement

    def test_loss_no_improvement(self, tmp_path: Path) -> None:
        """For loss metrics, higher values are not treated as improvement."""
        model = self._make_model()
        _, new_best = save_best_and_last(model, tmp_path, 0.8, 0.5, "train_loss")
        assert new_best == 0.5  # Higher loss is worse; best unchanged

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """output_dir is created if it does not exist."""
        model = self._make_model()
        nested = tmp_path / "a" / "b" / "c"
        save_best_and_last(model, nested, 0.5, 0.0, "val_iou")
        assert nested.exists()

    def test_saved_weights_loadable(self, tmp_path: Path) -> None:
        """Saved state_dict can be loaded back into a model."""
        model = self._make_model()
        # Set known weights
        with torch.no_grad():
            model.weight.fill_(3.14)
        save_best_and_last(model, tmp_path, 0.5, 0.0, "val_iou")

        new_model = self._make_model()
        state = torch.load(
            tmp_path / "best_model.pth", map_location="cpu", weights_only=True
        )
        new_model.load_state_dict(state)
        assert float(new_model.weight[0, 0]) == pytest_approx(3.14)


def pytest_approx(val: float, rel: float = 1e-5) -> object:
    """Thin wrapper so the helper function stays importable without pytest."""
    import pytest

    return pytest.approx(val, rel=rel)
