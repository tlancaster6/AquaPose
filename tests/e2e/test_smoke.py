"""Pytest integration for the AquaPose pipeline smoke test.

Wraps SmokeTestRunner for CI integration. All tests are marked @slow and
@e2e and are excluded from the standard ``hatch run test`` fast test run.

Run with::

    hatch run test-all -k smoke
    hatch run test-all tests/e2e/test_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.smoke_test import SmokeTestRunner

# ---------------------------------------------------------------------------
# Known real-data paths (well-known defaults on the development machine)
# ---------------------------------------------------------------------------

_DEFAULT_CALIBRATION = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaCal/release_calibration/calibration.json"
)
_DEFAULT_VIDEO_DIR = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/videos/core_videos")
_DEFAULT_YOLO_WEIGHTS = Path("runs/detect/output/yolo_fish/train_v1/weights/best.pt")
_DEFAULT_UNET_WEIGHTS = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaPose/unet/run2/best_model.pth"
)

_CALIBRATION_PATH = _DEFAULT_CALIBRATION
"""AquaCal calibration JSON — required for synthetic mode."""

_PRODUCTION_CONFIG: Path | None = None
"""Full config path for production/diagnostic/benchmark tests.

Set this to a YAML config with real video_dir, calibration_path, and model
paths to enable non-synthetic smoke tests in this environment.
"""


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def calibration_path() -> Path:
    """Return the calibration path, skipping the session if unavailable.

    Returns:
        Path to AquaCal calibration JSON.
    """
    if not _CALIBRATION_PATH.exists():
        pytest.skip(f"Calibration file not found at {_CALIBRATION_PATH}")
    return _CALIBRATION_PATH


@pytest.fixture(scope="session")
def full_config_path() -> Path:
    """Return the full config path, skipping if unavailable.

    Returns:
        Path to a pipeline config YAML with real data paths.
    """
    if _PRODUCTION_CONFIG is None or not _PRODUCTION_CONFIG.exists():
        pytest.skip("No full pipeline config available — set _PRODUCTION_CONFIG")
    return _PRODUCTION_CONFIG


# ---------------------------------------------------------------------------
# Smoke test class
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.e2e
class TestSmoke:
    """Smoke tests for pipeline verification.

    Run with::

        hatch run test-all -k smoke

    Tests requiring real video data are marked @slow and skip when the
    required data is not available on the current machine.

    The synthetic mode test only requires the AquaCal calibration file,
    which is smaller and more portable than full video data.
    """

    def test_synthetic_mode(
        self,
        tmp_path: Path,
        calibration_path: Path,
    ) -> None:
        """Synthetic mode end-to-end test.

        Requires only the AquaCal calibration JSON — no video files,
        no YOLO weights, no U-Net weights.

        Args:
            tmp_path: Pytest temporary directory for artifacts.
            calibration_path: Session-scoped calibration file path fixture.
        """
        runner = SmokeTestRunner(
            output_base=tmp_path,
            calibration_path=calibration_path,
            frame_limit=5,
        )
        result = runner.run_single_mode("synthetic")
        assert result.passed, f"Synthetic mode failed: {result.error}"

    def test_benchmark_mode(
        self,
        tmp_path: Path,
        full_config_path: Path,
    ) -> None:
        """Benchmark mode smoke test with real data.

        Skips if full pipeline config is not configured on this machine.

        Args:
            tmp_path: Pytest temporary directory for artifacts.
            full_config_path: Session-scoped full pipeline config fixture.
        """
        runner = SmokeTestRunner(
            output_base=tmp_path,
            config_path=full_config_path,
            frame_limit=5,
        )
        result = runner.run_single_mode("benchmark")
        assert result.passed, f"Benchmark mode failed: {result.error}"

    def test_production_mode(
        self,
        tmp_path: Path,
        full_config_path: Path,
    ) -> None:
        """Production mode smoke test with real data.

        Skips if full pipeline config is not configured on this machine.

        Args:
            tmp_path: Pytest temporary directory for artifacts.
            full_config_path: Session-scoped full pipeline config fixture.
        """
        runner = SmokeTestRunner(
            output_base=tmp_path,
            config_path=full_config_path,
            frame_limit=5,
        )
        result = runner.run_single_mode("production")
        assert result.passed, f"Production mode failed: {result.error}"

    def test_diagnostic_mode(
        self,
        tmp_path: Path,
        full_config_path: Path,
    ) -> None:
        """Diagnostic mode smoke test with real data.

        Skips if full pipeline config is not configured on this machine.

        Args:
            tmp_path: Pytest temporary directory for artifacts.
            full_config_path: Session-scoped full pipeline config fixture.
        """
        runner = SmokeTestRunner(
            output_base=tmp_path,
            config_path=full_config_path,
            frame_limit=5,
        )
        result = runner.run_single_mode("diagnostic")
        assert result.passed, f"Diagnostic mode failed: {result.error}"

    def test_reproducibility(
        self,
        tmp_path: Path,
        full_config_path: Path,
    ) -> None:
        """Reproducibility test: run twice with identical config, compare outputs.

        Skips if full pipeline config is not configured on this machine.

        Args:
            tmp_path: Pytest temporary directory for artifacts.
            full_config_path: Session-scoped full pipeline config fixture.
        """
        runner = SmokeTestRunner(
            output_base=tmp_path,
            config_path=full_config_path,
            frame_limit=5,
        )
        result = runner.run_reproducibility_test()
        assert result.passed, f"Reproducibility check failed: {result.error}"
