"""Shared fixtures for AquaPose end-to-end tests.

Provides session-scoped path fixtures that skip when required data is not
present on the current machine, and a per-test output directory for saving
artifacts (reprojection videos, HDF5 outputs) for human review.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Well-known paths on the development machine
# ---------------------------------------------------------------------------

_DEFAULT_CALIBRATION = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaCal/release_calibration/calibration.json"
)
_DEFAULT_VIDEO_DIR = Path("C:/Users/tucke/Desktop/Aqua/Videos/test_videos")
_DEFAULT_YOLO_WEIGHTS = Path("runs/detect/output/yolo_fish/train_v1/weights/best.pt")
_DEFAULT_UNET_WEIGHTS = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaPose/unet/run2/best_model.pth"
)
_E2E_OUTPUT_DIR = Path("tests/e2e/output")


# ---------------------------------------------------------------------------
# Session-scoped fixtures — skip entire session if data is unavailable
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def calibration_path() -> Path:
    """Return the AquaCal calibration JSON path, skipping if not found.

    Returns:
        Path to the AquaCal calibration JSON file.
    """
    if not _DEFAULT_CALIBRATION.exists():
        pytest.skip(f"Calibration file not found: {_DEFAULT_CALIBRATION}")
    return _DEFAULT_CALIBRATION


@pytest.fixture(scope="session")
def test_video_dir() -> Path:
    """Return the test video directory path, skipping if not found.

    Returns:
        Path to the directory containing 4-camera test videos.
    """
    if not _DEFAULT_VIDEO_DIR.exists():
        pytest.skip(f"Test video directory not found: {_DEFAULT_VIDEO_DIR}")
    return _DEFAULT_VIDEO_DIR


@pytest.fixture(scope="session")
def yolo_weights() -> Path:
    """Return YOLO weights path, skipping if not found.

    Returns:
        Path to the YOLO fish detection model weights.
    """
    # Resolve relative to project root (where pytest is invoked from)
    weights_path = Path.cwd() / _DEFAULT_YOLO_WEIGHTS
    if not weights_path.exists():
        # Try absolute path directly
        if not _DEFAULT_YOLO_WEIGHTS.is_absolute():
            pytest.skip(f"YOLO weights not found: {_DEFAULT_YOLO_WEIGHTS}")
        weights_path = _DEFAULT_YOLO_WEIGHTS
    return weights_path


@pytest.fixture(scope="session")
def unet_weights() -> Path:
    """Return U-Net weights path, skipping if not found.

    Returns:
        Path to the U-Net segmentation model weights.
    """
    if not _DEFAULT_UNET_WEIGHTS.exists():
        pytest.skip(f"U-Net weights not found: {_DEFAULT_UNET_WEIGHTS}")
    return _DEFAULT_UNET_WEIGHTS


# ---------------------------------------------------------------------------
# Output directory fixture — for saving artifacts for human review
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def e2e_output_dir() -> Path:
    """Return the e2e test artifact output directory (creates if not exists).

    Artifacts saved here (reprojection videos, HDF5 outputs) persist after
    tests complete for human review. Contents are gitignored.

    Returns:
        Path to the tests/e2e/output/ directory.
    """
    output_dir = _E2E_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
