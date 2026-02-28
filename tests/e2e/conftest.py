"""Shared fixtures for AquaPose end-to-end tests.

Provides session-scoped path fixtures that skip when required data is not
present on the current machine, and a per-test output directory for saving
artifacts (reprojection videos, HDF5 outputs) for human review.

Test data layout (default ``~/aquapose/testing/``)::

    ~/aquapose/testing/
    ├── aquapose.yaml
    ├── geometry/
    │   └── calibration.json
    ├── models/
    │   ├── yolo.pt
    │   └── unet.pth
    └── videos/
        ├── e3v831e-*.mp4
        ├── e3v8334-*.mp4
        ├── e3v83eb-*.mp4
        └── e3v83f0-*.mp4

Override the root with ``AQUAPOSE_TEST_DATA`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Test data root — ~/aquapose/testing/ by default, override via env var
# ---------------------------------------------------------------------------

_TEST_DATA_ROOT = Path(
    os.environ.get("AQUAPOSE_TEST_DATA", Path.home() / "aquapose" / "testing")
)


# ---------------------------------------------------------------------------
# Session-scoped fixtures — skip entire session if data is unavailable
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_data_root() -> Path:
    """Return the test data root directory, skipping if not found."""
    if not _TEST_DATA_ROOT.exists():
        pytest.skip(
            f"Test data root not found: {_TEST_DATA_ROOT} "
            f"(set AQUAPOSE_TEST_DATA to override)"
        )
    return _TEST_DATA_ROOT


@pytest.fixture(scope="session")
def calibration_path(test_data_root: Path) -> Path:
    """Return the AquaCal calibration JSON path, skipping if not found."""
    path = test_data_root / "geometry" / "calibration.json"
    if not path.exists():
        pytest.skip(f"Calibration file not found: {path}")
    return path


@pytest.fixture(scope="session")
def test_video_dir(test_data_root: Path) -> Path:
    """Return the test video directory path, skipping if not found."""
    path = test_data_root / "videos"
    if not path.exists():
        pytest.skip(f"Test video directory not found: {path}")
    return path


@pytest.fixture(scope="session")
def yolo_weights(test_data_root: Path) -> Path:
    """Return YOLO weights path, skipping if not found."""
    path = test_data_root / "models" / "yolo.pt"
    if not path.exists():
        pytest.skip(f"YOLO weights not found: {path}")
    return path


@pytest.fixture(scope="session")
def unet_weights(test_data_root: Path) -> Path:
    """Return U-Net weights path, skipping if not found."""
    path = test_data_root / "models" / "unet.pth"
    if not path.exists():
        pytest.skip(f"U-Net weights not found: {path}")
    return path
