"""Shared pytest fixtures for golden data loading and PipelineContext construction.

Fixtures are session-scoped for efficiency — golden data is read-only and
expensive to load from disk.

These fixtures load v1.0 pipeline outputs. The v1.0 pipeline had different
stage boundaries than the new 5-stage model, but the data itself remains the
golden reference for regression testing. See test_stage_harness.py module
docstring for the full mapping between v1.0 outputs and new stage boundaries.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
import torch

# Directory containing the golden fixture files (same as this conftest).
GOLDEN_DIR: Path = Path(__file__).parent

# Default numerical tolerance for golden data comparisons.
DEFAULT_ATOL: float = 1e-3

# Expected golden fixture filenames.
_GOLDEN_FILES: list[str] = [
    "metadata.pt",
    "golden_detection.pt",
    "golden_segmentation.pt.gz",
    "golden_tracking.pt",
    "golden_midline_extraction.pt",
    "golden_triangulation.pt",
]


def _check_golden_data_exists() -> bool:
    """Check whether all expected golden fixture files are present.

    Returns:
        True if every expected golden file exists in GOLDEN_DIR, False otherwise.
    """
    return all((GOLDEN_DIR / fname).exists() for fname in _GOLDEN_FILES)


@pytest.fixture(scope="session")
def golden_metadata() -> dict:
    """Load and return the golden metadata dict from metadata.pt.

    Skips the test session if the file is not present (golden data not yet
    generated).

    Returns:
        Metadata dict containing seed, versions, camera_ids, frame_count, etc.
    """
    path = GOLDEN_DIR / "metadata.pt"
    if not path.exists():
        pytest.skip(
            "Golden data not generated — run scripts/generate_golden_data.py first"
        )
    return torch.load(path, weights_only=False)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def golden_detections(golden_metadata: dict) -> list:
    """Load and return the golden detection data from golden_detection.pt.

    Args:
        golden_metadata: Session-scoped metadata fixture (ensures metadata
            loads first to provide a consistent skip guard).

    Returns:
        list[dict[str, list[Detection]]] — per-frame per-camera detections.
    """
    path = GOLDEN_DIR / "golden_detection.pt"
    if not path.exists():
        pytest.skip(
            "Golden detection data not found — run scripts/generate_golden_data.py first"
        )
    return torch.load(path, weights_only=False)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def golden_masks(golden_metadata: dict) -> list:
    """Load and return the golden segmentation data from golden_segmentation.pt.gz.

    The segmentation fixture is stored as a gzip-compressed .pt file because
    the raw numpy arrays are large.

    Args:
        golden_metadata: Session-scoped metadata fixture.

    Returns:
        list[dict[str, list[tuple[ndarray, CropRegion]]]] — per-frame masks.
    """
    path = GOLDEN_DIR / "golden_segmentation.pt.gz"
    if not path.exists():
        pytest.skip(
            "Golden segmentation data not found — run scripts/generate_golden_data.py first"
        )
    with gzip.open(path, "rb") as f:
        return torch.load(f, weights_only=False)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def golden_tracks(golden_metadata: dict) -> list:
    """Load and return the golden tracking data from golden_tracking.pt.

    Args:
        golden_metadata: Session-scoped metadata fixture.

    Returns:
        list[list[FishTrack]] — per-frame confirmed fish tracks.
    """
    path = GOLDEN_DIR / "golden_tracking.pt"
    if not path.exists():
        pytest.skip(
            "Golden tracking data not found — run scripts/generate_golden_data.py first"
        )
    return torch.load(path, weights_only=False)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def golden_midlines(golden_metadata: dict) -> list:
    """Load and return the golden midline extraction data from golden_midline_extraction.pt.

    Args:
        golden_metadata: Session-scoped metadata fixture.

    Returns:
        list[MidlineSet] — per-frame 2D midline sets (dict[int, dict[str, Midline2D]]).
    """
    path = GOLDEN_DIR / "golden_midline_extraction.pt"
    if not path.exists():
        pytest.skip(
            "Golden midline data not found — run scripts/generate_golden_data.py first"
        )
    return torch.load(path, weights_only=False)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def golden_triangulation(golden_metadata: dict) -> list:
    """Load and return the golden triangulation data from golden_triangulation.pt.

    Args:
        golden_metadata: Session-scoped metadata fixture.

    Returns:
        list[dict[int, Midline3D]] — per-frame 3D midline results.
    """
    path = GOLDEN_DIR / "golden_triangulation.pt"
    if not path.exists():
        pytest.skip(
            "Golden triangulation data not found — run scripts/generate_golden_data.py first"
        )
    return torch.load(path, weights_only=False)  # type: ignore[no-any-return]
