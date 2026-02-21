"""Unit tests for Midline3DWriter HDF5 serialization and round-trip correctness."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from aquapose.io import Midline3DWriter, read_midline3d_results
from aquapose.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
    Midline3D,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_midline3d(
    fish_id: int,
    frame_index: int = 0,
    n_cameras: int = 3,
    mean_residual: float = 1.5,
    max_residual: float = 3.0,
    is_low_confidence: bool = False,
) -> Midline3D:
    """Create a synthetic Midline3D for testing.

    Uses the module-level SPLINE_KNOTS, SPLINE_K, N_SAMPLE_POINTS constants.
    """
    rng = np.random.default_rng(seed=fish_id + frame_index * 100)
    control_points = rng.random((7, 3)).astype(np.float32)
    half_widths = rng.random(N_SAMPLE_POINTS).astype(np.float32) * 0.01
    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=control_points,
        knots=SPLINE_KNOTS.astype(np.float32),
        degree=SPLINE_K,
        arc_length=0.3 + fish_id * 0.01,
        half_widths=half_widths,
        n_cameras=n_cameras,
        mean_residual=mean_residual,
        max_residual=max_residual,
        is_low_confidence=is_low_confidence,
    )


# ---------------------------------------------------------------------------
# Test 1: round-trip correctness
# ---------------------------------------------------------------------------


def test_write_read_round_trip(tmp_path: Path) -> None:
    """Write 3 frames with 2 fish each, read back, verify all fields match."""
    out_path = tmp_path / "midlines.h5"
    max_fish = 4

    frames_data: list[dict[int, Midline3D]] = []
    for frame_i in range(3):
        frames_data.append(
            {
                j: _make_midline3d(
                    fish_id=j,
                    frame_index=frame_i,
                    n_cameras=4,
                    mean_residual=1.0 + j * 0.5,
                    max_residual=2.0 + j * 0.5,
                    is_low_confidence=(j == 1),
                )
                for j in range(2)
            }
        )

    with Midline3DWriter(out_path, max_fish=max_fish) as w:
        for frame_i, midlines in enumerate(frames_data):
            w.write_frame(frame_i, midlines)

    result = read_midline3d_results(out_path)

    # Shape checks
    assert result["frame_index"].shape == (3,)
    assert result["fish_id"].shape == (3, max_fish)
    assert result["control_points"].shape == (3, max_fish, 7, 3)
    assert result["arc_length"].shape == (3, max_fish)
    assert result["half_widths"].shape == (3, max_fish, N_SAMPLE_POINTS)
    assert result["n_cameras"].shape == (3, max_fish)
    assert result["mean_residual"].shape == (3, max_fish)
    assert result["max_residual"].shape == (3, max_fish)
    assert result["is_low_confidence"].shape == (3, max_fish)

    # Value checks for fish 0 across all frames
    for frame_i in range(3):
        expected = frames_data[frame_i][0]
        assert result["fish_id"][frame_i, 0] == 0
        np.testing.assert_allclose(
            result["control_points"][frame_i, 0],
            expected.control_points,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result["half_widths"][frame_i, 0],
            expected.half_widths,
            atol=1e-6,
        )
        assert result["n_cameras"][frame_i, 0] == 4
        assert abs(result["mean_residual"][frame_i, 0] - expected.mean_residual) < 1e-5
        assert abs(result["max_residual"][frame_i, 0] - expected.max_residual) < 1e-5
        assert bool(result["is_low_confidence"][frame_i, 0]) is False

    # is_low_confidence for fish 1 should be True
    for frame_i in range(3):
        assert bool(result["is_low_confidence"][frame_i, 1]) is True


# ---------------------------------------------------------------------------
# Test 2: chunk flush
# ---------------------------------------------------------------------------


def test_chunk_flush(tmp_path: Path) -> None:
    """Write chunk_frames+1 rows; verify first chunk is flushed to disk."""
    out_path = tmp_path / "midlines.h5"
    chunk_frames = 2

    writer = Midline3DWriter(out_path, max_fish=3, chunk_frames=chunk_frames)

    # After 2 frames (== chunk_frames), first chunk should be flushed
    for i in range(chunk_frames):
        writer.write_frame(i, {0: _make_midline3d(0, frame_index=i)})

    # Verify flush occurred by checking HDF5 file on disk
    with h5py.File(out_path, "r") as f:
        assert f["midlines"]["frame_index"].shape[0] == chunk_frames

    # Write one more frame (starts a new buffer)
    writer.write_frame(chunk_frames, {1: _make_midline3d(1, frame_index=chunk_frames)})
    writer.close()

    # All 3 frames should be present
    result = read_midline3d_results(out_path)
    assert result["frame_index"].shape[0] == chunk_frames + 1
    np.testing.assert_array_equal(
        result["frame_index"], np.arange(chunk_frames + 1, dtype=np.int64)
    )


# ---------------------------------------------------------------------------
# Test 3: context manager
# ---------------------------------------------------------------------------


def test_context_manager(tmp_path: Path) -> None:
    """Verify context manager protocol flushes and closes file."""
    out_path = tmp_path / "midlines.h5"

    with Midline3DWriter(out_path, max_fish=3) as w:
        w.write_frame(0, {0: _make_midline3d(0, frame_index=0)})
        w.write_frame(1, {0: _make_midline3d(0, frame_index=1)})

    # File is closed; should be readable after context exit
    result = read_midline3d_results(out_path)
    assert result["frame_index"].shape[0] == 2


# ---------------------------------------------------------------------------
# Test 4: fill-values for missing fish
# ---------------------------------------------------------------------------


def test_fillvalues_for_missing_fish(tmp_path: Path) -> None:
    """Write a frame with only 1 fish (max_fish=3); verify unfilled slots."""
    out_path = tmp_path / "midlines.h5"
    max_fish = 3

    with Midline3DWriter(out_path, max_fish=max_fish) as w:
        w.write_frame(0, {0: _make_midline3d(0, frame_index=0)})

    result = read_midline3d_results(out_path)

    # Slot 0 should be populated
    assert result["fish_id"][0, 0] == 0

    # Slots 1 and 2 should have fill-values
    for slot in range(1, max_fish):
        assert result["fish_id"][0, slot] == -1
        assert np.all(np.isnan(result["control_points"][0, slot]))
        assert np.all(np.isnan(result["half_widths"][0, slot]))
        assert result["n_cameras"][0, slot] == 0
        assert result["mean_residual"][0, slot] == pytest.approx(-1.0)
        assert result["max_residual"][0, slot] == pytest.approx(-1.0)
        assert bool(result["is_low_confidence"][0, slot]) is False


# ---------------------------------------------------------------------------
# Test 5: SPLINE_KNOTS and SPLINE_K stored as group attributes
# ---------------------------------------------------------------------------


def test_knots_and_degree_as_attrs(tmp_path: Path) -> None:
    """Verify SPLINE_KNOTS and SPLINE_K are group attributes, not datasets."""
    out_path = tmp_path / "midlines.h5"

    with Midline3DWriter(out_path, max_fish=2) as w:
        w.write_frame(0, {})

    with h5py.File(out_path, "r") as f:
        grp = f["midlines"]

        # Knots and degree must be attributes
        assert "SPLINE_KNOTS" in grp.attrs
        assert "SPLINE_K" in grp.attrs

        # Knots must NOT be a dataset
        assert "SPLINE_KNOTS" not in grp

        # Values match constants
        np.testing.assert_allclose(grp.attrs["SPLINE_KNOTS"], SPLINE_KNOTS)
        assert int(grp.attrs["SPLINE_K"]) == SPLINE_K


# ---------------------------------------------------------------------------
# Test 6: empty frame (no fish)
# ---------------------------------------------------------------------------


def test_empty_frame(tmp_path: Path) -> None:
    """Write a frame with empty dict; verify all slots have fill-values."""
    out_path = tmp_path / "midlines.h5"
    max_fish = 3

    with Midline3DWriter(out_path, max_fish=max_fish) as w:
        w.write_frame(42, {})

    result = read_midline3d_results(out_path)

    assert result["frame_index"][0] == 42
    # All fish slots should be fill-values
    assert np.all(result["fish_id"][0] == -1)
    assert np.all(np.isnan(result["control_points"][0]))
    assert np.all(np.isnan(result["half_widths"][0]))
    assert np.all(result["n_cameras"][0] == 0)
    assert np.all(result["mean_residual"][0] == pytest.approx(-1.0))
    assert np.all(result["max_residual"][0] == pytest.approx(-1.0))
    assert not np.any(result["is_low_confidence"][0])
