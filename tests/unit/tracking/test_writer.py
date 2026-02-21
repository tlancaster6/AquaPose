"""Unit tests for TrackingWriter HDF5 serialization and round-trip correctness."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from aquapose.tracking import (
    AssociationResult,
    FishTrack,
    FishTracker,
    FrameAssociations,
    TrackingWriter,
    read_tracking_results,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assoc(
    fish_id: int,
    centroid: tuple[float, float, float],
    n_cameras: int = 3,
    camera_detections: dict[str, int] | None = None,
    confidence: float = 1.0,
) -> AssociationResult:
    """Create a synthetic AssociationResult for testing."""
    return AssociationResult(
        fish_id=fish_id,
        centroid_3d=np.array(centroid, dtype=np.float32),
        reprojection_residual=0.5,
        camera_detections=camera_detections or {},
        n_cameras=n_cameras,
        confidence=confidence,
        is_low_confidence=False,
    )


def _make_track(
    fish_id: int,
    centroid: tuple[float, float, float],
    is_confirmed: bool = True,
    camera_detections: dict[str, int] | None = None,
    bboxes: dict[str, tuple[int, int, int, int]] | None = None,
    n_cameras: int = 3,
    confidence: float = 1.0,
    reprojection_residual: float = 0.5,
) -> FishTrack:
    """Create a synthetic FishTrack for testing."""
    track = FishTrack(fish_id=fish_id)
    track.positions = deque([np.array(centroid, dtype=np.float32)], maxlen=2)
    track.is_confirmed = is_confirmed
    track.camera_detections = camera_detections or {}
    track.bboxes = bboxes or {}
    track.n_cameras = n_cameras
    track.confidence = confidence
    track.reprojection_residual = reprojection_residual
    return track


CAMERAS = ["cam_a", "cam_b", "cam_c"]


# ---------------------------------------------------------------------------
# Test 1: round-trip correctness
# ---------------------------------------------------------------------------


def test_write_read_roundtrip(tmp_path: Path) -> None:
    """Write 5 frames with 3 tracks each; read back and verify all fields."""
    out_path = tmp_path / "tracking.h5"
    max_fish = 5

    tracks_per_frame: list[list[FishTrack]] = []
    for frame_i in range(5):
        tracks = [
            _make_track(
                fish_id=j,
                centroid=(float(frame_i + j), float(j), 0.5),
                is_confirmed=True,
                n_cameras=3,
                confidence=0.9,
                reprojection_residual=0.3,
            )
            for j in range(3)
        ]
        tracks_per_frame.append(tracks)

    with TrackingWriter(out_path, CAMERAS, max_fish=max_fish) as w:
        for frame_i, tracks in enumerate(tracks_per_frame):
            w.write_frame(frame_i, tracks)

    result = read_tracking_results(out_path)

    # Shape checks
    assert result["frame_index"].shape == (5,)
    assert result["fish_id"].shape == (5, max_fish)
    assert result["centroid_3d"].shape == (5, max_fish, 3)
    assert result["confidence"].shape == (5, max_fish)
    assert result["n_cameras"].shape == (5, max_fish)
    assert result["is_confirmed"].shape == (5, max_fish)

    # Value checks for slot 0 across all frames
    for frame_i in range(5):
        assert result["fish_id"][frame_i, 0] == 0
        np.testing.assert_allclose(
            result["centroid_3d"][frame_i, 0],
            [float(frame_i), 0.0, 0.5],
            atol=1e-5,
        )
        assert result["is_confirmed"][frame_i, 0]
        assert result["n_cameras"][frame_i, 0] == 3
        assert abs(result["confidence"][frame_i, 0] - 0.9) < 1e-5


# ---------------------------------------------------------------------------
# Test 2: chunked flush
# ---------------------------------------------------------------------------


def test_chunked_flush(tmp_path: Path) -> None:
    """Write 7 frames with chunk_frames=3; verify flush timing and final size."""
    out_path = tmp_path / "tracking.h5"
    chunk_frames = 3

    writer = TrackingWriter(out_path, CAMERAS, max_fish=3, chunk_frames=chunk_frames)

    # After 3 frames, first chunk should be flushed
    for i in range(3):
        writer.write_frame(i, [_make_track(0, (float(i), 0.0, 0.0))])

    # Check HDF5 dataset has grown to 3 after first flush
    import h5py

    with h5py.File(out_path, "r") as f:
        assert f["tracking"]["frame_index"].shape[0] == 3

    # Write 4 more frames
    for i in range(3, 7):
        writer.write_frame(i, [_make_track(0, (float(i), 0.0, 0.0))])

    writer.close()

    # All 7 frames should be present
    result = read_tracking_results(out_path)
    assert result["frame_index"].shape[0] == 7
    np.testing.assert_array_equal(result["frame_index"], np.arange(7, dtype=np.int64))


# ---------------------------------------------------------------------------
# Test 3: context manager
# ---------------------------------------------------------------------------


def test_context_manager(tmp_path: Path) -> None:
    """Verify context manager properly closes file and data is readable after."""
    out_path = tmp_path / "tracking.h5"

    with TrackingWriter(out_path, CAMERAS, max_fish=3) as w:
        w.write_frame(0, [_make_track(0, (1.0, 2.0, 3.0))])
        w.write_frame(1, [_make_track(0, (4.0, 5.0, 6.0))])

    # File should be readable after context exits
    result = read_tracking_results(out_path)
    assert result["frame_index"].shape[0] == 2
    assert result["fish_id"][0, 0] == 0
    assert result["fish_id"][1, 0] == 0


# ---------------------------------------------------------------------------
# Test 4: camera assignments and bboxes
# ---------------------------------------------------------------------------


def test_camera_assignments_and_bboxes(tmp_path: Path) -> None:
    """Write frames with camera_detections and bboxes; read back and verify."""
    out_path = tmp_path / "tracking.h5"
    cameras = ["cam_a", "cam_b"]

    track = _make_track(
        fish_id=0,
        centroid=(1.0, 2.0, 3.0),
        camera_detections={"cam_a": 2, "cam_b": 0},
        bboxes={"cam_a": (10, 20, 30, 40), "cam_b": (5, 15, 25, 35)},
        n_cameras=2,
    )

    with TrackingWriter(out_path, cameras, max_fish=3) as w:
        w.write_frame(0, [track])

    result = read_tracking_results(out_path)

    # Camera assignments for slot 0
    cam_assignments = result["camera_assignments"]
    assert cam_assignments["cam_a"][0, 0] == 2
    assert cam_assignments["cam_b"][0, 0] == 0

    # Absent fish (slots 1, 2) should be -1
    assert cam_assignments["cam_a"][0, 1] == -1
    assert cam_assignments["cam_b"][0, 2] == -1

    # Bboxes for slot 0
    bboxes = result["bboxes"]
    np.testing.assert_array_equal(bboxes["cam_a"][0, 0], [10, 20, 30, 40])
    np.testing.assert_array_equal(bboxes["cam_b"][0, 0], [5, 15, 25, 35])

    # Absent fish bbox slots should be -1
    np.testing.assert_array_equal(bboxes["cam_a"][0, 1], [-1, -1, -1, -1])


# ---------------------------------------------------------------------------
# Test 5: absent fish fill-values
# ---------------------------------------------------------------------------


def test_absent_fish_fillvalues(tmp_path: Path) -> None:
    """Write a frame with 2 of 9 fish; verify remaining 7 slots have fill-values."""
    out_path = tmp_path / "tracking.h5"
    max_fish = 9

    tracks = [
        _make_track(0, (0.0, 0.0, 0.5)),
        _make_track(1, (1.0, 0.0, 0.5)),
    ]

    with TrackingWriter(out_path, CAMERAS, max_fish=max_fish) as w:
        w.write_frame(0, tracks)

    result = read_tracking_results(out_path)

    # Slots 0 and 1 should be populated
    assert result["fish_id"][0, 0] == 0
    assert result["fish_id"][0, 1] == 1

    # Slots 2-8 should be fill-values
    for slot in range(2, max_fish):
        assert result["fish_id"][0, slot] == -1
        assert np.all(np.isnan(result["centroid_3d"][0, slot]))
        assert result["confidence"][0, slot] == pytest.approx(-1.0)
        assert result["n_cameras"][0, slot] == 0
        assert result["is_confirmed"][0, slot] is np.bool_(False)


# ---------------------------------------------------------------------------
# Test 6: camera_names attribute
# ---------------------------------------------------------------------------


def test_camera_names_attribute(tmp_path: Path) -> None:
    """Verify camera_names attribute is stored on /tracking group."""
    out_path = tmp_path / "tracking.h5"
    cameras = ["front", "side_left", "top"]

    with TrackingWriter(out_path, cameras) as w:
        w.write_frame(0, [])

    result = read_tracking_results(out_path)
    assert result["camera_names"] == cameras


# ---------------------------------------------------------------------------
# Test 7: empty file
# ---------------------------------------------------------------------------


def test_empty_file(tmp_path: Path) -> None:
    """Create writer and close immediately; file exists with empty datasets."""
    out_path = tmp_path / "tracking.h5"

    with TrackingWriter(out_path, CAMERAS, max_fish=3) as _w:
        pass  # write nothing

    assert out_path.exists()

    result = read_tracking_results(out_path)
    assert result["frame_index"].shape == (0,)
    assert result["fish_id"].shape == (0, 3)
    assert result["centroid_3d"].shape == (0, 3, 3)
    assert result["camera_names"] == CAMERAS


# ---------------------------------------------------------------------------
# Test 8: integration tracker -> writer
# ---------------------------------------------------------------------------


def test_integration_tracker_to_writer(tmp_path: Path) -> None:
    """Feed FishTracker output into TrackingWriter; verify HDF5 fish IDs match."""
    out_path = tmp_path / "tracking.h5"

    cameras = ["cam_a", "cam_b"]
    tracker = FishTracker(min_hits=1, max_age=3, expected_count=3)

    # Build 3 frames of FrameAssociations with 3 fish
    frame_assocs = []
    for frame_i in range(3):
        assocs = [
            _make_assoc(
                fish_id=j,
                centroid=(float(j), float(frame_i) * 0.1, 0.5),
                n_cameras=2,
                camera_detections={"cam_a": j, "cam_b": j},
            )
            for j in range(3)
        ]
        fa = FrameAssociations(associations=assocs, frame_index=frame_i)
        frame_assocs.append(fa)

    with TrackingWriter(out_path, cameras, max_fish=5) as w:
        for frame_i, fa in enumerate(frame_assocs):
            confirmed = tracker.update(fa)
            w.write_frame(frame_i, confirmed)

    result = read_tracking_results(out_path)

    # 3 frames written
    assert result["frame_index"].shape[0] == 3

    # After frame 0 with min_hits=1, confirmed tracks exist — check fish IDs
    # are non-negative for populated slots
    for frame_i in range(3):
        populated = result["fish_id"][frame_i][result["fish_id"][frame_i] >= 0]
        assert len(populated) > 0, f"No confirmed fish in frame {frame_i}"

    # Centroids for confirmed fish should be finite (non-NaN)
    for frame_i in range(3):
        for slot in range(5):
            if result["fish_id"][frame_i, slot] >= 0:
                assert np.all(np.isfinite(result["centroid_3d"][frame_i, slot])), (
                    f"NaN centroid for fish {result['fish_id'][frame_i, slot]}"
                    f" in frame {frame_i}"
                )
