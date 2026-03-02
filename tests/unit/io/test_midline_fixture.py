"""Round-trip and validation tests for load_midline_fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aquapose.core.association.types import TrackletGroup
from aquapose.core.context import PipelineContext
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.core.tracking.types import Tracklet2D
from aquapose.core.types.detection import Detection
from aquapose.core.types.midline import Midline2D
from aquapose.engine.diagnostic_observer import DiagnosticObserver
from aquapose.engine.events import StageComplete
from aquapose.io.midline_fixture import (
    CalibBundle,
    MidlineFixture,
    load_midline_fixture,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_midline2d(
    fish_id: int,
    camera_id: str,
    frame_index: int,
    n_points: int = 5,
    confidence: float | None = 0.9,
) -> Midline2D:
    """Create a synthetic Midline2D for test use."""
    points = np.arange(n_points * 2, dtype=np.float32).reshape(n_points, 2)
    half_widths = np.ones(n_points, dtype=np.float32) * 2.0
    point_confidence = (
        np.ones(n_points, dtype=np.float32) * confidence
        if confidence is not None
        else None
    )
    return Midline2D(
        points=points,
        half_widths=half_widths,
        fish_id=fish_id,
        camera_id=camera_id,
        frame_index=frame_index,
        is_head_to_tail=True,
        point_confidence=point_confidence,
    )


def _make_tracklet2d(
    camera_id: str,
    track_id: int,
    frames: tuple[int, ...],
) -> Tracklet2D:
    """Create a Tracklet2D with default centroid at each frame."""
    centroids = tuple((float(f) * 10.0, float(f) * 5.0) for f in frames)
    bboxes = tuple((0.0, 0.0, 10.0, 10.0) for _ in frames)
    status = tuple("detected" for _ in frames)
    return Tracklet2D(
        camera_id=camera_id,
        track_id=track_id,
        frames=frames,
        centroids=centroids,
        bboxes=bboxes,
        frame_status=status,
    )


def _make_annotated_detection(
    midline: Midline2D | None,
    centroid: tuple[float, float],
) -> AnnotatedDetection:
    """Build an AnnotatedDetection with a synthetic Detection."""
    cx, cy = centroid
    bbox = (int(cx - 5), int(cy - 5), 10, 10)
    det = Detection(bbox=bbox, mask=None, area=100, confidence=0.95)
    return AnnotatedDetection(detection=det, midline=midline)


def _fire_association_stage(
    observer: DiagnosticObserver, groups: list[TrackletGroup]
) -> None:
    """Simulate an AssociationStage StageComplete event."""
    ctx = PipelineContext()
    ctx.tracklet_groups = groups
    ctx.frame_count = 1
    observer.on_event(
        StageComplete(
            stage_name="AssociationStage",
            stage_index=2,
            elapsed_seconds=0.1,
            context=ctx,
        )
    )


def _fire_midline_stage(
    observer: DiagnosticObserver,
    annotated_detections: list,
    frame_count: int,
    camera_ids: list[str],
) -> None:
    """Simulate a MidlineStage StageComplete event."""
    ctx = PipelineContext()
    ctx.annotated_detections = annotated_detections
    ctx.frame_count = frame_count
    ctx.camera_ids = camera_ids
    observer.on_event(
        StageComplete(
            stage_name="MidlineStage",
            stage_index=3,
            elapsed_seconds=0.2,
            context=ctx,
        )
    )


def _write_fixture(tmp_path: Path, observer: DiagnosticObserver) -> Path:
    """Export midline fixture NPZ from observer, return path."""
    out_path = tmp_path / "midline_fixtures.npz"
    observer.export_midline_fixtures(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_round_trip_single_fish_single_camera(tmp_path: Path) -> None:
    """export_midline_fixtures -> load_midline_fixture round-trips correctly."""
    observer = DiagnosticObserver()
    fish_id = 0
    cam = "cam_a"
    frames = (0,)

    tracklet = _make_tracklet2d(cam, 0, frames)
    group = TrackletGroup(
        fish_id=fish_id,
        tracklets=(tracklet,),
        confidence=0.9,
        per_frame_confidence=(0.9,),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group])

    midline = _make_midline2d(fish_id, cam, 0)
    centroid = tracklet.centroids[0]
    frame0_annot = {cam: [_make_annotated_detection(midline, centroid)]}
    _fire_midline_stage(observer, [frame0_annot], 1, [cam])

    npz_path = _write_fixture(tmp_path, observer)
    fixture = load_midline_fixture(npz_path)

    assert isinstance(fixture, MidlineFixture)
    assert len(fixture.frames) == 1
    assert len(fixture.frame_indices) == 1
    assert fixture.frame_indices[0] == 0

    loaded_midline = fixture.frames[0][fish_id][cam]
    assert np.allclose(loaded_midline.points, midline.points, atol=1e-6)
    assert np.allclose(loaded_midline.half_widths, midline.half_widths, atol=1e-6)
    assert loaded_midline.fish_id == fish_id
    assert loaded_midline.camera_id == cam
    assert loaded_midline.frame_index == 0
    assert loaded_midline.is_head_to_tail == midline.is_head_to_tail


def test_round_trip_multi_fish_multi_camera(tmp_path: Path) -> None:
    """Two fish, three cameras, two frames all survive round-trip."""
    observer = DiagnosticObserver()
    cam_ids = ("cam_a", "cam_b", "cam_c")
    frames = (0, 1)

    # Fish 0 observed in cam_a and cam_b; fish 1 in cam_b and cam_c
    tracklets_fish0 = tuple(
        _make_tracklet2d(c, i, frames) for i, c in enumerate(cam_ids[:2])
    )
    tracklets_fish1 = tuple(
        _make_tracklet2d(c, i + 10, frames) for i, c in enumerate(cam_ids[1:])
    )

    group0 = TrackletGroup(
        fish_id=0,
        tracklets=tracklets_fish0,
        confidence=0.9,
        per_frame_confidence=(0.9, 0.9),
        consensus_centroids=None,
    )
    group1 = TrackletGroup(
        fish_id=1,
        tracklets=tracklets_fish1,
        confidence=0.8,
        per_frame_confidence=(0.8, 0.8),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group0, group1])

    # Build annotated detections for two frames
    annotated_frames: list[dict] = []
    for fidx, frame_idx in enumerate(frames):
        frame_annot: dict = {}
        # Fish 0 in cam_a and cam_b
        for ti, cam in enumerate(cam_ids[:2]):
            midline = _make_midline2d(0, cam, frame_idx)
            centroid = tracklets_fish0[ti].centroids[fidx]
            frame_annot.setdefault(cam, []).append(
                _make_annotated_detection(midline, centroid)
            )
        # Fish 1 in cam_b and cam_c
        for ti, cam in enumerate(cam_ids[1:]):
            midline = _make_midline2d(1, cam, frame_idx)
            centroid = tracklets_fish1[ti].centroids[fidx]
            frame_annot.setdefault(cam, []).append(
                _make_annotated_detection(midline, centroid)
            )
        annotated_frames.append(frame_annot)

    _fire_midline_stage(observer, annotated_frames, len(frames), list(cam_ids))

    npz_path = _write_fixture(tmp_path, observer)
    fixture = load_midline_fixture(npz_path)

    assert len(fixture.frames) == 2
    assert set(fixture.frame_indices) == {0, 1}

    # Verify fish 0 in cam_a at frame 0
    assert 0 in fixture.frames[0]
    assert "cam_a" in fixture.frames[0][0]
    loaded = fixture.frames[0][0]["cam_a"]
    expected = _make_midline2d(0, "cam_a", 0)
    assert np.allclose(loaded.points, expected.points, atol=1e-6)

    # Verify fish 1 in cam_c at frame 1
    assert 1 in fixture.frames[1]
    assert "cam_c" in fixture.frames[1][1]


def test_load_raises_on_missing_version(tmp_path: Path) -> None:
    """ValueError raised with 'Missing meta/version' when key absent."""
    npz_path = tmp_path / "bad.npz"
    # Write NPZ without meta/version
    np.savez(str(npz_path), **{"meta/frame_indices": np.array([0], dtype=np.int64)})

    with pytest.raises(ValueError, match="Missing meta/version"):
        load_midline_fixture(npz_path)


def test_load_raises_on_wrong_version(tmp_path: Path) -> None:
    """ValueError raised with 'Unsupported fixture version' when version mismatch."""
    npz_path = tmp_path / "bad.npz"
    np.savez(
        str(npz_path),
        **{
            "meta/version": np.array("99.0"),
            "meta/frame_indices": np.array([0], dtype=np.int64),
            "meta/camera_ids": np.array([], dtype=object),
        },
    )

    with pytest.raises(ValueError, match="Unsupported fixture version"):
        load_midline_fixture(npz_path)


def test_empty_fixture_round_trip(tmp_path: Path) -> None:
    """Empty fixture (no midlines) exports and loads without error."""
    observer = DiagnosticObserver()
    fish_id = 0
    cam = "cam_a"
    frames = (0,)

    tracklet = _make_tracklet2d(cam, 0, frames)
    group = TrackletGroup(
        fish_id=fish_id,
        tracklets=(tracklet,),
        confidence=0.9,
        per_frame_confidence=(0.9,),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group])

    # Frame 0 has no midlines (None)
    centroid = tracklet.centroids[0]
    frame0_annot = {cam: [_make_annotated_detection(None, centroid)]}
    _fire_midline_stage(observer, [frame0_annot], 1, [cam])

    npz_path = _write_fixture(tmp_path, observer)
    fixture = load_midline_fixture(npz_path)

    assert isinstance(fixture, MidlineFixture)
    # No frames with midlines should produce empty frames tuple
    assert len(fixture.frames) == 0


def test_point_confidence_preserved(tmp_path: Path) -> None:
    """Non-uniform point_confidence arrays survive the round-trip exactly."""
    observer = DiagnosticObserver()
    fish_id = 0
    cam = "cam_a"
    frames = (0,)
    n_points = 7

    tracklet = _make_tracklet2d(cam, 0, frames)
    group = TrackletGroup(
        fish_id=fish_id,
        tracklets=(tracklet,),
        confidence=0.9,
        per_frame_confidence=(0.9,),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group])

    # Build midline with non-uniform point confidence
    midline = _make_midline2d(fish_id, cam, 0, n_points=n_points, confidence=None)
    midline.point_confidence = np.array(
        [0.1, 0.2, 0.5, 0.8, 0.95, 0.3, 0.7], dtype=np.float32
    )
    centroid = tracklet.centroids[0]
    frame0_annot = {cam: [_make_annotated_detection(midline, centroid)]}
    _fire_midline_stage(observer, [frame0_annot], 1, [cam])

    npz_path = _write_fixture(tmp_path, observer)
    fixture = load_midline_fixture(npz_path)

    loaded_midline = fixture.frames[0][fish_id][cam]
    assert loaded_midline.point_confidence is not None
    assert np.allclose(
        loaded_midline.point_confidence, midline.point_confidence, atol=1e-6
    )


# ---------------------------------------------------------------------------
# CalibBundle and v2.0 NPZ format tests (Task 1 of 41-01)
# ---------------------------------------------------------------------------


def _make_v20_npz(
    tmp_path: Path,
    cam_ids: tuple[str, ...] = ("cam_a", "cam_b"),
    n_points: int = 5,
) -> Path:
    """Write a minimal v2.0 NPZ file with calib/ keys and midline data.

    Returns the path to the written file.
    """
    npz_path = tmp_path / "fixture_v20.npz"

    arrays: dict[str, object] = {
        "meta/version": np.array("2.0", dtype=object),
        "meta/camera_ids": np.array(list(cam_ids), dtype=object),
        "meta/frame_indices": np.array([0], dtype=np.int64),
        "meta/frame_count": np.array(1, dtype=np.int64),
        "meta/timestamp": np.array("2026-01-01T00:00:00+00:00", dtype=object),
        # calib shared keys
        "calib/water_z": np.float32(-0.5),
        "calib/n_air": np.float32(1.0),
        "calib/n_water": np.float32(1.333),
        "calib/interface_normal": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    # Per-camera calib arrays and one midline entry per camera
    for i, cam_id in enumerate(cam_ids):
        K = np.eye(3, dtype=np.float32) * float(i + 1)
        R = np.eye(3, dtype=np.float32)
        t = np.array([float(i), 0.0, 0.0], dtype=np.float32)
        arrays[f"calib/{cam_id}/K_new"] = K
        arrays[f"calib/{cam_id}/R"] = R
        arrays[f"calib/{cam_id}/t"] = t

        prefix = f"midline/0/0/{cam_id}"
        pts = np.arange(n_points * 2, dtype=np.float32).reshape(n_points, 2)
        arrays[f"{prefix}/points"] = pts
        arrays[f"{prefix}/half_widths"] = np.ones(n_points, dtype=np.float32)
        arrays[f"{prefix}/point_confidence"] = np.ones(n_points, dtype=np.float32)
        arrays[f"{prefix}/is_head_to_tail"] = np.array(True, dtype=np.bool_)

    np.savez_compressed(str(npz_path), **arrays)
    return npz_path


def test_v10_backward_compat_calib_bundle_is_none(tmp_path: Path) -> None:
    """load_midline_fixture on a v1.0 NPZ returns MidlineFixture with calib_bundle=None."""
    # Write a standard v1.0 fixture using the round-trip helper from existing tests
    observer = DiagnosticObserver()
    cam = "cam_a"
    frames = (0,)
    tracklet = _make_tracklet2d(cam, 0, frames)
    group = TrackletGroup(
        fish_id=0,
        tracklets=(tracklet,),
        confidence=0.9,
        per_frame_confidence=(0.9,),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group])
    midline = _make_midline2d(0, cam, 0)
    centroid = tracklet.centroids[0]
    frame0_annot = {cam: [_make_annotated_detection(midline, centroid)]}
    _fire_midline_stage(observer, [frame0_annot], 1, [cam])

    npz_path = _write_fixture(tmp_path, observer)
    fixture = load_midline_fixture(npz_path)

    assert isinstance(fixture, MidlineFixture)
    assert fixture.calib_bundle is None


def test_v20_calib_bundle_populated(tmp_path: Path) -> None:
    """load_midline_fixture on a v2.0 NPZ returns MidlineFixture with populated CalibBundle."""
    cam_ids = ("cam_a", "cam_b")
    npz_path = _make_v20_npz(tmp_path, cam_ids=cam_ids)
    fixture = load_midline_fixture(npz_path)

    assert isinstance(fixture, MidlineFixture)
    assert fixture.calib_bundle is not None
    assert isinstance(fixture.calib_bundle, CalibBundle)


def test_v20_calib_bundle_fields_match(tmp_path: Path) -> None:
    """CalibBundle fields match the arrays written to calib/ keys."""
    cam_ids = ("cam_a", "cam_b")
    npz_path = _make_v20_npz(tmp_path, cam_ids=cam_ids)
    fixture = load_midline_fixture(npz_path)

    cb = fixture.calib_bundle
    assert cb is not None

    assert abs(cb.water_z - (-0.5)) < 1e-6
    assert abs(cb.n_air - 1.0) < 1e-6
    assert abs(cb.n_water - 1.333) < 1e-4
    np.testing.assert_allclose(
        cb.interface_normal, np.array([0.0, 0.0, -1.0], dtype=np.float32), atol=1e-6
    )

    for i, cam_id in enumerate(cam_ids):
        expected_K = np.eye(3, dtype=np.float32) * float(i + 1)
        expected_R = np.eye(3, dtype=np.float32)
        expected_t = np.array([float(i), 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(cb.K_new[cam_id], expected_K, atol=1e-6)
        np.testing.assert_allclose(cb.R[cam_id], expected_R, atol=1e-6)
        np.testing.assert_allclose(cb.t[cam_id], expected_t, atol=1e-6)


def test_v20_calib_bundle_camera_ids_match(tmp_path: Path) -> None:
    """CalibBundle.camera_ids matches the cameras found in calib/ keys."""
    cam_ids = ("cam_a", "cam_b")
    npz_path = _make_v20_npz(tmp_path, cam_ids=cam_ids)
    fixture = load_midline_fixture(npz_path)

    cb = fixture.calib_bundle
    assert cb is not None
    assert set(cb.camera_ids) == set(cam_ids)


def test_calib_bundle_is_frozen_dataclass() -> None:
    """CalibBundle is a frozen dataclass that raises on assignment."""
    import dataclasses

    cb = CalibBundle(
        camera_ids=("cam_a",),
        K_new={"cam_a": np.eye(3, dtype=np.float32)},
        R={"cam_a": np.eye(3, dtype=np.float32)},
        t={"cam_a": np.zeros(3, dtype=np.float32)},
        water_z=-0.5,
        interface_normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        n_air=1.0,
        n_water=1.333,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cb.water_z = 0.0  # type: ignore[misc]
