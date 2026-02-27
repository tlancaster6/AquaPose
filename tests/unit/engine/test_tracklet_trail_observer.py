"""Unit tests for TrackletTrailObserver — trail rendering, mosaic composition, color assignment."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

from aquapose.core.association.types import TrackletGroup
from aquapose.core.tracking.types import Tracklet2D
from aquapose.engine.events import PipelineComplete, StageComplete
from aquapose.engine.observer_factory import _OBSERVER_MAP, build_observers
from aquapose.engine.tracklet_trail_observer import (
    FISH_COLORS_BGR,
    TrackletTrailObserver,
    _coasted_color,
)

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_observer(tmp_path: Path) -> TrackletTrailObserver:
    """Return a TrackletTrailObserver wired to a temp directory."""
    return TrackletTrailObserver(
        output_dir=tmp_path / "output",
        video_dir=tmp_path / "videos",
        calibration_path=tmp_path / "calibration.json",
    )


def _make_tracklet(
    camera_id: str,
    track_id: int,
    frames: tuple,
    centroids: tuple,
    statuses: tuple | None = None,
) -> Tracklet2D:
    """Create a Tracklet2D with simple bboxes derived from centroids."""
    if statuses is None:
        statuses = tuple("detected" for _ in frames)
    bboxes = tuple((u - 10, v - 10, 20.0, 20.0) for u, v in centroids)
    return Tracklet2D(
        camera_id=camera_id,
        track_id=track_id,
        frames=frames,
        centroids=centroids,
        bboxes=bboxes,
        frame_status=statuses,
    )


def _make_tracklet_group(fish_id: int, tracklets: list[Tracklet2D]) -> TrackletGroup:
    return TrackletGroup(fish_id=fish_id, tracklets=tuple(tracklets))


# ---------------------------------------------------------------------------
# Test 1: ignores non-PipelineComplete events
# ---------------------------------------------------------------------------


def test_ignores_non_pipeline_complete_events(tmp_path: Path) -> None:
    """Observer must not perform any video I/O on non-PipelineComplete events."""
    observer = _make_observer(tmp_path)
    event = StageComplete(stage_name="detection", stage_index=0)

    # No exception and no filesystem side effects.
    with patch("aquapose.engine.tracklet_trail_observer.cv2") as mock_cv2:
        observer.on_event(event)
        mock_cv2.VideoWriter.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: skips gracefully when tracks_2d is None
# ---------------------------------------------------------------------------


def test_skips_when_tracks_2d_is_none(tmp_path: Path) -> None:
    """Observer must no-op when context.tracks_2d is None."""
    observer = _make_observer(tmp_path)
    ctx = MagicMock()
    ctx.tracks_2d = None
    ctx.tracklet_groups = []
    event = PipelineComplete(run_id="test", context=ctx)

    # Should log a warning and return without raising.
    observer.on_event(event)
    # No video directory created means no I/O attempted.
    assert not (tmp_path / "output" / "observers" / "diagnostics").exists()


# ---------------------------------------------------------------------------
# Test 3: color assignment from tracklet_groups
# ---------------------------------------------------------------------------


def test_color_assignment_from_tracklet_groups(tmp_path: Path) -> None:
    """_build_color_map must return correct fish_id->BGR and (cam, track)->fish_id maps."""
    observer = _make_observer(tmp_path)

    t0 = _make_tracklet("cam1", 10, (0, 1), ((100.0, 200.0), (110.0, 205.0)))
    t1 = _make_tracklet("cam2", 20, (0, 1), ((300.0, 150.0), (305.0, 152.0)))
    t2 = _make_tracklet("cam1", 30, (0, 1), ((50.0, 50.0), (55.0, 52.0)))

    groups = [
        _make_tracklet_group(0, [t0]),
        _make_tracklet_group(1, [t1]),
        _make_tracklet_group(2, [t2]),
    ]

    fish_color_map, track_to_fish = observer._build_color_map(groups)

    # Each fish_id gets the correct palette color.
    assert fish_color_map[0] == FISH_COLORS_BGR[0 % len(FISH_COLORS_BGR)]
    assert fish_color_map[1] == FISH_COLORS_BGR[1 % len(FISH_COLORS_BGR)]
    assert fish_color_map[2] == FISH_COLORS_BGR[2 % len(FISH_COLORS_BGR)]

    # Reverse mapping is correct.
    assert track_to_fish[("cam1", 10)] == 0
    assert track_to_fish[("cam2", 20)] == 1
    assert track_to_fish[("cam1", 30)] == 2


# ---------------------------------------------------------------------------
# Test 4: coasted color is lighter than base
# ---------------------------------------------------------------------------


def test_coasted_color_lighter_than_base() -> None:
    """Each channel of the coasted color must be closer to 128 than the base."""
    base = (20, 180, 50)
    coasted = _coasted_color(base)

    for base_ch, coasted_ch in zip(base, coasted, strict=True):
        dist_base = abs(base_ch - 128)
        dist_coasted = abs(coasted_ch - 128)
        assert dist_coasted < dist_base, (
            f"Channel {base_ch} -> coasted {coasted_ch} is not closer to 128"
        )


# ---------------------------------------------------------------------------
# Test 5: frame lookup structure
# ---------------------------------------------------------------------------


def test_trail_lookup_structure(tmp_path: Path) -> None:
    """_build_frame_lookup must correctly index tracklets per camera per frame."""
    observer = _make_observer(tmp_path)

    t_cam1_a = _make_tracklet(
        "cam1", 1, (0, 1, 2), ((10.0, 10.0), (11.0, 11.0), (12.0, 12.0))
    )
    t_cam1_b = _make_tracklet("cam1", 2, (1, 2), ((50.0, 50.0), (51.0, 51.0)))
    t_cam2_a = _make_tracklet("cam2", 3, (0, 1), ((200.0, 200.0), (201.0, 201.0)))

    tracks_2d = {
        "cam1": [t_cam1_a, t_cam1_b],
        "cam2": [t_cam2_a],
    }
    track_to_fish = {
        ("cam1", 1): 0,
        ("cam1", 2): 1,
        ("cam2", 3): 0,
    }

    lookup = observer._build_frame_lookup(tracks_2d, track_to_fish)

    # cam1 frame 0: only tracklet 1 (idx 0)
    cam1_f0 = lookup["cam1"][0]
    assert len(cam1_f0) == 1
    tracklet_obj, idx_in_tracklet, fish_id = cam1_f0[0]
    assert tracklet_obj is t_cam1_a
    assert idx_in_tracklet == 0
    assert fish_id == 0

    # cam1 frame 1: both tracklets
    cam1_f1 = lookup["cam1"][1]
    assert len(cam1_f1) == 2

    # cam1 frame 2: both tracklets
    cam1_f2 = lookup["cam1"][2]
    assert len(cam1_f2) == 2

    # cam2 frame 0: tracklet 3 (idx 0)
    cam2_f0 = lookup["cam2"][0]
    assert len(cam2_f0) == 1
    assert cam2_f0[0][2] == 0  # fish_id

    # cam2 frame 2: not present
    assert 2 not in lookup["cam2"]


# ---------------------------------------------------------------------------
# Test 6: mosaic_dims for various camera counts
# ---------------------------------------------------------------------------


def test_mosaic_dims() -> None:
    """_mosaic_dims must return correct grid dimensions."""
    tile_w, tile_h = 100, 80

    # 12 cameras → ceil(sqrt(12)) = 4 cols, ceil(12/4) = 3 rows
    mh, mw = TrackletTrailObserver._mosaic_dims(12, tile_w, tile_h)
    assert mw == 4 * tile_w
    assert mh == 3 * tile_h

    # 1 camera → 1 col, 1 row
    mh, mw = TrackletTrailObserver._mosaic_dims(1, tile_w, tile_h)
    assert mw == tile_w
    assert mh == tile_h

    # 7 cameras → ceil(sqrt(7)) = 3 cols, ceil(7/3) = 3 rows
    n_cols = math.ceil(math.sqrt(7))
    n_rows = math.ceil(7 / n_cols)
    mh, mw = TrackletTrailObserver._mosaic_dims(7, tile_w, tile_h)
    assert mw == n_cols * tile_w
    assert mh == n_rows * tile_h


# ---------------------------------------------------------------------------
# Test 7: build_observers diagnostic mode includes TrackletTrailObserver
# ---------------------------------------------------------------------------


def test_build_observers_diagnostic_mode_includes_tracklet_trail(
    tmp_path: Path,
) -> None:
    """build_observers in diagnostic mode must include a TrackletTrailObserver."""
    config = MagicMock()
    config.output_dir = str(tmp_path / "output")
    config.video_dir = str(tmp_path / "videos")
    config.calibration_path = str(tmp_path / "calibration.json")

    observers = build_observers(
        config=config,
        mode="diagnostic",
        verbose=False,
        total_stages=5,
    )

    trail_observers = [o for o in observers if isinstance(o, TrackletTrailObserver)]
    assert len(trail_observers) >= 1, (
        "diagnostic mode must include TrackletTrailObserver"
    )


# ---------------------------------------------------------------------------
# Test 8: observer map registration
# ---------------------------------------------------------------------------


def test_observer_in_observer_map() -> None:
    """'tracklet_trail' must be registered in _OBSERVER_MAP and map to TrackletTrailObserver."""
    assert "tracklet_trail" in _OBSERVER_MAP
    assert _OBSERVER_MAP["tracklet_trail"] is TrackletTrailObserver


# ---------------------------------------------------------------------------
# Test 9: engine __init__ exports
# ---------------------------------------------------------------------------


def test_engine_init_exports() -> None:
    """TrackletTrailObserver must be importable from aquapose.engine and listed in __all__."""
    import aquapose.engine as engine

    assert hasattr(engine, "TrackletTrailObserver")
    assert engine.TrackletTrailObserver is TrackletTrailObserver
    assert "TrackletTrailObserver" in engine.__all__
