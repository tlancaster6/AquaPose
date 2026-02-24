"""Unit tests for FishTracker with track-driven association."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.segmentation.detector import Detection
from aquapose.tracking import FishTrack, FishTracker
from aquapose.tracking.tracker import TrackHealth, TrackState

# ---------------------------------------------------------------------------
# Synthetic rig helpers (reused from test_associate.py patterns)
# ---------------------------------------------------------------------------


def _make_overhead_camera(
    cam_x: float,
    cam_y: float,
    water_z: float = 1.0,
    fx: float = 1400.0,
    cx: float = 800.0,
    cy: float = 600.0,
) -> RefractiveProjectionModel:
    """Build a downward-looking camera at world position (cam_x, cam_y, 0)."""
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.tensor([-cam_x, -cam_y, 0.0], dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(
        K=K, R=R, t=t, water_z=water_z, normal=normal, n_air=1.0, n_water=1.333
    )


def _make_3_camera_rig() -> dict[str, RefractiveProjectionModel]:
    """Build 3 overhead cameras arranged in a triangle."""
    water_z = 1.0
    positions = [(-0.5, -0.4), (0.5, -0.4), (0.0, 0.5)]
    cam_ids = ["cam_a", "cam_b", "cam_c"]
    return {
        cam_id: _make_overhead_camera(x, y, water_z)
        for cam_id, (x, y) in zip(cam_ids, positions, strict=True)
    }


def _blob_mask(
    u: float, v: float, radius: int = 10, H: int = 1200, W: int = 1600
) -> np.ndarray:
    """Create a circular blob mask at pixel (u, v)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    uu = np.arange(W)
    vv = np.arange(H)
    UU, VV = np.meshgrid(uu, vv)
    mask[(UU - u) ** 2 + (VV - v) ** 2 <= radius**2] = 255
    return mask


def _project_to_detections(
    positions_3d: list[np.ndarray],
    models: dict[str, RefractiveProjectionModel],
    H: int = 1200,
    W: int = 1600,
) -> dict[str, list[Detection]]:
    """Project multiple 3D positions into all cameras, build Detection dicts.

    Returns a detections_per_camera dict with all fish in all cameras.
    """
    detections_per_camera: dict[str, list[Detection]] = {cam: [] for cam in models}
    for pos in positions_3d:
        pt = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
        for cam_id, model in models.items():
            with torch.no_grad():
                pixels, valid = model.project(pt)
            if valid[0]:
                u = float(pixels[0, 0])
                v = float(pixels[0, 1])
                if 0 <= u < W and 0 <= v < H:
                    mask = _blob_mask(u, v, radius=10, H=H, W=W)
                    area = int(np.count_nonzero(mask))
                    bbox = (max(0, int(u) - 10), max(0, int(v) - 10), 20, 20)
                    det = Detection(bbox=bbox, mask=mask, area=area, confidence=1.0)
                    detections_per_camera[cam_id].append(det)
    return detections_per_camera


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def models() -> dict[str, RefractiveProjectionModel]:
    return _make_3_camera_rig()


# ---------------------------------------------------------------------------
# Test 1: Single fish tracked across 5 frames
# ---------------------------------------------------------------------------


def test_single_fish_track_across_frames(
    models: dict[str, RefractiveProjectionModel],
) -> None:
    """Same fish_id maintained across 5 consecutive frames."""
    tracker = FishTracker(
        min_hits=2, max_age=7, expected_count=9, reprojection_threshold=25.0
    )
    seen_ids: set[int] = set()

    for f in range(5):
        x = float(f) * 0.01
        pos = [np.array([x, 0.0, 1.5], dtype=np.float32)]
        dets = _project_to_detections(pos, models)
        confirmed = tracker.update(dets, models, frame_index=f)

        if f >= 1:
            assert len(confirmed) == 1, f"Expected 1 confirmed track at frame {f}"
            seen_ids.add(confirmed[0].fish_id)

    assert len(seen_ids) == 1, f"Expected stable fish_id, got IDs: {seen_ids}"


# ---------------------------------------------------------------------------
# Test 2: Two fish no swap
# ---------------------------------------------------------------------------


def test_two_fish_no_swap(
    models: dict[str, RefractiveProjectionModel],
) -> None:
    """Two fish with separated trajectories maintain stable IDs."""
    tracker = FishTracker(
        min_hits=2,
        max_age=7,
        expected_count=9,
        reprojection_threshold=25.0,
        min_cameras_birth=2,
    )

    id_a: int | None = None
    id_b: int | None = None

    for f in range(10):
        t = float(f) * 0.005
        pos_a = np.array([-0.1 + t, -0.1, 1.5], dtype=np.float32)
        pos_b = np.array([0.1 - t, 0.1, 1.5], dtype=np.float32)
        dets = _project_to_detections([pos_a, pos_b], models)
        confirmed = tracker.update(dets, models, frame_index=f)

        if f >= 1:
            assert len(confirmed) == 2, f"Expected 2 confirmed at frame {f}"

            by_y = {}
            for tr in confirmed:
                y_val = round(float(list(tr.positions)[-1][1]), 1)
                by_y[y_val] = tr.fish_id
            fish_at_neg_y = by_y.get(-0.1)
            fish_at_pos_y = by_y.get(0.1)

            if id_a is None:
                id_a = fish_at_neg_y
                id_b = fish_at_pos_y
            else:
                assert fish_at_neg_y == id_a, f"Fish A swapped ID at frame {f}"
                assert fish_at_pos_y == id_b, f"Fish B swapped ID at frame {f}"


# ---------------------------------------------------------------------------
# Test 3: Birth confirmation (PROBATIONARY → CONFIRMED)
# ---------------------------------------------------------------------------


def test_birth_confirmation(
    models: dict[str, RefractiveProjectionModel],
) -> None:
    """Track is tentative for first min_hits-1 frames, then confirmed."""
    tracker = FishTracker(min_hits=2, max_age=7, reprojection_threshold=25.0)
    pos = [np.array([0.0, 0.0, 1.5], dtype=np.float32)]
    dets = _project_to_detections(pos, models)

    confirmed_0 = tracker.update(dets, models, frame_index=0)
    assert confirmed_0 == [], "Should be tentative on frame 0"

    all_tracks = tracker.get_all_tracks()
    assert len(all_tracks) >= 1
    assert all_tracks[0].state == TrackState.PROBATIONARY

    confirmed_1 = tracker.update(dets, models, frame_index=1)
    assert len(confirmed_1) == 1
    assert confirmed_1[0].is_confirmed
    assert confirmed_1[0].state == TrackState.CONFIRMED


# ---------------------------------------------------------------------------
# Test 4: Grace period and death (CONFIRMED → COASTING → DEAD)
# ---------------------------------------------------------------------------


def test_grace_period_and_death(
    models: dict[str, RefractiveProjectionModel],
) -> None:
    """Track survives max_age missed frames, dies after exceeding it."""
    max_age = 7
    tracker = FishTracker(min_hits=1, max_age=max_age, reprojection_threshold=25.0)
    pos = [np.array([0.0, 0.0, 1.5], dtype=np.float32)]
    dets = _project_to_detections(pos, models)

    # Seed the track
    tracker.update(dets, models, frame_index=0)

    # Feed empty frames
    empty: dict[str, list[Detection]] = {cam: [] for cam in models}
    for missed in range(1, max_age + 2):
        tracker.update(empty, models, frame_index=missed)

        alive_tracks = tracker.get_all_tracks()
        if missed <= max_age:
            assert len(alive_tracks) >= 1, (
                f"Track should still be alive after {missed} missed frame(s)"
            )
        else:
            assert alive_tracks == [], (
                f"Track should be dead after {missed} missed frame(s)"
            )


# ---------------------------------------------------------------------------
# Test 5: Population constraint (TRACK-04) — dead track ID re-used
# ---------------------------------------------------------------------------


def test_population_constraint_relinking(
    models: dict[str, RefractiveProjectionModel],
) -> None:
    """New observation in the same frame as a track death inherits the dead ID."""
    # max_age=1 means confirmed tracks die after 2 missed frames.
    # min_hits=1 for immediate confirmation. Use reprojection_threshold=25.0
    # for synthetic overhead cameras.
    tracker = FishTracker(min_hits=1, max_age=1, reprojection_threshold=25.0)

    pos_a = [np.array([0.0, 0.0, 1.5], dtype=np.float32)]
    dets_a = _project_to_detections(pos_a, models)
    tracker.update(dets_a, models, frame_index=0)

    original_id = tracker.get_all_tracks()[0].fish_id

    # Frame 1: empty — track misses (confirmed → coasting, fsu=1)
    empty: dict[str, list[Detection]] = {cam: [] for cam in models}
    tracker.update(empty, models, frame_index=1)
    assert len(tracker.get_all_tracks()) == 1, "Track should still be alive (coasting)"

    # Frame 2: track dies (fsu=2 > max_age=1), new fish at different position
    # Use (0.1, 0.1) — far enough from (0.0, 0.0) to not be claimed (~93px),
    # but still within camera image bounds.
    pos_b = [np.array([0.1, 0.1, 1.5], dtype=np.float32)]
    dets_b = _project_to_detections(pos_b, models)
    tracker.update(dets_b, models, frame_index=2)

    all_tracks = tracker.get_all_tracks()
    ids = {t.fish_id for t in all_tracks}
    assert original_id in ids, f"Expected recycled ID {original_id} in tracks {ids}"


# ---------------------------------------------------------------------------
# Test 6: Constant-velocity prediction
# ---------------------------------------------------------------------------


def test_constant_velocity_prediction() -> None:
    """predict() extrapolates from velocity correctly."""
    track = FishTrack(fish_id=0)
    track.positions.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    track.positions.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    track.velocity = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    predicted = track.predict()
    np.testing.assert_allclose(predicted, [2.0, 0.0, 0.0], atol=1e-6)


def test_constant_velocity_single_position() -> None:
    """predict() with one position and zero velocity returns that position."""
    track = FishTrack(fish_id=0)
    track.positions.append(np.array([3.0, 1.0, 0.5], dtype=np.float32))

    predicted = track.predict()
    np.testing.assert_allclose(predicted, [3.0, 1.0, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# Test 7: First-frame batch init — IDs sorted by X coordinate
# ---------------------------------------------------------------------------


def test_first_frame_batch_init(
    models: dict[str, RefractiveProjectionModel],
) -> None:
    """Multiple associations on frame 0 create tracks with IDs sorted by X."""
    tracker = FishTracker(min_hits=1, reprojection_threshold=25.0)

    # 3 fish at different X positions
    positions = [
        np.array([0.15, 0.0, 1.5], dtype=np.float32),
        np.array([-0.15, 0.0, 1.5], dtype=np.float32),
        np.array([0.0, 0.0, 1.5], dtype=np.float32),
    ]
    dets = _project_to_detections(positions, models)
    tracker.update(dets, models, frame_index=0)

    tracks = tracker.get_all_tracks()
    assert len(tracks) >= 3, f"Expected >= 3 tracks, got {len(tracks)}"

    # Verify IDs are sorted by X
    id_to_x = {t.fish_id: float(list(t.positions)[-1][0]) for t in tracks}
    sorted_ids = sorted(id_to_x.keys())
    xs = [id_to_x[i] for i in sorted_ids]
    assert xs == sorted(xs), f"Track IDs should be ordered by X, got: {xs}"


# ---------------------------------------------------------------------------
# Test 8: get_seed_points returns predicted positions
# ---------------------------------------------------------------------------


def test_get_seed_points(
    models: dict[str, RefractiveProjectionModel],
) -> None:
    """get_seed_points() returns predicted positions of confirmed tracks."""
    tracker = FishTracker(min_hits=2, max_age=7, reprojection_threshold=25.0)
    assert tracker.get_seed_points() is None

    pos = [np.array([0.05, 0.02, 1.5], dtype=np.float32)]
    dets = _project_to_detections(pos, models)

    tracker.update(dets, models, frame_index=0)
    assert tracker.get_seed_points() is None  # not confirmed yet

    tracker.update(dets, models, frame_index=1)
    seeds = tracker.get_seed_points()
    assert seeds is not None
    assert len(seeds) == 1


# ---------------------------------------------------------------------------
# Test 9: Coasting velocity damping
# ---------------------------------------------------------------------------


def test_coasting_velocity_damping() -> None:
    """During COASTING, each mark_missed advances prediction by damped velocity step.

    After 2 mark_missed calls (2 coasting frames) with velocity=[1, 0, 0] and
    damping=0.5, the cumulative prediction is:
      step1: last_pos + vel*0.5 = [0.5, 0, 1]
      step2: [0.5, 0, 1] + vel*0.5 = [1.0, 0, 1]
    """
    track = FishTrack(fish_id=0, velocity_damping=0.5)
    track.state = TrackState.CONFIRMED
    track.positions.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    track.velocity = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # First missed frame: CONFIRMED -> COASTING, prediction seeded
    track.mark_missed()
    assert track.state == TrackState.COASTING
    # After first miss: prediction = [0, 0, 1] + [1, 0, 0]*0.5 = [0.5, 0, 1]
    np.testing.assert_allclose(track.predict(), [0.5, 0.0, 1.0], atol=1e-6)

    # Second missed frame: prediction advances
    track.mark_missed()
    # After second miss: [0.5, 0, 1] + [1, 0, 0]*0.5 = [1.0, 0, 1]
    np.testing.assert_allclose(track.predict(), [1.0, 0.0, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Test 10: Probationary short leash
# ---------------------------------------------------------------------------


def test_probationary_short_leash() -> None:
    """Probationary tracks die after 2 missed frames (not max_age)."""
    track = FishTrack(fish_id=0, max_age=10)
    track.state = TrackState.PROBATIONARY

    track.mark_missed()
    assert not track.is_dead  # frames_since_update=1
    track.mark_missed()
    assert not track.is_dead  # frames_since_update=2
    track.mark_missed()
    assert track.is_dead  # frames_since_update=3 > 2


# ---------------------------------------------------------------------------
# Test 11: Full state lifecycle transitions
# ---------------------------------------------------------------------------


def test_state_lifecycle_transitions() -> None:
    """PROBATIONARY → CONFIRMED → COASTING → back to CONFIRMED on re-acquisition."""
    track = FishTrack(fish_id=0, min_hits=2, max_age=5)

    # Starts PROBATIONARY
    assert track.state == TrackState.PROBATIONARY

    # First update: still PROBATIONARY (consecutive_hits=1 < min_hits=2)
    track.update_from_claim(
        centroid_3d=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        camera_detections={"cam_a": 0},
        reprojection_residual=5.0,
        n_cameras=2,
    )
    assert track.state == TrackState.PROBATIONARY

    # Second update: CONFIRMED (consecutive_hits=2 >= min_hits=2)
    track.update_from_claim(
        centroid_3d=np.array([0.01, 0.0, 1.0], dtype=np.float32),
        camera_detections={"cam_a": 0},
        reprojection_residual=5.0,
        n_cameras=2,
    )
    assert track.state == TrackState.CONFIRMED

    # Miss: COASTING
    track.mark_missed()
    assert track.state == TrackState.COASTING

    # Re-acquire: back to CONFIRMED
    track.update_from_claim(
        centroid_3d=np.array([0.02, 0.0, 1.0], dtype=np.float32),
        camera_detections={"cam_a": 0},
        reprojection_residual=5.0,
        n_cameras=2,
    )
    assert track.state == TrackState.CONFIRMED


# ---------------------------------------------------------------------------
# Test 12: Single-view freezes velocity
# ---------------------------------------------------------------------------


def test_single_view_freezes_velocity() -> None:
    """update_position_only freezes velocity while updating position."""
    track = FishTrack(fish_id=0, min_hits=1)
    track.velocity = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    track.positions.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    track.update_position_only(
        centroid_3d=np.array([0.5, 0.0, 1.0], dtype=np.float32),
        camera_detections={"cam_a": 0},
        reprojection_residual=3.0,
        n_cameras=1,
    )

    # Velocity should still be the original value (frozen)
    np.testing.assert_allclose(track.velocity, [1.0, 0.0, 0.0], atol=1e-6)
    # But position was updated
    np.testing.assert_allclose(list(track.positions)[-1], [0.5, 0.0, 1.0], atol=1e-6)
    # degraded_frames incremented
    assert track.health.degraded_frames == 1


# ---------------------------------------------------------------------------
# Test 13: TrackHealth statistics
# ---------------------------------------------------------------------------


def test_track_health_statistics() -> None:
    """TrackHealth correctly tracks running statistics."""
    health = TrackHealth()

    assert health.mean_residual == 0.0
    assert health.mean_cameras == 0.0

    health.residual_history.append(5.0)
    health.residual_history.append(10.0)
    health.cameras_per_frame.append(3)
    health.cameras_per_frame.append(5)

    assert health.mean_residual == 7.5
    assert health.mean_cameras == 4.0


# ---------------------------------------------------------------------------
# Test 14: is_confirmed property for writer compat
# ---------------------------------------------------------------------------


def test_is_confirmed_property() -> None:
    """is_confirmed returns True for CONFIRMED and COASTING states."""
    track = FishTrack(fish_id=0)

    track.state = TrackState.PROBATIONARY
    assert not track.is_confirmed

    track.state = TrackState.CONFIRMED
    assert track.is_confirmed

    track.state = TrackState.COASTING
    assert track.is_confirmed


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


def _make_duplicate_tracks(
    fish_id_a: int,
    fish_id_b: int,
    pos: np.ndarray,
    *,
    shared_cameras: bool = False,
    window: int = 10,
) -> tuple[FishTrack, FishTrack]:
    """Create two confirmed tracks at (nearly) the same position.

    If shared_cameras is False, track A claims {cam_a, cam_b} and track B
    claims {cam_c} — simulating a detection split with low co-visibility.
    If True, both claim {cam_a, cam_b} — simulating two real fish.
    """
    a = FishTrack(fish_id=fish_id_a, state=TrackState.CONFIRMED)
    b = FishTrack(fish_id=fish_id_b, state=TrackState.CONFIRMED)

    offset = np.array([0.005, 0.005, 0.0], dtype=np.float32)

    for _ in range(window):
        a.positions.append(pos.copy())
        b.positions.append((pos + offset).copy())

        if shared_cameras:
            a.camera_history.append(frozenset(["cam_a", "cam_b"]))
            b.camera_history.append(frozenset(["cam_a", "cam_b"]))
        else:
            a.camera_history.append(frozenset(["cam_a", "cam_b"]))
            b.camera_history.append(frozenset(["cam_c"]))

    return a, b


def test_dedup_kills_younger_duplicate() -> None:
    """Two confirmed tracks splitting one fish's detections are deduplicated."""
    pos = np.array([0.1, 0.2, 1.5], dtype=np.float32)
    a, b = _make_duplicate_tracks(0, 1, pos, shared_cameras=False, window=10)

    tracker = FishTracker(dedup_distance=0.04, dedup_window=10)
    tracker.tracks = [a, b]
    tracker._dedup_confirmed_tracks()

    # Younger track (ID 1) should be killed
    assert a.state == TrackState.CONFIRMED
    assert b.state == TrackState.DEAD


def test_dedup_spares_covisible_tracks() -> None:
    """Two confirmed tracks with high camera co-visibility are not deduplicated."""
    pos = np.array([0.1, 0.2, 1.5], dtype=np.float32)
    a, b = _make_duplicate_tracks(0, 1, pos, shared_cameras=True, window=10)

    tracker = FishTracker(dedup_distance=0.04, dedup_window=10)
    tracker.tracks = [a, b]
    tracker._dedup_confirmed_tracks()

    # Both should survive — they both claim the same cameras (real schooling fish)
    assert a.state == TrackState.CONFIRMED
    assert b.state == TrackState.CONFIRMED


def test_dedup_spares_distant_tracks() -> None:
    """Two confirmed tracks far apart are not deduplicated."""
    pos_a = np.array([0.1, 0.2, 1.5], dtype=np.float32)
    pos_b = np.array([0.3, 0.4, 1.5], dtype=np.float32)

    a = FishTrack(fish_id=0, state=TrackState.CONFIRMED)
    b = FishTrack(fish_id=1, state=TrackState.CONFIRMED)
    for _ in range(10):
        a.positions.append(pos_a.copy())
        b.positions.append(pos_b.copy())
        a.camera_history.append(frozenset(["cam_a", "cam_b"]))
        b.camera_history.append(frozenset(["cam_c"]))

    tracker = FishTracker(dedup_distance=0.04, dedup_window=10)
    tracker.tracks = [a, b]
    tracker._dedup_confirmed_tracks()

    assert a.state == TrackState.CONFIRMED
    assert b.state == TrackState.CONFIRMED


def test_dedup_requires_full_window() -> None:
    """Tracks with insufficient history are not deduplicated."""
    pos = np.array([0.1, 0.2, 1.5], dtype=np.float32)
    a, b = _make_duplicate_tracks(0, 1, pos, shared_cameras=False, window=5)

    tracker = FishTracker(dedup_distance=0.04, dedup_window=10)
    tracker.tracks = [a, b]
    tracker._dedup_confirmed_tracks()

    # Only 5 frames of history, need 10 — no dedup
    assert a.state == TrackState.CONFIRMED
    assert b.state == TrackState.CONFIRMED


# ---------------------------------------------------------------------------
# Coasting prediction accumulation regression test
# ---------------------------------------------------------------------------


def test_coasting_prediction_tracks_fish() -> None:
    """Coasting prediction accumulates over missed frames, tracking fish motion.

    Regression test for the jerk-freeze-die bug: the old predict() computed
    last_pos + vel*damping^fsu, which stalled near last_pos as fsu grew.
    After max_age=7 missed frames at fish speed 0.05 m/s, prediction error
    was ~36 px (>> reprojection_threshold=15 px), preventing re-acquisition.

    The fix advances _coasting_prediction by vel*damping each mark_missed()
    call, keeping the prediction close to the fish trajectory.
    """
    # Simulate a fish at 0.05 m/s (1.67 mm/frame at 30fps)
    vel = np.array([0.00167, 0.0, 0.0], dtype=np.float32)
    start_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)

    track = FishTrack(fish_id=0, velocity_damping=0.8)
    track.state = TrackState.CONFIRMED
    track.positions.append(start_pos.copy())
    track.velocity = vel.copy()

    # Simulate max_age=7 coasting frames
    for _ in range(7):
        track.mark_missed()

    # True fish position after 7 frames
    true_pos_after_7 = start_pos + vel * 7  # ~11.7 mm from start

    # Predicted position from coasting model
    predicted = track.predict()

    # With the fix: error should be small (damping=0.8 means each step is
    # slightly shorter, but 7*vel*0.8 ≈ 5.6*vel vs 7*vel → error = 1.4*vel ≈ 2.3mm)
    # Without the fix (old bug): error would be last_pos + vel*0.8^7 ≈ last_pos + 0.37*vel
    # → error from true pos = (7 - 0.37)*vel ≈ 6.63*vel ≈ 11mm ≈ 36px
    error_mm = float(np.linalg.norm(predicted - true_pos_after_7)) * 1000
    assert error_mm < 5.0, (
        f"Coasting prediction error {error_mm:.1f}mm is too large. "
        f"Predicted: {predicted}, true: {true_pos_after_7}"
    )
