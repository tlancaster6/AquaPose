"""Unit tests for FishTracker lifecycle and Hungarian assignment."""

from __future__ import annotations

import numpy as np

from aquapose.tracking import FishTrack, FishTracker
from aquapose.tracking.associate import AssociationResult, FrameAssociations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assoc(
    fish_id: int,
    centroid: tuple[float, float, float],
    *,
    residual: float = 0.5,
    confidence: float = 1.0,
    n_cameras: int = 3,
    camera_detections: dict[str, int] | None = None,
    is_low_confidence: bool = False,
) -> AssociationResult:
    """Build a synthetic AssociationResult."""
    if camera_detections is None:
        camera_detections = {f"cam{i}": i for i in range(n_cameras)}
    return AssociationResult(
        fish_id=fish_id,
        centroid_3d=np.array(centroid, dtype=np.float32),
        reprojection_residual=residual,
        camera_detections=camera_detections,
        n_cameras=n_cameras,
        confidence=confidence,
        is_low_confidence=is_low_confidence,
    )


def _make_frame(
    associations: list[AssociationResult],
    frame_index: int = 0,
) -> FrameAssociations:
    """Build a FrameAssociations from a list of AssociationResult objects."""
    return FrameAssociations(associations=associations, frame_index=frame_index)


# ---------------------------------------------------------------------------
# Test 1: Single fish tracked across 5 frames
# ---------------------------------------------------------------------------


def test_single_fish_track_across_frames() -> None:
    """Same fish_id maintained across 5 consecutive frames; confirmed after frame 2."""
    tracker = FishTracker(min_hits=2, max_age=7)
    seen_ids: set[int] = set()

    for f in range(5):
        x = float(f) * 0.01  # linear motion in X
        assoc = _make_assoc(0, (x, 0.0, 0.5))
        frame = _make_frame([assoc], frame_index=f)
        confirmed = tracker.update(frame)

        if f >= 1:  # confirmed after min_hits=2 (frames 0 and 1)
            assert len(confirmed) == 1, f"Expected 1 confirmed track at frame {f}"
            seen_ids.add(confirmed[0].fish_id)

    # All confirmed returns should carry the same ID
    assert len(seen_ids) == 1, f"Expected stable fish_id, got IDs: {seen_ids}"


def test_single_fish_confirmed_after_min_hits() -> None:
    """Track is NOT confirmed on frame 0, IS confirmed from frame 1 onward."""
    tracker = FishTracker(min_hits=2, max_age=7)

    assoc = _make_assoc(0, (0.0, 0.0, 0.5))

    # Frame 0: no confirmed tracks yet (hit_streak=1 < min_hits=2)
    confirmed_0 = tracker.update(_make_frame([assoc], frame_index=0))
    assert confirmed_0 == [], "Track should not be confirmed on frame 0"

    # Frame 1: hit_streak=2 >= min_hits=2 → confirmed
    confirmed_1 = tracker.update(_make_frame([assoc], frame_index=1))
    assert len(confirmed_1) == 1, "Track should be confirmed on frame 1"
    assert confirmed_1[0].is_confirmed


# ---------------------------------------------------------------------------
# Test 2: Two fish moving in opposite directions — no ID swaps
# ---------------------------------------------------------------------------


def test_two_fish_no_swap() -> None:
    """Two fish with crossing XY trajectories maintain stable IDs."""
    tracker = FishTracker(min_hits=2, max_age=7, max_distance=0.1)

    # Fish A starts at x=-0.5, Fish B at x=+0.5; both move slowly
    id_a: int | None = None
    id_b: int | None = None

    for f in range(10):
        t = float(f) * 0.005  # small increments (< max_distance)
        # Fish A moves right, Fish B moves left — they are separated in Y
        assoc_a = _make_assoc(0, (-0.5 + t, -0.2, 0.5))
        assoc_b = _make_assoc(1, (0.5 - t, +0.2, 0.5))
        frame = _make_frame([assoc_a, assoc_b], frame_index=f)
        confirmed = tracker.update(frame)

        if f >= 1:
            assert len(confirmed) == 2, f"Expected 2 confirmed at frame {f}"

            by_pos = {(round(t.positions[-1][1], 1)): t.fish_id for t in confirmed}
            fish_at_neg_y = by_pos.get(-0.2) or by_pos.get(-0.1)
            fish_at_pos_y = by_pos.get(0.2) or by_pos.get(0.1)

            if id_a is None:
                id_a = fish_at_neg_y
                id_b = fish_at_pos_y
            else:
                assert fish_at_neg_y == id_a, f"Fish A swapped ID at frame {f}"
                assert fish_at_pos_y == id_b, f"Fish B swapped ID at frame {f}"


# ---------------------------------------------------------------------------
# Test 3: Birth confirmation
# ---------------------------------------------------------------------------


def test_birth_confirmation() -> None:
    """Track is tentative for first min_hits-1 frames, then confirmed."""
    tracker = FishTracker(min_hits=2, max_age=7)
    assoc = _make_assoc(0, (0.0, 0.0, 0.5))

    confirmed_0 = tracker.update(_make_frame([assoc], frame_index=0))
    assert confirmed_0 == [], "Should be tentative on frame 0"

    all_tracks = tracker.get_all_tracks()
    assert len(all_tracks) == 1
    assert not all_tracks[0].is_confirmed

    confirmed_1 = tracker.update(_make_frame([assoc], frame_index=1))
    assert len(confirmed_1) == 1
    assert confirmed_1[0].is_confirmed


# ---------------------------------------------------------------------------
# Test 4: Grace period and death
# ---------------------------------------------------------------------------


def test_grace_period_and_death() -> None:
    """Track survives max_age missed frames, dies on max_age+1."""
    max_age = 7
    tracker = FishTracker(
        min_hits=1, max_age=max_age
    )  # min_hits=1 for immediate confirm
    assoc = _make_assoc(0, (0.0, 0.0, 0.5))

    # Seed the track
    tracker.update(_make_frame([assoc], frame_index=0))

    # Feed empty frames — no detections
    for missed in range(1, max_age + 2):
        empty_frame = _make_frame([], frame_index=missed)
        tracker.update(empty_frame)

        alive_tracks = tracker.get_all_tracks()
        if missed <= max_age:
            assert len(alive_tracks) == 1, (
                f"Track should still be alive after {missed} missed frame(s)"
            )
            assert alive_tracks[0].frames_since_update == missed
        else:
            # frames_since_update == max_age+1 → is_dead → pruned
            assert alive_tracks == [], (
                f"Track should be dead after {missed} missed frame(s)"
            )


# ---------------------------------------------------------------------------
# Test 5: Population constraint (TRACK-04) — dead track ID re-used
# ---------------------------------------------------------------------------


def test_population_constraint_relinking() -> None:
    """New observation in the same frame as a track death inherits the dead ID."""
    tracker = FishTracker(min_hits=1, max_age=0)  # max_age=0: dies after 1 missed frame

    # Create a track for fish A
    assoc_a = _make_assoc(0, (0.0, 0.0, 0.5))
    tracker.update(_make_frame([assoc_a], frame_index=0))

    original_id = tracker.get_all_tracks()[0].fish_id

    # Frame 1: fish A is gone, but a new observation appears far away
    # Fish A's track will miss this frame → is_dead (frames_since_update=1 > max_age=0)
    new_obs = _make_assoc(0, (5.0, 5.0, 0.5))  # far from old position
    tracker.update(_make_frame([new_obs], frame_index=1))

    all_tracks = tracker.get_all_tracks()
    # Population constraint: new track should have inherited original_id
    ids = {t.fish_id for t in all_tracks}
    assert original_id in ids, f"Expected recycled ID {original_id} in tracks {ids}"


# ---------------------------------------------------------------------------
# Test 6: Constant-velocity prediction
# ---------------------------------------------------------------------------


def test_constant_velocity_prediction() -> None:
    """predict() extrapolates from two positions correctly."""
    track = FishTrack(fish_id=0)
    track.positions.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    track.positions.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    predicted = track.predict()
    np.testing.assert_allclose(predicted, [2.0, 0.0, 0.0], atol=1e-6)


def test_constant_velocity_single_position() -> None:
    """predict() with one position returns that position (zero velocity)."""
    track = FishTrack(fish_id=0)
    track.positions.append(np.array([3.0, 1.0, 0.5], dtype=np.float32))

    predicted = track.predict()
    np.testing.assert_allclose(predicted, [3.0, 1.0, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# Test 7: XY-only cost matrix — Z-noise does not cause ID swaps
# ---------------------------------------------------------------------------


def test_xy_only_cost_matrix() -> None:
    """Tracks matched correctly despite Z differences; XY determines cost."""
    tracker = FishTracker(min_hits=1, max_age=7, max_distance=0.5)

    # Two fish at same XY, different Z
    assoc_a = _make_assoc(0, (0.0, 0.0, 1.0))
    assoc_b = _make_assoc(1, (0.3, 0.0, 0.0))
    tracker.update(_make_frame([assoc_a, assoc_b], frame_index=0))

    id_at_x0 = next(
        t.fish_id for t in tracker.get_all_tracks() if t.positions[-1][0] < 0.2
    )
    id_at_x03 = next(
        t.fish_id for t in tracker.get_all_tracks() if t.positions[-1][0] > 0.2
    )

    # Frame 1: same XY but Z is swapped (noisy Z)
    assoc_a2 = _make_assoc(0, (0.0, 0.0, 0.0))  # z swapped
    assoc_b2 = _make_assoc(1, (0.3, 0.0, 1.0))  # z swapped
    tracker.update(_make_frame([assoc_a2, assoc_b2], frame_index=1))

    confirmed = tracker.get_all_tracks()
    id_at_x0_frame1 = next(
        t.fish_id for t in confirmed if abs(t.positions[-1][0]) < 0.2
    )
    id_at_x03_frame1 = next(t.fish_id for t in confirmed if t.positions[-1][0] > 0.2)

    assert id_at_x0_frame1 == id_at_x0, "Fish at x=0 should keep its ID despite Z swap"
    assert id_at_x03_frame1 == id_at_x03, (
        "Fish at x=0.3 should keep its ID despite Z swap"
    )


# ---------------------------------------------------------------------------
# Test 8: Max-distance gate — no match when observation is too far
# ---------------------------------------------------------------------------


def test_max_distance_gate() -> None:
    """Observation > max_distance from all tracks creates a new track."""
    tracker = FishTracker(min_hits=1, max_age=7, max_distance=0.1)

    # Seed one track near origin
    assoc_near = _make_assoc(0, (0.0, 0.0, 0.5))
    tracker.update(_make_frame([assoc_near], frame_index=0))

    assert len(tracker.get_all_tracks()) == 1
    original_id = tracker.get_all_tracks()[0].fish_id

    # Frame 1: observation 1.0 m away in XY (>> max_distance=0.1 m)
    assoc_far = _make_assoc(0, (1.0, 0.0, 0.5))
    tracker.update(_make_frame([assoc_far], frame_index=1))

    all_tracks = tracker.get_all_tracks()
    # Original track missed → frames_since_update=1; new track created
    ids = {t.fish_id for t in all_tracks}
    assert len(ids) == 2, f"Expected 2 tracks, got {len(ids)}: {ids}"
    assert original_id in ids, "Original track should still exist (within grace)"


# ---------------------------------------------------------------------------
# Test 9: First-frame batch init — IDs sorted by X coordinate
# ---------------------------------------------------------------------------


def test_first_frame_batch_init() -> None:
    """9 associations on frame 0 create 9 tracks with IDs sorted by X."""
    tracker = FishTracker()

    # Create associations in shuffled X order
    x_values = [float(i) * 0.1 for i in [3, 7, 1, 9, 5, 0, 8, 2, 6]]
    associations = [_make_assoc(i, (x, 0.0, 0.5)) for i, x in enumerate(x_values)]

    tracker.update(_make_frame(associations, frame_index=0))

    tracks = tracker.get_all_tracks()
    assert len(tracks) == 9, f"Expected 9 tracks, got {len(tracks)}"

    # Extract X positions in ID order (ID 0 should have smallest X)
    id_to_x = {t.fish_id: float(t.positions[-1][0]) for t in tracks}
    sorted_x = [id_to_x[i] for i in range(9)]
    assert sorted_x == sorted(sorted_x), (
        f"Track IDs should be ordered by X, got x-values: {sorted_x}"
    )


# ---------------------------------------------------------------------------
# Test 10: get_seed_points — returns centroids or None
# ---------------------------------------------------------------------------


def test_get_seed_points() -> None:
    """get_seed_points() returns centroids of confirmed tracks; None if none confirmed."""
    tracker = FishTracker(min_hits=2, max_age=7)

    # Frame 0: no confirmed tracks yet → None
    assert tracker.get_seed_points() is None

    assoc = _make_assoc(0, (0.5, 0.2, 0.3))
    tracker.update(_make_frame([assoc], frame_index=0))

    # Still no confirmed tracks (min_hits=2, frame 0 gives streak=1)
    assert tracker.get_seed_points() is None

    tracker.update(_make_frame([assoc], frame_index=1))

    # Now confirmed
    seeds = tracker.get_seed_points()
    assert seeds is not None
    assert len(seeds) == 1
    np.testing.assert_allclose(seeds[0][:2], [0.5, 0.2], atol=1e-5)


def test_get_seed_points_multiple_confirmed() -> None:
    """get_seed_points returns one entry per confirmed track."""
    tracker = FishTracker(min_hits=2, max_age=7)

    assocs = [_make_assoc(i, (float(i) * 0.1, 0.0, 0.5)) for i in range(3)]

    tracker.update(_make_frame(assocs, frame_index=0))
    tracker.update(_make_frame(assocs, frame_index=1))

    seeds = tracker.get_seed_points()
    assert seeds is not None
    assert len(seeds) == 3
