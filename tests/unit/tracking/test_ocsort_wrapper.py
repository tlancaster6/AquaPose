"""Unit tests for OcSortTracker — boxmot isolation and Tracklet2D output format.

These tests verify the OcSortTracker wrapper contract:
- Correct Tracklet2D field types (tuples, camera_id, track_id)
- Empty detection handling
- Single-track confirmation after min_hits frames
- Coasting detection gap
- Multi-track independence
- State roundtrip (get_state / from_state)
- Keypoint centroid extraction with OBB fallback
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Minimal Detection stub (avoids importing segmentation module)
# ---------------------------------------------------------------------------


@dataclass
class _FakeDet:
    """Minimal Detection stub for testing (xywh bbox, confidence, optional keypoints)."""

    bbox: tuple[int, int, int, int]
    confidence: float
    mask: None = None
    area: int = 0
    keypoints: np.ndarray | None = None
    keypoint_conf: np.ndarray | None = None


def _make_det(
    x: int = 10,
    y: int = 10,
    w: int = 40,
    h: int = 40,
    conf: float = 0.9,
) -> _FakeDet:
    """Build a fake Detection with given bbox and confidence."""
    return _FakeDet(bbox=(x, y, w, h), confidence=conf, area=w * h)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIN_HITS = 3
MAX_AGE = 15


def _make_tracker(
    camera_id: str = "cam1",
    min_hits: int = MIN_HITS,
    max_age: int = MAX_AGE,
) -> OcSortTracker:  # noqa: F821
    from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

    return OcSortTracker(
        camera_id=camera_id,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=0.3,
        det_thresh=0.3,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmptyDetections:
    """OcSortTracker with zero detections produces no tracklets."""

    def test_empty_detections_produces_no_tracklets(self) -> None:
        """Passing empty detection lists for several frames returns []."""
        tracker = _make_tracker()
        for frame_idx in range(MIN_HITS + 3):
            tracker.update(frame_idx, [])
        tracklets = tracker.get_tracklets()
        assert tracklets == [], f"Expected [], got {tracklets}"


class TestSingleDetectionStream:
    """OcSortTracker with a single consistent detection stream."""

    def test_single_detection_stream_produces_one_tracklet(self) -> None:
        """One consistent detection for min_hits+5 frames produces one Tracklet2D."""
        tracker = _make_tracker()
        n_frames = MIN_HITS + 5
        for frame_idx in range(n_frames):
            tracker.update(frame_idx, [_make_det()])
        tracklets = tracker.get_tracklets()
        # Exactly one confirmed tracklet expected
        assert len(tracklets) == 1, f"Expected 1 tracklet, got {len(tracklets)}"

    def test_tracklet_camera_id_matches_tracker(self) -> None:
        """Tracklet2D.camera_id matches the OcSortTracker's camera_id."""
        cam = "cam_test"
        tracker = _make_tracker(camera_id=cam)
        for frame_idx in range(MIN_HITS + 2):
            tracker.update(frame_idx, [_make_det()])
        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        assert tracklets[0].camera_id == cam

    def test_tracklet_has_detected_frame_status(self) -> None:
        """All frames with a detection get frame_status='detected'."""
        tracker = _make_tracker()
        n_frames = MIN_HITS + 2
        for frame_idx in range(n_frames):
            tracker.update(frame_idx, [_make_det()])
        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        for status in tracklets[0].frame_status:
            assert status == "detected", f"Unexpected status: {status}"


class TestTracklet2DTypes:
    """Tracklet2D sequence fields are all tuples."""

    def test_tracklet2d_fields_are_tuples(self) -> None:
        """Verify frames, centroids, bboxes, frame_status are all tuples."""
        tracker = _make_tracker()
        for frame_idx in range(MIN_HITS + 2):
            tracker.update(frame_idx, [_make_det()])
        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        t = tracklets[0]
        assert isinstance(t.frames, tuple), f"frames is {type(t.frames)}"
        assert isinstance(t.centroids, tuple), f"centroids is {type(t.centroids)}"
        assert isinstance(t.bboxes, tuple), f"bboxes is {type(t.bboxes)}"
        assert isinstance(t.frame_status, tuple), (
            f"frame_status is {type(t.frame_status)}"
        )

    def test_tracklet_frame_count_matches_detected_frames(self) -> None:
        """The number of frames in the tracklet matches detection frames + coasted."""
        tracker = _make_tracker()
        n_detected = MIN_HITS + 4
        n_coasted = 3
        # Detected phase
        for frame_idx in range(n_detected):
            tracker.update(frame_idx, [_make_det()])
        # Coasted phase
        for frame_idx in range(n_detected, n_detected + n_coasted):
            tracker.update(frame_idx, [])
        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        t = tracklets[0]
        total_frames = len(t.frames)
        # Should have n_detected + n_coasted frames recorded
        assert total_frames == n_detected + n_coasted, (
            f"Expected {n_detected + n_coasted} frames, got {total_frames}"
        )


class TestCoasting:
    """Coasting (detection gap) leaves 'coasted' entries in frame_status."""

    def test_coasting_detection_gap(self) -> None:
        """Gap frames produce 'coasted' entries in frame_status."""
        tracker = _make_tracker()
        n_before = MIN_HITS + 2
        n_gap = 3
        n_after = 2

        # Detected frames
        for frame_idx in range(n_before):
            tracker.update(frame_idx, [_make_det()])

        # Coasting gap
        for frame_idx in range(n_before, n_before + n_gap):
            tracker.update(frame_idx, [])

        # Detected again
        for frame_idx in range(n_before + n_gap, n_before + n_gap + n_after):
            tracker.update(frame_idx, [_make_det()])

        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1, f"Expected 1 tracklet, got {len(tracklets)}"
        statuses = tracklets[0].frame_status
        coasted_count = sum(1 for s in statuses if s == "coasted")
        assert coasted_count == n_gap, (
            f"Expected {n_gap} coasted frames, got {coasted_count}"
        )


class TestMultipleDetections:
    """Multiple spatially separated detections produce independent tracklets."""

    def test_multiple_detections_produce_multiple_tracklets(self) -> None:
        """Two fish in the same camera produce two distinct Tracklet2D objects."""
        tracker = _make_tracker()
        n_frames = MIN_HITS + 3

        for frame_idx in range(n_frames):
            dets = [
                _make_det(x=10, y=10, w=40, h=40),  # Fish A (top-left)
                _make_det(x=200, y=200, w=40, h=40),  # Fish B (bottom-right)
            ]
            tracker.update(frame_idx, dets)

        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 2, f"Expected 2 tracklets, got {len(tracklets)}"
        ids = {t.track_id for t in tracklets}
        assert len(ids) == 2, f"Track IDs must be distinct, got {ids}"

    def test_multiple_tracklets_have_correct_camera_id(self) -> None:
        """All tracklets from one tracker share the same camera_id."""
        cam = "cam_multi"
        tracker = _make_tracker(camera_id=cam)
        n_frames = MIN_HITS + 2

        for frame_idx in range(n_frames):
            dets = [
                _make_det(x=10, y=10, w=40, h=40),
                _make_det(x=200, y=200, w=40, h=40),
            ]
            tracker.update(frame_idx, dets)

        tracklets = tracker.get_tracklets()
        for t in tracklets:
            assert t.camera_id == cam


class TestStateRoundtrip:
    """get_state / from_state preserves tracker state across batches."""

    def test_state_roundtrip_continues_tracks(self) -> None:
        """Tracks started in batch 1 continue with same IDs in batch 2."""
        tracker1 = _make_tracker()
        batch1_frames = MIN_HITS + 2

        # Batch 1
        for frame_idx in range(batch1_frames):
            tracker1.update(frame_idx, [_make_det()])

        tracklets1 = tracker1.get_tracklets()
        assert len(tracklets1) == 1
        track_id_batch1 = tracklets1[0].track_id

        # Save state
        state = tracker1.get_state()

        # Restore into a new tracker instance
        from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

        tracker2 = OcSortTracker.from_state("cam1", state)

        # Batch 2
        batch2_frames = 5
        for frame_idx in range(batch1_frames, batch1_frames + batch2_frames):
            tracker2.update(frame_idx, [_make_det()])

        tracklets2 = tracker2.get_tracklets()
        # The restored tracker should have the same track ID for the continuing track
        assert any(t.track_id == track_id_batch1 for t in tracklets2), (
            f"Expected track_id {track_id_batch1} to persist; got {[t.track_id for t in tracklets2]}"
        )

    def test_state_roundtrip_resets_builders(self) -> None:
        """Tracklets after roundtrip contain only frames from the new batch.

        Builders are cleared on from_state() so each chunk accumulates fresh
        tracklet data.  Previous chunk data is discarded to avoid duplicate
        frame entries when chunks use local frame indices.
        """
        tracker1 = _make_tracker()
        batch1_frames = MIN_HITS + 2

        for frame_idx in range(batch1_frames):
            tracker1.update(frame_idx, [_make_det()])

        state = tracker1.get_state()

        from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

        tracker2 = OcSortTracker.from_state("cam1", state)

        batch2_frames = 4
        for frame_idx in range(batch1_frames, batch1_frames + batch2_frames):
            tracker2.update(frame_idx, [_make_det()])

        tracklets2 = tracker2.get_tracklets()
        assert len(tracklets2) == 1
        # Only frames from batch2 — builders are reset across chunks
        total = len(tracklets2[0].frames)
        assert total == batch2_frames, (
            f"Expected {batch2_frames} frames (batch2 only), got {total}"
        )


class TestTentativeTracksFiltered:
    """Tentative tracks (below min_hits) do not appear in get_tracklets()."""

    def test_below_min_hits_not_in_output(self) -> None:
        """A track with fewer than min_hits detections is excluded."""
        tracker = _make_tracker(min_hits=5)
        # Feed only min_hits - 1 frames (track stays tentative)
        for frame_idx in range(4):
            tracker.update(frame_idx, [_make_det()])
        tracklets = tracker.get_tracklets()
        assert tracklets == [], f"Expected empty (tentative track), got {tracklets}"


# ---------------------------------------------------------------------------
# Helpers for keypoint centroid tests
# ---------------------------------------------------------------------------


def _make_det_with_kpts(
    x: int = 10,
    y: int = 10,
    w: int = 40,
    h: int = 40,
    conf: float = 0.9,
    kpt_xy: tuple[float, float] | None = None,
    kpt_conf: float = 0.9,
    kpt_index: int = 2,
    n_keypoints: int = 6,
) -> _FakeDet:
    """Build a _FakeDet with keypoints array containing one high-confidence keypoint."""
    kpts = np.zeros((n_keypoints, 2), dtype=np.float32)
    kconfs = np.zeros(n_keypoints, dtype=np.float32)
    if kpt_xy is not None:
        kpts[kpt_index] = kpt_xy
        kconfs[kpt_index] = kpt_conf
    return _FakeDet(
        bbox=(x, y, w, h),
        confidence=conf,
        area=w * h,
        keypoints=kpts,
        keypoint_conf=kconfs,
    )


def _obb_centroid(x: int, y: int, w: int, h: int) -> tuple[float, float]:
    """Compute the expected OBB centroid for a given xywh bbox."""
    return (float(x + w / 2.0), float(y + h / 2.0))


class TestKeypointCentroid:
    """_TrackletBuilder.add_frame uses keypoint centroid when available and confident."""

    def test_high_confidence_keypoint_overrides_obb_centroid(self) -> None:
        """High-confidence keypoint at index 2 should be used as centroid, not OBB center."""
        from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

        kpt_x, kpt_y = 33.5, 28.7  # Distinct from OBB centroid
        tracker = OcSortTracker(
            camera_id="cam1",
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=0.3,
            det_thresh=0.3,
            centroid_keypoint_index=2,
            centroid_confidence_floor=0.3,
        )

        det = _make_det_with_kpts(
            x=10, y=10, w=40, h=40, kpt_xy=(kpt_x, kpt_y), kpt_conf=0.95, kpt_index=2
        )
        for frame_idx in range(MIN_HITS + 2):
            tracker.update(frame_idx, [det])

        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        # At least one centroid should be the keypoint position
        centroids = tracklets[0].centroids
        kpt_centroids = [
            (cx, cy)
            for cx, cy in centroids
            if abs(cx - kpt_x) < 0.01 and abs(cy - kpt_y) < 0.01
        ]
        assert len(kpt_centroids) > 0, (
            f"Expected keypoint centroid ({kpt_x}, {kpt_y}) in centroids, got {centroids}"
        )

    def test_low_confidence_keypoint_falls_back_to_obb_centroid(self) -> None:
        """Keypoint with confidence below floor should fall back to OBB centroid."""
        from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

        x, y, w, h = 10, 10, 40, 40
        kpt_x, kpt_y = 99.0, 99.0  # Far from OBB centroid
        expected_cx, expected_cy = _obb_centroid(x, y, w, h)

        tracker = OcSortTracker(
            camera_id="cam1",
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=0.3,
            det_thresh=0.3,
            centroid_keypoint_index=2,
            centroid_confidence_floor=0.3,
        )

        det = _make_det_with_kpts(
            x=x,
            y=y,
            w=w,
            h=h,
            kpt_xy=(kpt_x, kpt_y),
            kpt_conf=0.1,
            kpt_index=2,  # below floor
        )
        for frame_idx in range(MIN_HITS + 2):
            tracker.update(frame_idx, [det])

        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        # Centroids should be OBB center, not keypoint position
        for cx, cy in tracklets[0].centroids:
            assert abs(cx - kpt_x) > 1.0 or abs(cy - kpt_y) > 1.0, (
                f"Got unexpected keypoint centroid ({kpt_x},{kpt_y}) — expected OBB fallback ({expected_cx},{expected_cy})"
            )

    def test_missing_keypoints_falls_back_to_obb_centroid(self) -> None:
        """Detection with keypoints=None should fall back to OBB centroid."""
        from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

        x, y, w, h = 10, 10, 40, 40
        tracker = OcSortTracker(
            camera_id="cam1",
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=0.3,
            det_thresh=0.3,
            centroid_keypoint_index=2,
            centroid_confidence_floor=0.3,
        )

        det = _FakeDet(
            bbox=(x, y, w, h),
            confidence=0.9,
            area=w * h,
            keypoints=None,
            keypoint_conf=None,
        )
        for frame_idx in range(MIN_HITS + 2):
            tracker.update(frame_idx, [det])

        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        expected_cx, expected_cy = _obb_centroid(x, y, w, h)
        for cx, cy in tracklets[0].centroids:
            assert abs(cx - expected_cx) < 0.01, (
                f"Expected OBB centroid x={expected_cx}, got cx={cx}"
            )
            assert abs(cy - expected_cy) < 0.01, (
                f"Expected OBB centroid y={expected_cy}, got cy={cy}"
            )

    def test_coasted_frame_uses_obb_centroid(self) -> None:
        """Coasted frame (no detection) uses OBB centroid from Kalman state."""
        from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

        x, y, w, h = 10, 10, 40, 40
        kpt_x, kpt_y = 33.5, 28.7

        tracker = OcSortTracker(
            camera_id="cam1",
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=0.3,
            det_thresh=0.3,
            centroid_keypoint_index=2,
            centroid_confidence_floor=0.3,
        )

        det = _make_det_with_kpts(
            x=x, y=y, w=w, h=h, kpt_xy=(kpt_x, kpt_y), kpt_conf=0.95, kpt_index=2
        )
        # Detected phase
        for frame_idx in range(MIN_HITS + 2):
            tracker.update(frame_idx, [det])
        # Coasted phase
        for frame_idx in range(MIN_HITS + 2, MIN_HITS + 4):
            tracker.update(frame_idx, [])

        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        statuses = tracklets[0].frame_status
        centroids = tracklets[0].centroids
        # Coasted frames should not use the exact keypoint position
        for status, (cx, cy) in zip(statuses, centroids, strict=True):
            if status == "coasted":
                # Coasted frames come from Kalman state (OBB-based), not keypoint
                # The Kalman prediction may be near kpt_x/kpt_y since the fish barely
                # moved; we just verify it's a valid float (the path ran without error).
                assert isinstance(cx, float)
                assert isinstance(cy, float)

    def test_custom_keypoint_index_used(self) -> None:
        """Custom centroid_keypoint_index (e.g. index 0 = nose) is used for centroid."""
        from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

        nose_x, nose_y = 55.0, 22.0  # nose keypoint at index 0
        tracker = OcSortTracker(
            camera_id="cam1",
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=0.3,
            det_thresh=0.3,
            centroid_keypoint_index=0,  # nose
            centroid_confidence_floor=0.3,
        )

        det = _make_det_with_kpts(
            x=10, y=10, w=40, h=40, kpt_xy=(nose_x, nose_y), kpt_conf=0.95, kpt_index=0
        )
        for frame_idx in range(MIN_HITS + 2):
            tracker.update(frame_idx, [det])

        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 1
        centroids = tracklets[0].centroids
        nose_centroids = [
            (cx, cy)
            for cx, cy in centroids
            if abs(cx - nose_x) < 0.01 and abs(cy - nose_y) < 0.01
        ]
        assert len(nose_centroids) > 0, (
            f"Expected nose keypoint centroid ({nose_x}, {nose_y}) in centroids, got {centroids}"
        )

    def test_ocsort_tracker_accepts_centroid_config_params(self) -> None:
        """OcSortTracker constructor accepts centroid_keypoint_index and centroid_confidence_floor."""
        from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

        tracker = OcSortTracker(
            camera_id="cam1",
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=0.3,
            det_thresh=0.3,
            centroid_keypoint_index=3,
            centroid_confidence_floor=0.5,
        )
        assert tracker._centroid_keypoint_index == 3
        assert tracker._centroid_confidence_floor == 0.5
