"""Unit tests for TrackingStage — Stage 2 contract and carry-forward.

Tests verify:
- Empty detections produce an empty tracks_2d dict
- Single camera single fish produces a Tracklet2D with correct fields
- Multiple cameras are tracked independently
- CarryForward preserves tracker state between batches (same track IDs)
- Tracklet2D fields conform to spec (tuples, correct types)
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Minimal Detection and TrackingConfig stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeDet:
    """Minimal Detection stub (xywh bbox, confidence)."""

    bbox: tuple[int, int, int, int]
    confidence: float
    mask: None = None
    area: int = 0


@dataclass(frozen=True)
class _FakeTrackingConfig:
    """Minimal TrackingConfig stub for testing."""

    tracker_kind: str = "ocsort"
    max_coast_frames: int = 15
    n_init: int = 3
    iou_threshold: float = 0.3
    det_thresh: float = 0.3


def _make_det(
    x: int = 10,
    y: int = 10,
    w: int = 40,
    h: int = 40,
    conf: float = 0.9,
) -> _FakeDet:
    return _FakeDet(bbox=(x, y, w, h), confidence=conf, area=w * h)


def _make_stage() -> TrackingStage:  # noqa: F821
    from aquapose.core.tracking.stage import TrackingStage

    return TrackingStage(config=_FakeTrackingConfig())


MIN_HITS = 3  # matches _FakeTrackingConfig.n_init


def _make_context(
    camera_ids: list[str],
    detections: list[dict],
) -> PipelineContext:  # noqa: F821
    from aquapose.core.context import PipelineContext

    ctx = PipelineContext()
    ctx.camera_ids = camera_ids
    ctx.detections = detections
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmptyDetections:
    """TrackingStage with no detections produces empty tracks_2d."""

    def test_empty_detections_no_cameras_produces_empty_dict(self) -> None:
        """No cameras + empty detections -> tracks_2d is {}."""
        stage = _make_stage()
        ctx = _make_context(camera_ids=[], detections=[])
        result_ctx, _ = stage.run(ctx)
        assert result_ctx.tracks_2d == {}

    def test_cameras_with_empty_detections_produce_empty_lists(self) -> None:
        """Cameras with no detections produce empty tracklet lists."""
        stage = _make_stage()
        n_frames = MIN_HITS + 2
        # N frames, each with no detections for camera
        detections = [{"cam1": []} for _ in range(n_frames)]
        ctx = _make_context(camera_ids=["cam1"], detections=detections)
        result_ctx, _ = stage.run(ctx)
        assert result_ctx.tracks_2d is not None
        assert "cam1" in result_ctx.tracks_2d
        assert result_ctx.tracks_2d["cam1"] == []


class TestSingleCameraSingleFish:
    """TrackingStage with one camera and one consistent fish."""

    def test_single_camera_single_fish_produces_tracklet(self) -> None:
        """One fish visible for enough frames produces one Tracklet2D."""
        stage = _make_stage()
        n_frames = MIN_HITS + 5
        detections = [{"cam1": [_make_det()]} for _ in range(n_frames)]
        ctx = _make_context(camera_ids=["cam1"], detections=detections)
        result_ctx, _ = stage.run(ctx)
        assert result_ctx.tracks_2d is not None
        cam_tracklets = result_ctx.tracks_2d.get("cam1", [])
        assert len(cam_tracklets) == 1, f"Expected 1 tracklet, got {len(cam_tracklets)}"

    def test_tracklet_camera_id_is_set(self) -> None:
        """Tracklet2D.camera_id matches the source camera."""
        stage = _make_stage()
        n_frames = MIN_HITS + 2
        detections = [{"cam_a": [_make_det()]} for _ in range(n_frames)]
        ctx = _make_context(camera_ids=["cam_a"], detections=detections)
        result_ctx, _ = stage.run(ctx)
        cam_tracklets = result_ctx.tracks_2d.get("cam_a", [])
        assert len(cam_tracklets) == 1
        assert cam_tracklets[0].camera_id == "cam_a"

    def test_context_tracks_2d_is_populated(self) -> None:
        """context.tracks_2d is set (not None) after run()."""
        stage = _make_stage()
        ctx = _make_context(camera_ids=["cam1"], detections=[])
        result_ctx, _ = stage.run(ctx)
        assert result_ctx.tracks_2d is not None


class TestCarryForward:
    """CarryForward preserves tracker state between batches."""

    def test_carry_forward_preserves_track_ids(self) -> None:
        """Track IDs from batch 1 persist into batch 2."""
        stage = _make_stage()
        n_frames_batch1 = MIN_HITS + 3

        # Batch 1
        dets1 = [{"cam1": [_make_det()]} for _ in range(n_frames_batch1)]
        ctx1 = _make_context(camera_ids=["cam1"], detections=dets1)
        ctx1, carry1 = stage.run(ctx1)
        tracklets1 = ctx1.tracks_2d.get("cam1", [])
        assert len(tracklets1) == 1
        track_id_b1 = tracklets1[0].track_id

        # Batch 2 (same stage instance, using carry from batch 1)
        n_frames_batch2 = 5
        dets2 = [{"cam1": [_make_det()]} for _ in range(n_frames_batch2)]
        ctx2 = _make_context(camera_ids=["cam1"], detections=dets2)
        ctx2, _ = stage.run(ctx2, carry=carry1)
        tracklets2 = ctx2.tracks_2d.get("cam1", [])

        # The track should continue with the same local ID
        assert any(t.track_id == track_id_b1 for t in tracklets2), (
            f"Expected track_id {track_id_b1} to continue; got {[t.track_id for t in tracklets2]}"
        )

    def test_no_carry_creates_fresh_trackers(self) -> None:
        """run() with carry=None creates fresh trackers."""
        stage = _make_stage()
        n_frames = MIN_HITS + 2
        dets = [{"cam1": [_make_det()]} for _ in range(n_frames)]

        ctx1 = _make_context(camera_ids=["cam1"], detections=dets)
        ctx1, _ = stage.run(ctx1, carry=None)
        # Should produce a confirmed tracklet with a fresh local ID (starts at 0)
        tracklets = ctx1.tracks_2d.get("cam1", [])
        assert len(tracklets) == 1
        assert tracklets[0].track_id == 0

    def test_carry_returned_has_camera_state(self) -> None:
        """CarryForward returned by run() has per-camera state."""
        from aquapose.core.context import CarryForward

        stage = _make_stage()
        dets = [{"cam1": [_make_det()], "cam2": [_make_det()]}]
        ctx = _make_context(camera_ids=["cam1", "cam2"], detections=dets)
        _, carry = stage.run(ctx)

        assert isinstance(carry, CarryForward)
        assert "cam1" in carry.tracks_2d_state
        assert "cam2" in carry.tracks_2d_state


class TestMultipleCameras:
    """Multiple cameras are tracked independently."""

    def test_two_cameras_both_produce_tracks(self) -> None:
        """Two cameras with detections each produce their own tracklets."""
        stage = _make_stage()
        n_frames = MIN_HITS + 4
        dets = [
            {
                "cam_left": [_make_det(x=10, y=10)],
                "cam_right": [_make_det(x=100, y=100)],
            }
            for _ in range(n_frames)
        ]
        ctx = _make_context(camera_ids=["cam_left", "cam_right"], detections=dets)
        result_ctx, _ = stage.run(ctx)

        assert "cam_left" in result_ctx.tracks_2d
        assert "cam_right" in result_ctx.tracks_2d
        assert len(result_ctx.tracks_2d["cam_left"]) == 1
        assert len(result_ctx.tracks_2d["cam_right"]) == 1

    def test_camera_ids_are_independent(self) -> None:
        """Track IDs from different cameras do not collide (both start at 0)."""
        stage = _make_stage()
        n_frames = MIN_HITS + 2
        dets = [
            {
                "cam_a": [_make_det()],
                "cam_b": [_make_det()],
            }
            for _ in range(n_frames)
        ]
        ctx = _make_context(camera_ids=["cam_a", "cam_b"], detections=dets)
        result_ctx, _ = stage.run(ctx)

        tracklets_a = result_ctx.tracks_2d.get("cam_a", [])
        tracklets_b = result_ctx.tracks_2d.get("cam_b", [])
        assert len(tracklets_a) == 1
        assert len(tracklets_b) == 1
        # Track IDs within each camera start from 0 independently
        assert tracklets_a[0].camera_id == "cam_a"
        assert tracklets_b[0].camera_id == "cam_b"


class TestTrackletFieldsSpec:
    """Tracklet2D fields conform to the spec."""

    def test_tracklet_sequence_fields_are_tuples(self) -> None:
        """frames, centroids, bboxes, frame_status are all tuples."""
        stage = _make_stage()
        n_frames = MIN_HITS + 2
        dets = [{"cam1": [_make_det()]} for _ in range(n_frames)]
        ctx = _make_context(camera_ids=["cam1"], detections=dets)
        result_ctx, _ = stage.run(ctx)

        tracklets = result_ctx.tracks_2d.get("cam1", [])
        assert len(tracklets) == 1
        t = tracklets[0]
        assert isinstance(t.frames, tuple), f"frames is {type(t.frames)}"
        assert isinstance(t.centroids, tuple), f"centroids is {type(t.centroids)}"
        assert isinstance(t.bboxes, tuple), f"bboxes is {type(t.bboxes)}"
        assert isinstance(t.frame_status, tuple), (
            f"frame_status is {type(t.frame_status)}"
        )

    def test_tracklet_track_id_is_int(self) -> None:
        """Tracklet2D.track_id is an int."""
        stage = _make_stage()
        n_frames = MIN_HITS + 2
        dets = [{"cam1": [_make_det()]} for _ in range(n_frames)]
        ctx = _make_context(camera_ids=["cam1"], detections=dets)
        result_ctx, _ = stage.run(ctx)

        tracklets = result_ctx.tracks_2d.get("cam1", [])
        assert len(tracklets) == 1
        assert isinstance(tracklets[0].track_id, int)

    def test_tracklet_frames_are_sequential(self) -> None:
        """frames contains monotonically increasing frame indices."""
        stage = _make_stage()
        n_frames = MIN_HITS + 3
        dets = [{"cam1": [_make_det()]} for _ in range(n_frames)]
        ctx = _make_context(camera_ids=["cam1"], detections=dets)
        result_ctx, _ = stage.run(ctx)

        tracklets = result_ctx.tracks_2d.get("cam1", [])
        assert len(tracklets) == 1
        frames = tracklets[0].frames
        assert len(frames) > 0
        # Frames should be strictly increasing
        for i in range(1, len(frames)):
            assert frames[i] > frames[i - 1], f"Non-monotonic frames: {frames}"
