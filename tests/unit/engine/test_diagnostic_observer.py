"""Unit tests for DiagnosticObserver."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aquapose.core.association.types import TrackletGroup
from aquapose.core.context import PipelineContext
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.core.tracking.types import Tracklet2D
from aquapose.core.types.detection import Detection
from aquapose.core.types.midline import Midline2D
from aquapose.engine.diagnostic_observer import DiagnosticObserver
from aquapose.engine.events import StageComplete
from aquapose.engine.observers import Observer


def test_diagnostic_observer_satisfies_protocol() -> None:
    """DiagnosticObserver satisfies the Observer protocol via isinstance check."""
    observer = DiagnosticObserver()
    assert isinstance(observer, Observer)


def test_captures_stage_output() -> None:
    """StageComplete with context populates observer.stages with a snapshot."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.detections = [[{"cam1": [1, 2]}]]
    ctx.frame_count = 1

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.5,
            context=ctx,
        )
    )

    assert "DetectionStage" in observer.stages
    snapshot = observer.stages["DetectionStage"]
    assert snapshot.detections is ctx.detections


def test_captures_multiple_stages() -> None:
    """Multiple StageComplete events populate separate snapshot entries."""
    observer = DiagnosticObserver()

    ctx1 = PipelineContext()
    ctx1.detections = [[{"cam1": [1]}]]
    ctx1.frame_count = 1

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.3,
            context=ctx1,
        )
    )

    ctx2 = PipelineContext()
    ctx2.detections = [[{"cam1": [1]}]]
    ctx2.annotated_detections = [[{"cam1": [{"midline": [1, 2]}]}]]
    ctx2.frame_count = 1

    observer.on_event(
        StageComplete(
            stage_name="MidlineStage",
            stage_index=1,
            elapsed_seconds=0.7,
            context=ctx2,
        )
    )

    assert len(observer.stages) == 2
    assert "DetectionStage" in observer.stages
    assert "MidlineStage" in observer.stages
    assert observer.stages["MidlineStage"].annotated_detections is not None


def test_snapshot_getitem() -> None:
    """StageSnapshot[frame_idx] returns dict of per-frame fields."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.detections = [{"cam1": [1]}, {"cam1": [2]}]
    ctx.frame_count = 2

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.1,
            context=ctx,
        )
    )

    snapshot = observer.stages["DetectionStage"]
    frame_0 = snapshot[0]
    assert "detections" in frame_0
    assert frame_0["detections"] == {"cam1": [1]}


def test_stores_references_not_copies() -> None:
    """Snapshot fields are the same objects as PipelineContext (identity check)."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.detections = [[{"cam1": [1]}]]
    ctx.frame_count = 1

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.2,
            context=ctx,
        )
    )

    assert observer.stages["DetectionStage"].detections is ctx.detections


def test_skips_if_no_context() -> None:
    """StageComplete without context field does not create a snapshot."""
    observer = DiagnosticObserver()

    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.1,
        )
    )

    assert len(observer.stages) == 0


def test_stage_timing_captured() -> None:
    """Snapshot preserves elapsed_seconds from the StageComplete event."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.frame_count = 0

    observer.on_event(
        StageComplete(
            stage_name="TrackingStubStage",
            stage_index=1,
            elapsed_seconds=1.23,
            context=ctx,
        )
    )

    assert observer.stages["TrackingStubStage"].elapsed_seconds == 1.23


def test_all_stages_captured_in_full_sequence() -> None:
    """Five StageComplete events produce five snapshot entries (v2.1 stage names)."""
    observer = DiagnosticObserver()
    stage_names = [
        "DetectionStage",
        "TrackingStage",
        "AssociationStage",
        "MidlineStage",
        "ReconstructionStage",
    ]

    ctx = PipelineContext()
    ctx.frame_count = 1
    ctx.detections = [[{}]]

    for i, name in enumerate(stage_names):
        observer.on_event(
            StageComplete(
                stage_name=name,
                stage_index=i,
                elapsed_seconds=float(i) * 0.5,
                context=ctx,
            )
        )

    assert len(observer.stages) == 5
    for name in stage_names:
        assert name in observer.stages


# ---------------------------------------------------------------------------
# Helpers for export_pipeline_diagnostics tests
# ---------------------------------------------------------------------------


def _make_tracklet2d(
    camera_id: str,
    track_id: int,
    frames: tuple[int, ...],
    centroids: tuple[tuple[float, float], ...] | None = None,
) -> Tracklet2D:
    """Create a Tracklet2D with default centroid (0, 0) at each frame."""
    if centroids is None:
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


def _make_group_with_consensus(
    fish_id: int,
    frames: tuple[int, ...],
    cam_ids: tuple[str, ...],
) -> TrackletGroup:
    """Create a TrackletGroup with synthetic consensus_centroids."""
    tracklets = tuple(
        _make_tracklet2d(cam_id, i, frames) for i, cam_id in enumerate(cam_ids)
    )
    consensus_centroids = tuple(
        (f, np.array([float(f), 0.0, 1.0], dtype=np.float64)) for f in frames
    )
    return TrackletGroup(
        fish_id=fish_id,
        tracklets=tracklets,
        confidence=0.9,
        per_frame_confidence=tuple(0.9 for _ in frames),
        consensus_centroids=consensus_centroids,
    )


def _fire_detection_stage(
    observer: DiagnosticObserver,
    detections: list,
    frame_count: int,
    camera_ids: list[str],
) -> None:
    """Simulate a DetectionStage StageComplete event."""
    ctx = PipelineContext()
    ctx.detections = detections
    ctx.frame_count = frame_count
    ctx.camera_ids = camera_ids
    observer.on_event(
        StageComplete(
            stage_name="DetectionStage",
            stage_index=0,
            elapsed_seconds=0.1,
            context=ctx,
        )
    )


def _fire_tracking_stage(
    observer: DiagnosticObserver,
    tracks_2d: dict,
    frame_count: int,
    camera_ids: list[str],
) -> None:
    """Simulate a TrackingStage StageComplete event."""
    ctx = PipelineContext()
    ctx.tracks_2d = tracks_2d
    ctx.frame_count = frame_count
    ctx.camera_ids = camera_ids
    observer.on_event(
        StageComplete(
            stage_name="TrackingStage",
            stage_index=1,
            elapsed_seconds=0.1,
            context=ctx,
        )
    )


def _fire_association_stage(
    observer: DiagnosticObserver,
    groups: list[TrackletGroup],
    frame_count: int = 1,
    camera_ids: list[str] | None = None,
) -> None:
    """Simulate an AssociationStage StageComplete event."""
    ctx = PipelineContext()
    ctx.tracklet_groups = groups
    ctx.frame_count = frame_count
    if camera_ids is not None:
        ctx.camera_ids = camera_ids
    observer.on_event(
        StageComplete(
            stage_name="AssociationStage",
            stage_index=2,
            elapsed_seconds=0.1,
            context=ctx,
        )
    )


# ---------------------------------------------------------------------------
# Tests for export_pipeline_diagnostics
# ---------------------------------------------------------------------------


def test_export_pipeline_diagnostics_all_sections(tmp_path: Path) -> None:
    """export_pipeline_diagnostics writes NPZ with all 5 sections."""
    observer = DiagnosticObserver()
    cam_ids = ["cam_a", "cam_b"]
    frames = (0, 1, 2)
    n_frames = len(frames)

    # Detection stage
    detections = [
        {"cam_a": [object(), object()], "cam_b": [object()]},
        {"cam_a": [object()], "cam_b": []},
        {"cam_a": [], "cam_b": [object(), object(), object()]},
    ]
    _fire_detection_stage(observer, detections, n_frames, cam_ids)

    # Tracking stage
    tracklet_a = _make_tracklet2d("cam_a", 0, frames)
    tracklet_b = _make_tracklet2d("cam_b", 1, frames)
    _fire_tracking_stage(
        observer,
        {"cam_a": [tracklet_a], "cam_b": [tracklet_b]},
        n_frames,
        cam_ids,
    )

    # Association stage
    group = _make_group_with_consensus(
        fish_id=0, frames=frames, cam_ids=("cam_a", "cam_b")
    )
    _fire_association_stage(observer, [group], n_frames, cam_ids)

    # Midline stage
    midline_a0 = _make_midline2d(0, "cam_a", 0)
    midline_b0 = _make_midline2d(0, "cam_b", 0)
    annotated = [
        {
            "cam_a": [_make_annotated_detection(midline_a0, tracklet_a.centroids[0])],
            "cam_b": [_make_annotated_detection(midline_b0, tracklet_b.centroids[0])],
        },
        {
            "cam_a": [_make_annotated_detection(None, tracklet_a.centroids[1])],
            "cam_b": [_make_annotated_detection(None, tracklet_b.centroids[1])],
        },
        {
            "cam_a": [],
            "cam_b": [],
        },
    ]
    _fire_midline_stage(observer, annotated, n_frames, cam_ids)

    out_path = tmp_path / "pipeline_diagnostics.npz"
    result = observer.export_pipeline_diagnostics(out_path)

    assert result.exists()
    data = np.load(str(result), allow_pickle=True)

    # --- Section 1: tracking ---
    assert data["tracking/camera_ids"].shape == (6,)  # 2 tracklets * 3 frames
    assert data["tracking/track_ids"].shape == (6,)
    assert data["tracking/frame_indices"].shape == (6,)
    assert data["tracking/centroids"].shape == (6, 2)
    assert data["tracking/frame_status"].shape == (6,)

    # --- Section 2: groups ---
    assert data["groups/fish_ids"].shape == (1,)
    assert data["groups/n_cameras"].tolist() == [2]
    assert data["groups/n_frames"].tolist() == [3]
    assert data["groups/confidence"].shape == (1,)
    assert "cam_a" in str(data["groups/camera_ids_csv"][0])

    # --- Section 3: correspondences ---
    n_corr = n_frames * 2  # 3 frames * 2 cameras
    assert data["correspondences/fish_ids"].shape == (n_corr,)
    assert data["correspondences/points_3d"].shape == (n_corr, 3)
    assert data["correspondences/centroids_2d"].shape == (n_corr, 2)

    # --- Section 4: detection_counts ---
    assert data["detection_counts/matrix"].shape == (n_frames, 2)
    assert data["detection_counts/camera_ids"].shape == (2,)
    assert data["detection_counts/frame_indices"].shape == (n_frames,)
    # Frame 0: cam_a=2, cam_b=1
    assert data["detection_counts/matrix"][0, 0] == 2
    assert data["detection_counts/matrix"][0, 1] == 1

    # --- Section 5: midline_counts ---
    assert data["midline_counts/matrix"].shape == (n_frames, 2)
    # Frame 0: both cameras have 1 midline each
    assert data["midline_counts/matrix"][0, 0] == 1
    assert data["midline_counts/matrix"][0, 1] == 1
    # Frame 1: no midlines (both are None)
    assert data["midline_counts/matrix"][1, 0] == 0
    assert data["midline_counts/matrix"][1, 1] == 0


def test_export_pipeline_diagnostics_empty_observer(tmp_path: Path) -> None:
    """export_pipeline_diagnostics works with no stage snapshots (all empty)."""
    observer = DiagnosticObserver()
    out_path = tmp_path / "pipeline_diagnostics.npz"
    result = observer.export_pipeline_diagnostics(out_path)

    assert result.exists()
    data = np.load(str(result), allow_pickle=True)

    # All sections should have zero-length arrays
    assert data["tracking/camera_ids"].shape == (0,)
    assert data["tracking/centroids"].shape == (0, 2)
    assert data["groups/fish_ids"].shape == (0,)
    assert data["correspondences/fish_ids"].shape == (0,)
    assert data["correspondences/points_3d"].shape == (0, 3)
    assert data["detection_counts/matrix"].shape == (0, 0)
    assert data["midline_counts/matrix"].shape == (0, 0)


def test_export_pipeline_diagnostics_correspondences_skips_none_consensus(
    tmp_path: Path,
) -> None:
    """Groups with consensus_centroids=None produce no correspondence rows."""
    observer = DiagnosticObserver()
    frames = (0, 1)
    group = TrackletGroup(
        fish_id=0,
        tracklets=(
            _make_tracklet2d("cam_a", 0, frames),
            _make_tracklet2d("cam_b", 1, frames),
        ),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group])

    out_path = tmp_path / "pipeline_diagnostics.npz"
    result = observer.export_pipeline_diagnostics(out_path)

    data = np.load(str(result), allow_pickle=True)
    assert data["correspondences/fish_ids"].shape == (0,)
    assert data["correspondences/points_3d"].shape == (0, 3)


# ---------------------------------------------------------------------------
# Tests for MidlineFixture dataclass (Task 1)
# ---------------------------------------------------------------------------


def test_midline_fixture_importable() -> None:
    """MidlineFixture is importable from aquapose.io.midline_fixture."""
    from aquapose.io.midline_fixture import NPZ_VERSION, MidlineFixture  # noqa: F401

    assert NPZ_VERSION == "2.0"


def test_midline_fixture_is_frozen_dataclass() -> None:
    """MidlineFixture is a frozen dataclass with the correct fields."""
    from aquapose.io.midline_fixture import MidlineFixture

    fixture = MidlineFixture(
        frames=(),
        frame_indices=(),
        camera_ids=(),
        metadata={},
    )
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        fixture.frames = ()  # type: ignore[misc]


def test_snapshot_has_tracks_2d_and_tracklet_groups_fields() -> None:
    """StageSnapshot has tracks_2d and tracklet_groups fields (v2.1)."""
    observer = DiagnosticObserver()
    ctx = PipelineContext()
    ctx.frame_count = 1
    ctx.tracks_2d = {"cam1": []}
    ctx.tracklet_groups = []

    observer.on_event(
        StageComplete(
            stage_name="TrackingStubStage",
            stage_index=1,
            elapsed_seconds=0.1,
            context=ctx,
        )
    )

    snapshot = observer.stages["TrackingStubStage"]
    assert snapshot.tracks_2d is ctx.tracks_2d
    assert snapshot.tracklet_groups is ctx.tracklet_groups


# ---------------------------------------------------------------------------
# Helpers for export_midline_fixtures tests (Task 2)
# ---------------------------------------------------------------------------


def _make_midline2d(
    fish_id: int,
    camera_id: str,
    frame_index: int,
    n_points: int = 5,
) -> Midline2D:
    """Create a synthetic Midline2D for test use."""
    points = np.arange(n_points * 2, dtype=np.float32).reshape(n_points, 2)
    half_widths = np.ones(n_points, dtype=np.float32) * 2.0
    return Midline2D(
        points=points,
        half_widths=half_widths,
        fish_id=fish_id,
        camera_id=camera_id,
        frame_index=frame_index,
        is_head_to_tail=True,
        point_confidence=np.ones(n_points, dtype=np.float32) * 0.9,
    )


def _make_annotated_detection(
    midline: Midline2D | None,
    centroid: tuple[float, float],
) -> AnnotatedDetection:
    """Build an AnnotatedDetection whose Detection centroid matches given centroid."""
    cx, cy = centroid
    # bbox (x, y, w, h) such that centre is (cx, cy)
    bbox = (int(cx - 5), int(cy - 5), 10, 10)
    det = Detection(bbox=bbox, mask=None, area=100, confidence=0.95)
    return AnnotatedDetection(detection=det, midline=midline)


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


# ---------------------------------------------------------------------------
# Tests for export_midline_fixtures (Task 2)
# ---------------------------------------------------------------------------


def test_export_midline_fixtures_writes_npz(tmp_path: Path) -> None:
    """export_midline_fixtures writes NPZ with expected keys for captured midlines."""
    observer = DiagnosticObserver()
    fish_id = 0
    cam_a, cam_b = "cam_a", "cam_b"
    frames = (0, 1)

    # Build tracklet group: fish_id=0 observed in cam_a and cam_b for frames 0,1
    tracklet_a = _make_tracklet2d(cam_a, 0, frames)
    tracklet_b = _make_tracklet2d(cam_b, 1, frames)
    group = TrackletGroup(
        fish_id=fish_id,
        tracklets=(tracklet_a, tracklet_b),
        confidence=0.9,
        per_frame_confidence=(0.9, 0.9),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group])

    # Build annotated detections: frame 0 has midlines in cam_a and cam_b
    #  frame 1 has midline only in cam_a (cam_b returns None)
    midline_a0 = _make_midline2d(fish_id, cam_a, 0)
    midline_b0 = _make_midline2d(fish_id, cam_b, 0)
    midline_a1 = _make_midline2d(fish_id, cam_a, 1)

    # annotated_detections is list[dict[str, list[AnnotatedDetection]]]
    centroid_a0 = tracklet_a.centroids[0]
    centroid_b0 = tracklet_b.centroids[0]
    centroid_a1 = tracklet_a.centroids[1]
    centroid_b1 = tracklet_b.centroids[1]

    frame0_annot = {
        cam_a: [_make_annotated_detection(midline_a0, centroid_a0)],
        cam_b: [_make_annotated_detection(midline_b0, centroid_b0)],
    }
    frame1_annot = {
        cam_a: [_make_annotated_detection(midline_a1, centroid_a1)],
        cam_b: [_make_annotated_detection(None, centroid_b1)],
    }
    _fire_midline_stage(observer, [frame0_annot, frame1_annot], 2, [cam_a, cam_b])

    out_path = tmp_path / "midline_fixtures.npz"
    result = observer.export_midline_fixtures(out_path)

    assert result.exists()
    data = np.load(str(result), allow_pickle=True)
    # Check meta keys
    assert "meta/version" in data
    assert "meta/frame_indices" in data
    # Check midline key for frame 0, fish 0, cam_a
    key = f"midline/0/{fish_id}/{cam_a}/points"
    assert key in data
    assert data[key].shape[1] == 2


def test_export_midline_fixtures_raises_without_snapshots(tmp_path: Path) -> None:
    """ValueError raised when required snapshots are missing."""
    import pytest

    observer = DiagnosticObserver()
    out_path = tmp_path / "out.npz"

    # Neither AssociationStage nor MidlineStage snapshot fired
    with pytest.raises(ValueError, match="AssociationStage"):
        observer.export_midline_fixtures(out_path)


def test_export_midline_fixtures_includes_all_frames(tmp_path: Path) -> None:
    """Frames with even 1 midline are included (no min_cameras filter)."""
    observer = DiagnosticObserver()
    fish_id = 0
    cam_a = "cam_a"
    frames = (0, 1, 2)

    # Single-camera tracklet group
    tracklet_a = _make_tracklet2d(cam_a, 0, frames)
    group = TrackletGroup(
        fish_id=fish_id,
        tracklets=(tracklet_a,),
        confidence=0.9,
        per_frame_confidence=(0.9, 0.9, 0.9),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group])

    # Each frame has 1 midline from cam_a only
    annotated: list[dict] = []
    for fidx, frame_idx in enumerate(frames):
        midline = _make_midline2d(fish_id, cam_a, frame_idx)
        centroid = tracklet_a.centroids[fidx]
        annotated.append({cam_a: [_make_annotated_detection(midline, centroid)]})

    _fire_midline_stage(observer, annotated, len(frames), [cam_a])

    out_path = tmp_path / "midline_fixtures.npz"
    result = observer.export_midline_fixtures(out_path)
    data = np.load(str(result), allow_pickle=True)

    frame_indices = data["meta/frame_indices"]
    assert len(frame_indices) == 3, "All 3 single-camera frames should be included"


def test_on_pipeline_complete_exports_midline_fixtures(tmp_path: Path) -> None:
    """PipelineComplete auto-exports midline_fixtures.npz to output_dir."""
    from aquapose.engine.events import PipelineComplete

    observer = DiagnosticObserver(output_dir=tmp_path)
    fish_id = 0
    cam_a = "cam_a"
    frames = (0,)

    tracklet_a = _make_tracklet2d(cam_a, 0, frames)
    group = TrackletGroup(
        fish_id=fish_id,
        tracklets=(tracklet_a,),
        confidence=0.9,
        per_frame_confidence=(0.9,),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group])

    midline_a0 = _make_midline2d(fish_id, cam_a, 0)
    centroid_a0 = tracklet_a.centroids[0]
    frame0_annot = {cam_a: [_make_annotated_detection(midline_a0, centroid_a0)]}
    _fire_midline_stage(observer, [frame0_annot], 1, [cam_a])

    observer.on_event(PipelineComplete())

    out_file = tmp_path / "midline_fixtures.npz"
    assert out_file.exists(), (
        "midline_fixtures.npz should be auto-exported on PipelineComplete"
    )

    diag_file = tmp_path / "pipeline_diagnostics.npz"
    assert diag_file.exists(), (
        "pipeline_diagnostics.npz should be auto-exported on PipelineComplete"
    )


# ---------------------------------------------------------------------------
# Helpers for v2.0 export tests (Task 2 of 41-01)
# ---------------------------------------------------------------------------


class _MockProjectionModel:
    """Minimal mock matching RefractiveProjectionModel attribute structure."""

    def __init__(
        self,
        cam_id: str,
        offset: float = 0.0,
    ) -> None:
        import torch

        self.K = torch.eye(3, dtype=torch.float32) * (offset + 1.0)
        self.R = torch.eye(3, dtype=torch.float32)
        self.t = torch.tensor([offset, 0.0, 0.0], dtype=torch.float32)
        self.water_z = -0.5
        self.normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        self.n_air = 1.0
        self.n_water = 1.333


def _make_models_dict(cam_ids: tuple[str, ...]) -> dict[str, _MockProjectionModel]:
    """Build a mock models dict keyed by camera_id."""
    return {
        cam_id: _MockProjectionModel(cam_id, float(i))
        for i, cam_id in enumerate(cam_ids)
    }


def _make_full_fixture_observer(
    cam_ids: tuple[str, ...],
    frames: tuple[int, ...],
    fish_id: int = 0,
) -> tuple[DiagnosticObserver, list]:
    """Create observer and annotated_detections list for a full fixture scenario."""
    observer = DiagnosticObserver()

    tracklets = tuple(
        _make_tracklet2d(cam_id, i, frames) for i, cam_id in enumerate(cam_ids)
    )
    group = TrackletGroup(
        fish_id=fish_id,
        tracklets=tracklets,
        confidence=0.9,
        per_frame_confidence=tuple(0.9 for _ in frames),
        consensus_centroids=None,
    )
    _fire_association_stage(observer, [group])

    annotated: list[dict] = []
    for fidx, frame_idx in enumerate(frames):
        frame_annot: dict = {}
        for ti, cam_id in enumerate(cam_ids):
            midline = _make_midline2d(fish_id, cam_id, frame_idx)
            centroid = tracklets[ti].centroids[fidx]
            frame_annot[cam_id] = [_make_annotated_detection(midline, centroid)]
        annotated.append(frame_annot)

    _fire_midline_stage(observer, annotated, len(frames), list(cam_ids))
    return observer, annotated


# ---------------------------------------------------------------------------
# Tests for export_midline_fixtures v2.0 (Task 2 of 41-01)
# ---------------------------------------------------------------------------


def test_export_without_models_writes_v10(tmp_path: Path) -> None:
    """export_midline_fixtures with models=None writes version 1.0 (backward compat)."""
    cam_ids = ("cam_a", "cam_b")
    observer, _ = _make_full_fixture_observer(cam_ids=cam_ids, frames=(0,))

    out_path = tmp_path / "fixture.npz"
    observer.export_midline_fixtures(out_path)

    data = np.load(str(out_path), allow_pickle=True)
    assert str(data["meta/version"]) == "1.0"
    # No calib/ keys should be written
    calib_keys = [k for k in data.files if k.startswith("calib/")]
    assert len(calib_keys) == 0


def test_export_with_models_writes_v20(tmp_path: Path) -> None:
    """export_midline_fixtures with models dict writes version 2.0."""
    cam_ids = ("cam_a", "cam_b")
    observer, _ = _make_full_fixture_observer(cam_ids=cam_ids, frames=(0,))
    models = _make_models_dict(cam_ids)

    out_path = tmp_path / "fixture.npz"
    observer.export_midline_fixtures(out_path, models=models)

    data = np.load(str(out_path), allow_pickle=True)
    assert str(data["meta/version"]) == "2.0"


def test_export_with_models_writes_calib_keys(tmp_path: Path) -> None:
    """export_midline_fixtures with models writes expected calib/ keys."""
    cam_ids = ("cam_a", "cam_b")
    observer, _ = _make_full_fixture_observer(cam_ids=cam_ids, frames=(0,))
    models = _make_models_dict(cam_ids)

    out_path = tmp_path / "fixture.npz"
    observer.export_midline_fixtures(out_path, models=models)

    data = np.load(str(out_path), allow_pickle=True)

    # Shared keys
    assert "calib/water_z" in data
    assert "calib/n_air" in data
    assert "calib/n_water" in data
    assert "calib/interface_normal" in data
    assert data["calib/interface_normal"].shape == (3,)

    # Per-camera keys
    for cam_id in cam_ids:
        assert f"calib/{cam_id}/K_new" in data
        assert data[f"calib/{cam_id}/K_new"].shape == (3, 3)
        assert f"calib/{cam_id}/R" in data
        assert data[f"calib/{cam_id}/R"].shape == (3, 3)
        assert f"calib/{cam_id}/t" in data
        assert data[f"calib/{cam_id}/t"].shape == (3,)


def test_export_with_models_round_trip(tmp_path: Path) -> None:
    """export with models -> load_midline_fixture -> CalibBundle matches originals."""
    import torch

    from aquapose.io.midline_fixture import load_midline_fixture

    cam_ids = ("cam_a", "cam_b")
    observer, _ = _make_full_fixture_observer(cam_ids=cam_ids, frames=(0,))
    models = _make_models_dict(cam_ids)

    out_path = tmp_path / "fixture.npz"
    observer.export_midline_fixtures(out_path, models=models)

    fixture = load_midline_fixture(out_path)
    cb = fixture.calib_bundle

    assert cb is not None
    assert set(cb.camera_ids) == set(cam_ids)
    assert abs(cb.water_z - (-0.5)) < 1e-6
    assert abs(cb.n_air - 1.0) < 1e-6
    assert abs(cb.n_water - 1.333) < 1e-4

    for i, cam_id in enumerate(cam_ids):
        expected_K = (torch.eye(3, dtype=torch.float32) * float(i + 1)).numpy()
        np.testing.assert_allclose(cb.K_new[cam_id], expected_K, atol=1e-6)
        expected_t = np.array([float(i), 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(cb.t[cam_id], expected_t, atol=1e-6)
