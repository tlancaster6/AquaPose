"""Interface tests for ReconstructionStage -- Stage 5 of the AquaPose pipeline.

Validates:
- ReconstructionStage satisfies the Stage Protocol via structural typing
- run() correctly uses tracklet_groups for camera membership (no RANSAC)
- run() produces empty midlines_3d when tracklet_groups is empty (stub path)
- min_cameras enforcement: frames below threshold are dropped
- Gap interpolation: short gaps are filled with confidence=0
- Gap > max_interp_gap: NOT interpolated (stays dropped)
- Triangulation backend delegates to triangulate_midlines()
- Curve optimizer backend delegates to CurveOptimizer.optimize_midlines()
- Backend selection via "triangulation" and "curve_optimizer" strings
- Backend registry raises ValueError for unknown kinds
- Import boundary (ENG-07): no engine/ runtime imports in core/reconstruction/
- All 5 stages importable from core/
- build_stages() returns 5 ordered Stage instances
- PosePipeline instantiable with build_stages(config)
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.core.association.types import TrackletGroup
from aquapose.core.context import PipelineContext, Stage
from aquapose.core.reconstruction import ReconstructionStage
from aquapose.core.reconstruction.backends import get_backend
from aquapose.core.tracking.types import Tracklet2D
from aquapose.reconstruction.midline import Midline2D
from aquapose.reconstruction.triangulation import Midline3D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_midline2d(
    fish_id: int = 0,
    camera_id: str = "cam1",
    frame_index: int = 0,
    n_pts: int = 15,
) -> Midline2D:
    """Create a synthetic Midline2D with uniform points."""
    points = np.linspace([10.0, 10.0], [100.0, 100.0], n_pts).astype(np.float32)
    half_widths = np.full(n_pts, 3.0, dtype=np.float32)
    return Midline2D(
        points=points,
        half_widths=half_widths,
        fish_id=fish_id,
        camera_id=camera_id,
        frame_index=frame_index,
    )


def _make_midline3d(
    fish_id: int = 0,
    frame_index: int = 0,
    n_ctrl: int = 7,
) -> Midline3D:
    """Create a synthetic Midline3D for test assertions."""
    rng = np.random.RandomState(fish_id * 100 + frame_index)
    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=rng.randn(n_ctrl, 3).astype(np.float32),
        knots=np.linspace(0, 1, n_ctrl + 4).astype(np.float32),
        degree=3,
        arc_length=float(rng.uniform(10, 50)),
        half_widths=np.full(15, 2.0, dtype=np.float32),
        n_cameras=4,
        mean_residual=1.5,
        max_residual=5.0,
        is_low_confidence=False,
    )


def _make_annotated_detection(
    midline: Midline2D | None,
    centroid: tuple[float, float] = (50.0, 50.0),
) -> SimpleNamespace:
    """Create a mock AnnotatedDetection with a midline and detection centroid."""
    det = SimpleNamespace(centroid=centroid)
    return SimpleNamespace(midline=midline, detection=det)


def _make_tracklet(
    camera_id: str,
    track_id: int,
    frames: tuple[int, ...],
    centroids: tuple | None = None,
    frame_status: tuple | None = None,
) -> Tracklet2D:
    """Create a Tracklet2D with given camera/frames."""
    if centroids is None:
        centroids = tuple((50.0, 50.0) for _ in frames)
    if frame_status is None:
        frame_status = tuple("detected" for _ in frames)
    bboxes = tuple((40.0, 40.0, 20.0, 20.0) for _ in frames)
    return Tracklet2D(
        camera_id=camera_id,
        track_id=track_id,
        frames=frames,
        centroids=centroids,
        bboxes=bboxes,
        frame_status=frame_status,
    )


def _build_stage(
    tmp_path: Path,
    mock_backend: MagicMock | None = None,
    min_cameras: int = 3,
    max_interp_gap: int = 5,
) -> ReconstructionStage:
    """Build a ReconstructionStage with mocked backend (no real calibration)."""
    calib_file = tmp_path / "calibration.json"
    calib_file.write_text("{}")

    stage = ReconstructionStage.__new__(ReconstructionStage)
    stage._calibration_path = calib_file
    stage._min_cameras = min_cameras
    stage._max_interp_gap = max_interp_gap
    stage._n_control_points = 7

    if mock_backend is not None:
        stage._backend = mock_backend
    else:
        default_backend = MagicMock()
        default_backend.reconstruct_frame = MagicMock(return_value={})
        stage._backend = default_backend

    return stage


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_reconstruction_stage_satisfies_protocol(tmp_path: Path) -> None:
    """ReconstructionStage is a Stage (structural typing) even before run()."""
    stage = _build_stage(tmp_path)
    assert isinstance(stage, Stage), (
        "ReconstructionStage must satisfy the Stage Protocol via structural typing"
    )


# ---------------------------------------------------------------------------
# Tracklet-group-driven reconstruction
# ---------------------------------------------------------------------------


def test_reconstruction_with_tracklet_groups(tmp_path: Path) -> None:
    """run() uses tracklet_groups camera membership to build MidlineSets."""
    ml1 = _make_midline2d(fish_id=0, camera_id="cam1")
    ml2 = _make_midline2d(fish_id=0, camera_id="cam2")
    ml3 = _make_midline2d(fish_id=0, camera_id="cam3")

    expected_m3d = _make_midline3d(fish_id=0, frame_index=0)

    mock_backend = MagicMock()
    mock_backend.reconstruct_frame.return_value = {0: expected_m3d}

    stage = _build_stage(tmp_path, mock_backend=mock_backend, min_cameras=3)

    # Build annotated detections: 3 cameras, 2 frames
    ann1 = _make_annotated_detection(ml1, centroid=(50.0, 50.0))
    ann2 = _make_annotated_detection(ml2, centroid=(50.0, 50.0))
    ann3 = _make_annotated_detection(ml3, centroid=(50.0, 50.0))

    annotated = [
        {"cam1": [ann1], "cam2": [ann2], "cam3": [ann3]},
        {"cam1": [ann1], "cam2": [ann2], "cam3": [ann3]},
    ]

    # Build tracklet group with 3 cameras, frames 0-1
    t1 = _make_tracklet("cam1", 1, (0, 1))
    t2 = _make_tracklet("cam2", 2, (0, 1))
    t3 = _make_tracklet("cam3", 3, (0, 1))
    group = TrackletGroup(fish_id=0, tracklets=(t1, t2, t3))

    ctx = PipelineContext()
    ctx.frame_count = 2
    ctx.tracklet_groups = [group]
    ctx.annotated_detections = annotated

    result = stage.run(ctx)

    assert result is ctx
    assert ctx.midlines_3d is not None
    assert len(ctx.midlines_3d) == 2
    assert 0 in ctx.midlines_3d[0]
    mock_backend.reconstruct_frame.assert_called()


def test_reconstruction_two_fish(tmp_path: Path) -> None:
    """run() with 2 fish produces per-fish results in midlines_3d."""
    m3d_0 = _make_midline3d(fish_id=0, frame_index=0)
    m3d_1 = _make_midline3d(fish_id=1, frame_index=0)

    mock_backend = MagicMock()
    mock_backend.reconstruct_frame.side_effect = lambda **kw: {
        next(iter(kw["midline_set"].keys())): (
            m3d_0 if 0 in kw["midline_set"] else m3d_1
        )
    }

    stage = _build_stage(tmp_path, mock_backend=mock_backend, min_cameras=3)

    ml = _make_midline2d()
    ann = _make_annotated_detection(ml, centroid=(50.0, 50.0))
    annotated = [{"cam1": [ann, ann], "cam2": [ann], "cam3": [ann]}]

    t_a1 = _make_tracklet("cam1", 1, (0,))
    t_a2 = _make_tracklet("cam2", 2, (0,))
    t_a3 = _make_tracklet("cam3", 3, (0,))
    group_a = TrackletGroup(fish_id=0, tracklets=(t_a1, t_a2, t_a3))

    t_b1 = _make_tracklet("cam1", 4, (0,))
    t_b2 = _make_tracklet("cam2", 5, (0,))
    t_b3 = _make_tracklet("cam3", 6, (0,))
    group_b = TrackletGroup(fish_id=1, tracklets=(t_b1, t_b2, t_b3))

    ctx = PipelineContext()
    ctx.frame_count = 1
    ctx.tracklet_groups = [group_a, group_b]
    ctx.annotated_detections = annotated

    stage.run(ctx)

    assert len(ctx.midlines_3d) == 1
    # Both fish should be present in frame 0
    assert 0 in ctx.midlines_3d[0]
    assert 1 in ctx.midlines_3d[0]


# ---------------------------------------------------------------------------
# min_cameras enforcement
# ---------------------------------------------------------------------------


def test_min_cameras_drops_insufficient_views(tmp_path: Path) -> None:
    """Fish with fewer cameras than min_cameras produces dropped frame."""
    mock_backend = MagicMock()
    mock_backend.reconstruct_frame.return_value = {}
    stage = _build_stage(tmp_path, mock_backend=mock_backend, min_cameras=3)

    ml = _make_midline2d()
    ann = _make_annotated_detection(ml, centroid=(50.0, 50.0))
    annotated = [{"cam1": [ann], "cam2": [ann]}]

    # Only 2 cameras -> below min_cameras=3
    t1 = _make_tracklet("cam1", 1, (0,))
    t2 = _make_tracklet("cam2", 2, (0,))
    group = TrackletGroup(fish_id=0, tracklets=(t1, t2))

    ctx = PipelineContext()
    ctx.frame_count = 1
    ctx.tracklet_groups = [group]
    ctx.annotated_detections = annotated

    stage.run(ctx)

    assert len(ctx.midlines_3d) == 1
    assert ctx.midlines_3d[0] == {}  # No fish reconstructed
    mock_backend.reconstruct_frame.assert_not_called()


# ---------------------------------------------------------------------------
# Gap interpolation
# ---------------------------------------------------------------------------


def test_gap_interpolation_short_gap(tmp_path: Path) -> None:
    """Short gap (<=max_interp_gap) is interpolated with is_low_confidence=True."""
    m3d_0 = _make_midline3d(fish_id=0, frame_index=0)
    m3d_5 = _make_midline3d(fish_id=0, frame_index=5)

    mock_backend = MagicMock()

    def reconstruct_side_effect(**kw: object) -> dict:
        fi = kw["frame_idx"]
        if fi == 0:
            return {0: m3d_0}
        if fi == 5:
            return {0: m3d_5}
        return {}

    mock_backend.reconstruct_frame.side_effect = reconstruct_side_effect

    # max_interp_gap=5, so gap of 4 (frames 1-4) should be interpolated
    stage = _build_stage(
        tmp_path, mock_backend=mock_backend, min_cameras=3, max_interp_gap=5
    )

    ml = _make_midline2d()
    ann = _make_annotated_detection(ml, centroid=(50.0, 50.0))
    # Only frames 0 and 5 have 3 cameras; frames 1-4 have 0 cameras
    annotated = []
    for i in range(6):
        if i in (0, 5):
            annotated.append({"cam1": [ann], "cam2": [ann], "cam3": [ann]})
        else:
            annotated.append({})

    t1 = _make_tracklet("cam1", 1, (0, 5))
    t2 = _make_tracklet("cam2", 2, (0, 5))
    t3 = _make_tracklet("cam3", 3, (0, 5))
    group = TrackletGroup(fish_id=0, tracklets=(t1, t2, t3))

    ctx = PipelineContext()
    ctx.frame_count = 6
    ctx.tracklet_groups = [group]
    ctx.annotated_detections = annotated

    stage.run(ctx)

    # Frames 0 and 5 are real; frames 1-4 are interpolated
    assert 0 in ctx.midlines_3d[0]
    assert 0 in ctx.midlines_3d[5]

    # Interpolated frames
    for f in (1, 2, 3, 4):
        assert 0 in ctx.midlines_3d[f], f"Fish 0 missing in interpolated frame {f}"
        m = ctx.midlines_3d[f][0]
        assert m.is_low_confidence is True, (
            f"Interpolated frame {f} should have is_low_confidence=True"
        )


def test_gap_too_long_not_interpolated(tmp_path: Path) -> None:
    """Gap > max_interp_gap is NOT interpolated (stays missing)."""
    m3d_0 = _make_midline3d(fish_id=0, frame_index=0)
    m3d_8 = _make_midline3d(fish_id=0, frame_index=8)

    mock_backend = MagicMock()

    def reconstruct_side_effect(**kw: object) -> dict:
        fi = kw["frame_idx"]
        if fi == 0:
            return {0: m3d_0}
        if fi == 8:
            return {0: m3d_8}
        return {}

    mock_backend.reconstruct_frame.side_effect = reconstruct_side_effect

    # max_interp_gap=3, gap of 7 (frames 1-7) should NOT be interpolated
    stage = _build_stage(
        tmp_path, mock_backend=mock_backend, min_cameras=3, max_interp_gap=3
    )

    ml = _make_midline2d()
    ann = _make_annotated_detection(ml, centroid=(50.0, 50.0))
    annotated = []
    for i in range(9):
        if i in (0, 8):
            annotated.append({"cam1": [ann], "cam2": [ann], "cam3": [ann]})
        else:
            annotated.append({})

    t1 = _make_tracklet("cam1", 1, (0, 8))
    t2 = _make_tracklet("cam2", 2, (0, 8))
    t3 = _make_tracklet("cam3", 3, (0, 8))
    group = TrackletGroup(fish_id=0, tracklets=(t1, t2, t3))

    ctx = PipelineContext()
    ctx.frame_count = 9
    ctx.tracklet_groups = [group]
    ctx.annotated_detections = annotated

    stage.run(ctx)

    # Only frames 0 and 8 have results
    assert 0 in ctx.midlines_3d[0]
    assert 0 in ctx.midlines_3d[8]
    # Gap frames should be empty
    for f in range(1, 8):
        assert 0 not in ctx.midlines_3d[f], (
            f"Fish 0 should NOT be in frame {f} (gap > max_interp_gap)"
        )


# ---------------------------------------------------------------------------
# Empty tracklet_groups (stub path)
# ---------------------------------------------------------------------------


def test_reconstruction_stage_empty_tracklet_groups_stub_path(tmp_path: Path) -> None:
    """run() produces empty midlines_3d when tracklet_groups is [] (stub path)."""
    mock_backend = MagicMock()
    stage = _build_stage(tmp_path, mock_backend=mock_backend)

    ctx = PipelineContext()
    ctx.frame_count = 3
    ctx.tracklet_groups = []

    result = stage.run(ctx)

    assert result is ctx
    assert ctx.midlines_3d is not None
    assert len(ctx.midlines_3d) == 3
    for frame_result in ctx.midlines_3d:
        assert frame_result == {}
    mock_backend.reconstruct_frame.assert_not_called()


def test_empty_tracklet_groups_still_produces_empty_midlines(tmp_path: Path) -> None:
    """Empty tracklet_groups with frame_count=0 produces empty midlines."""
    stage = _build_stage(tmp_path)
    ctx = PipelineContext()
    ctx.frame_count = 0
    ctx.tracklet_groups = []

    stage.run(ctx)
    assert ctx.midlines_3d == []


# ---------------------------------------------------------------------------
# Coasted frames are excluded
# ---------------------------------------------------------------------------


def test_coasted_frames_excluded_from_camera_count(tmp_path: Path) -> None:
    """Coasted frames don't count toward min_cameras."""
    mock_backend = MagicMock()
    mock_backend.reconstruct_frame.return_value = {}
    stage = _build_stage(tmp_path, mock_backend=mock_backend, min_cameras=3)

    ml = _make_midline2d()
    ann = _make_annotated_detection(ml, centroid=(50.0, 50.0))
    annotated = [{"cam1": [ann], "cam2": [ann], "cam3": [ann]}]

    # cam3 is coasted in frame 0 -> only 2 detected cameras -> dropped
    t1 = _make_tracklet("cam1", 1, (0,), frame_status=("detected",))
    t2 = _make_tracklet("cam2", 2, (0,), frame_status=("detected",))
    t3 = _make_tracklet("cam3", 3, (0,), frame_status=("coasted",))
    group = TrackletGroup(fish_id=0, tracklets=(t1, t2, t3))

    ctx = PipelineContext()
    ctx.frame_count = 1
    ctx.tracklet_groups = [group]
    ctx.annotated_detections = annotated

    stage.run(ctx)

    assert ctx.midlines_3d[0] == {}
    mock_backend.reconstruct_frame.assert_not_called()


# ---------------------------------------------------------------------------
# Backend delegation
# ---------------------------------------------------------------------------


def test_triangulation_backend_delegates(tmp_path: Path) -> None:
    """TriangulationBackend.reconstruct_frame delegates to triangulate_midlines."""
    from aquapose.core.reconstruction.backends.triangulation import TriangulationBackend

    calib_file = tmp_path / "calibration.json"
    calib_file.write_text("{}")

    mock_models = {"cam1": MagicMock(), "cam2": MagicMock()}
    ml = _make_midline2d(fish_id=0, camera_id="cam1")
    midline_set = {0: {"cam1": ml}}

    expected_result = {0: MagicMock()}

    with (
        patch(
            "aquapose.core.reconstruction.backends.triangulation.TriangulationBackend._load_models",
            return_value=mock_models,
        ),
        patch(
            "aquapose.core.reconstruction.backends.triangulation.triangulate_midlines",
            return_value=expected_result,
        ) as mock_tri,
    ):
        backend = TriangulationBackend(calibration_path=calib_file)
        result = backend.reconstruct_frame(frame_idx=5, midline_set=midline_set)

    assert result == expected_result
    mock_tri.assert_called_once()
    call_kwargs = mock_tri.call_args
    assert call_kwargs.kwargs["frame_index"] == 5
    assert call_kwargs.kwargs["midline_set"] == midline_set


def test_curve_optimizer_backend_delegates(tmp_path: Path) -> None:
    """CurveOptimizerBackend.reconstruct_frame delegates to CurveOptimizer.optimize_midlines."""
    from aquapose.core.reconstruction.backends.curve_optimizer import (
        CurveOptimizerBackend,
    )

    calib_file = tmp_path / "calibration.json"
    calib_file.write_text("{}")

    mock_models = {"cam1": MagicMock()}
    ml = _make_midline2d(fish_id=0, camera_id="cam1")
    midline_set = {0: {"cam1": ml}}

    expected_result = {0: MagicMock()}

    with patch(
        "aquapose.core.reconstruction.backends.curve_optimizer.CurveOptimizerBackend._load_models",
        return_value=mock_models,
    ):
        backend = CurveOptimizerBackend(calibration_path=calib_file)
        backend._optimizer.optimize_midlines = MagicMock(return_value=expected_result)
        result = backend.reconstruct_frame(frame_idx=3, midline_set=midline_set)

    assert result == expected_result
    backend._optimizer.optimize_midlines.assert_called_once()
    call_kwargs = backend._optimizer.optimize_midlines.call_args
    assert call_kwargs.kwargs["frame_index"] == 3


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_backend_selection_triangulation(tmp_path: Path) -> None:
    """Backend kind 'triangulation' resolves to TriangulationBackend."""
    from aquapose.core.reconstruction.backends.triangulation import TriangulationBackend

    calib_file = tmp_path / "calibration.json"
    calib_file.write_text("{}")

    with patch(
        "aquapose.core.reconstruction.backends.triangulation.TriangulationBackend._load_models",
        return_value={},
    ):
        backend = get_backend("triangulation", calibration_path=calib_file)

    assert isinstance(backend, TriangulationBackend)


def test_backend_selection_curve_optimizer(tmp_path: Path) -> None:
    """Backend kind 'curve_optimizer' resolves to CurveOptimizerBackend."""
    from aquapose.core.reconstruction.backends.curve_optimizer import (
        CurveOptimizerBackend,
    )

    calib_file = tmp_path / "calibration.json"
    calib_file.write_text("{}")

    with patch(
        "aquapose.core.reconstruction.backends.curve_optimizer.CurveOptimizerBackend._load_models",
        return_value={},
    ):
        backend = get_backend("curve_optimizer", calibration_path=calib_file)

    assert isinstance(backend, CurveOptimizerBackend)


# ---------------------------------------------------------------------------
# Backend registry -- unknown kind
# ---------------------------------------------------------------------------


def test_backend_registry_unknown_raises() -> None:
    """Backend registry raises ValueError for unknown backend kind."""
    with pytest.raises(ValueError, match="Unknown reconstruction backend kind"):
        get_backend("not_a_backend", calibration_path="fake.json")


# ---------------------------------------------------------------------------
# Import boundary (ENG-07)
# ---------------------------------------------------------------------------


def test_import_boundary() -> None:
    """core/reconstruction/ modules must not import from aquapose.engine at runtime."""
    modules_to_check = [
        "aquapose.core.reconstruction",
        "aquapose.core.reconstruction.stage",
        "aquapose.core.reconstruction.backends",
        "aquapose.core.reconstruction.backends.triangulation",
        "aquapose.core.reconstruction.backends.curve_optimizer",
    ]

    for mod_name in modules_to_check:
        mod = importlib.import_module(mod_name)
        source = inspect.getsource(mod)

        lines = source.split("\n")
        in_type_checking_block = False
        for line in lines:
            stripped = line.strip()
            if stripped == "if TYPE_CHECKING:" or stripped.startswith(
                "if TYPE_CHECKING:"
            ):
                in_type_checking_block = True
                continue
            if (
                in_type_checking_block
                and stripped
                and not line.startswith(" ")
                and not line.startswith("\t")
            ):
                in_type_checking_block = False
            if not in_type_checking_block:
                assert "from aquapose.engine" not in line, (
                    f"Module {mod_name} has a potential runtime import from "
                    f"aquapose.engine outside TYPE_CHECKING: {line!r}"
                )


# ---------------------------------------------------------------------------
# Active stages importable
# ---------------------------------------------------------------------------


def test_active_stages_importable() -> None:
    """Active stage classes are importable from aquapose.core (v2.1 set)."""
    from aquapose.core import (
        DetectionStage,
        MidlineStage,
        ReconstructionStage,
    )

    assert DetectionStage is not None
    assert MidlineStage is not None
    assert ReconstructionStage is not None

    from aquapose.core.association import AssociationStage
    from aquapose.core.tracking import TrackingStage

    assert TrackingStage is not None
    assert AssociationStage is not None


# ---------------------------------------------------------------------------
# build_stages factory
# ---------------------------------------------------------------------------


def test_build_stages_returns_stages(tmp_path: Path) -> None:
    """build_stages(config) returns an ordered list of 5 Stage instances."""
    from aquapose.core import (
        DetectionStage,
        MidlineStage,
        ReconstructionStage,
    )
    from aquapose.core.association import AssociationStage
    from aquapose.core.tracking import TrackingStage
    from aquapose.engine.config import PipelineConfig
    from aquapose.engine.pipeline import build_stages

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    calib_file = tmp_path / "calibration.json"
    calib_file.write_text("{}")
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"fake")
    weights_file = tmp_path / "weights.pth"
    weights_file.write_bytes(b"fake")

    config = PipelineConfig(
        video_dir=str(video_dir),
        calibration_path=str(calib_file),
        detection=__import__(
            "aquapose.engine.config", fromlist=["DetectionConfig"]
        ).DetectionConfig(model_path=str(model_file)),
        midline=__import__(
            "aquapose.engine.config", fromlist=["MidlineConfig"]
        ).MidlineConfig(weights_path=str(weights_file)),
    )

    with (
        patch(
            "aquapose.core.detection.stage.DetectionStage.__init__", return_value=None
        ),
        patch("aquapose.core.midline.stage.MidlineStage.__init__", return_value=None),
        patch(
            "aquapose.core.reconstruction.stage.ReconstructionStage.__init__",
            return_value=None,
        ),
    ):
        stages = build_stages(config)

    assert isinstance(stages, list)
    assert len(stages) == 5, f"Expected 5 stages, got {len(stages)}"
    assert isinstance(stages[0], DetectionStage)
    assert isinstance(stages[1], TrackingStage)
    assert isinstance(stages[2], AssociationStage)
    assert isinstance(stages[3], MidlineStage)
    assert isinstance(stages[4], ReconstructionStage)

    from aquapose.core.context import Stage

    for stage in stages:
        if not isinstance(stage, AssociationStage):
            assert isinstance(stage, Stage), (
                f"{type(stage).__name__} must satisfy Stage Protocol"
            )


# ---------------------------------------------------------------------------
# PosePipeline instantiable with build_stages
# ---------------------------------------------------------------------------


def test_pose_pipeline_instantiable_with_build_stages(tmp_path: Path) -> None:
    """PosePipeline(stages=build_stages(config), config=config) instantiates."""
    from aquapose.engine.config import PipelineConfig
    from aquapose.engine.pipeline import PosePipeline, build_stages

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    calib_file = tmp_path / "calibration.json"
    calib_file.write_text("{}")

    config = PipelineConfig(
        video_dir=str(video_dir),
        calibration_path=str(calib_file),
    )

    with (
        patch(
            "aquapose.core.detection.stage.DetectionStage.__init__", return_value=None
        ),
        patch("aquapose.core.midline.stage.MidlineStage.__init__", return_value=None),
        patch(
            "aquapose.core.reconstruction.stage.ReconstructionStage.__init__",
            return_value=None,
        ),
    ):
        stages = build_stages(config)
        pipeline = PosePipeline(stages=stages, config=config)

    assert pipeline is not None


# ---------------------------------------------------------------------------
# Validation -- missing context fields
# ---------------------------------------------------------------------------


def test_run_raises_if_annotated_detections_missing(tmp_path: Path) -> None:
    """run() raises ValueError if both tracklet_groups and annotated_detections are None."""
    stage = _build_stage(tmp_path)
    ctx = PipelineContext()

    with pytest.raises(ValueError, match=r"annotated_detections"):
        stage.run(ctx)
