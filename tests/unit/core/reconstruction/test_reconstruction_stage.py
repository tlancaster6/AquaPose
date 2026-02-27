"""Interface tests for ReconstructionStage — Stage 5 of the AquaPose pipeline.

Validates:
- ReconstructionStage satisfies the Stage Protocol via structural typing
- run() correctly assembles MidlineSet and populates PipelineContext.midlines_3d
- run() produces empty midlines_3d when tracklet_groups is empty (stub path, Phase 22)
- Triangulation backend delegates to triangulate_midlines()
- Curve optimizer backend delegates to CurveOptimizer.optimize_midlines()
- Backend selection via "triangulation" and "curve_optimizer" strings
- Backend registry raises ValueError for unknown kinds
- Import boundary (ENG-07): no engine/ runtime imports in core/reconstruction/
- MidlineSet assembly from FishTrack.camera_detections + annotated_detections
- All 5 stages importable from core/
- build_stages() returns 5 ordered Stage instances
- PosePipeline instantiable with build_stages(config)
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.core.context import PipelineContext, Stage
from aquapose.core.reconstruction import ReconstructionStage
from aquapose.core.reconstruction.backends import get_backend
from aquapose.core.tracking import FishTrack, TrackState
from aquapose.reconstruction.midline import Midline2D

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


def _make_annotated_detection(
    midline: Midline2D | None, camera_id: str = "cam1"
) -> MagicMock:
    """Create a mock AnnotatedDetection with a midline attribute."""
    ann = MagicMock()
    ann.midline = midline
    ann.camera_id = camera_id
    return ann


def _make_fish_track(
    fish_id: int = 0,
    camera_detections: dict[str, int] | None = None,
) -> FishTrack:
    """Create a FishTrack with given camera_detections mapping."""
    track = FishTrack(fish_id=fish_id)
    track.camera_detections = camera_detections or {}
    track.state = TrackState.CONFIRMED
    track.positions.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    return track


def _build_stage(
    tmp_path: Path, mock_backend: MagicMock | None = None
) -> ReconstructionStage:
    """Build a ReconstructionStage with mocked backend (no real calibration)."""
    calib_file = tmp_path / "calibration.json"
    calib_file.write_text("{}")

    stage = ReconstructionStage.__new__(ReconstructionStage)
    stage._calibration_path = calib_file
    stage._skip_camera_id = "e3v8250"

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
# Context population
# ---------------------------------------------------------------------------


def test_reconstruction_stage_populates_midlines_3d(tmp_path: Path) -> None:
    """run() populates context.midlines_3d as a list of per-frame dicts.

    v2.1 transition: ReconstructionStage iterates annotated_detections only.
    Fish identity via TrackletGroup is wired in Phase 26.
    tracklet_groups must be None (not empty list) to reach the annotated_detections path.
    """
    ml1 = _make_midline2d(fish_id=0, camera_id="cam1")
    ann1 = MagicMock()
    ann1.midline = ml1

    synthetic_annotated = [
        {"cam1": [ann1]},
        {"cam1": []},
    ]

    mock_backend = MagicMock()
    mock_backend.reconstruct_frame.return_value = {}
    stage = _build_stage(tmp_path, mock_backend=mock_backend)

    ctx = PipelineContext()
    ctx.annotated_detections = synthetic_annotated
    # tracklet_groups must be None (not []) to reach the annotated_detections path.
    # When tracklet_groups == [], run() early-returns with empty midlines_3d.
    ctx.tracklet_groups = None

    result = stage.run(ctx)

    assert result is ctx, "run() must return the same context object"
    assert ctx.midlines_3d is not None
    assert isinstance(ctx.midlines_3d, list)
    assert len(ctx.midlines_3d) == 2, "midlines_3d must have one entry per frame"


def test_reconstruction_stage_empty_tracklet_groups_stub_path(tmp_path: Path) -> None:
    """run() produces empty midlines_3d when tracklet_groups is [] (stub path, Phase 22)."""
    mock_backend = MagicMock()
    stage = _build_stage(tmp_path, mock_backend=mock_backend)

    ctx = PipelineContext()
    ctx.frame_count = 3
    ctx.tracklet_groups = []  # empty list = stub output from AssociationStubStage

    result = stage.run(ctx)

    assert result is ctx, "run() must return the same context object"
    assert ctx.midlines_3d is not None
    assert isinstance(ctx.midlines_3d, list)
    assert len(ctx.midlines_3d) == 3, "midlines_3d must have one entry per frame"
    for frame_result in ctx.midlines_3d:
        assert frame_result == {}, "Each frame must be an empty dict when stub"
    mock_backend.reconstruct_frame.assert_not_called()


# ---------------------------------------------------------------------------
# MidlineSet assembly
# ---------------------------------------------------------------------------


def test_midline_set_assembly(tmp_path: Path) -> None:
    """_assemble_midline_set correctly builds dict[fish_id, dict[cam_id, Midline2D]]."""
    ml_cam1 = _make_midline2d(fish_id=0, camera_id="cam1")
    ml_cam2 = _make_midline2d(fish_id=0, camera_id="cam2")

    ann_cam1 = _make_annotated_detection(ml_cam1, camera_id="cam1")
    ann_cam2 = _make_annotated_detection(ml_cam2, camera_id="cam2")

    # Fish 0 is visible in cam1 (det_idx=0) and cam2 (det_idx=1)
    track = _make_fish_track(fish_id=0, camera_detections={"cam1": 0, "cam2": 0})
    # Fish 1 has a None midline — should be excluded
    track2 = _make_fish_track(fish_id=1, camera_detections={"cam1": 1})
    ann_no_midline = _make_annotated_detection(None, camera_id="cam1")

    frame_annotated: dict[str, list] = {
        "cam1": [ann_cam1, ann_no_midline],
        "cam2": [ann_cam2],
    }
    frame_tracks = [track, track2]

    stage = _build_stage(tmp_path)
    midline_set = stage._assemble_midline_set(
        frame_idx=0,
        frame_tracks=frame_tracks,
        frame_annotated=frame_annotated,
    )

    assert 0 in midline_set, "Fish 0 should be in MidlineSet"
    assert "cam1" in midline_set[0], "cam1 should be in fish 0's midlines"
    assert "cam2" in midline_set[0], "cam2 should be in fish 0's midlines"
    assert midline_set[0]["cam1"] is ml_cam1
    assert midline_set[0]["cam2"] is ml_cam2

    # Fish 1 has None midline — excluded
    assert 1 not in midline_set, "Fish 1 with None midline should be excluded"


def test_midline_set_assembly_empty_camera_detections(tmp_path: Path) -> None:
    """Fish with no camera_detections are excluded from MidlineSet."""
    track = _make_fish_track(fish_id=0, camera_detections={})
    stage = _build_stage(tmp_path)

    midline_set = stage._assemble_midline_set(
        frame_idx=0,
        frame_tracks=[track],
        frame_annotated={},
    )
    assert midline_set == {}, (
        "Fish with no camera_detections must produce empty MidlineSet"
    )


def test_midline_set_assembly_out_of_range_det_idx(tmp_path: Path) -> None:
    """Out-of-range det_idx is handled gracefully (logged and skipped)."""
    ml = _make_midline2d(fish_id=0)
    ann = _make_annotated_detection(ml)
    track = _make_fish_track(fish_id=0, camera_detections={"cam1": 99})  # out of range
    frame_annotated: dict[str, list] = {"cam1": [ann]}
    stage = _build_stage(tmp_path)

    midline_set = stage._assemble_midline_set(
        frame_idx=0,
        frame_tracks=[track],
        frame_annotated=frame_annotated,
    )
    assert midline_set == {}, "Out-of-range det_idx must be skipped gracefully"


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
# Backend registry — unknown kind
# ---------------------------------------------------------------------------


def test_backend_registry_unknown_raises() -> None:
    """Backend registry raises ValueError for unknown backend kind."""
    with pytest.raises(ValueError, match="Unknown reconstruction backend kind"):
        get_backend("not_a_backend", calibration_path="fake.json")


# ---------------------------------------------------------------------------
# Import boundary (ENG-07)
# ---------------------------------------------------------------------------


def test_import_boundary() -> None:
    """core/reconstruction/ modules must not import from aquapose.engine at runtime.

    The ENG-07 import boundary forbids runtime imports from aquapose.engine in
    core/ modules. Imports under TYPE_CHECKING are allowed (annotations only).

    Strategy: verify that aquapose.engine is not in sys.modules as a side-effect
    of importing the reconstruction modules, and that no module-level code in
    core/reconstruction/ brings in engine symbols at import time.
    """

    modules_to_check = [
        "aquapose.core.reconstruction",
        "aquapose.core.reconstruction.stage",
        "aquapose.core.reconstruction.types",
        "aquapose.core.reconstruction.backends",
        "aquapose.core.reconstruction.backends.triangulation",
        "aquapose.core.reconstruction.backends.curve_optimizer",
    ]

    for mod_name in modules_to_check:
        mod = importlib.import_module(mod_name)
        source = inspect.getsource(mod)

        # Check that aquapose.engine does NOT appear as a bare import anywhere
        # OUTSIDE a TYPE_CHECKING block. Use a simple state machine that tracks
        # whether we are inside `if TYPE_CHECKING:` (indented) blocks.
        lines = source.split("\n")
        in_type_checking_block = False
        for line in lines:
            stripped = line.strip()
            # Detect `if TYPE_CHECKING:` at any indent level
            if stripped == "if TYPE_CHECKING:" or stripped.startswith(
                "if TYPE_CHECKING:"
            ):
                in_type_checking_block = True
                continue
            # A non-blank, non-indented line that is NOT part of the if-block ends it
            # We detect this by checking if indentation is 0 for a non-empty line
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
# Active stages importable (v2.1 — stubs in engine/pipeline.py, not core/)
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

    # AssociationStubStage lives in engine/pipeline.py (stub until Phase 25)
    # TrackingStage lives in aquapose.core.tracking (replaced TrackingStubStage in Phase 24)
    from aquapose.core.tracking import TrackingStage
    from aquapose.engine.pipeline import AssociationStubStage

    assert TrackingStage is not None
    assert AssociationStubStage is not None


# ---------------------------------------------------------------------------
# build_stages factory (v2.1 — 5 stages in production, 4 in synthetic)
# ---------------------------------------------------------------------------


def test_build_stages_returns_stages(tmp_path: Path) -> None:
    """build_stages(config) returns an ordered list of 5 Stage instances (production mode)."""
    from aquapose.core import (
        DetectionStage,
        MidlineStage,
        ReconstructionStage,
    )
    from aquapose.core.tracking import TrackingStage
    from aquapose.engine.config import PipelineConfig
    from aquapose.engine.pipeline import (
        AssociationStubStage,
        build_stages,
    )

    # Create dummy files so path checks don't fail at import time
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

    # Mock all heavy init operations
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
    assert isinstance(stages[2], AssociationStubStage)
    assert isinstance(stages[3], MidlineStage)
    assert isinstance(stages[4], ReconstructionStage)

    # All satisfy Stage Protocol (stubs are duck-typed, not runtime_checkable)
    from aquapose.core.context import Stage

    for stage in stages:
        if not isinstance(stage, AssociationStubStage):
            assert isinstance(stage, Stage), (
                f"{type(stage).__name__} must satisfy Stage Protocol"
            )


# ---------------------------------------------------------------------------
# PosePipeline instantiable with build_stages
# ---------------------------------------------------------------------------


def test_pose_pipeline_instantiable_with_build_stages(tmp_path: Path) -> None:
    """PosePipeline(stages=build_stages(config), config=config) instantiates without error."""
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

    # Mock all stage constructors to avoid loading real models/calibration
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

    assert pipeline is not None, "PosePipeline must instantiate without error"


# ---------------------------------------------------------------------------
# Validation — missing context fields
# ---------------------------------------------------------------------------


def test_run_raises_if_annotated_detections_missing(tmp_path: Path) -> None:
    """run() raises ValueError if context.annotated_detections is None (and tracklet_groups is also None)."""
    stage = _build_stage(tmp_path)
    ctx = PipelineContext()
    # Both annotated_detections and tracklet_groups are None — should raise
    # (tracklet_groups=None means neither the stub path nor the full path applies)

    with pytest.raises(ValueError, match=r"annotated_detections"):
        stage.run(ctx)
