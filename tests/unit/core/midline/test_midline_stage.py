"""Interface tests for MidlineStage — Stage 2 of the AquaPose pipeline.

Validates:
- MidlineStage satisfies the Stage Protocol via structural typing
- run() correctly populates PipelineContext.annotated_detections
- Direct pose backend raises NotImplementedError
- Backend registry raises ValueError for unknown kinds
- Import boundary (ENG-07): no engine/ runtime imports in core/midline/
- Fail-fast on missing U-Net weights
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aquapose.core.midline import AnnotatedDetection, MidlineStage
from aquapose.core.midline.backends import get_backend
from aquapose.engine.stages import PipelineContext, Stage
from aquapose.segmentation.detector import Detection

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_midline_stage_satisfies_protocol(tmp_path: Path) -> None:
    """MidlineStage is a Stage (structural typing) even before run() is called."""
    stage = _build_stage(tmp_path)
    assert isinstance(stage, Stage), (
        "MidlineStage must satisfy the Stage Protocol via structural typing"
    )


# ---------------------------------------------------------------------------
# Context population
# ---------------------------------------------------------------------------


def test_midline_stage_populates_annotated_detections(tmp_path: Path) -> None:
    """run() populates context.annotated_detections from context.detections."""
    det = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    synthetic_detections: list[dict[str, list[Detection]]] = [
        {"cam1": [det], "cam2": []},
        {"cam1": [], "cam2": [det]},
    ]

    # Build expected annotated output (one AnnotatedDetection per Detection)
    synthetic_annotated: list[dict[str, list[AnnotatedDetection]]] = []
    for frame_dets in synthetic_detections:
        frame_annotated: dict[str, list[AnnotatedDetection]] = {}
        for cam_id, cam_dets in frame_dets.items():
            frame_annotated[cam_id] = [
                AnnotatedDetection(
                    detection=d,
                    mask=None,
                    crop_region=None,
                    midline=None,
                    camera_id=cam_id,
                    frame_index=i,
                )
                for i, d in enumerate(cam_dets)
            ]
        synthetic_annotated.append(frame_annotated)

    stage = _build_stage(tmp_path, synthetic_annotated=synthetic_annotated)

    ctx = PipelineContext()
    ctx.detections = synthetic_detections
    ctx.camera_ids = ["cam1", "cam2"]
    result = stage.run(ctx)

    assert result is ctx, "run() must return the same context object"
    assert ctx.annotated_detections is not None
    assert isinstance(ctx.annotated_detections, list)
    assert len(ctx.annotated_detections) == len(synthetic_detections)

    # Validate per-frame structure
    for frame_annotated in ctx.annotated_detections:
        assert isinstance(frame_annotated, dict)
        for cam_id, cam_list in frame_annotated.items():
            assert isinstance(cam_id, str)
            assert isinstance(cam_list, list)


# ---------------------------------------------------------------------------
# Direct pose backend
# ---------------------------------------------------------------------------


def test_direct_pose_backend_raises() -> None:
    """Constructing the direct pose backend raises NotImplementedError."""
    from aquapose.core.midline.backends.direct_pose import DirectPoseBackend

    with pytest.raises(NotImplementedError, match="planned alternative backend"):
        DirectPoseBackend()


def test_direct_pose_backend_process_frame_raises() -> None:
    """DirectPoseBackend.process_frame raises NotImplementedError.

    Tests the process_frame stub directly without going through the registry,
    by patching __init__ to not raise so the instance can be created.
    """
    from aquapose.core.midline.backends.direct_pose import DirectPoseBackend

    with patch.object(DirectPoseBackend, "__init__", return_value=None):
        instance = DirectPoseBackend()  # type: ignore[call-arg]
    with pytest.raises(NotImplementedError):
        instance.process_frame(0, {}, {}, [])


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


def test_backend_registry_unknown_raises() -> None:
    """get_backend raises ValueError for an unrecognized backend kind."""
    with pytest.raises(ValueError, match="Unknown midline backend kind"):
        get_backend("nonexistent_backend")


def test_backend_registry_direct_pose_raises() -> None:
    """get_backend('direct_pose') propagates NotImplementedError from the stub."""
    with pytest.raises(NotImplementedError):
        get_backend("direct_pose")


def test_backend_registry_segment_then_extract_constructs(tmp_path: Path) -> None:
    """get_backend('segment_then_extract') constructs without error (mocked model)."""
    fake_weights = tmp_path / "model.pth"
    fake_weights.write_bytes(b"fake")

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_segmentor = MagicMock()
    mock_segmentor.get_model.return_value = mock_model

    with (
        patch(
            "aquapose.segmentation.model.UNetSegmentor",
            return_value=mock_segmentor,
        ),
        patch(
            "aquapose.core.midline.backends.segment_then_extract.Path.exists",
            return_value=True,
        ),
    ):
        backend = get_backend(
            "segment_then_extract",
            weights_path=str(fake_weights),
            device="cpu",
        )
        assert backend is not None


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------

_CORE_MIDLINE_MODULES = [
    "aquapose.core.midline",
    "aquapose.core.midline.stage",
    "aquapose.core.midline.types",
    "aquapose.core.midline.backends",
    "aquapose.core.midline.backends.segment_then_extract",
    "aquapose.core.midline.backends.direct_pose",
]


def test_import_boundary_no_engine_imports() -> None:
    """No core/midline/ module may import from aquapose.engine at module level.

    TYPE_CHECKING-guarded imports are permitted, but no runtime import of
    aquapose.engine is allowed (ENG-07).
    """
    for mod_name in _CORE_MIDLINE_MODULES:
        module = importlib.import_module(mod_name)
        source = inspect.getsource(module)
        lines = source.splitlines()

        in_type_checking_block = False
        for line in lines:
            stripped = line.strip()

            # Detect entry into TYPE_CHECKING block
            if "TYPE_CHECKING" in stripped and "if" in stripped:
                in_type_checking_block = True
                continue

            # Exit TYPE_CHECKING block when we return to top-level indentation
            if in_type_checking_block and stripped and not line.startswith(" "):
                in_type_checking_block = False

            if not in_type_checking_block:
                assert "from aquapose.engine" not in stripped, (
                    f"{mod_name}: runtime import from aquapose.engine found: {line!r}. "
                    "Use TYPE_CHECKING guard for annotation-only imports (ENG-07)."
                )
                assert stripped != "import aquapose.engine", (
                    f"{mod_name}: runtime import found: {line!r}. "
                    "Use TYPE_CHECKING guard (ENG-07)."
                )
                assert not stripped.startswith("import aquapose.engine."), (
                    f"{mod_name}: runtime import found: {line!r}. "
                    "Use TYPE_CHECKING guard (ENG-07)."
                )


# ---------------------------------------------------------------------------
# Fail-fast on missing weights
# ---------------------------------------------------------------------------


def test_missing_weights_raises_at_construction(tmp_path: Path) -> None:
    """SegmentThenExtractBackend raises FileNotFoundError when weights don't exist."""
    from aquapose.core.midline.backends.segment_then_extract import (
        SegmentThenExtractBackend,
    )

    nonexistent_weights = tmp_path / "nonexistent_weights.pth"
    # Do NOT create the file — it must not exist

    with pytest.raises(FileNotFoundError, match=r"nonexistent_weights\.pth"):
        SegmentThenExtractBackend(weights_path=str(nonexistent_weights))


def test_missing_weights_raises_via_midline_stage(tmp_path: Path) -> None:
    """MidlineStage raises FileNotFoundError when U-Net weights do not exist."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "cam1-video.mp4").write_bytes(b"fake")

    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    nonexistent_weights = tmp_path / "nonexistent.pth"

    mock_cam = MagicMock()
    mock_calib = MagicMock()
    mock_calib.cameras = {"cam1": mock_cam}

    with (
        patch(
            "aquapose.calibration.loader.load_calibration_data",
            return_value=mock_calib,
        ),
        patch(
            "aquapose.calibration.loader.compute_undistortion_maps",
            return_value=MagicMock(),
        ),
        pytest.raises(FileNotFoundError, match=r"nonexistent\.pth"),
    ):
        MidlineStage(
            video_dir=video_dir,
            calibration_path=calib_path,
            weights_path=str(nonexistent_weights),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_stage(
    tmp_path: Path,
    synthetic_annotated: list[dict[str, list[AnnotatedDetection]]] | None = None,
) -> MidlineStage:
    """Build a MidlineStage with all heavy I/O mocked.

    The stage's run() is replaced with a lightweight stub that returns the
    supplied *synthetic_annotated* instead of opening real videos or running
    U-Net inference.

    Args:
        tmp_path: Temporary directory for synthetic stub files.
        synthetic_annotated: Per-frame annotated detection lists to return from run().
            Defaults to a single empty frame for two cameras.

    Returns:
        Constructed and wired MidlineStage.
    """
    if synthetic_annotated is None:
        synthetic_annotated = [{"cam1": [], "cam2": []}]

    video_dir = tmp_path / "videos"
    video_dir.mkdir(exist_ok=True)
    (video_dir / "cam1-test.mp4").write_bytes(b"fake")
    (video_dir / "cam2-test.mp4").write_bytes(b"fake")

    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    fake_weights = tmp_path / "model.pth"
    fake_weights.write_bytes(b"fake")

    mock_cam1 = MagicMock()
    mock_cam2 = MagicMock()
    mock_calib = MagicMock()
    mock_calib.cameras = {"cam1": mock_cam1, "cam2": mock_cam2}
    mock_undist = MagicMock()
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    mock_segmentor = MagicMock()
    mock_segmentor.get_model.return_value = mock_model

    with (
        patch(
            "aquapose.calibration.loader.load_calibration_data",
            return_value=mock_calib,
        ),
        patch(
            "aquapose.calibration.loader.compute_undistortion_maps",
            return_value=mock_undist,
        ),
        patch(
            "aquapose.segmentation.model.UNetSegmentor",
            return_value=mock_segmentor,
        ),
        patch(
            "aquapose.core.midline.backends.segment_then_extract.Path.exists",
            return_value=True,
        ),
    ):
        stage = MidlineStage(
            video_dir=video_dir,
            calibration_path=calib_path,
            weights_path=str(fake_weights),
            device="cpu",
        )

    # Replace run() with a deterministic stub that returns synthetic data
    n_frames = len(synthetic_annotated)

    def _stub_run(ctx: PipelineContext) -> PipelineContext:
        detections = ctx.detections or []
        frames_to_process = min(n_frames, len(detections))
        ctx.annotated_detections = synthetic_annotated[:frames_to_process]
        return ctx

    stage.run = _stub_run  # type: ignore[method-assign]
    return stage
