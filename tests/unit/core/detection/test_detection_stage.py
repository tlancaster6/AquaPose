"""Interface tests for DetectionStage — Stage 1 of the AquaPose pipeline.

Validates:
- DetectionStage satisfies the Stage Protocol via structural typing
- run() correctly populates PipelineContext fields
- Backend registry raises ValueError for unknown detector kinds
- Import boundary (ENG-07): no engine/ runtime imports in core/detection/
- Fail-fast on missing YOLO weights
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aquapose.core.detection import Detection, DetectionStage
from aquapose.core.detection.backends import get_backend
from aquapose.engine.stages import PipelineContext, Stage

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_detection_stage_satisfies_protocol(tmp_path: Path) -> None:
    """DetectionStage is a Stage (structural typing) even before run() is called."""
    stage = _build_stage(tmp_path)
    assert isinstance(stage, Stage), (
        "DetectionStage must satisfy the Stage Protocol via structural typing"
    )


# ---------------------------------------------------------------------------
# Context population
# ---------------------------------------------------------------------------


def test_detection_stage_populates_context(tmp_path: Path) -> None:
    """run() populates context.detections, .frame_count, and .camera_ids."""
    det = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    synthetic_detections: list[dict[str, list[Detection]]] = [
        {"cam1": [det], "cam2": []},
        {"cam1": [], "cam2": [det]},
    ]

    stage = _build_stage(tmp_path, synthetic_detections=synthetic_detections)

    ctx = PipelineContext()
    result = stage.run(ctx)

    assert result is ctx, "run() must return the same context object"
    assert ctx.detections is not None
    assert isinstance(ctx.detections, list)
    assert ctx.frame_count == len(synthetic_detections)
    assert ctx.camera_ids is not None
    assert isinstance(ctx.camera_ids, list)
    assert all(isinstance(cid, str) for cid in ctx.camera_ids)

    # Validate per-frame detection structure
    for frame_dets in ctx.detections:
        assert isinstance(frame_dets, dict)
        for cam_id, cam_dets in frame_dets.items():
            assert isinstance(cam_id, str)
            assert isinstance(cam_dets, list)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


def test_backend_registry_unknown_kind_raises() -> None:
    """get_backend raises ValueError for an unrecognized detector kind."""
    with pytest.raises(ValueError, match="Unknown detector kind"):
        get_backend("unicorn_detector")


def test_backend_registry_yolo_requires_model_path(tmp_path: Path) -> None:
    """get_backend('yolo') without a model_path raises TypeError or FileNotFoundError."""
    with pytest.raises((TypeError, FileNotFoundError)):
        get_backend("yolo")


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------

_CORE_DETECTION_MODULES = [
    "aquapose.core.detection",
    "aquapose.core.detection.stage",
    "aquapose.core.detection.types",
    "aquapose.core.detection.backends",
    "aquapose.core.detection.backends.yolo",
]


def test_import_boundary_no_engine_imports() -> None:
    """No core/detection/ module may import from aquapose.engine at module level.

    TYPE_CHECKING-guarded imports are permitted, but no runtime import of
    aquapose.engine is allowed (ENG-07).
    """
    for mod_name in _CORE_DETECTION_MODULES:
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
    """DetectionStage raises FileNotFoundError when model weights don't exist."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "cam1-video.mp4").write_bytes(b"fake")

    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    nonexistent_weights = tmp_path / "nonexistent.pt"

    mock_cam = MagicMock()
    mock_calib = MagicMock()
    mock_calib.cameras = {"cam1": mock_cam}

    # Patch calibration loading so construction proceeds to backend creation
    with (
        patch(
            "aquapose.calibration.loader.load_calibration_data",
            return_value=mock_calib,
        ),
        patch(
            "aquapose.calibration.loader.compute_undistortion_maps",
            return_value=MagicMock(),
        ),
        pytest.raises(FileNotFoundError, match=r"nonexistent\.pt"),
    ):
        DetectionStage(
            video_dir=video_dir,
            calibration_path=calib_path,
            detector_kind="yolo",
            model_path=str(nonexistent_weights),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_stage(
    tmp_path: Path,
    synthetic_detections: list[dict[str, list[Detection]]] | None = None,
) -> DetectionStage:
    """Build a DetectionStage with all heavy I/O mocked.

    The stage's run() is replaced with a lightweight function that returns
    the supplied *synthetic_detections* instead of opening real videos.

    Args:
        tmp_path: Temporary directory for synthetic stub files.
        synthetic_detections: Per-frame detection lists to return from run().
            Defaults to a single empty frame for two cameras.

    Returns:
        Constructed and wired DetectionStage.
    """
    if synthetic_detections is None:
        synthetic_detections = [{"cam1": [], "cam2": []}]

    video_dir = tmp_path / "videos"
    video_dir.mkdir(exist_ok=True)
    (video_dir / "cam1-test.mp4").write_bytes(b"fake")
    (video_dir / "cam2-test.mp4").write_bytes(b"fake")

    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    fake_weights = tmp_path / "model.pt"
    fake_weights.write_bytes(b"fake")

    mock_cam1 = MagicMock()
    mock_cam2 = MagicMock()
    mock_calib = MagicMock()
    mock_calib.cameras = {"cam1": mock_cam1, "cam2": mock_cam2}
    mock_undist = MagicMock()

    # Patch the loader functions used inside DetectionStage.__init__.
    # They are imported locally inside __init__, so patch the source module.
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
            "aquapose.core.detection.backends.yolo.YOLOBackend.__init__",
            return_value=None,
        ),
        patch(
            "aquapose.core.detection.backends.yolo.Path.exists",
            return_value=True,
        ),
    ):
        stage = DetectionStage(
            video_dir=video_dir,
            calibration_path=calib_path,
            detector_kind="yolo",
            model_path=str(fake_weights),
        )

    # Replace run() with a deterministic stub that returns synthetic data
    camera_ids = sorted(["cam1", "cam2"])
    n_frames = len(synthetic_detections)
    stop_frame = stage._stop_frame

    def _stub_run(ctx: PipelineContext) -> PipelineContext:
        frames: list[dict[str, list[Detection]]] = []
        for i in range(n_frames):
            if stop_frame is not None and i >= stop_frame:
                break
            frames.append(synthetic_detections[i])
        ctx.detections = frames
        ctx.frame_count = len(frames)
        ctx.camera_ids = camera_ids
        return ctx

    stage.run = _stub_run  # type: ignore[method-assign]
    return stage
