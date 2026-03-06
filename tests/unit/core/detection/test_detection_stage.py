"""Interface tests for DetectionStage — Stage 1 of the AquaPose pipeline.

Validates:
- DetectionStage satisfies the Stage Protocol via structural typing
- run() correctly populates PipelineContext fields
- Backend registry raises ValueError for unknown detector kinds
- Import boundary (ENG-07): no engine/ runtime imports in core/detection/
- Fail-fast on missing YOLO weights
- Batched detection: detect_batch() called once per timestep (not detect() per camera)
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.core.context import PipelineContext, Stage
from aquapose.core.detection import Detection, DetectionStage
from aquapose.core.detection.backends import get_backend

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


def test_backend_registry_yolo_requires_weights_path(tmp_path: Path) -> None:
    """get_backend('yolo') without a weights_path raises TypeError or FileNotFoundError."""
    with pytest.raises((TypeError, FileNotFoundError)):
        get_backend("yolo")


def test_backend_registry_yolo_obb_requires_weights_path() -> None:
    """get_backend('yolo_obb') without a weights_path raises TypeError or FileNotFoundError."""
    with pytest.raises((TypeError, FileNotFoundError)):
        get_backend("yolo_obb")


def test_backend_registry_yolo_obb_with_weights(tmp_path: Path) -> None:
    """get_backend('yolo_obb') with a valid weights file returns a YOLOOBBBackend."""
    from aquapose.core.detection.backends.yolo_obb import YOLOOBBBackend

    fake_weights = tmp_path / "model_obb.pt"
    fake_weights.write_bytes(b"fake")

    with patch("ultralytics.YOLO.__init__", return_value=None):
        backend = get_backend("yolo_obb", weights_path=str(fake_weights))

    assert isinstance(backend, YOLOOBBBackend)


def test_backend_registry_unknown_lists_both_kinds() -> None:
    """get_backend raises ValueError mentioning both 'yolo' and 'yolo_obb'."""
    with pytest.raises(ValueError, match=r"yolo_obb"):
        get_backend("unknown_kind")


def test_yolo_obb_detect_populates_angle_and_obb_points(tmp_path: Path) -> None:
    """YOLOOBBBackend.detect() returns Detections with negated angle and obb_points."""
    import torch

    from aquapose.core.detection.backends.yolo_obb import YOLOOBBBackend

    fake_weights = tmp_path / "model_obb.pt"
    fake_weights.write_bytes(b"fake")

    # Build fake OBB result tensors
    # xywhr: (cx, cy, w, h, angle_cw_rad)
    angle_cw = 0.5  # clockwise radians (ultralytics convention)
    fake_xywhr = torch.tensor([[100.0, 150.0, 60.0, 20.0, angle_cw]])
    # xyxyxyxy: (1, 4, 2) four corners
    fake_corners = torch.tensor(
        [[[80.0, 140.0], [120.0, 140.0], [120.0, 160.0], [80.0, 160.0]]]
    )
    fake_conf = torch.tensor([0.85])

    mock_obb = MagicMock()
    mock_obb.xywhr = fake_xywhr
    mock_obb.xyxyxyxy = fake_corners
    mock_obb.conf = fake_conf

    mock_result = MagicMock()
    mock_result.obb = mock_obb

    with patch("ultralytics.YOLO.__init__", return_value=None):
        backend = YOLOOBBBackend(weights_path=fake_weights)

    backend._model = MagicMock()
    backend._model.predict.return_value = [mock_result]

    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    detections = backend.detect(frame)

    assert len(detections) == 1
    det = detections[0]
    # Angle must be negated (CW -> CCW)
    assert det.angle is not None
    assert abs(det.angle - (-angle_cw)) < 1e-6
    # obb_points must be shape (4, 2)
    assert det.obb_points is not None
    assert det.obb_points.shape == (4, 2)
    assert det.confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------

_CORE_DETECTION_MODULES = [
    "aquapose.core.detection",
    "aquapose.core.detection.stage",
    "aquapose.core.detection.backends",
    "aquapose.core.detection.backends.yolo",
    "aquapose.core.detection.backends.yolo_obb",
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
    from aquapose.core.types.frame_source import FrameSource

    nonexistent_weights = tmp_path / "nonexistent.pt"

    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.camera_ids = ["cam1"]

    with pytest.raises(FileNotFoundError, match=r"nonexistent\.pt"):
        DetectionStage(
            frame_source=mock_frame_source,
            detector_kind="yolo",
            weights_path=str(nonexistent_weights),
        )


# ---------------------------------------------------------------------------
# Batched detection
# ---------------------------------------------------------------------------


def test_batched_detection_calls_detect_batch_once(tmp_path: Path) -> None:
    """run() calls detect_batch() once per timestep instead of detect() per camera."""
    from aquapose.core.types.frame_source import FrameSource

    fake_weights = tmp_path / "model.pt"
    fake_weights.write_bytes(b"fake")

    # Build a stub FrameSource that yields 2 timesteps with 3 cameras each
    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.camera_ids = ["cam1", "cam2", "cam3"]

    frames_t0 = {
        "cam1": np.zeros((100, 100, 3), dtype=np.uint8),
        "cam2": np.zeros((100, 100, 3), dtype=np.uint8),
        "cam3": np.zeros((100, 100, 3), dtype=np.uint8),
    }
    frames_t1 = {
        "cam1": np.zeros((100, 100, 3), dtype=np.uint8),
        "cam2": np.zeros((100, 100, 3), dtype=np.uint8),
        "cam3": np.zeros((100, 100, 3), dtype=np.uint8),
    }
    mock_frame_source.__iter__ = MagicMock(
        return_value=iter([(0, frames_t0), (1, frames_t1)])
    )
    mock_frame_source.__enter__ = MagicMock(return_value=mock_frame_source)
    mock_frame_source.__exit__ = MagicMock(return_value=False)

    det = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)

    with (
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
            frame_source=mock_frame_source,
            detector_kind="yolo",
            weights_path=str(fake_weights),
        )

    # Mock the detect_batch method to return detections for each frame
    mock_detect_batch = MagicMock(side_effect=lambda frames: [[det] for _ in frames])
    stage._detector.detect_batch = mock_detect_batch  # type: ignore[attr-defined]
    stage._detector.detect = MagicMock()  # should NOT be called

    ctx = PipelineContext()
    result = stage.run(ctx)

    # detect_batch called once per timestep (2 timesteps)
    assert mock_detect_batch.call_count == 2
    # detect should never be called
    stage._detector.detect.assert_not_called()

    # Results are correct
    assert result.frame_count == 2
    assert len(result.detections) == 2
    for frame_dets in result.detections:
        assert set(frame_dets.keys()) == {"cam1", "cam2", "cam3"}
        for cam_dets in frame_dets.values():
            assert len(cam_dets) == 1
            assert cam_dets[0].confidence == 0.9


def test_batched_detection_handles_missing_cameras(tmp_path: Path) -> None:
    """run() gracefully handles missing cameras with empty detection lists."""
    from aquapose.core.types.frame_source import FrameSource

    fake_weights = tmp_path / "model.pt"
    fake_weights.write_bytes(b"fake")

    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.camera_ids = ["cam1", "cam2", "cam3"]

    # cam2 is missing from this timestep
    frames_t0 = {
        "cam1": np.zeros((100, 100, 3), dtype=np.uint8),
        "cam3": np.zeros((100, 100, 3), dtype=np.uint8),
    }
    mock_frame_source.__iter__ = MagicMock(return_value=iter([(0, frames_t0)]))
    mock_frame_source.__enter__ = MagicMock(return_value=mock_frame_source)
    mock_frame_source.__exit__ = MagicMock(return_value=False)

    det = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)

    with (
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
            frame_source=mock_frame_source,
            detector_kind="yolo",
            weights_path=str(fake_weights),
        )

    # detect_batch receives only 2 frames (cam1, cam3), returns 2 results
    mock_detect_batch = MagicMock(side_effect=lambda frames: [[det] for _ in frames])
    stage._detector.detect_batch = mock_detect_batch  # type: ignore[attr-defined]

    ctx = PipelineContext()
    result = stage.run(ctx)

    # detect_batch called with 2 frames (not 3)
    assert mock_detect_batch.call_count == 1
    batch_call_frames = mock_detect_batch.call_args[0][0]
    assert len(batch_call_frames) == 2

    # Missing camera gets empty list
    frame_dets = result.detections[0]
    assert frame_dets["cam2"] == []
    assert len(frame_dets["cam1"]) == 1
    assert len(frame_dets["cam3"]) == 1


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
    from aquapose.core.types.frame_source import FrameSource

    if synthetic_detections is None:
        synthetic_detections = [{"cam1": [], "cam2": []}]

    fake_weights = tmp_path / "model.pt"
    fake_weights.write_bytes(b"fake")

    # Build a stub FrameSource that yields synthetic frames
    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.camera_ids = sorted(["cam1", "cam2"])

    with (
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
            frame_source=mock_frame_source,
            detector_kind="yolo",
            weights_path=str(fake_weights),
        )

    # Replace run() with a deterministic stub that returns synthetic data
    camera_ids = sorted(["cam1", "cam2"])
    n_frames = len(synthetic_detections)

    def _stub_run(ctx: PipelineContext) -> PipelineContext:
        frames: list[dict[str, list[Detection]]] = list(synthetic_detections)
        ctx.detections = frames
        ctx.frame_count = n_frames
        ctx.camera_ids = camera_ids
        return ctx

    stage.run = _stub_run  # type: ignore[method-assign]
    return stage
