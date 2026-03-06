"""Interface tests for MidlineStage — Stage 4 of the AquaPose pipeline.

Validates:
- MidlineStage satisfies the Stage Protocol via structural typing
- run() correctly populates PipelineContext.annotated_detections
- Backend registry raises ValueError for unknown kinds
- SegmentationBackend stub instantiates and returns midline=None
- Import boundary (ENG-07): no engine/ runtime imports in core/midline/
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aquapose.core.context import PipelineContext, Stage
from aquapose.core.midline import AnnotatedDetection, MidlineStage
from aquapose.core.midline.backends import get_backend
from aquapose.core.midline.backends.segmentation import (
    SegmentationBackend,
)
from aquapose.core.types.detection import Detection

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
# Backend registry
# ---------------------------------------------------------------------------


def test_backend_registry_unknown_raises() -> None:
    """get_backend raises ValueError for an unrecognized backend kind."""
    with pytest.raises(ValueError, match="Unknown midline backend kind"):
        get_backend("nonexistent_backend")


# ---------------------------------------------------------------------------
# Stub backend behavior
# ---------------------------------------------------------------------------


def test_segmentation_backend_stub_returns_none_midlines() -> None:
    """SegmentationBackend stub instantiates cleanly and returns midline=None."""
    import numpy as np

    backend = SegmentationBackend()

    det1 = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    det2 = Detection(bbox=(60, 60, 30, 30), mask=None, area=900, confidence=0.8)
    frame_dets: dict[str, list[Detection]] = {"cam1": [det1, det2]}
    frames: dict[str, np.ndarray] = {"cam1": np.zeros((480, 640, 3), dtype=np.uint8)}

    result = backend.process_frame(
        frame_idx=0,
        frame_dets=frame_dets,
        frames=frames,
        camera_ids=["cam1"],
    )

    assert "cam1" in result
    assert len(result["cam1"]) == 2
    for ann in result["cam1"]:
        assert isinstance(ann, AnnotatedDetection)
        assert ann.midline is None, "Stub must return midline=None for all detections"


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------

_CORE_MIDLINE_MODULES = [
    "aquapose.core.midline",
    "aquapose.core.midline.stage",
    "aquapose.core.midline.types",
    "aquapose.core.midline.backends",
    "aquapose.core.midline.backends.segmentation",
    "aquapose.core.midline.backends.pose_estimation",
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
# Batched inference path
# ---------------------------------------------------------------------------


def test_batched_run_calls_process_batch(tmp_path: Path) -> None:
    """MidlineStage.run() calls process_batch once per frame via batched path."""
    import numpy as np

    from aquapose.core.types.crop import AffineCrop
    from aquapose.core.types.frame_source import FrameSource

    det1 = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    det2 = Detection(bbox=(60, 60, 30, 30), mask=None, area=900, confidence=0.8)

    detections: list[dict[str, list[Detection]]] = [
        {"cam1": [det1], "cam2": [det2]},
    ]

    # Build a mock frame source that yields one frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.__enter__ = MagicMock(return_value=mock_frame_source)
    mock_frame_source.__exit__ = MagicMock(return_value=False)
    mock_frame_source.__iter__ = MagicMock(
        return_value=iter([(0, {"cam1": dummy_frame, "cam2": dummy_frame})])
    )

    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    stage = MidlineStage(
        frame_source=mock_frame_source,
        calibration_path=calib_path,
        device="cpu",
    )

    # Mock the backend's _extract_crop and process_batch
    mock_crop = AffineCrop(
        image=np.zeros((64, 128, 3), dtype=np.uint8),
        M=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        crop_size=(128, 64),
        frame_shape=(480, 640),
    )
    stage._backend._extract_crop = MagicMock(return_value=mock_crop)  # type: ignore[union-attr]

    def _mock_process_batch(
        crops: list[AffineCrop],
        metadata: list[tuple[Detection, str, int]],
    ) -> list[AnnotatedDetection]:
        return [
            AnnotatedDetection(
                detection=det,
                mask=None,
                crop_region=None,
                midline=None,
                camera_id=cam_id,
                frame_index=frame_idx,
            )
            for det, cam_id, frame_idx in metadata
        ]

    stage._backend.process_batch = _mock_process_batch  # type: ignore[union-attr]

    ctx = PipelineContext()
    ctx.detections = detections
    ctx.camera_ids = ["cam1", "cam2"]

    result = stage.run(ctx)

    assert result is ctx
    assert ctx.annotated_detections is not None
    assert len(ctx.annotated_detections) == 1

    frame_result = ctx.annotated_detections[0]
    assert "cam1" in frame_result
    assert "cam2" in frame_result
    assert len(frame_result["cam1"]) == 1
    assert len(frame_result["cam2"]) == 1

    # Verify _extract_crop was called for each detection (2 total)
    assert stage._backend._extract_crop.call_count == 2  # type: ignore[union-attr]


def test_batched_run_redistributes_correctly(tmp_path: Path) -> None:
    """Batched results are redistributed to correct camera/detection slots."""
    import numpy as np

    from aquapose.core.types.crop import AffineCrop
    from aquapose.core.types.frame_source import FrameSource

    det_a = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    det_b = Detection(bbox=(60, 60, 30, 30), mask=None, area=900, confidence=0.8)
    det_c = Detection(bbox=(120, 120, 40, 40), mask=None, area=1600, confidence=0.7)

    # 2 dets on cam1, 1 det on cam2
    detections: list[dict[str, list[Detection]]] = [
        {"cam1": [det_a, det_b], "cam2": [det_c]},
    ]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.__enter__ = MagicMock(return_value=mock_frame_source)
    mock_frame_source.__exit__ = MagicMock(return_value=False)
    mock_frame_source.__iter__ = MagicMock(
        return_value=iter([(0, {"cam1": dummy_frame, "cam2": dummy_frame})])
    )

    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    stage = MidlineStage(
        frame_source=mock_frame_source,
        calibration_path=calib_path,
        device="cpu",
    )

    mock_crop = AffineCrop(
        image=np.zeros((64, 128, 3), dtype=np.uint8),
        M=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        crop_size=(128, 64),
        frame_shape=(480, 640),
    )
    stage._backend._extract_crop = MagicMock(return_value=mock_crop)  # type: ignore[union-attr]

    def _mock_process_batch(
        crops: list[AffineCrop],
        metadata: list[tuple[Detection, str, int]],
    ) -> list[AnnotatedDetection]:
        return [
            AnnotatedDetection(
                detection=det,
                mask=None,
                crop_region=None,
                midline=None,
                camera_id=cam_id,
                frame_index=frame_idx,
            )
            for det, cam_id, frame_idx in metadata
        ]

    stage._backend.process_batch = _mock_process_batch  # type: ignore[union-attr]

    ctx = PipelineContext()
    ctx.detections = detections
    ctx.camera_ids = ["cam1", "cam2"]

    result = stage.run(ctx)

    frame_result = result.annotated_detections[0]
    assert len(frame_result["cam1"]) == 2, "cam1 should have 2 annotated detections"
    assert len(frame_result["cam2"]) == 1, "cam2 should have 1 annotated detection"

    # Verify detection identity
    assert frame_result["cam1"][0].detection is det_a
    assert frame_result["cam1"][1].detection is det_b
    assert frame_result["cam2"][0].detection is det_c


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
    model inference.

    Args:
        tmp_path: Temporary directory for synthetic stub files.
        synthetic_annotated: Per-frame annotated detection lists to return from run().
            Defaults to a single empty frame for two cameras.

    Returns:
        Constructed and wired MidlineStage.
    """
    from aquapose.core.types.frame_source import FrameSource

    if synthetic_annotated is None:
        synthetic_annotated = [{"cam1": [], "cam2": []}]

    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.camera_ids = ["cam1", "cam2"]

    stage = MidlineStage(
        frame_source=mock_frame_source,
        calibration_path=calib_path,
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
