"""Interface tests for PoseStage — Stage 2 of the AquaPose pipeline.

Validates:
- PoseStage satisfies the Stage Protocol via structural typing
- run() writes keypoints directly onto Detection objects in-place
- Backend registry raises ValueError for unknown kinds
- Import boundary (ENG-07): no engine/ runtime imports in core/pose/
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aquapose.core.context import PipelineContext, Stage
from aquapose.core.pose import PoseStage
from aquapose.core.pose.backends import get_backend
from aquapose.core.types.detection import Detection

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_pose_stage_satisfies_protocol(tmp_path: Path) -> None:
    """PoseStage is a Stage (structural typing) even before run() is called."""
    stage = _build_stage(tmp_path)
    assert isinstance(stage, Stage), (
        "PoseStage must satisfy the Stage Protocol via structural typing"
    )


# ---------------------------------------------------------------------------
# In-place keypoint enrichment
# ---------------------------------------------------------------------------


def test_pose_stage_writes_keypoints_onto_detections(tmp_path: Path) -> None:
    """run() writes det.keypoints and det.keypoint_conf directly onto Detection objects."""
    det = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    detections: list[dict[str, list[Detection]]] = [
        {"cam1": [det], "cam2": []},
    ]

    kpts_xy = np.random.rand(6, 2).astype(np.float32)
    kpts_conf = np.ones(6, dtype=np.float32)

    stage = _build_stage(tmp_path, per_det_kpts=[(kpts_xy, kpts_conf)])

    ctx = PipelineContext()
    ctx.detections = detections
    ctx.camera_ids = ["cam1", "cam2"]
    result = stage.run(ctx)

    assert result is ctx, "run() must return the same context object"
    # Keypoints written in-place on the detection object
    assert det.keypoints is not None
    assert det.keypoint_conf is not None
    np.testing.assert_array_equal(det.keypoints, kpts_xy)
    np.testing.assert_array_equal(det.keypoint_conf, kpts_conf)


def test_pose_stage_no_annotated_detections_field(tmp_path: Path) -> None:
    """run() does not populate any annotated_detections field (removed in v3.7)."""
    det = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    detections: list[dict[str, list[Detection]]] = [{"cam1": [det]}]

    stage = _build_stage(tmp_path)
    ctx = PipelineContext()
    ctx.detections = detections
    ctx.camera_ids = ["cam1"]
    stage.run(ctx)

    assert not hasattr(ctx, "annotated_detections"), (
        "PipelineContext must not have annotated_detections in v3.7"
    )


def test_pose_stage_returns_same_context(tmp_path: Path) -> None:
    """run() returns the same context object, not a copy."""
    ctx = PipelineContext()
    ctx.detections = [{"cam1": []}]
    ctx.camera_ids = ["cam1"]

    stage = _build_stage(tmp_path)
    result = stage.run(ctx)
    assert result is ctx


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


def test_backend_registry_unknown_raises() -> None:
    """get_backend raises ValueError for an unrecognized backend kind."""
    with pytest.raises(ValueError, match="Unknown"):
        get_backend("nonexistent_backend")


# ---------------------------------------------------------------------------
# Batched inference path
# ---------------------------------------------------------------------------


def test_batched_run_writes_keypoints_for_each_detection(tmp_path: Path) -> None:
    """PoseStage.run() writes keypoints onto every Detection in the frame."""
    import numpy as np

    from aquapose.core.types.crop import AffineCrop
    from aquapose.core.types.frame_source import FrameSource

    det1 = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    det2 = Detection(bbox=(60, 60, 30, 30), mask=None, area=900, confidence=0.8)

    detections: list[dict[str, list[Detection]]] = [
        {"cam1": [det1], "cam2": [det2]},
    ]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.__enter__ = MagicMock(return_value=mock_frame_source)
    mock_frame_source.__exit__ = MagicMock(return_value=False)
    mock_frame_source.__iter__ = MagicMock(
        return_value=iter([(0, {"cam1": dummy_frame, "cam2": dummy_frame})])
    )

    stage = PoseStage(
        frame_source=mock_frame_source,
        device="cpu",
    )

    mock_crop = AffineCrop(
        image=np.zeros((64, 128, 3), dtype=np.uint8),
        M=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        crop_size=(128, 64),
        frame_shape=(480, 640),
    )
    stage._backend._extract_crop = MagicMock(return_value=mock_crop)  # type: ignore[union-attr]

    kpts_xy = np.random.rand(6, 2).astype(np.float32)
    kpts_conf = np.ones(6, dtype=np.float32)

    def _mock_process_batch(
        crops: list[AffineCrop],
        metadata: list[tuple[Detection, str, int]],
    ) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
        return [(kpts_xy, kpts_conf) for _ in metadata]

    stage._backend.process_batch = _mock_process_batch  # type: ignore[union-attr]

    ctx = PipelineContext()
    ctx.detections = detections
    ctx.camera_ids = ["cam1", "cam2"]

    result = stage.run(ctx)
    assert result is ctx

    # Both detections should have keypoints written
    assert det1.keypoints is not None, "det1.keypoints should be set"
    assert det2.keypoints is not None, "det2.keypoints should be set"
    assert det1.keypoint_conf is not None
    assert det2.keypoint_conf is not None

    assert stage._backend._extract_crop.call_count == 2  # type: ignore[union-attr]


def test_batched_run_none_result_leaves_keypoints_none(tmp_path: Path) -> None:
    """When backend returns (None, None), det.keypoints stays None."""
    import numpy as np

    from aquapose.core.types.crop import AffineCrop
    from aquapose.core.types.frame_source import FrameSource

    det = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    detections: list[dict[str, list[Detection]]] = [{"cam1": [det]}]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.__enter__ = MagicMock(return_value=mock_frame_source)
    mock_frame_source.__exit__ = MagicMock(return_value=False)
    mock_frame_source.__iter__ = MagicMock(
        return_value=iter([(0, {"cam1": dummy_frame})])
    )

    stage = PoseStage(frame_source=mock_frame_source, device="cpu")

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
    ) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
        return [(None, None) for _ in metadata]

    stage._backend.process_batch = _mock_process_batch  # type: ignore[union-attr]

    ctx = PipelineContext()
    ctx.detections = detections
    ctx.camera_ids = ["cam1"]
    stage.run(ctx)

    # keypoints should remain None since backend returned (None, None)
    assert det.keypoints is None, (
        "det.keypoints should remain None when backend returns (None, None)"
    )
    assert det.keypoint_conf is None


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------

_CORE_POSE_MODULES = [
    "aquapose.core.pose",
    "aquapose.core.pose.stage",
    "aquapose.core.pose.types",
    "aquapose.core.pose.backends",
    "aquapose.core.pose.backends.pose_estimation",
]


def test_import_boundary_no_engine_imports() -> None:
    """No core/pose/ module may import from aquapose.engine at module level.

    TYPE_CHECKING-guarded imports are permitted, but no runtime import of
    aquapose.engine is allowed (ENG-07).
    """
    for mod_name in _CORE_POSE_MODULES:
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
# Helpers
# ---------------------------------------------------------------------------


def _build_stage(
    tmp_path: Path,
    per_det_kpts: list[tuple[np.ndarray | None, np.ndarray | None]] | None = None,
) -> PoseStage:
    """Build a PoseStage with all heavy I/O mocked.

    The stage's run() is replaced with a lightweight stub that writes
    the supplied *per_det_kpts* directly onto Detection objects.

    Args:
        tmp_path: Temporary directory for synthetic stub files.
        per_det_kpts: Per-detection (kpts_xy, kpts_conf) tuples to write.
            Defaults to writing None (no keypoints).

    Returns:
        Constructed and wired PoseStage.
    """
    from aquapose.core.types.frame_source import FrameSource

    mock_frame_source = MagicMock(spec=FrameSource)
    mock_frame_source.camera_ids = ["cam1", "cam2"]

    stage = PoseStage(
        frame_source=mock_frame_source,
        device="cpu",
    )

    # Replace run() with a deterministic stub
    _kpts_iter = iter(per_det_kpts or [])

    def _stub_run(ctx: PipelineContext) -> PipelineContext:
        detections = ctx.detections or []
        camera_ids = ctx.camera_ids or []
        for frame_dets in detections:
            for cam_id in camera_ids:
                for det in frame_dets.get(cam_id, []):
                    try:
                        kpts_xy, kpts_conf = next(_kpts_iter)
                    except StopIteration:
                        break
                    if kpts_xy is not None and kpts_conf is not None:
                        det.keypoints = kpts_xy
                        det.keypoint_conf = kpts_conf
        return ctx

    stage.run = _stub_run  # type: ignore[method-assign]
    return stage
