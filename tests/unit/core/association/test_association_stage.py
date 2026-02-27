"""Interface tests for AssociationStage — Stage 3 of the AquaPose pipeline.

Validates:
- AssociationStage satisfies the Stage Protocol via structural typing
- run() correctly populates PipelineContext.associated_bundles
- AssociationBundle has the correct fields
- Backend registry raises ValueError for unknown kinds
- Import boundary (ENG-07): no engine/ runtime imports in core/association/
- Empty detections produce empty bundle list
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.core.association import AssociationBundle, AssociationStage
from aquapose.core.association.backends import get_backend
from aquapose.core.context import PipelineContext, Stage
from aquapose.segmentation.detector import Detection

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_association_stage_satisfies_protocol(tmp_path: Path) -> None:
    """AssociationStage is a Stage (structural typing) even before run() is called."""
    stage = _build_stage(tmp_path)
    assert isinstance(stage, Stage), (
        "AssociationStage must satisfy the Stage Protocol via structural typing"
    )


# ---------------------------------------------------------------------------
# Bundle structure
# ---------------------------------------------------------------------------


def test_association_bundle_structure() -> None:
    """AssociationBundle has the expected fields with correct types."""
    bundle = AssociationBundle(
        fish_idx=0,
        centroid_3d=np.array([1.0, 2.0, 0.5]),
        camera_detections={"cam1": 0, "cam2": 1},
        n_cameras=2,
        reprojection_residual=3.5,
        confidence=0.95,
    )

    assert bundle.fish_idx == 0
    assert isinstance(bundle.centroid_3d, np.ndarray)
    assert bundle.centroid_3d.shape == (3,)
    assert isinstance(bundle.camera_detections, dict)
    assert bundle.n_cameras == 2
    assert isinstance(bundle.reprojection_residual, float)
    assert isinstance(bundle.confidence, float)


# ---------------------------------------------------------------------------
# Context population
# ---------------------------------------------------------------------------


def test_association_stage_populates_bundles(tmp_path: Path) -> None:
    """run() populates context.associated_bundles from context.detections."""
    det = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    synthetic_detections: list[dict[str, list[Detection]]] = [
        {"cam1": [det], "cam2": [det], "cam3": [det]},
        {"cam1": [det], "cam2": []},
    ]

    # Build expected bundles (two fish in frame 0, zero in frame 1)
    frame0_bundles = [
        AssociationBundle(
            fish_idx=0,
            centroid_3d=np.array([0.0, 0.0, 0.5]),
            camera_detections={"cam1": 0, "cam2": 0, "cam3": 0},
            n_cameras=3,
            reprojection_residual=2.0,
            confidence=1.0,
        )
    ]
    frame1_bundles: list[AssociationBundle] = []
    synthetic_bundles = [frame0_bundles, frame1_bundles]

    stage = _build_stage(tmp_path, synthetic_bundles=synthetic_bundles)

    ctx = PipelineContext()
    ctx.detections = synthetic_detections
    result = stage.run(ctx)

    assert result is ctx, "run() must return the same context object"
    assert ctx.associated_bundles is not None
    assert isinstance(ctx.associated_bundles, list)
    assert len(ctx.associated_bundles) == len(synthetic_detections)

    # Validate per-frame structure
    for frame_bundles in ctx.associated_bundles:
        assert isinstance(frame_bundles, list)

    # Frame 0: one bundle; Frame 1: no bundles
    assert len(ctx.associated_bundles[0]) == 1
    assert len(ctx.associated_bundles[1]) == 0


def test_association_stage_reads_annotated_detections(tmp_path: Path) -> None:
    """run() also accepts context.annotated_detections (Stage 2 output)."""
    from aquapose.core.midline.types import AnnotatedDetection

    det = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    ad = AnnotatedDetection(detection=det, camera_id="cam1", frame_index=0)
    synthetic_annotated: list[dict[str, list[AnnotatedDetection]]] = [
        {"cam1": [ad], "cam2": [ad], "cam3": [ad]},
    ]

    expected_bundle = AssociationBundle(
        fish_idx=0,
        centroid_3d=np.array([0.0, 0.0, 0.5]),
        camera_detections={"cam1": 0},
        n_cameras=1,
        reprojection_residual=0.0,
        confidence=1.0,
    )

    stage = _build_stage(tmp_path, synthetic_bundles=[[expected_bundle]])

    ctx = PipelineContext()
    ctx.annotated_detections = synthetic_annotated
    result = stage.run(ctx)

    assert result is ctx
    assert ctx.associated_bundles is not None
    assert len(ctx.associated_bundles) == 1


def test_association_stage_no_detections_raises(tmp_path: Path) -> None:
    """run() raises ValueError if neither detections nor annotated_detections is set."""
    stage = _build_stage(tmp_path)
    ctx = PipelineContext()

    with pytest.raises(
        ValueError, match=r"context\.detections or context\.annotated_detections"
    ):
        stage.run(ctx)


# ---------------------------------------------------------------------------
# Empty detections
# ---------------------------------------------------------------------------


def test_empty_detections_produces_empty_bundles(tmp_path: Path) -> None:
    """A frame with no detections produces an empty bundle list."""
    empty_detections: list[dict[str, list[Detection]]] = [
        {"cam1": [], "cam2": [], "cam3": []},
    ]

    stage = _build_stage(tmp_path, synthetic_bundles=[[]])

    ctx = PipelineContext()
    ctx.detections = empty_detections
    stage.run(ctx)

    assert ctx.associated_bundles is not None
    assert len(ctx.associated_bundles) == 1
    assert ctx.associated_bundles[0] == []


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


def test_backend_registry_unknown_raises() -> None:
    """get_backend raises ValueError for an unrecognized backend kind."""
    with pytest.raises(ValueError, match="Unknown association backend kind"):
        get_backend("nonexistent_backend")


def test_backend_registry_ransac_centroid_constructs(tmp_path: Path) -> None:
    """get_backend('ransac_centroid') constructs without error (mocked calibration)."""
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    mock_calib = MagicMock()
    mock_calib.cameras = {}
    mock_calib.water_z = 0.0
    mock_calib.interface_normal = [0.0, 0.0, 1.0]
    mock_calib.n_air = 1.0
    mock_calib.n_water = 1.33

    with patch(
        "aquapose.calibration.loader.load_calibration_data",
        return_value=mock_calib,
    ):
        backend = get_backend("ransac_centroid", calibration_path=str(calib_path))
    assert backend is not None


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


_CORE_ASSOCIATION_MODULES = [
    "aquapose.core.association",
    "aquapose.core.association.stage",
    "aquapose.core.association.types",
    "aquapose.core.association.backends",
    "aquapose.core.association.backends.ransac_centroid",
]


def test_import_boundary_no_engine_imports() -> None:
    """No core/association/ module may import from aquapose.engine at module level.

    TYPE_CHECKING-guarded imports are permitted, but no runtime import of
    aquapose.engine is allowed (ENG-07).
    """
    for mod_name in _CORE_ASSOCIATION_MODULES:
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
    synthetic_bundles: list[list[AssociationBundle]] | None = None,
) -> AssociationStage:
    """Build an AssociationStage with all heavy I/O mocked.

    The calibration loading and RefractiveProjectionModel construction are
    patched out. The backend's ``associate_frame`` is replaced with a mock
    that returns ``synthetic_bundles`` frames in order.

    Args:
        tmp_path: Temporary directory for fake calibration file.
        synthetic_bundles: Per-frame bundle lists to return from the mock
            backend. If None, returns empty lists for each call.

    Returns:
        A configured AssociationStage ready for testing.
    """
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text("{}")

    mock_calib = MagicMock()
    mock_calib.cameras = {}
    mock_calib.water_z = 0.0
    mock_calib.interface_normal = [0.0, 0.0, 1.0]
    mock_calib.n_air = 1.0
    mock_calib.n_water = 1.33

    with patch(
        "aquapose.calibration.loader.load_calibration_data",
        return_value=mock_calib,
    ):
        stage = AssociationStage(calibration_path=str(calib_path))

    # Replace backend.associate_frame with a side_effect that returns per-frame data
    if synthetic_bundles is not None:
        call_results = iter(synthetic_bundles)
        stage._backend.associate_frame = MagicMock(  # type: ignore[union-attr]
            side_effect=lambda dets: next(call_results)
        )
    else:
        stage._backend.associate_frame = MagicMock(return_value=[])  # type: ignore[union-attr]

    return stage
