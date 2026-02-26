"""Tests for Stage protocol conformance and PipelineContext behavior."""

from __future__ import annotations

import importlib
import inspect

import pytest

from aquapose.engine.stages import PipelineContext, Stage

# ---------------------------------------------------------------------------
# Stage structural typing
# ---------------------------------------------------------------------------


def test_stage_structural_typing() -> None:
    """A plain class with run() is recognized as Stage without inheritance (ENG-01)."""

    class MyStage:
        def run(self, context: PipelineContext) -> PipelineContext:
            return context

    instance = MyStage()
    assert isinstance(instance, Stage), (
        "MyStage should satisfy Stage protocol via structural typing"
    )


def test_non_conforming_class_rejected() -> None:
    """A class without run() is NOT a Stage."""

    class NotAStage:
        def process(self, context: PipelineContext) -> PipelineContext:
            return context

    instance = NotAStage()
    assert not isinstance(instance, Stage), (
        "NotAStage (no run() method) should not satisfy Stage protocol"
    )


# ---------------------------------------------------------------------------
# PipelineContext accumulation
# ---------------------------------------------------------------------------


def test_pipeline_context_accumulates_fields() -> None:
    """Setting multiple fields does not overwrite each other (ENG-02)."""
    ctx = PipelineContext()
    ctx.detections = [{"cam1": []}]
    ctx.tracks = [[]]

    assert ctx.detections == [{"cam1": []}]
    assert ctx.tracks == [[]]


def test_pipeline_context_defaults_none() -> None:
    """All Optional fields default to None on a fresh PipelineContext."""
    ctx = PipelineContext()

    assert ctx.frame_count is None
    assert ctx.camera_ids is None
    # Stage 1 — Detection
    assert ctx.detections is None
    # Stage 2 — Midline
    assert ctx.annotated_detections is None
    # Stage 3 — Association
    assert ctx.associated_bundles is None
    # Stage 4 — Tracking
    assert ctx.tracks is None
    # Stage 5 — Reconstruction
    assert ctx.midlines_3d is None
    # stage_timing should be an empty dict (not None)
    assert ctx.stage_timing == {}


def test_pipeline_context_get_raises_on_none() -> None:
    """get() raises ValueError with a descriptive message when field is None."""
    ctx = PipelineContext()

    with pytest.raises(ValueError, match="detections") as exc_info:
        ctx.get("detections")

    # Message should explain what to do
    assert (
        "stage" in str(exc_info.value).lower() or "run" in str(exc_info.value).lower()
    )


def test_pipeline_context_get_returns_value() -> None:
    """get() returns the field value when it is not None."""
    ctx = PipelineContext()
    ctx.detections = [{"cam1": [{"id": 0}]}]

    result = ctx.get("detections")
    assert result == [{"cam1": [{"id": 0}]}]


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------

_FORBIDDEN_MODULES = [
    "aquapose.calibration",
    "aquapose.segmentation",
    "aquapose.reconstruction",
    "aquapose.tracking",
    "aquapose.mesh",
    "aquapose.initialization",
    "aquapose.io",
    "aquapose.visualization",
    "aquapose.synthetic",
]


def test_import_boundary_no_computation_imports() -> None:
    """engine.stages must not import from any computation module (ENG-07).

    We inspect the source code directly to catch both runtime and TYPE_CHECKING
    imports, since the plan explicitly forbids even TYPE_CHECKING exceptions.
    """
    module = importlib.import_module("aquapose.engine.stages")
    source = inspect.getsource(module)

    for forbidden in _FORBIDDEN_MODULES:
        assert forbidden not in source, (
            f"engine.stages imports from forbidden module '{forbidden}'. "
            "Engine package must only use stdlib types (ENG-07)."
        )
