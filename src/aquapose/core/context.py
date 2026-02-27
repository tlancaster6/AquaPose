"""Stage Protocol and PipelineContext — core data contracts for the pipeline.

Defines the structural typing contract that all pipeline stages must satisfy,
and the typed accumulator that flows data between stages.

These types live in core/ because they are pure data containers with no engine
logic. Placing them here eliminates all TYPE_CHECKING backdoors where core/
stage files previously needed to import from engine/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Stage(Protocol):
    """Structural protocol for all pipeline stages.

    Any class that implements a ``run(context: PipelineContext) -> PipelineContext``
    method is automatically a Stage — no inheritance required.

    Example::

        class MyStage:
            def run(self, context: PipelineContext) -> PipelineContext:
                context.camera_ids = ["cam1", "cam2"]
                return context
    """

    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute this stage, read from context, populate output fields, return context.

        Args:
            context: Accumulated pipeline state from prior stages.

        Returns:
            The same context object with this stage's output fields populated.

        """
        ...


@dataclass
class PipelineContext:
    """Typed accumulator for inter-stage data flow in the 5-stage pipeline.

    Stages populate their output field(s) and return the context. Fields are
    None until the producing stage has run. Use :meth:`get` to retrieve a
    field with a clear error if the upstream stage has not yet executed.

    Fields use generic stdlib types only to preserve the engine import
    boundary (ENG-07). Actual element types are documented below.

    The 5-stage data flow is:
    1. Detection  -> ``detections``
    2. Midline    -> ``annotated_detections``
    3. Association -> ``associated_bundles``
    4. Tracking   -> ``tracks``
    5. Reconstruction -> ``midlines_3d``

    Attributes:
        frame_count: Number of frames processed. Set by the Detection stage (Stage 1).
        camera_ids: Active camera IDs. Set by the Detection stage (Stage 1).
            Type: ``list[str]``
        detections: Stage 1 (Detection) output. Per-frame per-camera detection results.
            Indexed by frame_idx. Each entry is a dict mapping camera_id to list of
            Detection objects.
            Type: ``list[dict[str, list[Detection]]]``
        annotated_detections: Stage 2 (Midline) output. Detections enriched with 15-point
            2D midlines and half-widths. Same structure as ``detections`` but each
            Detection carries midline data.
            Type: ``list[dict[str, list[Detection]]]``
        associated_bundles: Stage 3 (Association) output. Cross-camera matched detection
            groups per fish, per frame. Each entry is a list of AssociationBundle objects
            (one per identified fish), where each bundle groups detections from multiple
            cameras and provides a triangulated 3D centroid.
            Type: ``list[list[AssociationBundle]]``
        tracks: Stage 4 (Tracking) output. Per-frame confirmed fish track objects with
            persistent fish IDs and lifecycle state.
            Type: ``list[list[FishTrack]]``
        midlines_3d: Stage 5 (Reconstruction) output. Per-frame 3D midline results.
            Each entry maps fish_id to a Spline3D (or Midline3D) object.
            Type: ``list[dict[int, Spline3D]]``
        stage_timing: Wall-clock seconds per stage, keyed by stage class name.

    """

    frame_count: int | None = None
    camera_ids: list[str] | None = None
    detections: list[dict[str, list]] | None = None
    annotated_detections: list[dict[str, list]] | None = None
    associated_bundles: list[list] | None = None
    tracks: list[list] | None = None
    midlines_3d: list[dict] | None = None
    stage_timing: dict[str, float] = field(default_factory=dict)

    def get(self, field_name: str) -> object:
        """Return the value of a field, raising ValueError if it is None.

        Args:
            field_name: Name of the PipelineContext field to retrieve.

        Returns:
            The field value (guaranteed non-None).

        Raises:
            ValueError: If the field is None, indicating the producing stage
                has not yet run.
            AttributeError: If ``field_name`` is not a valid field on this dataclass.

        """
        value = getattr(self, field_name)
        if value is None:
            raise ValueError(
                f"PipelineContext.{field_name} is None — the stage that produces "
                f"'{field_name}' has not run yet. Check stage ordering.",
            )
        return value
