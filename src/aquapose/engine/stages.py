"""Stage Protocol and PipelineContext dataclass for the engine pipeline.

Defines the structural typing contract that all pipeline stages must satisfy,
and the typed accumulator that flows data between stages.
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
    """Typed accumulator for inter-stage data flow.

    Stages populate their output field(s) and return the context. Fields are
    None until the producing stage has run. Use :meth:`get` to retrieve a
    field with a clear error if the upstream stage has not yet executed.

    Fields use generic stdlib types only to preserve the engine import
    boundary (ENG-07). Actual element types are documented below.

    Attributes:
        frame_count: Number of frames processed. Set by the detection stage.
        camera_ids: Active camera IDs. Set by the detection stage.
            Type: ``list[str]``
        detections: Per-frame detection results. Indexed by frame_idx.
            Each entry is a dict mapping camera_id to list of Detection objects.
            Type: ``list[dict[str, list[Detection]]]``
        masks: Per-frame segmentation masks. Indexed by frame_idx.
            Each entry is a dict mapping camera_id to list of (mask_uint8, CropRegion) tuples.
            Type: ``list[dict[str, list[tuple[np.ndarray, CropRegion]]]]``
        tracks: Per-frame tracking results. Indexed by frame_idx.
            Each entry is a list of confirmed FishTrack objects.
            Type: ``list[list[FishTrack]]``
        midline_sets: Per-frame 2D midline sets. Indexed by frame_idx.
            Each entry is a MidlineSet (dict[int, dict[str, Midline2D]]).
            Type: ``list[MidlineSet]``
        midlines_3d: Per-frame 3D midline results. Indexed by frame_idx.
            Each entry maps fish_id to Midline3D.
            Type: ``list[dict[int, Midline3D]]``
        stage_timing: Wall-clock seconds per stage, keyed by stage name.
    """

    frame_count: int | None = None
    camera_ids: list[str] | None = None
    detections: list[dict[str, list]] | None = None
    masks: list[dict[str, list]] | None = None
    tracks: list[list] | None = None
    midline_sets: list | None = None
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
                f"'{field_name}' has not run yet. Check stage ordering."
            )
        return value
