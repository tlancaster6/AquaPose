"""PosePipeline orchestrator — single canonical entrypoint for pipeline execution.

PosePipeline wires together Stage instances, manages execution order, emits
lifecycle events via EventBus, and writes the serialized config as the first
artifact before any stage runs (ENG-06, ENG-08).

The :func:`build_stages` factory is the canonical way to construct all 5 stages
from a :class:`~aquapose.engine.config.PipelineConfig` and wire them into
:class:`PosePipeline`. Stage 3 uses ``AssociationStage`` from
``aquapose.core.association`` for Leiden-based cross-camera clustering.

Stage 2 (TrackingStage) uses a different ``run()`` signature —
``(context, carry) -> (context, carry)`` — because it maintains per-camera
OC-SORT state across batches. The pipeline runner detects this stage via
``isinstance(stage, TrackingStage)`` and dispatches accordingly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from aquapose.calibration.luts import load_forward_luts, load_inverse_luts
from aquapose.core.context import PipelineContext, Stage
from aquapose.engine.config import PipelineConfig, serialize_config

if TYPE_CHECKING:
    from aquapose.core.types.frame_source import VideoFrameSource
from aquapose.engine.events import (
    Event,
    PipelineComplete,
    PipelineFailed,
    PipelineStart,
    StageComplete,
    StageStart,
)
from aquapose.engine.observers import EventBus, Observer

logger = logging.getLogger(__name__)

# Maps stage class name to the PipelineContext output fields it populates.
# Used by PosePipeline.run() to detect pre-populated stages when initial_context
# is provided, enabling skip of already-completed stages.
_STAGE_OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "DetectionStage": ("frame_count", "camera_ids", "detections"),
    "SyntheticDataStage": (
        "frame_count",
        "camera_ids",
        "detections",
        "annotated_detections",
    ),
    "TrackingStage": ("tracks_2d",),
    "AssociationStage": ("tracklet_groups",),
    "MidlineStage": ("annotated_detections",),
    "ReconstructionStage": ("midlines_3d",),
}


class PosePipeline:
    """Orchestrates the AquaPose pipeline by executing stages in order.

    PosePipeline is the single canonical entrypoint (``run()``) that:

    1. Creates the output directory and writes the serialized config as the
       first artifact (``config.yaml``) before any stage runs.
    2. Emits lifecycle events (PipelineStart, StageStart, StageComplete,
       PipelineComplete, PipelineFailed) via an internal EventBus.
    3. Executes stages in the order provided, passing a shared
       :class:`PipelineContext` through each stage.
    4. Records per-stage wall-clock timing in ``context.stage_timing``.

    Observers are purely additive — removing all observers produces identical
    execution results.

    Example::

        config = load_config(run_id="test_run", cli_overrides={"output_dir": "/tmp/test"})
        pipeline = PosePipeline(
            stages=[DetectionStage(), SegmentationStage()],
            config=config,
            observers=[TimingObserver()],
        )
        context = pipeline.run()

    Args:
        stages: Ordered list of Stage instances to execute. Stages run in the
            order given — first stage first, last stage last.
        config: Frozen PipelineConfig for this run. Serialized as the first
            artifact before any stage executes.
        observers: Optional list of Observer instances. Each observer is
            subscribed to the base ``Event`` type so it receives all events.

    """

    def __init__(
        self,
        stages: list[Stage],
        config: PipelineConfig,
        observers: list[Observer] | None = None,
    ) -> None:
        self._stages = list(stages)
        self._config = config
        self._bus = EventBus()

        if observers:
            for observer in observers:
                self._bus.subscribe(Event, observer)

    def add_observer(
        self,
        observer: Observer,
        event_type: type[Event] = Event,
    ) -> None:
        """Subscribe *observer* to receive *event_type* events.

        Args:
            observer: Any object satisfying the Observer protocol.
            event_type: Event class to subscribe to.  Defaults to ``Event``
                (receives all events).

        """
        self._bus.subscribe(event_type, observer)

    def remove_observer(
        self,
        observer: Observer,
        event_type: type[Event] = Event,
    ) -> None:
        """Unsubscribe *observer* from *event_type* events.

        No-op if the observer is not currently subscribed.

        Args:
            observer: Observer instance to remove.
            event_type: Event class to unsubscribe from.  Defaults to ``Event``.

        """
        self._bus.unsubscribe(event_type, observer)

    def run(self, initial_context: PipelineContext | None = None) -> PipelineContext:
        """Execute all stages in order and return the accumulated context.

        When *initial_context* is provided, stages whose output fields are
        already populated (non-None) in that context are skipped automatically.
        Skipped stages still emit ``StageStart`` and ``StageComplete`` events
        (with ``elapsed_seconds=0.0`` and ``summary={"skipped": True}``) so that
        observers see a complete event timeline.

        Steps:

        1. Resolve and create the output directory.
        2. Write ``config.yaml`` (serialized config) as the first artifact.
        3. Emit ``PipelineStart``.
        4. For each stage: if outputs are already populated and
           *initial_context* was provided, emit skipped events and continue;
           otherwise emit ``StageStart``, call ``stage.run(context)``,
           record timing in ``context.stage_timing``, emit ``StageComplete``.
        5. On success: emit ``PipelineComplete`` and return the context.
        6. On failure: emit ``PipelineFailed``, then re-raise the exception.

        Args:
            initial_context: Optional pre-populated PipelineContext. When
                provided, stages whose output fields are all non-None are
                skipped and their carry_forward state is extracted from this
                context. Defaults to None (fresh context).

        Returns:
            The final :class:`PipelineContext` after all stages have run.

        Raises:
            Exception: Re-raises any exception thrown by a stage after emitting
                ``PipelineFailed``.

        """
        pipeline_start = time.monotonic()

        # --- 1. Resolve and create output directory -----------------------
        output_dir = Path(self._config.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 2. Write config artifact (ENG-08) ----------------------------
        config_path = output_dir / "config.yaml"
        config_path.write_text(serialize_config(self._config), encoding="utf-8")

        # --- 3. Emit PipelineStart ----------------------------------------
        self._bus.emit(PipelineStart(run_id=self._config.run_id, config=self._config))

        # --- 4. Initialize context ----------------------------------------
        context = initial_context if initial_context is not None else PipelineContext()

        # --- 5. Execute stages in order -----------------------------------
        # TrackingStage uses a different run() signature:
        # (context, carry) -> (context, carry).
        # We maintain carry state across all batches on the pipeline instance.
        carry: object | None = None
        if initial_context is not None:
            carry = initial_context.carry_forward

        try:
            for i, stage in enumerate(self._stages):
                from aquapose.core.tracking import TrackingStage

                stage_name = type(stage).__name__
                output_fields = _STAGE_OUTPUT_FIELDS.get(stage_name, ())
                already_populated = bool(output_fields) and all(
                    getattr(context, f, None) is not None for f in output_fields
                )

                if already_populated:
                    logger.info(
                        "Skipping %s -- outputs already populated in context",
                        stage_name,
                    )
                    self._bus.emit(StageStart(stage_name=stage_name, stage_index=i))
                    self._bus.emit(
                        StageComplete(
                            stage_name=stage_name,
                            stage_index=i,
                            elapsed_seconds=0.0,
                            summary={"skipped": True},
                            context=context,
                        ),
                    )
                    continue

                self._bus.emit(StageStart(stage_name=stage_name, stage_index=i))
                stage_start = time.monotonic()
                if isinstance(stage, TrackingStage):
                    context, carry = stage.run(context, carry)  # type: ignore[arg-type]
                    context.carry_forward = carry
                else:
                    context = stage.run(context)
                elapsed = time.monotonic() - stage_start
                context.stage_timing[stage_name] = elapsed
                self._bus.emit(
                    StageComplete(
                        stage_name=stage_name,
                        stage_index=i,
                        elapsed_seconds=elapsed,
                        summary={},
                        context=context,
                    ),
                )

        except Exception as exc:
            total_elapsed = time.monotonic() - pipeline_start
            self._bus.emit(
                PipelineFailed(
                    run_id=self._config.run_id,
                    error=str(exc),
                    elapsed_seconds=total_elapsed,
                ),
            )
            raise

        # --- 6. Emit PipelineComplete -------------------------------------
        total_elapsed = time.monotonic() - pipeline_start
        self._bus.emit(
            PipelineComplete(
                run_id=self._config.run_id,
                elapsed_seconds=total_elapsed,
                context=context,
            ),
        )

        return context


# ---------------------------------------------------------------------------
# Stage factory
# ---------------------------------------------------------------------------


def build_stages(
    config: PipelineConfig, frame_source: VideoFrameSource | None = None
) -> list:
    """Construct pipeline stages from a :class:`PipelineConfig`.

    This factory is the canonical way to wire stages into :class:`PosePipeline`.
    It imports all stage classes from ``aquapose.core`` (never the reverse) and
    constructs each stage from its corresponding sub-config in *config*.

    v2.1 pipeline ordering: Detection → 2D Tracking → Association → Midline →
    Reconstruction. Stage 3 (AssociationStage) performs Leiden-based cross-camera
    tracklet clustering using ray-ray geometry and ghost-point penalties.

    When ``config.mode == "synthetic"``, returns a 4-stage list with
    SyntheticDataStage replacing both DetectionStage and MidlineStage:
    SyntheticDataStage → TrackingStage → AssociationStage → ReconstructionStage.

    For all other modes, returns the full 5-stage list:
    DetectionStage → TrackingStage → AssociationStage → MidlineStage → ReconstructionStage.

    Example::

        stages = build_stages(config)
        pipeline = PosePipeline(stages=stages, config=config)
        context = pipeline.run()

    Args:
        config: Frozen pipeline config providing calibration path, model paths,
            backend selection, and all stage-specific parameters.
        frame_source: Optional pre-created FrameSource to inject into detection
            and midline stages. When None (default), a new VideoFrameSource is
            created internally for non-synthetic modes.

    Returns:
        Ordered list of stage instances (mix of Stage protocol and stub types).

    Raises:
        FileNotFoundError: If required paths in *config* do not exist.
        ValueError: If any backend kind string is unrecognized.

    """
    from aquapose.core import (
        DetectionStage,
        MidlineStage,
        ReconstructionStage,
        SyntheticDataStage,
    )
    from aquapose.core.association import AssociationStage
    from aquapose.core.tracking import TrackingStage
    from aquapose.core.types import VideoFrameSource

    STAGE_NAMES: dict[str, type] = {
        "detection": DetectionStage,
        "synthetic": SyntheticDataStage,
        "tracking": TrackingStage,
        "association": AssociationStage,
        "midline": MidlineStage,
        "reconstruction": ReconstructionStage,
    }

    tracking_stage = TrackingStage(config=config.tracking)

    reconstruction_stage = ReconstructionStage(
        calibration_path=config.calibration_path,
        backend=config.reconstruction.backend,
        min_cameras=config.reconstruction.min_cameras,
        max_interp_gap=config.reconstruction.max_interp_gap,
        n_control_points=config.reconstruction.n_control_points,
        plane_projection_enabled=config.reconstruction.plane_projection.enabled,
    )

    def _truncate(stages: list) -> list:
        """Truncate stage list at ``config.stop_after`` if set."""
        if config.stop_after is None:
            return stages
        target_cls = STAGE_NAMES.get(config.stop_after)
        if target_cls is None:
            raise ValueError(
                f"Unknown stop_after stage {config.stop_after!r}. "
                f"Valid values: {sorted(STAGE_NAMES)}"
            )
        for i, stage in enumerate(stages):
            if isinstance(stage, target_cls):
                return stages[: i + 1]
        # Target stage not in the list (e.g. "midline" in synthetic mode)
        return stages

    def _check_luts_if_needed(stages: list) -> None:
        """Raise early if association is in the stage list but LUTs are missing."""
        has_association = any(isinstance(s, AssociationStage) for s in stages)
        if has_association and config.calibration_path:
            fwd = load_forward_luts(config.calibration_path, config.lut)
            if fwd is None:
                raise FileNotFoundError(
                    "LUTs not found. Run: aquapose prep generate-luts --config <path>"
                )
            inv = load_inverse_luts(config.calibration_path, config.lut)
            if inv is None:
                raise FileNotFoundError(
                    "LUTs not found. Run: aquapose prep generate-luts --config <path>"
                )

    # --- Synthetic mode: SyntheticDataStage replaces Detection + Midline
    if config.mode == "synthetic":
        synthetic_stage = SyntheticDataStage(
            calibration_path=config.calibration_path,
            synthetic_config=config.synthetic,
            n_points=config.n_sample_points,
        )

        stages = _truncate(
            [
                synthetic_stage,
                tracking_stage,
                AssociationStage(config),
                reconstruction_stage,
            ]
        )
        _check_luts_if_needed(stages)
        return stages

    # --- Production (and all other) modes: full 5-stage pipeline
    if frame_source is None:
        frame_source = VideoFrameSource(
            video_dir=config.video_dir,
            calibration_path=config.calibration_path,
        )

    detection_stage = DetectionStage(
        frame_source=frame_source,
        detector_kind=config.detection.detector_kind,
        detection_batch_frames=config.detection.detection_batch_frames,
        weights_path=config.detection.weights_path,
        device=config.device,
    )

    midline_stage = MidlineStage(
        frame_source=frame_source,
        calibration_path=config.calibration_path,
        weights_path=config.midline.weights_path,
        confidence_threshold=config.midline.confidence_threshold,
        backend=config.midline.backend,
        device=config.device,
        n_points=config.n_sample_points,
        min_area=config.midline.min_area,
        lut_config=config.lut,
        midline_config=config.midline,
        crop_size=tuple(config.detection.crop_size),
        midline_batch_crops=config.midline.midline_batch_crops,
    )

    stages = _truncate(
        [
            detection_stage,
            tracking_stage,
            AssociationStage(config),
            midline_stage,
            reconstruction_stage,
        ]
    )
    _check_luts_if_needed(stages)
    return stages
