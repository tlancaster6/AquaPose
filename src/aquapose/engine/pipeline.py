"""PosePipeline orchestrator — single canonical entrypoint for pipeline execution.

PosePipeline wires together Stage instances, manages execution order, emits
lifecycle events via EventBus, and writes the serialized config as the first
artifact before any stage runs (ENG-06, ENG-08).
"""

from __future__ import annotations

import time
from pathlib import Path

from aquapose.engine.config import PipelineConfig, serialize_config
from aquapose.engine.events import (
    Event,
    PipelineComplete,
    PipelineFailed,
    PipelineStart,
    StageComplete,
    StageStart,
)
from aquapose.engine.observers import EventBus, Observer
from aquapose.engine.stages import PipelineContext, Stage


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

    def run(self) -> PipelineContext:
        """Execute all stages in order and return the accumulated context.

        Steps:

        1. Resolve and create the output directory.
        2. Write ``config.yaml`` (serialized config) as the first artifact.
        3. Emit ``PipelineStart``.
        4. For each stage: emit ``StageStart``, call ``stage.run(context)``,
           record timing in ``context.stage_timing``, emit ``StageComplete``.
        5. On success: emit ``PipelineComplete`` and return the context.
        6. On failure: emit ``PipelineFailed``, then re-raise the exception.

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
        context = PipelineContext()

        # --- 5. Execute stages in order -----------------------------------
        try:
            for i, stage in enumerate(self._stages):
                stage_name = type(stage).__name__
                self._bus.emit(StageStart(stage_name=stage_name, stage_index=i))
                stage_start = time.monotonic()
                context = stage.run(context)
                elapsed = time.monotonic() - stage_start
                context.stage_timing[stage_name] = elapsed
                self._bus.emit(
                    StageComplete(
                        stage_name=stage_name,
                        stage_index=i,
                        elapsed_seconds=elapsed,
                        summary={},
                    )
                )

        except Exception as exc:
            total_elapsed = time.monotonic() - pipeline_start
            self._bus.emit(
                PipelineFailed(
                    run_id=self._config.run_id,
                    error=str(exc),
                    elapsed_seconds=total_elapsed,
                )
            )
            raise

        # --- 6. Emit PipelineComplete -------------------------------------
        total_elapsed = time.monotonic() - pipeline_start
        self._bus.emit(
            PipelineComplete(
                run_id=self._config.run_id,
                elapsed_seconds=total_elapsed,
            )
        )

        return context
