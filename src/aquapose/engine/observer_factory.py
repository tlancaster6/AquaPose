"""Observer assembly factory for the AquaPose pipeline engine.

Provides :func:`build_observers` to construct the appropriate observer list
from a :class:`~aquapose.engine.config.PipelineConfig`, execution mode, and
optional additive observer names. Centralises observer business logic in the
engine layer rather than the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aquapose.engine.animation_observer import Animation3DObserver
from aquapose.engine.config import PipelineConfig
from aquapose.engine.console_observer import ConsoleObserver
from aquapose.engine.diagnostic_observer import DiagnosticObserver
from aquapose.engine.hdf5_observer import HDF5ExportObserver
from aquapose.engine.observers import Observer
from aquapose.engine.overlay_observer import Overlay2DObserver
from aquapose.engine.timing import TimingObserver
from aquapose.engine.tracklet_trail_observer import TrackletTrailObserver

if TYPE_CHECKING:
    from aquapose.core.types.frame_source import FrameSource

__all__ = ["build_observers"]

# ---------------------------------------------------------------------------
# Observer name -> class mapping
# ---------------------------------------------------------------------------

_OBSERVER_MAP: dict[str, type] = {
    "timing": TimingObserver,
    "hdf5": HDF5ExportObserver,
    "overlay2d": Overlay2DObserver,
    "animation3d": Animation3DObserver,
    "diagnostic": DiagnosticObserver,
    "console": ConsoleObserver,
    "tracklet_trail": TrackletTrailObserver,
}


def build_observers(
    config: PipelineConfig,
    mode: str,
    verbose: bool,
    total_stages: int,
    extra_observers: tuple[str, ...] = (),
    frame_source: FrameSource | None = None,
) -> list[Observer]:
    """Assemble the observer list based on execution mode and additive flags.

    Constructs a baseline observer set for the given *mode*, then appends any
    additional observers requested via *extra_observers*. The returned list is
    ready to pass to :class:`~aquapose.engine.pipeline.PosePipeline`.

    Mode behaviour:
    - ``"production"`` / ``"synthetic"``: ConsoleObserver + TimingObserver +
      HDF5ExportObserver.
    - ``"diagnostic"``: All production observers plus Overlay2DObserver,
      Animation3DObserver, DiagnosticObserver, and TrackletTrailObserver.
    - ``"benchmark"``: ConsoleObserver + TimingObserver only.
    - Any other mode: ConsoleObserver only.

    Args:
        config: Pipeline configuration (used for output and video paths).
        mode: Execution mode preset (production, diagnostic, benchmark,
            synthetic).
        verbose: Whether to enable verbose console output.
        total_stages: Number of stages in the pipeline (used by
            ConsoleObserver for progress display).
        extra_observers: Additional observer names from ``--add-observer``
            flags. Supported names match the keys of the internal
            ``_OBSERVER_MAP``.
        frame_source: Optional FrameSource providing camera frames for
            Overlay2DObserver and TrackletTrailObserver. When None (synthetic
            mode or no video dir), those observers fall back to synthetic
            black frames.

    Returns:
        List of configured Observer instances for the pipeline.
    """
    observers: list[Observer] = [
        ConsoleObserver(verbose=verbose, total_stages=total_stages),
    ]

    output_dir = Path(config.output_dir)

    if mode in ("production", "synthetic"):
        observers.append(TimingObserver(output_path=output_dir / "timing.txt"))
        observers.append(HDF5ExportObserver(output_dir=config.output_dir))
        if mode == "synthetic":
            observers.append(
                Overlay2DObserver(
                    output_dir=config.output_dir,
                    calibration_path=config.calibration_path,
                    frame_source=frame_source,
                )
            )
            observers.append(Animation3DObserver(output_dir=config.output_dir))
            observers.append(
                TrackletTrailObserver(
                    output_dir=config.output_dir,
                    calibration_path=config.calibration_path,
                    frame_source=frame_source,
                )
            )

    elif mode == "diagnostic":
        observers.append(TimingObserver(output_path=output_dir / "timing.txt"))
        observers.append(HDF5ExportObserver(output_dir=config.output_dir))
        observers.append(
            Overlay2DObserver(
                output_dir=config.output_dir,
                calibration_path=config.calibration_path,
                frame_source=frame_source,
                show_bbox=True,
            )
        )
        observers.append(Animation3DObserver(output_dir=config.output_dir))
        observers.append(
            DiagnosticObserver(
                output_dir=config.output_dir,
            )
        )
        observers.append(
            TrackletTrailObserver(
                output_dir=config.output_dir,
                calibration_path=config.calibration_path,
                frame_source=frame_source,
            )
        )

    elif mode == "benchmark":
        observers.append(TimingObserver(output_path=output_dir / "timing.txt"))

    # Additive --add-observer flags
    for name in extra_observers:
        cls = _OBSERVER_MAP.get(name)
        if cls is None:
            continue
        if cls is TimingObserver:
            observers.append(TimingObserver(output_path=output_dir / "timing.txt"))
        elif cls is HDF5ExportObserver:
            observers.append(HDF5ExportObserver(output_dir=config.output_dir))
        elif cls is Overlay2DObserver:
            observers.append(
                Overlay2DObserver(
                    output_dir=config.output_dir,
                    calibration_path=config.calibration_path,
                    frame_source=frame_source,
                )
            )
        elif cls is Animation3DObserver:
            observers.append(Animation3DObserver(output_dir=config.output_dir))
        elif cls is DiagnosticObserver:
            observers.append(
                DiagnosticObserver(
                    output_dir=config.output_dir,
                )
            )
        elif cls is ConsoleObserver:
            observers.append(
                ConsoleObserver(verbose=verbose, total_stages=total_stages)
            )
        elif cls is TrackletTrailObserver:
            observers.append(
                TrackletTrailObserver(
                    output_dir=config.output_dir,
                    calibration_path=config.calibration_path,
                    frame_source=frame_source,
                )
            )

    return observers
