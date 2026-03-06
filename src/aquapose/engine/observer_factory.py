"""Observer assembly factory for the AquaPose pipeline engine.

Provides :func:`build_observers` to construct the appropriate observer list
from a :class:`~aquapose.engine.config.PipelineConfig`, execution mode, and
optional additive observer names. Centralises observer business logic in the
engine layer rather than the CLI.
"""

from __future__ import annotations

from pathlib import Path

from aquapose.engine.config import PipelineConfig
from aquapose.engine.console_observer import ConsoleObserver
from aquapose.engine.diagnostic_observer import DiagnosticObserver
from aquapose.engine.observers import Observer
from aquapose.engine.timing import TimingObserver

__all__ = ["build_observers"]

# ---------------------------------------------------------------------------
# Observer name -> class mapping
# ---------------------------------------------------------------------------

_OBSERVER_MAP: dict[str, type] = {
    "timing": TimingObserver,
    "diagnostic": DiagnosticObserver,
    "console": ConsoleObserver,
}


def build_observers(
    config: PipelineConfig,
    mode: str,
    verbose: bool,
    total_stages: int,
    extra_observers: tuple[str, ...] = (),
    chunk_idx: int = 0,
    chunk_start: int = 0,
) -> list[Observer]:
    """Assemble the observer list based on execution mode and additive flags.

    Constructs a baseline observer set for the given *mode*, then appends any
    additional observers requested via *extra_observers*. The returned list is
    ready to pass to :class:`~aquapose.engine.pipeline.PosePipeline`.

    Mode behaviour:

    - ``"production"`` / ``"synthetic"``: ConsoleObserver + TimingObserver.
    - ``"diagnostic"``: ConsoleObserver + TimingObserver + DiagnosticObserver.
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
        chunk_idx: Zero-based chunk index forwarded to DiagnosticObserver so
            it writes caches to the correct ``diagnostics/chunk_NNN/`` subdirectory.
            Default 0 (single-chunk or non-diagnostic mode).
        chunk_start: Global frame index of the first frame in this chunk,
            forwarded to DiagnosticObserver so it writes the correct
            ``start_frame`` value in ``manifest.json``. Default 0.

    Returns:
        List of configured Observer instances for the pipeline.
    """
    observers: list[Observer] = [
        ConsoleObserver(verbose=verbose, total_stages=total_stages),
    ]

    output_dir = Path(config.output_dir)

    if mode in ("production", "synthetic", "benchmark"):
        observers.append(TimingObserver(output_path=output_dir / "timing.txt"))

    elif mode == "diagnostic":
        observers.append(TimingObserver(output_path=output_dir / "timing.txt"))
        observers.append(
            DiagnosticObserver(
                output_dir=config.output_dir,
                chunk_idx=chunk_idx,
                chunk_start=chunk_start,
            )
        )

    # Additive --add-observer flags
    for name in extra_observers:
        cls = _OBSERVER_MAP.get(name)
        if cls is None:
            continue
        if cls is TimingObserver:
            observers.append(TimingObserver(output_path=output_dir / "timing.txt"))
        elif cls is DiagnosticObserver:
            observers.append(
                DiagnosticObserver(
                    output_dir=config.output_dir,
                    chunk_idx=chunk_idx,
                    chunk_start=chunk_start,
                )
            )
        elif cls is ConsoleObserver:
            observers.append(
                ConsoleObserver(verbose=verbose, total_stages=total_stages)
            )

    return observers
