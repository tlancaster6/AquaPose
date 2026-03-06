"""Visualization utilities for post-run diagnostic rendering from cached chunk data.

Provides CLI-facing functions that load chunk caches and produce overlay videos,
3D animations, and tracklet trail videos without re-running the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aquapose.evaluation.viz.animation import generate_animation
from aquapose.evaluation.viz.overlay import generate_overlay
from aquapose.evaluation.viz.trails import generate_trails

logger = logging.getLogger(__name__)


def generate_all(
    run_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Path | Exception]:
    """Attempt every visualization and skip gracefully on failure.

    Calls generate_overlay, generate_animation, and generate_trails in order.
    Each visualization is wrapped in a try/except — failures are logged as
    warnings and recorded in the returned status dict rather than propagated.

    Args:
        run_dir: Path to the pipeline run directory.
        output_dir: Directory for all outputs. Defaults to ``{run_dir}/viz/``.

    Returns:
        Dict mapping visualization name to either the output Path (success)
        or the Exception that was raised (failure).
    """
    results: dict[str, Path | Exception] = {}

    for name, fn in [
        ("overlay", lambda: generate_overlay(run_dir, output_dir)),
        ("animation", lambda: generate_animation(run_dir, output_dir)),
        ("trails", lambda: generate_trails(run_dir, output_dir)),
    ]:
        try:
            out = fn()
            results[name] = out
        except Exception as exc:
            logger.warning("Visualization '%s' failed: %s", name, exc, exc_info=True)
            results[name] = exc

    return results


__all__ = [
    "generate_all",
    "generate_animation",
    "generate_overlay",
    "generate_trails",
]
