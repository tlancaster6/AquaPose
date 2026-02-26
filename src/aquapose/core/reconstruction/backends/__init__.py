"""Backend registry for the Reconstruction stage.

Provides a factory function that resolves reconstruction backend kind strings
to configured backend instances. Supports "triangulation" (default) and
"curve_optimizer".
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create a reconstruction backend by kind name.

    Args:
        kind: Backend identifier. Supported values:

            - ``"triangulation"``: Multi-view RANSAC triangulation + B-spline
              fitting backend. Delegates to ``triangulate_midlines()``.
            - ``"curve_optimizer"``: Correspondence-free 3D B-spline optimization
              via chamfer distance. Delegates to ``CurveOptimizer.optimize_midlines()``.

        **kwargs: Forwarded to the backend constructor. For ``"triangulation"``,
            accepted kwargs are: ``calibration_path``, ``skip_camera_id``,
            ``inlier_threshold``, ``snap_threshold``, ``max_depth``.
            For ``"curve_optimizer"``, same plus optimizer-specific params.

    Returns:
        A configured backend instance with a
        ``reconstruct_frame(frame_idx, midline_set)`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
    """
    if kind == "triangulation":
        from aquapose.core.reconstruction.backends.triangulation import (
            TriangulationBackend,
        )

        return TriangulationBackend(**kwargs)

    if kind == "curve_optimizer":
        from aquapose.core.reconstruction.backends.curve_optimizer import (
            CurveOptimizerBackend,
        )

        return CurveOptimizerBackend(**kwargs)

    raise ValueError(
        f"Unknown reconstruction backend kind: {kind!r}. "
        f"Supported kinds: ['triangulation', 'curve_optimizer']"
    )
