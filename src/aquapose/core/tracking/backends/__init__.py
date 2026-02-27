"""Backend registry for the Tracking stage.

Provides a factory function that resolves tracking backend kind strings to
configured backend instances. Currently only "hungarian" is implemented.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create a tracking backend by kind name.

    Args:
        kind: Backend identifier. Supported values:

            - ``"hungarian"``: Hungarian-3D temporal tracking backend wrapping
              the v1.0 ``FishTracker`` for exact behavioral equivalence.

        **kwargs: Forwarded to the backend constructor. For ``"hungarian"``,
            accepted kwargs are: ``calibration_path``, ``expected_count``,
            ``min_hits``, ``max_age``, ``reprojection_threshold``,
            ``birth_interval``, ``min_cameras_birth``, ``velocity_damping``,
            ``velocity_window``.

    Returns:
        A configured backend instance with a
        ``track_frame(frame_idx, bundles)`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
    """
    if kind == "hungarian":
        from aquapose.core.tracking.backends.hungarian import HungarianBackend

        return HungarianBackend(**kwargs)

    raise ValueError(
        f"Unknown tracking backend kind: {kind!r}. Supported kinds: ['hungarian']"
    )
