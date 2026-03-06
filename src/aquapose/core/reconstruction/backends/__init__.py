"""Backend registry for the Reconstruction stage.

Provides a factory function that resolves reconstruction backend kind strings
to configured backend instances. Only the DLT backend is supported.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create a reconstruction backend by kind name.

    Args:
        kind: Backend identifier. Supported values:

            - ``"dlt"``: Confidence-weighted DLT triangulation with single-pass
              outlier rejection. Accepted kwargs: ``calibration_path``,
              ``outlier_threshold``, ``n_control_points``,
              ``low_confidence_fraction``.

        **kwargs: Forwarded to the backend constructor.

    Returns:
        A configured backend instance with a
        ``reconstruct_frame(frame_idx, midline_set)`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
    """
    if kind == "dlt":
        from aquapose.core.reconstruction.backends.dlt import DltBackend

        return DltBackend(**kwargs)

    raise ValueError(
        f"Unknown reconstruction backend kind: {kind!r}. Supported kinds: ['dlt']"
    )
