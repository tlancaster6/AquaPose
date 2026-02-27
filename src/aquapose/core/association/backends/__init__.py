"""Backend registry for the Association stage.

Provides a factory function that resolves association backend kind strings to
configured backend instances. Currently only "ransac_centroid" is implemented.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create an association backend by kind name.

    Args:
        kind: Backend identifier. Supported values:

            - ``"ransac_centroid"``: RANSAC centroid ray clustering using the
              v1.0 ``discover_births`` algorithm.

        **kwargs: Forwarded to the backend constructor. For
            ``"ransac_centroid"``, accepted kwargs are: ``calibration_path``,
            ``expected_count``, ``min_cameras``, ``reprojection_threshold``.

    Returns:
        A configured backend instance with an
        ``associate_frame(detections_per_camera)`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
    """
    if kind == "ransac_centroid":
        from aquapose.core.association.backends.ransac_centroid import (
            RansacCentroidBackend,
        )

        return RansacCentroidBackend(**kwargs)

    raise ValueError(
        f"Unknown association backend kind: {kind!r}. "
        f"Supported kinds: ['ransac_centroid']"
    )
