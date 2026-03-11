"""Backend registry for the Pose stage.

Provides a factory function that resolves pose backend kind strings to
configured backend instances.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create a pose backend by kind name.

    Args:
        kind: Backend identifier. Supported values:

            - ``"pose_estimation"``: YOLO-pose keypoint inference that
              returns raw anatomical keypoints on Detection objects.

        **kwargs: Forwarded to the backend constructor.

    Returns:
        A configured backend instance.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
    """
    if kind == "pose_estimation":
        from aquapose.core.pose.backends.pose_estimation import PoseEstimationBackend

        return PoseEstimationBackend(**kwargs)

    raise ValueError(
        f"Unknown pose backend kind: {kind!r}. Supported kinds: ['pose_estimation']"
    )
