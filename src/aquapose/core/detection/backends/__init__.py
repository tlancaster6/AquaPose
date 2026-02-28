"""Backend registry for the Detection stage.

Provides a factory function that resolves detector kind strings to
configured detector instances. Supported kinds: ``"yolo"``, ``"yolo_obb"``.
"""

from __future__ import annotations

from typing import Any, Union

from aquapose.core.detection.backends.yolo import YOLOBackend
from aquapose.core.detection.backends.yolo_obb import YOLOOBBBackend

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> YOLOBackend | YOLOOBBBackend:
    """Create a detection backend by kind name.

    Args:
        kind: Backend identifier. Supported values: ``"yolo"``, ``"yolo_obb"``.
        **kwargs: Forwarded to the backend constructor. For ``"yolo"`` and
            ``"yolo_obb"``, ``model_path`` is required.

    Returns:
        A configured detector instance with a
        ``detect(frame: np.ndarray) -> list[Detection]`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
    """
    if kind == "yolo":
        return YOLOBackend(**kwargs)
    if kind == "yolo_obb":
        return YOLOOBBBackend(**kwargs)
    raise ValueError(
        f"Unknown detector kind: {kind!r}. Supported kinds: ['yolo', 'yolo_obb']"
    )
