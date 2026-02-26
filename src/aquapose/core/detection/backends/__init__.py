"""Backend registry for the Detection stage.

Provides a factory function that resolves detector kind strings to
configured detector instances. Currently only "yolo" is supported;
MOG2 is deferred to a future plan.
"""

from __future__ import annotations

from typing import Any

from aquapose.core.detection.backends.yolo import YOLOBackend
from aquapose.segmentation.detector import Detection

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> YOLOBackend:
    """Create a detection backend by kind name.

    Args:
        kind: Backend identifier. Currently only ``"yolo"`` is supported.
        **kwargs: Forwarded to the backend constructor. For ``"yolo"``,
            ``model_path`` is required.

    Returns:
        A configured detector instance with a
        ``detect(frame: np.ndarray) -> list[Detection]`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
    """
    if kind == "yolo":
        return YOLOBackend(**kwargs)
    raise ValueError(f"Unknown detector kind: {kind!r}. Supported kinds: ['yolo']")
