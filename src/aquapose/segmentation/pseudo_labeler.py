"""SAM2 pseudo-label generation from fish detections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from sam2.sam2_image_predictor import (  # pyright: ignore[reportMissingImports]
        SAM2ImagePredictor,
    )

from .detector import Detection


@dataclass
class FrameAnnotation:
    """A frame with pseudo-label masks for Label Studio export.

    Attributes:
        frame_id: Unique identifier for this frame.
        image_path: Path to the source image file.
        masks: List of binary masks (uint8, 0/255), one per detection.
        camera_id: Camera identifier for multi-view tracking.
    """

    frame_id: str
    image_path: Path
    masks: list[np.ndarray]
    camera_id: str


@dataclass
class AnnotatedFrame:
    """A frame with human-corrected annotation masks.

    Attributes:
        frame_id: Unique identifier for this frame.
        image_path: Path to the source image file.
        masks: List of binary masks (uint8, 0/255), one per annotation.
        camera_id: Camera identifier for multi-view tracking.
    """

    frame_id: str
    image_path: Path
    masks: list[np.ndarray]
    camera_id: str


def _mask_to_logits(mask: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Convert a binary mask to logits and resize for SAM2 input.

    SAM2's mask_input expects logits (not binary). Values > 0 mean foreground.
    Resizes to target_size x target_size as SAM2 expects 256x256 input.

    Args:
        mask: Binary mask (uint8, 0/255) of any size.
        target_size: Target spatial dimension for SAM2 (default 256).

    Returns:
        Logits array of shape (1, target_size, target_size), float32.
    """
    # Convert 0/255 to logits: foreground=+4.0, background=-4.0
    logits = np.where(mask > 0, 4.0, -4.0).astype(np.float32)
    # Resize to SAM2 expected input size
    resized = cv2.resize(
        logits, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )
    return resized[np.newaxis, :, :]  # (1, H, W)


class SAMPseudoLabeler:
    """Generate high-quality pseudo-label masks from fish detections using SAM2.

    Wraps SAM2ImagePredictor to refine rough detection masks (from MOG2 or YOLO)
    into precise segmentation masks suitable for human review in Label Studio.

    The SAM2 model is lazily loaded on first use to avoid GPU memory
    allocation on import.

    Args:
        model_variant: SAM2 model identifier for Hugging Face hub.
        device: Torch device string. Auto-detects cuda/cpu if None.
    """

    def __init__(
        self,
        model_variant: str = "facebook/sam2.1-hiera-large",
        device: str | None = None,
    ) -> None:
        self._model_variant = model_variant
        self._device = device
        self._predictor: SAM2ImagePredictor | None = None

    def _load_predictor(self) -> SAM2ImagePredictor:
        """Lazily load SAM2ImagePredictor on first use."""
        if self._predictor is not None:
            return self._predictor

        import torch
        from sam2.sam2_image_predictor import (  # pyright: ignore[reportMissingImports]
            SAM2ImagePredictor,
        )

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._predictor = SAM2ImagePredictor.from_pretrained(
            self._model_variant, device=device
        )
        return self._predictor

    def predict(
        self,
        image: np.ndarray,
        detections: list[Detection],
        use_mask_prompt: bool = True,
    ) -> list[np.ndarray]:
        """Generate refined masks from fish detections using SAM2.

        Sets the full image on the SAM2 predictor (not cropped -- SAM2
        uses full image context for better mask boundaries). For each
        detection, prompts SAM2 with the bounding box and optionally the
        rough foreground mask.

        Args:
            image: Full-frame BGR image as uint8 array of shape (H, W, 3).
            detections: List of Detection objects from any detector (MOG2Detector
                or YOLODetector via make_detector).
            use_mask_prompt: Whether to pass the detection mask as a logit
                prompt alongside the box. Falls back to box-only if False.

        Returns:
            List of binary masks (uint8, 0/255, full frame size),
            one per detection. Empty list if detections is empty.
        """
        if not detections:
            return []

        import torch

        predictor = self._load_predictor()

        # Convert BGR to RGB for SAM2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            predictor.set_image(image_rgb)

            masks_out: list[np.ndarray] = []
            for det in detections:
                # Convert bbox (x, y, w, h) to SAM2 format [x1, y1, x2, y2]
                x, y, w, h = det.bbox
                box = np.array([x, y, x + w, y + h], dtype=np.float32)

                predict_kwargs: dict = {
                    "box": box,
                    "multimask_output": False,
                }

                if use_mask_prompt:
                    mask_logits = _mask_to_logits(det.mask)
                    predict_kwargs["mask_input"] = mask_logits

                masks, _scores, _logits = predictor.predict(**predict_kwargs)

                # masks shape: (1, H, W) with multimask_output=False
                binary_mask = (masks[0] > 0.5).astype(np.uint8) * 255
                masks_out.append(binary_mask)

        return masks_out
