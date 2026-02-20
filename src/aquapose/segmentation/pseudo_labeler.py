"""SAM2 pseudo-label generation from fish detections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from sam2.sam2_image_predictor import (  # pyright: ignore[reportMissingImports]
        SAM2ImagePredictor,
    )

from .crop import compute_crop_region, extract_crop, paste_mask
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

    Each detection is cropped from the full frame (with padding) before being
    fed to SAM2. This gives SAM2 better scale context for small objects and
    reduces background leakage. The resulting crop mask is pasted back into
    the full frame.

    The SAM2 model is lazily loaded on first use to avoid GPU memory
    allocation on import.

    Args:
        model_variant: SAM2 model identifier for Hugging Face hub.
        device: Torch device string. Auto-detects cuda/cpu if None.
        crop_padding: Fractional padding around detection bbox for cropping.
            0.25 means 25% of bbox dimension added on each side.
    """

    def __init__(
        self,
        model_variant: str = "facebook/sam2.1-hiera-large",
        device: str | None = None,
        crop_padding: float = 0.25,
    ) -> None:
        self._model_variant = model_variant
        self._device = device
        self._crop_padding = crop_padding
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

        For each detection, crops the image around the bounding box (with
        padding), runs SAM2 on the crop for better scale context, then
        pastes the mask back into the full frame.

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
        frame_h, frame_w = image.shape[:2]

        masks_out: list[np.ndarray] = []

        with torch.inference_mode():
            for det in detections:
                # Compute padded crop region around the detection bbox
                region = compute_crop_region(
                    det.bbox, frame_h, frame_w, padding=self._crop_padding
                )

                # Extract crop and convert BGR to RGB for SAM2
                crop_bgr = extract_crop(image, region)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

                # Set the crop as SAM2's image (not the full frame)
                predictor.set_image(crop_rgb)

                # Bbox coordinates relative to the crop
                bx, by, bw, bh = det.bbox
                box_in_crop = np.array(
                    [
                        bx - region.x1,
                        by - region.y1,
                        bx + bw - region.x1,
                        by + bh - region.y1,
                    ],
                    dtype=np.float32,
                )

                predict_kwargs: dict = {
                    "box": box_in_crop,
                    "multimask_output": False,
                }

                if use_mask_prompt:
                    # Extract the detection mask crop and convert to logits
                    crop_mask = extract_crop(det.mask, region)
                    mask_logits = _mask_to_logits(crop_mask)
                    predict_kwargs["mask_input"] = mask_logits

                masks, _scores, _logits = predictor.predict(**predict_kwargs)

                # masks shape: (1, crop_H, crop_W) with multimask_output=False
                crop_binary = (masks[0] > 0.5).astype(np.uint8) * 255

                # Paste crop mask back into full frame
                full_mask = paste_mask(crop_binary, region)
                masks_out.append(full_mask)

        return masks_out


def to_coco_dataset(
    annotated_frames: list[AnnotatedFrame],
    output_path: Path,
) -> Path:
    """Convert annotated frames to COCO JSON format with RLE masks.

    Produces a COCO-format annotation file suitable for training with
    torchvision Mask R-CNN or other COCO-compatible detectors.

    Args:
        annotated_frames: List of AnnotatedFrame objects with binary masks.
        output_path: Path to write the COCO JSON file.

    Returns:
        Path to the written COCO JSON file.
    """
    import pycocotools.mask as mask_util

    images = []
    annotations = []
    annotation_id = 1

    for img_id, frame in enumerate(annotated_frames, start=1):
        # Determine image dimensions from first mask or default
        if frame.masks:
            h, w = frame.masks[0].shape[:2]
        else:
            h, w = 0, 0

        images.append(
            {
                "id": img_id,
                "file_name": str(frame.image_path.name),
                "width": int(w),
                "height": int(h),
                "frame_id": frame.frame_id,
                "camera_id": frame.camera_id,
            }
        )

        for mask in frame.masks:
            # Encode mask as COCO RLE (Fortran order required)
            mask_binary = (mask > 0).astype(np.uint8)
            mask_f = np.asfortranarray(mask_binary)
            rle = mask_util.encode(mask_f)

            # Convert bytes to string for JSON serialization
            counts = rle["counts"]
            if isinstance(counts, bytes):
                counts = counts.decode("utf-8")
            rle_json = {
                "size": rle["size"],
                "counts": counts,
            }

            # Compute bounding box from mask
            bbox = mask_util.toBbox(rle).tolist()  # [x, y, w, h]
            area = float(mask_util.area(rle))

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": rle_json,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fish"}],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return output_path
