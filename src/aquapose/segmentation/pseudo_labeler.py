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


def _select_largest_mask(masks: np.ndarray) -> np.ndarray:
    """Select the mask with the most nonzero pixels from a stack of masks.

    Used when SAM2 returns multiple candidate masks (multimask_output=True)
    to keep the largest area segmentation.

    Args:
        masks: Array of shape (N, H, W), bool or float32. Values > 0.5 are
            treated as foreground.

    Returns:
        Single mask of shape (H, W), uint8 with values 0 or 255.
    """
    # Count foreground pixels per mask
    counts = [(masks[i] > 0.5).sum() for i in range(len(masks))]
    best_idx = int(np.argmax(counts))
    return (masks[best_idx] > 0.5).astype(np.uint8) * 255


def filter_mask(
    mask: np.ndarray,
    detection: Detection,
    min_conf: float = 0.3,
    min_fill: float = 0.15,
    max_fill: float = 0.85,
    min_area: int = 150,
) -> np.ndarray | None:
    """Apply quality filters to a SAM2-generated mask.

    Filters on detection confidence, mask area, and fill ratio relative to
    the detection bounding box. Also reduces the mask to its largest
    connected component to remove stray pixels.

    Args:
        mask: Binary mask (uint8, 0/255) in full-frame coordinates.
        detection: Detection object that produced this mask.
        min_conf: Minimum YOLO detection confidence threshold.
        min_fill: Minimum mask fill ratio of bbox area.
        max_fill: Maximum mask fill ratio of bbox area.
        min_area: Minimum mask pixel area.

    Returns:
        Cleaned mask (largest connected component only) or None if filtered.
    """
    # Step 1: Confidence gate
    if detection.confidence < min_conf:
        return None

    # Step 2: Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels > 1:
        # Label 0 is background; find largest among 1..N
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas)) + 1
        cleaned = np.zeros_like(mask)
        cleaned[labels == best_label] = 255
        mask = cleaned

    # Step 3: Area gate
    mask_area = int(cv2.countNonZero(mask))
    if mask_area < min_area:
        return None

    # Step 4: Fill ratio gate relative to detection bbox
    _, _, bw, bh = detection.bbox
    bbox_area = bw * bh
    if bbox_area == 0:
        return None

    fill = mask_area / bbox_area
    if fill < min_fill or fill > max_fill:
        return None

    return mask


class SAMPseudoLabeler:
    """Generate high-quality pseudo-label masks from fish detections using SAM2.

    Wraps SAM2ImagePredictor to refine rough detection masks (from MOG2 or YOLO)
    into precise segmentation masks suitable for training data generation.

    Each detection is cropped from the full frame (with padding) before being
    fed to SAM2. This gives SAM2 better scale context for small objects and
    reduces background leakage. The resulting crop mask is pasted back into
    the full frame.

    SAM2 always uses box-only prompting (no mask prompt). This produces
    dramatically better masks than mask-prompted mode per empirical testing.

    The SAM2 model is lazily loaded on first use to avoid GPU memory
    allocation on import.

    Args:
        model_variant: SAM2 model identifier for Hugging Face hub.
        device: Torch device string. Auto-detects cuda/cpu if None.
        crop_padding: Fractional padding around detection bbox for cropping.
            0.25 means 25% of bbox dimension added on each side.
        draw_pseudolabels: When True, saves annotated debug images (with
            mask contours overlaid) to a ``debug/`` subdirectory next to
            each output image. For developer use only.
    """

    def __init__(
        self,
        model_variant: str = "facebook/sam2.1-hiera-large",
        device: str | None = None,
        crop_padding: float = 0.25,
        draw_pseudolabels: bool = False,
    ) -> None:
        self._model_variant = model_variant
        self._device = device
        self._crop_padding = crop_padding
        self._draw_pseudolabels = draw_pseudolabels
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
    ) -> list[np.ndarray]:
        """Generate refined masks from fish detections using SAM2 box-only prompting.

        For each detection, crops the image around the bounding box (with
        padding), runs SAM2 on the crop with box-only prompting for better
        scale context, then pastes the mask back into the full frame.

        Args:
            image: Full-frame BGR image as uint8 array of shape (H, W, 3).
            detections: List of Detection objects from any detector (MOG2Detector
                or YOLODetector via make_detector).

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

                # Box-only prompting — no mask input
                masks, _scores, _logits = predictor.predict(
                    box=box_in_crop,
                    multimask_output=False,
                )

                # masks shape: (1, crop_H, crop_W) with multimask_output=False
                # Use _select_largest_mask in case SAM2 ever returns multimask
                if len(masks) > 1:
                    crop_binary = _select_largest_mask(masks)
                else:
                    crop_binary = (masks[0] > 0.5).astype(np.uint8) * 255

                # Paste crop mask back into full frame
                full_mask = paste_mask(crop_binary, region)

                # Optionally save debug visualization
                if self._draw_pseudolabels:
                    self._save_debug(crop_bgr, crop_binary, det)

                masks_out.append(full_mask)

        return masks_out

    def _save_debug(
        self,
        crop_bgr: np.ndarray,
        crop_mask: np.ndarray,
        det: Detection,
    ) -> None:
        """Save annotated debug image with mask contours overlaid.

        Args:
            crop_bgr: Cropped BGR image.
            crop_mask: Binary mask (uint8, 0/255) at crop resolution.
            det: Detection that produced this mask.
        """
        debug_img = crop_bgr.copy()
        contours, _ = cv2.findContours(
            crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)

        # Save to debug/ subdirectory next to the image path
        debug_dir = Path("debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_name = f"debug_conf{det.confidence:.2f}.jpg"
        cv2.imwrite(str(debug_dir / debug_name), debug_img)


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
