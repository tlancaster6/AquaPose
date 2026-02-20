"""Mask R-CNN segmentation model wrapper using torchvision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .crop import CropRegion


@dataclass
class SegmentationResult:
    """A single instance segmentation result in crop-space coordinates.

    The mask is in crop-space (relative to the crop, not the full frame).
    Callers who need a full-frame mask should call
    ``paste_mask(result.mask, result.crop_region)``.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in crop-space pixel coordinates.
        mask: Binary mask in crop-space, shape (H_crop, W_crop), dtype uint8
            with values 0 or 255. NOT full-frame.
        confidence: Detection confidence score.
        label: Class label (always 1 for fish).
        crop_region: Metadata for reconstructing the full-frame mask via
            ``paste_mask(result.mask, result.crop_region)``.
    """

    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    confidence: float
    label: int
    crop_region: CropRegion


class MaskRCNNSegmentor:
    """Mask R-CNN instance segmentation model for fish detection.

    Wraps ``torchvision.models.detection.maskrcnn_resnet50_fpn_v2`` with
    proper head replacement for the target number of classes. Supports
    loading trained weights and batch inference.

    The primary inference entry point is :meth:`segment`, which accepts
    pre-cropped images and their associated :class:`~.crop.CropRegion`
    metadata. This fits the detect -> crop -> segment pipeline:

    1. **detect**: MOG2 or YOLO detector finds fish bounding boxes.
    2. **crop**: :func:`~.crop.extract_crop` cuts the region.
    3. **segment**: This method runs Mask R-CNN on the crop and returns
       crop-space masks + metadata for full-frame reconstruction.

    Args:
        num_classes: Number of classes including background (default 2: bg + fish).
        weights_path: Optional path to a saved state_dict. If None, uses
            ImageNet-pretrained backbone with fresh detection heads.
        confidence_threshold: Minimum confidence score to keep a detection.
    """

    def __init__(
        self,
        num_classes: int = 2,
        weights_path: Path | None = None,
        confidence_threshold: float = 0.1,
    ) -> None:
        self._model = self._build_model(num_classes)
        self._confidence_threshold = confidence_threshold

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state_dict)

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        """Build Mask R-CNN with replaced box and mask heads.

        Args:
            num_classes: Number of output classes (including background).

        Returns:
            torchvision Mask R-CNN model ready for fine-tuning.
        """
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        )
        # Replace box predictor head
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features  # pyright: ignore[reportAttributeAccessIssue]
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
        # Replace mask predictor head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # pyright: ignore[reportAttributeAccessIssue]
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 256, num_classes
        )
        return model

    def segment(
        self,
        crops: list[np.ndarray],
        crop_regions: list[CropRegion],
    ) -> list[list[SegmentationResult]]:
        """Run inference on a batch of pre-cropped images.

        This is the primary inference entry point in the detect -> crop ->
        segment pipeline. Accepts crops produced by
        :func:`~.crop.extract_crop` together with their associated
        :class:`~.crop.CropRegion` metadata.

        All crops are processed in a single forward pass through the model
        for GPU throughput. Mask R-CNN's FPN + RoI pooling handles crops of
        different sizes natively.

        Args:
            crops: List of BGR uint8 crop images of shape (H, W, 3). Sizes
                may differ between crops.
            crop_regions: List of :class:`~.crop.CropRegion` objects
                corresponding 1-to-1 with ``crops``.

        Returns:
            Nested list: outer per crop, inner per detection.
            Each detection is a :class:`SegmentationResult` with
            crop-space ``mask`` and ``bbox``, plus the ``crop_region``
            needed to paste back into the full frame.
        """
        import cv2

        if len(crops) != len(crop_regions):
            raise ValueError(
                f"crops and crop_regions must have equal length, "
                f"got {len(crops)} vs {len(crop_regions)}"
            )

        self._model.eval()
        device = next(self._model.parameters()).device

        # Convert crops to float tensors [C, H, W] in [0, 1]
        tensors: list[torch.Tensor] = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensors.append(t.to(device))

        with torch.no_grad():
            outputs = self._model(tensors)

        results: list[list[SegmentationResult]] = []
        for output, region in zip(outputs, crop_regions, strict=True):
            detections: list[SegmentationResult] = []
            scores = output["scores"].cpu().numpy()
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"].cpu().numpy()  # (N, 1, H_crop, W_crop)

            for i, score in enumerate(scores):
                if score < self._confidence_threshold:
                    continue

                # Threshold mask at 0.5 to get binary uint8 (0/255)
                binary_mask = (masks[i, 0] > 0.5).astype(np.uint8) * 255

                x1, y1, x2, y2 = boxes[i].astype(int)
                detections.append(
                    SegmentationResult(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        mask=binary_mask,
                        confidence=float(score),
                        label=int(labels[i]),
                        crop_region=region,
                    )
                )
            results.append(detections)

        return results

    def predict(self, images: list[np.ndarray]) -> list[list[SegmentationResult]]:
        """Run inference on a batch of images (backward-compatible entry point).

        Internally calls :meth:`segment` with trivial :class:`~.crop.CropRegion`
        objects covering the full image. Use :meth:`segment` directly when
        you have pre-cropped images from :func:`~.crop.extract_crop`.

        Args:
            images: List of BGR uint8 images of shape (H, W, 3).

        Returns:
            Nested list: outer per image, inner per detection.
            Each detection is a :class:`SegmentationResult` with a trivial
            ``crop_region`` covering the full image.
        """
        crop_regions = [
            CropRegion(
                x1=0,
                y1=0,
                x2=img.shape[1],
                y2=img.shape[0],
                frame_h=img.shape[0],
                frame_w=img.shape[1],
            )
            for img in images
        ]
        return self.segment(images, crop_regions)

    def get_model(self) -> nn.Module:
        """Return the underlying torchvision model for training access.

        Returns:
            The ``nn.Module`` Mask R-CNN model instance.
        """
        return self._model
