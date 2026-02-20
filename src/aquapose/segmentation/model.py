"""Mask R-CNN segmentation model wrapper using torchvision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


@dataclass
class SegmentationResult:
    """A single instance segmentation result.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        mask_rle: pycocotools RLE dict with 'counts' (bytes) and 'size' [h, w].
        confidence: Detection confidence score.
        label: Class label (always 1 for fish).
    """

    bbox: tuple[int, int, int, int]
    mask_rle: dict
    confidence: float
    label: int


class MaskRCNNSegmentor:
    """Mask R-CNN instance segmentation model for fish detection.

    Wraps ``torchvision.models.detection.maskrcnn_resnet50_fpn_v2`` with
    proper head replacement for the target number of classes. Supports
    loading trained weights and batch inference.

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

    def predict(self, images: list[np.ndarray]) -> list[list[SegmentationResult]]:
        """Run inference on a batch of images.

        Args:
            images: List of BGR uint8 images of shape (H, W, 3).

        Returns:
            Nested list: outer per image, inner per detection.
            Each detection is a :class:`SegmentationResult`.
        """
        import cv2

        self._model.eval()
        device = next(self._model.parameters()).device

        # Convert images to float tensors [C, H, W] in [0, 1]
        tensors = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensors.append(t.to(device))

        with torch.no_grad():
            outputs = self._model(tensors)

        results: list[list[SegmentationResult]] = []
        for output in outputs:
            detections: list[SegmentationResult] = []
            scores = output["scores"].cpu().numpy()
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"].cpu().numpy()  # (N, 1, H, W)

            for i, score in enumerate(scores):
                if score < self._confidence_threshold:
                    continue

                # Threshold mask at 0.5 to get binary
                binary_mask = (masks[i, 0] > 0.5).astype(np.uint8)

                # Encode as pycocotools RLE
                mask_f = np.asfortranarray(binary_mask)
                rle = mask_util.encode(mask_f)

                x1, y1, x2, y2 = boxes[i].astype(int)
                detections.append(
                    SegmentationResult(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        mask_rle=rle,  # pyright: ignore[reportArgumentType]
                        confidence=float(score),
                        label=int(labels[i]),
                    )
                )
            results.append(detections)

        return results

    def get_model(self) -> nn.Module:
        """Return the underlying torchvision model for training access.

        Returns:
            The ``nn.Module`` Mask R-CNN model instance.
        """
        return self._model
