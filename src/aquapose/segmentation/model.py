"""Segmentation models for fish crop binary segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .crop import CropRegion

# Fixed input size for UNet training and inference
UNET_INPUT_SIZE = 128


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


class _UNet(nn.Module):
    """Lightweight U-Net with MobileNetV3-Small encoder for binary segmentation.

    Uses 4 skip-connection levels from the MobileNetV3-Small feature hierarchy.
    Decoder uses bilinear upsampling + conv (no transposed convolutions).
    Output is a single-channel sigmoid probability map.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small(
            weights=(
                torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                if pretrained
                else None
            ),
        )
        features = backbone.features

        # MobileNetV3-Small feature layers and their output channels:
        #   features[0]    -> 16ch,  stride 2   (level 0)
        #   features[1]    -> 16ch,  stride 4   (level 1)
        #   features[2:4]  -> 24ch,  stride 8   (level 2)
        #   features[4:9]  -> 48ch,  stride 16  (level 3)
        #   features[9:12] -> 96ch,  stride 32  (level 4 / bottleneck)
        # Skip features[12] (576ch 1x1 expansion — not needed for U-Net).
        self.enc0 = features[0:1]  # -> 16ch
        self.enc1 = features[1:2]  # -> 16ch
        self.enc2 = features[2:4]  # -> 24ch
        self.enc3 = features[4:9]  # -> 48ch
        self.enc4 = features[9:12]  # -> 96ch (bottleneck)

        # Decoder: upsample + concat skip + conv block
        self.up4 = _DecoderBlock(96, 48, 48)  # concat enc3(48) -> 96 in
        self.up3 = _DecoderBlock(48, 24, 24)  # concat enc2(24) -> 48 in
        self.up2 = _DecoderBlock(24, 16, 16)  # concat enc1(16) -> 32 in
        self.up1 = _DecoderBlock(16, 16, 16)  # concat enc0(16) -> 32 in

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W) in [0, 1].

        Returns:
            Sigmoid probability map of shape (B, 1, H, W).
        """
        e0 = self.enc0(x)  # (B, 16, H/2, W/2)
        e1 = self.enc1(e0)  # (B, 16, H/4, W/4)
        e2 = self.enc2(e1)  # (B, 24, H/8, W/8)
        e3 = self.enc3(e2)  # (B, 48, H/16, W/16)
        e4 = self.enc4(e3)  # (B, 96, H/32, W/32)

        d3 = self.up4(e4, e3)  # (B, 48, H/16, W/16)
        d2 = self.up3(d3, e2)  # (B, 24, H/8, W/8)
        d1 = self.up2(d2, e1)  # (B, 16, H/4, W/4)
        d0 = self.up1(d1, e0)  # (B, 16, H/2, W/2)

        return torch.sigmoid(self.final(d0))  # (B, 1, H, W)


class _DecoderBlock(nn.Module):
    """Single decoder level: upsample, concat skip, conv-bn-relu x2."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample x, concat with skip, apply conv block."""
        x = self.up(x)
        # Handle odd spatial dimensions from encoder
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetSegmentor:
    """Lightweight U-Net binary segmentor for fish crops.

    Replaces Mask R-CNN in the detect -> crop -> segment pipeline.
    Since crops already contain a single centered fish (from YOLO detection),
    a simple binary segmentation model is sufficient and much lighter.

    All crops are resized to 128x128 for inference, then masks are resized
    back to original crop dimensions.

    Args:
        weights_path: Optional path to a saved state_dict. If None, uses
            ImageNet-pretrained MobileNetV3 encoder with fresh decoder.
        confidence_threshold: Minimum mean foreground probability to produce
            a result. If the mean probability of foreground pixels is below
            this, an empty list is returned for that crop.
    """

    def __init__(
        self,
        weights_path: Path | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self._model = _UNet(pretrained=(weights_path is None))
        self._confidence_threshold = confidence_threshold

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state_dict)

    def segment(
        self,
        crops: list[np.ndarray],
        crop_regions: list[CropRegion],
    ) -> list[list[SegmentationResult]]:
        """Run inference on a batch of pre-cropped images.

        All crops are resized to 128x128, passed through the U-Net in a
        single batched forward pass, then masks are resized back to the
        original crop dimensions.

        Args:
            crops: List of BGR uint8 crop images of shape (H, W, 3).
            crop_regions: List of CropRegion objects corresponding 1-to-1
                with ``crops``.

        Returns:
            Nested list: outer per crop, inner 0 or 1 SegmentationResult.
            U-Net produces one binary mask per crop (not multiple detections).
        """
        if len(crops) != len(crop_regions):
            raise ValueError(
                f"crops and crop_regions must have equal length, "
                f"got {len(crops)} vs {len(crop_regions)}"
            )

        self._model.eval()
        device = next(self._model.parameters()).device
        sz = UNET_INPUT_SIZE

        # Resize and convert crops to batched tensor
        tensors: list[torch.Tensor] = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            probs = self._model(batch)  # (B, 1, 128, 128)

        results: list[list[SegmentationResult]] = []
        for i, (crop, region) in enumerate(zip(crops, crop_regions, strict=True)):
            prob_map = probs[i, 0].cpu().numpy()  # (128, 128)
            binary_128 = (prob_map > 0.5).astype(np.uint8)

            # Check confidence: mean probability of foreground pixels
            fg_pixels = prob_map[binary_128 == 1]
            if (
                fg_pixels.size == 0
                or float(fg_pixels.mean()) < self._confidence_threshold
            ):
                results.append([])
                continue

            confidence = float(fg_pixels.mean())

            # Resize mask back to original crop dimensions
            crop_h, crop_w = crop.shape[:2]
            if (crop_h, crop_w) == (sz, sz):
                binary_crop = binary_128 * 255
            else:
                binary_crop = (
                    cv2.resize(
                        binary_128, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
                    )
                    * 255
                )

            # Compute tight bbox from mask
            ys, xs = np.where(binary_crop > 0)
            if len(ys) == 0:
                results.append([])
                continue

            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

            results.append(
                [
                    SegmentationResult(
                        bbox=(x1, y1, x2, y2),
                        mask=binary_crop.astype(np.uint8),
                        confidence=confidence,
                        label=1,
                        crop_region=region,
                    )
                ]
            )

        return results

    def predict(self, images: list[np.ndarray]) -> list[list[SegmentationResult]]:
        """Run inference on a batch of images (backward-compatible entry point).

        Internally calls :meth:`segment` with trivial CropRegion objects
        covering the full image.

        Args:
            images: List of BGR uint8 images of shape (H, W, 3).

        Returns:
            Nested list: outer per image, inner 0 or 1 SegmentationResult.
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
        """Return the underlying nn.Module for training access.

        Returns:
            The ``_UNet`` model instance.
        """
        return self._model


class MaskRCNNSegmentor:
    """Mask R-CNN instance segmentation model for fish detection.

    Wraps ``torchvision.models.detection.maskrcnn_resnet50_fpn_v2`` with
    proper head replacement for the target number of classes. Supports
    loading trained weights and batch inference.

    .. deprecated::
        Use :class:`UNetSegmentor` instead. MaskRCNNSegmentor is retained
        for backward compatibility but is no longer the default.

    Args:
        num_classes: Number of classes including background (default 2: bg + fish).
        weights_path: Optional path to a saved state_dict.
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

        Args:
            crops: List of BGR uint8 crop images of shape (H, W, 3).
            crop_regions: List of CropRegion objects corresponding 1-to-1
                with ``crops``.

        Returns:
            Nested list: outer per crop, inner per detection.
        """
        if len(crops) != len(crop_regions):
            raise ValueError(
                f"crops and crop_regions must have equal length, "
                f"got {len(crops)} vs {len(crop_regions)}"
            )

        self._model.eval()
        device = next(self._model.parameters()).device

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
            masks = output["masks"].cpu().numpy()

            for i, score in enumerate(scores):
                if score < self._confidence_threshold:
                    continue

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

        Args:
            images: List of BGR uint8 images of shape (H, W, 3).

        Returns:
            Nested list: outer per image, inner per detection.
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
