"""COCO-format dataset for Mask R-CNN training on fish crops."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.utils.data import Dataset


def stratified_split(
    dataset: CropDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split dataset indices with per-camera stratification.

    Groups images by ``camera_id`` field in the COCO JSON, then for each
    camera independently samples ``val_fraction`` of its images for
    validation and the rest for training. This ensures every camera is
    proportionally represented in both splits.

    Args:
        dataset: A CropDataset instance to split.
        val_fraction: Fraction of each camera's images to use for validation.
            Must be in (0, 1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(train_indices, val_indices)`` as lists of integer
        indices into ``dataset``.
    """
    rng = random.Random(seed)

    # Group indices by camera_id
    camera_to_indices: dict[str, list[int]] = {}
    for idx, img_info in enumerate(dataset._images):
        cam_id = img_info.get("camera_id", "unknown")
        camera_to_indices.setdefault(cam_id, []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for _cam_id, indices in sorted(camera_to_indices.items()):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    return train_indices, val_indices


class CropDataset(Dataset):  # type: ignore[type-arg]
    """Dataset loading COCO-format annotations at native crop resolution.

    Each sample is an ``(image_tensor, target_dict)`` tuple compatible
    with torchvision Mask R-CNN training. Images are returned at their
    native resolution (no forced resize), since Mask R-CNN handles
    variable-size inputs natively via its FPN + RoI pooling.

    Args:
        coco_json: Path to COCO-format annotation JSON file.
        image_root: Root directory containing the source images.
        augment: Whether to apply data augmentation (flips, rotation, jitter).
    """

    def __init__(
        self,
        coco_json: Path,
        image_root: Path,
        augment: bool = False,
    ) -> None:
        self._image_root = Path(image_root)
        self._augment = augment

        with open(coco_json) as f:
            coco = json.load(f)

        self._images: list[dict] = coco["images"]  # type: ignore[assignment]
        self._categories: list[dict] = coco.get("categories", [])  # type: ignore[assignment]

        # Build annotation index: image_id -> list of annotation dicts
        self._ann_index: dict[int, list[dict]] = {}  # type: ignore[assignment]
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self._ann_index.setdefault(img_id, []).append(ann)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load image and annotations at native resolution, optionally augment.

        Loads the crop at its native resolution. No forced resize — Mask R-CNN
        handles variable input sizes natively via FPN. The collate_fn returns
        a tuple (not a stacked batch) so variable sizes are compatible.

        Args:
            idx: Index into the image list.

        Returns:
            Tuple of (image_tensor, target_dict) where image_tensor is
            float32 [C, H, W] in [0, 1] and target_dict contains
            'boxes', 'labels', 'masks' tensors.
        """
        img_info = self._images[idx]
        img_id = img_info["id"]

        # Load image at native resolution
        img_path = self._image_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback: create blank image using COCO-recorded dimensions
            h = img_info.get("height", 256)
            w = img_info.get("width", 256)
            image = np.zeros((h, w, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        # Get annotations for this image
        anns = self._ann_index.get(img_id, [])

        # Decode masks and bboxes
        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        for ann in anns:
            seg = ann["segmentation"]
            # Handle RLE format (counts may be string or bytes)
            if isinstance(seg, dict) and "counts" in seg:
                rle = {
                    "size": seg["size"],
                    "counts": (
                        seg["counts"].encode("utf-8")
                        if isinstance(seg["counts"], str)
                        else seg["counts"]
                    ),
                }
                mask = mask_util.decode(rle).astype(np.uint8)  # pyright: ignore[reportArgumentType]
                masks.append(mask)
            # bbox from annotation: [x, y, w, h] -> [x1, y1, x2, y2]
            bx, by, bw, bh = ann["bbox"]
            boxes.append([bx, by, bx + bw, by + bh])

        # Apply augmentation (at native resolution, before tensor conversion)
        if self._augment and masks:
            image, masks, boxes = self._apply_augmentation(image, masks, boxes)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones(len(boxes), dtype=torch.int64)
            masks_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            # Negative frame: empty tensors with correct spatial dims
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
            masks_tensor = torch.zeros((0, img_h, img_w), dtype=torch.uint8)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
        }

        return image_tensor, target

    def _apply_augmentation(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        boxes: list[list[float]],
    ) -> tuple[np.ndarray, list[np.ndarray], list[list[float]]]:
        """Apply random augmentations consistently to image, masks, and boxes.

        Args:
            image: RGB image array (H, W, 3).
            masks: List of binary mask arrays (H, W).
            boxes: List of [x1, y1, x2, y2] bounding boxes.

        Returns:
            Augmented (image, masks, boxes) tuple.
        """
        h, w = image.shape[:2]

        # Horizontal flip (50%)
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            masks = [np.fliplr(m).copy() for m in masks]
            boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]

        # Vertical flip (50%)
        if random.random() < 0.5:
            image = np.flipud(image).copy()
            masks = [np.flipud(m).copy() for m in masks]
            boxes = [[b[0], h - b[3], b[2], h - b[1]] for b in boxes]

        # Random rotation (0, 90, 180, 270)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k).copy()
            masks = [np.rot90(m, k).copy() for m in masks]
            for i, b in enumerate(boxes):
                for _ in range(k):
                    x1, y1, x2, y2 = b
                    b = [y1, w - x2, y2, w - x1]
                boxes[i] = b
            # After rotation, h/w may swap for odd k
            if k in (1, 3):
                h, w = w, h

        # Brightness/contrast jitter (image only, not masks)
        brightness = random.uniform(0.7, 1.3)
        contrast = random.uniform(0.7, 1.3)
        image = image.astype(np.float32)
        image = (image - 128) * contrast + 128
        image = image * brightness
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image, masks, boxes
