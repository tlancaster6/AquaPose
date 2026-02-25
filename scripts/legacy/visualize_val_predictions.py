"""Visualize U-Net predictions vs pseudo-label ground truth on val set."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

from aquapose.segmentation.model import UNET_INPUT_SIZE, UNetSegmentor

DATA_ROOT = Path(r"C:\Users\tucke\Desktop\Aqua\AquaPose\unet\training_data")
MODEL_PATH = Path(r"C:\Users\tucke\Desktop\Aqua\AquaPose\unet\best_model.pth")
OUTPUT_PATH = Path(r"/output/val_sidebyside.png")


def _decode_gt_mask(ann: dict, img_h: int, img_w: int) -> np.ndarray:
    """Decode a COCO RLE annotation to a binary mask."""
    seg = ann["segmentation"]
    rle = {
        "size": seg["size"],
        "counts": (
            seg["counts"].encode("utf-8")
            if isinstance(seg["counts"], str)
            else seg["counts"]
        ),
    }
    return mask_util.decode(rle).astype(np.uint8)


def _overlay_mask(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.4
) -> np.ndarray:
    """Overlay a binary mask on an image with transparency."""
    out = image.copy()
    overlay = np.full_like(image, color, dtype=np.uint8)
    region = mask > 0
    out[region] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[region]
    # Draw contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def main() -> None:
    val_json = DATA_ROOT / "val.json"
    image_root = DATA_ROOT / "images"

    with open(val_json) as f:
        coco = json.load(f)

    # Build annotation index
    ann_index: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        ann_index.setdefault(ann["image_id"], []).append(ann)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmentor = UNetSegmentor(weights_path=MODEL_PATH, confidence_threshold=0.0)
    model = segmentor.get_model()
    model.to(device)
    model.eval()

    sz = UNET_INPUT_SIZE
    panels: list[np.ndarray] = []

    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_path = image_root / img_info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        # Get GT mask (merged)
        anns = ann_index.get(img_id, [])
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            m = _decode_gt_mask(ann, h, w)
            gt_mask = np.maximum(gt_mask, m)

        # Run U-Net prediction
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
        inp = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        inp = inp.to(device)

        with torch.no_grad():
            prob = model(inp)[0, 0].cpu().numpy()

        pred_mask_128 = (prob > 0.5).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask_128, (w, h), interpolation=cv2.INTER_NEAREST)

        # Compute IoU
        intersection = (gt_mask & pred_mask).sum()
        union = (gt_mask | pred_mask).sum()
        iou = intersection / max(union, 1)

        # Build side-by-side: original | GT overlay | Pred overlay
        gt_overlay = _overlay_mask(image, gt_mask * 255, (0, 255, 0))  # green
        pred_overlay = _overlay_mask(image, pred_mask * 255, (0, 100, 255))  # orange

        # Add labels
        cam_id = img_info.get("camera_id", "?")
        cv2.putText(
            gt_overlay, "GT", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        cv2.putText(
            pred_overlay,
            f"Pred IoU={iou:.2f}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 100, 255),
            1,
        )
        cv2.putText(
            image,
            f"{cam_id}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        panel = np.hstack([image, gt_overlay, pred_overlay])
        panels.append((iou, panel, cam_id))  # type: ignore[arg-type]

    # Sort by IoU ascending (worst first) for easy inspection
    panels.sort(key=lambda x: x[0])  # type: ignore[index,return-value]

    # Resize all panels to same width for stacking
    target_w = 600  # 200px per column
    resized_panels = []
    for _iou, panel, _cam in panels:  # type: ignore[misc]
        scale = target_w / panel.shape[1]
        new_h = int(panel.shape[0] * scale)
        resized_panels.append(cv2.resize(panel, (target_w, new_h)))

    # Arrange in a grid (max ~6 rows, then wrap to new column set)
    n = len(resized_panels)
    cols = math.ceil(n / 8)
    rows_per_col = math.ceil(n / cols)

    # Pad panels to uniform height per column group
    col_groups: list[np.ndarray] = []
    for col_idx in range(cols):
        start = col_idx * rows_per_col
        end = min(start + rows_per_col, n)
        group = resized_panels[start:end]
        col_groups.append(np.vstack(group))

    # Pad columns to same height
    max_h = max(c.shape[0] for c in col_groups)
    padded = []
    for c in col_groups:
        if c.shape[0] < max_h:
            pad = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
            c = np.vstack([c, pad])
        padded.append(c)

    mosaic = np.hstack(padded)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), mosaic)
    print(f"Saved {n} side-by-side panels to {OUTPUT_PATH}")
    print(f"Image size: {mosaic.shape[1]}x{mosaic.shape[0]}")

    # Print IoU summary
    ious = [x[0] for x in panels]  # type: ignore[index]
    print(
        f"IoU: min={min(ious):.3f} median={sorted(ious)[len(ious) // 2]:.3f} mean={np.mean(ious):.3f} max={max(ious):.3f}"
    )


if __name__ == "__main__":
    main()
