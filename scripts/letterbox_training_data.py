"""One-time script: letterbox training crops to 128x128 and adjust keypoints.

Reads the original rectangular OBB crops, scales each to fit within 128x128
while preserving aspect ratio (letterbox), centers on a black canvas, and
shifts keypoint coordinates to match. Writes output to a sibling directory.

Usage:
    python scripts/letterbox_training_data.py \
        --src C:/Users/tucke/aquapose/projects/YH/models/pose/training_set \
        --dst C:/Users/tucke/aquapose/projects/YH/models/pose/training_set_letterboxed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def letterbox_image(
    img: np.ndarray, target_size: int = 128
) -> tuple[np.ndarray, float, float, float]:
    """Scale image to fit in target_size x target_size, pad with black.

    Returns:
        (letterboxed_image, scale, pad_x, pad_y)
    """
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0
    x0 = round(pad_x)
    y0 = round(pad_y)
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    return canvas, scale, pad_x, pad_y


def transform_keypoints(
    keypoints: list[float],
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    target_size: int = 128,
) -> list[float]:
    """Shift keypoint coordinates from original crop space to letterboxed space."""
    out = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if v > 0:
            x_new = x * scale + pad_x
            y_new = y * scale + pad_y
            # Clip to canvas and mark OOB as invisible
            if 0 <= x_new < target_size and 0 <= y_new < target_size:
                out.extend([x_new, y_new, v])
            else:
                out.extend([0.0, 0.0, 0.0])
        else:
            out.extend([0.0, 0.0, 0.0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Letterbox training crops to 128x128")
    parser.add_argument("--src", required=True, help="Source training_set directory")
    parser.add_argument("--dst", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=128, help="Target square size")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    target_size = args.size

    # Load annotations
    ann_path = src / "annotations.json"
    with open(ann_path) as f:
        coco = json.load(f)

    # Create output dirs
    img_dst = dst / "images"
    img_dst.mkdir(parents=True, exist_ok=True)

    img_src = src / "images" if (src / "images").is_dir() else src

    # Process each image
    new_images = []
    new_annotations = []

    ann_by_img = {}
    for ann in coco.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_info in coco["images"]:
        img_id = img_info["id"]
        fname = img_info["file_name"]
        orig_w = img_info["width"]
        orig_h = img_info["height"]

        # Load image
        img = cv2.imread(str(img_src / fname))
        if img is None:
            print(f"WARNING: could not read {fname}, skipping")
            continue

        # Letterbox
        canvas, scale, pad_x, pad_y = letterbox_image(img, target_size)
        cv2.imwrite(str(img_dst / fname), canvas)

        # Update image info
        new_images.append({**img_info, "width": target_size, "height": target_size})

        # Update annotations
        for ann in ann_by_img.get(img_id, []):
            new_kps = transform_keypoints(
                ann["keypoints"], orig_w, orig_h, scale, pad_x, pad_y, target_size
            )
            new_annotations.append({**ann, "keypoints": new_kps})

    # Write output annotations
    new_coco = {**coco, "images": new_images, "annotations": new_annotations}
    with open(dst / "annotations.json", "w") as f:
        json.dump(new_coco, f, indent=2)

    print(f"Done: {len(new_images)} images letterboxed to {target_size}x{target_size}")
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
