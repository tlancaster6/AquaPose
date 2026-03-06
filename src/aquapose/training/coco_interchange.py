"""Bidirectional COCO Keypoints <-> YOLO-Pose format conversion."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def yolo_pose_to_coco(pose_dir: Path, n_keypoints: int) -> dict:
    """Convert a YOLO-Pose dataset directory to COCO Keypoints dict.

    Reads all label files from ``pose_dir/labels/train/*.txt`` and
    image dimensions from ``pose_dir/images/train/*.{jpg,png}``.

    Args:
        pose_dir: Directory containing ``images/train/`` and ``labels/train/``.
        n_keypoints: Number of keypoints per annotation.

    Returns:
        COCO-format dict ready for ``json.dump()``.
    """
    img_dir = pose_dir / "images" / "train"
    lbl_dir = pose_dir / "labels" / "train"

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 0
    ann_id = 0

    for label_path in sorted(lbl_dir.glob("*.txt")):
        stem = label_path.stem

        # Find matching image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            logger.warning("No image found for label %s, skipping", label_path)
            continue

        # Read image dimensions (fast, no decode)
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        image_id += 1
        images.append(
            {
                "id": image_id,
                "file_name": f"images/train/{img_path.name}",
                "width": img_w,
                "height": img_h,
            }
        )

        # Parse label lines
        text = label_path.read_text().strip()
        if not text:
            continue

        for line in text.split("\n"):
            vals = line.strip().split()
            if not vals:
                continue

            # Format: cls cx cy w h  x1 y1 v1  x2 y2 v2 ...
            # cls = vals[0], bbox = vals[1:5], keypoints = vals[5:]
            cx = float(vals[1])
            cy = float(vals[2])
            bw = float(vals[3])
            bh = float(vals[4])

            # Convert normalized bbox to absolute [x_min, y_min, w, h]
            abs_bw = bw * img_w
            abs_bh = bh * img_h
            x_min = (cx - bw / 2) * img_w
            y_min = (cy - bh / 2) * img_h

            # Parse and convert keypoints
            kp_flat: list[float] = []
            num_visible = 0
            for ki in range(n_keypoints):
                base = 5 + ki * 3
                kx = float(vals[base])
                ky = float(vals[base + 1])
                kv = float(vals[base + 2])

                if kv == 0 and kx == 0 and ky == 0:
                    # Not labeled
                    kp_flat.extend([0.0, 0.0, 0])
                else:
                    kp_flat.extend([kx * img_w, ky * img_h, int(kv)])
                    if int(kv) > 0:
                        num_visible += 1

            ann_id += 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x_min, y_min, abs_bw, abs_bh],
                    "area": abs_bw * abs_bh,
                    "keypoints": kp_flat,
                    "num_keypoints": num_visible,
                    "iscrowd": 0,
                }
            )

    categories = [
        {
            "id": 1,
            "name": "fish",
            "supercategory": "animal",
            "keypoints": [f"kp_{i}" for i in range(n_keypoints)],
            "skeleton": [],
        }
    ]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def write_coco_keypoints(pose_dir: Path, n_keypoints: int) -> Path:
    """Write ``coco_keypoints.json`` next to ``dataset.yaml`` in *pose_dir*.

    Args:
        pose_dir: YOLO-Pose dataset directory.
        n_keypoints: Number of keypoints per annotation.

    Returns:
        Path to the written JSON file.
    """
    coco = yolo_pose_to_coco(pose_dir, n_keypoints)
    out = pose_dir / "coco_keypoints.json"
    out.write_text(json.dumps(coco, indent=2))
    return out


def coco_to_yolo_pose(
    coco_path: Path,
    output_labels_dir: Path,
    img_dir: Path | None = None,
) -> int:
    """Convert COCO Keypoints JSON to YOLO-Pose label files.

    Args:
        coco_path: Path to a COCO Keypoints JSON file.
        output_labels_dir: Directory to write ``.txt`` label files into.
        img_dir: Optional image directory to read dimensions from.
            If ``None``, uses dimensions from COCO image entries.

    Returns:
        Number of label files written.
    """
    coco = json.loads(coco_path.read_text())

    # Build image_id -> image_info lookup
    id_to_img: dict[int, dict] = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    for image_id, img_info in id_to_img.items():
        anns = anns_by_img.get(image_id, [])
        if not anns:
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        # Derive output filename from image file_name stem
        stem = Path(img_info["file_name"]).stem
        lines: list[str] = []

        for ann in anns:
            # Convert absolute bbox [x_min, y_min, w, h] to normalized [cx, cy, w, h]
            x_min, y_min, bw, bh = ann["bbox"]
            cx = (x_min + bw / 2) / img_w
            cy = (y_min + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h

            # Convert absolute keypoints to normalized
            kps = ann["keypoints"]
            n_kp = len(kps) // 3
            kp_parts: list[str] = []
            for ki in range(n_kp):
                kx = kps[ki * 3]
                ky = kps[ki * 3 + 1]
                kv = kps[ki * 3 + 2]

                if kv == 0 and kx == 0 and ky == 0:
                    kp_parts.extend(["0", "0", "0"])
                else:
                    kp_parts.extend(
                        [
                            str(kx / img_w),
                            str(ky / img_h),
                            str(int(kv)),
                        ]
                    )

            parts = ["0", str(cx), str(cy), str(nw), str(nh), *kp_parts]
            lines.append(" ".join(parts))

        out_path = output_labels_dir / f"{stem}.txt"
        out_path.write_text("\n".join(lines) + "\n")
        n_written += 1

    return n_written
