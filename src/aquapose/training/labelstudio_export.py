"""Export YOLO datasets to Label Studio pre-annotation JSON."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from PIL import Image


def _obb_corners_to_rotated_rect(corners: list[float], img_w: int, img_h: int) -> dict:
    """Convert 4 normalized corner points to Label Studio rotated rectangle.

    Args:
        corners: [x1,y1, x2,y2, x3,y3, x4,y4] normalized 0-1, order TL,TR,BR,BL.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Dict with x, y, width, height (percentages), rotation (degrees).
    """
    pts = [(corners[i] * img_w, corners[i + 1] * img_h) for i in range(0, 8, 2)]
    tl, tr, _br, bl = pts

    w_px = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    h_px = math.hypot(bl[0] - tl[0], bl[1] - tl[1])

    # Rotation angle from TL→TR vector (0° = horizontal rightward)
    angle_deg = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))

    # Label Studio rotates around TL anchor, so (x, y) = TL position
    x_pct = tl[0] / img_w * 100
    y_pct = tl[1] / img_h * 100
    w_pct = w_px / img_w * 100
    h_pct = h_px / img_h * 100

    return {
        "x": x_pct,
        "y": y_pct,
        "width": w_pct,
        "height": h_pct,
        "rotation": angle_deg,
    }


def _parse_obb_label(
    label_path: Path, img_w: int, img_h: int, class_names: dict[int, str]
) -> list[dict]:
    """Parse a YOLO-OBB label file into Label Studio results."""
    results = []
    for line in label_path.read_text().strip().splitlines():
        tokens = line.strip().split()
        cls_id = int(float(tokens[0]))
        corners = [float(t) for t in tokens[1:9]]
        rect = _obb_corners_to_rotated_rect(corners, img_w, img_h)
        rect["rectanglelabels"] = [class_names.get(cls_id, str(cls_id))]
        results.append(
            {
                "id": uuid.uuid4().hex[:8],
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": img_w,
                "original_height": img_h,
                "image_rotation": 0,
                "value": rect,
            }
        )
    return results


def _parse_pose_label(
    label_path: Path,
    img_w: int,
    img_h: int,
    class_names: dict[int, str],
    num_kpts: int,
    kpt_names: list[str] | None = None,
) -> list[dict]:
    """Parse a YOLO-Pose label file into Label Studio results."""
    results = []
    for line in label_path.read_text().strip().splitlines():
        tokens = line.strip().split()
        cls_id = int(float(tokens[0]))
        cx, cy, w, h = (float(t) for t in tokens[1:5])
        x_pct = (cx - w / 2) * 100
        y_pct = (cy - h / 2) * 100
        w_pct = w * 100
        h_pct = h * 100

        results.append(
            {
                "id": uuid.uuid4().hex[:8],
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": img_w,
                "original_height": img_h,
                "image_rotation": 0,
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "rotation": 0,
                    "rectanglelabels": [class_names.get(cls_id, str(cls_id))],
                },
            }
        )

        kpt_tokens = tokens[5:]
        for k in range(num_kpts):
            base = k * 3
            if base + 2 >= len(kpt_tokens):
                break
            kx = float(kpt_tokens[base])
            ky = float(kpt_tokens[base + 1])
            vis = int(float(kpt_tokens[base + 2]))
            if vis == 0:
                continue
            results.append(
                {
                    "id": uuid.uuid4().hex[:8],
                    "type": "keypointlabels",
                    "from_name": "keypoint",
                    "to_name": "image",
                    "original_width": img_w,
                    "original_height": img_h,
                    "image_rotation": 0,
                    "value": {
                        "x": kx * 100,
                        "y": ky * 100,
                        "width": 0.5,
                        "keypointlabels": [
                            kpt_names[k]
                            if kpt_names and k < len(kpt_names)
                            else f"kp{k}"
                        ],
                    },
                }
            )
    return results


def export_labelstudio_json(
    dataset_dir: Path,
    output_json: Path,
    task: str,
    image_url_prefix: str,
    class_names: dict[int, str] | None = None,
    num_kpts: int = 6,
    kpt_names: list[str] | None = None,
) -> int:
    """Convert a YOLO dataset directory to Label Studio pre-annotation JSON.

    Args:
        dataset_dir: Directory with images/{train,val}/ and labels/{train,val}/.
        output_json: Output JSON file path.
        task: ``"obb"`` or ``"pose"``.
        image_url_prefix: Full URL prefix including document-root-relative path,
            e.g. ``"/data/local-files/?d=round1_selected/obb/images/"``.
        class_names: Class index to name mapping. Defaults to ``{0: "fish"}``.
        num_kpts: Number of keypoints for pose task.
        kpt_names: Keypoint label names. If None, reads ``kpt_names`` from
            ``dataset.yaml`` in *dataset_dir*, falling back to
            ``["kp0", "kp1", ...]``.

    Returns:
        Number of tasks written.
    """
    if class_names is None:
        class_names = {0: "fish"}

    # Auto-discover keypoint names from dataset.yaml
    if kpt_names is None and task == "pose":
        ds_yaml_path = dataset_dir / "dataset.yaml"
        if ds_yaml_path.exists():
            import yaml

            ds_cfg = yaml.safe_load(ds_yaml_path.read_text()) or {}
            kpt_names = ds_cfg.get("kpt_names")

    tasks: list[dict] = []
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in (
                ".jpg",
                ".jpeg",
                ".png",
            ):
                continue

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            with Image.open(img_path) as img:
                img_w, img_h = img.size

            if task == "obb":
                results = _parse_obb_label(lbl_path, img_w, img_h, class_names)
            elif task == "pose":
                results = _parse_pose_label(
                    lbl_path, img_w, img_h, class_names, num_kpts, kpt_names
                )
            else:
                raise ValueError(f"Unsupported task: {task}")

            tasks.append(
                {
                    "data": {
                        "image": f"{image_url_prefix}{split}/{img_path.name}",
                    },
                    "predictions": [
                        {
                            "model_version": f"pseudo-label-{task}",
                            "result": results,
                        }
                    ],
                }
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(tasks, indent=2))
    return len(tasks)
