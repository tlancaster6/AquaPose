"""Label Studio export/import utilities for annotation workflows."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from label_studio_converter.brush import decode_rle, mask2rle

from .pseudo_labeler import AnnotatedFrame, FrameAnnotation


def export_to_label_studio(
    frames: list[FrameAnnotation],
    output_dir: Path,
    project_name: str = "aquapose",
) -> Path:
    """Export pseudo-label masks to Label Studio task JSON with brush RLE format.

    Creates a tasks JSON file importable by Label Studio with pre-annotated
    brush masks. Images are copied (or symlinked) to the output directory.
    Negative frames (no masks) are included with empty predictions.

    Args:
        frames: List of FrameAnnotation objects with masks to export.
        output_dir: Directory to write images and tasks JSON.
        project_name: Name prefix for the tasks JSON file.

    Returns:
        Path to the generated tasks JSON file.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    tasks = []

    for frame in frames:
        # Copy image to output directory
        dest_image = images_dir / frame.image_path.name
        if not dest_image.exists():
            shutil.copy2(frame.image_path, dest_image)

        relative_path = f"/data/local-files/?d=images/{frame.image_path.name}"

        predictions_results = []
        for mask in frame.masks:
            h, w = mask.shape[:2]
            # Label Studio expects 0/255 uint8 masks for mask2rle
            mask_uint8 = mask.astype(np.uint8)
            if mask_uint8.max() == 1:
                mask_uint8 = mask_uint8 * 255

            rle = mask2rle(mask_uint8)

            brush_result = {
                "type": "brushlabels",
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": ["fish"],
                },
                "original_width": int(w),
                "original_height": int(h),
                "from_name": "label",
                "to_name": "image",
            }
            predictions_results.append(brush_result)

        task: dict[str, Any] = {
            "data": {
                "image": relative_path,
                "frame_id": frame.frame_id,
                "camera_id": frame.camera_id,
            },
        }

        if predictions_results:
            task["predictions"] = [{"result": predictions_results}]
        else:
            # Negative frame -- empty predictions
            task["predictions"] = []

        tasks.append(task)

    tasks_path = output_dir / f"{project_name}_tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(tasks, f, indent=2)

    return tasks_path


def import_from_label_studio(annotations_path: Path) -> list[AnnotatedFrame]:
    """Import corrected annotations from Label Studio export JSON.

    Reads a Label Studio JSON export and decodes brush RLE masks back
    to binary numpy arrays.

    Args:
        annotations_path: Path to the Label Studio JSON export file.

    Returns:
        List of AnnotatedFrame objects with decoded binary masks.
    """
    with open(annotations_path) as f:
        tasks = json.load(f)

    annotated_frames: list[AnnotatedFrame] = []

    for task in tasks:
        data = task["data"]
        image_path = Path(data.get("image", ""))
        frame_id = data.get("frame_id", "")
        camera_id = data.get("camera_id", "")

        masks: list[np.ndarray] = []

        # Get annotations (Label Studio uses "annotations" key in exports)
        annotations = task.get("annotations", [])
        if annotations:
            # Use first annotation set (primary annotator)
            results = annotations[0].get("result", [])
            for result in results:
                if result.get("type") != "brushlabels":
                    continue

                value = result.get("value", {})
                rle = value.get("rle", [])
                orig_w = result.get("original_width", 0)
                orig_h = result.get("original_height", 0)

                if rle and orig_w > 0 and orig_h > 0:
                    # decode_rle returns a flat array; reshape to image dims
                    flat_mask = decode_rle(rle)
                    # decode_rle may return values in [0, 255] range
                    binary_mask = np.array(flat_mask, dtype=np.uint8).reshape(
                        orig_h, orig_w, 4
                    )
                    # Take first channel (all channels should be same for binary)
                    binary_mask = binary_mask[:, :, 0]
                    # Ensure binary 0/255
                    binary_mask = np.where(binary_mask > 0, 255, 0).astype(np.uint8)
                    masks.append(binary_mask)

        annotated_frames.append(
            AnnotatedFrame(
                frame_id=frame_id,
                image_path=image_path,
                masks=masks,
                camera_id=camera_id,
            )
        )

    return annotated_frames


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
