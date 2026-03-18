"""Generate wall-fish inpaint augmentations for OBB training data."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_obb_label(label_path: Path) -> list[tuple[str, np.ndarray]]:
    """Parse a YOLO-OBB label file into (raw_line, corners_norm) pairs.

    Args:
        label_path: Path to the label .txt file.

    Returns:
        List of (original_line, corners) where corners is shape (4, 2)
        with normalized coordinates.
    """
    annotations = []
    for line in label_path.read_text().strip().splitlines():
        tokens = line.strip().split()
        if len(tokens) < 9:
            continue
        coords = [float(t) for t in tokens[1:9]]
        corners = np.array(coords, dtype=np.float64).reshape(4, 2)
        annotations.append((line.strip(), corners))
    return annotations


def is_wall_fish(
    corners_px: np.ndarray,
    gray: np.ndarray,
    threshold: int,
    patch_size: int,
) -> bool:
    """Determine if an annotation is a wall-fish based on corner brightness.

    Args:
        corners_px: Denormalized corner coordinates, shape (4, 2).
        gray: Grayscale image array.
        threshold: Brightness threshold (0-255).
        patch_size: Side length of the sampling patch.

    Returns:
        True if 2+ corners have mean brightness >= threshold.
    """
    h, w = gray.shape
    half = patch_size // 2
    bright_count = 0
    for cx, cy in corners_px:
        ix, iy = round(cx), round(cy)
        y0 = max(0, iy - half)
        y1 = min(h, iy + half + 1)
        x0 = max(0, ix - half)
        x1 = min(w, ix + half + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        patch = gray[y0:y1, x0:x1]
        if patch.mean() >= threshold:
            bright_count += 1
    return bright_count >= 1


def build_inpaint_mask(
    sand_corners_list: list[np.ndarray],
    img_shape: tuple[int, int],
    dilate_px: int = 5,
) -> np.ndarray:
    """Build a binary inpainting mask from sand-fish OBB corners.

    Args:
        sand_corners_list: List of denormalized corner arrays, each (4, 2).
        img_shape: (height, width) of the image.
        dilate_px: Dilation radius in pixels.

    Returns:
        uint8 mask with 255 for regions to inpaint.
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    for corners_px in sand_corners_list:
        pts = corners_px.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask


def sample_sand_patch(
    image: np.ndarray, patch_h: int = 128, patch_w: int = 128
) -> np.ndarray:
    """Sample a sand texture patch from the lower-center of the image.

    The lower-center is always sand substrate with no fish or walls.

    Args:
        image: BGR image array.
        patch_h: Height of patch to sample.
        patch_w: Width of patch to sample.

    Returns:
        BGR patch of shape (patch_h, patch_w, 3).
    """
    img_h, img_w = image.shape[:2]
    # Center horizontally, 75% down vertically
    cy = int(img_h * 0.75)
    cx = img_w // 2
    y0 = cy - patch_h // 2
    x0 = cx - patch_w // 2
    return image[y0 : y0 + patch_h, x0 : x0 + patch_w].copy()


def sand_tile_fill(
    image: np.ndarray,
    mask: np.ndarray,
    feather_px: int = 15,
    patch_h: int = 128,
    patch_w: int = 128,
) -> np.ndarray:
    """Fill masked regions by tiling a sand texture patch with feathered blending.

    Samples a clean sand patch from the lower-center of the image, tiles it
    across the full image, then alpha-blends into the masked regions using a
    Gaussian-feathered mask edge for seamless transitions.

    Args:
        image: BGR image array.
        mask: uint8 binary mask (255 = fill region).
        feather_px: Gaussian blur radius for feathering the mask edge.
        patch_h: Height of the sand texture patch.
        patch_w: Width of the sand texture patch.

    Returns:
        BGR image with masked regions filled by tiled sand texture.
    """
    img_h, img_w = image.shape[:2]
    patch = sample_sand_patch(image, patch_h, patch_w)

    # Tile the patch across the full image
    tiles_y = (img_h + patch_h - 1) // patch_h
    tiles_x = (img_w + patch_w - 1) // patch_w
    tiled = np.tile(patch, (tiles_y, tiles_x, 1))[:img_h, :img_w]

    # Feather the mask edges for smooth blending
    ksize = feather_px * 2 + 1
    alpha = cv2.GaussianBlur(mask, (ksize, ksize), 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]  # (H, W, 1) for broadcasting

    # Blend: tiled sand where mask is hot, original where mask is cold
    result = tiled.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha)
    return result.astype(np.uint8)


def process_image(
    img_path: Path,
    label_path: Path,
    output_img_dir: Path,
    output_lbl_dir: Path,
    brightness_threshold: int,
    patch_size: int,
    dry_run: bool,
) -> dict[str, int] | None:
    """Process a single image/label pair.

    Returns:
        Dict with counts if image was processed, None if skipped.
    """
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    annotations = parse_obb_label(label_path)
    if not annotations:
        return None

    wall_lines: list[str] = []
    sand_corners: list[np.ndarray] = []

    for raw_line, corners_norm in annotations:
        corners_px = corners_norm.copy()
        corners_px[:, 0] *= img_w
        corners_px[:, 1] *= img_h

        if is_wall_fish(corners_px, gray, brightness_threshold, patch_size):
            wall_lines.append(raw_line)
        else:
            sand_corners.append(corners_px)

    n_wall = len(wall_lines)
    n_sand = len(sand_corners)

    # Only process images with both wall-fish and sand-fish
    if n_wall == 0 or n_sand == 0:
        return None

    if dry_run:
        return {"wall": n_wall, "sand": n_sand}

    # Fill sand-fish regions with tiled sand texture
    mask = build_inpaint_mask(sand_corners, (img_h, img_w))
    inpainted = sand_tile_fill(image, mask)

    # Save outputs
    cv2.imwrite(str(output_img_dir / img_path.name), inpainted)
    (output_lbl_dir / label_path.name).write_text("\n".join(wall_lines) + "\n")

    return {"wall": n_wall, "sand": n_sand}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate wall-fish inpaint augmentations for OBB training data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.home() / "aquapose/projects/YH/training_data/obb",
        help="Root OBB directory containing images/ and labels/ subdirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "aquapose/projects/YH/training_data/obb/wall_augmented",
        help="Output directory for inpainted images and filtered labels.",
    )
    parser.add_argument(
        "--brightness-threshold",
        type=int,
        default=135,
        help="Corner brightness threshold for wall detection (0-255).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=11,
        help="Side length of brightness sampling patch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report detections without writing output files.",
    )
    args = parser.parse_args()

    img_dir = args.input_dir / "images"
    lbl_dir = args.input_dir / "labels"

    if not img_dir.is_dir():
        print(f"Error: images directory not found: {img_dir}")
        return
    if not lbl_dir.is_dir():
        print(f"Error: labels directory not found: {lbl_dir}")
        return

    output_img_dir = args.output_dir / "images"
    output_lbl_dir = args.output_dir / "labels"
    if not args.dry_run:
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    total_processed = 0

    for img_path in image_paths:
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        result = process_image(
            img_path,
            label_path,
            output_img_dir,
            output_lbl_dir,
            args.brightness_threshold,
            args.patch_size,
            args.dry_run,
        )
        if result is not None:
            total_processed += 1
            print(
                f"{'[DRY] ' if args.dry_run else ''}"
                f"{img_path.name}: {result['wall']} wall, {result['sand']} sand"
            )

    print(
        f"\n{'Would process' if args.dry_run else 'Processed'} {total_processed} images."
    )


if __name__ == "__main__":
    main()
