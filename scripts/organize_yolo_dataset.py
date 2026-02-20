"""Organize YOLO annotation export into ultralytics dataset structure.

Creates a stratified 80/20 train/val split per camera.
Expects image and label filenames to match (same stem).
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


def extract_camera_id(name: str) -> str:
    """Extract short camera ID (e3v8250) from filename."""
    match = re.match(r"(e3v[a-f0-9]+)", name)
    return match.group(1) if match else name.split("_")[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize YOLO export into ultralytics dataset structure"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to directory with images/ and labels/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo_fish"),
        help="Output dataset directory (default: data/yolo_fish)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of images per camera for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for split (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = args.input_dir / "images"
    labels_dir = args.input_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Match images to labels by stem, group by camera
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    camera_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unmatched = []

    for img_file in sorted(images_dir.glob("*.jpg")):
        stem = img_file.stem
        if stem in label_stems:
            camera_id = extract_camera_id(stem)
            camera_pairs[camera_id].append((img_file.name, f"{stem}.txt"))
        else:
            unmatched.append(img_file.name)

    matched = sum(len(v) for v in camera_pairs.values())
    print(f"Matched {matched}/{len(list(images_dir.glob('*.jpg')))} images to labels")
    if unmatched:
        print(f"WARNING: {len(unmatched)} images have no matching label")

    # Create output directories
    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Stratified split per camera
    train_count = 0
    val_count = 0
    print(f"\n{'Camera':<10} {'Total':>5} {'Train':>5} {'Val':>5}")
    print("-" * 30)

    for camera_id in sorted(camera_pairs):
        pairs = camera_pairs[camera_id]
        random.shuffle(pairs)
        n_val = max(1, round(len(pairs) * args.val_fraction))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        for img_name, lbl_name in train_pairs:
            shutil.copy2(
                images_dir / img_name, args.output_dir / "images" / "train" / img_name
            )
            shutil.copy2(
                labels_dir / lbl_name, args.output_dir / "labels" / "train" / lbl_name
            )

        for img_name, lbl_name in val_pairs:
            shutil.copy2(
                images_dir / img_name, args.output_dir / "images" / "val" / img_name
            )
            shutil.copy2(
                labels_dir / lbl_name, args.output_dir / "labels" / "val" / lbl_name
            )

        print(
            f"{camera_id:<10} {len(pairs):>5} {len(train_pairs):>5} {len(val_pairs):>5}"
        )
        train_count += len(train_pairs)
        val_count += len(val_pairs)

    print("-" * 30)
    print(f"{'TOTAL':<10} {train_count + val_count:>5} {train_count:>5} {val_count:>5}")

    # Create dataset.yaml
    dataset_yaml = args.output_dir / "dataset.yaml"
    abs_path = args.output_dir.resolve().as_posix()
    dataset_yaml.write_text(
        f"path: {abs_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: fish\n"
    )
    print(f"\nDataset config: {dataset_yaml}")


if __name__ == "__main__":
    main()
