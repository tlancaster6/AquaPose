"""Train YOLOv8n on the annotated fish bounding-box dataset.

Usage
-----
Run with default settings (expects data/yolo_fish/dataset.yaml)::

    python scripts/train_yolo.py

Override resolution or batch size if GPU VRAM is limited::

    python scripts/train_yolo.py --imgsz 1280 --batch 1

Trained weights are saved to::

    {--project}/{--name}/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for YOLO training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n fish detector on the annotated dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/yolo_fish/dataset.yaml",
        help="Path to dataset YAML file (ultralytics format).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model weights (COCO pretrained). Use yolov8s.pt for a larger model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1600,
        help="Training image size (long edge). Use 1280 if OOM at 1600.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size. -1 for auto (profiles GPU VRAM). Reduce to 1 if OOM.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience (epochs without val improvement).",
    )
    parser.add_argument(
        "--project",
        default="output/yolo_fish",
        help="Root output directory.",
    )
    parser.add_argument(
        "--name",
        default="train_v1",
        help="Run name (subdirectory under --project).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device: '0' for GPU 0, 'cpu' for CPU-only.",
    )
    return parser.parse_args()


def main() -> None:
    """Run YOLO training with the given arguments."""
    args = parse_args()

    from ultralytics import YOLO

    data_path = str(Path(args.data).resolve())

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    print(
        f"Starting training — data={data_path}, epochs={args.epochs}, "
        f"imgsz={args.imgsz}, batch={args.batch}, device={args.device}"
    )
    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        single_cls=True,
        augment=True,
        project=args.project,
        name=args.name,
        workers=0,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights saved to: {best_weights}")


if __name__ == "__main__":
    main()
