"""Click CLI commands for training data management (import, convert)."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import click
import yaml

logger = logging.getLogger(__name__)


@click.group("data")
def data_group() -> None:
    """Manage training data stores."""


@data_group.command("import")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Project config YAML for path resolution.",
)
@click.option(
    "--store",
    required=True,
    type=click.Choice(["obb", "pose"]),
    help="Which store to import into.",
)
@click.option(
    "--source",
    required=True,
    type=click.Choice(["manual", "corrected", "pseudo"]),
    help="Data source type.",
)
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True),
    help="YOLO-format directory with images/ and labels/.",
)
@click.option(
    "--augment", is_flag=True, help="Generate elastic augmentation variants on import."
)
@click.option(
    "--augment-count",
    type=int,
    default=4,
    show_default=True,
    help="Number of variants per sample when --augment is set.",
)
@click.option(
    "--augment-angle-range",
    type=float,
    default=15.0,
    show_default=True,
    help="Max deformation angle in degrees.",
)
@click.option("--batch-id", type=str, default=None, help="Shared import batch ID.")
@click.option(
    "--metadata-json",
    type=str,
    default=None,
    help="JSON string of additional metadata to attach to all imported samples.",
)
def import_cmd(
    config: str,
    store: str,
    source: str,
    input_dir: str,
    augment: bool,
    augment_count: int,
    augment_angle_range: float,
    batch_id: str | None,
    metadata_json: str | None,
) -> None:
    """Import YOLO-format training data into the store."""
    import cv2

    from .elastic_deform import generate_variants, parse_pose_label
    from .store import SampleStore

    # Resolve project dir from config
    config_data = yaml.safe_load(Path(config).read_text())
    project_dir = Path(config_data["project_dir"])
    store_db = project_dir / "training_data" / store / "store.db"

    # Parse extra metadata
    metadata = json.loads(metadata_json) if metadata_json else None

    input_path = Path(input_dir)

    # Scan for image files
    image_files = sorted(
        list(input_path.glob("images/**/*.jpg"))
        + list(input_path.glob("images/**/*.png"))
    )

    if not image_files:
        click.echo("No image files found in input directory.")
        return

    # Augmentation skip message for OBB
    augment_pose = augment
    if augment and store == "obb":
        click.echo(
            "Augmentation only applies to pose (crop-space) data, "
            "skipping --augment for OBB store."
        )
        augment_pose = False

    counts = {"imported": 0, "upserted": 0, "skipped": 0, "augmented": 0}
    cascade_deleted = 0

    with SampleStore(store_db) as sample_store:
        # Count children before upserts to detect cascade deletions
        def _count_children(sample_id: str) -> int:
            conn = sample_store._connect()
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM samples WHERE parent_id = ?",
                (sample_id,),
            ).fetchone()
            return row["cnt"]

        for img_path in image_files:
            # Find matching label by stem in labels/ directory
            rel = img_path.relative_to(input_path / "images")
            label_rel = rel.parent / f"{img_path.stem}.txt"
            label_path = input_path / "labels" / label_rel

            if not label_path.exists():
                continue

            # Check pre-existing children count for cascade tracking
            content_hash = SampleStore.compute_content_hash(img_path)
            conn = sample_store._connect()
            existing = conn.execute(
                "SELECT id FROM samples WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            pre_children = 0
            if existing is not None:
                pre_children = _count_children(existing["id"])

            sample_id, action = sample_store.import_sample(
                image_path=img_path,
                label_path=label_path,
                source=source,
                metadata=metadata,
                import_batch_id=batch_id,
            )
            counts[action] += 1

            if action == "upserted" and pre_children > 0:
                cascade_deleted += pre_children

            # Augmentation for pose store
            if augment_pose and action == "imported":
                img = cv2.imread(str(sample_store.images_dir / f"{sample_id}.jpg"))
                if img is None:
                    continue

                crop_h, crop_w = img.shape[:2]
                label_store_path = sample_store.labels_dir / f"{sample_id}.txt"
                coords, visible = parse_pose_label(label_store_path, crop_w, crop_h)

                lateral_pad = crop_h * 0.18

                variants = generate_variants(
                    image=img,
                    coords=coords,
                    visible=visible,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    lateral_pad=lateral_pad,
                    angle_range=(5.0, augment_angle_range),
                )

                for variant in variants:
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    ) as tmp_img:
                        cv2.imwrite(tmp_img.name, variant["image"])
                        tmp_img_path = Path(tmp_img.name)

                    with tempfile.NamedTemporaryFile(
                        suffix=".txt", delete=False, mode="w"
                    ) as tmp_lbl:
                        line = " ".join(str(round(v, 6)) for v in variant["pose_line"])
                        tmp_lbl.write(line + "\n")
                        tmp_lbl_path = Path(tmp_lbl.name)

                    sample_store.add_augmented(
                        parent_id=sample_id,
                        image_path=tmp_img_path,
                        label_path=tmp_lbl_path,
                        metadata={"variant_tag": variant["variant_tag"]},
                    )
                    counts["augmented"] += 1

                    # Clean up temp files
                    tmp_img_path.unlink(missing_ok=True)
                    tmp_lbl_path.unlink(missing_ok=True)

    # Print cascade deletion warning
    if cascade_deleted > 0:
        click.echo(
            f"Warning: {cascade_deleted} augmented variants were cascade-deleted "
            "during upserts. Re-run with --augment to regenerate."
        )

    # Print summary
    click.echo(
        f"Import complete: {counts['imported']} imported, "
        f"{counts['upserted']} upserted, {counts['skipped']} skipped, "
        f"{counts['augmented']} augmented variants created"
    )


@data_group.command("convert")
@click.option(
    "--coco-file",
    required=True,
    type=click.Path(exists=True),
    help="Path to COCO keypoints JSON.",
)
@click.option(
    "--images-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing source images.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for YOLO-format data.",
)
@click.option(
    "--type",
    "convert_type",
    required=True,
    type=click.Choice(["obb", "pose", "both"]),
    help="What to generate.",
)
@click.option(
    "--n-keypoints", type=int, default=6, show_default=True, help="Number of keypoints."
)
@click.option(
    "--crop-width",
    type=int,
    default=128,
    show_default=True,
    help="Crop width for pose.",
)
@click.option(
    "--crop-height",
    type=int,
    default=64,
    show_default=True,
    help="Crop height for pose.",
)
@click.option(
    "--lateral-ratio",
    type=float,
    default=0.18,
    show_default=True,
    help="OBB lateral padding as fraction of median arc length.",
)
@click.option(
    "--min-visible",
    type=int,
    default=4,
    show_default=True,
    help="Minimum visible keypoints to include (pose).",
)
@click.option(
    "--val-split",
    type=float,
    default=0.2,
    show_default=True,
    help="Fraction for validation.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
@click.option(
    "--edge-threshold-factor",
    type=float,
    default=2.0,
    show_default=True,
    help="Multiplier for edge keypoint extrapolation.",
)
def convert_cmd(
    coco_file: str,
    images_dir: str,
    output_dir: str,
    convert_type: str,
    n_keypoints: int,
    crop_width: int,
    crop_height: int,
    lateral_ratio: float,
    min_visible: int,
    val_split: float,
    seed: int,
    edge_threshold_factor: float,
) -> None:
    """Convert COCO annotations to YOLO-OBB and/or YOLO-Pose format."""
    from .coco_convert import (
        compute_median_arc_length,
        generate_obb_dataset,
        generate_pose_dataset,
        load_coco,
    )

    coco_path = Path(coco_file)
    img_dir = Path(images_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading COCO JSON: {coco_path}")
    coco = load_coco(coco_path)

    all_annotations = coco.get("annotations", [])
    click.echo(f"  Images: {len(coco.get('images', []))}")
    click.echo(f"  Annotations: {len(all_annotations)}")

    median_arc = compute_median_arc_length(all_annotations, n_keypoints)
    click.echo(f"  Median arc length: {median_arc:.1f} px")

    if convert_type in ("obb", "both"):
        click.echo("Generating YOLO-OBB dataset...")
        obb_train, obb_val = generate_obb_dataset(
            coco,
            images_dir=img_dir,
            output_dir=out_dir,
            median_arc=median_arc,
            lateral_ratio=lateral_ratio,
            edge_factor=edge_threshold_factor,
            val_split=val_split,
            seed=seed,
            n_keypoints=n_keypoints,
        )
        click.echo(f"OBB: {obb_train} train, {obb_val} val")

    if convert_type in ("pose", "both"):
        click.echo("Generating YOLO-Pose dataset...")
        pose_train, pose_val = generate_pose_dataset(
            coco,
            images_dir=img_dir,
            output_dir=out_dir,
            median_arc=median_arc,
            lateral_ratio=lateral_ratio,
            edge_factor=edge_threshold_factor,
            crop_w=crop_width,
            crop_h=crop_height,
            min_visible=min_visible,
            val_split=val_split,
            seed=seed,
        )
        click.echo(f"Pose: {pose_train} train, {pose_val} val")

    click.echo("Conversion complete.")
