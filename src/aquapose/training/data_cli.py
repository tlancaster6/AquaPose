"""Click CLI commands for training data management."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import click

from aquapose.cli_utils import get_project_dir

if TYPE_CHECKING:
    from .store import SampleStore

logger = logging.getLogger(__name__)


@click.group("data")
def data_group() -> None:
    """Manage training data stores."""


@data_group.command("import")
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
@click.pass_context
def import_cmd(
    ctx: click.Context,
    store: str,
    source: str,
    input_dir: str,
    augment: bool,
    augment_count: int,  # reserved for generate_variants n_variants
    augment_angle_range: float,
    batch_id: str | None,
    metadata_json: str | None,
) -> None:
    """Import YOLO-format training data into the store."""
    import cv2

    from .elastic_deform import generate_variants, parse_pose_label
    from .store import SampleStore

    # Resolve project dir from context
    project_dir = get_project_dir(ctx)
    store_db = project_dir / "training_data" / store / "store.db"

    # Parse extra metadata
    metadata = json.loads(metadata_json) if metadata_json else None

    input_path = Path(input_dir)

    # Auto-detect confidence.json sidecar in input directory
    sidecar_data: dict = {}
    sidecar_path = input_path / "confidence.json"
    if sidecar_path.exists():
        sidecar_data = json.loads(sidecar_path.read_text())
        click.echo(f"Found confidence.json sidecar ({len(sidecar_data)} entries)")

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

            # Build per-sample metadata: sidecar + curvature + static overlay
            sample_meta: dict = {}

            # Sidecar metadata (keyed by image stem)
            sidecar_entry = sidecar_data.get(img_path.stem, {})
            if sidecar_entry:
                labels_list = sidecar_entry.get("labels", [])
                if labels_list:
                    first_label = labels_list[0]
                    for key in (
                        "confidence",
                        "gap_reason",
                        "n_source_cameras",
                        "raw_metrics",
                        "source",
                    ):
                        if key in first_label:
                            sample_meta[key] = first_label[key]

            # 2D curvature computation (pose store only)
            if store == "pose" and label_path.exists():
                from .elastic_deform import parse_pose_label
                from .pseudo_labels import compute_curvature

                try:
                    # Use crop_w=crop_h=1 since curvature is scale-invariant
                    kps, vis = parse_pose_label(label_path, 1, 1)
                    vis_pts = kps[vis]
                    if len(vis_pts) >= 3:
                        sample_meta["curvature"] = compute_curvature(vis_pts)
                except ValueError:
                    pass  # multi-line label (multi-fish crop), skip curvature

            # Static metadata from --metadata-json overrides sidecar
            if metadata:
                sample_meta.update(metadata)

            sample_id, action = sample_store.import_sample(
                image_path=img_path,
                label_path=label_path,
                source=source,
                metadata=sample_meta or None,
                import_batch_id=batch_id,
            )
            counts[action] += 1

            # Tag samples from val/ subdirectory
            if len(rel.parts) > 1 and rel.parts[0] == "val":
                conn = sample_store._connect()
                row = conn.execute(
                    "SELECT tags FROM samples WHERE id = ?", (sample_id,)
                ).fetchone()
                if row is not None:
                    tags = json.loads(row["tags"])
                    if "val" not in tags:
                        tags.append("val")
                        conn.execute(
                            "UPDATE samples SET tags = ? WHERE id = ?",
                            (json.dumps(tags), sample_id),
                        )
                        conn.commit()

            if action == "upserted" and pre_children > 0:
                cascade_deleted += pre_children

            # Augmentation for pose store (skip val samples to avoid leakage)
            is_val = len(rel.parts) > 1 and rel.parts[0] == "val"
            if augment_pose and action == "imported" and not is_val:
                img = cv2.imread(str(sample_store.images_dir / f"{sample_id}.jpg"))
                if img is None:
                    continue

                crop_h, crop_w = img.shape[:2]
                label_store_path = sample_store.labels_dir / f"{sample_id}.txt"
                try:
                    coords, visible = parse_pose_label(label_store_path, crop_w, crop_h)
                except ValueError:
                    # Skip multi-fish crops for augmentation
                    continue

                if len(coords) == 0:
                    # No visible keypoints — nothing to deform
                    continue

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
@click.option(
    "--split-mode",
    type=click.Choice(["temporal", "random"]),
    default="random",
    show_default=True,
    help="Split strategy: 'temporal' groups by frame index, 'random' shuffles.",
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
    split_mode: str,
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
            split_mode=split_mode,
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
            split_mode=split_mode,
        )
        click.echo(f"Pose: {pose_train} train, {pose_val} val")

    click.echo("Conversion complete.")


def _resolve_store_from_ctx(ctx: click.Context, store: str) -> Path:
    """Resolve the store DB path from click context and store type.

    Args:
        ctx: Click context with project resolution.
        store: Store type (``"obb"`` or ``"pose"``).

    Returns:
        Path to the store database file.
    """
    project_dir = get_project_dir(ctx)
    return project_dir / "training_data" / store / "store.db"


def _resolve_ids_by_filename(
    sample_store: SampleStore,
    filenames: list[str],
) -> list[str]:
    """Resolve original filenames (image stems) to sample IDs.

    Args:
        sample_store: Open sample store instance.
        filenames: List of original image stems to look up.

    Returns:
        List of resolved sample IDs.
    """
    conn = sample_store._connect()
    resolved: list[str] = []
    for fname in filenames:
        # Match against the stored image_path stem
        rows = conn.execute(
            "SELECT id, image_path FROM samples WHERE parent_id IS NULL"
        ).fetchall()
        matches = [r["id"] for r in rows if Path(r["image_path"]).stem == fname]
        if not matches:
            click.echo(f"Warning: no sample found for filename '{fname}'")
        else:
            resolved.extend(matches)
    return resolved


@data_group.command("assemble")
@click.option(
    "--store",
    required=True,
    type=click.Choice(["obb", "pose"]),
    help="Which store to assemble from.",
)
@click.option("--name", required=True, type=str, help="Dataset name.")
@click.option(
    "--source",
    type=str,
    multiple=True,
    help="Filter by source type(s). Can be repeated.",
)
@click.option(
    "--tags-include",
    type=str,
    multiple=True,
    help="Include samples with ALL listed tags.",
)
@click.option(
    "--tags-exclude",
    type=str,
    multiple=True,
    help="Exclude samples with ANY listed tag.",
)
@click.option(
    "--min-confidence",
    type=float,
    default=None,
    help="Minimum confidence threshold.",
)
@click.option(
    "--val-fraction",
    type=float,
    default=0.2,
    show_default=True,
    help="Fraction for validation split.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
@click.option(
    "--pseudo-in-val",
    is_flag=True,
    help="Allow pseudo-labels in validation split.",
)
@click.option(
    "--split-mode",
    type=click.Choice(["tagged", "random"]),
    default="random",
    show_default=True,
    help="Split strategy: 'tagged' uses val tag, 'random' shuffles.",
)
@click.option(
    "--val-candidates",
    type=str,
    default=None,
    help="Tag that marks val-eligible samples (random mode only).",
)
@click.option(
    "--include-excluded",
    is_flag=True,
    help="Include excluded samples (for uncurated A/B comparison).",
)
@click.option(
    "--max-aug-per-parent",
    type=int,
    default=None,
    help="Max augmented children per parent in train. None = all.",
)
@click.pass_context
def assemble_cmd(
    ctx: click.Context,
    store: str,
    name: str,
    source: tuple[str, ...],
    tags_include: tuple[str, ...],
    tags_exclude: tuple[str, ...],
    min_confidence: float | None,
    val_fraction: float,
    seed: int,
    pseudo_in_val: bool,
    split_mode: str,
    val_candidates: str | None,
    include_excluded: bool,
    max_aug_per_parent: int | None,
) -> None:
    """Assemble a training dataset with symlinks from store."""
    from .store import SampleStore

    store_db = _resolve_store_from_ctx(ctx, store)

    # Build query dict from filter options
    query: dict = {}
    if len(source) == 1:
        query["source"] = source[0]
    if tags_include:
        query["tags_include"] = list(tags_include)
    if tags_exclude:
        query["tags_exclude"] = list(tags_exclude)
    if min_confidence is not None:
        query["min_confidence"] = min_confidence
    if include_excluded:
        query["exclude_excluded"] = False

    with SampleStore(store_db) as sample_store:
        ds_path = sample_store.assemble(
            name=name,
            query=query,
            val_fraction=val_fraction,
            seed=seed,
            pseudo_in_val=pseudo_in_val,
            split_mode=split_mode,
            val_candidates_tag=val_candidates,
            max_aug_per_parent=max_aug_per_parent,
        )

        # Count results
        train_count = len(list((ds_path / "images" / "train").iterdir()))
        val_count = len(list((ds_path / "images" / "val").iterdir()))

    click.echo(f"Assembled dataset '{name}': {train_count} train, {val_count} val")
    click.echo(f"Path: {ds_path}")


@data_group.command("status")
@click.pass_context
def status_cmd(ctx: click.Context) -> None:
    """Show cross-store summary of training data."""
    from .store import SampleStore

    project_dir = get_project_dir(ctx)

    click.echo("Training Data Status")
    click.echo("=" * 40)

    for store_type in ("obb", "pose"):
        store_db = project_dir / "training_data" / store_type / "store.db"
        label = store_type.upper()

        if not store_db.exists():
            click.echo(f"{label} Store: No data (run `aquapose data import` first)")
            continue

        with SampleStore(store_db) as store:
            s = store.summary()
            source_parts = ", ".join(
                f"{count} {src}" for src, count in sorted(s["by_source"].items())
            )
            click.echo(f"{label} Store: {s['total']} samples ({source_parts})")

            # Build exclusion breakdown by reason
            excluded_str = str(s["excluded_count"])
            if s["excluded_count"] > 0:
                conn = store._connect()
                excluded_rows = conn.execute(
                    "SELECT tags FROM samples WHERE "
                    "EXISTS (SELECT 1 FROM json_each(tags) "
                    "WHERE json_each.value = 'excluded')"
                ).fetchall()
                reason_counts: dict[str, int] = {}
                reserved_tags = {"excluded", "val", "augmented"}
                for row in excluded_rows:
                    tags = json.loads(row["tags"])
                    for tag in tags:
                        if tag not in reserved_tags:
                            reason_counts[tag] = reason_counts.get(tag, 0) + 1
                if reason_counts:
                    parts = ", ".join(
                        f"{cnt} {reason}"
                        for reason, cnt in sorted(reason_counts.items())
                    )
                    excluded_str = f"{s['excluded_count']} ({parts})"

            click.echo(
                f"  Augmented: {s['augmented_count']} | "
                f"Excluded: {excluded_str} | "
                f"Datasets: {s['dataset_count']} | "
                f"Models: {s['model_count']}"
            )

    click.echo("=" * 40)


@data_group.command("list")
@click.option(
    "--store",
    required=True,
    type=click.Choice(["obb", "pose"]),
    help="Which store to list.",
)
@click.option("--verbose", is_flag=True, help="Show individual samples.")
@click.pass_context
def list_cmd(ctx: click.Context, store: str, verbose: bool) -> None:
    """List store contents with summary statistics."""
    from .store import SampleStore

    store_db = _resolve_store_from_ctx(ctx, store)

    with SampleStore(store_db) as sample_store:
        s = sample_store.summary()

        click.echo(f"{store.upper()} Store Summary")
        click.echo(f"  Total: {s['total']}")
        for src, count in sorted(s["by_source"].items()):
            click.echo(f"  {src}: {count}")
        click.echo(f"  Augmented: {s['augmented_count']}")
        click.echo(f"  Excluded: {s['excluded_count']}")
        click.echo(f"  Datasets: {s['dataset_count']}")

        if verbose:
            samples = sample_store.query(exclude_excluded=False)
            click.echo("")
            click.echo(f"{'ID':38s} {'Source':10s} {'Tags':20s} {'Created':25s}")
            click.echo("-" * 93)
            for sample in samples:
                tags = (
                    json.loads(sample["tags"])
                    if isinstance(sample["tags"], str)
                    else sample["tags"]
                )
                click.echo(
                    f"{sample['id']:38s} {sample['source']:10s} "
                    f"{','.join(tags) if tags else '-':20s} "
                    f"{sample['created_at']:25s}"
                )


@data_group.command("exclude")
@click.option(
    "--store",
    required=True,
    type=click.Choice(["obb", "pose"]),
    help="Which store.",
)
@click.option("--ids", type=str, multiple=True, help="Sample IDs to exclude.")
@click.option(
    "--source", type=str, default=None, help="Exclude all samples from this source."
)
@click.option(
    "--reason",
    type=str,
    default=None,
    help="Reason tag for exclusion (e.g. 'bad_crop', 'occluded').",
)
@click.option(
    "--by-filename",
    is_flag=True,
    help="Treat --ids as original filenames (image stems) instead of sample IDs.",
)
@click.pass_context
def exclude_cmd(
    ctx: click.Context,
    store: str,
    ids: tuple[str, ...],
    source: str | None,
    reason: str | None,
    by_filename: bool,
) -> None:
    """Soft-delete samples (reversible via include)."""
    from .store import SampleStore

    store_db = _resolve_store_from_ctx(ctx, store)

    with SampleStore(store_db) as sample_store:
        if ids:
            if by_filename:
                sample_ids = _resolve_ids_by_filename(sample_store, list(ids))
            else:
                sample_ids = list(ids)
        elif source:
            samples = sample_store.query(source=source)
            sample_ids = [s["id"] for s in samples]
        else:
            click.echo("Error: Provide --ids or --source to select samples.")
            return

        count = sample_store.exclude(sample_ids, reason=reason)
        click.echo(f"Excluded {count} samples")


@data_group.command("include")
@click.option(
    "--store",
    required=True,
    type=click.Choice(["obb", "pose"]),
    help="Which store.",
)
@click.option("--ids", type=str, multiple=True, help="Sample IDs to include.")
@click.option(
    "--source",
    type=str,
    default=None,
    help="Include all excluded samples from this source.",
)
@click.pass_context
def include_cmd(
    ctx: click.Context, store: str, ids: tuple[str, ...], source: str | None
) -> None:
    """Reverse exclusion (remove 'excluded' tag)."""
    from .store import SampleStore

    store_db = _resolve_store_from_ctx(ctx, store)

    with SampleStore(store_db) as sample_store:
        if ids:
            sample_ids = list(ids)
        elif source:
            samples = sample_store.query(source=source, exclude_excluded=False)
            sample_ids = [s["id"] for s in samples]
        else:
            click.echo("Error: Provide --ids or --source to select samples.")
            return

        count = sample_store.include(sample_ids)
        click.echo(f"Included {count} samples")


@data_group.command("remove")
@click.option(
    "--store",
    required=True,
    type=click.Choice(["obb", "pose"]),
    help="Which store.",
)
@click.option("--ids", type=str, multiple=True, help="Sample IDs to remove.")
@click.option(
    "--source", type=str, default=None, help="Remove all samples from this source."
)
@click.option("--purge", is_flag=True, help="Confirm permanent deletion.")
@click.pass_context
def remove_cmd(
    ctx: click.Context,
    store: str,
    ids: tuple[str, ...],
    source: str | None,
    purge: bool,
) -> None:
    """Hard-delete samples (permanent, cascades to children)."""
    from .store import SampleStore

    if not purge:
        click.echo(
            "Error: Pass --purge to confirm permanent deletion of files "
            "and database records."
        )
        return

    store_db = _resolve_store_from_ctx(ctx, store)

    with SampleStore(store_db) as sample_store:
        if ids:
            sample_ids = list(ids)
        elif source:
            samples = sample_store.query(source=source, exclude_excluded=False)
            sample_ids = [s["id"] for s in samples]
        else:
            click.echo("Error: Provide --ids or --source to select samples.")
            return

        count = sample_store.remove(sample_ids)
        click.echo(f"Removed {count} samples permanently")
