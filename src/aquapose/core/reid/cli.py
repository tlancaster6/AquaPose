"""Click command group for fish re-identification operations."""

from __future__ import annotations

import click

from aquapose.cli_utils import get_project_dir, resolve_run


@click.group("reid")
def reid_group() -> None:
    """Fish re-identification: embed, repair, mine-crops, fine-tune."""


@reid_group.command("embed")
@click.argument("run", default=None, required=False)
@click.option(
    "--weights",
    type=click.Path(exists=True, path_type=None),
    default=None,
    help="Path to fine-tuned projection head weights (.pt). "
    "Omit to use zero-shot MegaDescriptor-T embeddings.",
)
@click.option(
    "--frame-stride",
    type=int,
    default=1,
    show_default=True,
    help="Embed every Nth frame (1 = all frames).",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing embeddings.npz if present.",
)
@click.pass_context
def embed_cmd(
    ctx: click.Context,
    run: str | None,
    weights: str | None,
    frame_stride: int,
    overwrite: bool,
) -> None:
    """Embed fish crops from a completed pipeline run.

    Iterates chunk caches, extracts OBB-aligned crops, and writes
    reid/embeddings.npz with zero-shot MegaDescriptor-T vectors.
    Automatically prints zero-shot ReID metrics after embedding.
    """

    from types import SimpleNamespace

    from aquapose.core.reid.runner import EmbedRunner

    run_dir = resolve_run(run, get_project_dir(ctx))
    embeddings_path = run_dir / "reid" / "embeddings.npz"

    if embeddings_path.exists() and not overwrite:
        raise click.ClickException(
            f"Embeddings already exist at {embeddings_path}. "
            "Use --overwrite to replace them."
        )

    if weights is not None:
        click.echo(
            "WARNING: --weights flag accepted but projection head application "
            "is not yet wired into embed. Running zero-shot embedding instead."
        )

    # Build a config namespace matching EmbedRunner's expected attributes.
    # Uses the same defaults as ReidConfig in engine/config.py.
    config = SimpleNamespace(
        model_name="hf-hub:BVRA/MegaDescriptor-T-224",
        batch_size=32,
        crop_size=224,
        device="cuda",
        embedding_dim=768,
    )
    runner = EmbedRunner(run_dir, config, frame_stride=frame_stride)
    output_path = runner.run()
    click.echo(f"Embeddings written to {output_path}")


@reid_group.command("mine-crops")
@click.argument("run", default=None, required=False)
@click.option(
    "--window-size",
    type=int,
    default=300,
    show_default=True,
    help="Frames per temporal window.",
)
@click.option(
    "--window-stride",
    type=int,
    default=150,
    show_default=True,
    help="Stride between windows.",
)
@click.option(
    "--min-cooccurring",
    type=int,
    default=3,
    show_default=True,
    help="Minimum fish per grouping.",
)
@click.option(
    "--min-cameras",
    type=int,
    default=3,
    show_default=True,
    help="Minimum cameras for quality gate.",
)
@click.option(
    "--max-residual",
    type=float,
    default=5.0,
    show_default=True,
    help="Maximum mean residual (px).",
)
@click.option(
    "--min-duration",
    type=int,
    default=10,
    show_default=True,
    help="Minimum contiguous frames per segment.",
)
@click.option(
    "--crops-per-fish",
    type=int,
    default=8,
    show_default=True,
    help="Target crops per fish per grouping.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing reid_crops/ directory.",
)
@click.pass_context
def mine_crops_cmd(
    ctx: click.Context,
    run: str | None,
    window_size: int,
    window_stride: int,
    min_cooccurring: int,
    min_cameras: int,
    max_residual: float,
    min_duration: int,
    crops_per_fish: int,
    overwrite: bool,
) -> None:
    """Mine training crops from a completed pipeline run for ReID fine-tuning.

    Slides temporal windows across the run, applies quality gates, and extracts
    OBB-aligned crops organized into groupings suitable for contrastive learning.
    """
    import shutil

    from aquapose.core.reid.miner import MinerConfig, TrainingDataMiner

    run_dir = resolve_run(run, get_project_dir(ctx))

    config = MinerConfig(
        window_size=window_size,
        window_stride=window_stride,
        min_cooccurring=min_cooccurring,
        min_cameras=min_cameras,
        max_residual=max_residual,
        min_duration=min_duration,
        crops_per_fish=crops_per_fish,
    )

    if overwrite:
        crops_dir = run_dir / "reid_crops"
        if crops_dir.exists():
            shutil.rmtree(crops_dir)
            click.echo(f"Removed existing {crops_dir}")

    miner = TrainingDataMiner(run_dir, config=config)
    stats = miner.run()

    click.echo(f"Mined {stats['n_groups']} groupings, {stats['n_crops']} total crops")
    for fid, count in sorted(stats["per_fish_counts"].items()):
        click.echo(f"  Fish {fid}: {count} crops")


@reid_group.command("fine-tune")
@click.argument("run", default=None, required=False)
@click.option(
    "--epochs",
    type=int,
    default=50,
    show_default=True,
    help="Maximum training epochs.",
)
@click.option(
    "--lr",
    type=float,
    default=3e-4,
    show_default=True,
    help="Learning rate for the projection head.",
)
@click.option(
    "--auc-gate",
    type=float,
    default=0.75,
    show_default=True,
    help="Female-female AUC threshold; re-embed only if this gate passes.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Torch device (e.g. cuda, cpu). Auto-detect if omitted.",
)
@click.pass_context
def fine_tune_cmd(
    ctx: click.Context,
    run: str | None,
    epochs: int,
    lr: float,
    auc_gate: float,
    device: str | None,
) -> None:
    """Fine-tune a ReID projection head and conditionally re-embed.

    Orchestrates the full workflow: build backbone feature cache, train
    projection head, evaluate AUC gate, and re-embed detections if gate passes.
    Prerequisites: run `reid mine-crops` and `reid embed` first.
    """
    import sys

    import numpy as np
    import torch

    from aquapose.training.reid_training import (
        ProjectionHead,
        ReidTrainingConfig,
        build_feature_cache,
        train_reid_head,
    )

    run_dir = resolve_run(run, get_project_dir(ctx))
    reid_crops_dir = run_dir / "reid_crops"
    output_dir = run_dir / "reid" / "fine_tune"
    embeddings_path = run_dir / "reid" / "embeddings.npz"

    if not reid_crops_dir.exists():
        raise click.ClickException(
            f"ReID crops directory not found: {reid_crops_dir}. "
            "Run `reid mine-crops` first."
        )

    if not embeddings_path.exists():
        raise click.ClickException(
            f"Embeddings file not found: {embeddings_path}. Run `reid embed` first."
        )

    resolved_device = device or "cuda"

    config = ReidTrainingConfig(
        reid_crops_dir=reid_crops_dir,
        output_dir=output_dir,
        epochs=epochs,
        lr_head=lr,
        device=resolved_device,
    )

    # Step 1: Build backbone feature cache
    click.echo("[Step 1] Building backbone feature cache...")
    cache = build_feature_cache(reid_crops_dir, config)
    n_crops = len(cache["labels"])
    n_groups = len(set(cache["group_ids"].tolist()))
    unique_fish = sorted({int(x) for x in cache["labels"]})
    click.echo(
        f"  Cache: {n_crops} crops, {n_groups} groups, {len(unique_fish)} fish IDs"
    )

    # Step 2: Train projection head
    click.echo("[Step 2] Training projection head...")
    result = train_reid_head(cache, config)
    click.echo(f"  Best AUC:   {result['best_auc']:.4f}")
    click.echo(f"  Best epoch: {result['best_epoch']}")

    # Step 3: AUC gate decision
    if result["best_auc"] >= auc_gate:
        click.echo(f"GATE PASSED (AUC {result['best_auc']:.4f} >= {auc_gate})")

        # Re-embed all detections using the trained projection head.
        click.echo("[Step 4] Re-embedding detections with fine-tuned head...")
        data = np.load(str(embeddings_path), allow_pickle=True)
        zs_embeddings = data["embeddings"]

        model_path = output_dir / "best_reid_model.pt"
        head = ProjectionHead(
            in_dim=config.backbone_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.embedding_dim,
        )
        head.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        head.eval()

        with torch.no_grad():
            zs_tensor = torch.from_numpy(zs_embeddings).float()
            ft_embeddings = head(zs_tensor).cpu().numpy()

        finetuned_path = run_dir / "reid" / "embeddings_finetuned.npz"
        np.savez(
            finetuned_path,
            embeddings=ft_embeddings,
            frame_index=data["frame_index"],
            fish_id=data["fish_id"],
            camera_id=data["camera_id"],
            detection_confidence=data["detection_confidence"],
        )
        click.echo(f"Fine-tuned embeddings saved to {finetuned_path}")
    else:
        click.echo(
            f"GATE FAILED (AUC {result['best_auc']:.4f} < {auc_gate}). "
            "Consider more training data or adjusted hyperparameters."
        )
        sys.exit(1)


@reid_group.command("repair")
@click.argument("run", default=None, required=False)
@click.option(
    "--mode",
    type=click.Choice(["seeded", "scan"]),
    default="scan",
    show_default=True,
    help="Swap detection mode: seeded (fast, body-length triggered) or "
    "scan (thorough, embedding-based proximity scan).",
)
@click.option(
    "--cosine-margin",
    type=float,
    default=0.15,
    show_default=True,
    help="Minimum cosine margin to confirm a swap event.",
)
@click.option(
    "--weights",
    type=click.Path(exists=True, path_type=None),
    default=None,
    help="Path to trained projection head (.pt) for embedding transform.",
)
@click.pass_context
def repair_cmd(
    ctx: click.Context,
    run: str | None,
    mode: str,
    cosine_margin: float,
    weights: str | None,
) -> None:
    """Detect and repair fish identity swaps in a completed run.

    Loads embeddings.npz, runs the SwapDetector in the chosen mode,
    and writes a corrected midlines_reid.h5 with relabeled fish IDs.
    """
    from pathlib import Path

    from aquapose.core.reid.swap_detector import SwapDetector, SwapDetectorConfig

    run_dir = resolve_run(run, get_project_dir(ctx))

    weights_path = Path(weights) if weights is not None else None

    cfg = SwapDetectorConfig(cosine_margin_threshold=cosine_margin)
    detector = SwapDetector(run_dir, config=cfg, projection_head_path=weights_path)

    click.echo(f"Running swap detection (mode={mode})...")
    events = detector.run(mode=mode)

    confirmed = [e for e in events if e.action == "confirmed"]
    repaired_path = detector.repair(confirmed)

    click.echo(
        f"Detected {len(events)} events, confirmed {len(confirmed)} swaps, "
        f"repaired {len(confirmed)} fish ID relabelings."
    )
    click.echo(f"Corrected midlines written to {repaired_path}")
