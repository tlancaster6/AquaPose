"""Click command group for fish re-identification operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from aquapose.cli_utils import get_project_dir, resolve_run

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from aquapose.training.reid_training import ReidTrainingConfig


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
@click.option(
    "--unfreeze-blocks",
    type=int,
    default=0,
    show_default=True,
    help="Number of Swin blocks to unfreeze (0 = frozen, use cached features).",
)
@click.option(
    "--lr-backbone-factor",
    type=float,
    default=0.1,
    show_default=True,
    help="Backbone LR = head LR * this factor.",
)
@click.pass_context
def fine_tune_cmd(
    ctx: click.Context,
    run: str | None,
    epochs: int,
    lr: float,
    auc_gate: float,
    device: str | None,
    unfreeze_blocks: int,
    lr_backbone_factor: float,
) -> None:
    """Fine-tune a ReID projection head and conditionally re-embed.

    Orchestrates the full workflow: build backbone feature cache, train
    projection head, evaluate AUC gate, and re-embed detections if gate passes.
    Prerequisites: run `reid mine-crops` and `reid embed` first.

    With --unfreeze-blocks > 0, trains the backbone end-to-end (no cached
    features) and re-embeds using the fine-tuned backbone + head.
    """
    import sys

    import numpy as np
    import torch

    from aquapose.training.reid_training import (
        ProjectionHead,
        ReidTrainingConfig,
        build_feature_cache,
        train_reid_end_to_end,
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
        unfreeze_blocks=unfreeze_blocks,
        lr_backbone_factor=lr_backbone_factor,
    )

    def _epoch_progress(epoch: int, total: int, loss: float, auc: float) -> None:
        marker = " *" if auc == result_holder.get("best_auc", -1) else ""
        click.echo(
            f"  Epoch {epoch:3d}/{total}  loss={loss:.4f}  val_auc={auc:.4f}{marker}"
        )

    result_holder: dict[str, float] = {}

    def _epoch_cb(epoch: int, total: int, loss: float, auc: float) -> None:
        if auc > result_holder.get("best_auc", -1):
            result_holder["best_auc"] = auc
        _epoch_progress(epoch, total, loss, auc)

    if unfreeze_blocks > 0:
        # End-to-end training path: skip feature caching, train backbone + head.
        click.echo(
            f"[Step 1] End-to-end training (unfreezing last {unfreeze_blocks} blocks)..."
        )
        result = train_reid_end_to_end(config, epoch_callback=_epoch_cb)
    else:
        # Frozen path: cached features + projection head only.
        click.echo("[Step 1] Building backbone feature cache...")
        cache = build_feature_cache(reid_crops_dir, config)
        n_crops = len(cache["labels"])
        n_groups = len(set(cache["group_ids"].tolist()))
        unique_fish = sorted({int(x) for x in cache["labels"]})
        click.echo(
            f"  Cache: {n_crops} crops, {n_groups} groups, {len(unique_fish)} fish IDs"
        )

        click.echo("[Step 2] Training projection head...")
        result = train_reid_head(cache, config, epoch_callback=_epoch_cb)

    click.echo(f"  Best AUC:   {result['best_auc']:.4f}")
    click.echo(f"  Best epoch: {result['best_epoch']}")

    # Print per-pair AUC breakdown if available.
    if result.get("per_pair_auc"):
        click.echo("  Per-pair AUC (females):")
        for (fish_a, fish_b), auc in sorted(result["per_pair_auc"].items()):
            click.echo(f"    Fish {fish_a} vs {fish_b}: {auc:.4f}")

    # AUC gate decision.
    if result["best_auc"] < auc_gate:
        click.echo(
            f"GATE FAILED (AUC {result['best_auc']:.4f} < {auc_gate}). "
            "Consider more training data or adjusted hyperparameters."
        )
        sys.exit(1)

    click.echo(f"GATE PASSED (AUC {result['best_auc']:.4f} >= {auc_gate})")

    # Re-embed all detections.
    model_path = output_dir / "best_reid_model.pt"
    data = np.load(str(embeddings_path), allow_pickle=True)

    if unfreeze_blocks > 0:
        # Fine-tuned backbone: re-embed by running crops through backbone + head.
        click.echo(
            "[Step 3] Re-embedding detections with fine-tuned backbone + head..."
        )
        ft_embeddings = _reembed_finetuned(
            run_dir=run_dir,
            checkpoint_path=model_path,
            config=config,
        )
    else:
        # Frozen path: apply projection head to zero-shot embeddings.
        click.echo("[Step 3] Re-embedding detections with fine-tuned head...")
        zs_embeddings = data["embeddings"]

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


def _reembed_finetuned(
    run_dir: Path,
    checkpoint_path: Path,
    config: ReidTrainingConfig,
) -> np.ndarray:
    """Re-embed all detections using a fine-tuned backbone + projection head.

    Runs ``EmbedRunner`` with a monkey-patched ``FishEmbedder`` that wraps
    the fine-tuned backbone and projection head.  The runner handles all crop
    extraction from video; the patched embedder replaces only the forward pass.

    The original ``embeddings.npz`` is backed up before re-embedding and
    restored afterward so that only ``embeddings_finetuned.npz`` (written
    by the caller) reflects the fine-tuned model.

    Args:
        run_dir: Path to the completed pipeline run directory.
        checkpoint_path: Path to the combined checkpoint
            (backbone_state_dict + head_state_dict + config).
        config: Training config (for model_name, crop_size, device, dims).

    Returns:
        Fine-tuned embeddings array of shape ``(N, embedding_dim)``.
    """
    import shutil
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import timm
    import torch

    import aquapose.core.reid.runner as runner_module
    from aquapose.core.reid.runner import EmbedRunner
    from aquapose.training.reid_training import ProjectionHead

    device = torch.device(config.device)

    # Load checkpoint.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Build fine-tuned backbone + head.
    backbone = timm.create_model(config.model_name, num_classes=0, pretrained=False)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.to(device)
    backbone.eval()

    head_cfg = ckpt.get("config", {})
    head = ProjectionHead(
        in_dim=head_cfg.get("backbone_dim", config.backbone_dim),
        hidden_dim=head_cfg.get("hidden_dim", config.hidden_dim),
        out_dim=head_cfg.get("embedding_dim", config.embedding_dim),
    )
    head.load_state_dict(ckpt["head_state_dict"])
    head.to(device)
    head.eval()

    # FishEmbedder-compatible wrapper using fine-tuned backbone + head.
    class _FineTunedEmbedder:
        """Drop-in for FishEmbedder using fine-tuned backbone + head."""

        def __init__(self, _config: object) -> None:
            self._crop_size = config.crop_size
            self._batch_size = config.batch_size
            self._embedding_dim = config.embedding_dim

        def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
            """Embed BGR crops through fine-tuned backbone + head."""
            import cv2

            if len(crops) == 0:
                return np.empty((0, config.embedding_dim), dtype=np.float32)

            all_feats = []
            with torch.no_grad():
                for i in range(0, len(crops), self._batch_size):
                    sub = crops[i : i + self._batch_size]
                    tensors = []
                    for crop in sub:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        resized = cv2.resize(
                            rgb,
                            (self._crop_size, self._crop_size),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        arr = resized.astype(np.float32) / 255.0
                        arr = (arr - 0.5) / 0.5
                        tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))
                    batch_tensor = torch.stack(tensors).to(device)
                    raw_feats = backbone(batch_tensor)
                    emb = head(raw_feats)  # ProjectionHead L2-normalizes
                    all_feats.append(emb.cpu().numpy())
                    del batch_tensor, raw_feats, emb

            return np.concatenate(all_feats, axis=0).astype(np.float32)

    # Config namespace for EmbedRunner (matches ReidConfigLike protocol).
    runner_config = SimpleNamespace(
        model_name=config.model_name,
        batch_size=config.batch_size,
        crop_size=config.crop_size,
        device=config.device,
        embedding_dim=config.embedding_dim,
    )

    # Back up original embeddings.npz so EmbedRunner.run() doesn't destroy it.
    embeddings_path = Path(run_dir) / "reid" / "embeddings.npz"
    backup_path = embeddings_path.with_suffix(".npz.bak")
    if embeddings_path.exists():
        shutil.copy2(embeddings_path, backup_path)

    runner = EmbedRunner(run_dir, runner_config, frame_stride=1)

    # Monkey-patch FishEmbedder so EmbedRunner uses our fine-tuned model.
    original_cls = runner_module.FishEmbedder  # type: ignore[attr-defined]
    runner_module.FishEmbedder = _FineTunedEmbedder  # type: ignore[assignment,attr-defined]
    try:
        output_path = runner.run()
    finally:
        runner_module.FishEmbedder = original_cls  # type: ignore[assignment,attr-defined]

    # Read fine-tuned embeddings from the file EmbedRunner wrote.
    data = np.load(str(output_path), allow_pickle=True)
    ft_embeddings: np.ndarray = data["embeddings"]

    # Restore original embeddings.npz.
    if backup_path.exists():
        shutil.move(str(backup_path), str(embeddings_path))

    return ft_embeddings


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

    # Margin distribution summary.
    margins = [e.cosine_margin for e in events if e.cosine_margin is not None]
    if margins:
        import numpy as np

        arr = np.array(margins)
        click.echo(
            f"  Margin stats: min={arr.min():.3f} max={arr.max():.3f} "
            f"mean={arr.mean():.3f} median={np.median(arr):.3f}"
        )
        click.echo(
            f"  Margins > 0: {(arr > 0).sum()}/{len(arr)}, "
            f"> threshold ({cosine_margin}): {(arr > cosine_margin).sum()}/{len(arr)}"
        )
    skipped = [e for e in events if e.action == "skipped_insufficient_data"]
    if skipped:
        click.echo(
            f"  Skipped {len(skipped)} events due to insufficient embedding data"
        )

    click.echo(f"Corrected midlines written to {repaired_path}")
