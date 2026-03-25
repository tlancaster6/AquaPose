#!/usr/bin/env python3
"""Train ReID projection head and evaluate discriminability gate.

Standalone driver script for fine-tuning MegaDescriptor-T embeddings to
discriminate individual fish (especially females). Orchestrates the full
workflow: build backbone cache, train projection head, evaluate with AUC
gate, and conditionally re-embed all detections.

Usage:
    hatch run python scripts/train_reid_head.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# =============================================================================
# Configuration — edit these paths for your setup
# =============================================================================

PROJECT_DIR = Path.home() / "aquapose" / "projects" / "YH"
RUN_DIR = PROJECT_DIR / "runs" / "run_20260307_140127"  # Phase 72 baseline run
REID_CROPS_DIR = RUN_DIR / "reid" / "reid_crops"
OUTPUT_DIR = RUN_DIR / "reid" / "fine_tune"
EMBEDDINGS_PATH = RUN_DIR / "reid" / "embeddings.npz"

AUC_GATE = 0.75

# Fish sex labels (YH project specific)
FEMALE_IDS = frozenset({0, 1, 2, 3, 4, 8})
MALE_IDS = frozenset({5, 6, 7})

# Known swap: fish 2 <-> fish 4 around frame 600

# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    """Run the full fine-tuning workflow."""
    from aquapose.training.reid_training import (
        ReidTrainingConfig,
        build_feature_cache,
        train_reid_head,
    )

    print("=" * 60)
    print("ReID Projection Head Fine-Tuning")
    print("=" * 60)
    print(f"  Reid crops:   {REID_CROPS_DIR}")
    print(f"  Output:       {OUTPUT_DIR}")
    print(f"  Embeddings:   {EMBEDDINGS_PATH}")
    print(f"  AUC gate:     {AUC_GATE}")
    print()

    config = ReidTrainingConfig(
        reid_crops_dir=REID_CROPS_DIR,
        output_dir=OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # Step 1: Build or load backbone feature cache
    # ------------------------------------------------------------------
    print("[Step 1] Building backbone feature cache...")
    t0 = time.time()
    cache = build_feature_cache(REID_CROPS_DIR, config)
    elapsed = time.time() - t0

    n_crops = len(cache["labels"])
    n_groups = len(set(cache["group_ids"].tolist()))
    unique_fish = sorted(set(int(x) for x in cache["labels"]))
    print(f"  Cache: {n_crops} crops, {n_groups} groups, {len(unique_fish)} fish IDs")
    for fid in unique_fish:
        count = int(np.sum(cache["labels"] == fid))
        print(f"    Fish {fid}: {count} crops")
    print(f"  Time: {elapsed:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Step 2: Train projection head
    # ------------------------------------------------------------------
    print("[Step 2] Training projection head...")
    t0 = time.time()
    result = train_reid_head(cache, config)
    elapsed = time.time() - t0

    print(f"  Best AUC:     {result['best_auc']:.4f}")
    print(f"  Best epoch:   {result['best_epoch']}")
    print(f"  Epochs run:   {result['epochs_run']}")
    print(f"  Time: {elapsed:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Step 3: Evaluate with per-pair breakdown
    # ------------------------------------------------------------------
    print("[Step 3] Evaluation Results")
    print("-" * 40)
    print(f"  Overall female-female AUC: {result['best_auc']:.4f}")
    print()

    if result["per_pair_auc"]:
        print("  Per-pair AUC breakdown:")
        sorted_pairs = sorted(result["per_pair_auc"].items(), key=lambda x: x[1])
        for (fi, fj), auc in sorted_pairs:
            marker = " <-- WORST" if (fi, fj) == sorted_pairs[0][0] else ""
            sex_i = "F" if fi in FEMALE_IDS else "M"
            sex_j = "F" if fj in FEMALE_IDS else "M"
            print(
                f"    Fish {fi}({sex_i}) vs Fish {fj}({sex_j}): AUC = {auc:.4f}{marker}"
            )

        worst_pair, worst_auc = sorted_pairs[0]
        print()
        print(
            f"  Bottleneck pair: Fish {worst_pair[0]} vs Fish {worst_pair[1]} (AUC = {worst_auc:.4f})"
        )
    print()

    # ------------------------------------------------------------------
    # Step 4: AUC gate decision
    # ------------------------------------------------------------------
    if result["best_auc"] >= AUC_GATE:
        print("=" * 60)
        print(f"  GATE PASSED  (AUC {result['best_auc']:.4f} >= {AUC_GATE})")
        print("=" * 60)
        print()

        # Re-embed all detections using trained projection head.
        _reembed_detections(config, result)

        print()
        print("Ready for Phase 105: Swap Detection and Repair")
    else:
        print("=" * 60)
        print(f"  GATE FAILED  (AUC {result['best_auc']:.4f} < {AUC_GATE})")
        print("=" * 60)
        print()

        if result["per_pair_auc"]:
            print("  Worst pairs contributing to low AUC:")
            sorted_pairs = sorted(result["per_pair_auc"].items(), key=lambda x: x[1])
            for (fi, fj), auc in sorted_pairs[:5]:
                print(f"    Fish {fi} vs Fish {fj}: AUC = {auc:.4f}")

        print()
        print("  Suggested next steps:")
        print("    1. Try progressive backbone unfreezing (last 2 Swin-T stages)")
        print("    2. Increase training data (mine more temporal windows)")
        print("    3. Adjust ArcFace margin/scale hyperparameters")
        print("    4. If AUC remains < 0.75, downscope to male-female only")
        sys.exit(1)


def _reembed_detections(config: object, result: dict) -> None:
    """Re-embed all detections using the trained projection head.

    Loads the existing zero-shot 768-dim embeddings, passes them through
    the trained ProjectionHead to produce 128-dim fine-tuned embeddings,
    and saves as embeddings_finetuned.npz with the same metadata arrays.

    Args:
        config: Training configuration.
        result: Training result dict (unused here but kept for API consistency).
    """
    from aquapose.training.reid_training import ProjectionHead

    print("[Step 4] Re-embedding all detections with fine-tuned head...")

    if not EMBEDDINGS_PATH.exists():
        print(f"  WARNING: Zero-shot embeddings not found at {EMBEDDINGS_PATH}")
        print("  Skipping re-embedding step.")
        return

    # Load zero-shot embeddings.
    data = np.load(str(EMBEDDINGS_PATH), allow_pickle=True)
    zs_embeddings = data["embeddings"]  # (N, 768) L2-normalized
    frame_index = data["frame_index"]
    fish_id = data["fish_id"]
    camera_id = data["camera_id"]
    detection_confidence = data["detection_confidence"]

    print(
        f"  Loaded {zs_embeddings.shape[0]} zero-shot embeddings ({zs_embeddings.shape[1]}-dim)"
    )

    # Load trained projection head.
    model_path = config.output_dir / "best_reid_model.pt"
    head = ProjectionHead(
        in_dim=config.backbone_dim,
        hidden_dim=config.hidden_dim,
        out_dim=config.embedding_dim,
    )
    head.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    head.eval()

    # Forward pass (CPU is fast enough for the small head).
    with torch.no_grad():
        zs_tensor = torch.from_numpy(zs_embeddings).float()
        ft_embeddings = head(zs_tensor).numpy()

    print(f"  Fine-tuned embeddings: {ft_embeddings.shape} ({ft_embeddings.dtype})")

    # Before/after within-ID cosine similarity comparison.
    print()
    print("  Before/after within-ID cosine similarity (females):")
    for fid in sorted(FEMALE_IDS):
        mask = fish_id == fid
        if mask.sum() < 2:
            continue

        # Zero-shot (768-dim).
        zs_sub = zs_embeddings[mask]
        zs_sim = zs_sub @ zs_sub.T
        n = len(zs_sub)
        zs_pairs = [zs_sim[i, j] for i in range(n) for j in range(i + 1, n)]
        zs_mean = float(np.mean(zs_pairs)) if zs_pairs else 0.0

        # Fine-tuned (128-dim).
        ft_sub = ft_embeddings[mask]
        ft_sim = ft_sub @ ft_sub.T
        ft_pairs = [ft_sim[i, j] for i in range(n) for j in range(i + 1, n)]
        ft_mean = float(np.mean(ft_pairs)) if ft_pairs else 0.0

        delta = ft_mean - zs_mean
        arrow = "+" if delta > 0 else ""
        print(f"    Fish {fid}: {zs_mean:.4f} -> {ft_mean:.4f} ({arrow}{delta:.4f})")

    # Save fine-tuned embeddings.
    output_path = config.output_dir / "embeddings_finetuned.npz"
    np.savez(
        output_path,
        embeddings=ft_embeddings,
        frame_index=frame_index,
        fish_id=fish_id,
        camera_id=camera_id,
        detection_confidence=detection_confidence,
    )
    print(f"\n  Saved fine-tuned embeddings to {output_path}")


if __name__ == "__main__":
    main()
