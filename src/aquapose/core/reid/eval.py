"""Zero-shot re-identification evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_reid_metrics(
    npz_path: Path,
    n_eval_frames: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Compute zero-shot ReID metrics from an embeddings NPZ file.

    Loads embeddings, samples ``n_eval_frames`` unique frame indices, and
    computes pairwise cosine similarity statistics plus retrieval metrics
    (Rank-1 accuracy and mAP).

    Args:
        npz_path: Path to ``embeddings.npz`` with arrays: ``embeddings``,
            ``frame_index``, ``fish_id``, ``camera_id``.
        n_eval_frames: Number of frames to sample for evaluation.
        seed: Random seed for reproducible frame sampling.

    Returns:
        Dict with keys: ``mean_within_sim``, ``mean_between_sim``,
        ``sim_gap``, ``rank1``, ``mAP``, ``n_eval_embeddings``,
        ``n_eval_frames``.
    """
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    frame_index = data["frame_index"]
    fish_ids = data["fish_id"]
    camera_ids = data["camera_id"]

    # Sample eval frames
    rng = np.random.default_rng(seed)
    unique_frames = np.unique(frame_index)
    n_sample = min(n_eval_frames, len(unique_frames))
    sampled_frames = set(rng.choice(unique_frames, size=n_sample, replace=False))

    # Filter to sampled frames
    mask = np.array([f in sampled_frames for f in frame_index])
    eval_embs = embeddings[mask]
    eval_fids = fish_ids[mask]
    eval_frames = frame_index[mask]
    eval_cams = camera_ids[mask]

    n_eval = len(eval_embs)
    if n_eval < 2:
        return {
            "mean_within_sim": 0.0,
            "mean_between_sim": 0.0,
            "sim_gap": 0.0,
            "rank1": 0.0,
            "mAP": 0.0,
            "n_eval_embeddings": n_eval,
            "n_eval_frames": n_sample,
        }

    # Cosine similarity matrix (embeddings are already L2-normalized)
    sim_matrix = eval_embs @ eval_embs.T

    # Build masks
    same_id = eval_fids[:, None] == eval_fids[None, :]
    diagonal = np.eye(n_eval, dtype=bool)
    # Same-frame same-camera mask (for excluding self-retrieval)
    same_frame_cam = (eval_frames[:, None] == eval_frames[None, :]) & np.array(
        [[ci == cj for cj in eval_cams] for ci in eval_cams]
    )

    # Within-ID and between-ID similarity (exclude diagonal)
    within_mask = same_id & ~diagonal
    between_mask = ~same_id & ~diagonal

    mean_within = float(sim_matrix[within_mask].mean()) if within_mask.any() else 0.0
    mean_between = float(sim_matrix[between_mask].mean()) if between_mask.any() else 0.0

    # Rank-1 accuracy
    # For each query, find highest-sim non-self, non-same-frame-cam embedding
    n_correct = 0
    n_queries = 0
    all_aps: list[float] = []

    for i in range(n_eval):
        # Valid gallery: exclude same-frame same-camera
        valid = ~same_frame_cam[i]
        valid[i] = False  # exclude self

        if not valid.any():
            continue

        sims_i = sim_matrix[i][valid]
        fids_i = eval_fids[valid]
        query_fid = eval_fids[i]

        # Sort by similarity descending
        order = np.argsort(-sims_i)
        sorted_fids = fids_i[order]

        # Rank-1
        n_queries += 1
        if sorted_fids[0] == query_fid:
            n_correct += 1

        # Average Precision
        relevant = sorted_fids == query_fid
        if not relevant.any():
            all_aps.append(0.0)
            continue

        cum_relevant = np.cumsum(relevant)
        precision_at_k = cum_relevant / np.arange(1, len(relevant) + 1)
        ap = float((precision_at_k * relevant).sum() / relevant.sum())
        all_aps.append(ap)

    rank1 = n_correct / n_queries if n_queries > 0 else 0.0
    map_score = float(np.mean(all_aps)) if all_aps else 0.0

    return {
        "mean_within_sim": mean_within,
        "mean_between_sim": mean_between,
        "sim_gap": mean_within - mean_between,
        "rank1": rank1,
        "mAP": map_score,
        "n_eval_embeddings": n_eval,
        "n_eval_frames": n_sample,
    }


def print_reid_report(metrics: dict[str, float]) -> None:
    """Pretty-print zero-shot ReID evaluation metrics.

    Args:
        metrics: Dict returned by :func:`compute_reid_metrics`.
    """
    print("=== Zero-Shot ReID Evaluation ===")
    print(
        f"Eval frames:    {metrics['n_eval_frames']:.0f} "
        f"({metrics['n_eval_embeddings']:.0f} embeddings)"
    )
    print(f"Within-ID sim:  {metrics['mean_within_sim']:.3f}")
    print(f"Between-ID sim: {metrics['mean_between_sim']:.3f}")
    print(f"Sim gap:        {metrics['sim_gap']:.3f}")
    print(f"Rank-1 acc:     {metrics['rank1'] * 100:.1f}%")
    print(f"mAP:            {metrics['mAP'] * 100:.1f}%")
