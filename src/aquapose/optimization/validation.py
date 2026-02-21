"""Cross-view holdout validation and visual overlay utilities for pose reconstruction."""

from __future__ import annotations

import numpy as np
import torch

from aquapose.mesh.builder import build_fish_mesh
from aquapose.mesh.state import FishState

from .loss import soft_iou_loss


def evaluate_holdout_iou(
    state: FishState,
    held_out_camera,
    held_out_mask: torch.Tensor,
    renderer,
    crop_region: tuple[int, int, int, int] | None = None,
) -> float:
    """Evaluate IoU of the rendered mesh against a held-out camera mask.

    Renders the fish mesh into the held-out camera (without gradients) and
    computes the soft IoU against the provided binary mask.

    Args:
        state: Optimised FishState representing the fish pose.
        held_out_camera: RefractiveProjectionModel for the held-out camera.
        held_out_mask: Binary mask tensor, shape (H, W), float32, values in {0, 1}.
        renderer: RefractiveSilhouetteRenderer instance with a ``render`` method.
        crop_region: Optional (y1, x1, y2, x2) to restrict IoU computation to the
            fish bounding box. If None, the full frame is used.

    Returns:
        IoU as a Python float in [0, 1]. Higher is better.
    """
    with torch.no_grad():
        meshes = build_fish_mesh([state])
        cam_id = "__holdout__"
        alpha_maps = renderer.render(meshes, [held_out_camera], [cam_id])
        alpha = alpha_maps[cam_id]

        loss = soft_iou_loss(alpha, held_out_mask, crop_region=crop_region)
        iou = 1.0 - float(loss.item())

    return iou


def run_holdout_validation(
    states: list[FishState],
    frames_data: list[dict],
    cameras: list,
    camera_ids: list[str],
    renderer,
    optimizer,
) -> dict:
    """Run cross-view holdout validation over a sequence of frames.

    For each frame, holds out one camera (rotating round-robin across cameras)
    and evaluates the IoU of the optimised mesh on the excluded camera.

    The optimization is performed using all N-1 cameras; the held-out camera is
    only used for IoU evaluation, not during optimization.

    Args:
        states: List of already-optimised FishState objects, one per frame.
            If provided (non-empty), the optimizer is skipped and existing states
            are used directly. Pass an empty list to run optimization.
        frames_data: List of dicts, one per frame. Each dict must have:
            - ``"target_masks"``: Dict camera_id -> mask tensor, shape (H, W).
            - ``"crop_regions"``: Dict camera_id -> (y1, x1, y2, x2) or None.
        cameras: List of RefractiveProjectionModel instances.
        camera_ids: Camera identifier strings matching ``cameras``.
        renderer: RefractiveSilhouetteRenderer instance.
        optimizer: FishOptimizer instance (used when ``states`` is empty or
            needs to be re-run per held-out camera).

    Returns:
        Dict with keys:
            - ``"global_mean_iou"``: float, average IoU across all frame-camera pairs.
            - ``"per_camera_iou"``: dict[str, float], mean IoU per held-out camera.
            - ``"per_frame_iou"``: list of dicts, each with ``"frame_idx"``,
              ``"held_out_camera"``, and ``"iou"``.
            - ``"min_camera_iou"``: float, worst per-camera mean IoU.
            - ``"target_met_080"``: bool, global mean >= 0.80.
            - ``"target_met_060_floor"``: bool, all cameras >= 0.60.
    """
    n_cameras = len(camera_ids)
    n_frames = len(frames_data)

    per_frame_results: list[dict] = []
    per_camera_accum: dict[str, list[float]] = {cam_id: [] for cam_id in camera_ids}

    for frame_idx in range(n_frames):
        frame = frames_data[frame_idx]
        target_masks = frame["target_masks"]
        crop_regions = frame["crop_regions"]

        # Rotate held-out camera round-robin across frames.
        held_out_idx = frame_idx % n_cameras
        held_out_id = camera_ids[held_out_idx]
        held_out_camera = cameras[held_out_idx]
        held_out_mask = target_masks.get(held_out_id)

        if held_out_mask is None:
            continue

        # Build training camera subset (N-1 cameras).
        train_camera_ids = [cid for cid in camera_ids if cid != held_out_id]
        train_cameras = [
            cam
            for cam, cid in zip(cameras, camera_ids, strict=True)
            if cid != held_out_id
        ]
        train_masks = {cid: m for cid, m in target_masks.items() if cid != held_out_id}
        train_crops = {cid: c for cid, c in crop_regions.items() if cid != held_out_id}

        # Use pre-optimised state if available, otherwise optimize on N-1 cameras.
        if states and frame_idx < len(states):
            opt_state = states[frame_idx]
        else:
            # Optimize on N-1 cameras for this frame.
            frame_data_subset = [
                {"target_masks": train_masks, "crop_regions": train_crops}
            ]
            if frame_idx == 0:
                init_state = _get_initial_state(
                    target_masks, train_cameras, train_camera_ids
                )
                opt_states = optimizer.optimize_sequence(
                    init_state, frame_data_subset, train_cameras, train_camera_ids
                )
            else:
                # For holdout on subsequent frames, use the first available state as init.
                prev_state = (
                    states[frame_idx - 1]
                    if (states and frame_idx - 1 < len(states))
                    else None
                )
                if prev_state is None:
                    continue
                opt_states = optimizer.optimize_sequence(
                    prev_state, frame_data_subset, train_cameras, train_camera_ids
                )
            opt_state = opt_states[0]

        # Evaluate IoU on held-out camera.
        held_out_crop = crop_regions.get(held_out_id)
        iou = evaluate_holdout_iou(
            opt_state,
            held_out_camera,
            held_out_mask,
            renderer,
            crop_region=held_out_crop,
        )

        per_frame_results.append(
            {
                "frame_idx": frame_idx,
                "held_out_camera": held_out_id,
                "iou": iou,
            }
        )
        per_camera_accum[held_out_id].append(iou)

    # Aggregate metrics.
    per_camera_iou = {
        cam_id: float(np.mean(ious)) if ious else 0.0
        for cam_id, ious in per_camera_accum.items()
    }

    all_ious = [r["iou"] for r in per_frame_results]
    global_mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    min_camera_iou = min(per_camera_iou.values()) if per_camera_iou else 0.0

    target_met_080 = global_mean_iou >= 0.80
    target_met_060_floor = all(v >= 0.60 for v in per_camera_iou.values() if v > 0.0)

    # Print summary.
    print("\n=== Holdout Validation Results ===")
    print(
        f"  Global mean IoU : {global_mean_iou:.4f}  (target: >= 0.80 {'PASS' if target_met_080 else 'FAIL'})"
    )
    print(
        f"  Min camera IoU  : {min_camera_iou:.4f}  (floor: >= 0.60 {'PASS' if target_met_060_floor else 'FAIL'})"
    )
    print(f"  Frames evaluated: {len(per_frame_results)} / {n_frames}")
    print("\n  Per-camera breakdown:")
    for cam_id, iou in sorted(per_camera_iou.items()):
        flag = "" if iou >= 0.60 else " [BELOW FLOOR]"
        print(f"    {cam_id:20s}: {iou:.4f}{flag}")

    return {
        "global_mean_iou": global_mean_iou,
        "per_camera_iou": per_camera_iou,
        "per_frame_iou": per_frame_results,
        "min_camera_iou": min_camera_iou,
        "target_met_080": target_met_080,
        "target_met_060_floor": target_met_060_floor,
    }


def _get_initial_state(
    target_masks: dict,
    cameras: list,
    camera_ids: list[str],
) -> FishState:
    """Create a default initial FishState at the origin for testing/fallback.

    Args:
        target_masks: Dict of camera_id -> mask (unused; present for signature symmetry).
        cameras: List of camera models (unused).
        camera_ids: Camera IDs (unused).

    Returns:
        A FishState at world origin with default pose.
    """
    return FishState(
        p=torch.tensor([0.0, 0.0, 1.5]),
        psi=torch.tensor(0.0),
        theta=torch.tensor(0.0),
        kappa=torch.tensor(0.0),
        s=torch.tensor(0.15),
    )


def render_overlay(
    frame_bgr: np.ndarray,
    alpha_map: np.ndarray,
    crop_region: tuple[int, int, int, int] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    opacity: float = 0.4,
) -> np.ndarray:
    """Overlay a rendered mesh silhouette onto a camera frame.

    Blends the alpha map (rendered silhouette) onto the original BGR frame as a
    colored transparent overlay. Where ``alpha_map`` is high (near 1.0), the
    frame is blended toward ``color`` at the given ``opacity``.

    Args:
        frame_bgr: Original camera frame, shape (H, W, 3), uint8, BGR color order.
        alpha_map: Rendered silhouette alpha map. Can be:
            - Shape (H, W): full-frame alpha, matched directly to ``frame_bgr``.
            - Shape (h, w): crop-sized alpha; ``crop_region`` must be provided to
              specify where to paste it into the full frame.
        crop_region: Optional (y1, x1, y2, x2) in frame pixel coordinates. If
            provided, ``alpha_map`` is pasted into a zero canvas at this region
            before blending. Required when ``alpha_map`` is crop-sized.
        color: BGR overlay color, default (0, 255, 0) (green).
        opacity: Blend strength in [0, 1]. 0 means no overlay; 1 means solid color
            wherever alpha > 0.

    Returns:
        BGR overlay image, shape (H, W, 3), uint8. Pixels where alpha is nonzero
        are blended with ``color``; background pixels are unchanged.
    """
    frame_h, frame_w = frame_bgr.shape[:2]

    # Convert alpha_map to float32 in [0, 1] if needed.
    if alpha_map.dtype != np.float32:
        alpha_f = alpha_map.astype(np.float32)
        if alpha_f.max() > 1.0:
            alpha_f = alpha_f / 255.0
    else:
        alpha_f = alpha_map

    # If crop_region given, paste crop-sized alpha into full-frame canvas.
    if crop_region is not None:
        y1, x1, y2, x2 = crop_region
        full_alpha = np.zeros((frame_h, frame_w), dtype=np.float32)
        # Clip to frame bounds for safety.
        cy1, cx1 = max(0, y1), max(0, x1)
        cy2, cx2 = min(frame_h, y2), min(frame_w, x2)
        ah = cy2 - cy1
        aw = cx2 - cx1
        if ah > 0 and aw > 0:
            full_alpha[cy1:cy2, cx1:cx2] = alpha_f[:ah, :aw]
        alpha_f = full_alpha

    # Reshape alpha for broadcasting: (H, W, 1).
    alpha_3d = alpha_f[:, :, np.newaxis]  # (H, W, 1)

    # Build color array matching frame shape.
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)  # (1, 1, 3)

    # Blend: output = frame * (1 - alpha * opacity) + color * alpha * opacity
    frame_f = frame_bgr.astype(np.float32)
    blended = frame_f * (1.0 - alpha_3d * opacity) + color_arr * (alpha_3d * opacity)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended
