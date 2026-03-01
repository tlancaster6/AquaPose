"""Training visualization utilities: augmented data grids and validation prediction grids."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _tensor_to_hwc(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) float32 tensor in [0, 1] to an (H, W, 3) uint8 array.

    Args:
        image_tensor: Float32 tensor of shape (3, H, W) with values in [0, 1].

    Returns:
        uint8 numpy array of shape (H, W, 3) with values in [0, 255].
    """
    arr = image_tensor.cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def save_unet_augmented_grid(
    dataset: object,
    output_path: Path,
    n_samples: int = 16,
) -> None:
    """Save a grid of augmented training samples with GT mask overlays.

    Samples ``n_samples`` items from a ``BinaryMaskDataset`` (or Subset) and
    renders each as an image with the binary GT mask overlaid in red at
    alpha=0.4.  The grid is arranged as ceil(n_samples / 4) rows x 4 columns.

    Args:
        dataset: A dataset whose ``__getitem__`` returns
            ``(image_tensor [3,H,W] float32, mask_tensor [1,H,W] float32)``.
        output_path: Path where the PNG file will be saved.
        n_samples: Number of samples to include in the grid.
    """
    import cv2

    torch.manual_seed(0)
    n_available = len(dataset)  # type: ignore[arg-type]
    n = min(n_samples, n_available)
    indices = random.sample(range(n_available), n)

    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes_flat = np.array(axes).flatten()

    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]  # type: ignore[index]
        img_hwc = _tensor_to_hwc(image_tensor)
        mask_np = mask_tensor.cpu().numpy()[0]  # (H, W) in {0, 1}

        # Build contour overlay on a copy of the image
        overlay = img_hwc.copy()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Fill mask region with semi-transparent red
        red_overlay = overlay.copy()
        red_overlay[mask_uint8 > 0] = [200, 50, 50]
        combined = cv2.addWeighted(overlay, 0.6, red_overlay, 0.4, 0)
        # Draw contour outlines clearly
        cv2.drawContours(combined, contours, -1, (255, 80, 80), 1)

        axes_flat[i].imshow(combined)
        axes_flat[i].axis("off")
        axes_flat[i].set_title(f"idx={idx}", fontsize=7)

    # Hide unused subplot axes
    for j in range(len(indices), len(axes_flat)):
        axes_flat[j].axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Augmented Training Data — GT Mask Overlay", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_unet_val_grid(
    model: torch.nn.Module,
    val_dataset: object,
    device: str,
    output_path: Path,
    n_samples: int = 16,
) -> None:
    """Save a grid of validation samples with GT and predicted mask overlays.

    Each grid cell shows the image with the GT mask contour in green and the
    predicted mask contour in red, so discrepancies are immediately visible.
    The model must already have best-checkpoint weights loaded before this
    function is called.

    Args:
        model: Trained ``_UNet`` model in eval mode.
        val_dataset: Validation dataset whose ``__getitem__`` returns
            ``(image_tensor [3,H,W] float32, mask_tensor [1,H,W] float32)``.
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        output_path: Path where the PNG file will be saved.
        n_samples: Number of samples to include in the grid.
    """
    import cv2

    torch.manual_seed(0)
    n_available = len(val_dataset)  # type: ignore[arg-type]
    n = min(n_samples, n_available)
    indices = random.sample(range(n_available), n)

    model.eval()

    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes_flat = np.array(axes).flatten()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_tensor, mask_tensor = val_dataset[idx]  # type: ignore[index]
            img_hwc = _tensor_to_hwc(image_tensor)

            # Run inference
            inp = image_tensor.unsqueeze(0).to(device)
            pred_prob = model(inp)  # (1, 1, H, W)
            pred_binary = (pred_prob > 0.5).float().cpu().numpy()[0, 0]  # (H, W)
            gt_mask = mask_tensor.cpu().numpy()[0]  # (H, W)

            # Extract contours for GT (green) and prediction (red)
            gt_uint8 = (gt_mask * 255).astype(np.uint8)
            pred_uint8 = (pred_binary * 255).astype(np.uint8)

            gt_contours, _ = cv2.findContours(
                gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            pred_contours, _ = cv2.findContours(
                pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            combined = img_hwc.copy()
            cv2.drawContours(combined, gt_contours, -1, (50, 200, 50), 1)
            cv2.drawContours(combined, pred_contours, -1, (200, 50, 50), 1)

            axes_flat[i].imshow(combined)
            axes_flat[i].axis("off")
            axes_flat[i].set_title(f"idx={idx}", fontsize=7)

    for j in range(len(indices), len(axes_flat)):
        axes_flat[j].axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(
        "Validation Predictions — GT (green) vs Pred (red) Contours", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pose_augmented_grid(
    dataset: object,
    output_path: Path,
    n_samples: int = 16,
    input_size: tuple[int, int] = (128, 64),
) -> None:
    """Save a grid of augmented pose training samples with GT keypoint overlays.

    Each grid cell shows the image with visible keypoints as colored circles
    and invisible keypoints as gray X marks.

    Args:
        dataset: A dataset whose ``__getitem__`` returns
            ``(image_tensor [3,H,W] float32, kp_flat [n_kp*2] float32,
            visibility [n_kp] bool)``.
        output_path: Path where the PNG file will be saved.
        n_samples: Number of samples to include in the grid.
        input_size: ``(width, height)`` of the model input image in pixels.
            Used to de-normalise keypoint coordinates.
    """
    torch.manual_seed(0)
    n_available = len(dataset)  # type: ignore[arg-type]
    n = min(n_samples, n_available)
    indices = random.sample(range(n_available), n)

    width, height = input_size
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes_flat = np.array(axes).flatten()

    for i, idx in enumerate(indices):
        image_tensor, kp_flat, visibility = dataset[idx]  # type: ignore[index]
        img_hwc = _tensor_to_hwc(image_tensor)

        kp_np = kp_flat.cpu().numpy()  # (n_kp * 2,)
        vis_np = visibility.cpu().numpy()  # (n_kp,)
        n_kp = len(vis_np)

        xs = kp_np[0::2] * width  # de-normalise x coords
        ys = kp_np[1::2] * height  # de-normalise y coords

        ax = axes_flat[i]
        ax.imshow(img_hwc)

        # Colourmap for keypoints — use a fixed colour cycle
        cmap = plt.cm.get_cmap("tab10", n_kp)

        for k in range(n_kp):
            color = cmap(k)
            if vis_np[k]:
                ax.scatter(xs[k], ys[k], s=20, color=color, zorder=3)
            else:
                ax.scatter(xs[k], ys[k], s=20, color="gray", marker="x", zorder=3)

        ax.axis("off")
        ax.set_title(f"idx={idx}", fontsize=7)

    for j in range(len(indices), len(axes_flat)):
        axes_flat[j].axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(
        "Augmented Training Data — GT Keypoints (colored=visible, gray x=invisible)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pose_val_grid(
    model: torch.nn.Module,
    val_dataset: object,
    device: str,
    output_path: Path,
    n_samples: int = 16,
    input_size: tuple[int, int] = (128, 64),
) -> None:
    """Save a grid of validation pose samples with GT and predicted keypoints.

    Each cell shows the image with GT keypoints as green circles and predicted
    keypoints as red circles, connected by thin lines showing error magnitude.
    Only visible GT keypoints are shown.

    Args:
        model: Trained ``_PoseModel`` with best-checkpoint weights loaded.
        val_dataset: Validation dataset whose ``__getitem__`` returns
            ``(image_tensor [3,H,W] float32, kp_flat [n_kp*2] float32,
            visibility [n_kp] bool)``.
        device: Torch device string.
        output_path: Path where the PNG file will be saved.
        n_samples: Number of samples to include in the grid.
        input_size: ``(width, height)`` of the model input in pixels.
    """
    torch.manual_seed(0)
    n_available = len(val_dataset)  # type: ignore[arg-type]
    n = min(n_samples, n_available)
    indices = random.sample(range(n_available), n)

    model.eval()

    width, height = input_size
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes_flat = np.array(axes).flatten()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_tensor, kp_flat, visibility = val_dataset[idx]  # type: ignore[index]
            img_hwc = _tensor_to_hwc(image_tensor)

            # Run inference
            inp = image_tensor.unsqueeze(0).to(device)
            pred_flat = model(inp).squeeze(0).cpu()  # (n_kp*2,)

            gt_np = kp_flat.cpu().numpy()
            pred_np = pred_flat.numpy()
            vis_np = visibility.cpu().numpy()

            gt_xs = gt_np[0::2] * width
            gt_ys = gt_np[1::2] * height
            pred_xs = pred_np[0::2] * width
            pred_ys = pred_np[1::2] * height
            n_kp = len(vis_np)

            ax = axes_flat[i]
            ax.imshow(img_hwc)

            for k in range(n_kp):
                if not vis_np[k]:
                    continue
                # Draw connecting line between GT and predicted
                ax.plot(
                    [gt_xs[k], pred_xs[k]],
                    [gt_ys[k], pred_ys[k]],
                    color="yellow",
                    linewidth=0.8,
                    zorder=2,
                    alpha=0.7,
                )
                # GT keypoint (green)
                ax.scatter(gt_xs[k], gt_ys[k], s=20, color="lime", zorder=3)
                # Predicted keypoint (red)
                ax.scatter(pred_xs[k], pred_ys[k], s=20, color="red", zorder=3)

            ax.axis("off")
            ax.set_title(f"idx={idx}", fontsize=7)

    for j in range(len(indices), len(axes_flat)):
        axes_flat[j].axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Validation Predictions — GT (green) vs Pred (red)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
