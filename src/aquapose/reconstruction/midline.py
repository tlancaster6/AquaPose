"""2D medial axis extraction and arc-length sampling pipeline.

Extracts ordered 15-point 2D midlines with half-widths from binary fish masks.
Produces Midline2D structs in full-frame pixel coordinates. Point ordering is
arbitrary (BFS traversal); cross-camera flip alignment is handled downstream.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.interpolate
import scipy.ndimage
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from aquapose.segmentation.crop import CropRegion

if TYPE_CHECKING:
    from aquapose.tracking.tracker import FishTrack

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class Midline2D:
    """Ordered 2D midline for a single fish in a single camera.

    Attributes:
        points: Full-frame pixel coordinates, shape (N, 2), float32.
            Column order is (x, y). Ordered from head to tail when
            ``is_head_to_tail`` is True.
        half_widths: Half-width of the fish at each midline point,
            shape (N,), float32, in full-frame pixels.
        fish_id: Globally unique fish identifier.
        camera_id: Camera identifier string.
        frame_index: Frame index within the video.
        is_head_to_tail: True when point[0] is the head end. False when
            orientation has not yet been established (first few frames).
    """

    points: np.ndarray
    half_widths: np.ndarray
    fish_id: int
    camera_id: str
    frame_index: int
    is_head_to_tail: bool = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_skip_mask(
    mask: np.ndarray,
    crop_region: CropRegion,
    min_area: int = 300,
) -> str | None:
    """Return a skip reason if the mask should be discarded, else None.

    Skips masks that are too small or have foreground pixels touching the
    full-frame image boundary (indicating the fish is partially outside
    the camera's field of view). Only checks crop edges that coincide
    with the full-frame boundary.

    Args:
        mask: Binary mask, uint8, values 0 or 255 (or 0/1). Shape (H, W).
        crop_region: CropRegion with full-frame position and frame dimensions.
        min_area: Minimum number of nonzero pixels required.

    Returns:
        Skip reason string, or None if the mask is valid.
    """
    nonzero = np.count_nonzero(mask)
    if nonzero < min_area:
        return f"too small: {nonzero} < {min_area}"

    # Only check edges where the crop touches the full-frame boundary.
    # Mask pixels at interior crop edges are fine (fish continues in frame).
    if crop_region.y1 == 0 and np.any(mask[0, :]):
        return "boundary-clipped: mask touches top frame edge"
    if crop_region.y2 >= crop_region.frame_h and np.any(mask[-1, :]):
        return "boundary-clipped: mask touches bottom frame edge"
    if crop_region.x1 == 0 and np.any(mask[:, 0]):
        return "boundary-clipped: mask touches left frame edge"
    if crop_region.x2 >= crop_region.frame_w and np.any(mask[:, -1]):
        return "boundary-clipped: mask touches right frame edge"

    return None


def _adaptive_smooth(mask: np.ndarray) -> np.ndarray:
    """Morphological closing then opening with an adaptive elliptical kernel.

    Kernel radius is derived from the minor axis length of the mask via
    skimage regionprops. This adapts smoothing to fish size.

    Args:
        mask: Binary mask, uint8. Shape (H, W).

    Returns:
        Smoothed binary mask, uint8 (0/255). Shape (H, W).
    """
    bool_mask = mask > 0

    # Keep only the largest connected component to discard stray blobs
    # (e.g. a second fish partially visible in the crop).
    labeled, n_labels = scipy.ndimage.label(bool_mask)
    if n_labels > 1:
        component_sizes = scipy.ndimage.sum(bool_mask, labeled, range(1, n_labels + 1))
        largest_label = int(np.argmax(component_sizes)) + 1
        bool_mask = labeled == largest_label

    props = regionprops(bool_mask.astype(np.uint8))
    minor = props[0].axis_minor_length if props else 10.0
    radius = max(3, int(minor) // 8)

    # Build elliptical structuring element
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((x**2 + y**2) <= radius**2).astype(np.uint8)

    # Closing: fills holes; Opening: removes small protrusions
    closed = scipy.ndimage.binary_closing(bool_mask, structure=kernel)
    opened = scipy.ndimage.binary_opening(closed, structure=kernel)
    return opened.astype(np.uint8) * 255


def _skeleton_and_widths(
    smooth_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute skeleton and distance transform from a smoothed mask.

    Args:
        smooth_mask: Smoothed binary mask, uint8 (0/255). Shape (H, W).

    Returns:
        skeleton_bool: Boolean skeleton array. Shape (H, W).
        distance_transform: Euclidean distance transform of the mask.
            Values represent half-width at each pixel. Shape (H, W).
    """
    bool_mask = smooth_mask > 0
    skeleton_bool: np.ndarray = np.asarray(skeletonize(bool_mask), dtype=bool)
    distance_transform: np.ndarray = np.asarray(
        scipy.ndimage.distance_transform_edt(bool_mask)
    )
    return skeleton_bool, distance_transform


def _longest_path_bfs(skel: np.ndarray) -> list[tuple[int, int]]:
    """Find the longest head-to-tail path in a skeleton via two-pass BFS.

    Uses two BFS passes to find the endpoints of the longest path, then
    reconstructs the ordered pixel sequence. Uses 8-connectivity.

    Args:
        skel: Boolean skeleton array. Shape (H, W).

    Returns:
        Ordered list of (row, col) pixel tuples, longest path from
        endpoint to endpoint. Returns [] if skeleton is empty.
    """
    if not np.any(skel):
        return []

    # Build pixel set for O(1) membership lookup
    yx_set: set[tuple[int, int]] = set(zip(*np.where(skel), strict=True))  # type: ignore[arg-type]

    def _neighbors(r: int, c: int) -> list[tuple[int, int]]:
        """8-connected neighbors that are on the skeleton."""
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in yx_set:
                    nbrs.append(nb)
        return nbrs

    def _bfs_farthest(
        start: tuple[int, int],
    ) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
        """BFS from start; return (farthest node, parent dict)."""
        visited: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        queue: deque[tuple[int, int]] = deque([start])
        farthest = start
        while queue:
            node = queue.popleft()
            farthest = node
            for nb in _neighbors(*node):
                if nb not in visited:
                    visited[nb] = node
                    queue.append(nb)
        return farthest, visited

    # Pick any skeleton pixel as seed
    seed = next(iter(yx_set))

    # Pass 1: find one endpoint (farthest from seed)
    end_a, _ = _bfs_farthest(seed)

    # Pass 2: find true farthest endpoint from end_a
    end_b, parents = _bfs_farthest(end_a)

    # Reconstruct path from end_b back to end_a
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end_b
    while cur is not None:
        path.append(cur)
        cur = parents[cur]

    return path  # ordered end_b -> end_a


def _resample_arc_length(
    path_yx: list[tuple[int, int]],
    dt: np.ndarray,
    n_points: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a skeleton path to n_points evenly-spaced arc-length positions.

    Computes cumulative Euclidean arc-length along the path, normalises to
    [0, 1], then interpolates x, y, and half-width at n_points uniform
    positions.

    Args:
        path_yx: Ordered list of (row, col) skeleton pixel tuples.
        dt: Distance transform array (half-widths at each pixel). Shape (H, W).
        n_points: Number of output points. Default 15.

    Returns:
        xy_crop: Resampled (x, y) crop-space coordinates, shape (N, 2), float32.
            Note: x = col, y = row.
        half_widths: Half-widths at each resampled point, shape (N,), float32.
    """
    rows = np.array([p[0] for p in path_yx], dtype=np.float64)
    cols = np.array([p[1] for p in path_yx], dtype=np.float64)

    # Cumulative arc-length
    dr = np.diff(rows)
    dc = np.diff(cols)
    seg_lengths = np.sqrt(dr**2 + dc**2)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    # Normalise to [0, 1]
    total_len = arc[-1]
    if total_len < 1e-6:
        # Degenerate path — return repeated single point
        xy = np.tile(np.array([[cols[0], rows[0]]], dtype=np.float32), (n_points, 1))
        hw = np.zeros(n_points, dtype=np.float32)
        return xy, hw

    arc_norm = arc / total_len

    # Half-widths along path from distance transform
    hw_path = np.array(
        [dt[int(r), int(c)] for r, c in zip(rows, cols, strict=True)],
        dtype=np.float64,
    )

    # Interpolators
    interp_x = scipy.interpolate.interp1d(arc_norm, cols, kind="linear")
    interp_y = scipy.interpolate.interp1d(arc_norm, rows, kind="linear")
    interp_hw = scipy.interpolate.interp1d(arc_norm, hw_path, kind="linear")

    # Evaluate at evenly-spaced positions
    t_eval = np.linspace(0.0, 1.0, n_points)
    x_out = interp_x(t_eval).astype(np.float32)
    y_out = interp_y(t_eval).astype(np.float32)
    hw_out = interp_hw(t_eval).astype(np.float32)

    xy_crop = np.stack([x_out, y_out], axis=1)  # (N, 2) in (x, y) = (col, row)
    return xy_crop, hw_out


def _crop_to_frame(
    xy_crop: np.ndarray,
    half_widths: np.ndarray,
    crop_region: CropRegion,
    crop_h: int,
    crop_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform crop-space coordinates to full-frame pixel coordinates.

    Accounts for the resize from crop_h x crop_w (the mask resolution,
    e.g. 128x128 from U-Net) to the actual crop region dimensions, then
    translates by the crop origin.

    Args:
        xy_crop: Crop-space (x, y) coordinates, shape (N, 2), float32.
            x = column, y = row in the resized mask.
        half_widths: Half-widths in crop-space pixels, shape (N,), float32.
        crop_region: CropRegion defining the full-frame position and size.
        crop_h: Height of the mask array (may differ from crop_region.height
            due to U-Net resize).
        crop_w: Width of the mask array (may differ from crop_region.width
            due to U-Net resize).

    Returns:
        xy_frame: Full-frame (x, y) pixel coordinates, shape (N, 2), float32.
        hw_frame: Half-widths in full-frame pixels, shape (N,), float32.
    """
    scale_x = crop_region.width / max(crop_w, 1)
    scale_y = crop_region.height / max(crop_h, 1)

    xy_frame = xy_crop.copy().astype(np.float32)
    xy_frame[:, 0] = xy_frame[:, 0] * scale_x + crop_region.x1
    xy_frame[:, 1] = xy_frame[:, 1] * scale_y + crop_region.y1

    avg_scale = (scale_x + scale_y) / 2.0
    hw_frame = (half_widths * avg_scale).astype(np.float32)

    return xy_frame, hw_frame


# ---------------------------------------------------------------------------
# Stateful extractor
# ---------------------------------------------------------------------------


class MidlineExtractor:
    """Extracts ordered 2D midlines from per-camera binary masks.

    Handles the full pipeline: mask validation, morphological smoothing,
    skeletonization, longest-path pruning, arc-length resampling, and
    coordinate transform. Head-to-tail orientation is deferred to the
    triangulation stage (cross-camera flip alignment).

    Args:
        n_points: Number of midline points to produce. Default 15.
        min_area: Minimum mask area (pixels) to attempt extraction. Default 300.
    """

    def __init__(
        self,
        n_points: int = 15,
        min_area: int = 300,
    ) -> None:
        self.n_points = n_points
        self.min_area = min_area

    def extract_midlines(
        self,
        tracks: list[FishTrack],
        masks_per_camera: dict[str, list[np.ndarray]],
        crop_regions_per_camera: dict[str, list[CropRegion]],
        detections_per_camera: dict[str, list],
        frame_index: int,
    ) -> dict[int, dict[str, Midline2D]]:
        """Extract 2D midlines for all tracks in a single frame.

        For each confirmed track, attempts midline extraction in every camera
        where it has a detection. Midline point ordering is arbitrary (BFS
        traversal order); cross-camera flip alignment is handled downstream
        in the triangulation stage.

        Args:
            tracks: List of FishTrack objects from the current frame.
            masks_per_camera: Per-camera list of binary masks (uint8).
            crop_regions_per_camera: Per-camera list of CropRegion objects.
            detections_per_camera: Per-camera detection lists (used for
                index lookup; content not directly inspected here).
            frame_index: Current frame index.

        Returns:
            Nested dict: fish_id → camera_id → Midline2D.
            Only includes entries where extraction succeeded.
        """
        results: dict[int, dict[str, Midline2D]] = {}

        for track in tracks:
            fish_id = track.fish_id
            fish_results: dict[str, Midline2D] = {}

            for camera_id, det_idx in track.camera_detections.items():
                masks = masks_per_camera.get(camera_id)
                crop_regions = crop_regions_per_camera.get(camera_id)
                if masks is None or crop_regions is None:
                    continue
                if det_idx >= len(masks) or det_idx >= len(crop_regions):
                    continue

                mask = masks[det_idx]
                crop_region = crop_regions[det_idx]
                crop_h, crop_w = mask.shape[:2]

                # 1. Check skip conditions
                skip_reason = _check_skip_mask(mask, crop_region, self.min_area)
                if skip_reason:
                    logger.debug(
                        "Skipping fish %d camera %s frame %d: %s",
                        fish_id,
                        camera_id,
                        frame_index,
                        skip_reason,
                    )
                    continue

                # 2. Smooth
                smooth = _adaptive_smooth(mask)

                # 3. Skeleton + distance transform
                skeleton_bool, dt = _skeleton_and_widths(smooth)

                # 4. Check skeleton has enough pixels
                n_skel = int(np.sum(skeleton_bool))
                if n_skel < self.n_points:
                    logger.debug(
                        "Skeleton too short (%d < %d) for fish %d camera %s frame %d",
                        n_skel,
                        self.n_points,
                        fish_id,
                        camera_id,
                        frame_index,
                    )
                    continue

                # 5. Longest-path BFS
                path_yx = _longest_path_bfs(skeleton_bool)
                if not path_yx:
                    continue

                # 6. Arc-length resample
                xy_crop, half_widths = _resample_arc_length(path_yx, dt, self.n_points)

                # 7. Crop to frame
                xy_frame, hw_frame = _crop_to_frame(
                    xy_crop, half_widths, crop_region, crop_h, crop_w
                )

                midline = Midline2D(
                    points=xy_frame,
                    half_widths=hw_frame,
                    fish_id=fish_id,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    is_head_to_tail=False,
                )
                fish_results[camera_id] = midline

            if fish_results:
                if fish_id not in results:
                    results[fish_id] = {}
                results[fish_id].update(fish_results)

        return results
