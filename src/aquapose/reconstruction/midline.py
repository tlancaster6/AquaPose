"""2D medial axis extraction and arc-length sampling pipeline.

Extracts ordered 15-point 2D midlines with half-widths from binary fish masks.
Produces Midline2D structs in full-frame pixel coordinates with consistent
head-to-tail orientation via 3D velocity cues and back-correction buffering.
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
    is_head_to_tail: bool = True


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_skip_mask(mask: np.ndarray, min_area: int = 300) -> str | None:
    """Return a skip reason if the mask should be discarded, else None.

    Skips masks that are too small or have foreground pixels touching any
    edge of the crop (indicating the fish is partially outside the crop).

    Args:
        mask: Binary mask, uint8, values 0 or 255 (or 0/1). Shape (H, W).
        min_area: Minimum number of nonzero pixels required.

    Returns:
        Skip reason string, or None if the mask is valid.
    """
    nonzero = np.count_nonzero(mask)
    if nonzero < min_area:
        return f"too small: {nonzero} < {min_area}"

    # Check if any foreground pixel touches any edge
    if (
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    ):
        return "boundary-clipped: mask touches crop edge"

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


def _orient_midline(
    xy_frame: np.ndarray,
    track: FishTrack,
    camera_id: str,
    projection_models: dict,
    fps: float,
    body_length_m: float,
) -> tuple[np.ndarray, bool]:
    """Orient midline endpoints so point[0] is closest to the head.

    Projects a predicted 3D head position into this camera and compares
    distance to both endpoints. If the fish is moving too slowly to
    determine orientation, returns is_established=False so the caller
    can apply inheritance.

    Args:
        xy_frame: Full-frame midline points, shape (N, 2), float32.
        track: FishTrack with positions and velocity.
        camera_id: Camera identifier (key into projection_models).
        projection_models: Dict mapping camera_id to projection model
            with a project(points) method.
        fps: Frames per second (for speed threshold conversion).
        body_length_m: Fish body length in metres (for speed threshold).

    Returns:
        oriented_xy: Midline oriented head-to-tail, shape (N, 2), float32.
        is_established: True if orientation was determined from velocity;
            False if speed was below threshold (inheritance required).
    """
    velocity_threshold_bls = 0.5  # body-lengths per second
    speed_threshold_mps = velocity_threshold_bls * body_length_m  # m/s
    speed_threshold_mpf = speed_threshold_mps / fps  # m/frame

    # Compute current speed
    vel = track.velocity  # shape (3,)
    speed = float(np.linalg.norm(vel))

    if speed < speed_threshold_mpf:
        return xy_frame, False

    # Predicted head position: centroid + velocity_unit * half_body
    if len(track.positions) == 0:
        return xy_frame, False

    last_pos = list(track.positions)[-1]  # shape (3,)
    vel_unit = vel / (speed + 1e-12)
    half_body = body_length_m / 2.0
    head_3d = last_pos + vel_unit * half_body  # shape (3,)

    # Project head position into this camera
    model = projection_models.get(camera_id)
    if model is None:
        return xy_frame, False

    try:
        import torch

        head_tensor = torch.tensor(head_3d, dtype=torch.float32).unsqueeze(0)  # (1, 3)
        pixels, valid = model.project(head_tensor)
        if not valid[0]:
            return xy_frame, False
        head_2d = pixels[0].numpy()  # (2,) = (u, v)
    except Exception:
        return xy_frame, False

    # Compare distances from head_2d to both endpoints
    p0 = xy_frame[0]  # (2,)
    p1 = xy_frame[-1]  # (2,)
    dist0 = float(np.linalg.norm(p0 - head_2d))
    dist1 = float(np.linalg.norm(p1 - head_2d))

    if dist1 < dist0:
        # Endpoint[-1] is closer to head — flip
        return xy_frame[::-1].copy(), True

    return xy_frame, True


# ---------------------------------------------------------------------------
# Stateful extractor
# ---------------------------------------------------------------------------


class MidlineExtractor:
    """Extracts ordered 2D midlines from per-camera binary masks.

    Handles the full pipeline: mask validation, morphological smoothing,
    skeletonization, longest-path pruning, arc-length resampling,
    coordinate transform, and head-to-tail orientation with inheritance
    and back-correction buffering.

    Args:
        n_points: Number of midline points to produce. Default 15.
        min_area: Minimum mask area (pixels) to attempt extraction. Default 300.
        fps: Video frame rate, used to convert speed thresholds. Default 30.0.
        body_length_m: Typical fish body length in metres. Default 0.15.
        velocity_threshold_bls: Speed threshold in body-lengths/second below
            which orientation is considered ambiguous. Default 0.5.
    """

    def __init__(
        self,
        n_points: int = 15,
        min_area: int = 300,
        fps: float = 30.0,
        body_length_m: float = 0.15,
        velocity_threshold_bls: float = 0.5,
    ) -> None:
        self.n_points = n_points
        self.min_area = min_area
        self.fps = fps
        self.body_length_m = body_length_m
        self.velocity_threshold_bls = velocity_threshold_bls

        # fish_id → last known head-first boolean
        self._orientations: dict[int, bool] = {}

        # fish_id → list of (frame_idx, camera_id, Midline2D) awaiting orientation
        self._back_correction_buffers: dict[int, list[tuple[int, str, Midline2D]]] = {}

        # fish_id → frames since track start (for back-correction cap)
        self._back_correction_frame_counts: dict[int, int] = {}

    @property
    def _back_correction_cap(self) -> int:
        """Back-correction buffer cap: min(30, fps) frames."""
        return min(30, int(self.fps))

    def extract_midlines(
        self,
        tracks: list[FishTrack],
        masks_per_camera: dict[str, list[np.ndarray]],
        crop_regions_per_camera: dict[str, list[CropRegion]],
        detections_per_camera: dict[str, list],
        projection_models: dict,
        frame_index: int,
    ) -> dict[int, dict[str, Midline2D]]:
        """Extract 2D midlines for all tracks in a single frame.

        For each confirmed track, attempts midline extraction in every camera
        where it has a detection. Applies orientation inheritance and
        back-correction buffering.

        Args:
            tracks: List of FishTrack objects from the current frame.
            masks_per_camera: Per-camera list of binary masks (uint8).
            crop_regions_per_camera: Per-camera list of CropRegion objects.
            detections_per_camera: Per-camera detection lists (used for
                index lookup; content not directly inspected here).
            projection_models: Dict mapping camera_id to projection model.
            frame_index: Current frame index.

        Returns:
            Nested dict: fish_id → camera_id → Midline2D.
            Only includes entries where extraction succeeded.
        """
        results: dict[int, dict[str, Midline2D]] = {}

        for track in tracks:
            fish_id = track.fish_id

            # Initialise back-correction tracking for new fish
            if fish_id not in self._back_correction_frame_counts:
                self._back_correction_frame_counts[fish_id] = 0
                self._back_correction_buffers[fish_id] = []

            self._back_correction_frame_counts[fish_id] += 1
            frame_count = self._back_correction_frame_counts[fish_id]

            # Check if back-correction window has expired; commit buffer as-is
            cap = self._back_correction_cap
            if frame_count > cap and fish_id in self._back_correction_buffers:
                buf = self._back_correction_buffers.get(fish_id, [])
                if buf:
                    # Commit buffer midlines as-is (no orientation flip)
                    for _fi, _cam, _ml in buf:
                        if fish_id not in results:
                            results[fish_id] = {}
                        results[fish_id][_cam] = _ml
                    self._back_correction_buffers[fish_id] = []

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
                skip_reason = _check_skip_mask(mask, self.min_area)
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

                # 8. Orientation
                oriented_xy, is_established = _orient_midline(
                    xy_frame,
                    track,
                    camera_id,
                    projection_models,
                    self.fps,
                    self.body_length_m,
                )

                # Handle orientation logic
                if is_established:
                    # Determine if flip was applied by comparing endpoints
                    was_flipped = not np.allclose(oriented_xy[0], xy_frame[0])

                    midline = Midline2D(
                        points=oriented_xy,
                        half_widths=hw_frame,
                        fish_id=fish_id,
                        camera_id=camera_id,
                        frame_index=frame_index,
                        is_head_to_tail=True,
                    )

                    # If first orientation established, back-correct buffer
                    if fish_id not in self._orientations:
                        buf = self._back_correction_buffers.get(fish_id, [])
                        if buf:
                            # Buffer was built with arbitrary (un-flipped) orientation.
                            # If current result was flipped, buffer needs flipping too.
                            if was_flipped:
                                for _fi, _cam, _ml in buf:
                                    _ml.points = _ml.points[::-1].copy()
                                    _ml.is_head_to_tail = True
                            else:
                                for _fi, _cam, _ml in buf:
                                    _ml.is_head_to_tail = True
                            # Commit buffer
                            if fish_id not in results:
                                results[fish_id] = {}
                            for _fi, _cam, _ml in buf:
                                results[fish_id][_cam] = _ml
                            self._back_correction_buffers[fish_id] = []

                    self._orientations[fish_id] = True  # head-first = point[0] is head
                    fish_results[camera_id] = midline

                else:
                    # Orientation not established this frame
                    if fish_id in self._orientations:
                        # Inherit: keep current order, mark as unestablished
                        midline = Midline2D(
                            points=xy_frame,
                            half_widths=hw_frame,
                            fish_id=fish_id,
                            camera_id=camera_id,
                            frame_index=frame_index,
                            is_head_to_tail=False,
                        )
                        fish_results[camera_id] = midline
                    else:
                        # No previous orientation; store in back-correction buffer
                        midline = Midline2D(
                            points=xy_frame,
                            half_widths=hw_frame,
                            fish_id=fish_id,
                            camera_id=camera_id,
                            frame_index=frame_index,
                            is_head_to_tail=False,
                        )
                        buf = self._back_correction_buffers.setdefault(fish_id, [])
                        if frame_count <= cap:
                            buf.append((frame_index, camera_id, midline))
                        # Do NOT add to fish_results yet — committed on
                        # orientation establishment or cap expiry

            if fish_results:
                if fish_id not in results:
                    results[fish_id] = {}
                results[fish_id].update(fish_results)

        return results
