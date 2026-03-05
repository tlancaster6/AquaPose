"""Geometry utilities for OBB computation and keypoint annotation formatting."""

from __future__ import annotations

import cv2
import numpy as np


def pca_obb(
    coords: np.ndarray,
    visible: np.ndarray,
    lateral_pad: float,
) -> np.ndarray:
    """Compute a PCA-derived oriented bounding box for visible keypoints.

    Uses PCA on the visible keypoint coordinates to find the long axis of the
    fish. Projects all visible points onto the PCA axes to determine extent.
    The OBB half-width along the main axis is the max projection plus a small
    epsilon; the half-width perpendicular is ``lateral_pad``.

    Args:
        coords: float array of shape ``(N, 2)`` with (x, y) pixel coordinates.
        visible: bool array of shape ``(N,)``, True if the keypoint is visible.
        lateral_pad: Half-width of OBB in the lateral (perpendicular) direction,
            in pixels.

    Returns:
        OBB corners as float64 array of shape ``(4, 2)``, in clockwise order
        starting from the top-left corner in OBB frame (TL, TR, BR, BL in
        rotated space).
    """
    vis_pts = coords[visible]

    # Degenerate case: 0 or 1 visible keypoint — return a default 20x20 box
    if len(vis_pts) <= 1:
        if len(vis_pts) == 1:
            cx, cy = float(vis_pts[0, 0]), float(vis_pts[0, 1])
        else:
            cx, cy = 0.0, 0.0
        half = 10.0
        return np.array(
            [
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ],
            dtype=np.float64,
        )

    centroid = vis_pts.mean(axis=0)
    centered = vis_pts - centroid

    if len(vis_pts) >= 2:
        # PCA via SVD
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        main_axis = vh[0]  # first principal component (long axis)
    else:
        main_axis = np.array([1.0, 0.0])

    # Ensure main_axis is a unit vector
    norm = np.linalg.norm(main_axis)
    main_axis = np.array([1.0, 0.0]) if norm < 1e-9 else main_axis / norm

    perp_axis = np.array([-main_axis[1], main_axis[0]])

    # Project visible points onto the main axis to determine extent
    proj_main = centered @ main_axis

    eps = 2.0  # small buffer along main axis
    half_main = float(max(abs(proj_main.max()), abs(proj_main.min()))) + eps
    half_perp = lateral_pad

    # Build corners in OBB frame, then rotate back to image space
    # Order: TL, TR, BR, BL in OBB frame (main axis = horizontal)
    corners_local = np.array(
        [
            [-half_main, -half_perp],
            [half_main, -half_perp],
            [half_main, half_perp],
            [-half_main, half_perp],
        ],
        dtype=np.float64,
    )

    # Rotation matrix: columns are main_axis and perp_axis
    rot = np.stack([main_axis, perp_axis], axis=1)  # (2, 2)
    corners_world = corners_local @ rot.T + centroid  # (4, 2)

    return corners_world


def affine_warp_crop(
    image: np.ndarray,
    obb_corners: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp OBB region to an axis-aligned rectangle.

    Maps 3 of the 4 OBB corners to corresponding destination corners of a
    ``(crop_w, crop_h)`` rectangle using ``cv2.getAffineTransform``.

    The OBB is assumed to have corners in order: TL, TR, BR, BL. The mapping
    ensures the fish is in landscape orientation (long axis = horizontal).

    Args:
        image: BGR image array of shape ``(H, W, 3)``.
        obb_corners: float64 array of shape ``(4, 2)`` with OBB corner
            coordinates in image space (TL, TR, BR, BL).
        crop_w: Output crop width in pixels.
        crop_h: Output crop height in pixels.

    Returns:
        Tuple of:
        - warped: uint8 BGR crop of shape ``(crop_h, crop_w, 3)``.
        - affine: float64 affine matrix of shape ``(2, 3)``.
    """
    # Source points: TL, TR, BL (3 corners for affine) -- must be float32 for cv2
    src = np.array([obb_corners[0], obb_corners[1], obb_corners[3]], dtype=np.float32)
    # Destination: top-left, top-right, bottom-left of crop
    dst = np.array([[0, 0], [crop_w - 1, 0], [0, crop_h - 1]], dtype=np.float32)

    affine = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(image, affine, (crop_w, crop_h))
    return warped, affine.astype(np.float64)


def transform_keypoints(
    coords: np.ndarray,
    visible: np.ndarray,
    affine_matrix: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 2x3 affine matrix to keypoint coordinates.

    Out-of-bounds keypoints (after transform) are marked invisible.

    Args:
        coords: float array of shape ``(N, 2)`` with (x, y) pixel coordinates.
        visible: bool array of shape ``(N,)``, True if keypoint is visible.
        affine_matrix: float64 affine transform matrix of shape ``(2, 3)``.
        crop_w: Output crop width in pixels.
        crop_h: Output crop height in pixels.

    Returns:
        Tuple of:
        - coords_out: float64 array of shape ``(N, 2)`` with transformed
          coordinates, clamped to ``[0, crop_w) x [0, crop_h)``.
        - visible_out: bool array of shape ``(N,)``, marking OOB points
          invisible.
    """
    n = len(coords)
    # Homogeneous coordinates
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([coords, ones])  # (N, 3)

    transformed = (affine_matrix @ pts_h.T).T  # (N, 2)

    # Check out-of-bounds
    oob = (
        (transformed[:, 0] < 0)
        | (transformed[:, 0] >= crop_w)
        | (transformed[:, 1] < 0)
        | (transformed[:, 1] >= crop_h)
    )

    coords_out = np.clip(transformed, [0, 0], [crop_w - 1, crop_h - 1])
    visible_out = visible & ~oob

    return coords_out, visible_out


def clip_obb_to_image(obb_corners: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Clip OBB corner coordinates to image bounds.

    Args:
        obb_corners: float64 array of shape ``(4, 2)`` with corner (x, y).
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Clipped copy of obb_corners with coordinates in
        ``[0, img_w-1] x [0, img_h-1]``.
    """
    clipped = obb_corners.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, img_w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, img_h - 1)
    return clipped


def extrapolate_edge_keypoints(
    coords: np.ndarray,
    visible: np.ndarray,
    img_w: int,
    img_h: int,
    lateral_pad: float,
    edge_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the polyline toward image edges for near-boundary keypoints.

    Checks whether the first or last visible keypoint in the chain is within
    ``lateral_pad * edge_factor`` pixels of any image boundary. If so,
    extrapolates toward the nearest edge along the local chain direction.

    Args:
        coords: float array of shape ``(N, 2)`` with (x, y) pixel coordinates.
        visible: bool array of shape ``(N,)``, True if the keypoint is visible.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        lateral_pad: Base lateral padding (half-width of OBB) in pixels.
        edge_factor: Multiplier applied to ``lateral_pad`` to set proximity
            threshold. Default is 2.0.

    Returns:
        Tuple of:
        - coords_out: Updated copy of coords with extrapolated keypoints.
        - visible_out: Updated copy of visible with extrapolated points set
            True.
    """
    coords_out = coords.copy()
    visible_out = visible.copy()

    threshold = lateral_pad * edge_factor
    vis_indices = [i for i in range(len(visible)) if visible[i]]

    if len(vis_indices) < 2:
        return coords_out, visible_out

    def _nearest_edge_dist(pt: np.ndarray) -> float:
        """Return distance to nearest image boundary."""
        x, y = pt
        return float(min(x, y, img_w - x, img_h - y))

    def _extrapolate_toward_edge(
        anchor: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Snap anchor perpendicularly to the nearest image boundary."""
        x, y = anchor
        dists = [x, img_w - 1 - x, y, img_h - 1 - y]
        idx = int(np.argmin(dists))
        new_pt = anchor.copy()
        if idx == 0:  # left
            new_pt[0] = 0.0
        elif idx == 1:  # right
            new_pt[0] = float(img_w - 1)
        elif idx == 2:  # top
            new_pt[1] = 0.0
        else:  # bottom
            new_pt[1] = float(img_h - 1)
        return new_pt

    # Check first visible keypoint (nose end)
    first_idx = vis_indices[0]
    first_pt = coords_out[first_idx]
    if _nearest_edge_dist(first_pt) < threshold:
        second_idx = vis_indices[1]
        direction = coords_out[first_idx] - coords_out[second_idx]
        new_pt = _extrapolate_toward_edge(first_pt.copy(), direction)
        coords_out[first_idx] = new_pt

    # Check last visible keypoint (tail end)
    last_idx = vis_indices[-1]
    last_pt = coords_out[last_idx]
    if _nearest_edge_dist(last_pt) < threshold:
        second_last_idx = vis_indices[-2]
        direction = coords_out[last_idx] - coords_out[second_last_idx]
        new_pt = _extrapolate_toward_edge(last_pt.copy(), direction)
        coords_out[last_idx] = new_pt

    return coords_out, visible_out


def format_obb_annotation(
    obb_corners: np.ndarray,
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> list[float]:
    """Format one OBB annotation as a flat array for a YOLO-OBB .txt label line.

    Returns a flat list ``[cls, x1, y1, x2, y2, x3, y3, x4, y4]`` with
    corners normalized to [0, 1] by image dimensions.

    Args:
        obb_corners: float64 array of shape ``(4, 2)`` with corner (x, y) in
            image pixel space, order: TL, TR, BR, BL.
        img_w: Image width in pixels (for normalization).
        img_h: Image height in pixels (for normalization).
        class_id: YOLO class index.

    Returns:
        Flat list ``[cls, x1, y1, x2, y2, x3, y3, x4, y4]`` with normalized
        coords.
    """
    row: list[float] = [float(class_id)]
    for corner in obb_corners:
        x_norm = float(np.clip(corner[0] / img_w, 0.0, 1.0))
        y_norm = float(np.clip(corner[1] / img_h, 0.0, 1.0))
        row.append(round(x_norm, 6))
        row.append(round(y_norm, 6))
    return row


def format_pose_annotation(
    cx: float,
    cy: float,
    w: float,
    h: float,
    keypoints: np.ndarray,
    visible: np.ndarray,
    crop_w: int,
    crop_h: int,
    class_id: int = 0,
) -> list[float]:
    """Format one pose annotation as a flat array for a YOLO-Pose .txt label line.

    Returns a flat list ``[cls, cx, cy, w, h, x1, y1, v1, x2, y2, v2, ...]``
    with bbox and keypoints normalized to [0, 1]. Invisible keypoints are
    output as ``0, 0, 0``. Visible keypoints use COCO visible=2 convention.

    Args:
        cx: Bounding box center x, normalized to [0, 1].
        cy: Bounding box center y, normalized to [0, 1].
        w: Bounding box width, normalized to [0, 1].
        h: Bounding box height, normalized to [0, 1].
        keypoints: float array of shape ``(N, 2)`` with (x, y) in crop pixels.
        visible: bool array of shape ``(N,)``, True if keypoint is visible.
        crop_w: Crop width in pixels (for normalization).
        crop_h: Crop height in pixels (for normalization).
        class_id: YOLO class index.

    Returns:
        Flat list ``[cls, cx, cy, w, h, x1, y1, v1, ...]`` with normalized
        coords.
    """
    row: list[float] = [
        float(class_id),
        round(cx, 6),
        round(cy, 6),
        round(w, 6),
        round(h, 6),
    ]
    for k in range(len(keypoints)):
        if visible[k]:
            x_norm = float(np.clip(keypoints[k, 0] / crop_w, 0.0, 1.0))
            y_norm = float(np.clip(keypoints[k, 1] / crop_h, 0.0, 1.0))
            row.extend([round(x_norm, 6), round(y_norm, 6), 2])
        else:
            row.extend([0, 0, 0])
    return row
