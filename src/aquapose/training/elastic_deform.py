"""Elastic midline deformation for pose training data augmentation."""

from __future__ import annotations

import numpy as np

from .geometry import format_obb_annotation, format_pose_annotation, pca_obb


def deform_keypoints_c_curve(
    coords: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Displace keypoints along a uniform circular arc.

    Applies a C-curve deformation by bending the keypoint polyline into an arc
    with the specified total bending angle. The centroid of the deformed points
    matches the centroid of the original.

    Args:
        coords: Keypoint coordinates of shape ``(N, 2)`` in pixel space.
        angle_deg: Signed total bending angle in degrees. Positive bends one
            way, negative the other. Zero is identity.

    Returns:
        Deformed keypoint coordinates of shape ``(N, 2)``.
    """
    if abs(angle_deg) < 1e-12:
        return coords.copy()

    n = len(coords)
    # Compute chord direction and length
    chord_vec = coords[-1] - coords[0]
    chord_length = float(np.linalg.norm(chord_vec))
    if chord_length < 1e-12:
        return coords.copy()

    # Unit vectors along and perpendicular to chord
    tangent = chord_vec / chord_length
    normal = np.array([-tangent[1], tangent[0]])

    # Arc-length parameterization along the polyline
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_arc = cum_lengths[-1]
    if total_arc < 1e-12:
        return coords.copy()
    t = cum_lengths / total_arc  # [0, 1]

    total_angle_rad = np.radians(angle_deg)
    half_angle = total_angle_rad / 2.0

    # Radius of curvature
    r = chord_length / (2.0 * np.sin(half_angle))

    # Displacement along arc for each keypoint
    # Map t to angle: theta(t) = -half_angle + t * total_angle
    theta = -half_angle + t * total_angle_rad

    # Arc position relative to center of arc
    # x_arc = R * sin(theta), y_arc = R * (1 - cos(theta))
    # Subtract the linear interpolation to get displacement only
    x_along = r * np.sin(theta)
    y_perp = r * (1.0 - np.cos(theta))

    # Linear interpolation from start to end
    x_linear = r * np.sin(-half_angle) + t * (
        r * np.sin(half_angle) - r * np.sin(-half_angle)
    )
    y_linear = r * (1.0 - np.cos(-half_angle)) + t * (
        r * (1.0 - np.cos(half_angle)) - r * (1.0 - np.cos(-half_angle))
    )

    dx_along = x_along - x_linear
    dy_perp = y_perp - y_linear

    # Build deformed coordinates
    deformed = coords.copy()
    for i in range(n):
        deformed[i] = coords[i] + dx_along[i] * tangent + dy_perp[i] * normal

    # Re-center to preserve centroid
    centroid_shift = deformed.mean(axis=0) - coords.mean(axis=0)
    deformed -= centroid_shift

    return deformed


def deform_keypoints_s_curve(
    coords: np.ndarray,
    amplitude_deg: float,
) -> np.ndarray:
    """Displace keypoints along a sinusoidal (S-curve) path.

    Applies a sinusoidal lateral displacement proportional to
    ``sin(pi * t)`` where ``t`` is the arc-length fraction along the polyline.

    Args:
        coords: Keypoint coordinates of shape ``(N, 2)`` in pixel space.
        amplitude_deg: Signed amplitude in degrees. The pixel amplitude is
            derived as ``chord_length * tan(amplitude_deg * pi / 180)``.

    Returns:
        Deformed keypoint coordinates of shape ``(N, 2)``.
    """
    if abs(amplitude_deg) < 1e-12:
        return coords.copy()

    # Chord direction and length
    chord_vec = coords[-1] - coords[0]
    chord_length = float(np.linalg.norm(chord_vec))
    if chord_length < 1e-12:
        return coords.copy()

    tangent = chord_vec / chord_length
    normal = np.array([-tangent[1], tangent[0]])

    # Arc-length parameterization
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_arc = cum_lengths[-1]
    if total_arc < 1e-12:
        return coords.copy()
    t = cum_lengths / total_arc

    # Pixel amplitude from angular amplitude
    a_px = chord_length * np.tan(np.radians(amplitude_deg))

    # Sinusoidal displacement: sin(pi * t) has zero at endpoints, max at midpoint
    displacement = a_px * np.sin(np.pi * t)

    deformed = coords.copy()
    for i in range(len(coords)):
        deformed[i] = coords[i] + displacement[i] * normal

    # Re-center to preserve centroid
    centroid_shift = deformed.mean(axis=0) - coords.mean(axis=0)
    deformed -= centroid_shift

    return deformed


def tps_warp_image(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> np.ndarray:
    """Warp an image using thin-plate spline interpolation.

    Uses 6 keypoint control points plus 4 corner identity anchors (10 total)
    to define the warp. Corner anchors pin the image borders in place.

    Args:
        image: Input BGR image of shape ``(crop_h, crop_w, 3)``.
        src_points: Source control points of shape ``(N, 2)`` in pixel coords.
        dst_points: Destination control points of shape ``(N, 2)``.
        crop_w: Output width in pixels.
        crop_h: Output height in pixels.

    Returns:
        Warped image of shape ``(crop_h, crop_w, 3)``.
    """
    import cv2
    from scipy.interpolate import RBFInterpolator

    # Add corner anchors (identity mapping for border stability)
    corners = np.array(
        [[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]],
        dtype=np.float64,
    )
    all_src = np.vstack([src_points, corners])
    all_dst = np.vstack([dst_points, corners])

    # Build backward mapping: for each output pixel, find where it comes from
    # in the input. We fit RBF from dst -> src (backward warp).
    interp_x = RBFInterpolator(all_dst, all_src[:, 0], kernel="thin_plate_spline")
    interp_y = RBFInterpolator(all_dst, all_src[:, 1], kernel="thin_plate_spline")

    # Create grid of output pixel coordinates
    grid_y, grid_x = np.mgrid[0:crop_h, 0:crop_w]
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Map output pixels back to input coordinates
    map_x = interp_x(grid_pts).reshape(crop_h, crop_w).astype(np.float32)
    map_y = interp_y(grid_pts).reshape(crop_h, crop_w).astype(np.float32)

    # Remap image using backward mapping
    result = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return result


def generate_deformed_labels(
    deformed_coords: np.ndarray,
    visible: np.ndarray,
    crop_w: int,
    crop_h: int,
    lateral_pad: float,
) -> dict:
    """Generate YOLO label lines from deformed keypoint coordinates.

    Computes a new OBB via ``pca_obb`` on the deformed keypoints, then formats
    both OBB and pose annotation lines.

    Args:
        deformed_coords: Deformed keypoints of shape ``(N, 2)`` in crop pixels.
        visible: Visibility array of shape ``(N,)`` (bool).
        crop_w: Crop image width in pixels.
        crop_h: Crop image height in pixels.
        lateral_pad: OBB lateral padding in pixels.

    Returns:
        Dict with keys ``obb_line`` (list[float]), ``pose_line`` (list[float]),
        and ``obb_corners`` (ndarray of shape ``(4, 2)``).
    """
    obb_corners = pca_obb(deformed_coords, visible, lateral_pad)
    obb_line = format_obb_annotation(obb_corners, crop_w, crop_h)

    # Compute normalized bbox from OBB for pose label
    vis_pts = deformed_coords[visible]
    if len(vis_pts) > 0:
        x_min, y_min = vis_pts.min(axis=0)
        x_max, y_max = vis_pts.max(axis=0)
        # Add lateral_pad to bbox
        x_min = max(0, x_min - lateral_pad)
        y_min = max(0, y_min - lateral_pad)
        x_max = min(crop_w, x_max + lateral_pad)
        y_max = min(crop_h, y_max + lateral_pad)
        cx = ((x_min + x_max) / 2.0) / crop_w
        cy = ((y_min + y_max) / 2.0) / crop_h
        w = (x_max - x_min) / crop_w
        h = (y_max - y_min) / crop_h
    else:
        cx, cy, w, h = 0.5, 0.5, 0.1, 0.1

    pose_line = format_pose_annotation(
        cx, cy, w, h, deformed_coords, visible, crop_w, crop_h
    )

    return {
        "obb_line": obb_line,
        "pose_line": pose_line,
        "obb_corners": obb_corners,
    }


def generate_variants(
    image: np.ndarray,
    coords: np.ndarray,
    visible: np.ndarray,
    crop_w: int,
    crop_h: int,
    lateral_pad: float,
    angle_range: tuple[float, float] = (10.0, 30.0),
) -> list[dict]:
    """Generate 4 deformed variants of a fish image and keypoints.

    Produces 2 C-curve variants (positive and negative angle) and 2 S-curve
    variants (positive and negative amplitude). The deformation angle is the
    midpoint of ``angle_range``.

    Args:
        image: Input BGR crop of shape ``(crop_h, crop_w, 3)``.
        coords: Keypoints of shape ``(N, 2)`` in crop pixel coordinates.
        visible: Visibility array of shape ``(N,)`` (bool).
        crop_w: Crop width in pixels.
        crop_h: Crop height in pixels.
        lateral_pad: OBB lateral padding in pixels.
        angle_range: ``(min_angle, max_angle)`` in degrees for deformation
            magnitude. The midpoint is used.

    Returns:
        List of 4 variant dicts, each with keys: ``image``, ``coords``,
        ``visible``, ``obb_line``, ``pose_line``, ``variant_tag``.
    """
    angle = (angle_range[0] + angle_range[1]) / 2.0

    deform_specs = [
        ("c_pos", deform_keypoints_c_curve, angle),
        ("c_neg", deform_keypoints_c_curve, -angle),
        ("s_pos", deform_keypoints_s_curve, angle),
        ("s_neg", deform_keypoints_s_curve, -angle),
    ]

    variants: list[dict] = []
    for tag, deform_fn, deform_angle in deform_specs:
        deformed = deform_fn(coords, deform_angle)
        warped = tps_warp_image(image, coords, deformed, crop_w, crop_h)
        labels = generate_deformed_labels(
            deformed, visible, crop_w, crop_h, lateral_pad
        )
        variants.append(
            {
                "image": warped,
                "coords": deformed,
                "visible": visible.copy(),
                "obb_line": labels["obb_line"],
                "pose_line": labels["pose_line"],
                "variant_tag": tag,
            }
        )

    return variants
