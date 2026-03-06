"""Elastic midline deformation for pose training data augmentation."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .geometry import format_obb_annotation, format_pose_annotation, pca_obb


def deform_keypoints_c_curve(
    coords: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Displace keypoints along a uniform circular arc (C-shape bend).

    Applies lateral displacement following a parabolic profile (zero at
    endpoints, maximum at midpoint) scaled by
    ``chord_length * tan(angle_deg)``. This produces a single-direction
    bend matching the amplitude convention used by ``deform_keypoints_s_curve``.

    Args:
        coords: Keypoint coordinates of shape ``(N, 2)`` in pixel space.
        angle_deg: Signed bending angle in degrees. Positive bends one
            way, negative the other. Zero is identity.

    Returns:
        Deformed keypoint coordinates of shape ``(N, 2)``.
    """
    if abs(angle_deg) < 1e-12:
        return coords.copy()

    # Compute chord direction and length
    chord_vec = coords[-1] - coords[0]
    chord_length = float(np.linalg.norm(chord_vec))
    if chord_length < 1e-12:
        return coords.copy()

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

    # Pixel amplitude from angular parameter (same convention as S-curve)
    a_px = chord_length * np.tan(np.radians(angle_deg))

    # Parabolic C-curve profile: 4*t*(1-t) is zero at endpoints, 1 at midpoint
    displacement = a_px * 4.0 * t * (1.0 - t)

    deformed = coords.copy()
    for i in range(len(coords)):
        deformed[i] = coords[i] + displacement[i] * normal

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

    # Full-period sine: sin(2*pi*t) gives a true S-curve — zero at endpoints
    # and midpoint, positive hump in the first half, negative in the second.
    displacement = a_px * np.sin(2.0 * np.pi * t)

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

    # Build flanking control points on both sides of each keypoint along the
    # local midline normal.  This ensures the TPS moves the full fish cross-
    # section uniformly instead of tapering displacement off-midline → blur.
    flank_dist = 25.0  # half fish body width, pixels
    n_kp = len(src_points)
    src_list = [src_points]
    dst_list = [dst_points]
    for i in range(n_kp):
        # Local normal: perpendicular to segment connecting neighbours
        if n_kp == 1:
            normal = np.array([0.0, 1.0])
        else:
            if i == 0:
                seg = src_points[1] - src_points[0]
            elif i == n_kp - 1:
                seg = src_points[-1] - src_points[-2]
            else:
                seg = src_points[i + 1] - src_points[i - 1]
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-12:
                normal = np.array([0.0, 1.0])
            else:
                normal = np.array([-seg[1], seg[0]]) / seg_len

        for sign in (+1.0, -1.0):
            offset = sign * flank_dist * normal
            src_list.append((src_points[i] + offset).reshape(1, 2))
            dst_list.append((dst_points[i] + offset).reshape(1, 2))

    # Add corner anchors (identity mapping for border stability)
    corners = np.array(
        [[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]],
        dtype=np.float64,
    )
    src_list.append(corners)
    dst_list.append(corners)

    all_src = np.vstack(src_list)
    all_dst = np.vstack(dst_list)

    # Build backward mapping: for each output pixel, find where it comes from
    # in the input. We fit RBF from dst -> src (backward warp).
    interp_x = RBFInterpolator(
        all_dst, all_src[:, 0], kernel="thin_plate_spline", smoothing=1e-3
    )
    interp_y = RBFInterpolator(
        all_dst, all_src[:, 1], kernel="thin_plate_spline", smoothing=1e-3
    )

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
    vis_coords = coords[visible]
    for tag, deform_fn, deform_angle in deform_specs:
        # Deform only visible keypoints so chord length (and thus amplitude)
        # isn't polluted by invisible (0,0) coordinates.
        deformed_vis = deform_fn(vis_coords, deform_angle)
        # Reconstruct full coord array with invisible points unchanged.
        deformed = coords.copy()
        deformed[visible] = deformed_vis
        warped = tps_warp_image(image, vis_coords, deformed_vis, crop_w, crop_h)
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


def parse_pose_label(
    label_path: Path,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a YOLO pose label file and extract keypoint coordinates.

    Reads the first line of the label file in YOLO pose format:
    ``cls cx cy w h x1 y1 v1 x2 y2 v2 ...`` with 6 keypoints.

    Args:
        label_path: Path to the ``.txt`` label file.
        crop_w: Image width in pixels (for denormalization).
        crop_h: Image height in pixels (for denormalization).

    Returns:
        Tuple of:
        - coords: float64 array of shape ``(6, 2)`` with (x, y) in pixels.
        - visible: bool array of shape ``(6,)``, True if visibility flag >= 1.
    """
    text = label_path.read_text().strip()
    vals = [float(v) for v in text.split()]
    # Skip cls, cx, cy, w, h (5 values), then read 6 keypoint triplets
    kp_vals = vals[5:]
    n_kpts = len(kp_vals) // 3
    coords = np.zeros((n_kpts, 2), dtype=np.float64)
    visible = np.zeros(n_kpts, dtype=bool)
    for i in range(n_kpts):
        x_norm = kp_vals[i * 3]
        y_norm = kp_vals[i * 3 + 1]
        v = kp_vals[i * 3 + 2]
        coords[i] = [x_norm * crop_w, y_norm * crop_h]
        visible[i] = v >= 1
    return coords, visible


def write_yolo_dataset(
    input_dir: Path,
    output_dir: Path,
    lateral_pad: float,
    angle_range: tuple[float, float] = (10.0, 30.0),
) -> None:
    """Generate a YOLO-format dataset with elastic deformation variants.

    Reads images and pose labels from ``input_dir`` (YOLO format), generates
    4 deformed variants per image, and writes originals + variants to
    ``output_dir`` in YOLO format.

    Args:
        input_dir: Source directory with ``images/train/`` and ``labels/train/``.
        output_dir: Destination directory for augmented dataset.
        lateral_pad: OBB lateral padding in pixels.
        angle_range: ``(min_angle, max_angle)`` in degrees.
    """
    import cv2
    import yaml

    img_in = input_dir / "images" / "train"
    lbl_in = input_dir / "labels" / "train"

    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(img_in.glob("*.jpg")):
        stem = img_path.stem
        lbl_path = lbl_in / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        crop_h, crop_w = image.shape[:2]

        # Copy original
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(lbl_path, lbl_out / lbl_path.name)

        # Parse keypoints from label
        coords, visible = parse_pose_label(lbl_path, crop_w, crop_h)

        # Generate variants
        variants = generate_variants(
            image, coords, visible, crop_w, crop_h, lateral_pad, angle_range
        )

        for variant in variants:
            tag = variant["variant_tag"]
            # Write variant image
            var_img_path = img_out / f"{stem}_{tag}.jpg"
            cv2.imwrite(str(var_img_path), variant["image"])

            # Write variant pose label
            var_lbl_path = lbl_out / f"{stem}_{tag}.txt"
            line = " ".join(str(round(v, 6)) for v in variant["pose_line"])
            var_lbl_path.write_text(line + "\n")

    # Write dataset.yaml
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "names": {0: "fish"},
        "nc": 1,
        "kpt_shape": [6, 3],
    }
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)


def generate_preview_grid(
    input_dir: Path,
    output_path: Path,
    lateral_pad: float,
    angle_range: tuple[float, float] = (10.0, 30.0),
    max_samples: int = 5,
) -> None:
    """Generate a preview grid showing original + 4 deformed variants.

    Creates a grid image with rows = samples and columns = [original, C+, C-,
    S+, S-]. Each cell shows the image with keypoints overlaid as colored
    circles connected by a polyline.

    Args:
        input_dir: YOLO-format source directory with ``images/train/`` and
            ``labels/train/``.
        output_path: Output path for the preview grid PNG.
        lateral_pad: OBB lateral padding in pixels.
        angle_range: ``(min_angle, max_angle)`` in degrees.
        max_samples: Maximum number of sample images to include.
    """
    import cv2

    img_dir = input_dir / "images" / "train"
    lbl_dir = input_dir / "labels" / "train"

    img_paths = sorted(img_dir.glob("*.jpg"))[:max_samples]
    if not img_paths:
        return

    cell_w = 160
    col_labels = ["Original", "C+", "C-", "S+", "S-"]
    n_cols = len(col_labels)
    header_h = 25
    kp_colors = [
        (0, 0, 255),
        (0, 128, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 128, 0),
        (255, 0, 0),
    ]

    rows: list[list[np.ndarray]] = []

    for img_path in img_paths:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        crop_h, crop_w = image.shape[:2]
        coords, visible = parse_pose_label(lbl_path, crop_w, crop_h)

        # Generate variants
        variants = generate_variants(
            image, coords, visible, crop_w, crop_h, lateral_pad, angle_range
        )

        # Build row of cells: [original, c_pos, c_neg, s_pos, s_neg]
        row_cells: list[np.ndarray] = []

        # Original cell
        orig_cell = _draw_keypoints_on_image(image, coords, visible, kp_colors)
        row_cells.append(orig_cell)

        # Variant cells in order: c_pos, c_neg, s_pos, s_neg
        tag_order = ["c_pos", "c_neg", "s_pos", "s_neg"]
        tag_to_variant = {v["variant_tag"]: v for v in variants}
        for tag in tag_order:
            v = tag_to_variant[tag]
            cell = _draw_keypoints_on_image(
                v["image"], v["coords"], v["visible"], kp_colors
            )
            row_cells.append(cell)

        rows.append(row_cells)

    if not rows:
        return

    # Compute cell height from aspect ratio of first image
    first_img = cv2.imread(str(img_paths[0]))
    if first_img is None:
        return
    aspect = first_img.shape[0] / first_img.shape[1]
    cell_h = int(cell_w * aspect)

    # Build grid
    grid_w = n_cols * cell_w
    grid_h = header_h + len(rows) * cell_h
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # light gray bg

    # Draw column headers
    for i, label in enumerate(col_labels):
        x = i * cell_w + 10
        cv2.putText(grid, label, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Draw cells
    for r, row_cells in enumerate(rows):
        for c, cell in enumerate(row_cells):
            resized = cv2.resize(cell, (cell_w, cell_h))
            y0 = header_h + r * cell_h
            x0 = c * cell_w
            grid[y0 : y0 + cell_h, x0 : x0 + cell_w] = resized

    cv2.imwrite(str(output_path), grid)


def _draw_keypoints_on_image(
    image: np.ndarray,
    coords: np.ndarray,
    visible: np.ndarray,
    colors: list[tuple[int, int, int]],
) -> np.ndarray:
    """Draw keypoints and connecting polyline on an image copy.

    Args:
        image: BGR image array.
        coords: Keypoints of shape ``(N, 2)`` in pixel coordinates.
        visible: Visibility array of shape ``(N,)``.
        colors: List of BGR color tuples, one per keypoint.

    Returns:
        Copy of the image with keypoints drawn.
    """
    import cv2

    canvas = image.copy()
    vis_pts = coords[visible].astype(np.int32)

    # Draw polyline connecting visible keypoints
    if len(vis_pts) >= 2:
        cv2.polylines(
            canvas, [vis_pts], isClosed=False, color=(255, 255, 255), thickness=1
        )

    # Draw circles at each visible keypoint
    for i in range(len(coords)):
        if visible[i]:
            pt = (int(coords[i, 0]), int(coords[i, 1]))
            color = colors[i % len(colors)]
            cv2.circle(canvas, pt, 3, color, -1)

    return canvas
