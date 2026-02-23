"""Fish shape generation and projection to 2D midlines.

Generates known ground truth 3D fish midlines (straight lines and circular
arcs) and projects them through refractive camera models to produce Midline2D
and MidlineSet objects compatible with the triangulation and curve optimization
pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.interpolate
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.reconstruction.midline import Midline2D
from aquapose.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
    Midline3D,
    MidlineSet,
)


@dataclass
class FishConfig:
    """Configuration for a single synthetic fish.

    Attributes:
        position: 3D world position of the fish centroid (x, y, z).
            Default places the fish at (0, 0, 1.25) — 0.5m below a
            water surface at z=0.75.
        heading_rad: Heading angle in radians, rotation around the Z axis.
            0 means the fish points along the positive X axis.
        curvature: Body curvature in m^-1. 0 produces a straight fish.
            Non-zero values produce a circular arc with radius=1/|curvature|.
        scale: Total body arc length in metres. Default 0.085 (85mm).
        n_points: Number of evenly spaced body points. Default N_SAMPLE_POINTS.
    """

    position: tuple[float, float, float] = (0.0, 0.0, 1.25)
    heading_rad: float = 0.0
    curvature: float = 0.0
    scale: float = 0.085
    n_points: int = N_SAMPLE_POINTS


def generate_fish_3d(config: FishConfig) -> np.ndarray:
    """Generate evenly spaced 3D body points for a synthetic fish.

    For straight fish (curvature == 0), produces a line segment along the
    heading direction centered at the fish position. For curved fish
    (curvature != 0), produces a circular arc with the given radius, subtending
    an arc length equal to ``config.scale``, rotated by ``config.heading_rad``
    around the Z axis.

    Args:
        config: FishConfig specifying shape, position, and heading.

    Returns:
        Body point positions, shape (n_points, 3), float32.
    """
    n = config.n_points
    cx, cy, cz = config.position
    heading = config.heading_rad
    kappa = config.curvature
    L = config.scale

    # Parameterize along arc: t in [-0.5, 0.5] * L gives arc-length coords
    t_vals = np.linspace(-0.5 * L, 0.5 * L, n)

    if abs(kappa) < 1e-9:
        # Straight line along heading direction
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        x = cx + t_vals * cos_h
        y = cy + t_vals * sin_h
        z = np.full(n, cz, dtype=np.float64)
    else:
        # Circular arc in XY plane, radius = 1/|kappa|
        r = 1.0 / abs(kappa)
        # Angle subtended by each arc-length step
        theta_vals = t_vals / r  # angle from arc midpoint

        # Unrotated arc: centred at origin, midpoint at angle 0
        # Arc points along the heading direction
        # Centre of curvature is perpendicular to heading at the fish centroid
        sign = 1.0 if kappa > 0 else -1.0

        # In the local frame (heading=0):
        # centre of curvature is at (0, sign*r) from fish centroid
        # arc points: x_local = r * sin(theta), y_local = sign * r * (1 - cos(theta))
        x_local = r * np.sin(theta_vals)
        y_local = sign * r * (1.0 - np.cos(theta_vals))

        # Rotate by heading_rad
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        x = cx + cos_h * x_local - sin_h * y_local
        y = cy + sin_h * x_local + cos_h * y_local
        z = np.full(n, cz, dtype=np.float64)

    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts


def generate_fish_half_widths(
    n_points: int = N_SAMPLE_POINTS,
    max_ratio: float = 0.08,
    scale: float = 0.085,
) -> np.ndarray:
    """Generate elliptical half-width profile for a synthetic fish.

    Models the fish body as an elliptical cross-section that tapers toward
    both the head (index 0) and tail (index n_points-1). The thickest point
    is at 40% of the body length from the head.

    Args:
        n_points: Number of body points. Default N_SAMPLE_POINTS.
        max_ratio: Maximum half-width as a fraction of ``scale``. Default 0.08
            gives approximately 6.8mm max half-width for an 85mm fish.
        scale: Fish body arc length in metres.

    Returns:
        Half-width values in world metres, shape (n_points,), float32.
    """
    t = np.linspace(0.0, 1.0, n_points)
    # Elliptical profile: peak at t=0.4 from head
    peak = 0.4
    # Normalised arc-length distance from peak, scaled to [0, 1]
    # Use a beta-distribution-like shape: zero at endpoints, max at peak
    # Simple approach: piecewise linear from 0 at t=0, max at t=peak, 0 at t=1
    hw = np.where(
        t <= peak,
        t / peak,
        (1.0 - t) / (1.0 - peak),
    )
    # Apply sqrt for rounder profile
    hw = np.sqrt(np.clip(hw, 0.0, 1.0))
    max_hw = max_ratio * scale
    return (hw * max_hw).astype(np.float32)


def make_ground_truth_midline3d(
    fish_id: int,
    frame_index: int,
    pts_3d: np.ndarray,
    half_widths: np.ndarray,
) -> Midline3D:
    """Construct a ground truth Midline3D by fitting a B-spline to 3D points.

    Fits a fixed ``SPLINE_N_CTRL``-control-point cubic B-spline using
    ``scipy.interpolate.make_lsq_spline`` with the canonical ``SPLINE_KNOTS``
    knot vector. The resulting Midline3D has zero residuals (it is the
    true ground truth, not a reconstruction).

    Args:
        fish_id: Fish identifier integer.
        frame_index: Frame index.
        pts_3d: Ground truth 3D body positions, shape (N, 3), float32 or float64.
        half_widths: Half-widths in world metres, shape (N,), float32 or float64.

    Returns:
        Midline3D with fitted B-spline control points and knots.
    """
    n = len(pts_3d)
    u_param = np.linspace(0.0, 1.0, n, dtype=np.float64)

    spl = scipy.interpolate.make_lsq_spline(
        u_param,
        pts_3d.astype(np.float64),
        SPLINE_KNOTS,
        k=SPLINE_K,
    )
    control_points = spl.c.astype(np.float32)  # shape (SPLINE_N_CTRL, 3)

    # Compute arc length via numerical integration
    u_fine = np.linspace(0.0, 1.0, 1000)
    curve_pts = spl(u_fine)  # shape (1000, 3)
    diffs = np.diff(curve_pts, axis=0)
    arc_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=control_points,
        knots=SPLINE_KNOTS.astype(np.float32),
        degree=SPLINE_K,
        arc_length=arc_length,
        half_widths=half_widths.astype(np.float32),
        n_cameras=99,
        mean_residual=0.0,
        max_residual=0.0,
    )


def project_fish_to_midline2d(
    pts_3d: np.ndarray,
    half_widths_3d: np.ndarray,
    model: RefractiveProjectionModel,
    fish_id: int,
    camera_id: str,
    frame_index: int,
) -> Midline2D | None:
    """Project 3D fish body points through a refractive camera model.

    Converts 3D body positions to 2D pixel coordinates using
    ``RefractiveProjectionModel.project``. Points where the projection is
    invalid (above water or behind camera) are replaced with NaN. Returns
    None if fewer than 3 points are visible.

    Half-widths are converted from world metres to pixels using the pinhole
    approximation: ``hw_px = hw_m * fx / depth``, where ``depth`` is the
    Z distance of each body point below the water surface.

    Args:
        pts_3d: Ground truth 3D body positions, shape (N, 3), float32.
        half_widths_3d: Half-widths in world metres, shape (N,), float32.
        model: Refractive projection model for the target camera.
        fish_id: Fish identifier integer.
        camera_id: Camera identifier string.
        frame_index: Frame index.

    Returns:
        Midline2D with projected pixel coordinates and converted half-widths,
        or None if fewer than 3 body points are visible in this camera.
    """
    pts_torch = torch.from_numpy(pts_3d.astype(np.float32))  # (N, 3)
    proj_px, valid = model.project(pts_torch)  # (N, 2), (N,)

    valid_np = valid.detach().cpu().numpy()
    n_visible = int(valid_np.sum())
    if n_visible < 3:
        return None

    proj_np = proj_px.detach().cpu().numpy()  # (N, 2)

    # Build pixel array with NaN for invalid projections
    points_out = np.where(
        valid_np[:, np.newaxis],
        proj_np,
        np.full_like(proj_np, np.nan),
    ).astype(np.float32)

    # Convert half-widths from world metres to pixels using pinhole approximation
    fx = float(model.K[0, 0].item())
    water_z = model.water_z
    hw_px = np.zeros(len(pts_3d), dtype=np.float32)
    for i, (pt, hw_m) in enumerate(zip(pts_3d, half_widths_3d, strict=True)):
        depth = float(pt[2]) - water_z
        if depth > 0 and valid_np[i]:
            hw_px[i] = float(hw_m) * fx / depth
        else:
            hw_px[i] = 0.0

    return Midline2D(
        points=points_out,
        half_widths=hw_px,
        fish_id=fish_id,
        camera_id=camera_id,
        frame_index=frame_index,
        is_head_to_tail=True,
    )


def generate_synthetic_midline_sets(
    models: dict[str, RefractiveProjectionModel],
    fish_configs: list[FishConfig] | None = None,
    n_frames: int = 1,
    frame_start: int = 0,
) -> tuple[list[MidlineSet], list[dict[int, Midline3D]]]:
    """Generate synthetic MidlineSets and ground truth Midline3D objects.

    For each frame, generates 3D fish body points for every fish config,
    projects them through all cameras to produce 2D Midline2D observations,
    and assembles them into MidlineSet dicts. Also produces ground truth
    Midline3D objects for comparison with reconstruction outputs.

    Args:
        models: Dict mapping camera ID to RefractiveProjectionModel.
        fish_configs: List of FishConfig objects, one per fish. If None,
            defaults to a single straight fish at (0, 0, 1.25) with heading=0.
        n_frames: Number of frames to generate. Default 1.
        frame_start: Starting frame index. Default 0.

    Returns:
        Tuple of (midline_sets, ground_truths) where:
        - midline_sets: List of MidlineSet dicts, one per frame. Each
          MidlineSet maps fish_id -> camera_id -> Midline2D.
        - ground_truths: List of dicts, one per frame. Each dict maps
          fish_id -> Midline3D ground truth.
    """
    if fish_configs is None:
        fish_configs = [FishConfig()]

    midline_sets: list[MidlineSet] = []
    ground_truths: list[dict[int, Midline3D]] = []

    for frame_offset in range(n_frames):
        frame_index = frame_start + frame_offset
        midline_set: MidlineSet = {}
        frame_gt: dict[int, Midline3D] = {}

        for fish_id, cfg in enumerate(fish_configs):
            pts_3d = generate_fish_3d(cfg)
            half_widths = generate_fish_half_widths(
                n_points=cfg.n_points,
                scale=cfg.scale,
            )

            # Ground truth Midline3D
            gt_midline = make_ground_truth_midline3d(
                fish_id=fish_id,
                frame_index=frame_index,
                pts_3d=pts_3d,
                half_widths=half_widths,
            )
            frame_gt[fish_id] = gt_midline

            # Project to each camera
            cam_midlines: dict[str, Midline2D] = {}
            for cam_id, model in models.items():
                midline_2d = project_fish_to_midline2d(
                    pts_3d=pts_3d,
                    half_widths_3d=half_widths,
                    model=model,
                    fish_id=fish_id,
                    camera_id=cam_id,
                    frame_index=frame_index,
                )
                if midline_2d is not None:
                    cam_midlines[cam_id] = midline_2d

            if cam_midlines:
                midline_set[fish_id] = cam_midlines

        midline_sets.append(midline_set)
        ground_truths.append(frame_gt)

    return midline_sets, ground_truths
