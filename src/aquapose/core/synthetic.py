"""SyntheticDataStage -- generates synthetic fish detections from known 3D geometry.

Replaces Detection + Midline stages in synthetic mode. Generates 3D fish
splines, projects them through the real refractive calibration model, and
produces Detection + AnnotatedDetection data structures matching the format
of the real pipeline stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from aquapose.calibration import RefractiveProjectionModel
from aquapose.calibration.loader import load_calibration_data
from aquapose.core.context import PipelineContext
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.reconstruction.midline import Midline2D
from aquapose.segmentation.crop import CropRegion
from aquapose.segmentation.detector import Detection

if TYPE_CHECKING:
    from aquapose.engine.config import SyntheticConfig


def _generate_fish_splines(
    fish_count: int,
    n_points: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate 3D fish splines as straight or slightly curved midlines.

    Args:
        fish_count: Number of fish to generate.
        n_points: Number of midline points per fish.
        rng: Numpy random generator for reproducibility.

    Returns:
        List of fish splines, each shape (n_points, 3).
    """
    splines = []
    for _i in range(fish_count):
        # Place fish at different x/y positions within approximate tank bounds
        cx = rng.uniform(-0.08, 0.08)
        cy = rng.uniform(-0.08, 0.08)
        cz = rng.uniform(0.02, 0.12)

        # Generate a midline as a slightly curved line
        t = np.linspace(-0.02, 0.02, n_points)
        heading = rng.uniform(0, 2 * np.pi)
        curvature = rng.uniform(-5.0, 5.0)

        x = cx + t * np.cos(heading) - curvature * t**2 * np.sin(heading)
        y = cy + t * np.sin(heading) + curvature * t**2 * np.cos(heading)
        z = np.full(n_points, cz)

        spline = np.stack([x, y, z], axis=-1).astype(np.float32)
        splines.append(spline)
    return splines


def _apply_frame_displacement(
    splines: list[np.ndarray],
    frame_index: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Apply small per-frame displacement to simulate swimming.

    Args:
        splines: List of base fish splines, each shape (n_points, 3).
        frame_index: Current frame index (used for deterministic motion).
        rng: Numpy random generator for reproducibility.

    Returns:
        List of displaced splines for this frame.
    """
    displaced = []
    for spline in splines:
        # Small per-frame displacement (swimming motion)
        dx = 0.0005 * frame_index + rng.normal(0, 0.0002)
        dy = 0.0005 * frame_index * 0.3 + rng.normal(0, 0.0002)
        offset = np.array([dx, dy, 0.0], dtype=np.float32)
        displaced.append(spline + offset)
    return displaced


class SyntheticDataStage:
    """Stage that generates synthetic detections and midlines from known 3D geometry.

    Satisfies the Stage protocol. Replaces both DetectionStage and MidlineStage
    in synthetic mode. Projects known 3D fish splines through the real refractive
    calibration model to produce realistic 2D detection data.

    Args:
        calibration_path: Path to AquaCal calibration JSON file.
        synthetic_config: Configuration for synthetic data generation.
    """

    def __init__(
        self,
        calibration_path: str,
        synthetic_config: SyntheticConfig,
    ) -> None:
        self._calibration_path = calibration_path
        self._config = synthetic_config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Generate synthetic fish data and populate context fields.

        Loads calibration, generates 3D fish splines, projects to 2D for each
        camera, and populates context with Detection and AnnotatedDetection data.

        Args:
            context: Pipeline context to populate.

        Returns:
            Context with detections, annotated_detections, frame_count, and
            camera_ids populated.
        """
        rng = np.random.default_rng(self._config.seed)
        cal_data = load_calibration_data(self._calibration_path)
        camera_ids = cal_data.ring_cameras

        # Build projection models per camera
        models: dict[str, RefractiveProjectionModel] = {}
        image_sizes: dict[str, tuple[int, int]] = {}
        for cam_name in camera_ids:
            cam = cal_data.cameras[cam_name]
            models[cam_name] = RefractiveProjectionModel(
                K=cam.K,
                R=cam.R,
                t=cam.t,
                water_z=cal_data.water_z,
                normal=cal_data.interface_normal,
                n_air=cal_data.n_air,
                n_water=cal_data.n_water,
            )
            image_sizes[cam_name] = cam.image_size

        # Generate base 3D splines
        n_points = 15
        base_splines = _generate_fish_splines(self._config.fish_count, n_points, rng)

        all_detections: list[dict[str, list]] = []
        all_annotated: list[dict[str, list]] = []

        for frame_idx in range(self._config.frame_count):
            # Per-frame RNG for displacement (deterministic per frame)
            frame_rng = np.random.default_rng(self._config.seed + frame_idx + 1)
            frame_splines = _apply_frame_displacement(
                base_splines, frame_idx, frame_rng
            )

            frame_dets: dict[str, list] = {cam: [] for cam in camera_ids}
            frame_annot: dict[str, list] = {cam: [] for cam in camera_ids}

            for fish_id, spline_3d in enumerate(frame_splines):
                for cam_name in camera_ids:
                    model = models[cam_name]
                    img_w, img_h = image_sizes[cam_name]

                    # Project 3D points to 2D
                    pts_tensor = torch.from_numpy(spline_3d).float()
                    pixels, valid = model.project(pts_tensor)
                    pixels_np = pixels.cpu().numpy()
                    valid_np = valid.cpu().numpy()

                    # Check visibility: all points must be valid and within image
                    if not valid_np.all():
                        continue
                    if (
                        pixels_np[:, 0].min() < 0
                        or pixels_np[:, 0].max() >= img_w
                        or pixels_np[:, 1].min() < 0
                        or pixels_np[:, 1].max() >= img_h
                    ):
                        continue

                    # Add noise if configured
                    if self._config.noise_std > 0:
                        noise = rng.normal(
                            0, self._config.noise_std, size=pixels_np.shape
                        ).astype(np.float32)
                        pixels_np = pixels_np + noise

                    # Create bounding box from projected points with padding
                    x_min = float(pixels_np[:, 0].min())
                    y_min = float(pixels_np[:, 1].min())
                    x_max = float(pixels_np[:, 0].max())
                    y_max = float(pixels_np[:, 1].max())
                    pad = 10
                    bbox_x = max(0, int(x_min - pad))
                    bbox_y = max(0, int(y_min - pad))
                    bbox_w = min(img_w, int(x_max + pad)) - bbox_x
                    bbox_h = min(img_h, int(y_max + pad)) - bbox_y

                    det = Detection(
                        bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
                        mask=None,
                        area=bbox_w * bbox_h,
                        confidence=1.0,
                    )
                    frame_dets[cam_name].append(det)

                    # Create Midline2D
                    midline = Midline2D(
                        points=pixels_np.astype(np.float32),
                        half_widths=np.full(n_points, 5.0, dtype=np.float32),
                        fish_id=fish_id,
                        camera_id=cam_name,
                        frame_index=frame_idx,
                        is_head_to_tail=True,
                    )

                    # Create CropRegion
                    crop = CropRegion(
                        x1=bbox_x,
                        y1=bbox_y,
                        x2=bbox_x + bbox_w,
                        y2=bbox_y + bbox_h,
                        frame_h=img_h,
                        frame_w=img_w,
                    )

                    annot = AnnotatedDetection(
                        detection=det,
                        mask=None,
                        crop_region=crop,
                        midline=midline,
                        camera_id=cam_name,
                        frame_index=frame_idx,
                    )
                    frame_annot[cam_name].append(annot)

            all_detections.append(frame_dets)
            all_annotated.append(frame_annot)

        context.frame_count = self._config.frame_count
        context.camera_ids = camera_ids
        context.detections = all_detections
        context.annotated_detections = all_annotated
        return context
