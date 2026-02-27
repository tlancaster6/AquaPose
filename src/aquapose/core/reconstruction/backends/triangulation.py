"""Triangulation backend for the Reconstruction stage.

Wraps the v1.0 ``triangulate_midlines()`` function, loading calibration
at construction time and delegating per-frame reconstruction without
reimplementing any triangulation logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aquapose.reconstruction.triangulation import (
    DEFAULT_INLIER_THRESHOLD,
    Midline3D,
    MidlineSet,
    triangulate_midlines,
)

__all__ = ["TriangulationBackend"]

logger = logging.getLogger(__name__)


class TriangulationBackend:
    """Multi-view RANSAC triangulation + B-spline fitting reconstruction backend.

    Loads calibration and constructs per-camera RefractiveProjectionModel
    instances at construction time (fail-fast: missing calibration raises
    immediately). Per-frame reconstruction delegates entirely to the v1.0
    ``triangulate_midlines()`` function.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        inlier_threshold: Maximum reprojection error (pixels) for RANSAC inliers.
        snap_threshold: Maximum pixel distance from epipolar curve for
            correspondence refinement.
        max_depth: Maximum allowed fish depth below the water surface (metres).
            None disables the upper depth bound.

    Raises:
        FileNotFoundError: If *calibration_path* does not exist.
    """

    def __init__(
        self,
        calibration_path: str | Path,
        inlier_threshold: float = DEFAULT_INLIER_THRESHOLD,
        snap_threshold: float = 20.0,
        max_depth: float | None = None,
    ) -> None:
        self._calibration_path = Path(calibration_path)
        self._inlier_threshold = inlier_threshold
        self._snap_threshold = snap_threshold
        self._max_depth = max_depth

        # Eagerly load calibration — fail-fast on missing file
        self._models = self._load_models(self._calibration_path)

    def reconstruct_frame(
        self,
        frame_idx: int,
        midline_set: MidlineSet,
    ) -> dict[int, Midline3D]:
        """Triangulate all fish midlines for a single frame.

        Delegates to ``triangulate_midlines()`` with the pre-loaded models.

        Args:
            frame_idx: Frame index to embed in output Midline3D structs.
            midline_set: Nested dict mapping fish_id to camera_id to Midline2D.

        Returns:
            Dict mapping fish_id to Midline3D. Only includes fish with
            sufficient valid body points for spline fitting.
        """
        return triangulate_midlines(
            midline_set=midline_set,
            models=self._models,
            frame_index=frame_idx,
            inlier_threshold=self._inlier_threshold,
            snap_threshold=self._snap_threshold,
            max_depth=self._max_depth,
        )

    @staticmethod
    def _load_models(
        calibration_path: Path,
    ) -> dict[str, Any]:
        """Load calibration and build per-camera RefractiveProjectionModel dict.

        Args:
            calibration_path: Path to AquaCal calibration JSON.

        Returns:
            Dict mapping camera_id to RefractiveProjectionModel.

        Raises:
            FileNotFoundError: If *calibration_path* does not exist.
        """
        from aquapose.calibration.loader import (
            compute_undistortion_maps,
            load_calibration_data,
        )
        from aquapose.calibration.projection import RefractiveProjectionModel

        calib = load_calibration_data(str(calibration_path))
        models: dict[str, Any] = {}
        for cam_id, cam_data in calib.cameras.items():
            maps = compute_undistortion_maps(cam_data)
            models[cam_id] = RefractiveProjectionModel(
                K=maps.K_new,
                R=cam_data.R,
                t=cam_data.t,
                water_z=calib.water_z,
                normal=calib.interface_normal,
                n_air=calib.n_air,
                n_water=calib.n_water,
            )
        return models
