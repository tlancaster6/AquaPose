"""RANSAC centroid clustering backend for the Association stage.

Wraps the v1.0 ``discover_births`` function from ``aquapose.tracking.associate``
as a Stage 3 backend. Loads calibration data at construction and exposes a
stateless ``associate_frame`` method that treats all detections as unclaimed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aquapose.core.association.types import AssociationBundle

__all__ = ["RansacCentroidBackend"]

logger = logging.getLogger(__name__)

# Camera to exclude — centre top-down wide-angle, poor association quality.
_DEFAULT_SKIP_CAMERA_ID = "e3v8250"


class RansacCentroidBackend:
    """RANSAC centroid clustering backend for cross-view fish association.

    Loads calibration data at construction (eager, fail-fast). Exposes
    ``associate_frame`` which treats every detection in a frame as unclaimed
    and delegates to the v1.0 ``discover_births`` RANSAC algorithm.

    The key difference from v1.0: in the original pipeline, ``discover_births``
    only saw detections not already claimed by existing tracks. Here, ALL
    detections are unclaimed because tracking (Stage 4) has not yet run.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        expected_count: Expected fish count, used as RANSAC stopping criterion.
        min_cameras: Minimum cameras required for a valid multi-view bundle.
        reprojection_threshold: Maximum pixel reprojection error for inliers.
        skip_camera_id: Camera ID to exclude from association.

    Raises:
        FileNotFoundError: If *calibration_path* does not exist.
    """

    def __init__(
        self,
        calibration_path: str | Path,
        expected_count: int = 9,
        min_cameras: int = 3,
        reprojection_threshold: float = 15.0,
        skip_camera_id: str = _DEFAULT_SKIP_CAMERA_ID,
    ) -> None:
        from aquapose.calibration.loader import (
            compute_undistortion_maps,
            load_calibration_data,
        )
        from aquapose.calibration.projection import RefractiveProjectionModel

        self._calibration_path = Path(calibration_path)
        self._expected_count = expected_count
        self._min_cameras = min_cameras
        self._reprojection_threshold = reprojection_threshold
        self._skip_camera_id = skip_camera_id

        if not self._calibration_path.exists():
            raise FileNotFoundError(
                f"calibration_path does not exist: {self._calibration_path}"
            )

        # Build per-camera refractive projection models at construction.
        calib = load_calibration_data(self._calibration_path)

        self._models: dict[str, RefractiveProjectionModel] = {}
        for cam_id, cam_data in calib.cameras.items():
            if cam_id == self._skip_camera_id:
                logger.info(
                    "RansacCentroidBackend: skipping excluded camera %s", cam_id
                )
                continue
            maps = compute_undistortion_maps(cam_data)
            self._models[cam_id] = RefractiveProjectionModel(
                K=maps.K_new,
                R=cam_data.R,
                t=cam_data.t,
                water_z=calib.water_z,
                normal=calib.interface_normal,
                n_air=calib.n_air,
                n_water=calib.n_water,
            )

        logger.info(
            "RansacCentroidBackend: loaded %d camera models: %s",
            len(self._models),
            sorted(self._models),
        )

    def associate_frame(
        self,
        detections_per_camera: dict[str, list],
    ) -> list[AssociationBundle]:
        """Run RANSAC centroid clustering on all detections in a frame.

        All detections are treated as unclaimed — tracking has not yet run
        at this point in the pipeline. Delegates to ``discover_births`` from
        ``aquapose.tracking.associate``.

        Args:
            detections_per_camera: Mapping from camera_id to list of
                Detection objects for that camera in this frame. Cameras not
                in the calibration models are ignored.

        Returns:
            List of AssociationBundle objects, one per identified fish.
            Empty list if the input has no detections or no camera models
            are available.
        """
        from aquapose.tracking.associate import discover_births

        if not detections_per_camera or not self._models:
            return []

        # Filter to cameras with both detections and calibration models.
        active_cameras = {
            cam_id: dets
            for cam_id, dets in detections_per_camera.items()
            if cam_id in self._models and len(dets) > 0
        }

        if not active_cameras:
            return []

        # All detections are unclaimed (no tracks yet in Stage 3).
        unclaimed_indices: dict[str, list[int]] = {
            cam_id: list(range(len(dets))) for cam_id, dets in active_cameras.items()
        }

        # Include all cameras (not just active) so discover_births has the
        # full model dict for reprojection scoring.
        results = discover_births(
            unclaimed_indices=unclaimed_indices,
            detections_per_camera=active_cameras,
            models=self._models,
            expected_count=self._expected_count,
            reprojection_threshold=self._reprojection_threshold,
            min_cameras=self._min_cameras,
        )

        bundles: list[AssociationBundle] = []
        for fish_idx, assoc in enumerate(results):
            bundles.append(
                AssociationBundle(
                    fish_idx=fish_idx,
                    centroid_3d=assoc.centroid_3d,
                    camera_detections=assoc.camera_detections,
                    n_cameras=assoc.n_cameras,
                    reprojection_residual=assoc.reprojection_residual,
                    confidence=assoc.confidence,
                )
            )

        return bundles
