"""Curve optimizer backend for the Reconstruction stage.

Wraps the v1.0 ``CurveOptimizer.optimize_midlines()`` method, loading
calibration at construction time. Maintains a stateful ``CurveOptimizer``
instance for warm-starting across frames, exactly as v1.0 did.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aquapose.reconstruction.curve_optimizer import CurveOptimizer, CurveOptimizerConfig
from aquapose.reconstruction.triangulation import Midline3D, MidlineSet

__all__ = ["CurveOptimizerBackend"]

logger = logging.getLogger(__name__)

# Default camera to exclude — centre top-down wide-angle, poor quality.
_DEFAULT_SKIP_CAMERA_ID = "e3v8250"


class CurveOptimizerBackend:
    """Correspondence-free 3D B-spline curve optimizer reconstruction backend.

    Loads calibration and constructs per-camera RefractiveProjectionModel
    instances at construction time (fail-fast: missing calibration raises
    immediately). Maintains a stateful ``CurveOptimizer`` for warm-starting
    across frames (persists across ``reconstruct_frame()`` calls).

    Per-frame reconstruction delegates entirely to the v1.0
    ``CurveOptimizer.optimize_midlines()`` method.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        skip_camera_id: Camera ID to exclude from optimization.
        lr: L-BFGS learning rate override. None uses CurveOptimizerConfig default.
        max_iter_coarse: Maximum L-BFGS coarse stage iterations. None uses default.
        max_iter_fine: Maximum L-BFGS fine stage iterations. None uses default.

    Raises:
        FileNotFoundError: If *calibration_path* does not exist.
    """

    def __init__(
        self,
        calibration_path: str | Path,
        skip_camera_id: str = _DEFAULT_SKIP_CAMERA_ID,
        lr: float | None = None,
        max_iter_coarse: int | None = None,
        max_iter_fine: int | None = None,
    ) -> None:
        self._calibration_path = Path(calibration_path)
        self._skip_camera_id = skip_camera_id

        # Build optimizer config from overrides (only override what was specified)
        config_kwargs: dict[str, Any] = {}
        if lr is not None:
            config_kwargs["lbfgs_lr"] = lr
        if max_iter_coarse is not None:
            config_kwargs["lbfgs_max_iter_coarse"] = max_iter_coarse
        if max_iter_fine is not None:
            config_kwargs["lbfgs_max_iter_fine"] = max_iter_fine

        optimizer_config = (
            CurveOptimizerConfig(**config_kwargs) if config_kwargs else None
        )

        # Stateful optimizer — persists across reconstruct_frame() calls for warm-starting
        self._optimizer = CurveOptimizer(config=optimizer_config)

        # Eagerly load calibration — fail-fast on missing file
        self._models = self._load_models(self._calibration_path, skip_camera_id)

    def reconstruct_frame(
        self,
        frame_idx: int,
        midline_set: MidlineSet,
    ) -> dict[int, Midline3D]:
        """Optimize all fish midlines for a single frame.

        Delegates to the stateful ``CurveOptimizer.optimize_midlines()`` with
        the pre-loaded models. The optimizer warm-starts from the previous
        frame's solution automatically.

        Args:
            frame_idx: Frame index (used for warm-start bookkeeping).
            midline_set: Nested dict mapping fish_id to camera_id to Midline2D.

        Returns:
            Dict mapping fish_id to Midline3D for all successfully optimized fish.
        """
        return self._optimizer.optimize_midlines(
            midline_set=midline_set,
            models=self._models,
            frame_index=frame_idx,
        )

    @staticmethod
    def _load_models(
        calibration_path: Path,
        skip_camera_id: str,
    ) -> dict[str, Any]:
        """Load calibration and build per-camera RefractiveProjectionModel dict.

        Args:
            calibration_path: Path to AquaCal calibration JSON.
            skip_camera_id: Camera ID to exclude from the returned dict.

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
            if cam_id == skip_camera_id:
                continue
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
