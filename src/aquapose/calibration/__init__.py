"""Calibration loading and refractive camera geometry."""

from .loader import (
    CalibrationData,
    CameraData,
    UndistortionMaps,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from .projection import RefractiveProjectionModel, triangulate_rays
from .uncertainty import (
    UncertaintyResult,
    build_rig_from_calibration,
    build_synthetic_rig,
    compute_triangulation_uncertainty,
    generate_uncertainty_report,
)

__all__ = [
    "CalibrationData",
    "CameraData",
    "RefractiveProjectionModel",
    "UncertaintyResult",
    "UndistortionMaps",
    "build_rig_from_calibration",
    "build_synthetic_rig",
    "compute_triangulation_uncertainty",
    "compute_undistortion_maps",
    "generate_uncertainty_report",
    "load_calibration_data",
    "triangulate_rays",
    "undistort_image",
]
