"""Calibration loading and refractive camera geometry."""

from .loader import (
    CalibrationData,
    CameraData,
    UndistortionMaps,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from .luts import (
    ForwardLUT,
    LutConfigLike,
    compute_lut_hash,
    generate_forward_luts,
    load_forward_luts,
    save_forward_luts,
    validate_forward_lut,
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
    "ForwardLUT",
    "LutConfigLike",
    "RefractiveProjectionModel",
    "UncertaintyResult",
    "UndistortionMaps",
    "build_rig_from_calibration",
    "build_synthetic_rig",
    "compute_lut_hash",
    "compute_triangulation_uncertainty",
    "compute_undistortion_maps",
    "generate_forward_luts",
    "generate_uncertainty_report",
    "load_calibration_data",
    "load_forward_luts",
    "save_forward_luts",
    "triangulate_rays",
    "undistort_image",
    "validate_forward_lut",
]
