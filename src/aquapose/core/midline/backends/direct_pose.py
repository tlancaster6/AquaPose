"""Direct pose estimation backend for the Midline stage.

Implements the DirectPoseBackend that directly regresses fish midline
keypoints from OBB-aligned affine crops, bypassing U-Net segmentation.

Uses a _PoseModel (U-Net encoder + regression head) to predict N anatomical
keypoints in crop-normalized coordinates, then back-projects to frame space,
fits a CubicSpline through visible keypoints, and resamples to exactly
n_sample_points. Points outside the observed arc-span are NaN-padded with
confidence=0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from aquapose.core.midline.types import AnnotatedDetection
from aquapose.reconstruction.midline import Midline2D
from aquapose.segmentation.crop import extract_affine_crop, invert_affine_points
from aquapose.segmentation.detector import Detection

__all__ = ["DirectPoseBackend"]

logger = logging.getLogger(__name__)


class DirectPoseBackend:
    """Direct keypoint regression backend for the Midline stage.

    Accepts detection bounding boxes and OBB angles, extracts rotation-aligned
    affine crops, runs a _PoseModel to predict anatomical keypoints, fits a
    CubicSpline through visible keypoints (confidence >= floor), and resamples
    to exactly n_points midline points. Invisible or out-of-span regions are
    NaN-padded with confidence=0.

    This backend produces Midline2D with identical shape and field structure
    to SegmentThenExtractBackend, enabling downstream stages to handle both
    backends uniformly.

    Args:
        weights_path: Path to _PoseModel state-dict (.pth file). Must exist.
        device: PyTorch device string (e.g. "cuda", "cpu"). Defaults to "cuda".
        n_points: Number of midline sample points to produce per detection.
        n_keypoints: Number of anatomical keypoints the model outputs.
        keypoint_t_values: Per-keypoint arc-fraction values in [0, 1] from
            nose (0) to tail (1). If None, uses uniform spacing
            ``linspace(0, 1, n_keypoints)``.
        confidence_floor: Minimum per-keypoint confidence to treat as visible.
            Keypoints below this floor are replaced by NaN + confidence=0.
        min_observed_keypoints: Minimum number of visible keypoints required to
            fit the spline. Fewer visible keypoints yields midline=None.
        crop_size: Crop canvas size (width, height) for affine warp. Must match
            the model's training resolution.
        **kwargs: Absorbed silently for compatibility with ``get_backend``
            keyword forwarding (e.g. ``confidence_threshold``, ``min_area``).

    Raises:
        FileNotFoundError: If *weights_path* does not exist.
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cuda",
        n_points: int = 15,
        n_keypoints: int = 6,
        keypoint_t_values: list[float] | None = None,
        confidence_floor: float = 0.1,
        min_observed_keypoints: int = 3,
        crop_size: tuple[int, int] = (128, 128),
        **kwargs: Any,
    ) -> None:
        import torch

        from aquapose.training.pose import _PoseModel

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"DirectPoseBackend: keypoint model weights not found: {weights_path}. "
                "Provide a valid path to a _PoseModel state-dict (.pth)."
            )

        self._n_points = n_points
        self._n_keypoints = n_keypoints
        self._conf_floor = confidence_floor
        self._min_observed = min_observed_keypoints
        self._crop_size = crop_size
        self._device = device

        # Uniform t-values as default (nose=0.0, tail=1.0)
        if keypoint_t_values is not None:
            self._t_values = np.array(keypoint_t_values, dtype=np.float64)
        else:
            self._t_values = np.linspace(0.0, 1.0, n_keypoints)

        # Eager model load — fail-fast at construction
        model = _PoseModel(n_keypoints=n_keypoints, pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(torch.device(device))
        model.eval()
        self._model = model

        logger.info(
            "DirectPoseBackend: loaded _PoseModel (weights=%s, device=%s, "
            "n_keypoints=%d, n_points=%d, conf_floor=%.2f, min_observed=%d)",
            weights_path,
            device,
            n_keypoints,
            n_points,
            confidence_floor,
            min_observed_keypoints,
        )

    def process_frame(
        self,
        frame_idx: int,
        frame_dets: dict[str, list[Detection]],
        frames: dict[str, np.ndarray],
        camera_ids: list[str],
    ) -> dict[str, list[AnnotatedDetection]]:
        """Process a single frame: run keypoint inference and fit midlines.

        For each camera in camera_ids, for each detection, extracts an
        OBB-aligned affine crop, runs _PoseModel inference, fits a CubicSpline
        through visible keypoints, resamples to n_points, and NaN-pads points
        outside the observed arc-span.

        Args:
            frame_idx: Current frame index (used for Midline2D metadata).
            frame_dets: Per-camera detection lists from Stage 1.
            frames: Per-camera undistorted frame images (BGR uint8).
            camera_ids: Active camera identifiers.

        Returns:
            Per-camera list of AnnotatedDetection objects, one per input
            detection. Midline is None if fewer than min_observed_keypoints
            are above confidence_floor.
        """
        annotated: dict[str, list[AnnotatedDetection]] = {}

        for cam_id in camera_ids:
            cam_dets = frame_dets.get(cam_id, [])
            frame = frames.get(cam_id)

            if not cam_dets or frame is None:
                annotated[cam_id] = []
                continue

            cam_annotated: list[AnnotatedDetection] = []

            for det in cam_dets:
                midline = self._process_single_detection(
                    det=det,
                    frame=frame,
                    cam_id=cam_id,
                    frame_idx=frame_idx,
                )
                cam_annotated.append(
                    AnnotatedDetection(
                        detection=det,
                        mask=None,
                        crop_region=None,
                        midline=midline,
                        camera_id=cam_id,
                        frame_index=frame_idx,
                    )
                )

            annotated[cam_id] = cam_annotated

        return annotated

    def _process_single_detection(
        self,
        det: Detection,
        frame: np.ndarray,
        cam_id: str,
        frame_idx: int,
    ) -> Midline2D | None:
        """Run inference on a single detection and return a Midline2D or None.

        Steps:
            1. Extract OBB-aligned affine crop.
            2. Run _PoseModel inference -> (n_keypoints, 2) normalized coords.
            3. Compute per-keypoint confidence heuristic.
            4. Apply confidence floor — count visible keypoints.
            5. If below min_observed_keypoints, return None.
            6. Back-project visible keypoints to frame space.
            7. Fit CubicSpline through visible (t, xy) pairs.
            8. Resample to n_points; NaN-pad outside [t_min, t_max].
            9. Interpolate confidence along spline; zero out NaN regions.
            10. Return Midline2D.

        Args:
            det: Detection from Stage 1.
            frame: Full undistorted frame (BGR uint8).
            cam_id: Camera identifier for metadata.
            frame_idx: Frame index for metadata.

        Returns:
            Midline2D with n_points points and per-point confidence, or None
            if fewer than min_observed_keypoints are visible.
        """
        import torch
        from scipy.interpolate import CubicSpline, interp1d

        # 1. Extract affine crop
        angle = det.angle if det.angle is not None else 0.0
        bx, by, bw, bh = det.bbox

        # Use true OBB dimensions from corner points when available.
        # Edge lengths of the OBB quadrilateral give the real oriented
        # width/height, while det.bbox is the axis-aligned bounding box
        # which over-estimates for rotated fish.
        if det.obb_points is not None:
            pts = np.asarray(det.obb_points, dtype=np.float64)  # (4, 2)
            edge0 = float(np.linalg.norm(pts[1] - pts[0]))
            edge1 = float(np.linalg.norm(pts[2] - pts[1]))
            obb_w_true = max(edge0, edge1)
            obb_h_true = min(edge0, edge1)
            center_xy: tuple[float, float] = (
                float(pts[:, 0].mean()),
                float(pts[:, 1].mean()),
            )
        else:
            obb_w_true = float(bw)
            obb_h_true = float(bh)
            if hasattr(det, "centroid") and det.centroid is not None:
                center_xy = (float(det.centroid[0]), float(det.centroid[1]))
            else:
                center_xy = (float(bx + bw / 2.0), float(by + bh / 2.0))

        affine = extract_affine_crop(
            frame=frame,
            center_xy=center_xy,
            angle_math_rad=angle,
            obb_w=obb_w_true,
            obb_h=obb_h_true,
            crop_size=self._crop_size,
            fit_obb=True,
            mask_background=True,
        )

        crop_img = affine.image
        crop_w, crop_h = self._crop_size  # (width, height)

        # Convert BGR -> RGB float32 tensor (1, 3, H, W) in [0, 1]
        if crop_img.ndim == 3 and crop_img.shape[2] == 3:
            rgb = crop_img[:, :, ::-1].astype(np.float32) / 255.0
        else:
            # Grayscale — expand to 3-channel
            gray = crop_img.astype(np.float32) / 255.0
            rgb = np.stack([gray, gray, gray], axis=2)

        img_tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
        )

        # 2. Run inference
        with torch.no_grad():
            output = self._model(img_tensor)  # (1, n_keypoints * 2) in [0,1]

        kp_norm = (
            output[0].cpu().numpy().reshape(self._n_keypoints, 2)
        )  # (K, 2) in [0,1]

        # 3. Confidence heuristic: 1 - 2 * max(|x-0.5|, |y-0.5|), clipped to [0,1]
        #    Points at crop centre => conf=1.0 (model confident position is interior)
        #    Points at crop boundary => conf=0.0 (model pushing point outside crop)
        x_dev = np.abs(kp_norm[:, 0] - 0.5)
        y_dev = np.abs(kp_norm[:, 1] - 0.5)
        conf = np.clip(1.0 - 2.0 * np.maximum(x_dev, y_dev), 0.0, 1.0)

        # 4. Apply confidence floor
        visible_mask = conf >= self._conf_floor
        n_visible = int(np.sum(visible_mask))

        # 5. Degenerate case: too few visible keypoints
        if n_visible < self._min_observed:
            logger.debug(
                "DirectPoseBackend: too few visible keypoints (%d < %d) "
                "for camera=%s frame=%d",
                n_visible,
                self._min_observed,
                cam_id,
                frame_idx,
            )
            return None

        # 6. Back-project visible keypoints to frame space
        #    Normalized [0,1] -> crop pixel coords -> frame coords via affine inverse.
        #    fit_obb=True in extract_affine_crop scaled the fish to fill the crop,
        #    so invert_affine_points correctly undoes rotation+scale+translation.
        visible_kp_norm = kp_norm[visible_mask]  # (V, 2)
        visible_t = self._t_values[visible_mask]  # (V,)
        visible_conf = conf[visible_mask]  # (V,)

        kp_crop_px = visible_kp_norm * np.array([crop_w, crop_h], dtype=np.float64)

        # 6a. Re-order keypoints by their crop-space x-coordinate.
        #
        #     During training the fish body was always oriented left-to-right
        #     in the crop (Nose at small x, Tailbase at large x).  Keypoints
        #     that were invisible in a training image are not reliably learned
        #     and may be predicted with x-values that violate the expected
        #     monotone order.  When CubicSpline connects non-monotone points
        #     parameterised by t=[0,0.2,...,1.0] it produces loops ("awareness
        #     ribbon" shape).
        #
        #     Sorting by crop-space x enforces the same ordering assumed during
        #     training.  For fish whose affine crop is oriented right-to-left
        #     the sorted order will be reversed relative to the anatomical
        #     labelling, but the resulting midline is still geometrically smooth;
        #     the orientation-resolution stage corrects head-tail direction.
        #
        #     t-values are re-assigned uniformly after sorting so that the
        #     spline parameter still runs from 0 (one end) to 1 (other end).
        if len(kp_crop_px) >= 2:
            x_order = np.argsort(kp_crop_px[:, 0])
            kp_crop_px = kp_crop_px[x_order]
            visible_conf = visible_conf[x_order]
            visible_t = np.linspace(0.0, 1.0, len(kp_crop_px))

        kp_frame = invert_affine_points(kp_crop_px, affine.M)  # (V, 2)

        # 7. Fit CubicSpline (requires at least 2 unique t-values for cubic)
        #    visible_t is already sorted and unique (linspace or pre-sorted
        #    anatomical values).  Keep the deduplication guard for safety.
        t_sorted = visible_t
        xy_sorted = kp_frame
        conf_sorted = visible_conf

        _, unique_idx = np.unique(t_sorted, return_index=True)
        if len(unique_idx) < 2:
            # Can't fit a spline with fewer than 2 unique t-values
            logger.debug(
                "DirectPoseBackend: insufficient unique t-values for camera=%s frame=%d",
                cam_id,
                frame_idx,
            )
            return None

        t_unique = t_sorted[unique_idx]
        xy_unique = xy_sorted[unique_idx]
        conf_unique = conf_sorted[unique_idx]

        # Choose CubicSpline or linear depending on point count
        if len(t_unique) >= 4:
            cs_x = CubicSpline(t_unique, xy_unique[:, 0], extrapolate=False)
            cs_y = CubicSpline(t_unique, xy_unique[:, 1], extrapolate=False)
        else:
            # Fall back to linear interpolation when < 4 unique points
            cs_x = interp1d(
                t_unique,
                xy_unique[:, 0],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            cs_y = interp1d(
                t_unique,
                xy_unique[:, 1],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

        # 8. Resample to n_points; NaN-pad outside observed arc-span
        t_eval = np.linspace(0.0, 1.0, self._n_points)
        t_min_obs = float(t_unique[0])
        t_max_obs = float(t_unique[-1])

        x_eval = cs_x(t_eval).astype(np.float32)
        y_eval = cs_y(t_eval).astype(np.float32)

        midline_pts = np.stack([x_eval, y_eval], axis=1)  # (n_points, 2)

        # NaN-pad points outside [t_min_obs, t_max_obs]
        outside_span = (t_eval < t_min_obs) | (t_eval > t_max_obs)
        midline_pts[outside_span] = np.nan

        # 9. Interpolate confidence along the spline
        conf_interp = interp1d(
            t_unique,
            conf_unique,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        midline_conf = conf_interp(t_eval).astype(np.float32)
        midline_conf[outside_span] = 0.0  # zero confidence for NaN regions

        # Clip to [0, 1] in case of interpolation overshoot
        midline_conf = np.clip(midline_conf, 0.0, 1.0)

        # 10. Build Midline2D
        fish_id = getattr(det, "fish_id", -1)
        if fish_id is None:
            fish_id = -1

        return Midline2D(
            points=midline_pts,
            half_widths=np.zeros(self._n_points, dtype=np.float32),
            fish_id=fish_id,
            camera_id=cam_id,
            frame_index=frame_idx,
            is_head_to_tail=True,
            point_confidence=midline_conf,
        )
