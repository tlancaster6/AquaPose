"""Pose estimation backend for the Midline stage.

Implements YOLO-pose inference with confidence-filtered keypoint spline
interpolation to produce Midline2D objects in full-frame coordinates.  When
OBB corner points are available, crops are prepared by directly mapping the
three OBB corners to the crop canvas corners via ``cv2.getAffineTransform``,
stretching the fish to fill the entire canvas — matching the training data
preparation.  Interpolated keypoints are back-projected to full-frame
coordinates via the inverse affine transform.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.interpolate import interp1d

from aquapose.core.midline.crop import extract_affine_crop, invert_affine_points
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.core.types.crop import AffineCrop
from aquapose.core.types.detection import Detection
from aquapose.core.types.midline import Midline2D

__all__ = ["PoseEstimationBackend"]

logger = logging.getLogger(__name__)


def _keypoints_to_midline(
    kpts_xy: np.ndarray,
    t_values: np.ndarray,
    confidences: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate visible keypoints to a dense midline via linear spline.

    Args:
        kpts_xy: Keypoint coordinates in crop space, shape ``(K, 2)``, float32.
        t_values: Arc-fraction parameter values for each keypoint, shape ``(K,)``.
            Values should span a subset of ``[0, 1]``.
        confidences: Per-keypoint confidence scores, shape ``(K,)``.
        n_points: Number of output midline points.

    Returns:
        xy: Resampled midline coordinates, shape ``(n_points, 2)``, float32.
        conf: Interpolated confidence values at each output point, shape
            ``(n_points,)``, float32.
    """
    t_eval = np.linspace(0.0, 1.0, n_points)

    interp_x = interp1d(
        t_values,
        kpts_xy[:, 0],
        kind="linear",
        fill_value="extrapolate",
    )
    interp_y = interp1d(
        t_values,
        kpts_xy[:, 1],
        kind="linear",
        fill_value="extrapolate",
    )
    interp_c = interp1d(
        t_values,
        confidences,
        kind="linear",
        bounds_error=False,
        fill_value=(float(confidences[0]), float(confidences[-1])),
    )

    x_out = interp_x(t_eval).astype(np.float32)
    y_out = interp_y(t_eval).astype(np.float32)
    conf_out = interp_c(t_eval).astype(np.float32)

    xy = np.stack([x_out, y_out], axis=1)
    return xy, conf_out


class PoseEstimationBackend:
    """YOLO-pose keypoint regression backend for the Midline stage.

    Runs a YOLO-pose model on OBB-aligned affine crops, extracts keypoints,
    filters by confidence, and interpolates visible keypoints to a dense
    Midline2D via linear spline in full-frame coordinates.

    Args:
        weights_path: Path to YOLO-pose model weights file.  If ``None`` or the
            file does not exist, the backend operates in no-model mode and
            returns ``midline=None`` for every detection.
        device: PyTorch device string passed to the YOLO model.  Default
            ``"cuda"``.
        n_points: Number of midline points to produce per detection.  Default 15.
        n_keypoints: Number of anatomical keypoints expected from the model.
            Default 6 (nose, head, spine1, spine2, spine3, tail).
        keypoint_t_values: Per-keypoint arc-fraction values in ``[0, 1]``.
            Required.  Raises ``ValueError`` if ``None``.
        confidence_floor: Minimum per-keypoint confidence to treat a keypoint
            as visible.  Default 0.3.
        min_observed_keypoints: Minimum number of visible keypoints required to
            fit the spline.  Default 3.
        crop_size: Output crop canvas size as ``(width, height)`` in pixels.
            Default ``(128, 64)``.
        conf: YOLO detection confidence threshold for model.predict().  Default 0.5.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cuda",
        n_points: int = 15,
        n_keypoints: int = 6,
        keypoint_t_values: list[float] | None = None,
        confidence_floor: float = 0.3,
        min_observed_keypoints: int = 3,
        crop_size: tuple[int, int] = (128, 64),
        conf: float = 0.5,
        **kwargs: object,
    ) -> None:
        self._device = device
        self._n_points = n_points
        self._n_keypoints = n_keypoints
        self._confidence_floor = confidence_floor
        self._min_observed_keypoints = min_observed_keypoints
        self._crop_size = crop_size
        self._conf = conf

        # Expose public attributes for introspection / tests
        self.weights_path = weights_path
        self.device = device
        self.n_points = n_points
        self.n_keypoints = n_keypoints
        self.confidence_floor = confidence_floor
        self.min_observed_keypoints = min_observed_keypoints
        self.crop_size = crop_size

        if keypoint_t_values is not None:
            self._keypoint_t_values = np.asarray(keypoint_t_values, dtype=np.float32)
        else:
            raise ValueError(
                "keypoint_t_values is None. Run: "
                "aquapose prep calibrate-keypoints --annotations <json> --config <yaml>"
            )
        self.keypoint_t_values = self._keypoint_t_values

        if weights_path is not None:
            from pathlib import Path

            if not Path(weights_path).exists():
                raise FileNotFoundError(
                    f"PoseEstimationBackend weights not found: {weights_path}. "
                    "Provide a valid path to a trained .pt weights file."
                )
            from ultralytics import YOLO

            self._model: object | None = YOLO(str(weights_path))
            logger.info("PoseEstimationBackend: loaded model from %s", weights_path)
        else:
            self._model = None
            logger.warning(
                "PoseEstimationBackend: no weights_path supplied — all midlines will be None."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_idx: int,
        frame_dets: dict[str, list[Detection]],
        frames: dict[str, np.ndarray],
        camera_ids: list[str],
    ) -> dict[str, list[AnnotatedDetection]]:
        """Run YOLO-pose inference and spline-interpolate keypoints for all detections.

        For each camera and each detection, extracts an OBB-aligned affine crop,
        runs YOLO-pose, filters keypoints by confidence, and interpolates visible
        keypoints to a dense midline in full-frame coordinates.  All failure cases
        (no model, no keypoints, too few visible keypoints) return ``midline=None``.

        Args:
            frame_idx: Current frame index stored in output metadata.
            frame_dets: Per-camera detection lists from Stage 1.
            frames: Per-camera undistorted frame images (BGR uint8).
            camera_ids: Active camera identifiers to process.

        Returns:
            Per-camera list of :class:`~aquapose.core.midline.types.AnnotatedDetection`
            objects, one per input detection.  ``midline`` is ``None`` when
            extraction failed.
        """
        annotated: dict[str, list[AnnotatedDetection]] = {}

        for cam_id in camera_ids:
            cam_dets = frame_dets.get(cam_id, [])
            frame = frames.get(cam_id)
            cam_results: list[AnnotatedDetection] = []

            for det in cam_dets:
                ann = self._process_detection(det, frame, cam_id, frame_idx)
                cam_results.append(ann)

            annotated[cam_id] = cam_results

        return annotated

    def process_batch(
        self,
        crops: list[AffineCrop],
        metadata: list[tuple[Detection, str, int]],
    ) -> list[AnnotatedDetection]:
        """Run batched YOLO-pose inference on pre-extracted crops.

        Accepts a list of crops and corresponding metadata tuples, runs a
        single batched ``model.predict()`` call, and returns one
        :class:`AnnotatedDetection` per input in positional correspondence.

        Args:
            crops: Pre-extracted affine crops for all detections.
            metadata: Parallel list of ``(detection, camera_id, frame_idx)``
                tuples for building output objects.

        Returns:
            List of :class:`AnnotatedDetection` in the same order as *crops*.
            Entries where keypoint extraction or interpolation fails have
            ``midline=None``.
        """
        if self._model is None or not crops:
            return [
                AnnotatedDetection(
                    detection=det,
                    mask=None,
                    crop_region=None,
                    midline=None,
                    camera_id=cam_id,
                    frame_index=frame_idx,
                )
                for det, cam_id, frame_idx in metadata
            ]

        crop_images = [c.image for c in crops]
        results = self._model.predict(  # type: ignore[union-attr]
            crop_images, conf=self._conf, verbose=False, batch=len(crop_images)
        )

        output: list[AnnotatedDetection] = []
        for result, crop, (det, cam_id, frame_idx) in zip(
            results, crops, metadata, strict=True
        ):
            _null = AnnotatedDetection(
                detection=det,
                mask=None,
                crop_region=None,
                midline=None,
                camera_id=cam_id,
                frame_index=frame_idx,
            )

            kpts_xy, kpts_conf = self._extract_keypoints([result])
            if kpts_xy is None or kpts_conf is None:
                output.append(_null)
                continue

            visible_mask = kpts_conf >= self._confidence_floor
            n_visible = int(visible_mask.sum())

            if n_visible < self._min_observed_keypoints:
                output.append(_null)
                continue

            visible_kpts = kpts_xy[visible_mask]
            visible_t = self._keypoint_t_values[visible_mask]
            visible_conf = kpts_conf[visible_mask]

            try:
                xy_crop, conf_resampled = _keypoints_to_midline(
                    visible_kpts, visible_t, visible_conf, self._n_points
                )
            except Exception:
                logger.debug(
                    "PoseEstimationBackend: keypoint interpolation failed",
                    exc_info=True,
                )
                output.append(_null)
                continue

            xy_frame = invert_affine_points(xy_crop, crop.M).astype(np.float32)

            midline = Midline2D(
                points=xy_frame,
                half_widths=np.zeros(self._n_points, dtype=np.float32),
                fish_id=0,
                camera_id=cam_id,
                frame_index=frame_idx,
                is_head_to_tail=True,
                point_confidence=conf_resampled,
            )

            output.append(
                AnnotatedDetection(
                    detection=det,
                    mask=None,
                    crop_region=None,
                    midline=midline,
                    camera_id=cam_id,
                    frame_index=frame_idx,
                )
            )

        return output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_detection(
        self,
        det: Detection,
        frame: np.ndarray | None,
        cam_id: str,
        frame_idx: int,
    ) -> AnnotatedDetection:
        """Process a single detection, returning an AnnotatedDetection.

        Args:
            det: Detection object with OBB metadata.
            frame: Full-frame image (BGR uint8) or None.
            cam_id: Camera identifier.
            frame_idx: Current frame index.

        Returns:
            AnnotatedDetection with midline populated on success, else None.
        """
        _null = AnnotatedDetection(
            detection=det,
            mask=None,
            crop_region=None,
            midline=None,
            camera_id=cam_id,
            frame_index=frame_idx,
        )

        if self._model is None or frame is None:
            return _null

        try:
            crop = self._extract_crop(det, frame)
        except Exception:
            logger.debug("PoseEstimationBackend: crop extraction failed", exc_info=True)
            return _null

        try:
            results = self._model.predict(  # type: ignore[union-attr]
                crop.image, conf=self._conf, verbose=False
            )
        except Exception:
            logger.debug("PoseEstimationBackend: YOLO inference failed", exc_info=True)
            return _null

        kpts_xy, kpts_conf = self._extract_keypoints(results)
        if kpts_xy is None or kpts_conf is None:
            return _null

        visible_mask = kpts_conf >= self._confidence_floor
        n_visible = int(visible_mask.sum())

        if n_visible < self._min_observed_keypoints:
            return _null

        visible_kpts = kpts_xy[visible_mask]
        visible_t = self._keypoint_t_values[visible_mask]
        visible_conf = kpts_conf[visible_mask]

        try:
            xy_crop, conf_resampled = _keypoints_to_midline(
                visible_kpts, visible_t, visible_conf, self._n_points
            )
        except Exception:
            logger.debug(
                "PoseEstimationBackend: keypoint interpolation failed", exc_info=True
            )
            return _null

        xy_frame = invert_affine_points(xy_crop, crop.M).astype(np.float32)

        midline = Midline2D(
            points=xy_frame,
            half_widths=np.zeros(self._n_points, dtype=np.float32),
            fish_id=0,
            camera_id=cam_id,
            frame_index=frame_idx,
            is_head_to_tail=True,
            point_confidence=conf_resampled,
        )

        return AnnotatedDetection(
            detection=det,
            mask=None,
            crop_region=None,
            midline=midline,
            camera_id=cam_id,
            frame_index=frame_idx,
        )

    def _extract_crop(
        self,
        det: Detection,
        frame: np.ndarray,
    ) -> AffineCrop:
        """Extract an OBB-aligned affine crop for a detection.

        When OBB corner points are available, uses ``cv2.getAffineTransform``
        to map TL→(0,0), TR→(W-1,0), BL→(0,H-1), stretching the OBB to
        fill the entire crop canvas — matching training data preparation.

        Ultralytics ``obb_points`` (``xyxyxyxy``) are ordered as
        [right-bottom, right-top, left-top, left-bottom] (i.e. ``pts[2]``
        is the true top-left corner, not ``pts[0]``).

        Falls back to ``extract_affine_crop`` at native scale when OBB corner
        points are absent.

        Args:
            det: Detection with optional OBB fields.
            frame: Full-frame BGR image.

        Returns:
            AffineCrop object.
        """
        crop_w, crop_h = self._crop_size

        if det.obb_points is not None and len(det.obb_points) >= 4:
            pts = np.asarray(det.obb_points, dtype=np.float32)
            # Ultralytics xyxyxyxy corner order (from xywhr2xyxyxyxy):
            #   pts[0] = right-bottom  (center + long_vec + perp_vec)
            #   pts[1] = right-top     (center + long_vec - perp_vec)
            #   pts[2] = left-top      (center - long_vec - perp_vec)  ← LT
            #   pts[3] = left-bottom   (center - long_vec + perp_vec)  ← LB
            #
            # Ultralytics does NOT guarantee w >= h.  When the fish long axis
            # happens to be in the "h" direction (LT→LB longer than LT→RT),
            # naively mapping LT→RT to crop-width produces a 90-degree-rotated,
            # aspect-ratio-flipped crop that does not match training.
            #
            # Fix: measure both sides and always map the LONG side to crop-width.
            #   Normal case  (LT→RT is long): src = [LT, RT, LB]
            #   Rotated case (LT→LB is long): rotate 90° → src = [LB, LT, RB]
            #     so that LB→LT (= long axis) maps to crop width, LB→RB maps to height.
            lt, rt, lb = pts[2], pts[1], pts[3]
            side_w = float(np.linalg.norm(rt - lt))  # LT→RT ("w" direction)
            side_h = float(np.linalg.norm(lb - lt))  # LT→LB ("h" direction)
            if side_h > side_w:
                # Long axis is h: rotate corner assignment so long axis → crop width
                rb = pts[0]
                src = np.array([lb, lt, rb], dtype=np.float32)
            else:
                src = np.array([lt, rt, lb], dtype=np.float32)
            dst = np.array([[0, 0], [crop_w - 1, 0], [0, crop_h - 1]], dtype=np.float32)
            M = cv2.getAffineTransform(src, dst)
            crop_image = cv2.warpAffine(
                frame,
                M,
                (crop_w, crop_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return AffineCrop(
                image=crop_image,
                M=M.astype(np.float64),
                crop_size=self._crop_size,
                frame_shape=frame.shape[:2],
            )

        # Fallback for non-OBB detectors
        x, y, w, h = det.bbox
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
        angle = float(det.angle) if det.angle is not None else 0.0

        if det.obb_points is not None and len(det.obb_points) >= 2:
            pts = np.asarray(det.obb_points, dtype=np.float32)
            side0 = float(np.linalg.norm(pts[1] - pts[0]))
            side1 = float(np.linalg.norm(pts[2] - pts[1]))
            obb_w, obb_h = side0, side1
        else:
            obb_w, obb_h = float(w), float(h)

        return extract_affine_crop(
            frame,
            center_xy=(cx, cy),
            angle_math_rad=angle,
            obb_w=obb_w,
            obb_h=obb_h,
            crop_size=self._crop_size,
        )

    def _extract_keypoints(
        self,
        results: list[object],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract keypoint coordinates and confidences from YOLO results.

        Uses pixel coordinates (``.xy``) not normalised (``.xyn``).
        All tensor-to-numpy conversions use ``.cpu().numpy()`` to support CUDA.

        Args:
            results: Ultralytics Results list returned by model.predict().

        Returns:
            Tuple of ``(kpts_xy, kpts_conf)`` each shape ``(K,)`` / ``(K, 2)``,
            or ``(None, None)`` when keypoints are unavailable.
        """
        if not results:
            return None, None
        res = results[0]
        if res.keypoints is None:
            return None, None
        kp = res.keypoints
        if len(kp.xy) == 0:
            return None, None

        kpts_xy = kp.xy[0].cpu().numpy().astype(np.float32)  # (K, 2) pixel coords
        if kp.conf is not None and len(kp.conf) > 0:
            kpts_conf = kp.conf[0].cpu().numpy().astype(np.float32)  # (K,)
        else:
            # No confidence available — treat all as maximally confident
            kpts_conf = np.ones(len(kpts_xy), dtype=np.float32)

        return kpts_xy, kpts_conf
