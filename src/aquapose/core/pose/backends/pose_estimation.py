"""Pose estimation backend for the Pose stage.

Implements YOLO-pose inference to produce raw anatomical keypoints in
full-frame coordinates on Detection objects.  When OBB corner points are
available, crops are prepared by directly mapping the three OBB corners to
the crop canvas corners via ``cv2.getAffineTransform``, stretching the fish
to fill the entire canvas — matching the training data preparation.
Keypoints are back-projected to full-frame coordinates via the inverse affine
transform and written directly onto Detection objects.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from aquapose.core.pose.crop import extract_affine_crop, invert_affine_points
from aquapose.core.types.crop import AffineCrop
from aquapose.core.types.detection import Detection

__all__ = ["PoseEstimationBackend"]

logger = logging.getLogger(__name__)


class PoseEstimationBackend:
    """YOLO-pose keypoint regression backend for the Pose stage.

    Runs a YOLO-pose model on OBB-aligned affine crops, extracts raw
    keypoints, filters by confidence, and back-projects to full-frame
    coordinates.  Returns raw keypoints and confidences directly so that
    PoseStage can write them onto Detection objects.

    Args:
        weights_path: Path to YOLO-pose model weights file.  If ``None`` or the
            file does not exist, the backend operates in no-model mode and
            returns None keypoints for every detection.
        device: PyTorch device string passed to the YOLO model.  Default
            ``"cuda"``.
        n_keypoints: Number of anatomical keypoints expected from the model.
            Default 6 (nose, head, spine1, spine2, spine3, tail).
        confidence_floor: Minimum per-keypoint confidence to treat a keypoint
            as visible.  Default 0.3.
        min_observed_keypoints: Minimum number of visible keypoints required
            for a successful pose result.  Default 3.
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
        self._n_keypoints = n_keypoints
        self._confidence_floor = confidence_floor
        self._min_observed_keypoints = min_observed_keypoints
        self._crop_size = crop_size
        self._conf = conf

        # Expose public attributes for introspection / tests
        self.weights_path = weights_path
        self.device = device
        self.n_keypoints = n_keypoints
        self.confidence_floor = confidence_floor
        self.min_observed_keypoints = min_observed_keypoints
        self.crop_size = crop_size

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
                "PoseEstimationBackend: no weights_path supplied — all keypoints will be None."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_batch(
        self,
        crops: list[AffineCrop],
        metadata: list[tuple[Detection, str, int]],
    ) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
        """Run batched YOLO-pose inference on pre-extracted crops.

        Returns raw keypoints in full-frame coordinates and confidences
        for each crop.  PoseStage writes these directly onto Detection objects.

        Args:
            crops: Pre-extracted affine crops for all detections.
            metadata: Parallel list of ``(detection, camera_id, frame_idx)``
                tuples.

        Returns:
            List of ``(kpts_xy_fullframe, kpts_conf)`` tuples in the same
            order as *crops*.  Both arrays are ``None`` when keypoint
            extraction fails or the model is not loaded.  On success,
            ``kpts_xy_fullframe`` has shape ``(K, 2)`` float32 and
            ``kpts_conf`` has shape ``(K,)`` float32.
        """
        if self._model is None or not crops:
            return [(None, None)] * len(metadata)

        crop_images = [c.image for c in crops]
        results = self._model.predict(  # type: ignore[union-attr]
            crop_images, conf=self._conf, verbose=False, batch=len(crop_images)
        )

        output: list[tuple[np.ndarray | None, np.ndarray | None]] = []
        for result, crop, (_det, _cam_id, _frame_idx) in zip(
            results, crops, metadata, strict=True
        ):
            kpts_xy, kpts_conf = self._extract_keypoints([result])
            if kpts_xy is None or kpts_conf is None:
                output.append((None, None))
                continue

            visible_mask = kpts_conf >= self._confidence_floor
            n_visible = int(visible_mask.sum())

            if n_visible < self._min_observed_keypoints:
                output.append((None, None))
                continue

            # Back-project all keypoints (not just visible) to full-frame coords
            kpts_xy_frame = invert_affine_points(kpts_xy, crop.M).astype(np.float32)
            output.append((kpts_xy_frame, kpts_conf))

        return output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
            #   pts[2] = left-top      (center - long_vec - perp_vec)  <- LT
            #   pts[3] = left-bottom   (center - long_vec + perp_vec)  <- LB
            #
            # Fix: measure both sides and always map the LONG side to crop-width.
            lt, rt, lb = pts[2], pts[1], pts[3]
            side_w = float(np.linalg.norm(rt - lt))  # LT->RT ("w" direction)
            side_h = float(np.linalg.norm(lb - lt))  # LT->LB ("h" direction)
            if side_h > side_w:
                # Long axis is h: rotate corner assignment so long axis -> crop width
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
            Tuple of ``(kpts_xy, kpts_conf)`` each shape ``(K, 2)`` / ``(K,)``,
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
            # No confidence available -- treat all as maximally confident
            kpts_conf = np.ones(len(kpts_xy), dtype=np.float32)

        return kpts_xy, kpts_conf
