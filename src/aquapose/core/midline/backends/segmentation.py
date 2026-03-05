"""Segmentation backend for the Midline stage.

Implements YOLO-seg inference with mask skeletonization to produce Midline2D
objects in full-frame coordinates.  When OBB corner points are available (from
a YOLO-OBB detector), crops are prepared by directly mapping the three OBB
corners to the crop canvas corners via ``cv2.getAffineTransform``, stretching
the fish to fill the entire canvas — matching the training data preparation in
``build_yolo_training_data.py``.  Skeleton points are back-projected to
full-frame coordinates via the inverse affine transform.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from aquapose.core.midline.crop import extract_affine_crop, invert_affine_points
from aquapose.core.midline.midline import (
    _adaptive_smooth,
    _longest_path_bfs,
    _resample_arc_length,
    _skeleton_and_widths,
)
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.core.types.crop import AffineCrop
from aquapose.core.types.detection import Detection
from aquapose.core.types.midline import Midline2D

__all__ = ["SegmentationBackend"]

logger = logging.getLogger(__name__)


class SegmentationBackend:
    """YOLO-seg segmentation backend for the Midline stage.

    Runs a YOLO-seg model on OBB-aligned affine crops, extracts binary masks,
    skeletonizes them, and back-projects the skeleton to full-frame coordinates
    as Midline2D objects.

    Args:
        weights_path: Path to YOLO-seg model weights file.  If ``None`` or the
            file does not exist, the backend operates in no-model mode and
            returns ``midline=None`` for every detection.
        confidence_threshold: Minimum confidence threshold passed to YOLO
            prediction.  Default 0.5.
        n_points: Number of midline points to produce per detection.
            Default 15.
        min_area: Minimum mask area in pixels to attempt skeletonization.
            Masks below this threshold are discarded (midline=None).  Default 300.
        device: PyTorch device string passed to the YOLO model.  Default
            ``"cuda"``.
        crop_size: Output crop canvas size as ``(width, height)`` in pixels.
            Default ``(128, 64)``.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        confidence_threshold: float = 0.5,
        n_points: int = 15,
        min_area: int = 300,
        device: str = "cuda",
        crop_size: tuple[int, int] = (128, 64),
        **kwargs: object,
    ) -> None:
        self._conf = confidence_threshold
        self._n_points = n_points
        self._min_area = min_area
        self._device = device
        self._crop_size = crop_size

        # Expose public attributes for introspection / tests
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.n_points = n_points
        self.min_area = min_area
        self.device = device
        self.crop_size = crop_size

        if weights_path is not None:
            from pathlib import Path

            if Path(weights_path).exists():
                from ultralytics import YOLO

                self._model: object | None = YOLO(str(weights_path))
                logger.info("SegmentationBackend: loaded model from %s", weights_path)
            else:
                self._model = None
                logger.warning(
                    "SegmentationBackend: weights_path '%s' does not exist — "
                    "all midlines will be None.",
                    weights_path,
                )
        else:
            self._model = None
            logger.warning(
                "SegmentationBackend: no weights_path supplied — all midlines will be None."
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
        """Run YOLO-seg inference and skeletonize masks for all detections.

        For each camera and each detection, extracts an OBB-aligned affine crop,
        runs YOLO-seg, skeletonizes the binary mask, and back-projects the
        skeleton to full-frame coordinates.  All failure cases (no model, no
        mask, mask too small, empty skeleton) return ``midline=None``.

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
        """Run batched YOLO-seg inference on pre-extracted crops.

        Accepts a list of crops and corresponding metadata tuples, runs a
        single batched ``model.predict()`` call, and returns one
        :class:`AnnotatedDetection` per input in positional correspondence.

        Args:
            crops: Pre-extracted affine crops for all detections.
            metadata: Parallel list of ``(detection, camera_id, frame_idx)``
                tuples for building output objects.

        Returns:
            List of :class:`AnnotatedDetection` in the same order as *crops*.
            Entries where mask extraction or skeletonization fails have
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

            mask_np = self._extract_mask([result], crop.image.shape)
            if mask_np is None:
                output.append(_null)
                continue

            if np.count_nonzero(mask_np) < self._min_area:
                output.append(_null)
                continue

            midline = self._skeletonize_and_project(mask_np, crop.M, cam_id, frame_idx)
            if midline is None:
                output.append(_null)
                continue

            output.append(
                AnnotatedDetection(
                    detection=det,
                    mask=mask_np,
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
            logger.debug("SegmentationBackend: crop extraction failed", exc_info=True)
            return _null

        try:
            results = self._model.predict(  # type: ignore[union-attr]
                crop.image, conf=self._conf, verbose=False
            )
        except Exception:
            logger.debug("SegmentationBackend: YOLO inference failed", exc_info=True)
            return _null

        mask_np = self._extract_mask(results, crop.image.shape)
        if mask_np is None:
            return _null

        # Area check
        if np.count_nonzero(mask_np) < self._min_area:
            return _null

        midline = self._skeletonize_and_project(mask_np, crop.M, cam_id, frame_idx)
        if midline is None:
            return _null

        return AnnotatedDetection(
            detection=det,
            mask=mask_np,
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

        When OBB corner points are available (from a YOLO-OBB detector),
        uses ``cv2.getAffineTransform`` to map TL->(0,0), TR->(W-1,0),
        BL->(0,H-1), stretching the OBB to fill the entire crop canvas —
        matching how the seg training data was prepared.

        Falls back to ``extract_affine_crop`` at native scale when OBB
        corner points are absent.

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
        return extract_affine_crop(
            frame,
            center_xy=(cx, cy),
            angle_math_rad=angle,
            obb_w=float(w),
            obb_h=float(h),
            crop_size=self._crop_size,
        )

    def _extract_mask(
        self,
        results: list[object],
        crop_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """Extract and resize the first YOLO-seg mask.

        Args:
            results: Ultralytics Results list returned by model.predict().
            crop_shape: Expected output shape ``(H, W, ...)``.

        Returns:
            Binary uint8 mask (0/255) sized to ``(crop_shape[0], crop_shape[1])``,
            or None when no mask is available.
        """
        if not results:
            return None
        res = results[0]
        if res.masks is None or len(res.masks.data) == 0:
            return None

        mask_tensor = res.masks.data[0]
        mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255

        target_h, target_w = crop_shape[:2]
        if mask_np.shape[0] != target_h or mask_np.shape[1] != target_w:
            mask_np = cv2.resize(
                mask_np,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

        return mask_np

    def _skeletonize_and_project(
        self,
        mask_np: np.ndarray,
        M: np.ndarray,
        cam_id: str,
        frame_idx: int,
    ) -> Midline2D | None:
        """Skeletonize a binary mask and back-project to frame coordinates.

        Args:
            mask_np: Binary uint8 mask (0/255) in crop space.
            M: Affine transform matrix ``(2, 3)`` mapping frame → crop.
            cam_id: Camera identifier for Midline2D metadata.
            frame_idx: Frame index for Midline2D metadata.

        Returns:
            Midline2D in full-frame coordinates, or None on failure.
        """
        smooth = _adaptive_smooth(mask_np)
        skeleton_bool, dt = _skeleton_and_widths(smooth)

        if int(np.sum(skeleton_bool)) < self._n_points:
            return None

        path_yx = _longest_path_bfs(skeleton_bool)
        if not path_yx:
            return None

        xy_crop, hw_crop = _resample_arc_length(path_yx, dt, self._n_points)

        # Back-project from crop space to full-frame
        xy_frame = invert_affine_points(xy_crop, M).astype(np.float32)

        # Scale half-widths by inverse of the affine scale factor
        scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
        hw_frame = (hw_crop / max(scale, 1e-6)).astype(np.float32)

        return Midline2D(
            points=xy_frame,
            half_widths=hw_frame,
            fish_id=0,
            camera_id=cam_id,
            frame_index=frame_idx,
            is_head_to_tail=False,
            point_confidence=None,
        )
