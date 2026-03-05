"""YOLO-OBB detection backend for the Detection stage.

Thin wrapper around the ultralytics YOLO model in OBB mode that loads the
model eagerly at construction time, converting oriented bounding box outputs
to :class:`~aquapose.core.types.detection.Detection` objects with
``angle`` and ``obb_points`` populated.

Angle convention contract (ONE conversion point):
    Ultralytics OBB outputs angles in radians, clockwise convention.
    This backend negates them to standard math convention (CCW, range
    ``[-pi, pi]``) before storing in :class:`Detection.angle`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aquapose.core.types.detection import Detection

__all__ = ["YOLOOBBBackend"]


class YOLOOBBBackend:
    """YOLO-OBB detection backend that produces oriented bounding boxes.

    Loads a YOLO OBB model eagerly at construction time. If the weights
    file does not exist a :class:`FileNotFoundError` is raised immediately
    rather than at run time.

    The angle convention conversion happens here (and only here): ultralytics
    outputs angles clockwise in radians; this backend negates them to standard
    math convention (counter-clockwise) before storing in
    :attr:`~aquapose.core.types.detection.Detection.angle`.

    Args:
        weights_path: Path to trained YOLO-OBB ``.pt`` weights file.
        conf_threshold: Minimum confidence score to keep a detection.
        iou_threshold: IoU threshold for non-max suppression.
        device: Device string (e.g. ``"cuda"``, ``"cpu"``). Accepted for
            interface consistency but YOLO auto-selects the device internally.

    Raises:
        FileNotFoundError: If *weights_path* does not point to an existing file.
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO-OBB weights not found: {weights_path}. "
                "Provide a valid path to a trained .pt weights file."
            )
        from ultralytics import YOLO  # lazy import — ultralytics may not be installed

        self._model = YOLO(str(weights_path))
        self._conf = conf_threshold
        self._iou = iou_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect fish in a single frame using OBB predictions.

        Runs YOLO-OBB inference on *frame* and converts each oriented bounding
        box to a :class:`~aquapose.core.types.detection.Detection`.  The
        ``angle`` field receives the standard math CCW angle (negated from
        ultralytics clockwise convention).  The ``obb_points`` field receives
        the four corner points of shape ``(4, 2)``.

        Args:
            frame: BGR image as uint8 array of shape ``(H, W, 3)``.

        Returns:
            List of :class:`~aquapose.core.types.detection.Detection` objects,
            one per detected fish, with ``angle`` and ``obb_points`` populated.
        """
        results = self._model.predict(
            frame, conf=self._conf, iou=self._iou, verbose=False
        )
        return self._parse_results(results)[0] if results else []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Detect fish in multiple frames using a single batched OBB prediction.

        Runs YOLO-OBB inference on all *frames* in a single ``predict()`` call,
        returning results in positional correspondence with input frames.

        Args:
            frames: List of BGR images as uint8 arrays of shape ``(H, W, 3)``.

        Returns:
            List of detection lists, one per input frame in positional order.
        """
        if not frames:
            return []
        results = self._model.predict(
            frames,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
            batch=len(frames),
        )
        return self._parse_results(results)

    def _parse_results(
        self,
        results: list,
    ) -> list[list[Detection]]:
        """Parse ultralytics OBB results into Detection objects.

        Each element in *results* corresponds to one input frame. The returned
        list has the same length as *results*, with each element being a list
        of :class:`Detection` objects for that frame.

        Args:
            results: Ultralytics Results list from ``model.predict()``.

        Returns:
            List of detection lists in positional correspondence with *results*.
        """
        all_detections: list[list[Detection]] = []

        for r in results:
            frame_dets: list[Detection] = []
            if r.obb is None:
                all_detections.append(frame_dets)
                continue

            # xywhr shape: (N, 5) — cx, cy, w, h, angle_cw_rad
            xywhr = r.obb.xywhr.cpu().numpy()
            # xyxyxyxy shape: (N, 4, 2) — four OBB corners
            corners_all = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()

            for i in range(len(xywhr)):
                _cx, _cy, w, h, angle_cw_rad = xywhr[i]

                # CRITICAL: negate to convert ultralytics CW -> standard math CCW
                detection_angle = -float(angle_cw_rad)

                # corners shape: (4, 2)
                corners = corners_all[i]  # (4, 2)

                # AABB fallback bbox from OBB corners
                x_min = int(corners[:, 0].min())
                y_min = int(corners[:, 1].min())
                x_max = int(corners[:, 0].max())
                y_max = int(corners[:, 1].max())
                aabb_w = x_max - x_min
                aabb_h = y_max - y_min
                bbox = (x_min, y_min, aabb_w, aabb_h)

                conf = float(confs[i])
                area = int(w * h)

                frame_dets.append(
                    Detection(
                        bbox=bbox,
                        mask=None,
                        area=area,
                        confidence=conf,
                        angle=detection_angle,
                        obb_points=corners,
                    )
                )

            all_detections.append(frame_dets)

        return all_detections
