"""MOG2 background-subtraction fish detector with morphological cleanup."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    """A single fish detection.

    Attributes:
        bbox: Bounding box as (x, y, w, h) in pixel coordinates.
        mask: Full-frame binary mask (uint8, 0/255) for this component only.
            ``None`` when the mask is equivalent to the bbox (e.g. YOLO
            detections without instance segmentation).
        area: Pixel area of the detected component.
        confidence: Detection confidence (always 1.0 for MOG2; placeholder
            for downstream compatibility).
    """

    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None
    area: int
    confidence: float


class MOG2Detector:
    """Fish detector using MOG2 background subtraction with morphological cleanup.

    Wraps ``cv2.BackgroundSubtractorMOG2`` with connected-component filtering
    to produce bounding boxes and per-component masks from video frames.
    Frames must be fed in temporal order per camera (caller responsibility).

    Args:
        history: Number of frames for background model history.
        var_threshold: Variance threshold for foreground classification.
        detect_shadows: Whether MOG2 should detect shadows (gray=127).
        min_area: Minimum connected-component area in pixels to keep.
        padding_fraction: Fraction of bbox dimension to add as padding.
        learning_rate: MOG2 learning rate. -1 (default) lets OpenCV auto-calculate
            as ``1/history``. Lower positive values (e.g. 0.0005) slow background
            adaptation, keeping stationary fish in the foreground longer.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 12,
        detect_shadows: bool = True,
        min_area: int = 200,
        padding_fraction: float = 0.15,
        learning_rate: float = -1,
    ) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._mog2.setShadowThreshold(200)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._min_area = min_area
        self._padding_fraction = padding_fraction
        self._learning_rate = learning_rate

    def apply(self, frame: np.ndarray) -> None:
        """Feed a frame to the background model without returning detections.

        Args:
            frame: BGR image as uint8 array of shape (H, W, 3).
        """
        self._mog2.apply(frame, learningRate=self._learning_rate)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect fish in a frame, returning bounding boxes and masks.

        Uses a two-stage approach to handle MOG2's shadow classification:

        1. **Blob finding** on foreground+shadow union (``fg_mask >= 127``),
           since fish often appear as shadow rather than foreground.
        2. **Watershed splitting** for merged blobs that contain multiple
           foreground (255) cores, using the foreground-only pixels as seeds
           and a distance transform to separate touching individuals.

        Args:
            frame: BGR image as uint8 array of shape (H, W, 3).

        Returns:
            List of :class:`Detection` objects, one per fish-sized blob.
        """
        fg_mask = self._mog2.apply(frame, learningRate=self._learning_rate)
        h_frame, w_frame = frame.shape[:2]

        # Stage 1: find blobs on foreground+shadow union
        combined = (fg_mask >= 127).astype(np.uint8) * 255
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self._kernel)

        # Foreground-only mask (for seed extraction in stage 2)
        fg_only = (fg_mask == 255).astype(np.uint8) * 255
        fg_only = cv2.morphologyEx(fg_only, cv2.MORPH_CLOSE, self._kernel)
        fg_only = cv2.morphologyEx(fg_only, cv2.MORPH_OPEN, self._kernel)

        # Connected components on the combined mask
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            combined, connectivity=8
        )

        detections: list[Detection] = []

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < self._min_area:
                continue

            blob_mask = (labels == label_id).astype(np.uint8) * 255

            # Stage 2: check for multiple foreground cores within this blob
            fg_in_blob = cv2.bitwise_and(fg_only, blob_mask)
            n_cores, core_labels = cv2.connectedComponents(fg_in_blob, connectivity=8)

            if n_cores <= 2:
                # 0 or 1 foreground core — emit as single detection
                detections.extend(self._make_detections(blob_mask, h_frame, w_frame))
            else:
                # Multiple foreground cores — watershed split
                split_masks = self._watershed_split(
                    blob_mask, fg_in_blob, core_labels, n_cores
                )
                for mask in split_masks:
                    detections.extend(self._make_detections(mask, h_frame, w_frame))

        return detections

    def _watershed_split(
        self,
        blob_mask: np.ndarray,
        fg_in_blob: np.ndarray,
        core_labels: np.ndarray,
        n_cores: int,
    ) -> list[np.ndarray]:
        """Split a merged blob using foreground cores as watershed seeds.

        Args:
            blob_mask: Binary mask of the combined blob (uint8, 0/255).
            fg_in_blob: Foreground-only pixels within the blob (uint8, 0/255).
            core_labels: Connected component labels for foreground cores.
            n_cores: Number of connected components (including background=0).

        Returns:
            List of binary masks, one per split region.
        """
        # Distance transform on the blob for watershed input
        dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)

        # Build marker image: each foreground core gets a unique label
        # (core_labels already has 0=bg, 1..n_cores-1 = cores)
        markers = core_labels.astype(np.int32)
        # Mark unknown region (blob but not a core) as 0
        # Mark definite background as a distinct label
        bg_label = n_cores
        markers[blob_mask == 0] = bg_label

        # Watershed needs a 3-channel image; use distance as grayscale
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dist_bgr = cv2.cvtColor(dist_norm, cv2.COLOR_GRAY2BGR)

        cv2.watershed(dist_bgr, markers)

        # Extract each core region (exclude background label and boundary=-1)
        masks = []
        for core_id in range(1, n_cores):
            region = np.zeros_like(blob_mask)
            region[markers == core_id] = 255
            # Intersect with the original blob to stay within bounds
            region = cv2.bitwise_and(region, blob_mask)
            if cv2.countNonZero(region) >= self._min_area:
                masks.append(region)

        # If watershed produced nothing useful, fall back to unsplit blob
        if not masks:
            masks = [blob_mask]

        return masks

    def _make_detections(
        self,
        mask: np.ndarray,
        h_frame: int,
        w_frame: int,
    ) -> list[Detection]:
        """Create Detection objects from a binary mask.

        Args:
            mask: Full-frame binary mask (uint8, 0/255).
            h_frame: Frame height.
            w_frame: Frame width.

        Returns:
            List containing a single Detection, or empty if below min_area.
        """
        area = int(cv2.countNonZero(mask))
        if area < self._min_area:
            return []

        # Bounding box from mask
        coords = cv2.findNonZero(mask)
        if coords is None:
            return []
        cx, cy, cw, ch = cv2.boundingRect(coords)

        # Pad bbox
        pad_x = int(cw * self._padding_fraction)
        pad_y = int(ch * self._padding_fraction)
        x1 = max(0, cx - pad_x)
        y1 = max(0, cy - pad_y)
        x2 = min(w_frame, cx + cw + pad_x)
        y2 = min(h_frame, cy + ch + pad_y)

        bbox = (x1, y1, x2 - x1, y2 - y1)

        return [Detection(bbox=bbox, mask=mask, area=area, confidence=1.0)]

    def warm_up(self, frames: Sequence[np.ndarray]) -> None:
        """Feed frames to build a stable background model before detection.

        Args:
            frames: Sequence of BGR images to feed through :meth:`apply`.
        """
        for frame in frames:
            self.apply(frame)


class YOLODetector:
    """Fish detector using YOLOv8 object detection.

    Wraps a trained ``ultralytics.YOLO`` model to produce :class:`Detection`
    objects interchangeable with :class:`MOG2Detector` output.  The
    ``ultralytics`` package is lazily imported inside ``__init__`` to avoid
    import failures when it is not installed.

    Args:
        model_path: Path to trained YOLOv8 ``.pt`` weights file.
        conf_threshold: Minimum confidence score to keep a detection.
        iou_threshold: IoU threshold for non-max suppression. Lower values
            suppress more overlapping boxes (ultralytics default is 0.7).
        padding_fraction: Fraction of bbox dimension to add as symmetric
            padding (clamped to frame bounds).
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        padding_fraction: float = 0.15,
    ) -> None:
        from ultralytics import YOLO  # lazy import — ultralytics may not be installed

        self._model = YOLO(str(model_path))
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._padding_fraction = padding_fraction

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect fish in a frame, returning bounding boxes.

        Runs YOLOv8 inference on *frame* and converts each predicted bounding
        box to a :class:`Detection`.  The ``mask`` field is ``None`` since
        the detection is fully described by the bbox.

        Args:
            frame: BGR image as uint8 array of shape ``(H, W, 3)``.

        Returns:
            List of :class:`Detection` objects, one per detected fish.
        """
        h_frame, w_frame = frame.shape[:2]
        results = self._model.predict(
            frame, conf=self._conf, iou=self._iou, verbose=False
        )
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # Apply symmetric padding, clamped to frame bounds
                bw, bh = x2 - x1, y2 - y1
                pad_x = bw * self._padding_fraction
                pad_y = bh * self._padding_fraction
                rx1 = max(0, int(x1 - pad_x))
                ry1 = max(0, int(y1 - pad_y))
                rx2 = min(w_frame, int(x2 + pad_x))
                ry2 = min(h_frame, int(y2 + pad_y))
                bbox = (rx1, ry1, rx2 - rx1, ry2 - ry1)
                area = (rx2 - rx1) * (ry2 - ry1)
                detections.append(
                    Detection(bbox=bbox, mask=None, area=area, confidence=conf)
                )
        return detections


def make_detector(kind: str, **kwargs: Any) -> MOG2Detector | YOLODetector:
    """Create a fish detector by name.

    Args:
        kind: Detector type — ``"mog2"`` or ``"yolo"``.
        **kwargs: Forwarded to the detector constructor.  For ``"yolo"``,
            ``model_path`` is required.

    Returns:
        Configured detector instance.

    Raises:
        ValueError: If *kind* is not recognized.
    """
    if kind == "mog2":
        return MOG2Detector(**kwargs)
    if kind == "yolo":
        return YOLODetector(**kwargs)
    raise ValueError(f"Unknown detector kind: {kind!r}")
