"""Pure-function evaluator for the detection stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aquapose.core.types.detection import Detection


@dataclass(frozen=True)
class DetectionMetrics:
    """Aggregated metrics for the detection stage.

    Attributes:
        total_detections: Sum of all detections across all frames and cameras.
        mean_confidence: Mean detection confidence across all detections.
        std_confidence: Standard deviation of detection confidence.
        mean_jitter: Mean absolute frame-to-frame delta in detection count per
            camera, averaged over cameras. A stable camera contributes 0.0;
            a flickering camera contributes a higher value.
        per_camera_counts: Total detections per camera.
    """

    total_detections: int
    mean_confidence: float
    std_confidence: float
    mean_jitter: float
    per_camera_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict of all metric fields.

        Returns:
            Dict with Python-native types (int, float, dict).
        """
        return {
            "total_detections": int(self.total_detections),
            "mean_confidence": float(self.mean_confidence),
            "std_confidence": float(self.std_confidence),
            "mean_jitter": float(self.mean_jitter),
            "per_camera_counts": {k: int(v) for k, v in self.per_camera_counts.items()},
        }


def _compute_jitter(counts_per_camera: dict[str, list[int]]) -> float:
    """Compute mean absolute frame-to-frame delta in detection count, averaged over cameras.

    Args:
        counts_per_camera: Maps camera_id to an ordered list of per-frame detection counts.

    Returns:
        Mean jitter across all cameras. Returns 0.0 if no cameras or all
        cameras have fewer than 2 frames of data.
    """
    jitters: list[float] = []
    for counts in counts_per_camera.values():
        if len(counts) < 2:
            continue
        arr = np.array(counts, dtype=float)
        jitters.append(float(np.mean(np.abs(np.diff(arr)))))
    if not jitters:
        return 0.0
    return float(np.mean(jitters))


def evaluate_detection(
    frames: list[dict[str, list[Detection]]],
) -> DetectionMetrics:
    """Compute detection-stage metrics from an ordered sequence of frames.

    Args:
        frames: List of per-frame dicts, each mapping camera_id to a list of
            Detection objects. List order corresponds to frame order.

    Returns:
        DetectionMetrics with aggregated counts, confidence statistics, jitter,
        and per-camera breakdown.
    """
    total_detections = 0
    all_confidences: list[float] = []
    per_camera_totals: dict[str, int] = {}
    per_camera_frame_counts: dict[str, list[int]] = {}

    for frame in frames:
        for cam_id, detections in frame.items():
            count = len(detections)
            total_detections += count

            # Accumulate confidences
            for det in detections:
                all_confidences.append(det.confidence)

            # Accumulate per-camera totals
            per_camera_totals[cam_id] = per_camera_totals.get(cam_id, 0) + count

            # Accumulate per-camera frame-level counts for jitter
            if cam_id not in per_camera_frame_counts:
                per_camera_frame_counts[cam_id] = []
            per_camera_frame_counts[cam_id].append(count)

    if all_confidences:
        conf_arr = np.array(all_confidences, dtype=float)
        mean_confidence = float(np.mean(conf_arr))
        std_confidence = float(np.std(conf_arr))
    else:
        mean_confidence = 0.0
        std_confidence = 0.0

    mean_jitter = _compute_jitter(per_camera_frame_counts)

    return DetectionMetrics(
        total_detections=total_detections,
        mean_confidence=mean_confidence,
        std_confidence=std_confidence,
        mean_jitter=mean_jitter,
        per_camera_counts=per_camera_totals,
    )
