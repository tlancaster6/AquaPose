"""OC-SORT tracker wrapper that fully isolates the boxmot dependency.

All imports from boxmot are confined to this module. Downstream code (TrackingStage
and all consumers) only sees the project's own Tracklet2D data contract.

Design notes:
- Input: list of Detection objects per frame (xywh bbox + confidence)
- Output: list of Tracklet2D frozen dataclasses per camera batch
- Coasting: when a confirmed track has no matched detection for a frame, its
  Kalman-predicted position is recorded with frame_status="coasted"
- Confirmation: only tracks that have been matched at least min_hits times
  appear in get_tracklets() output
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aquapose.core.tracking.types import Tracklet2D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private builder type
# ---------------------------------------------------------------------------


@dataclass
class _TrackletBuilder:
    """Mutable accumulator for one track's per-frame data.

    Converted to a frozen Tracklet2D by OcSortTracker.get_tracklets().

    Attributes:
        camera_id: Camera this tracklet belongs to.
        track_id: Local track ID (monotonically assigned by wrapper).
        frames: List of frame indices.
        centroids: List of (u, v) pixel centroids.
        bboxes: List of (x, y, w, h) bounding boxes.
        frame_status: List of "detected" or "coasted" per frame.
        detected_count: Number of frames with status "detected".
        active: Whether this track is still being updated.
    """

    camera_id: str
    track_id: int
    frames: list[int] = field(default_factory=list)
    centroids: list[tuple[float, float]] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    frame_status: list[str] = field(default_factory=list)
    detected_count: int = 0
    active: bool = True

    def add_frame(
        self,
        frame_idx: int,
        bbox_xyxy: tuple[float, float, float, float],
        status: str,
    ) -> None:
        """Append one frame of data.

        Args:
            frame_idx: Frame index.
            bbox_xyxy: Bounding box as (x1, y1, x2, y2).
            status: "detected" or "coasted".
        """
        x1, y1, x2, y2 = bbox_xyxy
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        self.frames.append(frame_idx)
        self.centroids.append((cx, cy))
        self.bboxes.append((x1, y1, w, h))
        self.frame_status.append(status)
        if status == "detected":
            self.detected_count += 1

    def to_tracklet2d(self) -> Tracklet2D:
        """Convert to a frozen Tracklet2D.

        Returns:
            Immutable Tracklet2D with tuple sequence fields.
        """
        return Tracklet2D(
            camera_id=self.camera_id,
            track_id=self.track_id,
            frames=tuple(self.frames),
            centroids=tuple(self.centroids),
            bboxes=tuple(self.bboxes),
            frame_status=tuple(self.frame_status),
        )


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class OcSortTracker:
    """Per-camera OC-SORT tracker wrapping boxmot, producing Tracklet2D output.

    This class is the sole location in the codebase that imports from boxmot.
    All downstream code interacts only with Tracklet2D frozen dataclasses.

    Args:
        camera_id: Camera identifier; stored on every Tracklet2D produced.
        max_age: Maximum frames to coast (Kalman predict with no observation)
            before dropping a track. Maps to boxmot ``max_age``.
        min_hits: Minimum number of matched detection frames before a track
            is "confirmed" and included in ``get_tracklets()`` output. Maps
            to boxmot ``min_hits``.
        iou_threshold: IoU threshold for matching detections to tracks.
        det_thresh: Minimum detection confidence to pass to tracker.
    """

    def __init__(
        self,
        camera_id: str,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        det_thresh: float = 0.3,
    ) -> None:
        self.camera_id = camera_id
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._det_thresh = det_thresh

        # Local ID management: boxmot assigns its own IDs; we map them to
        # clean monotonically increasing local IDs.
        self._next_local_id: int = 0
        self._boxmot_id_to_local: dict[int, int] = {}

        # Per-local-track builder dict
        self._builders: dict[int, _TrackletBuilder] = {}

        # Set of local track IDs that are still reported by boxmot as active
        self._active_local_ids: set[int] = set()

        # Instantiate the boxmot OcSort tracker (deferred import)
        self._tracker = self._create_tracker()

    def _create_tracker(self) -> Any:
        """Create and return a fresh boxmot OcSort tracker instance.

        Returns:
            boxmot OcSort tracker instance.
        """
        from boxmot import OcSort

        return OcSort(
            det_thresh=self._det_thresh,
            max_age=self._max_age,
            min_hits=self._min_hits,
            iou_threshold=self._iou_threshold,
        )

    def _assign_local_id(self, boxmot_id: int) -> int:
        """Return the local track ID for a boxmot track ID, creating one if needed.

        Args:
            boxmot_id: Track ID from boxmot output array (1-indexed).

        Returns:
            Local track ID (0-indexed, monotonically assigned).
        """
        if boxmot_id not in self._boxmot_id_to_local:
            local_id = self._next_local_id
            self._next_local_id += 1
            self._boxmot_id_to_local[boxmot_id] = local_id
            self._builders[local_id] = _TrackletBuilder(
                camera_id=self.camera_id,
                track_id=local_id,
            )
        return self._boxmot_id_to_local[boxmot_id]

    def update(self, frame_idx: int, detections: list) -> None:
        """Feed one frame of detections into the tracker.

        Args:
            frame_idx: Index of the current frame (used for Tracklet2D.frames).
            detections: List of Detection objects from aquapose.segmentation.detector.
                Each detection provides bbox (x, y, w, h) and confidence.
        """
        img_dummy = np.empty((0, 0, 3), dtype=np.uint8)

        # Build Nx6 detection array: [x1, y1, x2, y2, conf, cls]
        if detections:
            rows = []
            for det in detections:
                x, y, w, h = det.bbox
                x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
                conf = float(det.confidence)
                rows.append([x1, y1, x2, y2, conf, 0.0])  # cls=0 (single class)
            dets_array = np.array(rows, dtype=np.float32)
        else:
            dets_array = np.empty((0, 6), dtype=np.float32)

        # Run boxmot tracker
        result = self._tracker.update(dets_array, img_dummy)
        # result shape: (N, 8) — [x1, y1, x2, y2, track_id, conf, cls, idx]
        # track_id column is 1-indexed (boxmot internal)

        # Determine which boxmot track IDs are in this frame's output
        confirmed_this_frame: set[int] = set()
        if len(result) > 0:
            for row in result:
                boxmot_id = int(row[4])
                confirmed_this_frame.add(boxmot_id)
                local_id = self._assign_local_id(boxmot_id)
                bbox_xyxy = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                self._builders[local_id].add_frame(frame_idx, bbox_xyxy, "detected")

        # Capture coasting tracks: active_tracks with time_since_update > 0
        # that have graduated past min_hits (hit_streak >= min_hits at peak)
        # We identify coasting tracks as those in active_tracks but NOT in confirmed output.
        # Use the boxmot internal track ID (trk.id is 0-indexed internally; output is id+1).
        for trk in self._tracker.active_tracks:
            boxmot_id_out = trk.id + 1  # output uses id+1
            is_coasting = (
                boxmot_id_out not in confirmed_this_frame
                and trk.time_since_update > 0
                and boxmot_id_out in self._boxmot_id_to_local
            )
            if not is_coasting:
                continue
            local_id = self._boxmot_id_to_local[boxmot_id_out]
            # Get predicted bbox from Kalman filter state
            trk_state = trk.get_state()  # shape (1, 4) xyxy
            if trk_state is not None and len(trk_state) > 0:
                xyxy = trk_state[0]
                bbox_xyxy = (
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2]),
                    float(xyxy[3]),
                )
                self._builders[local_id].add_frame(frame_idx, bbox_xyxy, "coasted")

        # Update active local ID set
        self._active_local_ids = {
            self._boxmot_id_to_local[bid]
            for bid in self._boxmot_id_to_local
            if any((trk.id + 1) == bid for trk in self._tracker.active_tracks)
        }

        # Mark builders no longer in active_tracks as inactive
        for local_id, builder in self._builders.items():
            if local_id not in self._active_local_ids:
                builder.active = False

    def get_tracklets(self) -> list[Tracklet2D]:
        """Return all confirmed tracklets accumulated so far.

        Confirmed means the track has had at least ``min_hits`` "detected"
        frames. Tentative tracks that never graduated past probation are excluded.

        Returns:
            List of frozen Tracklet2D instances, one per confirmed track.
        """
        result = []
        for builder in self._builders.values():
            if builder.detected_count >= self._min_hits and builder.frames:
                result.append(builder.to_tracklet2d())
        return result

    def get_state(self) -> dict[str, Any]:
        """Return an opaque state blob for cross-batch carry.

        The returned dict captures the boxmot tracker object and all ID-mapping
        state needed to continue tracking in the next batch.

        Returns:
            State dict suitable for storage in CarryForward.tracks_2d_state.
        """
        return {
            "tracker": self._tracker,
            "next_local_id": self._next_local_id,
            "boxmot_id_to_local": dict(self._boxmot_id_to_local),
            "builders": dict(self._builders),
            "active_local_ids": set(self._active_local_ids),
            "camera_id": self.camera_id,
            "max_age": self._max_age,
            "min_hits": self._min_hits,
            "iou_threshold": self._iou_threshold,
            "det_thresh": self._det_thresh,
        }

    @classmethod
    def from_state(cls, camera_id: str, state: dict[str, Any]) -> OcSortTracker:
        """Reconstruct an OcSortTracker from a saved state blob.

        Args:
            camera_id: Camera ID (must match the state's camera_id).
            state: State dict previously returned by get_state().

        Returns:
            OcSortTracker with restored tracker state and ID mappings.
        """
        instance = cls.__new__(cls)
        instance.camera_id = camera_id
        instance._max_age = state["max_age"]
        instance._min_hits = state["min_hits"]
        instance._iou_threshold = state["iou_threshold"]
        instance._det_thresh = state["det_thresh"]
        instance._tracker = state["tracker"]
        instance._next_local_id = state["next_local_id"]
        instance._boxmot_id_to_local = dict(state["boxmot_id_to_local"])
        instance._builders = dict(state["builders"])
        instance._active_local_ids = set(state["active_local_ids"])
        return instance
