"""TrackingStage — Stage 2 of the AquaPose v2.1 pipeline.

Per-camera 2D tracking using OC-SORT via the boxmot package. Each camera's
detections are tracked independently, producing Tracklet2D objects that carry
temporal identity for the Association stage (Stage 3).

All boxmot internals are fully isolated in aquapose.tracking.ocsort_wrapper.
This stage only imports from that wrapper and from the project's own contracts.
"""

from __future__ import annotations

import logging
from typing import Any

from aquapose.core.context import CarryForward, PipelineContext

logger = logging.getLogger(__name__)


class TrackingStage:
    """Stage 2: Per-camera 2D tracking.

    Consumes ``context.detections`` (Stage 1 output) and produces
    ``context.tracks_2d`` — a ``dict[str, list[Tracklet2D]]`` keyed by
    camera ID.

    Each camera's detections are independently tracked by an OcSortTracker.
    Tracker state is preserved between pipeline batches via ``CarryForward``,
    enabling continuous track identity across batch boundaries.

    Only confirmed tracklets (those that have accumulated at least
    ``config.n_init`` matched detection frames) appear in the output.
    Tentative tracks that never graduated past probation are excluded.

    Args:
        config: Frozen TrackingConfig providing OC-SORT parameters.

    Example::

        stage = TrackingStage(config=TrackingConfig())
        context, carry = stage.run(context, carry=None)
        # context.tracks_2d: {cam_id: [Tracklet2D, ...], ...}
    """

    def __init__(self, config: Any) -> None:
        # Accept Any to avoid circular imports from engine/ into core/.
        # The config must have: max_coast_frames, n_init, iou_threshold, det_thresh.
        self._config = config

    def run(
        self,
        context: PipelineContext,
        carry: CarryForward | None = None,
    ) -> tuple[PipelineContext, CarryForward]:
        """Run per-camera OC-SORT tracking on this batch of frames.

        Reads ``context.detections`` (``list[dict[str, list[Detection]]]``) and
        ``context.camera_ids`` (``list[str]``). For each camera, restores or
        creates an OcSortTracker, feeds all frames to the tracker, then collects
        confirmed Tracklet2D objects. A new ``CarryForward`` is built from the
        per-camera tracker states for the next batch.

        Args:
            context: Accumulated pipeline state from the Detection stage.
                ``context.detections`` must be set.
            carry: Cross-batch carry state from a prior ``TrackingStage.run()``
                call. When ``None``, fresh trackers are created for all cameras.

        Returns:
            Tuple of ``(context, new_carry)``. ``context.tracks_2d`` is set to
            a ``dict[str, list[Tracklet2D]]`` keyed by camera ID.
            ``new_carry.tracks_2d_state`` holds per-camera tracker states.

        """
        from aquapose.tracking.ocsort_wrapper import OcSortTracker

        if carry is None:
            carry = CarryForward()

        detections: list = context.detections or []
        camera_ids: list[str] = context.camera_ids or []

        # Build or restore one OcSortTracker per camera
        trackers: dict[str, OcSortTracker] = {}
        for cam_id in camera_ids:
            if cam_id in carry.tracks_2d_state:
                trackers[cam_id] = OcSortTracker.from_state(
                    cam_id, carry.tracks_2d_state[cam_id]
                )
            else:
                trackers[cam_id] = OcSortTracker(
                    camera_id=cam_id,
                    max_age=self._config.max_coast_frames,
                    min_hits=self._config.n_init,
                    iou_threshold=self._config.iou_threshold,
                    det_thresh=self._config.det_thresh,
                )

        # Feed all frames to each camera's tracker
        for frame_idx, frame_dets in enumerate(detections):
            for cam_id in camera_ids:
                cam_dets = frame_dets.get(cam_id, [])
                trackers[cam_id].update(frame_idx, cam_dets)

        # Collect confirmed tracklets per camera
        tracks_2d: dict = {}
        for cam_id in camera_ids:
            tracks_2d[cam_id] = trackers[cam_id].get_tracklets()

        context.tracks_2d = tracks_2d

        # Build new carry from updated tracker states
        new_carry = CarryForward(
            tracks_2d_state={
                cam_id: trackers[cam_id].get_state() for cam_id in camera_ids
            }
        )

        return context, new_carry
