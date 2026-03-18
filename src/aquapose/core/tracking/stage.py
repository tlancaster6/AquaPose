"""TrackingStage -- Stage 2 of the AquaPose pipeline.

Per-camera 2D tracking producing Tracklet2D objects that carry temporal
identity for the Association stage (Stage 3).

Uses the custom single-pass keypoint tracker from
``aquapose.core.tracking.keypoint_tracker`` with ORU/OCR occlusion recovery.
"""

from __future__ import annotations

import logging
from typing import Any

from aquapose.core.context import ChunkHandoff, PipelineContext

logger = logging.getLogger(__name__)


class TrackingStage:
    """Stage 2: Per-camera 2D tracking.

    Consumes ``context.detections`` (Stage 1 output) and produces
    ``context.tracks_2d`` -- a ``dict[str, list[Tracklet2D]]`` keyed by
    camera ID.

    Each camera's detections are independently tracked by a KeypointTracker.
    Tracker state is preserved between pipeline chunks via ``ChunkHandoff``,
    enabling continuous track identity across chunk boundaries.

    Only confirmed tracklets (those that have accumulated at least
    ``config.n_init`` matched detection frames) appear in the output.
    Tentative tracks that never graduated past probation are excluded.

    Args:
        config: Frozen TrackingConfig.
        centroid_keypoint_index: Index into Detection.keypoints for tracklet
            centroid. Default 2 (spine1). Passed to the tracker backend.
        centroid_confidence_floor: Minimum keypoint confidence to use keypoint
            as centroid. Default 0.3. Passed to the tracker backend.

    Example::

        stage = TrackingStage(config=TrackingConfig())
        context, carry = stage.run(context, carry=None)
        # context.tracks_2d: {cam_id: [Tracklet2D, ...], ...}
    """

    def __init__(
        self,
        config: Any,
        centroid_keypoint_index: int = 2,
        centroid_confidence_floor: float = 0.3,
    ) -> None:
        # Accept Any to avoid circular imports from engine/ into core/.
        # The config must have: max_coast_frames, n_init, iou_threshold, det_thresh.
        self._config = config
        self._centroid_keypoint_index = centroid_keypoint_index
        self._centroid_confidence_floor = centroid_confidence_floor

    def run(
        self,
        context: PipelineContext,
        carry: object | None = None,
    ) -> tuple[PipelineContext, ChunkHandoff]:
        """Run per-camera tracking on this batch of frames.

        Reads ``context.detections`` (``list[dict[str, list[Detection]]]``) and
        ``context.camera_ids`` (``list[str]``). For each camera, restores or
        creates a tracker, feeds all frames, then collects confirmed Tracklet2D
        objects. A new ``ChunkHandoff`` is built from the per-camera tracker
        states for the next chunk.

        The tracker backend is ``"keypoint_oks"`` (KeypointTracker).

        Args:
            context: Accumulated pipeline state from the Detection stage.
                ``context.detections`` must be set.
            carry: Cross-chunk carry state from a prior ``TrackingStage.run()``
                call. When ``None``, fresh trackers are created for all cameras.
                Expected runtime type: ``ChunkHandoff | None``.

        Returns:
            Tuple of ``(context, new_carry)``. ``context.tracks_2d`` is set to
            a ``dict[str, list[Tracklet2D]]`` keyed by camera ID.
            ``new_carry.tracks_2d_state`` holds per-camera tracker states.

        """
        prev_tracks_2d_state: dict = {}
        if carry is not None and hasattr(carry, "tracks_2d_state"):
            prev_tracks_2d_state = carry.tracks_2d_state  # type: ignore[union-attr]

        detections: list = context.detections or []
        camera_ids: list[str] = context.camera_ids or []

        from aquapose.core.tracking.keypoint_tracker import KeypointTracker

        trackers: dict[str, Any] = {}
        for cam_id in camera_ids:
            if cam_id in prev_tracks_2d_state:
                trackers[cam_id] = KeypointTracker.from_state(
                    cam_id, prev_tracks_2d_state[cam_id]
                )
            else:
                trackers[cam_id] = KeypointTracker(
                    camera_id=cam_id,
                    max_age=self._config.max_coast_frames,
                    n_init=self._config.n_init,
                    det_thresh=self._config.det_thresh,
                    track_thresh=self._config.track_thresh,
                    birth_thresh=self._config.birth_thresh,
                    base_r=self._config.base_r,
                    lambda_ocm=self._config.lambda_ocm,
                    max_gap_frames=self._config.max_gap_frames,
                    match_cost_threshold=self._config.match_cost_threshold,
                    ocr_threshold=self._config.ocr_threshold,
                    max_match_distance=self._config.max_match_distance,
                    centroid_keypoint_index=self._centroid_keypoint_index,
                    centroid_confidence_floor=self._centroid_confidence_floor,
                    merger_distance=self._config.merger_distance_px,
                    merger_max_age=self._config.merger_max_coast_frames,
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

        # Build new carry from updated tracker states, preserving identity fields
        new_tracks_2d_state = {
            cam_id: trackers[cam_id].get_state() for cam_id in camera_ids
        }
        if isinstance(carry, ChunkHandoff):
            new_carry = ChunkHandoff(
                tracks_2d_state=new_tracks_2d_state,
                identity_map=carry.identity_map,
                fish_tracklet_sets=carry.fish_tracklet_sets,
                next_global_id=carry.next_global_id,
            )
        else:
            new_carry = ChunkHandoff(
                tracks_2d_state=new_tracks_2d_state,
                identity_map={},
                fish_tracklet_sets={},
                next_global_id=0,
            )

        return context, new_carry
