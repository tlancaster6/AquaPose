"""ReconstructionStage — Stage 5 of the 5-stage AquaPose pipeline.

Reads annotated detections (context.annotated_detections from Stage 4) and
tracklet groups (context.tracklet_groups from Stage 3), assembles per-frame
MidlineSets, and produces 3D B-spline midlines via the configured backend.
Populates PipelineContext.midlines_3d.

v2.1 transition: Stage will be fully updated in Phase 26 to consume
TrackletGroup objects. Until Phase 26, run() requires annotated_detections.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aquapose.core.context import PipelineContext
from aquapose.core.reconstruction.backends import get_backend
from aquapose.reconstruction.midline import Midline2D
from aquapose.reconstruction.triangulation import DEFAULT_INLIER_THRESHOLD

__all__ = ["ReconstructionStage"]

logger = logging.getLogger(__name__)


class ReconstructionStage:
    """Stage 5: Triangulates tracked fish 2D midlines into 3D B-spline midlines.

    Runs after TrackingStage (Stage 4). For each frame, assembles a MidlineSet
    from the confirmed FishTrack objects and their corresponding AnnotatedDetection
    midline data, then passes it to the configured reconstruction backend.

    The backend is created eagerly at construction time. A missing calibration
    file raises :class:`FileNotFoundError` immediately.

    MidlineSet assembly logic (new code bridging decoupled stages):
    - Iterates over confirmed FishTrack objects from Stage 4.
    - For each track, reads ``camera_detections`` (cam_id -> det_idx).
    - For each (cam_id, det_idx), looks up the corresponding AnnotatedDetection
      in ``annotated_detections[frame_idx][cam_id][det_idx]``.
    - If the AnnotatedDetection has a non-None midline, includes it in the set.
    - Builds ``dict[fish_id, dict[cam_id, Midline2D]]`` for the backend.

    In v1.0, MidlineExtractor had direct access to tracks and masks. In the
    new decoupled model, the Reconstruction stage assembles the same structure
    from Stage 2 and Stage 4 outputs.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        backend: Backend kind — ``"triangulation"`` (default) or
            ``"curve_optimizer"``.
        inlier_threshold: Maximum reprojection error (pixels) for RANSAC inliers.
            Only used by the triangulation backend.
        snap_threshold: Maximum pixel distance from epipolar curve for
            correspondence refinement. Only used by the triangulation backend.
        max_depth: Maximum allowed fish depth below the water surface (metres).
            None disables the upper depth bound. Only used by the triangulation backend.
        **backend_kwargs: Additional kwargs forwarded to the backend constructor.

    Raises:
        FileNotFoundError: If *calibration_path* does not exist.
        ValueError: If *backend* is not a recognized backend identifier.

    """

    def __init__(
        self,
        calibration_path: str | Path,
        backend: str = "triangulation",
        inlier_threshold: float = DEFAULT_INLIER_THRESHOLD,
        snap_threshold: float = 20.0,
        max_depth: float | None = None,
        **backend_kwargs: object,
    ) -> None:
        self._calibration_path = Path(calibration_path)

        # Build kwargs for the backend constructor
        combined_kwargs: dict[str, object] = {
            "calibration_path": calibration_path,
        }
        if backend == "triangulation":
            combined_kwargs["inlier_threshold"] = inlier_threshold
            combined_kwargs["snap_threshold"] = snap_threshold
            combined_kwargs["max_depth"] = max_depth
        combined_kwargs.update(backend_kwargs)

        self._backend = get_backend(backend, **combined_kwargs)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run 3D reconstruction across all frames.

        v2.1 transition: Reads ``context.annotated_detections`` (Stage 4) to
        assemble per-frame MidlineSets. Fish identity is populated from
        ``context.tracklet_groups`` when available (Phase 26+). Until Phase 26,
        reconstruction produces midlines without persistent fish IDs.

        Populates ``context.midlines_3d`` as a list (one entry per frame) of
        dicts mapping fish_id to Midline3D.

        Args:
            context: Accumulated pipeline state from prior stages. Must have
                ``annotated_detections`` (from Stage 4) populated.

        Returns:
            The same *context* object with ``midlines_3d`` populated.

        Raises:
            ValueError: If ``context.annotated_detections`` is not populated.

        """
        if context.annotated_detections is None:
            raise ValueError(
                "ReconstructionStage requires context.annotated_detections — "
                "it is not populated. Ensure Stage 4 (MidlineStage) has run.",
            )

        t0 = time.perf_counter()

        midlines_3d_per_frame: list[dict] = []

        # v2.1 transition: iterate frames using annotated_detections only.
        # Phase 26 will replace this with TrackletGroup-driven reconstruction.
        for frame_idx, frame_annotated in enumerate(context.annotated_detections):
            frame_tracks: list = []  # empty until Phase 26 wires tracklet_groups
            midline_set = self._assemble_midline_set(
                frame_idx=frame_idx,
                frame_tracks=frame_tracks,
                frame_annotated=frame_annotated,
            )

            if midline_set:
                frame_results = self._backend.reconstruct_frame(  # type: ignore[union-attr]
                    frame_idx=frame_idx,
                    midline_set=midline_set,
                )
            else:
                frame_results = {}

            midlines_3d_per_frame.append(frame_results)

        elapsed = time.perf_counter() - t0
        logger.info(
            "ReconstructionStage.run: %d frames, %.2fs",
            len(midlines_3d_per_frame),
            elapsed,
        )

        context.midlines_3d = midlines_3d_per_frame
        return context

    def _assemble_midline_set(
        self,
        frame_idx: int,
        frame_tracks: list,
        frame_annotated: dict[str, list],
    ) -> dict[int, dict[str, Midline2D]]:
        """Assemble a MidlineSet from FishTrack and AnnotatedDetection data.

        For each confirmed FishTrack, looks up the per-camera AnnotatedDetection
        at the detection index stored in ``track.camera_detections``. Includes
        only cameras where the AnnotatedDetection has a non-None midline.

        Args:
            frame_idx: Current frame index (for logging).
            frame_tracks: List of FishTrack objects for this frame (all states).
            frame_annotated: Dict mapping camera_id to list of AnnotatedDetection.

        Returns:
            MidlineSet: dict[fish_id, dict[cam_id, Midline2D]]. May be empty
            if no fish have sufficient midline observations.

        """
        midline_set: dict[int, dict[str, Midline2D]] = {}

        for track in frame_tracks:
            fish_id: int = track.fish_id
            camera_detections: dict[str, int] = track.camera_detections

            if not camera_detections:
                continue

            cam_midlines: dict[str, Midline2D] = {}

            for cam_id, det_idx in camera_detections.items():
                # Look up annotated detections for this camera
                cam_annots = frame_annotated.get(cam_id)
                if cam_annots is None:
                    continue
                if det_idx >= len(cam_annots):
                    logger.debug(
                        "Frame %d fish %d: det_idx=%d out of range for cam=%s (len=%d)",
                        frame_idx,
                        fish_id,
                        det_idx,
                        cam_id,
                        len(cam_annots),
                    )
                    continue

                annotated = cam_annots[det_idx]
                # Handle both AnnotatedDetection objects and raw Detection objects
                midline = getattr(annotated, "midline", None)
                if midline is None:
                    continue

                cam_midlines[cam_id] = midline

            if cam_midlines:
                midline_set[fish_id] = cam_midlines

        return midline_set
