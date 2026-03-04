"""AssociationStage — cross-camera tracklet association (Stage 3).

Scores all cross-camera tracklet pairs using ray-ray geometry and clusters
them into global fish identity groups via the Leiden algorithm. Replaces
AssociationStubStage from Phase 22.
"""

from __future__ import annotations

import logging
from typing import Any

from aquapose.core.context import PipelineContext

logger = logging.getLogger(__name__)

__all__ = ["AssociationStage"]


class AssociationStage:
    """Cross-camera tracklet association (Stage 3).

    Scores all cross-camera tracklet pairs using ray-ray geometry and
    clusters them into global fish identity groups via Leiden algorithm.
    Populates ``PipelineContext.tracklet_groups``.

    The config parameter is ``Any``-typed to avoid the circular
    engine -> core import (same pattern as TrackingStage in Phase 24).

    Args:
        config: PipelineConfig instance. Accessed as ``config.association``
            for scoring/clustering fields, ``config.lut`` for LUT config,
            ``config.calibration_path`` for calibration data path.
    """

    def __init__(self, config: Any) -> None:
        self._config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Score tracklet pairs and cluster into fish identity groups.

        Args:
            context: Accumulated pipeline state from the Tracking stage.

        Returns:
            Context with ``tracklet_groups`` populated.
        """
        from aquapose.calibration.luts import (
            generate_forward_luts,
            generate_inverse_lut,
            load_forward_luts,
            load_inverse_luts,
            save_forward_luts,
            save_inverse_luts,
        )
        from aquapose.core.association.clustering import (
            build_must_not_link,
            cluster_tracklets,
            merge_fragments,
        )
        from aquapose.core.association.scoring import score_all_pairs

        tracks_2d = context.get("tracks_2d")
        detections = context.detections

        calibration_path = self._config.calibration_path
        if not calibration_path:
            logger.warning("No calibration path -- association producing empty groups")
            context.tracklet_groups = []
            return context

        # Load LUTs from cache, auto-generate on miss
        forward_luts = load_forward_luts(calibration_path, self._config.lut)
        inverse_lut = load_inverse_luts(calibration_path, self._config.lut)

        if forward_luts is None or inverse_lut is None:
            from aquapose.calibration.loader import (
                compute_undistortion_maps,
                load_calibration_data,
            )

            logger.info(
                "LUTs not cached -- generating (first run may take a few minutes)"
            )
            calibration = load_calibration_data(calibration_path)

            # Compute undistortion maps so LUTs are built in undistorted pixel
            # space (K_new), matching the coordinate system of the tracklet
            # centroids produced by VideoFrameSource (which undistorts frames).
            undistortion_maps = {
                cam_id: compute_undistortion_maps(calibration.cameras[cam_id])
                for cam_id in calibration.ring_cameras
            }

            if forward_luts is None:
                forward_luts = generate_forward_luts(
                    calibration, self._config.lut, undistortion_maps=undistortion_maps
                )
                save_forward_luts(forward_luts, calibration_path, self._config.lut)

            if inverse_lut is None:
                inverse_lut = generate_inverse_lut(
                    calibration, self._config.lut, undistortion_maps=undistortion_maps
                )
                save_inverse_luts(inverse_lut, calibration_path, self._config.lut)

        # Step 1: Score all pairs
        frame_count = len(detections) if detections else None
        scores = score_all_pairs(
            tracks_2d,
            forward_luts,
            inverse_lut,
            self._config.association,
            frame_count=frame_count,
        )

        # Step 2-3: Build constraints and cluster
        mnl = build_must_not_link(tracks_2d)
        groups = cluster_tracklets(scores, tracks_2d, mnl, self._config.association)

        # Step 4: Merge fragments
        groups = merge_fragments(groups, self._config.association)

        # Step 5: Geometric refinement via 3D triangulation
        if forward_luts is not None:
            from aquapose.core.association.refinement import refine_clusters

            groups = refine_clusters(groups, forward_luts, self._config.association)

        context.tracklet_groups = groups
        return context
