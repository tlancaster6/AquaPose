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
            load_forward_luts,
            load_inverse_luts,
        )
        from aquapose.core.association.clustering import (
            build_must_not_link,
            cluster_tracklets,
            merge_fragments,
        )
        from aquapose.core.association.scoring import score_all_pairs

        tracks_2d = context.get("tracks_2d")
        detections = context.detections

        # Load LUTs from cache
        forward_luts = None
        inverse_lut = None

        calibration_path = self._config.calibration_path
        if calibration_path:
            forward_luts = load_forward_luts(calibration_path, self._config.lut)
            inverse_lut = load_inverse_luts(calibration_path, self._config.lut)

        if forward_luts is None or inverse_lut is None:
            logger.warning("LUTs not available -- association producing empty groups")
            context.tracklet_groups = []
            return context

        # Extract detection centroids for ghost penalty
        det_centroids = _extract_centroids(detections)

        # Step 1: Score all pairs
        scores = score_all_pairs(
            tracks_2d,
            forward_luts,
            inverse_lut,
            det_centroids,
            self._config.association,
        )

        # Step 2-3: Build constraints and cluster
        mnl = build_must_not_link(tracks_2d)
        groups = cluster_tracklets(scores, tracks_2d, mnl, self._config.association)

        # Step 4: Merge fragments
        groups = merge_fragments(groups, self._config.association)

        context.tracklet_groups = groups
        return context


def _extract_centroids(
    detections: list | None,
) -> list[dict[str, list[tuple[float, float]]]]:
    """Convert Detection objects to simple (u, v) centroid tuples.

    Args:
        detections: Per-frame per-camera detection lists from Stage 1.
            Each entry is ``dict[str, list[Detection]]``.

    Returns:
        Per-frame per-camera centroid lists suitable for ghost penalty scoring.
    """
    if detections is None:
        return []

    result: list[dict[str, list[tuple[float, float]]]] = []
    for frame_dets in detections:
        frame_centroids: dict[str, list[tuple[float, float]]] = {}
        for cam_id, det_list in frame_dets.items():
            centroids: list[tuple[float, float]] = []
            for det in det_list:
                # Detection objects have .centroid or .bbox — extract centroid
                if hasattr(det, "centroid"):
                    c = det.centroid
                    centroids.append((float(c[0]), float(c[1])))
                elif hasattr(det, "bbox"):
                    # Derive centroid from bbox (x, y, w, h)
                    bx, by, bw, bh = det.bbox
                    centroids.append((float(bx + bw / 2), float(by + bh / 2)))
            frame_centroids[cam_id] = centroids
        result.append(frame_centroids)

    return result
