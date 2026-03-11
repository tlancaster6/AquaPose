"""AssociationStage — cross-camera tracklet association (Stage 3).

Scores all cross-camera tracklet pairs using ray-ray geometry and clusters
them into global fish identity groups via the Leiden algorithm. Replaces
AssociationStubStage from Phase 22.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from aquapose.core.context import PipelineContext
from aquapose.core.tracking.types import Tracklet2D

logger = logging.getLogger(__name__)

__all__ = ["AssociationStage"]


class AssociationStage:
    """Cross-camera tracklet association (Stage 3).

    Scores all cross-camera tracklet pairs using ray-ray geometry and
    clusters them into global fish identity groups via Leiden algorithm.
    Populates ``PipelineContext.tracklet_groups``.

    LUTs must be pre-generated via ``aquapose prep generate-luts``.
    This stage loads them from disk but does not generate them lazily.

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

        Raises:
            FileNotFoundError: If pre-generated LUTs are not found on disk.
        """
        from aquapose.calibration.luts import (
            load_forward_luts,
            load_inverse_luts,
        )
        from aquapose.core.association.clustering import (
            build_must_not_link,
            cluster_tracklets,
        )
        from aquapose.core.association.scoring import score_all_pairs

        tracks_2d = cast("dict[str, list[Tracklet2D]]", context.get("tracks_2d"))
        detections = context.detections

        calibration_path = self._config.calibration_path
        if not calibration_path:
            logger.warning("No calibration path -- association producing empty groups")
            context.tracklet_groups = []
            return context

        # Load pre-generated LUTs (no lazy generation)
        forward_luts = load_forward_luts(calibration_path, self._config.lut)
        inverse_lut = load_inverse_luts(calibration_path, self._config.lut)

        if forward_luts is None or inverse_lut is None:
            lut_dir = Path(calibration_path).parent / "luts"
            raise FileNotFoundError(
                f"LUTs not found at {lut_dir}. "
                f"Run: aquapose prep generate-luts --config <path>"
            )

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

        # Step 4: Group validation via multi-keypoint residuals
        if forward_luts is not None:
            from aquapose.core.association.validation import validate_groups

            groups = validate_groups(groups, forward_luts, self._config.association)

        context.tracklet_groups = groups
        return context
