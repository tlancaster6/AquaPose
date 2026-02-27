"""TrackingStage — Stage 4 of the 5-stage AquaPose pipeline.

Reads pre-associated bundles from Stage 3 (context.associated_bundles) and
maintains persistent FishTrack identities across frames via the selected
tracking backend. Populates PipelineContext.tracks.

Stage 3 (AssociationStage) is a hard dependency — if ``context.associated_bundles``
is not populated, ``run()`` raises a precondition error. The pipeline's data
flow is honest: each stage consumes the previous stage's output.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aquapose.core.context import PipelineContext
from aquapose.core.tracking.backends import get_backend

__all__ = ["TrackingStage"]

logger = logging.getLogger(__name__)


class TrackingStage:
    """Stage 4: Assigns persistent fish identities across frames.

    Runs after AssociationStage (Stage 3). For each frame, calls the
    configured backend to maintain FishTrack objects with lifecycle state
    (probationary -> confirmed -> coasting -> dead). The tracker is
    constructed once and persists across frames, enabling continuous
    fish identity assignment.

    The backend is created eagerly at construction time. A missing calibration
    file raises :class:`FileNotFoundError` immediately.

    ``run()`` reads ``context.associated_bundles`` (Stage 3 output) as its
    primary input. Stage 3 (AssociationStage) is a hard dependency — if
    ``context.associated_bundles`` is ``None``, a ``ValueError`` is raised.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        expected_count: Expected number of fish; used as population constraint.
        backend: Backend kind — currently only ``"hungarian"`` is supported.
        **tracker_kwargs: Additional kwargs forwarded to the backend constructor
            (min_hits, max_age, reprojection_threshold, birth_interval, etc.).

    Raises:
        FileNotFoundError: If *calibration_path* does not exist.
        ValueError: If *backend* is not a recognized backend identifier.

    """

    def __init__(
        self,
        calibration_path: str | Path,
        expected_count: int = 9,
        backend: str = "hungarian",
        **tracker_kwargs: object,
    ) -> None:
        self._calibration_path = Path(calibration_path)
        self._backend = get_backend(
            backend,
            calibration_path=calibration_path,
            expected_count=expected_count,
            **tracker_kwargs,
        )

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run temporal tracking across all frames.

        Reads ``context.associated_bundles`` (pre-associated detection groups
        from Stage 3) and produces per-frame confirmed track lists.

        Populates ``context.tracks`` as a list (one entry per frame) of lists
        (one FishTrack per confirmed fish in that frame).

        Args:
            context: Accumulated pipeline state from prior stages. Must have
                ``associated_bundles`` (from Stage 3) populated.

        Returns:
            The same *context* object with ``tracks`` populated.

        Raises:
            ValueError: If ``context.associated_bundles`` is not populated.

        """
        if context.associated_bundles is None:
            raise ValueError(
                "TrackingStage requires context.associated_bundles — "
                "ensure Stage 3 (AssociationStage) has run.",
            )

        t0 = time.perf_counter()

        tracks_per_frame: list[list] = []

        for frame_idx, frame_bundles in enumerate(context.associated_bundles):
            frame_tracks = self._backend.track_frame(  # type: ignore[union-attr]
                frame_idx=frame_idx,
                bundles=frame_bundles,
            )
            tracks_per_frame.append(frame_tracks)

        elapsed = time.perf_counter() - t0
        logger.info(
            "TrackingStage.run: %d frames, %.2fs",
            len(tracks_per_frame),
            elapsed,
        )

        context.tracks = tracks_per_frame
        return context
