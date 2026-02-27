"""TrackingStage — Stage 4 of the 5-stage AquaPose pipeline.

Reads raw detections from Stage 1 (context.detections) and maintains
persistent FishTrack identities across frames via the selected tracking
backend. Populates PipelineContext.tracks.

Design note (v1.0 debt): Stage 4 reads context.detections (raw per-camera
detections from Stage 1), NOT context.associated_bundles (Stage 3 output).
The Hungarian backend re-derives cross-camera association internally via
FishTracker.update() to preserve exact v1.0 numerical equivalence. The
associated_bundles from Stage 3 are passed through to track_frame() for
potential future backends or observers, but the default Hungarian backend
does not consume them. This is documented design debt.
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

    Key design note: run() reads context.detections (Stage 1 raw output),
    not context.associated_bundles (Stage 3 output). The Hungarian backend
    re-derives cross-camera association internally to preserve v1.0 equivalence.
    See module docstring for full rationale.

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

        Reads ``context.detections`` (raw per-frame per-camera detection dicts
        from Stage 1) and produces per-frame confirmed track lists. The
        associated_bundles from Stage 3 are passed to the backend for
        potential future use, but the default Hungarian backend does not
        consume them.

        Populates ``context.tracks`` as a list (one entry per frame) of lists
        (one FishTrack per confirmed fish in that frame).

        Args:
            context: Accumulated pipeline state from prior stages. Must have
                ``detections`` (from Stage 1) populated.

        Returns:
            The same *context* object with ``tracks`` populated.

        Raises:
            ValueError: If ``context.detections`` is not populated.

        """
        if context.detections is None:
            raise ValueError(
                "TrackingStage requires context.detections — "
                "it is not populated. Ensure Stage 1 (DetectionStage) has run.",
            )

        t0 = time.perf_counter()

        tracks_per_frame: list[list] = []
        bundles_per_frame = context.associated_bundles or [
            [] for _ in context.detections
        ]

        for frame_idx, (frame_dets, frame_bundles) in enumerate(
            zip(context.detections, bundles_per_frame, strict=False),
        ):
            frame_tracks = self._backend.track_frame(  # type: ignore[union-attr]
                frame_idx=frame_idx,
                bundles=frame_bundles,
                detections_per_camera=frame_dets,
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
