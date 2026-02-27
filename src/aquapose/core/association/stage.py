"""AssociationStage — Stage 3 of the 5-stage AquaPose pipeline.

Reads annotated detections from Stage 2, groups detections across cameras into
per-fish bundles via RANSAC centroid clustering, and populates
PipelineContext.associated_bundles.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aquapose.core.association.backends import get_backend
from aquapose.core.context import PipelineContext

__all__ = ["AssociationStage"]

logger = logging.getLogger(__name__)


class AssociationStage:
    """Stage 3: Groups detections across cameras into per-fish bundles.

    Runs after MidlineStage (Stage 2). For each frame, calls the configured
    backend to cluster detections from multiple cameras into cross-view bundles,
    each representing one physical fish. All detections are treated as unclaimed
    at this stage — temporal tracking (Stage 4) has not yet run.

    The backend is created eagerly at construction time. A missing calibration
    file raises :class:`FileNotFoundError` immediately.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        expected_count: Expected number of fish; used as RANSAC stopping
            criterion.
        min_cameras: Minimum cameras required for a valid multi-view bundle.
        reprojection_threshold: Maximum pixel reprojection error for RANSAC
            inliers.
        backend: Backend kind — currently only ``"ransac_centroid"`` is
            supported.

    Raises:
        FileNotFoundError: If *calibration_path* does not exist.
        ValueError: If *backend* is not a recognized backend identifier.
    """

    def __init__(
        self,
        calibration_path: str | Path,
        expected_count: int = 9,
        min_cameras: int = 3,
        reprojection_threshold: float = 15.0,
        backend: str = "ransac_centroid",
    ) -> None:
        self._calibration_path = Path(calibration_path)
        self._backend = get_backend(
            backend,
            calibration_path=calibration_path,
            expected_count=expected_count,
            min_cameras=min_cameras,
            reprojection_threshold=reprojection_threshold,
        )

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run cross-view association across all frames.

        Reads ``context.detections`` (or ``context.annotated_detections`` if
        Stage 2 has run) and produces per-frame bundles.

        When ``context.annotated_detections`` is available (Stage 2 has run),
        its structure is used to extract Detection objects from each
        AnnotatedDetection wrapper. When only ``context.detections`` is
        available (Stage 2 skipped), raw Detection lists are used directly.

        Populates ``context.associated_bundles`` as a list (one entry per frame)
        of lists (one AssociationBundle per identified fish).

        Args:
            context: Accumulated pipeline state from prior stages. Must have
                either ``detections`` (from Stage 1) or ``annotated_detections``
                (from Stage 2) populated.

        Returns:
            The same *context* object with ``associated_bundles`` populated.

        Raises:
            ValueError: If neither ``detections`` nor ``annotated_detections``
                is populated on the context.
        """
        # Prefer annotated_detections (Stage 2 output) over raw detections.
        # Both have the same outer structure: list[dict[str, list[Detection]]].
        # AnnotatedDetection wraps Detection, so we unwrap here.
        if context.annotated_detections is not None:
            source = context.annotated_detections
            use_annotated = True
        elif context.detections is not None:
            source = context.detections
            use_annotated = False
        else:
            raise ValueError(
                "AssociationStage requires context.detections or "
                "context.annotated_detections — neither is populated. "
                "Ensure Stage 1 (DetectionStage) has run."
            )

        t0 = time.perf_counter()

        bundles_per_frame: list[list] = []

        for frame_dets in source:
            # Build detections_per_camera: dict[str, list[Detection]]
            # Unwrap AnnotatedDetection wrappers if from Stage 2.
            detections_per_camera: dict[str, list] = {}
            for cam_id, cam_list in frame_dets.items():
                if use_annotated:
                    # Extract the underlying Detection from each AnnotatedDetection
                    detections_per_camera[cam_id] = [ad.detection for ad in cam_list]
                else:
                    detections_per_camera[cam_id] = list(cam_list)

            frame_bundles = self._backend.associate_frame(  # type: ignore[union-attr]
                detections_per_camera
            )
            bundles_per_frame.append(frame_bundles)

        elapsed = time.perf_counter() - t0
        logger.info(
            "AssociationStage.run: %d frames, %.2fs",
            len(bundles_per_frame),
            elapsed,
        )

        context.associated_bundles = bundles_per_frame
        return context
