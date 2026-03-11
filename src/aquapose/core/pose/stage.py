"""PoseStage — Stage 2 of the AquaPose pipeline.

Reads detection bounding boxes from Stage 1, crops each detection, and
extracts raw anatomical keypoints via YOLO-pose inference. Writes keypoints
directly onto Detection objects as det.keypoints and det.keypoint_conf.
Runs on ALL detections (before tracking), enabling downstream stages to use
keypoints for OKS-based cost and keypoint centroid association.

Does not populate any dedicated PipelineContext field — enriches detections
in-place.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aquapose.core.context import PipelineContext
from aquapose.core.inference import BatchState, predict_with_oom_retry
from aquapose.core.pose.backends import get_backend

if TYPE_CHECKING:
    from aquapose.core.types.frame_source import FrameSource

__all__ = ["PoseStage"]

logger = logging.getLogger(__name__)

_DEFAULT_N_KEYPOINTS = 6
"""Number of anatomical keypoints for the pose_estimation backend model."""


class PoseStage:
    """Stage 2: Runs pose estimation on all detections, enriches Detection objects with raw anatomical keypoints.

    Runs after DetectionStage (Stage 1) and before TrackingStage (Stage 3).
    For each frame and camera, crops each detection and runs the pose
    estimation backend to produce raw 6-keypoint data. Writes keypoints
    directly onto Detection objects:

    - ``det.keypoints``: shape (6, 2) float32, full-frame pixel coordinates
    - ``det.keypoint_conf``: shape (6,) float32, per-keypoint confidence

    Does not create ``AnnotatedDetection`` wrapper objects. Does not upsample
    to 15-point midlines (moved to ReconstructionStage). Does not apply
    orientation resolution.

    The backend is created eagerly at construction time. A missing weights
    file raises :class:`FileNotFoundError` immediately.

    Frame I/O is delegated to the injected ``frame_source``, which handles
    video discovery, calibration loading, and undistortion.

    Args:
        frame_source: Multi-camera frame provider (e.g. VideoFrameSource).
            Must satisfy the FrameSource protocol.
        weights_path: Path to YOLO-pose model weights file. Raises
            FileNotFoundError if path does not exist. None skips model
            loading (stub mode).
        confidence_threshold: YOLO detection confidence threshold for
            model.predict(). Default 0.5.
        device: PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).
        pose_config: Optional PoseConfig-like object with pose fields
            (weights_path, keypoint_t_values, keypoint_confidence_floor,
            min_observed_keypoints). None uses defaults.
        crop_size: Output crop canvas size as (width, height) in pixels.
        pose_batch_crops: Maximum number of crops per YOLO pose batch.
            ``0`` means no limit (batch all crops in the chunk).

    Raises:
        FileNotFoundError: If required weights files do not exist.

    """

    def __init__(
        self,
        frame_source: FrameSource,
        weights_path: str | None = None,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        pose_config: Any | None = None,
        crop_size: tuple[int, int] = (128, 64),
        pose_batch_crops: int = 0,
    ) -> None:
        self._frame_source = frame_source
        self._pose_config = pose_config
        self._batch_size = pose_batch_crops
        self._batch_state = BatchState()

        # Build the pose estimation backend
        pc = pose_config
        self._backend = get_backend(
            "pose_estimation",
            weights_path=pc.weights_path if pc is not None else weights_path,
            device=device,
            n_keypoints=pc.n_keypoints if pc is not None else _DEFAULT_N_KEYPOINTS,
            keypoint_t_values=pc.keypoint_t_values if pc is not None else None,
            confidence_floor=(pc.keypoint_confidence_floor if pc is not None else 0.3),
            min_observed_keypoints=(pc.min_observed_keypoints if pc is not None else 3),
            crop_size=crop_size,
            conf=confidence_threshold,
        )

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run pose estimation across all cameras for all frames.

        Processes ALL detections (not filtered by tracklet groups) and
        writes raw keypoints directly onto each Detection object.

        Args:
            context: Accumulated pipeline state from prior stages. Must have
                ``detections`` and ``camera_ids`` populated by Stage 1.

        Returns:
            The same *context* object with detections enriched in-place.

        Raises:
            ValueError: If ``context.detections`` or ``context.camera_ids`` is
                None (Stage 1 has not yet run).

        """
        detections = context.get("detections")
        camera_ids = context.get("camera_ids")

        t0 = time.perf_counter()

        with self._frame_source:
            for frame_idx, frames in self._frame_source:
                if frame_idx >= len(detections):  # type: ignore[arg-type]
                    break

                frame_dets = detections[frame_idx]  # type: ignore[index]

                # -- Batched inference path --
                # 1. Collect phase (CPU): extract crops for all detections
                crops = []
                metadata = []
                det_refs: list[
                    Any
                ] = []  # references to Detection objects for in-place write

                for cam_id in camera_ids:  # type: ignore[union-attr]
                    cam_dets = frame_dets.get(cam_id, [])
                    frame = frames.get(cam_id)
                    for det in cam_dets:
                        if frame is None:
                            continue
                        try:
                            crop = self._backend._extract_crop(det, frame)  # type: ignore[union-attr]
                        except Exception:
                            logger.debug(
                                "PoseStage: crop extraction failed for %s frame %d",
                                cam_id,
                                frame_idx,
                                exc_info=True,
                            )
                            continue
                        crops.append(crop)
                        metadata.append((det, cam_id, frame_idx))
                        det_refs.append(det)

                # 2. Predict phase (GPU): batched inference with OOM retry
                if crops:

                    def _batch_predict(
                        items: list[Any],
                    ) -> list[tuple]:
                        crops_chunk = [item[0] for item in items]
                        meta_chunk = [item[1] for item in items]
                        return self._backend.process_batch(crops_chunk, meta_chunk)  # type: ignore[union-attr]

                    batch_results = predict_with_oom_retry(
                        _batch_predict,
                        list(zip(crops, metadata, strict=True)),
                        self._batch_size,
                        self._batch_state,
                    )

                    # 3. Write keypoints onto Detection objects in-place
                    for det, (kpts_xy, kpts_conf) in zip(
                        det_refs, batch_results, strict=True
                    ):
                        if kpts_xy is not None and kpts_conf is not None:
                            det.keypoints = kpts_xy
                            det.keypoint_conf = kpts_conf

        if self._batch_state.oom_occurred:
            logger.info(
                "Pose batch size was reduced to %d due to CUDA OOM. "
                "Consider setting pose.pose_batch_crops=%d in config.",
                self._batch_state.effective_batch_size,
                self._batch_state.effective_batch_size,
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "PoseStage.run: %d frames, %d cameras, %.2fs",
            len(detections),  # type: ignore[arg-type]
            len(camera_ids),  # type: ignore[arg-type]
            elapsed,
        )

        return context
