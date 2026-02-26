"""Segment-then-extract backend for the Midline stage.

Combines v1.0's run_segmentation and run_midline_extraction into a single
per-frame operation that does NOT require track assignments. It annotates ALL
detections with midline data (not just tracked fish), enabling downstream
Association and Tracking stages to use the midline information.

Data flow per frame:
  1. For each camera, compute crop regions from detection bboxes.
  2. Extract crops from the frame and batch-segment via UNetSegmentor.
  3. For each detection, run the midline pipeline (smooth → skeleton → BFS
     → arc-length resample → crop-to-frame).
  4. Return annotated detections (Detection + mask + CropRegion + Midline2D).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from aquapose.core.midline.types import AnnotatedDetection
from aquapose.reconstruction.midline import (
    Midline2D,
    _adaptive_smooth,
    _check_skip_mask,
    _crop_to_frame,
    _longest_path_bfs,
    _resample_arc_length,
    _skeleton_and_widths,
)
from aquapose.segmentation.crop import CropRegion, compute_crop_region, extract_crop
from aquapose.segmentation.detector import Detection

__all__ = ["SegmentThenExtractBackend"]

logger = logging.getLogger(__name__)


class SegmentThenExtractBackend:
    """Segment-then-extract midline backend.

    Eagerly loads UNetSegmentor and configures the midline extraction pipeline
    at construction time (fail-fast: missing weights raise immediately).

    Args:
        weights_path: Path to U-Net model weights file. Raises FileNotFoundError
            if the path does not exist (None uses pretrained ImageNet encoder).
        confidence_threshold: Minimum mean foreground probability for the
            segmentor to accept a mask result.
        n_points: Number of midline points to produce per detection.
        min_area: Minimum mask area (pixels) to attempt midline extraction.
        device: PyTorch device string (e.g. "cuda", "cpu").

    Raises:
        FileNotFoundError: If *weights_path* is provided and does not exist.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        n_points: int = 15,
        min_area: int = 300,
        device: str = "cuda",
    ) -> None:
        import torch

        from aquapose.segmentation.model import UNetSegmentor

        # Fail-fast: validate weights path before loading anything
        if weights_path is not None:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"U-Net weights file does not exist: {weights_path}. "
                    "Provide a valid path or pass weights_path=None to use "
                    "the pretrained ImageNet encoder."
                )

        self._n_points = n_points
        self._min_area = min_area
        self._device = device

        # Eager model load — construction fails immediately if weights are bad
        self._segmentor = UNetSegmentor(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
        )

        # Move model to target device
        self._segmentor.get_model().to(torch.device(device))

        logger.info(
            "SegmentThenExtractBackend: loaded U-Net (weights=%s, device=%s, "
            "n_points=%d, min_area=%d)",
            weights_path,
            device,
            n_points,
            min_area,
        )

    def process_frame(
        self,
        frame_idx: int,
        frame_dets: dict[str, list[Detection]],
        frames: dict[str, np.ndarray],
        camera_ids: list[str],
    ) -> dict[str, list[AnnotatedDetection]]:
        """Process a single frame: segment all detections and extract midlines.

        For each camera with detections, crops and segments each detection,
        then extracts a 2D midline. All detections are annotated regardless of
        tracking state (unlike v1.0 which only processed tracked fish).

        Args:
            frame_idx: Current frame index (used for Midline2D metadata).
            frame_dets: Per-camera detection lists from Stage 1.
            frames: Per-camera undistorted frame images (BGR uint8).
            camera_ids: Active camera identifiers.

        Returns:
            Per-camera list of AnnotatedDetection objects, one per input detection.
            Midline and mask may be None if extraction failed for that detection.
        """
        annotated: dict[str, list[AnnotatedDetection]] = {}

        for cam_id in camera_ids:
            cam_dets = frame_dets.get(cam_id, [])
            frame = frames.get(cam_id)

            if not cam_dets or frame is None:
                annotated[cam_id] = []
                continue

            h_frame, w_frame = frame.shape[:2]

            # Build crops for batch segmentation
            crops: list[np.ndarray] = []
            crop_regions: list[CropRegion] = []

            for det in cam_dets:
                region = compute_crop_region(
                    bbox=det.bbox,
                    frame_h=h_frame,
                    frame_w=w_frame,
                )
                crop = extract_crop(frame, region)
                crops.append(crop)
                crop_regions.append(region)

            # Batch segmentation — UNetSegmentor returns nested list
            seg_results = self._segmentor.segment(crops, crop_regions)

            # Extract midlines per detection
            cam_annotated: list[AnnotatedDetection] = []
            for det, region, per_crop_results in zip(
                cam_dets, crop_regions, seg_results, strict=True
            ):
                if not per_crop_results:
                    # Segmentor found nothing for this crop
                    cam_annotated.append(
                        AnnotatedDetection(
                            detection=det,
                            mask=None,
                            crop_region=None,
                            midline=None,
                            camera_id=cam_id,
                            frame_index=frame_idx,
                        )
                    )
                    continue

                mask = per_crop_results[0].mask  # UNet yields at most 1 result/crop
                midline = self._extract_midline_from_mask(
                    mask=mask,
                    crop_region=region,
                    det=det,
                    cam_id=cam_id,
                    frame_idx=frame_idx,
                )

                cam_annotated.append(
                    AnnotatedDetection(
                        detection=det,
                        mask=mask,
                        crop_region=region,
                        midline=midline,
                        camera_id=cam_id,
                        frame_index=frame_idx,
                    )
                )

            annotated[cam_id] = cam_annotated

        return annotated

    def _extract_midline_from_mask(
        self,
        mask: np.ndarray,
        crop_region: CropRegion,
        det: Detection,
        cam_id: str,
        frame_idx: int,
    ) -> Midline2D | None:
        """Run the full midline extraction pipeline on a single mask.

        Mirrors the per-fish logic from v1.0 MidlineExtractor.extract_midlines(),
        adapted for per-detection use (no fish_id yet — uses detection index
        placeholder -1 since tracking has not yet assigned IDs).

        Steps:
            1. Check skip conditions (too small, boundary-clipped).
            2. Adaptive morphological smoothing.
            3. Skeletonization + distance transform.
            4. Check skeleton length.
            5. BFS longest path.
            6. Arc-length resampling to n_points.
            7. Crop-to-frame coordinate transform.

        Args:
            mask: Binary crop-space mask (uint8, 0/255), shape (H_crop, W_crop).
            crop_region: CropRegion mapping the mask to the full frame.
            det: Source detection (used only for logging context).
            cam_id: Camera identifier for logging.
            frame_idx: Frame index for logging and Midline2D metadata.

        Returns:
            Midline2D in full-frame coordinates, or None if extraction failed.
        """
        crop_h, crop_w = mask.shape[:2]

        # 1. Skip check
        skip_reason = _check_skip_mask(mask, crop_region, self._min_area)
        if skip_reason:
            logger.debug(
                "Skipping detection camera=%s frame=%d bbox=%s: %s",
                cam_id,
                frame_idx,
                det.bbox,
                skip_reason,
            )
            return None

        # 2. Adaptive smooth
        smooth = _adaptive_smooth(mask)

        # 3. Skeleton + distance transform
        skeleton_bool, dt = _skeleton_and_widths(smooth)

        # 4. Check skeleton length
        n_skel = int(np.sum(skeleton_bool))
        if n_skel < self._n_points:
            logger.debug(
                "Skeleton too short (%d < %d) for camera=%s frame=%d",
                n_skel,
                self._n_points,
                cam_id,
                frame_idx,
            )
            return None

        # 5. BFS longest path
        path_yx = _longest_path_bfs(skeleton_bool)
        if not path_yx:
            return None

        # 6. Arc-length resample
        xy_crop, half_widths = _resample_arc_length(path_yx, dt, self._n_points)

        # 7. Crop-to-frame transform
        xy_frame, hw_frame = _crop_to_frame(
            xy_crop, half_widths, crop_region, crop_h, crop_w
        )

        # fish_id is -1 at this point — tracking has not yet assigned IDs.
        # Downstream stages (Association, Tracking) will replace this placeholder.
        return Midline2D(
            points=xy_frame,
            half_widths=hw_frame,
            fish_id=-1,
            camera_id=cam_id,
            frame_index=frame_idx,
            is_head_to_tail=False,
        )
