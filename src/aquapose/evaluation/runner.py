"""EvalRunner orchestrates per-stage evaluation from diagnostic run directories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from aquapose.core.context import load_stage_cache
from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import MidlineSet
from aquapose.evaluation.metrics import select_frames
from aquapose.evaluation.stages import (
    AssociationMetrics,
    DetectionMetrics,
    MidlineMetrics,
    ReconstructionMetrics,
    TrackingMetrics,
    evaluate_association,
    evaluate_detection,
    evaluate_midline,
    evaluate_reconstruction,
    evaluate_tracking,
)

if TYPE_CHECKING:
    from aquapose.core.context import PipelineContext

_STAGE_KEYS: tuple[str, ...] = (
    "detection",
    "tracking",
    "association",
    "midline",
    "reconstruction",
)

# Centroid match tolerance in pixels for TrackletGroup -> AnnotatedDetection matching.
_CENTROID_MATCH_TOLERANCE_PX = 5.0


@dataclass(frozen=True)
class EvalRunnerResult:
    """Aggregated evaluation result from a single pipeline run directory.

    Attributes:
        run_id: Pipeline run identifier from cache envelope metadata.
        stages_present: Set of stage keys with cache files present.
        detection: Detection stage metrics, or None if cache missing.
        tracking: Tracking stage metrics, or None if cache missing.
        association: Association stage metrics, or None if cache missing.
        midline: Midline stage metrics, or None if cache missing.
        reconstruction: Reconstruction stage metrics, or None if cache missing.
        frames_evaluated: Number of frames actually evaluated (after sampling).
        frames_available: Total frames available in the run.
    """

    run_id: str
    stages_present: frozenset[str]
    detection: DetectionMetrics | None
    tracking: TrackingMetrics | None
    association: AssociationMetrics | None
    midline: MidlineMetrics | None
    reconstruction: ReconstructionMetrics | None
    frames_evaluated: int
    frames_available: int

    def to_dict(self) -> dict[str, object]:
        """Return a fully JSON-serializable nested dict representation.

        Returns:
            Dict with run_id, stages_present (sorted list), frames_evaluated,
            frames_available, and a "stages" dict where each present stage key
            maps to its metrics' to_dict() output. Absent stages are omitted
            from the "stages" dict.
        """
        stages: dict[str, object] = {}
        if self.detection is not None:
            stages["detection"] = self.detection.to_dict()
        if self.tracking is not None:
            stages["tracking"] = self.tracking.to_dict()
        if self.association is not None:
            stages["association"] = self.association.to_dict()
        if self.midline is not None:
            stages["midline"] = self.midline.to_dict()
        if self.reconstruction is not None:
            stages["reconstruction"] = self.reconstruction.to_dict()

        return {
            "run_id": self.run_id,
            "stages_present": sorted(self.stages_present),
            "frames_evaluated": self.frames_evaluated,
            "frames_available": self.frames_available,
            "stages": stages,
        }


class EvalRunner:
    """Orchestrates per-stage evaluation from a pipeline diagnostic run directory.

    Discovers per-stage pickle caches in ``<run_dir>/diagnostics/``, loads each
    via :func:`~aquapose.core.context.load_stage_cache`, and calls the
    corresponding Phase 47 stage evaluator with the unpacked PipelineContext data.
    Stages whose cache files are missing are silently skipped.

    Example::

        runner = EvalRunner(run_dir)
        result = runner.run(n_frames=50)
        print(result.to_dict())
    """

    def __init__(self, run_dir: Path) -> None:
        """Initialize EvalRunner with a run directory path.

        Args:
            run_dir: Path to the pipeline run directory. Expected to contain
                a ``diagnostics/`` subdirectory with ``<stage>_cache.pkl`` files
                and a ``config.yaml`` file (required when association cache is
                present).
        """
        self._run_dir = Path(run_dir)

    def run(self, n_frames: int | None = None) -> EvalRunnerResult:
        """Discover caches, run evaluators, and return aggregated results.

        Args:
            n_frames: When not None, uniformly sample this many frames from the
                available frames before passing data to evaluators. When None,
                all frames are evaluated.

        Returns:
            EvalRunnerResult with per-stage metrics for all present stages.

        Raises:
            StaleCacheError: If any cache file is incompatible with the current
                codebase (propagated from load_stage_cache, not caught here).
            FileNotFoundError: If the association cache is present but
                config.yaml is missing from the run directory.
        """
        caches = self._discover_caches()

        if not caches:
            return EvalRunnerResult(
                run_id="",
                stages_present=frozenset(),
                detection=None,
                tracking=None,
                association=None,
                midline=None,
                reconstruction=None,
                frames_evaluated=0,
                frames_available=0,
            )

        # Determine run_id and frame_count from the first loaded cache.
        first_ctx = next(iter(caches.values()))
        run_id = getattr(first_ctx, "run_id", "") or ""
        frame_count = first_ctx.frame_count or 0

        # Determine sampled frame indices.
        if n_frames is not None and frame_count > 0:
            sampled_indices = select_frames(tuple(range(frame_count)), n_frames)
            frames_evaluated = len(sampled_indices)
        else:
            sampled_indices = list(range(frame_count))
            frames_evaluated = frame_count

        # Evaluate each present stage.
        detection_metrics: DetectionMetrics | None = None
        tracking_metrics: TrackingMetrics | None = None
        association_metrics: AssociationMetrics | None = None
        midline_metrics: MidlineMetrics | None = None
        reconstruction_metrics: ReconstructionMetrics | None = None

        if "detection" in caches:
            ctx = caches["detection"]
            frames = ctx.detections or []
            if sampled_indices and len(frames) > 0:
                frames = [frames[i] for i in sampled_indices if i < len(frames)]
            detection_metrics = evaluate_detection(frames)

        if "tracking" in caches:
            ctx = caches["tracking"]
            # Flatten all tracklets from all cameras into a single list.
            tracks_2d = ctx.tracks_2d or {}
            all_tracklets = [t for tracklets in tracks_2d.values() for t in tracklets]
            tracking_metrics = evaluate_tracking(all_tracklets)

        if "association" in caches:
            n_animals = self._read_n_animals()
            # Build MidlineSets from midline cache if available, else association cache.
            midline_sets_ctx = (
                caches.get("midline")
                or caches.get("reconstruction")
                or caches["association"]
            )
            midline_sets = self._build_midline_sets(midline_sets_ctx, sampled_indices)
            association_metrics = evaluate_association(midline_sets, n_animals)

        if "midline" in caches:
            ctx = caches["midline"]
            midline_sets_ctx = (
                caches.get("midline") or caches.get("reconstruction") or ctx
            )
            midline_sets = self._build_midline_sets(midline_sets_ctx, sampled_indices)
            # evaluate_midline takes list[dict[int, Midline2D]] — collapse per frame
            per_frame_midlines = [
                {
                    fish_id: cam_map[next(iter(cam_map))]
                    for fish_id, cam_map in ms.items()
                    if cam_map
                }
                for ms in midline_sets
            ]
            midline_metrics = evaluate_midline(per_frame_midlines)

        if "reconstruction" in caches:
            ctx = caches["reconstruction"]
            midlines_3d = ctx.midlines_3d or []
            frame_results = [
                (i, midlines_3d[i])
                for i in sampled_indices
                if i < len(midlines_3d) and midlines_3d[i]
            ]
            # fish_available: n_animals * frames_evaluated (or count from context)
            n_animals_recon = 0
            if "association" in caches:
                try:
                    n_animals_recon = self._read_n_animals()
                except FileNotFoundError:
                    n_animals_recon = 0
            fish_available = n_animals_recon * frames_evaluated
            reconstruction_metrics = evaluate_reconstruction(
                frame_results, fish_available
            )

        return EvalRunnerResult(
            run_id=run_id,
            stages_present=frozenset(caches.keys()),
            detection=detection_metrics,
            tracking=tracking_metrics,
            association=association_metrics,
            midline=midline_metrics,
            reconstruction=reconstruction_metrics,
            frames_evaluated=frames_evaluated,
            frames_available=frame_count,
        )

    def _discover_caches(self) -> dict[str, PipelineContext]:
        """Probe for each known stage cache file and load existing ones.

        Returns:
            Dict mapping stage_key to PipelineContext for each cache present.
            Missing cache files are silently skipped.

        Raises:
            StaleCacheError: Propagated from load_stage_cache when a cache
                file cannot be deserialized due to class evolution.
        """
        result: dict[str, PipelineContext] = {}
        diag_dir = self._run_dir / "diagnostics"

        for key in _STAGE_KEYS:
            cache_path = diag_dir / f"{key}_cache.pkl"
            if cache_path.exists():
                result[key] = load_stage_cache(cache_path)

        return result

    def _read_n_animals(self) -> int:
        """Read n_animals from config.yaml in the run directory.

        Returns:
            The n_animals value from the pipeline config.

        Raises:
            FileNotFoundError: If config.yaml does not exist in the run directory.
        """
        from aquapose.engine.config import load_config

        config_path = self._run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found in run directory '{self._run_dir}'. "
                f"This file is required when the association cache is present "
                f"to determine the expected number of animals (n_animals)."
            )
        config = load_config(config_path)
        return config.n_animals

    def _build_midline_sets(
        self,
        ctx: PipelineContext,
        sampled_indices: list[int],
    ) -> list[MidlineSet]:
        """Build per-frame MidlineSets from PipelineContext data.

        Assembles MidlineSets by matching TrackletGroup tracklet centroids to
        AnnotatedDetection midlines using a centroid proximity threshold.

        Args:
            ctx: PipelineContext from the midline (or reconstruction) stage cache.
                Must have both ``tracklet_groups`` and ``annotated_detections``.
            sampled_indices: Frame indices to include. Empty list means all frames.

        Returns:
            List of MidlineSet dicts (fish_id -> camera_id -> Midline2D), one
            entry per sampled frame index. Frames with no data yield empty dicts.
        """
        tracklet_groups = ctx.tracklet_groups or []
        annotated_detections = ctx.annotated_detections or []
        frame_count = ctx.frame_count or len(annotated_detections)

        effective_indices = (
            sampled_indices if sampled_indices else list(range(frame_count))
        )

        # Accumulate: frame_idx -> fish_id -> camera_id -> Midline2D
        collected: dict[int, MidlineSet] = {}

        for group in tracklet_groups:
            if not group.tracklets:
                continue
            fish_id = group.fish_id
            for tracklet in group.tracklets:
                cam_id: str = tracklet.camera_id  # type: ignore[union-attr]
                for tidx, (frame_idx, status) in enumerate(
                    zip(
                        tracklet.frames,  # type: ignore[union-attr]
                        tracklet.frame_status,  # type: ignore[union-attr]
                        strict=False,
                    )
                ):
                    if status != "detected":
                        continue
                    if frame_idx not in set(effective_indices):
                        continue
                    if frame_idx >= len(annotated_detections):
                        continue

                    frame_annot = annotated_detections[frame_idx]
                    if not isinstance(frame_annot, dict):
                        continue
                    cam_list = frame_annot.get(cam_id)
                    if not cam_list:
                        continue

                    centroid: tuple[float, float] = tracklet.centroids[tidx]  # type: ignore[union-attr]
                    ann_det = _match_annotated_by_centroid(cam_list, centroid)
                    if ann_det is None:
                        continue
                    midline = cast(Midline2D, ann_det.midline)  # type: ignore[union-attr]
                    if midline is None:
                        continue

                    collected.setdefault(frame_idx, {}).setdefault(fish_id, {})[
                        cam_id
                    ] = midline

        # Build result list in sampled frame order.
        result = []
        for frame_idx in effective_indices:
            result.append(collected.get(frame_idx, {}))

        return result


def _match_annotated_by_centroid(
    frame_annotated_for_cam: list,
    centroid: tuple[float, float],
) -> object | None:
    """Find the AnnotatedDetection whose Detection centroid is nearest to centroid.

    Args:
        frame_annotated_for_cam: List of AnnotatedDetection objects for a
            single (frame, camera) pair.
        centroid: ``(u, v)`` pixel coordinate from a Tracklet2D.

    Returns:
        The closest AnnotatedDetection within ``_CENTROID_MATCH_TOLERANCE_PX``
        pixels, or None if no match is found.
    """
    cx, cy = centroid
    best = None
    best_dist = _CENTROID_MATCH_TOLERANCE_PX

    for ann_det in frame_annotated_for_cam:
        x, y, w, h = ann_det.detection.bbox
        det_cx = x + w / 2.0
        det_cy = y + h / 2.0
        dist = math.sqrt((det_cx - cx) ** 2 + (det_cy - cy) ** 2)
        if dist <= best_dist:
            best_dist = dist
            best = ann_det

    return best


__all__ = ["EvalRunner", "EvalRunnerResult"]
