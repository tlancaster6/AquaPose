"""EvalRunner orchestrates per-stage evaluation from diagnostic run directories."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from aquapose.core.context import PipelineContext, load_chunk_cache
from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import MidlineSet
from aquapose.evaluation.metrics import select_frames
from aquapose.evaluation.stages import (
    AssociationMetrics,
    DetectionMetrics,
    FragmentationMetrics,
    MidlineMetrics,
    ReconstructionMetrics,
    SmoothingMetrics,
    StitchingMetrics,
    TrackingMetrics,
    evaluate_association,
    evaluate_detection,
    evaluate_fragmentation,
    evaluate_midline,
    evaluate_reconstruction,
    evaluate_smoothing,
    evaluate_stitching,
    evaluate_tracking,
)
from aquapose.evaluation.stages.reconstruction import (
    compute_curvature_stratified,
    compute_per_point_error,
)

# Centroid match tolerance in pixels for TrackletGroup -> AnnotatedDetection matching.
_CENTROID_MATCH_TOLERANCE_PX = 5.0


def load_run_context(
    run_dir: Path,
) -> tuple[PipelineContext | None, dict]:
    """Discover, load, and merge all chunk caches from a diagnostic run directory.

    Reads ``diagnostics/manifest.json`` if present to get chunk list and metadata.
    Falls back to globbing ``diagnostics/chunk_*/cache.pkl`` if no manifest is found.
    Merges all loaded chunk contexts into a single synthetic PipelineContext with
    correct frame offsets applied.

    Args:
        run_dir: Path to the pipeline run directory. Expected to contain a
            ``diagnostics/`` subdirectory with ``chunk_NNN/cache.pkl`` files.

    Returns:
        A tuple of (merged_context, metadata) where:
        - merged_context is a single PipelineContext spanning all chunks, or
          None if no chunks are found.
        - metadata is the manifest dict (or an empty dict if no manifest).

    Raises:
        StaleCacheError: Propagated from load_chunk_cache when a cache file
            cannot be deserialized.
    """
    run_dir = Path(run_dir)
    diag_dir = run_dir / "diagnostics"

    if not diag_dir.exists():
        return None, {}

    # Try to read manifest.json first
    manifest: dict = {}
    manifest_path = diag_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}

    # Discover chunk cache paths
    chunk_paths: list[
        tuple[int, int | None, Path]
    ] = []  # (chunk_idx, start_frame, path)

    if manifest and "chunks" in manifest:
        for chunk_entry in sorted(manifest["chunks"], key=lambda c: c.get("index", 0)):
            chunk_idx = chunk_entry.get("index", 0)
            start_frame = chunk_entry.get("start_frame")
            chunk_dir = diag_dir / f"chunk_{chunk_idx:03d}"
            cache_path = chunk_dir / "cache.pkl"
            if cache_path.exists():
                chunk_paths.append((chunk_idx, start_frame, cache_path))
    else:
        # Fallback: glob for chunk_*/cache.pkl sorted by directory name
        found = sorted(diag_dir.glob("chunk_*/cache.pkl"), key=lambda p: p.parent.name)
        for cache_path in found:
            # Parse chunk index from directory name (chunk_NNN)
            dir_name = cache_path.parent.name
            try:
                chunk_idx = int(dir_name.split("_")[1])
            except (IndexError, ValueError):
                chunk_idx = 0
            chunk_paths.append((chunk_idx, None, cache_path))

    if not chunk_paths:
        return None, manifest

    # Load each chunk context
    loaded_chunks: list[tuple[int, int | None, PipelineContext]] = []
    for chunk_idx, start_frame, cache_path in chunk_paths:
        ctx = load_chunk_cache(cache_path)
        loaded_chunks.append((chunk_idx, start_frame, ctx))

    if len(loaded_chunks) == 1:
        # Single chunk: return as-is (no merge needed)
        _, _, ctx = loaded_chunks[0]
        return ctx, manifest

    # Multi-chunk: compute start_frame offsets if not provided by manifest
    resolved_chunks: list[tuple[int, PipelineContext]] = []
    current_offset = 0
    for _chunk_idx, start_frame, ctx in loaded_chunks:
        if start_frame is None:
            start_frame = current_offset
        resolved_chunks.append((start_frame, ctx))
        current_offset = start_frame + (ctx.frame_count or 0)

    merged = _merge_chunk_contexts(resolved_chunks)
    return merged, manifest


def _merge_chunk_contexts(
    chunks: list[tuple[int, PipelineContext]],
) -> PipelineContext:
    """Merge multiple chunk PipelineContexts into a single synthetic context.

    Frame indices in per-frame lists and tracklet frames are offset by each
    chunk's start_frame so that they are globally consistent.

    Args:
        chunks: List of (start_frame, PipelineContext) tuples sorted by
            start_frame ascending.

    Returns:
        A single PipelineContext with all chunk data merged. camera_ids is
        taken from the first chunk.
    """
    # Determine camera_ids from the first chunk
    camera_ids = chunks[0][1].camera_ids if chunks else None

    # Merge per-frame lists: detections, midlines_3d
    merged_detections: list | None = None
    merged_midlines_3d: list | None = None

    for start_frame, ctx in chunks:
        if ctx.detections is not None:
            if merged_detections is None:
                merged_detections = []
            merged_detections.extend(ctx.detections)

        if ctx.midlines_3d is not None:
            if merged_midlines_3d is None:
                merged_midlines_3d = []
            for frame_entry in ctx.midlines_3d:
                if frame_entry is None or start_frame == 0:
                    merged_midlines_3d.append(frame_entry)
                else:
                    offset_entry = {}
                    for fish_id, m3d in frame_entry.items():
                        offset_entry[fish_id] = dataclasses.replace(
                            m3d, frame_index=m3d.frame_index + start_frame
                        )
                    merged_midlines_3d.append(offset_entry)

    # Merge tracks_2d: extend per-camera tracklet lists with frame-offset tracklets
    merged_tracks_2d: dict | None = None
    for start_frame, ctx in chunks:
        if ctx.tracks_2d is None:
            continue
        if merged_tracks_2d is None:
            merged_tracks_2d = {}
        for cam_id, tracklets in ctx.tracks_2d.items():
            if cam_id not in merged_tracks_2d:
                merged_tracks_2d[cam_id] = []
            for tracklet in tracklets:
                merged_tracks_2d[cam_id].append(
                    _offset_tracklet_frames(tracklet, start_frame)
                )

    # Merge tracklet_groups: concatenate with frame-offset tracklets
    merged_tracklet_groups: list | None = None
    for start_frame, ctx in chunks:
        if ctx.tracklet_groups is None:
            continue
        if merged_tracklet_groups is None:
            merged_tracklet_groups = []
        for group in ctx.tracklet_groups:
            merged_tracklet_groups.append(_offset_group_frames(group, start_frame))

    total_frames = sum(ctx.frame_count or 0 for _, ctx in chunks)

    return PipelineContext(
        frame_count=total_frames,
        camera_ids=camera_ids,
        detections=merged_detections,
        tracks_2d=merged_tracks_2d,
        tracklet_groups=merged_tracklet_groups,
        midlines_3d=merged_midlines_3d,
    )


def _offset_tracklet_frames(tracklet: Any, offset: int) -> Any:
    """Return a copy of a Tracklet2D with frame indices shifted by offset.

    Args:
        tracklet: A Tracklet2D instance.
        offset: Frame index offset to add to each frame index.

    Returns:
        A new Tracklet2D with frames shifted by offset.
    """
    import dataclasses

    frames = getattr(tracklet, "frames", None)
    if frames is None or offset == 0:
        return tracklet
    new_frames = tuple(f + offset for f in frames)
    return dataclasses.replace(tracklet, frames=new_frames)


def _offset_group_frames(group: Any, offset: int) -> Any:
    """Return a copy of a TrackletGroup with all tracklet frame indices shifted.

    Args:
        group: A TrackletGroup instance.
        offset: Frame index offset to add.

    Returns:
        A new TrackletGroup with all inner tracklet frames shifted by offset.
    """
    import dataclasses

    if offset == 0:
        return group
    tracklets = getattr(group, "tracklets", None)
    if not tracklets:
        return group
    new_tracklets = tuple(_offset_tracklet_frames(t, offset) for t in tracklets)
    return dataclasses.replace(group, tracklets=new_tracklets)


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
    fragmentation: FragmentationMetrics | None = None
    stitching: StitchingMetrics | None = None
    smoothing: SmoothingMetrics | None = None

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
        if self.fragmentation is not None:
            stages["fragmentation"] = self.fragmentation.to_dict()
        if self.stitching is not None:
            stages["stitching"] = self.stitching.to_dict()
        if self.smoothing is not None:
            stages["smoothing"] = self.smoothing.to_dict()

        return {
            "run_id": self.run_id,
            "stages_present": sorted(self.stages_present),
            "frames_evaluated": self.frames_evaluated,
            "frames_available": self.frames_available,
            "stages": stages,
        }


class EvalRunner:
    """Orchestrates per-stage evaluation from a pipeline diagnostic run directory.

    Discovers per-chunk cache files in ``<run_dir>/diagnostics/chunk_NNN/cache.pkl``,
    loads and merges them via :func:`load_run_context`, and calls the corresponding
    stage evaluator with the unpacked PipelineContext data. Stages whose data is
    absent in the merged context are silently skipped.

    Example::

        runner = EvalRunner(run_dir)
        result = runner.run(n_frames=50)
        print(result.to_dict())
    """

    def __init__(self, run_dir: Path) -> None:
        """Initialize EvalRunner with a run directory path.

        Args:
            run_dir: Path to the pipeline run directory. Expected to contain
                a ``diagnostics/`` subdirectory with ``chunk_NNN/cache.pkl`` files
                and a ``config.yaml`` file (required when association data is
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
                codebase (propagated from load_chunk_cache, not caught here).
            FileNotFoundError: If the association data is present but
                config.yaml is missing from the run directory.
        """
        ctx, _manifest = load_run_context(self._run_dir)

        if ctx is None:
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

        run_id = getattr(ctx, "run_id", "") or ""
        frame_count = ctx.frame_count or 0

        # Determine which stages are present from the merged context
        stages_present: set[str] = set()
        if ctx.detections is not None:
            stages_present.add("detection")
        if ctx.tracks_2d is not None:
            stages_present.add("tracking")
        if ctx.tracklet_groups is not None:
            stages_present.add("association")
        if ctx.midlines_3d is not None:
            stages_present.add("reconstruction")

        # Determine sampled frame indices.
        if n_frames is not None and frame_count > 0:
            sampled_indices = select_frames(tuple(range(frame_count)), n_frames)
            frames_evaluated = len(sampled_indices)
        else:
            sampled_indices = list(range(frame_count))
            frames_evaluated = frame_count

        # Read n_sample_points from run config (used for per-point eval)
        n_sample_points = 6
        try:
            from aquapose.engine.config import load_config

            config_path = self._run_dir / "config.yaml"
            if config_path.exists():
                run_config = load_config(config_path)
                n_sample_points = run_config.reconstruction.n_sample_points
        except Exception:
            pass

        # Evaluate each present stage.
        detection_metrics: DetectionMetrics | None = None
        tracking_metrics: TrackingMetrics | None = None
        association_metrics: AssociationMetrics | None = None
        midline_metrics: MidlineMetrics | None = None
        reconstruction_metrics: ReconstructionMetrics | None = None
        fragmentation_metrics: FragmentationMetrics | None = None

        if "detection" in stages_present:
            frames = ctx.detections or []
            if sampled_indices and len(frames) > 0:
                frames = [frames[i] for i in sampled_indices if i < len(frames)]
            detection_metrics = evaluate_detection(frames)

        if "tracking" in stages_present:
            tracks_2d = ctx.tracks_2d or {}
            all_tracklets = [t for tracklets in tracks_2d.values() for t in tracklets]
            tracking_metrics = evaluate_tracking(all_tracklets)

        if "association" in stages_present:
            n_animals = self._read_n_animals()
            midline_sets = self._build_midline_sets(ctx, sampled_indices)
            association_metrics = evaluate_association(midline_sets, n_animals)

        if "midline" in stages_present:
            midline_sets = self._build_midline_sets(ctx, sampled_indices)
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

        # These are populated by reconstruction eval and reused by smoothing eval.
        frame_results: list[tuple[int, dict]] | None = None
        midline_sets_by_frame: dict[int, MidlineSet] | None = None
        projection_models: dict[str, Any] | None = None

        if "reconstruction" in stages_present:
            midlines_3d = ctx.midlines_3d or []
            frame_results = [
                (i, midlines_3d[i])
                for i in sampled_indices
                if i < len(midlines_3d) and midlines_3d[i]
            ]
            n_animals_recon = 0
            if "association" in stages_present:
                try:
                    n_animals_recon = self._read_n_animals()
                except FileNotFoundError:
                    n_animals_recon = 0
            fish_available = n_animals_recon * frames_evaluated
            # Per-keypoint error and curvature stratification (EVAL-04, EVAL-05)
            per_point_result = None
            curvature_result = None

            # Build midline_sets_by_frame for per-keypoint and curvature analysis
            midline_sets_for_recon = self._build_midline_sets(ctx, sampled_indices)
            midline_sets_by_frame = {}
            for idx, ms in zip(sampled_indices, midline_sets_for_recon, strict=False):
                if ms:
                    midline_sets_by_frame[idx] = ms

            # Try loading projection models for per-keypoint error
            projection_models = self._load_projection_models()
            if projection_models:
                per_point_result = compute_per_point_error(
                    frame_results,
                    midline_sets_by_frame,
                    projection_models,
                    n_body_points=n_sample_points,
                )

            # Curvature stratification (no projection models needed)
            curvature_result = compute_curvature_stratified(
                frame_results, midline_sets_by_frame
            )

            reconstruction_metrics = evaluate_reconstruction(
                frame_results,
                fish_available,
                per_point_error=per_point_result,
                curvature_stratified=curvature_result,
            )

            # Fragmentation analysis from 3D midline data
            n_animals_frag = 0
            if "association" in stages_present:
                try:
                    n_animals_frag = self._read_n_animals()
                except FileNotFoundError:
                    n_animals_frag = 0
            fragmentation_metrics = evaluate_fragmentation(midlines_3d, n_animals_frag)

        # Stitch-quality evaluation (reads from H5, independent of cache.pkl)
        stitching_metrics: StitchingMetrics | None = None
        try:
            n_animals_stitch = self._read_n_animals()
            stitching_metrics = evaluate_stitching(self._run_dir, n_animals_stitch)
        except FileNotFoundError:
            pass

        # Smoothing-quality evaluation (reads from smoothed H5, reuses recon data)
        smoothing_metrics = evaluate_smoothing(
            self._run_dir,
            frame_results,
            midline_sets_by_frame,
            projection_models,
            n_body_points=n_sample_points,
        )

        return EvalRunnerResult(
            run_id=run_id,
            stages_present=frozenset(stages_present),
            detection=detection_metrics,
            tracking=tracking_metrics,
            association=association_metrics,
            midline=midline_metrics,
            reconstruction=reconstruction_metrics,
            frames_evaluated=frames_evaluated,
            frames_available=frame_count,
            fragmentation=fragmentation_metrics,
            stitching=stitching_metrics,
            smoothing=smoothing_metrics,
        )

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

    def _load_projection_models(self) -> dict[str, Any]:
        """Load refractive projection models from the run's config.yaml.

        Attempts to load calibration data and build RefractiveProjectionModel
        instances for each camera. Returns an empty dict on failure (graceful
        degradation).

        Returns:
            Dict mapping camera_id to RefractiveProjectionModel, or empty dict
            if calibration data is unavailable.
        """
        try:
            from aquapose.calibration import load_calibration_data
            from aquapose.calibration.loader import compute_undistortion_maps
            from aquapose.calibration.projection import RefractiveProjectionModel
            from aquapose.engine.config import load_config

            config_path = self._run_dir / "config.yaml"
            if not config_path.exists():
                return {}
            config = load_config(config_path)
            cal_data = load_calibration_data(config.calibration_path)
            models: dict[str, Any] = {}
            for cam_id, cam_data in cal_data.cameras.items():
                undist_maps = compute_undistortion_maps(cam_data)
                models[cam_id] = RefractiveProjectionModel(
                    K=undist_maps.K_new,
                    R=cam_data.R,
                    t=cam_data.t,
                    water_z=cal_data.water_z,
                    normal=cal_data.interface_normal,
                    n_air=cal_data.n_air,
                    n_water=cal_data.n_water,
                )
            return models
        except Exception:
            return {}

    def _build_midline_sets(
        self,
        ctx: PipelineContext,
        sampled_indices: list[int],
    ) -> list[MidlineSet]:
        """Build per-frame MidlineSets from PipelineContext data.

        Assembles MidlineSets from TrackletGroup keypoints (v3.7+) or by
        matching centroids to AnnotatedDetection midlines (legacy).

        Args:
            ctx: PipelineContext with tracklet_groups carrying keypoints (v3.7)
                or annotated_detections (legacy).
            sampled_indices: Frame indices to include. Empty list means all frames.

        Returns:
            List of MidlineSet dicts (fish_id -> camera_id -> Midline2D), one
            entry per sampled frame index. Frames with no data yield empty dicts.
        """
        import numpy as np

        from aquapose.core.reconstruction.stage import (
            _DEFAULT_KEYPOINT_T_VALUES,
            _keypoints_to_midline,
        )

        # Read keypoint_t_values and n_sample_points from run config if available
        keypoint_t_values = _DEFAULT_KEYPOINT_T_VALUES
        n_sample_points = 6
        try:
            from aquapose.engine.config import load_config

            config_path = self._run_dir / "config.yaml"
            if config_path.exists():
                run_config = load_config(config_path)
                if run_config.pose.keypoint_t_values is not None:
                    keypoint_t_values = np.array(
                        run_config.pose.keypoint_t_values, dtype=np.float64
                    )
                n_sample_points = run_config.reconstruction.n_sample_points
        except Exception:
            pass  # Fall back to defaults

        tracklet_groups = ctx.tracklet_groups or []

        # v3.7: keypoints live on Tracklet2D objects directly
        # Legacy: midlines live on AnnotatedDetection objects in ctx.annotated_detections
        annotated_detections: list = getattr(ctx, "annotated_detections", None) or []
        use_tracklet_keypoints = not annotated_detections

        frame_count = ctx.frame_count or len(annotated_detections)

        effective_indices = (
            sampled_indices if sampled_indices else list(range(frame_count))
        )
        effective_set = set(effective_indices)

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
                    if frame_idx not in effective_set:
                        continue

                    if use_tracklet_keypoints:
                        # v3.7 path: read keypoints directly from tracklet
                        if tracklet.keypoints is None:  # type: ignore[union-attr]
                            continue
                        kpts_xy = tracklet.keypoints[tidx]  # type: ignore[union-attr]
                        kpts_conf = (
                            tracklet.keypoint_conf[tidx]  # type: ignore[union-attr]
                            if tracklet.keypoint_conf is not None  # type: ignore[union-attr]
                            else np.ones(kpts_xy.shape[0], dtype=np.float32)
                        )
                        points, point_conf = _keypoints_to_midline(
                            kpts_xy,
                            keypoint_t_values,
                            kpts_conf,
                            n_points=n_sample_points,
                        )
                        midline = Midline2D(
                            points=points,
                            half_widths=np.zeros(len(points), dtype=np.float32),
                            fish_id=fish_id,
                            camera_id=cam_id,
                            frame_index=frame_idx,
                            point_confidence=point_conf,
                        )
                    else:
                        # Legacy path: match AnnotatedDetection by centroid, use .midline
                        centroid: tuple[float, float] = tracklet.centroids[tidx]  # type: ignore[union-attr]
                        if frame_idx >= len(annotated_detections):
                            continue
                        frame_annot = annotated_detections[frame_idx]
                        if not isinstance(frame_annot, dict):
                            continue
                        cam_list = frame_annot.get(cam_id)
                        if not cam_list:
                            continue

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

    Legacy function retained for backward compatibility with old diagnostic runs.

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


__all__ = ["EvalRunner", "EvalRunnerResult", "load_run_context"]
