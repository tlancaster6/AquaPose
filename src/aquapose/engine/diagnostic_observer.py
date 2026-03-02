"""Diagnostic observer for capturing intermediate stage outputs in memory."""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aquapose.engine.events import Event, PipelineComplete, StageComplete

_NPZ_VERSION_V1 = "1.0"

logger = logging.getLogger(__name__)

_ASSOCIATION_STAGE_NAME = "AssociationStage"
_MIDLINE_STAGE_NAME = "MidlineStage"
_CENTROID_MATCH_TOLERANCE_PX = 5.0

# PipelineContext field names that hold per-frame data (list-typed, indexed by frame).
# detections, annotated_detections, and midlines_3d are all list[...] indexed by frame.
_PER_FRAME_FIELDS = (
    "detections",
    "annotated_detections",
    "midlines_3d",
)

# PipelineContext field names that hold scalar or non-frame-indexed data.
# tracks_2d is dict[str, list[Tracklet2D]] (keyed by camera, not frame).
# tracklet_groups is list[TrackletGroup] (flat cross-batch list, not frame-indexed).
_SCALAR_FIELDS = (
    "frame_count",
    "camera_ids",
    "tracks_2d",
    "tracklet_groups",
)


@dataclass
class StageSnapshot:
    """Immutable snapshot of PipelineContext state after a stage completes.

    Stores *references* (not deep copies) to PipelineContext fields, relying
    on the freeze-on-populate invariant for correctness.

    Subscript access (``snapshot[frame_idx]``) returns a dict of all non-None
    per-frame fields at that index, enabling convenient exploration in Jupyter
    notebooks.

    Attributes:
        stage_name: Name of the stage that produced this snapshot.
        stage_index: Zero-based position in the pipeline sequence.
        elapsed_seconds: Wall-clock time for this stage.
        frame_count: Number of frames (from PipelineContext), or None.
        camera_ids: Active camera IDs (from PipelineContext), or None.
        detections: Reference to PipelineContext.detections, or None.
        annotated_detections: Reference to PipelineContext.annotated_detections, or None.
        tracks_2d: Reference to PipelineContext.tracks_2d (dict, not per-frame), or None.
        tracklet_groups: Reference to PipelineContext.tracklet_groups (flat list), or None.
        midlines_3d: Reference to PipelineContext.midlines_3d, or None.
    """

    stage_name: str = ""
    stage_index: int = 0
    elapsed_seconds: float = 0.0
    frame_count: int | None = None
    camera_ids: list[str] | None = None
    detections: list | None = None
    annotated_detections: list | None = None
    tracks_2d: dict | None = None
    tracklet_groups: list | None = None
    midlines_3d: list | None = None

    # Keep a frozen set of per-frame field names for __getitem__.
    _per_frame_fields: tuple[str, ...] = field(
        default=_PER_FRAME_FIELDS, init=False, repr=False, compare=False
    )

    def __getitem__(self, frame_idx: int) -> dict[str, object]:
        """Return a dict of all non-None per-frame fields at *frame_idx*.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            Dict mapping field name to the value at ``frame_idx``.

        Raises:
            IndexError: If *frame_idx* is out of range for any field.
        """
        result: dict[str, object] = {}
        for name in self._per_frame_fields:
            value = getattr(self, name, None)
            if value is not None and isinstance(value, list):
                result[name] = value[frame_idx]
        return result


class DiagnosticObserver:
    """Captures intermediate stage outputs in memory for post-hoc analysis.

    After each stage completes, the observer takes a snapshot of the
    PipelineContext (by reference, not deep copy) and stores it keyed by
    stage name. All 5 stages are captured without selective filtering.

    Designed for interactive exploration in Jupyter notebooks::

        observer = DiagnosticObserver()
        pipeline = PosePipeline(stages=stages, config=config, observers=[observer])
        context = pipeline.run()

        # Explore detection stage results
        snapshot = observer.stages["DetectionStage"]
        frame_0 = snapshot[0]  # dict of per-frame fields
        print(frame_0["detections"])
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.stages: dict[str, StageSnapshot] = {}
        self._output_dir = Path(output_dir) if output_dir is not None else None

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and capture stage snapshots.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if isinstance(event, PipelineComplete):
            self._on_pipeline_complete()
            return

        if not isinstance(event, StageComplete):
            return

        context = event.context
        if context is None:
            return

        snapshot = StageSnapshot(
            stage_name=event.stage_name,
            stage_index=event.stage_index,
            elapsed_seconds=event.elapsed_seconds,
            frame_count=getattr(context, "frame_count", None),
            camera_ids=getattr(context, "camera_ids", None),
            detections=getattr(context, "detections", None),
            annotated_detections=getattr(context, "annotated_detections", None),
            tracks_2d=getattr(context, "tracks_2d", None),
            tracklet_groups=getattr(context, "tracklet_groups", None),
            midlines_3d=getattr(context, "midlines_3d", None),
        )

        self.stages[event.stage_name] = snapshot

    def _on_pipeline_complete(self) -> None:
        """Auto-export centroid correspondences and midline fixtures when output_dir is set."""
        if self._output_dir is None:
            return

        # --- centroid correspondences ---
        if _ASSOCIATION_STAGE_NAME in self.stages:
            snapshot = self.stages[_ASSOCIATION_STAGE_NAME]
            if snapshot.tracklet_groups:
                try:
                    out = self.export_centroid_correspondences(
                        self._output_dir / "centroid_correspondences.npz"
                    )
                    logger.info("Centroid correspondences exported to %s", out)
                except Exception:
                    logger.warning(
                        "Failed to export centroid correspondences", exc_info=True
                    )

        # --- midline fixtures ---
        if (
            _ASSOCIATION_STAGE_NAME in self.stages
            and _MIDLINE_STAGE_NAME in self.stages
        ):
            try:
                out_mid = self.export_midline_fixtures(
                    self._output_dir / "midline_fixtures.npz"
                )
                logger.info("Midline fixtures exported to %s", out_mid)
            except Exception:
                logger.warning("Failed to export midline fixtures", exc_info=True)

    # ------------------------------------------------------------------
    # Midline fixture export
    # ------------------------------------------------------------------

    @staticmethod
    def _match_annotated_by_centroid(
        frame_annotated_for_cam: list,
        centroid: tuple[float, float],
    ) -> object | None:
        """Find the AnnotatedDetection whose Detection centroid is nearest to *centroid*.

        Scans *frame_annotated_for_cam* (a ``list[AnnotatedDetection]``) and
        returns the element whose bounding-box centre is within
        ``_CENTROID_MATCH_TOLERANCE_PX`` pixels of *centroid*.  Returns
        ``None`` if the list is empty or no element is within tolerance.

        Args:
            frame_annotated_for_cam: List of AnnotatedDetection objects for a
                single (frame, camera) pair.
            centroid: ``(u, v)`` pixel coordinate from a Tracklet2D.

        Returns:
            The closest AnnotatedDetection within tolerance, or None.
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

    def export_midline_fixtures(self, output_path: Path | str) -> Path:
        """Assemble per-frame MidlineSets from snapshot data and write an NPZ fixture.

        Builds a ``dict[frame_idx, dict[fish_id, dict[camera_id, Midline2D]]]``
        by matching tracklet centroids to AnnotatedDetections and extracting their
        ``.midline`` field.  Frames with at least one valid midline are serialised
        to the compressed NPZ using the key convention documented in
        ``aquapose.io.midline_fixture``.

        Args:
            output_path: Destination file path for the NPZ output.

        Returns:
            Resolved absolute path to the written NPZ file.

        Raises:
            ValueError: If either ``AssociationStage`` or ``MidlineStage``
                snapshots are missing.
        """
        if _ASSOCIATION_STAGE_NAME not in self.stages:
            raise ValueError(
                "No AssociationStage snapshot found. "
                "Run the pipeline before calling export_midline_fixtures()."
            )
        if _MIDLINE_STAGE_NAME not in self.stages:
            raise ValueError(
                "No MidlineStage snapshot found. "
                "Run the pipeline before calling export_midline_fixtures()."
            )

        assoc_snap = self.stages[_ASSOCIATION_STAGE_NAME]
        mid_snap = self.stages[_MIDLINE_STAGE_NAME]

        tracklet_groups = assoc_snap.tracklet_groups or []
        annotated_detections = mid_snap.annotated_detections or []
        camera_ids_list: list[str] = list(mid_snap.camera_ids or [])
        frame_count: int = mid_snap.frame_count or len(annotated_detections)

        # Accumulate per-frame MidlineSet data.
        # collected: frame_idx -> fish_id -> camera_id -> Midline2D
        collected: dict[int, dict[int, dict[str, object]]] = {}
        all_camera_ids: set[str] = set()

        for group in tracklet_groups:
            if group.tracklets is None:
                continue

            # Build per-tracklet frame membership: frame_idx -> (camera_id, local_idx)
            # Only "detected" frames contribute.
            for tracklet in group.tracklets:  # type: ignore[union-attr]
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
                    if frame_idx >= len(annotated_detections):
                        continue

                    frame_annot = annotated_detections[frame_idx]
                    if not isinstance(frame_annot, dict):
                        continue
                    cam_list = frame_annot.get(cam_id)
                    if not cam_list:
                        continue

                    centroid: tuple[float, float] = tracklet.centroids[tidx]  # type: ignore[union-attr]
                    ann_det = self._match_annotated_by_centroid(cam_list, centroid)
                    if ann_det is None:
                        continue
                    midline = ann_det.midline  # type: ignore[union-attr]
                    if midline is None:
                        continue

                    # Store result
                    all_camera_ids.add(cam_id)
                    fish_id = group.fish_id
                    collected.setdefault(frame_idx, {}).setdefault(fish_id, {})[
                        cam_id
                    ] = midline

        # Build flat NPZ arrays using the documented key convention.
        # Use object-typed dict to allow mixed array scalars alongside nd-arrays.
        npz_arrays: dict[str, object] = {}
        sorted_frame_indices = sorted(collected.keys())

        # Meta arrays
        timestamp_str = datetime.datetime.now(datetime.UTC).isoformat()
        npz_arrays["meta/version"] = np.array(_NPZ_VERSION_V1, dtype=object)
        npz_arrays["meta/camera_ids"] = np.array(
            camera_ids_list if camera_ids_list else sorted(all_camera_ids), dtype=object
        )
        npz_arrays["meta/frame_indices"] = np.array(
            sorted_frame_indices, dtype=np.int64
        )
        npz_arrays["meta/frame_count"] = np.array(frame_count, dtype=np.int64)
        npz_arrays["meta/timestamp"] = np.array(timestamp_str, dtype=object)

        # Midline data arrays (one set per fish x camera x frame)
        for frame_idx in sorted_frame_indices:
            fish_map = collected[frame_idx]
            for fish_id, cam_map in fish_map.items():
                for cam_id, midline in cam_map.items():
                    prefix = f"midline/{frame_idx}/{fish_id}/{cam_id}"
                    npz_arrays[f"{prefix}/points"] = midline.points.astype(  # type: ignore[union-attr]
                        np.float32
                    )
                    npz_arrays[f"{prefix}/half_widths"] = midline.half_widths.astype(  # type: ignore[union-attr]
                        np.float32
                    )
                    if midline.point_confidence is not None:  # type: ignore[union-attr]
                        conf = midline.point_confidence.astype(np.float32)  # type: ignore[union-attr]
                    else:
                        n = midline.points.shape[0]  # type: ignore[union-attr]
                        conf = np.ones(n, dtype=np.float32)
                    npz_arrays[f"{prefix}/point_confidence"] = conf
                    npz_arrays[f"{prefix}/is_head_to_tail"] = np.array(
                        midline.is_head_to_tail,  # type: ignore[union-attr]
                        dtype=np.bool_,
                    )

        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out), **npz_arrays)  # type: ignore[arg-type]
        return out

    def export_centroid_correspondences(self, output_path: Path | str) -> Path:
        """Export 2D-to-3D centroid correspondences from the Association stage.

        Iterates over all TrackletGroups that have non-None ``consensus_centroids``,
        and for each (frame_idx, point_3d) pair with a valid 3D point, collects
        the 2D pixel centroid from each contributing tracklet that has the frame.
        Saves results as a compressed NPZ file for calibration fine-tuning.

        Args:
            output_path: Destination file path for the NPZ output.

        Returns:
            Resolved absolute path to the written NPZ file.

        Raises:
            ValueError: If no AssociationStage snapshot is found, or if the
                snapshot contains no tracklet groups.
        """
        if _ASSOCIATION_STAGE_NAME not in self.stages:
            raise ValueError(
                "No AssociationStage snapshot found. "
                "Run the pipeline before calling export_centroid_correspondences()."
            )

        snapshot = self.stages[_ASSOCIATION_STAGE_NAME]
        if not snapshot.tracklet_groups:
            raise ValueError(
                "AssociationStage snapshot has no tracklet_groups. "
                "Ensure the pipeline ran with a populated association stage."
            )

        fish_ids_list: list[int] = []
        frame_indices_list: list[int] = []
        points_3d_list: list[np.ndarray] = []
        camera_ids_list: list[str] = []
        centroids_2d_list: list[tuple[float, float]] = []

        for group in snapshot.tracklet_groups:
            if group.consensus_centroids is None:
                continue

            # Build per-tracklet frame-to-centroid lookup.
            # group.tracklets is typed as generic tuple to preserve the core/ import
            # boundary; elements are Tracklet2D at runtime (see TrackletGroup docstring).
            tracklet_frame_maps: list[
                tuple[object, dict[int, tuple[float, float]]]
            ] = []
            for tracklet in group.tracklets:  # type: ignore[union-attr]
                frame_map: dict[int, tuple[float, float]] = {
                    f: tracklet.centroids[i]  # type: ignore[union-attr]
                    for i, f in enumerate(tracklet.frames)  # type: ignore[union-attr]
                }
                tracklet_frame_maps.append((tracklet, frame_map))

            for frame_idx, point_3d in group.consensus_centroids:
                if point_3d is None:
                    continue

                for tracklet, frame_map in tracklet_frame_maps:
                    if frame_idx not in frame_map:
                        continue
                    u, v = frame_map[frame_idx]
                    fish_ids_list.append(group.fish_id)
                    frame_indices_list.append(frame_idx)
                    points_3d_list.append(point_3d)
                    camera_ids_list.append(tracklet.camera_id)  # type: ignore[union-attr]
                    centroids_2d_list.append((u, v))

        n = len(fish_ids_list)
        fish_ids_arr = np.array(fish_ids_list, dtype=np.int64)
        frame_indices_arr = np.array(frame_indices_list, dtype=np.int64)
        points_3d_arr = (
            np.stack(points_3d_list, axis=0).astype(np.float64)
            if n > 0
            else np.empty((0, 3), dtype=np.float64)
        )
        camera_ids_arr = np.array(camera_ids_list, dtype=object)
        centroids_2d_arr = (
            np.array(centroids_2d_list, dtype=np.float64)
            if n > 0
            else np.empty((0, 2), dtype=np.float64)
        )

        out = Path(output_path).resolve()
        np.savez_compressed(
            str(out),
            fish_ids=fish_ids_arr,
            frame_indices=frame_indices_arr,
            points_3d=points_3d_arr,
            camera_ids=camera_ids_arr,
            centroids_2d=centroids_2d_arr,
        )
        return out
