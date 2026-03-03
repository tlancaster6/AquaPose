"""Diagnostic observer for capturing intermediate stage outputs in memory."""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aquapose.engine.events import Event, PipelineComplete, StageComplete
from aquapose.io.midline_fixture import NPZ_VERSION as _NPZ_VERSION_LATEST

_NPZ_VERSION_V1 = "1.0"

logger = logging.getLogger(__name__)

_DETECTION_STAGE_NAME = "DetectionStage"
_TRACKING_STAGE_NAME = "TrackingStage"
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

    def __init__(
        self,
        output_dir: str | Path | None = None,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.stages: dict[str, StageSnapshot] = {}
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._calibration_path = (
            Path(calibration_path) if calibration_path is not None else None
        )

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
        """Auto-export pipeline diagnostics and midline fixtures when output_dir is set."""
        if self._output_dir is None:
            return

        # --- pipeline diagnostics (unified NPZ) ---
        try:
            out = self.export_pipeline_diagnostics(
                self._output_dir / "pipeline_diagnostics.npz"
            )
            logger.info("Pipeline diagnostics exported to %s", out)
        except Exception:
            logger.warning("Failed to export pipeline diagnostics", exc_info=True)

        # --- midline fixtures ---
        if (
            _ASSOCIATION_STAGE_NAME in self.stages
            and _MIDLINE_STAGE_NAME in self.stages
        ):
            try:
                models = self._build_projection_models()
                out_mid = self.export_midline_fixtures(
                    self._output_dir / "midline_fixtures.npz",
                    models=models,
                )
                logger.info("Midline fixtures exported to %s", out_mid)
            except Exception:
                logger.warning("Failed to export midline fixtures", exc_info=True)

    def _build_projection_models(self) -> dict[str, Any] | None:
        """Build per-camera projection models from calibration data.

        Returns:
            Dict mapping camera_id to RefractiveProjectionModel, or None if
            calibration_path was not provided or camera_ids are unavailable.
        """
        if self._calibration_path is None:
            return None

        # Need camera_ids from any captured snapshot
        camera_ids: list[str] | None = None
        for snapshot in self.stages.values():
            if snapshot.camera_ids is not None:
                camera_ids = snapshot.camera_ids
                break
        if not camera_ids:
            return None

        from aquapose.calibration.loader import (
            compute_undistortion_maps,
            load_calibration_data,
        )
        from aquapose.calibration.projection import RefractiveProjectionModel

        calib_data = load_calibration_data(str(self._calibration_path))
        models: dict[str, Any] = {}
        for cam_id in camera_ids:
            if cam_id in calib_data.cameras:
                cam = calib_data.cameras[cam_id]
                undist_maps = compute_undistortion_maps(cam)
                models[cam_id] = RefractiveProjectionModel(
                    K=undist_maps.K_new,
                    R=cam.R,
                    t=cam.t,
                    water_z=calib_data.water_z,
                    normal=calib_data.interface_normal,
                    n_air=calib_data.n_air,
                    n_water=calib_data.n_water,
                )
        return models if models else None

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

    def export_midline_fixtures(
        self,
        output_path: Path | str,
        models: dict[str, Any] | None = None,
    ) -> Path:
        """Assemble per-frame MidlineSets from snapshot data and write an NPZ fixture.

        Builds a ``dict[frame_idx, dict[fish_id, dict[camera_id, Midline2D]]]``
        by matching tracklet centroids to AnnotatedDetections and extracting their
        ``.midline`` field.  Frames with at least one valid midline are serialised
        to the compressed NPZ using the key convention documented in
        ``aquapose.io.midline_fixture``.

        When ``models`` is provided, the fixture is written as v2.0 with
        calibration data bundled under ``calib/`` keys.  When ``models`` is
        ``None``, the fixture is written as v1.0 with no calibration data
        (backward-compatible behaviour).

        Args:
            output_path: Destination file path for the NPZ output.
            models: Optional dict mapping camera_id to
                ``RefractiveProjectionModel`` instances.  When provided, the NPZ
                version is set to "2.0" and calibration parameters are extracted
                from the models and written to ``calib/`` keys.  All models in a
                rig share the same ``water_z``, ``normal``, ``n_air``, and
                ``n_water`` — these are taken from the first model in the dict.

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

        # Determine version: v2.0 when models are provided, v1.0 otherwise.
        version_str = _NPZ_VERSION_LATEST if models is not None else _NPZ_VERSION_V1

        # Meta arrays
        timestamp_str = datetime.datetime.now(datetime.UTC).isoformat()
        npz_arrays["meta/version"] = np.array(version_str, dtype=object)
        npz_arrays["meta/camera_ids"] = np.array(
            camera_ids_list if camera_ids_list else sorted(all_camera_ids), dtype=object
        )
        npz_arrays["meta/frame_indices"] = np.array(
            sorted_frame_indices, dtype=np.int64
        )
        npz_arrays["meta/frame_count"] = np.array(frame_count, dtype=np.int64)
        npz_arrays["meta/timestamp"] = np.array(timestamp_str, dtype=object)

        # Calibration arrays (v2.0 only — written when models dict is provided)
        if models is not None:
            _write_calib_arrays(npz_arrays, models)

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

    def export_pipeline_diagnostics(self, output_path: Path | str) -> Path:
        """Export unified pipeline diagnostics NPZ with 5 sections.

        Sections:
            - ``tracking/``: Per-(tracklet, frame) rows from ``tracks_2d``.
            - ``groups/``: Per-TrackletGroup summary rows.
            - ``correspondences/``: Per-(fish, frame, camera) 3D consensus rows.
            - ``detection_counts/``: (n_frames, n_cameras) detection count matrix.
            - ``midline_counts/``: (n_frames, n_cameras) midline count matrix.

        Each section is populated from the corresponding stage snapshot when
        available.  Missing snapshots produce empty arrays for that section.

        Args:
            output_path: Destination file path for the NPZ output.

        Returns:
            Resolved absolute path to the written NPZ file.
        """
        npz: dict[str, object] = {}

        self._collect_tracking_section(npz)
        self._collect_groups_section(npz)
        self._collect_correspondences_section(npz)
        self._collect_detection_counts_section(npz)
        self._collect_midline_counts_section(npz)

        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out), **npz)  # type: ignore[arg-type]
        return out

    # ------------------------------------------------------------------
    # Pipeline diagnostics: per-section collectors
    # ------------------------------------------------------------------

    def _collect_tracking_section(self, npz: dict[str, object]) -> None:
        """Populate ``tracking/`` keys from the TrackingStage snapshot."""
        snap = self.stages.get(_TRACKING_STAGE_NAME)
        tracks_2d = snap.tracks_2d if snap else None

        cam_ids_list: list[str] = []
        track_ids_list: list[int] = []
        frame_indices_list: list[int] = []
        centroids_list: list[tuple[float, float]] = []
        status_list: list[str] = []

        if tracks_2d:
            for cam_id, tracklets in tracks_2d.items():
                for tracklet in tracklets:
                    for i, frame_idx in enumerate(tracklet.frames):  # type: ignore[union-attr]
                        cam_ids_list.append(cam_id)
                        track_ids_list.append(tracklet.track_id)  # type: ignore[union-attr]
                        frame_indices_list.append(frame_idx)
                        centroids_list.append(tracklet.centroids[i])  # type: ignore[union-attr]
                        status_list.append(tracklet.frame_status[i])  # type: ignore[union-attr]

        n = len(cam_ids_list)
        npz["tracking/camera_ids"] = np.array(cam_ids_list, dtype=object)
        npz["tracking/track_ids"] = np.array(track_ids_list, dtype=np.int64)
        npz["tracking/frame_indices"] = np.array(frame_indices_list, dtype=np.int64)
        npz["tracking/centroids"] = (
            np.array(centroids_list, dtype=np.float32)
            if n > 0
            else np.empty((0, 2), dtype=np.float32)
        )
        npz["tracking/frame_status"] = np.array(status_list, dtype=object)

    def _collect_groups_section(self, npz: dict[str, object]) -> None:
        """Populate ``groups/`` keys from the AssociationStage snapshot."""
        snap = self.stages.get(_ASSOCIATION_STAGE_NAME)
        groups = snap.tracklet_groups if snap else None

        fish_ids_list: list[int] = []
        n_cameras_list: list[int] = []
        n_frames_list: list[int] = []
        confidence_list: list[float] = []
        camera_ids_csv_list: list[str] = []

        if groups:
            for group in groups:
                fish_ids_list.append(group.fish_id)
                tracklets = group.tracklets or ()
                cam_ids = [t.camera_id for t in tracklets]  # type: ignore[union-attr]
                n_cameras_list.append(len(cam_ids))
                # Union of all tracklet frames
                all_frames: set[int] = set()
                for t in tracklets:
                    all_frames.update(t.frames)  # type: ignore[union-attr]
                n_frames_list.append(len(all_frames))
                confidence_list.append(
                    group.confidence if group.confidence is not None else 0.0
                )
                camera_ids_csv_list.append(",".join(cam_ids))

        npz["groups/fish_ids"] = np.array(fish_ids_list, dtype=np.int64)
        npz["groups/n_cameras"] = np.array(n_cameras_list, dtype=np.int64)
        npz["groups/n_frames"] = np.array(n_frames_list, dtype=np.int64)
        npz["groups/confidence"] = np.array(confidence_list, dtype=np.float32)
        npz["groups/camera_ids_csv"] = np.array(camera_ids_csv_list, dtype=object)

    def _collect_correspondences_section(self, npz: dict[str, object]) -> None:
        """Populate ``correspondences/`` keys from TrackletGroup consensus."""
        snap = self.stages.get(_ASSOCIATION_STAGE_NAME)
        groups = snap.tracklet_groups if snap else None

        fish_ids_list: list[int] = []
        frame_indices_list: list[int] = []
        points_3d_list: list[np.ndarray] = []
        camera_ids_list: list[str] = []
        centroids_2d_list: list[tuple[float, float]] = []

        if groups:
            for group in groups:
                if group.consensus_centroids is None:
                    continue

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
        npz["correspondences/fish_ids"] = np.array(fish_ids_list, dtype=np.int64)
        npz["correspondences/frame_indices"] = np.array(
            frame_indices_list, dtype=np.int64
        )
        npz["correspondences/points_3d"] = (
            np.stack(points_3d_list, axis=0).astype(np.float64)
            if n > 0
            else np.empty((0, 3), dtype=np.float64)
        )
        npz["correspondences/camera_ids"] = np.array(camera_ids_list, dtype=object)
        npz["correspondences/centroids_2d"] = (
            np.array(centroids_2d_list, dtype=np.float64)
            if n > 0
            else np.empty((0, 2), dtype=np.float64)
        )

    def _collect_detection_counts_section(self, npz: dict[str, object]) -> None:
        """Populate ``detection_counts/`` keys from the DetectionStage snapshot."""
        snap = self.stages.get(_DETECTION_STAGE_NAME)
        detections = snap.detections if snap else None
        camera_ids = snap.camera_ids if snap else None
        frame_count = snap.frame_count if snap else None

        if detections and camera_ids and frame_count:
            n_frames = frame_count
            cam_list = list(camera_ids)
            matrix = np.zeros((n_frames, len(cam_list)), dtype=np.int16)
            for fi, frame_dict in enumerate(detections):
                if not isinstance(frame_dict, dict):
                    continue
                for ci, cam_id in enumerate(cam_list):
                    dets = frame_dict.get(cam_id)
                    if dets is not None:
                        matrix[fi, ci] = len(dets)
            npz["detection_counts/matrix"] = matrix
            npz["detection_counts/camera_ids"] = np.array(cam_list, dtype=object)
            npz["detection_counts/frame_indices"] = np.arange(n_frames, dtype=np.int64)
        else:
            npz["detection_counts/matrix"] = np.empty((0, 0), dtype=np.int16)
            npz["detection_counts/camera_ids"] = np.array([], dtype=object)
            npz["detection_counts/frame_indices"] = np.array([], dtype=np.int64)

    def _collect_midline_counts_section(self, npz: dict[str, object]) -> None:
        """Populate ``midline_counts/`` keys from the MidlineStage snapshot."""
        snap = self.stages.get(_MIDLINE_STAGE_NAME)
        annotated = snap.annotated_detections if snap else None
        camera_ids = snap.camera_ids if snap else None
        frame_count = snap.frame_count if snap else None

        if annotated and camera_ids and frame_count:
            n_frames = frame_count
            cam_list = list(camera_ids)
            matrix = np.zeros((n_frames, len(cam_list)), dtype=np.int16)
            for fi, frame_dict in enumerate(annotated):
                if not isinstance(frame_dict, dict):
                    continue
                for ci, cam_id in enumerate(cam_list):
                    ann_list = frame_dict.get(cam_id)
                    if ann_list is not None:
                        count = sum(
                            1
                            for ann in ann_list
                            if getattr(ann, "midline", None) is not None
                        )
                        matrix[fi, ci] = count
            npz["midline_counts/matrix"] = matrix
            npz["midline_counts/camera_ids"] = np.array(cam_list, dtype=object)
            npz["midline_counts/frame_indices"] = np.arange(n_frames, dtype=np.int64)
        else:
            npz["midline_counts/matrix"] = np.empty((0, 0), dtype=np.int16)
            npz["midline_counts/camera_ids"] = np.array([], dtype=object)
            npz["midline_counts/frame_indices"] = np.array([], dtype=np.int64)


def _write_calib_arrays(
    npz_arrays: dict[str, object],
    models: dict[str, Any],
) -> None:
    """Populate *npz_arrays* with ``calib/`` keys extracted from *models*.

    Extracts calibration parameters from ``RefractiveProjectionModel``
    instances and writes them into *npz_arrays* in-place.  All models in a rig
    share the same ``water_z``, ``normal``, ``n_air``, and ``n_water`` values —
    these are taken from the first model in the dict.

    CUDA safety: all torch tensors are moved to CPU before converting to numpy.

    Args:
        npz_arrays: Mutable dict that will receive the ``calib/`` keys.
        models: Dict mapping camera_id to ``RefractiveProjectionModel``
            instances (or any object with the same attribute names).
    """
    if not models:
        return

    # Shared calibration parameters come from the first model.
    first_model = next(iter(models.values()))
    npz_arrays["calib/water_z"] = np.float32(first_model.water_z)
    npz_arrays["calib/n_air"] = np.float32(first_model.n_air)
    npz_arrays["calib/n_water"] = np.float32(first_model.n_water)
    npz_arrays["calib/interface_normal"] = (
        first_model.normal.cpu().numpy().astype(np.float32)
    )

    # Per-camera parameters
    for cam_id, model in models.items():
        npz_arrays[f"calib/{cam_id}/K_new"] = model.K.cpu().numpy().astype(np.float32)
        npz_arrays[f"calib/{cam_id}/R"] = model.R.cpu().numpy().astype(np.float32)
        npz_arrays[f"calib/{cam_id}/t"] = model.t.cpu().numpy().astype(np.float32)
