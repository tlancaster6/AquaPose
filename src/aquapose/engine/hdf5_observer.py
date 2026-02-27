"""HDF5 export observer for writing final 3D spline control points to outputs.h5.

Supports two layouts:
- Fish-first (v2.1+): ``/fish_{id}/spline_controls[T, N, 3]`` and
  ``/fish_{id}/confidence[T]`` when tracklet_groups are available.
- Frame-major (legacy): ``/frames/NNNN/fish_N/control_points`` when
  tracklet_groups are absent (backward compatibility).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from aquapose.engine.events import (
    Event,
    PipelineComplete,
    PipelineStart,
)

logger = logging.getLogger(__name__)


class HDF5ExportObserver:
    """Writes final 3D spline control points and metadata to ``outputs.h5``.

    Subscribes to PipelineStart (to capture config for hashing and root
    attributes) and PipelineComplete (to write the HDF5 file).

    When ``tracklet_groups`` is available in the context, uses the fish-first
    layout with per-fish arrays. Otherwise falls back to the frame-major
    layout for backward compatibility.

    Args:
        output_dir: Directory where ``outputs.h5`` will be written.

    Example::

        observer = HDF5ExportObserver(output_dir="/tmp/run_output")
        pipeline = PosePipeline(stages=stages, config=config, observers=[observer])
        context = pipeline.run()
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._config_hash: str = ""
        self._config: object = None

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and handle HDF5 export.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if isinstance(event, PipelineStart):
            self._capture_config(event)
        elif isinstance(event, PipelineComplete):
            self._write_hdf5(event)

    def _capture_config(self, event: PipelineStart) -> None:
        """Compute MD5 hash of the serialized config and store config ref.

        Args:
            event: PipelineStart event containing the config object.
        """
        self._config = event.config
        if event.config is None:
            return
        try:
            from aquapose.engine.config import serialize_config

            config_str = serialize_config(event.config)  # type: ignore[arg-type]
            self._config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
        except Exception:
            logger.warning("Failed to compute config hash", exc_info=True)

    def _write_hdf5(self, event: PipelineComplete) -> None:
        """Write HDF5 file from PipelineContext.midlines_3d.

        Selects fish-first or frame-major layout based on whether
        tracklet_groups is populated in the context.

        Args:
            event: PipelineComplete event with optional context field.
        """
        context = event.context
        if context is None:
            return

        midlines_3d = getattr(context, "midlines_3d", None)
        if midlines_3d is None or not isinstance(midlines_3d, list):
            return

        tracklet_groups = getattr(context, "tracklet_groups", None)
        has_groups = tracklet_groups is not None and len(tracklet_groups) > 0

        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / "outputs.h5"

        if has_groups:
            self._write_fish_first(output_path, event, midlines_3d, tracklet_groups)
        else:
            self._write_frame_major(output_path, event, midlines_3d)

        logger.info("Wrote HDF5 output to %s", output_path)

    def _write_root_attrs(
        self,
        f: h5py.File,
        event: PipelineComplete,
        midlines_3d: list,
    ) -> None:
        """Write root-level attributes common to both layouts.

        Args:
            f: Open HDF5 file handle.
            event: PipelineComplete event.
            midlines_3d: Per-frame midline dicts.
        """
        # Collect all unique fish IDs
        all_fish_ids: set[int] = set()
        for frame_dict in midlines_3d:
            if isinstance(frame_dict, dict):
                all_fish_ids.update(frame_dict.keys())
        sorted_fish_ids = sorted(all_fish_ids)

        f.attrs["run_id"] = event.run_id
        f.attrs["frame_count"] = len(midlines_3d)
        f.attrs["fish_ids"] = np.array(sorted_fish_ids, dtype=np.int64)
        f.attrs["run_timestamp"] = datetime.now(tz=UTC).isoformat()

        if self._config_hash:
            f.attrs["config_hash"] = self._config_hash

        # Additional root attrs from config
        if self._config is not None:
            calibration_path = getattr(self._config, "calibration_path", None)
            if calibration_path is not None:
                f.attrs["calibration_path"] = str(calibration_path)

            video_dir = getattr(self._config, "video_dir", None)
            if video_dir is not None:
                f.attrs["video_dir"] = str(video_dir)

    def _write_fish_first(
        self,
        output_path: Path,
        event: PipelineComplete,
        midlines_3d: list,
        tracklet_groups: list,
    ) -> None:
        """Write fish-first HDF5 layout.

        Structure: ``/fish_{id}/spline_controls[T, N, 3]`` and
        ``/fish_{id}/confidence[T]``.

        Args:
            output_path: Path to write the HDF5 file.
            event: PipelineComplete event.
            midlines_3d: Per-frame per-fish Midline3D dicts.
            tracklet_groups: TrackletGroup list from association.
        """
        # Collect all fish IDs
        all_fish_ids: set[int] = set()
        for frame_dict in midlines_3d:
            if isinstance(frame_dict, dict):
                all_fish_ids.update(frame_dict.keys())
        sorted_fish_ids = sorted(all_fish_ids)

        total_frames = len(midlines_3d)

        # Build per-frame confidence from tracklet groups
        group_confidence: dict[int, dict[int, float]] = {}
        for group in tracklet_groups:
            fish_id = group.fish_id
            if group.per_frame_confidence is not None:
                # Build frame->confidence mapping from the group's frame range
                all_frames: list[int] = []
                for tracklet in group.tracklets:
                    all_frames.extend(tracklet.frames)
                unique_frames = sorted(set(all_frames))
                if len(unique_frames) == len(group.per_frame_confidence):
                    group_confidence[fish_id] = dict(
                        zip(unique_frames, group.per_frame_confidence, strict=True)
                    )

        with h5py.File(str(output_path), "w") as f:
            self._write_root_attrs(f, event, midlines_3d)
            f.attrs["layout"] = "fish_first"

            # Determine N (control point count) from first available midline
            n_ctrl = 7  # default
            for frame_dict in midlines_3d:
                if isinstance(frame_dict, dict):
                    for spline in frame_dict.values():
                        cp = getattr(spline, "control_points", None)
                        if cp is not None:
                            n_ctrl = np.asarray(cp).shape[0]
                            break
                    if n_ctrl != 7:
                        break

            for fish_id in sorted_fish_ids:
                fish_grp = f.create_group(f"fish_{fish_id}")

                spline_controls = np.full(
                    (total_frames, n_ctrl, 3), np.nan, dtype=np.float32
                )
                confidence = np.full(total_frames, np.nan, dtype=np.float32)

                for frame_idx, frame_dict in enumerate(midlines_3d):
                    if not isinstance(frame_dict, dict):
                        continue
                    if fish_id not in frame_dict:
                        continue

                    spline = frame_dict[fish_id]
                    cp = getattr(spline, "control_points", None)
                    if cp is not None:
                        spline_controls[frame_idx] = np.asarray(cp, dtype=np.float32)

                    # Confidence: interpolated (is_low_confidence) -> 0.0
                    is_low = getattr(spline, "is_low_confidence", False)
                    if is_low:
                        confidence[frame_idx] = 0.0
                    elif (
                        fish_id in group_confidence
                        and frame_idx in group_confidence[fish_id]
                    ):
                        confidence[frame_idx] = group_confidence[fish_id][frame_idx]
                    else:
                        confidence[frame_idx] = 1.0

                fish_grp.create_dataset(
                    "spline_controls",
                    data=spline_controls,
                    compression="gzip",
                    compression_opts=4,
                )
                fish_grp.create_dataset(
                    "confidence",
                    data=confidence,
                    compression="gzip",
                    compression_opts=4,
                )

    def _write_frame_major(
        self,
        output_path: Path,
        event: PipelineComplete,
        midlines_3d: list,
    ) -> None:
        """Write frame-major HDF5 layout (legacy backward compatibility).

        Structure: ``/frames/NNNN/fish_N/control_points``.

        Args:
            output_path: Path to write the HDF5 file.
            event: PipelineComplete event.
            midlines_3d: Per-frame per-fish Midline3D dicts.
        """
        with h5py.File(str(output_path), "w") as f:
            self._write_root_attrs(f, event, midlines_3d)

            frames_grp = f.create_group("frames")
            for frame_idx, frame_dict in enumerate(midlines_3d):
                frame_name = f"{frame_idx:04d}"
                frame_grp = frames_grp.create_group(frame_name)

                if not isinstance(frame_dict, dict):
                    continue

                for fish_id, spline in frame_dict.items():
                    fish_grp = frame_grp.create_group(f"fish_{fish_id}")

                    control_points = getattr(spline, "control_points", None)
                    if control_points is not None:
                        cp = np.asarray(control_points, dtype=np.float32)
                        fish_grp.create_dataset("control_points", data=cp)

                    arc_length = getattr(spline, "arc_length", None)
                    if arc_length is not None:
                        fish_grp.attrs["arc_length"] = float(arc_length)
