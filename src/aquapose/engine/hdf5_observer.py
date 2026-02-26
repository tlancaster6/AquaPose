"""HDF5 export observer for writing final 3D spline control points to outputs.h5."""

from __future__ import annotations

import hashlib
import logging
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

    Subscribes to PipelineStart (to capture config for hashing) and
    PipelineComplete (to write the HDF5 file). The output uses a
    frame-major layout: ``/frames/NNNN/fish_N/control_points``.

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

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and handle HDF5 export.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if isinstance(event, PipelineStart):
            self._capture_config_hash(event)
        elif isinstance(event, PipelineComplete):
            self._write_hdf5(event)

    def _capture_config_hash(self, event: PipelineStart) -> None:
        """Compute MD5 hash of the serialized config string.

        Args:
            event: PipelineStart event containing the config object.
        """
        if event.config is None:
            return
        try:
            from aquapose.engine.config import serialize_config

            config_str = serialize_config(event.config)  # type: ignore[arg-type]
            self._config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
        except Exception:
            logger.warning("Failed to compute config hash", exc_info=True)

    def _write_hdf5(self, event: PipelineComplete) -> None:
        """Write the frame-major HDF5 file from PipelineContext.midlines_3d.

        Args:
            event: PipelineComplete event with optional context field.
        """
        context = event.context
        if context is None:
            return

        midlines_3d = getattr(context, "midlines_3d", None)
        if midlines_3d is None or not isinstance(midlines_3d, list):
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / "outputs.h5"

        # Collect all unique fish IDs across all frames.
        all_fish_ids: set[int] = set()
        for frame_dict in midlines_3d:
            if isinstance(frame_dict, dict):
                all_fish_ids.update(frame_dict.keys())
        sorted_fish_ids = sorted(all_fish_ids)

        with h5py.File(str(output_path), "w") as f:
            # Root-level metadata.
            f.attrs["run_id"] = event.run_id
            f.attrs["frame_count"] = len(midlines_3d)
            f.attrs["fish_ids"] = np.array(sorted_fish_ids, dtype=np.int64)
            if self._config_hash:
                f.attrs["config_hash"] = self._config_hash

            # Frame-major layout.
            frames_grp = f.create_group("frames")
            for frame_idx, frame_dict in enumerate(midlines_3d):
                frame_name = f"{frame_idx:04d}"
                frame_grp = frames_grp.create_group(frame_name)

                if not isinstance(frame_dict, dict):
                    continue

                for fish_id, spline in frame_dict.items():
                    fish_grp = frame_grp.create_group(f"fish_{fish_id}")

                    # Extract control points from Spline3D.
                    control_points = getattr(spline, "control_points", None)
                    if control_points is not None:
                        cp = np.asarray(control_points, dtype=np.float32)
                        fish_grp.create_dataset("control_points", data=cp)

                    # Extract arc length if available.
                    arc_length = getattr(spline, "arc_length", None)
                    if arc_length is not None:
                        fish_grp.attrs["arc_length"] = float(arc_length)

        logger.info("Wrote HDF5 output to %s", output_path)
