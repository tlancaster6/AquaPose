"""HDF5 chunked-append writer for Midline3D reconstruction results.

Modelled on :class:`~aquapose.tracking.writer.TrackingWriter` with a chunked
resizable dataset layout under ``/midlines/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np

from aquapose.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
)

if TYPE_CHECKING:
    from aquapose.reconstruction.triangulation import Midline3D

# Number of B-spline control points per midline
_N_CTRL = 7


class Midline3DWriter:
    """Write Midline3D results to a chunked HDF5 file.

    Buffers frames in memory and flushes full chunks to disk for efficient
    I/O on long videos. Suitable for downstream visualisation and analysis.

    The HDF5 layout under ``/midlines/`` is::

        frame_index       (N,)                   int64,   fillvalue=0
        fish_id           (N, max_fish)           int32,   fillvalue=-1
        control_points    (N, max_fish, 7, 3)     float32, fillvalue=NaN
        arc_length        (N, max_fish)           float32, fillvalue=NaN
        half_widths       (N, max_fish, 15)       float32, fillvalue=NaN
        n_cameras         (N, max_fish)           int32,   fillvalue=0
        mean_residual     (N, max_fish)           float32, fillvalue=-1.0
        max_residual      (N, max_fish)           float32, fillvalue=-1.0
        is_low_confidence (N, max_fish)           bool,    fillvalue=False

    Group attributes store the spline knot vector (``SPLINE_KNOTS``) and
    degree (``SPLINE_K``) as they are constant across all frames.

    Args:
        output_path: Path to the output ``.h5`` file. Created on open.
        max_fish: Maximum number of fish slots per frame (default 9).
        chunk_frames: Number of frames to buffer before flushing (default 1000).
    """

    def __init__(
        self,
        output_path: str | Path,
        max_fish: int = 9,
        chunk_frames: int = 1000,
    ) -> None:
        self._path = Path(output_path)
        self._max_fish = max_fish
        self._chunk_frames = chunk_frames
        self._buffer_idx: int = 0

        # Open HDF5 file and create datasets
        self._file = h5py.File(self._path, "w")
        grp = self._file.require_group("midlines")

        # Store spline constants as group attributes
        grp.attrs["SPLINE_KNOTS"] = SPLINE_KNOTS
        grp.attrs["SPLINE_K"] = SPLINE_K

        def _make(
            name: str,
            trailing_shape: tuple[int, ...],
            dtype: str,
            fill: object,
        ) -> h5py.Dataset:
            """Create a resizable, chunked, gzip-compressed HDF5 dataset."""
            full_shape = (0, *trailing_shape)
            chunk_shape = (chunk_frames, *trailing_shape)
            max_shape = (None, *trailing_shape)
            return grp.create_dataset(
                name,
                shape=full_shape,
                maxshape=max_shape,
                chunks=chunk_shape,
                dtype=dtype,
                fillvalue=fill,
                compression="gzip",
                compression_opts=4,
            )

        _make("frame_index", (), "int64", 0)
        _make("fish_id", (max_fish,), "int32", -1)
        _make("control_points", (max_fish, _N_CTRL, 3), "float32", np.nan)
        _make("arc_length", (max_fish,), "float32", np.nan)
        _make("half_widths", (max_fish, N_SAMPLE_POINTS), "float32", np.nan)
        _make("n_cameras", (max_fish,), "int32", 0)
        _make("mean_residual", (max_fish,), "float32", -1.0)
        _make("max_residual", (max_fish,), "float32", -1.0)
        _make("is_low_confidence", (max_fish,), "bool", False)

        # Allocate in-memory buffer arrays (pre-allocated to chunk_frames size)
        self._buf_frame_index = np.zeros(chunk_frames, dtype=np.int64)
        self._buf_fish_id = np.full((chunk_frames, max_fish), -1, dtype=np.int32)
        self._buf_control_points = np.full(
            (chunk_frames, max_fish, _N_CTRL, 3), np.nan, dtype=np.float32
        )
        self._buf_arc_length = np.full(
            (chunk_frames, max_fish), np.nan, dtype=np.float32
        )
        self._buf_half_widths = np.full(
            (chunk_frames, max_fish, N_SAMPLE_POINTS), np.nan, dtype=np.float32
        )
        self._buf_n_cameras = np.zeros((chunk_frames, max_fish), dtype=np.int32)
        self._buf_mean_residual = np.full(
            (chunk_frames, max_fish), -1.0, dtype=np.float32
        )
        self._buf_max_residual = np.full(
            (chunk_frames, max_fish), -1.0, dtype=np.float32
        )
        self._buf_is_low_confidence = np.zeros((chunk_frames, max_fish), dtype=bool)

    def write_frame(self, frame_index: int, midlines: dict[int, Midline3D]) -> None:
        """Write one frame of Midline3D results to the buffer.

        Fish are written in ascending ``fish_id`` order into slots 0..N-1.
        Unfilled slots retain their fill-values (-1, NaN, False, 0).

        Args:
            frame_index: Index of the video frame being written.
            midlines: Dict mapping fish_id to :class:`~aquapose.reconstruction.triangulation.Midline3D`.
                Pass an empty dict to write a frame with all fill-values.
        """
        i = self._buffer_idx

        # Reset this row to fill-values
        self._buf_frame_index[i] = frame_index
        self._buf_fish_id[i] = -1
        self._buf_control_points[i] = np.nan
        self._buf_arc_length[i] = np.nan
        self._buf_half_widths[i] = np.nan
        self._buf_n_cameras[i] = 0
        self._buf_mean_residual[i] = -1.0
        self._buf_max_residual[i] = -1.0
        self._buf_is_low_confidence[i] = False

        # Sort midlines by fish_id for deterministic slot ordering
        sorted_items = sorted(midlines.items(), key=lambda kv: kv[0])

        for slot, (fish_id, midline) in enumerate(sorted_items):
            if slot >= self._max_fish:
                break

            self._buf_fish_id[i, slot] = fish_id
            self._buf_control_points[i, slot] = midline.control_points.astype(
                np.float32
            )
            self._buf_arc_length[i, slot] = float(midline.arc_length)
            hw = midline.half_widths.astype(np.float32)
            n_hw = min(len(hw), N_SAMPLE_POINTS)
            self._buf_half_widths[i, slot, :n_hw] = hw[:n_hw]
            self._buf_n_cameras[i, slot] = int(midline.n_cameras)
            self._buf_mean_residual[i, slot] = float(midline.mean_residual)
            self._buf_max_residual[i, slot] = float(midline.max_residual)
            self._buf_is_low_confidence[i, slot] = bool(midline.is_low_confidence)

        self._buffer_idx += 1
        if self._buffer_idx == self._chunk_frames:
            self._flush()

    def _flush(self) -> None:
        """Flush buffered frames to disk and reset the buffer index."""
        n = self._buffer_idx
        if n == 0:
            return

        grp = cast(h5py.Group, self._file["midlines"])

        def _extend(ds_name: str, data: np.ndarray) -> None:
            ds = cast(h5py.Dataset, grp[ds_name])
            old_size = ds.shape[0]
            ds.resize(old_size + n, axis=0)
            ds[old_size : old_size + n] = data[:n]

        _extend("frame_index", self._buf_frame_index)
        _extend("fish_id", self._buf_fish_id)
        _extend("control_points", self._buf_control_points)
        _extend("arc_length", self._buf_arc_length)
        _extend("half_widths", self._buf_half_widths)
        _extend("n_cameras", self._buf_n_cameras)
        _extend("mean_residual", self._buf_mean_residual)
        _extend("max_residual", self._buf_max_residual)
        _extend("is_low_confidence", self._buf_is_low_confidence)

        self._file.flush()
        self._buffer_idx = 0

    def close(self) -> None:
        """Flush remaining buffered frames and close the HDF5 file."""
        self._flush()
        self._file.close()

    def __enter__(self) -> Midline3DWriter:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the file on context manager exit."""
        self.close()


def read_midline3d_results(path: str | Path) -> dict[str, Any]:
    """Read all Midline3D results from an HDF5 file into numpy arrays.

    Convenience reader for downstream consumers and round-trip verification.
    Reads all datasets under ``/midlines/`` and the ``SPLINE_KNOTS`` and
    ``SPLINE_K`` group attributes.

    Args:
        path: Path to the HDF5 file written by :class:`Midline3DWriter`.

    Returns:
        Dict with keys:

        - ``frame_index``: shape ``(N,)``, int64
        - ``fish_id``: shape ``(N, max_fish)``, int32
        - ``control_points``: shape ``(N, max_fish, 7, 3)``, float32
        - ``arc_length``: shape ``(N, max_fish)``, float32
        - ``half_widths``: shape ``(N, max_fish, 15)``, float32
        - ``n_cameras``: shape ``(N, max_fish)``, int32
        - ``mean_residual``: shape ``(N, max_fish)``, float32
        - ``max_residual``: shape ``(N, max_fish)``, float32
        - ``is_low_confidence``: shape ``(N, max_fish)``, bool
        - ``SPLINE_KNOTS``: shape ``(11,)``, float64
        - ``SPLINE_K``: int
    """
    with h5py.File(path, "r") as f:
        grp = cast(h5py.Group, f["midlines"])

        result: dict[str, Any] = {
            "frame_index": cast(h5py.Dataset, grp["frame_index"])[()],
            "fish_id": cast(h5py.Dataset, grp["fish_id"])[()],
            "control_points": cast(h5py.Dataset, grp["control_points"])[()],
            "arc_length": cast(h5py.Dataset, grp["arc_length"])[()],
            "half_widths": cast(h5py.Dataset, grp["half_widths"])[()],
            "n_cameras": cast(h5py.Dataset, grp["n_cameras"])[()],
            "mean_residual": cast(h5py.Dataset, grp["mean_residual"])[()],
            "max_residual": cast(h5py.Dataset, grp["max_residual"])[()],
            "is_low_confidence": cast(h5py.Dataset, grp["is_low_confidence"])[()],
            "SPLINE_KNOTS": grp.attrs["SPLINE_KNOTS"],
            "SPLINE_K": int(cast(int, grp.attrs["SPLINE_K"])),
        }

    return result
