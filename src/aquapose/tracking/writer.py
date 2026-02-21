"""HDF5 serialization for tracking results with chunked resizable datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np

if TYPE_CHECKING:
    from .tracker import FishTrack


class TrackingWriter:
    """Write tracking results to a chunked HDF5 file.

    Buffers frames in memory and flushes full chunks to disk for efficient
    I/O on hours-long videos. Suitable for downstream Phase 6 consumption.

    The HDF5 layout under ``/tracking/`` is::

        frame_index            (N,)            int64
        fish_id                (N, max_fish)   int32    fillvalue=-1
        centroid_3d            (N, max_fish, 3) float32  fillvalue=NaN
        confidence             (N, max_fish)   float32  fillvalue=-1.0
        reprojection_residual  (N, max_fish)   float32  fillvalue=-1.0
        n_cameras              (N, max_fish)   int32    fillvalue=0
        is_confirmed           (N, max_fish)   bool     fillvalue=False
        camera_assignments/{cam}  (N, max_fish)   int32  fillvalue=-1
        bboxes/{cam}              (N, max_fish, 4) int32  fillvalue=-1

    Args:
        output_path: Path to the output ``.h5`` file. Created on open.
        camera_names: Ordered list of camera IDs. Determines column order in
            ``camera_assignments`` and ``bboxes`` sub-groups.
        max_fish: Maximum number of fish slots per frame (default 9).
        chunk_frames: Number of frames to buffer before flushing (default 1000).
    """

    def __init__(
        self,
        output_path: str | Path,
        camera_names: list[str],
        max_fish: int = 9,
        chunk_frames: int = 1000,
    ) -> None:
        self._path = Path(output_path)
        self._camera_names = list(camera_names)
        self._max_fish = max_fish
        self._chunk_frames = chunk_frames
        self._buffer_idx: int = 0

        # Open HDF5 file and create datasets
        self._file = h5py.File(self._path, "w")
        grp = self._file.require_group("tracking")
        grp.attrs["camera_names"] = self._camera_names

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
        _make("centroid_3d", (max_fish, 3), "float32", np.nan)
        _make("confidence", (max_fish,), "float32", -1.0)
        _make("reprojection_residual", (max_fish,), "float32", -1.0)
        _make("n_cameras", (max_fish,), "int32", 0)
        _make("is_confirmed", (max_fish,), "bool", False)

        # Per-camera sub-groups
        cam_grp = grp.require_group("camera_assignments")
        bbox_grp = grp.require_group("bboxes")
        for cam in self._camera_names:
            cam_grp.create_dataset(
                cam,
                shape=(0, max_fish),
                maxshape=(None, max_fish),
                chunks=(chunk_frames, max_fish),
                dtype="int32",
                fillvalue=-1,
                compression="gzip",
                compression_opts=4,
            )
            bbox_grp.create_dataset(
                cam,
                shape=(0, max_fish, 4),
                maxshape=(None, max_fish, 4),
                chunks=(chunk_frames, max_fish, 4),
                dtype="int32",
                fillvalue=-1,
                compression="gzip",
                compression_opts=4,
            )

        # Allocate in-memory buffer arrays (pre-allocated to chunk_frames size)
        self._buf_frame_index = np.zeros(chunk_frames, dtype=np.int64)
        self._buf_fish_id = np.full((chunk_frames, max_fish), -1, dtype=np.int32)
        self._buf_centroid_3d = np.full(
            (chunk_frames, max_fish, 3), np.nan, dtype=np.float32
        )
        self._buf_confidence = np.full((chunk_frames, max_fish), -1.0, dtype=np.float32)
        self._buf_reprojection_residual = np.full(
            (chunk_frames, max_fish), -1.0, dtype=np.float32
        )
        self._buf_n_cameras = np.zeros((chunk_frames, max_fish), dtype=np.int32)
        self._buf_is_confirmed = np.zeros((chunk_frames, max_fish), dtype=bool)
        self._buf_cam_assign: dict[str, np.ndarray] = {
            cam: np.full((chunk_frames, max_fish), -1, dtype=np.int32)
            for cam in self._camera_names
        }
        self._buf_bboxes: dict[str, np.ndarray] = {
            cam: np.full((chunk_frames, max_fish, 4), -1, dtype=np.int32)
            for cam in self._camera_names
        }

    def write_frame(self, frame_index: int, tracks: list[FishTrack]) -> None:
        """Write one frame of tracking results to the buffer.

        Fish are written in ascending ``fish_id`` order into slots 0..N-1.
        Unfilled slots retain their fill-values (-1, NaN, False, 0).

        Args:
            frame_index: Index of the video frame being written.
            tracks: List of FishTrack objects (typically confirmed tracks from
                ``FishTracker.update()`` or ``get_all_tracks()``).
        """
        i = self._buffer_idx

        # Reset this row to fill-values
        self._buf_frame_index[i] = frame_index
        self._buf_fish_id[i] = -1
        self._buf_centroid_3d[i] = np.nan
        self._buf_confidence[i] = -1.0
        self._buf_reprojection_residual[i] = -1.0
        self._buf_n_cameras[i] = 0
        self._buf_is_confirmed[i] = False
        for cam in self._camera_names:
            self._buf_cam_assign[cam][i] = -1
            self._buf_bboxes[cam][i] = -1

        # Sort tracks by fish_id for deterministic slot ordering
        sorted_tracks = sorted(tracks, key=lambda t: t.fish_id)

        for slot, track in enumerate(sorted_tracks):
            if slot >= self._max_fish:
                break
            self._buf_fish_id[i, slot] = track.fish_id
            if len(track.positions) > 0:
                self._buf_centroid_3d[i, slot] = list(track.positions)[-1].astype(
                    np.float32
                )
            self._buf_confidence[i, slot] = float(track.confidence)
            self._buf_reprojection_residual[i, slot] = float(
                track.reprojection_residual
            )
            self._buf_n_cameras[i, slot] = int(track.n_cameras)
            self._buf_is_confirmed[i, slot] = bool(track.is_confirmed)

            for cam in self._camera_names:
                det_idx = track.camera_detections.get(cam, -1)
                self._buf_cam_assign[cam][i, slot] = int(det_idx)

            for cam in self._camera_names:
                bbox = track.bboxes.get(cam)
                if bbox is not None:
                    self._buf_bboxes[cam][i, slot] = np.array(bbox, dtype=np.int32)

        self._buffer_idx += 1
        if self._buffer_idx == self._chunk_frames:
            self._flush()

    def _flush(self) -> None:
        """Flush buffered frames to disk and reset the buffer index."""
        n = self._buffer_idx
        if n == 0:
            return

        grp = cast(h5py.Group, self._file["tracking"])

        def _extend(ds_name: str, data: np.ndarray) -> None:
            ds = cast(h5py.Dataset, grp[ds_name])
            old_size = ds.shape[0]
            ds.resize(old_size + n, axis=0)
            ds[old_size : old_size + n] = data[:n]

        _extend("frame_index", self._buf_frame_index)
        _extend("fish_id", self._buf_fish_id)
        _extend("centroid_3d", self._buf_centroid_3d)
        _extend("confidence", self._buf_confidence)
        _extend("reprojection_residual", self._buf_reprojection_residual)
        _extend("n_cameras", self._buf_n_cameras)
        _extend("is_confirmed", self._buf_is_confirmed)

        cam_grp = cast(h5py.Group, grp["camera_assignments"])
        bbox_grp = cast(h5py.Group, grp["bboxes"])
        for cam in self._camera_names:
            _extend_cam(cast(h5py.Dataset, cam_grp[cam]), self._buf_cam_assign[cam], n)
            _extend_cam(cast(h5py.Dataset, bbox_grp[cam]), self._buf_bboxes[cam], n)

        self._file.flush()
        self._buffer_idx = 0

    def close(self) -> None:
        """Flush remaining buffered frames and close the HDF5 file."""
        self._flush()
        self._file.close()

    def __enter__(self) -> TrackingWriter:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the file on context manager exit."""
        self.close()


def _extend_cam(ds: h5py.Dataset, buf: np.ndarray, n: int) -> None:
    """Resize and append ``n`` rows of ``buf`` to ``ds`` in place.

    Args:
        ds: An h5py Dataset with resizable axis 0.
        buf: Buffer array; first ``n`` rows are written.
        n: Number of rows to write.
    """
    old_size = ds.shape[0]
    ds.resize(old_size + n, axis=0)
    ds[old_size : old_size + n] = buf[:n]


def read_tracking_results(path: str | Path) -> dict[str, Any]:
    """Read all tracking results from an HDF5 file into numpy arrays.

    Convenience reader for downstream consumers (Phase 6) and round-trip
    verification. Reads all datasets under ``/tracking/`` and the
    ``camera_names`` attribute.

    Args:
        path: Path to the HDF5 file written by :class:`TrackingWriter`.

    Returns:
        Dict with keys:

        - ``frame_index``: shape ``(N,)``, int64
        - ``fish_id``: shape ``(N, max_fish)``, int32
        - ``centroid_3d``: shape ``(N, max_fish, 3)``, float32
        - ``confidence``: shape ``(N, max_fish)``, float32
        - ``reprojection_residual``: shape ``(N, max_fish)``, float32
        - ``n_cameras``: shape ``(N, max_fish)``, int32
        - ``is_confirmed``: shape ``(N, max_fish)``, bool
        - ``camera_assignments``: dict mapping camera_name -> array ``(N, max_fish)``
        - ``bboxes``: dict mapping camera_name -> array ``(N, max_fish, 4)``
        - ``camera_names``: list of str
    """
    with h5py.File(path, "r") as f:
        grp = cast(h5py.Group, f["tracking"])
        raw_cam_names = cast(list[str], grp.attrs["camera_names"])
        camera_names: list[str] = list(raw_cam_names)

        result: dict[str, Any] = {
            "frame_index": cast(h5py.Dataset, grp["frame_index"])[()],
            "fish_id": cast(h5py.Dataset, grp["fish_id"])[()],
            "centroid_3d": cast(h5py.Dataset, grp["centroid_3d"])[()],
            "confidence": cast(h5py.Dataset, grp["confidence"])[()],
            "reprojection_residual": cast(h5py.Dataset, grp["reprojection_residual"])[
                ()
            ],
            "n_cameras": cast(h5py.Dataset, grp["n_cameras"])[()],
            "is_confirmed": cast(h5py.Dataset, grp["is_confirmed"])[()],
            "camera_names": camera_names,
        }

        cam_assign_grp = cast(h5py.Group, grp["camera_assignments"])
        bbox_grp = cast(h5py.Group, grp["bboxes"])

        camera_assignments: dict[str, Any] = {}
        bboxes: dict[str, Any] = {}
        for cam in camera_names:
            camera_assignments[cam] = cast(h5py.Dataset, cam_assign_grp[cam])[()]
            bboxes[cam] = cast(h5py.Dataset, bbox_grp[cam])[()]

        result["camera_assignments"] = camera_assignments
        result["bboxes"] = bboxes

    return result
