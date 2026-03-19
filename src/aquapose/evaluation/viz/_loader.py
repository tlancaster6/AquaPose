"""Shared data loader utilities for viz modules."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aquapose.core.context import PipelineContext

logger = logging.getLogger(__name__)


def load_all_chunk_caches(run_dir: Path) -> list[PipelineContext]:
    """Load all chunk cache files from a run directory in chunk order.

    Reads ``diagnostics/manifest.json`` to discover chunk indices, then loads
    each ``diagnostics/chunk_NNN/cache.pkl`` in ascending order. Chunks with
    missing or unloadable cache files are skipped with a warning.

    Args:
        run_dir: Path to the pipeline run directory containing ``diagnostics/``.

    Returns:
        List of PipelineContext objects, one per chunk, in ascending chunk order.
        Returns an empty list if no diagnostics directory or manifest is found.
    """
    from aquapose.core.context import load_chunk_cache

    diag_dir = run_dir / "diagnostics"
    if not diag_dir.exists():
        logger.warning("No diagnostics directory found at %s", diag_dir)
        return []

    manifest_path = diag_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning("No manifest.json found at %s", manifest_path)
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read manifest.json: %s", exc)
        return []

    chunks = manifest.get("chunks", [])
    if not chunks:
        logger.warning("manifest.json has no chunk entries")
        return []

    # Sort by chunk index.
    chunks_sorted = sorted(chunks, key=lambda c: c.get("index", 0))

    contexts: list[PipelineContext] = []
    for chunk_entry in chunks_sorted:
        chunk_idx = chunk_entry.get("index", 0)
        cache_path = diag_dir / f"chunk_{chunk_idx:03d}" / "cache.pkl"
        if not cache_path.exists():
            logger.warning("Chunk cache not found: %s", cache_path)
            continue
        try:
            ctx = load_chunk_cache(cache_path)
            contexts.append(ctx)
        except Exception as exc:
            logger.warning("Failed to load chunk cache %s: %s", cache_path, exc)
            continue

    return contexts


def read_config_yaml(run_dir: Path) -> dict:
    """Read config.yaml from run directory as a raw dict.

    Args:
        run_dir: Path to the pipeline run directory.

    Returns:
        Parsed YAML dict, or empty dict if file is missing or unparseable.
    """
    import yaml

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        logger.warning("config.yaml not found in %s", run_dir)
        return {}
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to parse config.yaml: %s", exc)
        return {}


class H5Spline:
    """Lightweight spline-like object for eval_spline compatibility."""

    __slots__ = ("control_points", "degree", "knots", "points")

    def __init__(
        self,
        control_points: np.ndarray,
        knots: np.ndarray,
        degree: int,
        points: np.ndarray | None = None,
    ) -> None:
        self.control_points = control_points
        self.knots = knots
        self.degree = degree
        self.points = points


def load_midlines_from_h5(h5_path: Path) -> list[dict[int, H5Spline]]:
    """Load per-frame midline dicts from an HDF5 file.

    Args:
        h5_path: Path to a midlines HDF5 file.

    Returns:
        List of per-frame dicts mapping fish_id to H5Spline objects.
    """
    from aquapose.io.midline_writer import read_midline3d_results

    data = read_midline3d_results(h5_path)
    fish_ids = data["fish_id"]  # (N, max_fish)
    control_points = data["control_points"]  # (N, max_fish, 7, 3)
    raw_points = data.get("points")  # (N, max_fish, n_kpts, 3) or None
    knots = np.asarray(data["SPLINE_KNOTS"], dtype=np.float64)
    degree = int(data["SPLINE_K"])
    n_frames, max_fish = fish_ids.shape

    frames: list[dict[int, H5Spline]] = []
    for fi in range(n_frames):
        frame_dict: dict[int, H5Spline] = {}
        for s in range(max_fish):
            fid = int(fish_ids[fi, s])
            if fid >= 0:
                pts = raw_points[fi, s] if raw_points is not None else None
                frame_dict[fid] = H5Spline(
                    control_points[fi, s],
                    knots,
                    degree,
                    points=pts,
                )
        frames.append(frame_dict)
    return frames


def resolve_h5_path(run_dir: Path, *, unstitched: bool = False) -> Path | None:
    """Resolve the best midlines HDF5 path for a run directory.

    Args:
        run_dir: Path to the pipeline run directory.
        unstitched: If True, always use ``midlines.h5``.

    Returns:
        Path to the HDF5 file, or None if neither file exists.
    """
    if not unstitched:
        for name in (
            "midlines_stitched_smoothed.h5",
            "midlines_stitched.h5",
        ):
            path = run_dir / name
            if path.exists():
                return path
    for name in ("midlines_smoothed.h5", "midlines.h5"):
        path = run_dir / name
        if path.exists():
            return path
    return None
