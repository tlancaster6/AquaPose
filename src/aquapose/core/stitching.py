"""Post-hoc trajectory stitching: merge fragmented 3D fish IDs into true identities.

Loads midlines.h5 from a full pipeline run, builds per-trajectory summaries,
then partitions fragments into K identities using conflict-graph coloring with
spatial-cost tiebreaking.

Algorithm:
  1. Build conflict graph: two fragments that share frames are definitely
     different fish (conflict edge).
  2. Greedy left-to-right coloring: process fragments by start time. For each
     fragment, assign it to the chain whose last member has the lowest
     transition cost, among chains that don't conflict. When only one chain is
     valid (the common case), the assignment is forced by temporal exclusion
     alone — spatial cost is only consulted when multiple chains are feasible.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Down-weight Z due to ~2.9x reconstruction anisotropy (see memory: Z/XY Anisotropy)
Z_WEIGHT = 1.0 / 2.9

# Number of frames at each end used to compute endpoint centroid/velocity
ENDPOINT_WINDOW = 5

# Maximum gap (in frames) over which velocity extrapolation is applied
VELOCITY_HORIZON = 30


@dataclass
class TrajectoryInfo:
    """Summary statistics for one fish_id trajectory."""

    fish_id: int
    frames: list[int] = field(default_factory=list)
    frame_set: set[int] = field(default_factory=set)
    n_observations: int = 0
    first_frame: int = 0
    last_frame: int = 0
    mean_arc_length: float = 0.0
    mean_n_cameras: float = 0.0
    centroids: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    start_centroid: NDArray[np.float64] | None = None
    end_centroid: NDArray[np.float64] | None = None
    start_velocity: NDArray[np.float64] | None = None
    end_velocity: NDArray[np.float64] | None = None
    n_cameras_per_frame: dict[int, int] = field(default_factory=dict)
    mean_residual_per_frame: dict[int, float] = field(default_factory=dict)


def load_trajectories(
    h5_path: str | Path,
    min_frames: int = 5,
    min_mean_cameras: float = 2.0,
) -> tuple[list[TrajectoryInfo], list[int]]:
    """Load midlines.h5 and build per-trajectory summaries.

    Args:
        h5_path: Path to the midlines.h5 file.
        min_frames: Drop trajectories with fewer than this many frames.
        min_mean_cameras: Drop trajectories with mean camera count below this.

    Returns:
        Tuple of (list of TrajectoryInfo, list of dropped fish_ids).
    """
    with h5py.File(h5_path, "r") as f:
        grp = cast(h5py.Group, f["midlines"])
        frame_index = cast(h5py.Dataset, grp["frame_index"])[()]
        fish_id = cast(h5py.Dataset, grp["fish_id"])[()]
        points = cast(h5py.Dataset, grp["points"])[()]
        arc_length = cast(h5py.Dataset, grp["arc_length"])[()]
        n_cameras = cast(h5py.Dataset, grp["n_cameras"])[()]
        mean_residual = cast(h5py.Dataset, grp["mean_residual"])[()]

    fish_data: dict[int, dict] = {}
    n_rows, max_fish = fish_id.shape

    for row_idx in range(n_rows):
        frame = int(frame_index[row_idx])
        for slot in range(max_fish):
            fid = int(fish_id[row_idx, slot])
            if fid < 0:
                continue
            if fid not in fish_data:
                fish_data[fid] = {
                    "frames": [],
                    "arc_lengths": [],
                    "n_cameras_list": [],
                    "centroids": {},
                    "n_cameras_per_frame": {},
                    "mean_residual_per_frame": {},
                }

            fd = fish_data[fid]
            fd["frames"].append(frame)
            fd["arc_lengths"].append(float(arc_length[row_idx, slot]))
            fd["n_cameras_list"].append(int(n_cameras[row_idx, slot]))
            fd["n_cameras_per_frame"][frame] = int(n_cameras[row_idx, slot])
            resid = float(mean_residual[row_idx, slot])
            if resid >= 0:
                fd["mean_residual_per_frame"][frame] = resid

            pts = points[row_idx, slot]  # (n_keypoints, 3)
            valid_mask = ~np.isnan(pts).any(axis=1)
            if valid_mask.any():
                fd["centroids"][frame] = pts[valid_mask].mean(axis=0)

    trajectories: list[TrajectoryInfo] = []
    dropped: list[int] = []

    for fid, fd in sorted(fish_data.items()):
        frames = sorted(fd["frames"])
        n_obs = len(frames)
        mean_ncam = (
            float(np.mean(fd["n_cameras_list"])) if fd["n_cameras_list"] else 0.0
        )

        if n_obs < min_frames or mean_ncam < min_mean_cameras:
            dropped.append(fid)
            continue

        ti = TrajectoryInfo(
            fish_id=fid,
            frames=frames,
            frame_set=set(frames),
            n_observations=n_obs,
            first_frame=frames[0],
            last_frame=frames[-1],
            mean_arc_length=float(np.nanmean(fd["arc_lengths"])),
            mean_n_cameras=mean_ncam,
            centroids=fd["centroids"],
            n_cameras_per_frame=fd["n_cameras_per_frame"],
            mean_residual_per_frame=fd["mean_residual_per_frame"],
        )
        _compute_endpoint_stats(ti)
        trajectories.append(ti)

    return trajectories, dropped


def _compute_endpoint_stats(ti: TrajectoryInfo) -> None:
    """Compute start/end centroids and velocities for a trajectory."""
    frames = ti.frames
    centroids = ti.centroids
    w = ENDPOINT_WINDOW

    start_frames = [f for f in frames[:w] if f in centroids]
    if start_frames:
        ti.start_centroid = np.mean([centroids[f] for f in start_frames], axis=0)
    if len(start_frames) >= 2:
        dt = start_frames[-1] - start_frames[0]
        if dt > 0:
            ti.start_velocity = (
                centroids[start_frames[-1]] - centroids[start_frames[0]]
            ) / dt

    end_frames = [f for f in frames[-w:] if f in centroids]
    if end_frames:
        ti.end_centroid = np.mean([centroids[f] for f in end_frames], axis=0)
    if len(end_frames) >= 2:
        dt = end_frames[-1] - end_frames[0]
        if dt > 0:
            ti.end_velocity = (
                centroids[end_frames[-1]] - centroids[end_frames[0]]
            ) / dt


def _weighted_dist(a: NDArray, b: NDArray) -> float:
    """Euclidean distance with Z down-weighted by Z_WEIGHT."""
    diff = (a - b).copy()
    diff[2] *= Z_WEIGHT
    return float(np.linalg.norm(diff))


def _transition_cost(a: TrajectoryInfo, b: TrajectoryInfo) -> float:
    """Cost of linking fragment *a* -> *b* (a ends before b starts).

    Uses velocity-extrapolated position when the gap is short enough and
    the predecessor has sufficient observations for a reliable velocity.

    Args:
        a: Predecessor trajectory.
        b: Successor trajectory.

    Returns:
        Scalar transition cost (lower is better).
    """
    if a.end_centroid is None or b.start_centroid is None:
        return 10.0

    gap = b.first_frame - a.last_frame

    if (
        gap > 0
        and gap <= VELOCITY_HORIZON
        and a.end_velocity is not None
        and a.n_observations >= 10
    ):
        predicted = a.end_centroid + a.end_velocity * gap
        dist = _weighted_dist(predicted, b.start_centroid)
    else:
        dist = _weighted_dist(a.end_centroid, b.start_centroid)

    return dist


def build_conflict_graph(
    trajectories: list[TrajectoryInfo],
) -> list[set[int]]:
    """Build conflict graph: fragments sharing frames are different fish.

    Args:
        trajectories: List of TrajectoryInfo.

    Returns:
        List of conflict sets (conflicts[i] = set of indices conflicting with i).
    """
    n = len(trajectories)
    conflicts: list[set[int]] = [set() for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if trajectories[i].frame_set & trajectories[j].frame_set:
                conflicts[i].add(j)
                conflicts[j].add(i)

    return conflicts


def solve_coloring(
    trajectories: list[TrajectoryInfo],
    conflicts: list[set[int]],
    target_k: int,
) -> list[list[TrajectoryInfo]]:
    """Greedy left-to-right coloring with spatial-cost tiebreaking.

    Process fragments by start time. For each fragment, assign to the chain
    whose last member has the lowest transition cost, among chains without
    conflict. When only one chain is valid, the assignment is forced.

    Args:
        trajectories: List of TrajectoryInfo.
        conflicts: Conflict sets from build_conflict_graph.
        target_k: Number of fish identities.

    Returns:
        List of up to *target_k* chains (each a list of TrajectoryInfo
        sorted by first_frame).
    """
    n = len(trajectories)
    order = sorted(range(n), key=lambda i: trajectories[i].first_frame)

    chain_members: list[list[int]] = [[] for _ in range(target_k)]
    chain_last: list[int | None] = [None] * target_k

    n_forced = 0
    n_spatial = 0
    n_empty = 0

    for idx in order:
        valid: list[int] = []
        for k in range(target_k):
            if not (conflicts[idx] & set(chain_members[k])):
                valid.append(k)

        if not valid:
            logger.warning(
                "fid %d conflicts with all %d chains — forcing assignment",
                trajectories[idx].fish_id,
                target_k,
            )
            min_conflicts = target_k + 1
            best_k = 0
            for k in range(target_k):
                nc = len(conflicts[idx] & set(chain_members[k]))
                if nc < min_conflicts:
                    min_conflicts = nc
                    best_k = k
            valid = [best_k]

        if len(valid) == 1:
            best_k = valid[0]
            if chain_last[best_k] is not None:
                n_forced += 1
            else:
                n_empty += 1
        else:
            best_k = -1
            best_cost = float("inf")
            for k in valid:
                if chain_last[k] is None:
                    cost = 0.0
                else:
                    last_idx = chain_last[k]
                    assert last_idx is not None
                    cost = _transition_cost(trajectories[last_idx], trajectories[idx])
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            n_spatial += 1

        chain_members[best_k].append(idx)
        chain_last[best_k] = idx

    logger.info(
        "Assignments: %d initial, %d forced, %d spatial", n_empty, n_forced, n_spatial
    )

    chains: list[list[TrajectoryInfo]] = []
    for members in chain_members:
        if members:
            chain = sorted(
                [trajectories[i] for i in members], key=lambda t: t.first_frame
            )
            chains.append(chain)

    return chains


def write_remapped_h5(
    src_path: str | Path,
    dst_path: str | Path,
    chains: list[list[TrajectoryInfo]],
    dropped: list[int],
) -> None:
    """Copy midlines.h5 with fish_id remapped to 0..K-1.

    For identity-split frames (two old IDs map to the same new ID in the same
    frame), keeps the observation with more cameras and drops the other.

    Args:
        src_path: Path to the source midlines HDF5 file.
        dst_path: Path to write the remapped HDF5 file.
        chains: Chain assignments from solve_coloring.
        dropped: List of fish_ids to discard (set to -1).
    """
    remap: dict[int, int] = {}
    for new_id, chain in enumerate(chains):
        for member in chain:
            remap[member.fish_id] = new_id

    logger.info("Identity remap (%d entries):", len(remap))
    for old_id in sorted(remap.keys()):
        logger.info("  %3d -> %d", old_id, remap[old_id])
    logger.debug("Dropped IDs (set to -1): %s", dropped)

    shutil.copy2(src_path, dst_path)

    with h5py.File(dst_path, "r+") as f:
        grp = cast(h5py.Group, f["midlines"])
        fish_id_ds = cast(h5py.Dataset, grp["fish_id"])
        n_cameras_ds = cast(h5py.Dataset, grp["n_cameras"])
        data = fish_id_ds[()]
        ncam_data = n_cameras_ds[()]

        n_deduped = 0
        for row_idx in range(data.shape[0]):
            for slot in range(data.shape[1]):
                old = int(data[row_idx, slot])
                if old < 0:
                    continue
                data[row_idx, slot] = remap.get(old, -1)

            seen: dict[int, int] = {}
            for slot in range(data.shape[1]):
                new_id = int(data[row_idx, slot])
                if new_id < 0:
                    continue
                if new_id in seen:
                    prev_slot = seen[new_id]
                    if ncam_data[row_idx, slot] > ncam_data[row_idx, prev_slot]:
                        data[row_idx, prev_slot] = -1
                        seen[new_id] = slot
                    else:
                        data[row_idx, slot] = -1
                    n_deduped += 1
                else:
                    seen[new_id] = slot

        fish_id_ds[...] = data

    if n_deduped:
        logger.info(
            "Deduplicated %d identity-split observations (kept higher n_cameras)",
            n_deduped,
        )
    logger.info("Remapped HDF5 written to %s", dst_path)
