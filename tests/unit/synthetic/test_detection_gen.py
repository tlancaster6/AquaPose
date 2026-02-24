"""Unit tests for synthetic detection generation from 3D trajectories."""

from __future__ import annotations

from aquapose.segmentation.detector import Detection
from aquapose.synthetic import (
    NoiseConfig,
    SyntheticDataset,
    build_fabricated_rig,
    generate_detection_dataset,
    generate_trajectories,
)
from aquapose.synthetic.trajectory import TrajectoryConfig


def _make_trajectory(n_fish: int = 3, duration: float = 1.0, seed: int = 42):
    """Return a short trajectory for testing."""
    cfg = TrajectoryConfig(
        n_fish=n_fish,
        duration_seconds=duration,
        fps=10.0,
        random_seed=seed,
    )
    return generate_trajectories(cfg)


def _make_rig():
    """Return a 2x2 fabricated rig for fast tests."""
    return build_fabricated_rig(n_cameras_x=2, n_cameras_y=2)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


def test_detection_dataset_structure() -> None:
    """Produces correct number of frames, each with detections_per_camera keyed by camera IDs."""
    traj = _make_trajectory(n_fish=3, duration=1.5)
    rig = _make_rig()
    dataset = generate_detection_dataset(traj, rig)

    expected_frames = traj.n_frames
    assert isinstance(dataset, SyntheticDataset)
    assert len(dataset.frames) == expected_frames

    cam_ids = set(rig.keys())
    for frame in dataset.frames:
        assert set(frame.detections_per_camera.keys()) == cam_ids, (
            f"Frame {frame.frame_index} missing cameras"
        )


def test_detections_are_detection_type() -> None:
    """Each detection is an instance of Detection with a valid bbox tuple."""
    traj = _make_trajectory(n_fish=2)
    rig = _make_rig()
    noise = NoiseConfig(base_miss_rate=0.0, base_false_positive_rate=0.0)
    dataset = generate_detection_dataset(traj, rig, noise_config=noise)

    for frame in dataset.frames:
        for cam_id, dets in frame.detections_per_camera.items():
            for det in dets:
                assert isinstance(det, Detection), (
                    f"Frame {frame.frame_index} cam {cam_id}: expected Detection, got {type(det)}"
                )
                x, y, w, h = det.bbox
                assert w > 0, f"Bbox width non-positive: {det.bbox}"
                assert h > 0, f"Bbox height non-positive: {det.bbox}"
                assert x >= 0, f"Bbox x negative: {det.bbox}"
                assert y >= 0, f"Bbox y negative: {det.bbox}"


# ---------------------------------------------------------------------------
# Noise model tests
# ---------------------------------------------------------------------------


def test_no_noise_all_detected() -> None:
    """With base_miss_rate=0 and base_fp_rate=0, every valid projection produces a detection."""
    traj = _make_trajectory(n_fish=3, duration=2.0)
    rig = _make_rig()
    noise = NoiseConfig(
        base_miss_rate=0.0,
        base_false_positive_rate=0.0,
        velocity_miss_scale=0.0,
    )
    dataset = generate_detection_dataset(traj, rig, noise_config=noise)

    # Count total detections and ground truth detections across all frames/cameras
    total_detected = 0
    total_gt_detected = 0
    for frame in dataset.frames:
        for cam_id in rig:
            total_detected += len(frame.detections_per_camera[cam_id])
            total_gt_detected += sum(
                1 for entry in frame.ground_truth[cam_id] if entry.was_detected
            )

    assert total_detected == total_gt_detected, (
        f"With zero noise: detected {total_detected} != gt_detected {total_gt_detected}"
    )


def test_miss_rate_reduces_detections() -> None:
    """With base_miss_rate=0.5, total detections are significantly fewer than no-noise case."""
    traj = _make_trajectory(n_fish=5, duration=3.0, seed=7)
    rig = _make_rig()

    noise_no_miss = NoiseConfig(
        base_miss_rate=0.0,
        base_false_positive_rate=0.0,
        velocity_miss_scale=0.0,
    )
    noise_high_miss = NoiseConfig(
        base_miss_rate=0.5,
        base_false_positive_rate=0.0,
        velocity_miss_scale=0.0,
    )

    ds_no_miss = generate_detection_dataset(
        traj, rig, noise_config=noise_no_miss, random_seed=0
    )
    ds_high_miss = generate_detection_dataset(
        traj, rig, noise_config=noise_high_miss, random_seed=0
    )

    def total_dets(ds: SyntheticDataset) -> int:
        return sum(
            len(dets)
            for frame in ds.frames
            for dets in frame.detections_per_camera.values()
        )

    n_no_miss = total_dets(ds_no_miss)
    n_high_miss = total_dets(ds_high_miss)

    # With 50% miss rate, should have substantially fewer detections
    assert n_high_miss < n_no_miss * 0.85, (
        f"High miss rate ({n_high_miss}) not significantly less than no-miss ({n_no_miss})"
    )


def test_false_positives_added() -> None:
    """With high base_fp_rate, total detections exceed number of true fish projections."""
    traj = _make_trajectory(n_fish=3, duration=2.0)
    rig = _make_rig()

    noise_no_fp = NoiseConfig(
        base_miss_rate=0.0,
        base_false_positive_rate=0.0,
        velocity_miss_scale=0.0,
    )
    noise_high_fp = NoiseConfig(
        base_miss_rate=0.0,
        base_false_positive_rate=0.5,
        velocity_miss_scale=0.0,
    )

    ds_no_fp = generate_detection_dataset(
        traj, rig, noise_config=noise_no_fp, random_seed=0
    )
    ds_high_fp = generate_detection_dataset(
        traj, rig, noise_config=noise_high_fp, random_seed=0
    )

    def total_dets(ds: SyntheticDataset) -> int:
        return sum(
            len(dets)
            for frame in ds.frames
            for dets in frame.detections_per_camera.values()
        )

    n_no_fp = total_dets(ds_no_fp)
    n_high_fp = total_dets(ds_high_fp)

    assert n_high_fp > n_no_fp, (
        f"High FP rate ({n_high_fp}) should exceed no-FP ({n_no_fp})"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_with_seed() -> None:
    """Same seed produces identical datasets."""
    traj = _make_trajectory(n_fish=4, duration=2.0)
    rig = _make_rig()
    noise = NoiseConfig()

    ds1 = generate_detection_dataset(traj, rig, noise_config=noise, random_seed=99)
    ds2 = generate_detection_dataset(traj, rig, noise_config=noise, random_seed=99)

    assert len(ds1.frames) == len(ds2.frames)
    for f1, f2 in zip(ds1.frames, ds2.frames, strict=True):
        for cam_id in rig:
            dets1 = f1.detections_per_camera[cam_id]
            dets2 = f2.detections_per_camera[cam_id]
            assert len(dets1) == len(dets2), (
                f"Frame {f1.frame_index} cam {cam_id}: "
                f"det count differs {len(dets1)} vs {len(dets2)}"
            )
            for d1, d2 in zip(dets1, dets2, strict=True):
                assert d1.bbox == d2.bbox, f"Bbox mismatch: {d1.bbox} vs {d2.bbox}"
