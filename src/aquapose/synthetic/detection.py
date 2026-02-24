"""Project 3D fish trajectories to per-camera Detection objects with noise.

Converts TrajectoryResult (multi-fish 3D paths) into SyntheticDataset objects
containing per-frame, per-camera Detection instances compatible with
FishTracker.update(). Supports configurable miss rates, false positive rates,
and centroid jitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.segmentation.detector import Detection
from aquapose.synthetic.trajectory import TrajectoryResult


@dataclass
class NoiseConfig:
    """Noise model parameters for synthetic detection generation.

    Attributes:
        base_miss_rate: Base probability that a valid fish projection is missed
            (not detected) per camera per frame.
        base_false_positive_rate: Expected number of false positive detections
            per fish per camera per frame (Poisson rate).
        centroid_noise_std: Standard deviation of centroid position noise in
            pixels. Applied as isotropic Gaussian jitter.
        bbox_noise_std: Standard deviation of bounding box size noise in pixels.
            Applied independently to width and height.
        velocity_miss_scale: Additional miss rate contribution per unit of
            normalised fish speed (speed / speed_threshold). Higher-speed fish
            are harder to detect in real images.
        speed_threshold: Speed normalisation factor in m/s for velocity-dependent
            miss rate computation.
        velocity_noise_scale: Additional centroid noise scale factor multiplied
            by normalised fish speed. Faster fish produce blurrier detections.

    Note:
        Occlusion-dependent and coalescence noise are not implemented.
        These require bounding-box overlap computation and are deferred as a
        future extension.
    """

    base_miss_rate: float = 0.06
    base_false_positive_rate: float = 0.06
    centroid_noise_std: float = 3.0
    bbox_noise_std: float = 2.0
    velocity_miss_scale: float = 0.15
    speed_threshold: float = 0.3
    velocity_noise_scale: float = 0.5


@dataclass
class DetectionGenConfig:
    """Configuration for synthetic detection generation.

    Attributes:
        noise: Noise model parameters.
        fish_bbox_size: Approximate bounding box size (width, height) in pixels
            for an 8 cm fish at typical depth. Used as the nominal bbox before
            noise is applied. This is a simplification vs. projecting the full
            fish ellipsoid — adequate for tracker evaluation where exact bbox
            shape is not the variable under test.
    """

    noise: NoiseConfig = field(default_factory=NoiseConfig)
    fish_bbox_size: tuple[float, float] = (40.0, 25.0)


@dataclass
class GroundTruthEntry:
    """Ground truth record linking a detection back to its source fish.

    Attributes:
        fish_id: Integer fish identifier.
        true_centroid_px: True (noise-free) pixel centroid (u, v).
        was_detected: Whether this fish was actually detected in this camera.
    """

    fish_id: int
    true_centroid_px: tuple[float, float]
    was_detected: bool


@dataclass
class SyntheticFrame:
    """All detection data for a single frame.

    Attributes:
        frame_index: Frame number (0-based).
        detections_per_camera: Dict mapping camera ID to list of Detection
            objects. Format is identical to FishTracker.update() input.
        ground_truth: Dict mapping camera ID to list of GroundTruthEntry,
            one entry per fish that had a valid projection in that camera.
    """

    frame_index: int
    detections_per_camera: dict[str, list[Detection]]
    ground_truth: dict[str, list[GroundTruthEntry]]


@dataclass
class SyntheticDataset:
    """Complete multi-frame synthetic detection dataset.

    Attributes:
        frames: List of SyntheticFrame, one per simulated frame.
        metadata: Dict with simulation metadata (n_fish, n_frames, seed, config
            repr) for reproducibility and logging.
    """

    frames: list[SyntheticFrame]
    metadata: dict[str, object]


def _image_size_from_model(model: RefractiveProjectionModel) -> tuple[int, int]:
    """Estimate image dimensions from the camera intrinsic matrix.

    Uses the principal point (cx, cy) convention: W = round(2 * cx),
    H = round(2 * cy). This matches the Phase 04 convention used in
    RefractiveProjectionModel.

    Args:
        model: The camera projection model.

    Returns:
        Tuple (W, H) — image width and height in pixels.
    """
    cx = float(model.K[0, 2].item())
    cy = float(model.K[1, 2].item())
    return round(2 * cx), round(2 * cy)


def generate_detection_dataset(
    trajectory: TrajectoryResult,
    models: dict[str, RefractiveProjectionModel],
    noise_config: NoiseConfig | None = None,
    det_config: DetectionGenConfig | None = None,
    random_seed: int | None = None,
) -> SyntheticDataset:
    """Generate per-frame, per-camera Detection objects from a trajectory.

    Projects each fish's 3D position through every camera at every frame using
    RefractiveProjectionModel.project. Applies configurable miss rates, centroid
    jitter, and false positive generation to produce a realistic synthetic
    detection stream compatible with FishTracker.update().

    Args:
        trajectory: Multi-fish 3D trajectory from generate_trajectories().
        models: Dict mapping camera ID to RefractiveProjectionModel.
        noise_config: Noise model parameters. Defaults to NoiseConfig() if None.
        det_config: Detection generation config. Defaults to DetectionGenConfig()
            if None.
        random_seed: Seed for the noise RNG. If None, uses
            trajectory.config.random_seed + 1 (if available) for reproducibility.

    Returns:
        SyntheticDataset with one SyntheticFrame per trajectory frame.
    """
    if noise_config is None:
        noise_config = NoiseConfig()
    if det_config is None:
        det_config = DetectionGenConfig(noise=noise_config)

    # Determine seed
    if random_seed is None:
        cfg_seed = trajectory.config.random_seed
        random_seed = (cfg_seed + 1) if cfg_seed is not None else 0

    rng = np.random.default_rng(random_seed)

    nom_w, nom_h = det_config.fish_bbox_size
    states = trajectory.states  # (n_frames, n_fish, 7)
    n_fish = trajectory.n_fish
    n_frames = trajectory.n_frames

    # Precompute image sizes per camera
    img_sizes: dict[str, tuple[int, int]] = {
        cam_id: _image_size_from_model(model) for cam_id, model in models.items()
    }

    frames: list[SyntheticFrame] = []

    for frame_idx in range(n_frames):
        frame_positions = states[frame_idx, :, :3]  # (n_fish, 3)
        frame_speeds = states[frame_idx, :, 5]  # (n_fish,)

        pts_torch = torch.from_numpy(frame_positions.astype(np.float32))  # (n_fish, 3)

        detections_per_camera: dict[str, list[Detection]] = {}
        ground_truth: dict[str, list[GroundTruthEntry]] = {}

        for cam_id, model in models.items():
            cam_detections: list[Detection] = []
            cam_gt: list[GroundTruthEntry] = []

            # Project all fish for this frame/camera in one batch call
            pixels, valid = model.project(pts_torch)  # (n_fish, 2), (n_fish,)
            pixels_np = pixels.detach().cpu().numpy()  # (n_fish, 2)
            valid_np = valid.detach().cpu().numpy()  # (n_fish,)

            img_w, img_h = img_sizes[cam_id]

            for fish_idx in range(n_fish):
                if not valid_np[fish_idx]:
                    continue  # fish not visible in this camera

                u_true = float(pixels_np[fish_idx, 0])
                v_true = float(pixels_np[fish_idx, 1])

                # Velocity-dependent miss rate
                speed = float(frame_speeds[fish_idx])
                speed_norm = speed / (noise_config.speed_threshold + 1e-12)
                miss_prob = float(
                    np.clip(
                        noise_config.base_miss_rate
                        + noise_config.velocity_miss_scale * speed_norm,
                        0.0,
                        1.0,
                    )
                )

                detected = bool(rng.random() > miss_prob)

                cam_gt.append(
                    GroundTruthEntry(
                        fish_id=fish_idx,
                        true_centroid_px=(u_true, v_true),
                        was_detected=detected,
                    )
                )

                if not detected:
                    continue

                # Apply centroid jitter (velocity-scaled)
                jitter_scale = 1.0 + noise_config.velocity_noise_scale * speed_norm
                sigma_c = noise_config.centroid_noise_std * jitter_scale
                u_noisy = u_true + float(rng.normal(0.0, sigma_c))
                v_noisy = v_true + float(rng.normal(0.0, sigma_c))

                # Apply bbox size noise
                w = max(
                    1,
                    round(nom_w + float(rng.normal(0.0, noise_config.bbox_noise_std))),
                )
                h = max(
                    1,
                    round(nom_h + float(rng.normal(0.0, noise_config.bbox_noise_std))),
                )

                # Convert centroid to top-left bbox corner, clamped to image
                x = int(np.clip(round(u_noisy - w // 2), 0, img_w - 1))
                y = int(np.clip(round(v_noisy - h // 2), 0, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)

                cam_detections.append(
                    Detection(
                        bbox=(x, y, w, h),
                        mask=None,
                        area=w * h,
                        confidence=1.0,
                    )
                )

            # False positives: Poisson(base_fp_rate * n_fish) count
            n_fp = int(rng.poisson(noise_config.base_false_positive_rate * n_fish))
            for _ in range(n_fp):
                fp_u = float(rng.uniform(0, img_w))
                fp_v = float(rng.uniform(0, img_h))
                fp_w = max(
                    1,
                    round(nom_w + float(rng.normal(0.0, noise_config.bbox_noise_std))),
                )
                fp_h = max(
                    1,
                    round(nom_h + float(rng.normal(0.0, noise_config.bbox_noise_std))),
                )
                fp_x = int(np.clip(round(fp_u - fp_w // 2), 0, img_w - 1))
                fp_y = int(np.clip(round(fp_v - fp_h // 2), 0, img_h - 1))
                fp_w = min(fp_w, img_w - fp_x)
                fp_h = min(fp_h, img_h - fp_y)
                cam_detections.append(
                    Detection(
                        bbox=(fp_x, fp_y, fp_w, fp_h),
                        mask=None,
                        area=fp_w * fp_h,
                        confidence=0.5,
                    )
                )

            detections_per_camera[cam_id] = cam_detections
            ground_truth[cam_id] = cam_gt

        frames.append(
            SyntheticFrame(
                frame_index=frame_idx,
                detections_per_camera=detections_per_camera,
                ground_truth=ground_truth,
            )
        )

    metadata: dict[str, object] = {
        "n_fish": n_fish,
        "n_frames": n_frames,
        "seed": random_seed,
        "n_cameras": len(models),
        "camera_ids": list(models.keys()),
    }

    return SyntheticDataset(frames=frames, metadata=metadata)
