"""Unit tests for RANSAC centroid ray clustering (cross-view fish association)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.segmentation.detector import Detection
from aquapose.tracking.associate import (
    FrameAssociations,
    _compute_mask_centroid,
    ransac_centroid_cluster,
)

# ---------------------------------------------------------------------------
# Synthetic rig helpers
# ---------------------------------------------------------------------------


def _make_overhead_camera(
    cam_x: float,
    cam_y: float,
    water_z: float = 1.0,
    fx: float = 1400.0,
    cx: float = 800.0,
    cy: float = 600.0,
) -> RefractiveProjectionModel:
    """Build a downward-looking camera at world position (cam_x, cam_y, 0).

    Cameras are at Z=0 in world frame; water surface at Z=water_z; fish below.
    With R=I: camera center C = -t = (cam_x, cam_y, 0).
    """
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.tensor([-cam_x, -cam_y, 0.0], dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(
        K=K, R=R, t=t, water_z=water_z, normal=normal, n_air=1.0, n_water=1.333
    )


def _make_3_camera_rig() -> dict[str, RefractiveProjectionModel]:
    """Build 3 overhead cameras arranged in a triangle, looking straight down."""
    water_z = 1.0
    positions = [(-0.5, -0.4), (0.5, -0.4), (0.0, 0.5)]
    cam_ids = ["cam_a", "cam_b", "cam_c"]
    return {
        cam_id: _make_overhead_camera(x, y, water_z)
        for cam_id, (x, y) in zip(cam_ids, positions, strict=True)
    }


def _blob_mask(
    u: float, v: float, radius: int = 10, H: int = 1200, W: int = 1600
) -> np.ndarray:
    """Create a circular blob mask at pixel (u, v)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    uu = np.arange(W)
    vv = np.arange(H)
    UU, VV = np.meshgrid(uu, vv)
    mask[(UU - u) ** 2 + (VV - v) ** 2 <= radius**2] = 255
    return mask


def _project_fish_to_detections(
    fish_3d: torch.Tensor,
    models: dict[str, RefractiveProjectionModel],
    H: int = 1200,
    W: int = 1600,
) -> dict[str, list[Detection]]:
    """Project a 3D fish centroid into all cameras, build Detection objects."""
    detections_per_camera: dict[str, list[Detection]] = {}
    pt = fish_3d.unsqueeze(0)  # (1, 3)
    for cam_id, model in models.items():
        with torch.no_grad():
            pixels, valid = model.project(pt)
        if valid[0]:
            u = float(pixels[0, 0])
            v = float(pixels[0, 1])
            # Only create detection if pixel is inside image bounds
            if 0 <= u < W and 0 <= v < H:
                mask = _blob_mask(u, v, radius=10, H=H, W=W)
                area = int(np.count_nonzero(mask))
                bbox = (max(0, int(u) - 10), max(0, int(v) - 10), 20, 20)
                det = Detection(bbox=bbox, mask=mask, area=area, confidence=1.0)
                detections_per_camera[cam_id] = [det]
    return detections_per_camera


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_camera_models() -> dict[str, RefractiveProjectionModel]:
    """Three overhead cameras in a triangle layout."""
    return _make_3_camera_rig()


@pytest.fixture
def two_fish_positions() -> list[torch.Tensor]:
    """Two well-separated fish positions below the water surface."""
    return [
        torch.tensor([0.0, 0.1, 1.5], dtype=torch.float32),
        torch.tensor([-0.2, -0.15, 1.5], dtype=torch.float32),
    ]


# ---------------------------------------------------------------------------
# Test 1: Two fish across three cameras
# ---------------------------------------------------------------------------


class TestTwoFishThreeCameras:
    """Test basic multi-fish clustering with 3 cameras."""

    def test_associations_returned(
        self,
        three_camera_models: dict[str, RefractiveProjectionModel],
        two_fish_positions: list[torch.Tensor],
    ) -> None:
        """Two fish should yield 2 high-confidence associations."""
        models = three_camera_models
        fish_positions = two_fish_positions

        # Build per-camera detections (one detection per fish per camera)
        detections_per_camera: dict[str, list[Detection]] = {cam: [] for cam in models}
        for fish_3d in fish_positions:
            per_cam = _project_fish_to_detections(fish_3d, models)
            for cam_id, dets in per_cam.items():
                detections_per_camera[cam_id].extend(dets)

        result = ransac_centroid_cluster(
            detections_per_camera,
            models,
            expected_count=2,
            n_iter=300,
            reprojection_threshold=20.0,
            min_cameras=2,
        )

        assert isinstance(result, FrameAssociations)
        high_conf = [a for a in result.associations if not a.is_low_confidence]
        assert len(high_conf) == 2, (
            f"Expected 2 high-confidence associations, got {len(high_conf)}"
        )

    def test_each_association_covers_min_cameras(
        self,
        three_camera_models: dict[str, RefractiveProjectionModel],
        two_fish_positions: list[torch.Tensor],
    ) -> None:
        """Each association should span at least 2 cameras."""
        models = three_camera_models
        detections_per_camera: dict[str, list[Detection]] = {cam: [] for cam in models}
        for fish_3d in two_fish_positions:
            per_cam = _project_fish_to_detections(fish_3d, models)
            for cam_id, dets in per_cam.items():
                detections_per_camera[cam_id].extend(dets)

        result = ransac_centroid_cluster(
            detections_per_camera, models, expected_count=2, n_iter=300
        )

        high_conf = [a for a in result.associations if not a.is_low_confidence]
        for assoc in high_conf:
            assert assoc.n_cameras >= 2, (
                f"Association has only {assoc.n_cameras} cameras"
            )

    def test_3d_centroids_close_to_ground_truth(
        self,
        three_camera_models: dict[str, RefractiveProjectionModel],
        two_fish_positions: list[torch.Tensor],
    ) -> None:
        """Triangulated 3D centroids should be within 0.05m of ground truth (XY)."""
        models = three_camera_models
        fish_positions = two_fish_positions

        detections_per_camera: dict[str, list[Detection]] = {cam: [] for cam in models}
        for fish_3d in fish_positions:
            per_cam = _project_fish_to_detections(fish_3d, models)
            for cam_id, dets in per_cam.items():
                detections_per_camera[cam_id].extend(dets)

        result = ransac_centroid_cluster(
            detections_per_camera,
            models,
            expected_count=2,
            n_iter=300,
            reprojection_threshold=20.0,
        )

        high_conf = [a for a in result.associations if not a.is_low_confidence]
        assert len(high_conf) == 2

        # Check each association matches one of the ground truth fish (XY only)
        gt_xy = [p[:2].numpy() for p in fish_positions]
        for assoc in high_conf:
            pred_xy = assoc.centroid_3d[:2]
            dists = [np.linalg.norm(pred_xy - gt) for gt in gt_xy]
            assert min(dists) < 0.05, (
                f"Nearest GT distance = {min(dists):.4f}m, expected < 0.05m"
            )

    def test_reprojection_residual_within_threshold(
        self,
        three_camera_models: dict[str, RefractiveProjectionModel],
        two_fish_positions: list[torch.Tensor],
    ) -> None:
        """Reprojection residual should be below the threshold."""
        models = three_camera_models
        detections_per_camera: dict[str, list[Detection]] = {cam: [] for cam in models}
        for fish_3d in two_fish_positions:
            per_cam = _project_fish_to_detections(fish_3d, models)
            for cam_id, dets in per_cam.items():
                detections_per_camera[cam_id].extend(dets)

        threshold = 20.0
        result = ransac_centroid_cluster(
            detections_per_camera,
            models,
            expected_count=2,
            n_iter=300,
            reprojection_threshold=threshold,
        )

        high_conf = [a for a in result.associations if not a.is_low_confidence]
        for assoc in high_conf:
            assert assoc.reprojection_residual < threshold, (
                f"Residual {assoc.reprojection_residual:.2f} px exceeds threshold {threshold} px"
            )


# ---------------------------------------------------------------------------
# Test 2: Prior-guided seeding
# ---------------------------------------------------------------------------


class TestPriorGuidedSeeding:
    """Test that seed_points improve convergence."""

    def test_prior_guided_finds_associations_with_few_iterations(
        self,
        three_camera_models: dict[str, RefractiveProjectionModel],
        two_fish_positions: list[torch.Tensor],
    ) -> None:
        """With good seed points, clustering should work with n_iter=20."""
        models = three_camera_models
        fish_positions = two_fish_positions

        detections_per_camera: dict[str, list[Detection]] = {cam: [] for cam in models}
        for fish_3d in fish_positions:
            per_cam = _project_fish_to_detections(fish_3d, models)
            for cam_id, dets in per_cam.items():
                detections_per_camera[cam_id].extend(dets)

        seed_points = [p.numpy() for p in fish_positions]

        result = ransac_centroid_cluster(
            detections_per_camera,
            models,
            expected_count=2,
            n_iter=20,  # Very few random iterations — relies on prior
            reprojection_threshold=20.0,
            seed_points=seed_points,
        )

        high_conf = [a for a in result.associations if not a.is_low_confidence]
        assert len(high_conf) == 2, (
            f"Prior-guided expected 2 associations with n_iter=20, got {len(high_conf)}"
        )

    def test_prior_guided_same_quality_as_random(
        self,
        three_camera_models: dict[str, RefractiveProjectionModel],
        two_fish_positions: list[torch.Tensor],
    ) -> None:
        """Prior-guided mode should find all fish at least as well as random."""
        models = three_camera_models
        fish_positions = two_fish_positions

        detections_per_camera: dict[str, list[Detection]] = {cam: [] for cam in models}
        for fish_3d in fish_positions:
            per_cam = _project_fish_to_detections(fish_3d, models)
            for cam_id, dets in per_cam.items():
                detections_per_camera[cam_id].extend(dets)

        seed_points = [p.numpy() for p in fish_positions]

        # With seeds, n_iter=0 should still work
        result = ransac_centroid_cluster(
            detections_per_camera,
            models,
            expected_count=2,
            n_iter=0,
            reprojection_threshold=20.0,
            seed_points=seed_points,
        )

        high_conf = [a for a in result.associations if not a.is_low_confidence]
        assert len(high_conf) == 2


# ---------------------------------------------------------------------------
# Test 3: Single-view detection flagged as low confidence
# ---------------------------------------------------------------------------


class TestSingleViewDetectionFlagged:
    """Test that single-view detections are flagged as low confidence."""

    def test_single_view_flagged(
        self, three_camera_models: dict[str, RefractiveProjectionModel]
    ) -> None:
        """A detection visible in only 1 camera should be low-confidence."""
        models = three_camera_models
        H, W = 1200, 1600

        # One fish visible only in cam_a
        mask = _blob_mask(800.0, 600.0, radius=10, H=H, W=W)
        det = Detection(
            bbox=(790, 590, 20, 20),
            mask=mask,
            area=int(np.count_nonzero(mask)),
            confidence=0.7,
        )
        detections_per_camera: dict[str, list[Detection]] = {
            "cam_a": [det],
            "cam_b": [],
            "cam_c": [],
        }

        result = ransac_centroid_cluster(
            detections_per_camera,
            models,
            expected_count=9,
            n_iter=50,
            min_cameras=2,
        )

        assert len(result.associations) == 1
        assoc = result.associations[0]
        assert assoc.is_low_confidence is True
        assert assoc.n_cameras == 1

    def test_single_view_confidence_from_detection(
        self, three_camera_models: dict[str, RefractiveProjectionModel]
    ) -> None:
        """Single-view association confidence should equal detection confidence."""
        models = three_camera_models
        H, W = 1200, 1600

        detection_confidence = 0.65
        mask = _blob_mask(800.0, 600.0, radius=10, H=H, W=W)
        det = Detection(
            bbox=(790, 590, 20, 20),
            mask=mask,
            area=int(np.count_nonzero(mask)),
            confidence=detection_confidence,
        )
        detections_per_camera: dict[str, list[Detection]] = {
            "cam_a": [det],
            "cam_b": [],
            "cam_c": [],
        }

        result = ransac_centroid_cluster(
            detections_per_camera, models, n_iter=10, min_cameras=2
        )

        assert len(result.associations) == 1
        assert abs(result.associations[0].confidence - detection_confidence) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: No double assignment
# ---------------------------------------------------------------------------


class TestNoDoubleAssignment:
    """Test that each detection is assigned to at most one fish."""

    def test_no_double_assignment(
        self,
        three_camera_models: dict[str, RefractiveProjectionModel],
        two_fish_positions: list[torch.Tensor],
    ) -> None:
        """Each detection should appear in at most one AssociationResult."""
        models = three_camera_models
        fish_positions = two_fish_positions

        detections_per_camera: dict[str, list[Detection]] = {cam: [] for cam in models}
        for fish_3d in fish_positions:
            per_cam = _project_fish_to_detections(fish_3d, models)
            for cam_id, dets in per_cam.items():
                detections_per_camera[cam_id].extend(dets)

        result = ransac_centroid_cluster(
            detections_per_camera,
            models,
            expected_count=2,
            n_iter=300,
            reprojection_threshold=20.0,
        )

        # Build set of (camera_id, detection_index) pairs seen in associations
        seen: set[tuple[str, int]] = set()
        for assoc in result.associations:
            for cam_id, det_idx in assoc.camera_detections.items():
                key = (cam_id, det_idx)
                assert key not in seen, (
                    f"Detection ({cam_id}, {det_idx}) assigned to multiple fish"
                )
                seen.add(key)

    def test_no_double_assignment_close_fish(
        self, three_camera_models: dict[str, RefractiveProjectionModel]
    ) -> None:
        """Close fish should also not produce double assignments."""
        models = three_camera_models

        # Two fish fairly close together
        fish_positions = [
            torch.tensor([0.05, 0.05, 1.5], dtype=torch.float32),
            torch.tensor([-0.05, -0.05, 1.5], dtype=torch.float32),
        ]

        detections_per_camera: dict[str, list[Detection]] = {cam: [] for cam in models}
        for fish_3d in fish_positions:
            per_cam = _project_fish_to_detections(fish_3d, models)
            for cam_id, dets in per_cam.items():
                detections_per_camera[cam_id].extend(dets)

        result = ransac_centroid_cluster(
            detections_per_camera, models, expected_count=2, n_iter=300
        )

        seen: set[tuple[str, int]] = set()
        for assoc in result.associations:
            for cam_id, det_idx in assoc.camera_detections.items():
                key = (cam_id, det_idx)
                assert key not in seen, (
                    f"Detection ({cam_id}, {det_idx}) assigned to multiple fish"
                )
                seen.add(key)


# ---------------------------------------------------------------------------
# Test 5: Mask centroid vs bbox center
# ---------------------------------------------------------------------------


class TestMaskCentroidNotBboxCenter:
    """Test that _compute_mask_centroid returns mask center-of-mass."""

    def test_centroid_not_bbox_center_off_center_mask(self) -> None:
        """Off-center mask should return pixel centroid, not bbox center."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=np.uint8)
        # Foreground blob in top-left of a large bbox
        mask[10:20, 10:20] = 255

        u, v = _compute_mask_centroid(mask)

        # Expected centroid is near the center of the 10x10 blob: (14.5, 14.5)
        assert abs(u - 14.5) < 1.0, f"u={u:.2f}, expected ~14.5"
        assert abs(v - 14.5) < 1.0, f"v={v:.2f}, expected ~14.5"

        # Bbox center would be (50, 50) if bbox covered the full image
        assert u < 50, "Centroid should not be at bbox center"
        assert v < 50, "Centroid should not be at bbox center"

    def test_centroid_on_symmetric_blob(self) -> None:
        """Symmetric circular blob should have centroid at its center."""
        H, W = 200, 200
        mask = np.zeros((H, W), dtype=np.uint8)
        center_u, center_v = 120, 80
        radius = 15
        uu = np.arange(W)
        vv = np.arange(H)
        UU, VV = np.meshgrid(uu, vv)
        mask[(UU - center_u) ** 2 + (VV - center_v) ** 2 <= radius**2] = 255

        u, v = _compute_mask_centroid(mask)

        assert abs(u - center_u) < 1.0, f"u={u:.2f}, expected {center_u}"
        assert abs(v - center_v) < 1.0, f"v={v:.2f}, expected {center_v}"

    def test_centroid_uses_foreground_pixels_only(self) -> None:
        """Centroid should ignore background (0) pixels."""
        mask = np.zeros((50, 100), dtype=np.uint8)
        # Foreground only on the right side
        mask[:, 80:90] = 255

        u, _v = _compute_mask_centroid(mask)

        # u should be around 84.5 (middle of cols 80-89), not near 50 (image center)
        assert u > 70, f"u={u:.2f}, should be near right side (>70)"

    def test_centroid_raises_on_empty_mask(self) -> None:
        """Should raise ValueError on a mask with no foreground pixels."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="no foreground"):
            _compute_mask_centroid(mask)


# ---------------------------------------------------------------------------
# Test 6: Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Test graceful handling of empty inputs."""

    def test_empty_detections_per_camera(
        self, three_camera_models: dict[str, RefractiveProjectionModel]
    ) -> None:
        """Empty detections_per_camera should return empty FrameAssociations."""
        result = ransac_centroid_cluster(
            detections_per_camera={},
            models=three_camera_models,
        )

        assert isinstance(result, FrameAssociations)
        assert result.associations == []
        assert result.unassigned == []

    def test_all_empty_camera_lists(
        self, three_camera_models: dict[str, RefractiveProjectionModel]
    ) -> None:
        """Dict with cameras but empty detection lists should return no associations."""
        detections_per_camera = {cam: [] for cam in three_camera_models}
        result = ransac_centroid_cluster(
            detections_per_camera=detections_per_camera,
            models=three_camera_models,
        )

        assert isinstance(result, FrameAssociations)
        assert result.associations == []

    def test_frame_index_preserved(
        self, three_camera_models: dict[str, RefractiveProjectionModel]
    ) -> None:
        """frame_index should be stored in FrameAssociations."""
        result = ransac_centroid_cluster(
            detections_per_camera={},
            models=three_camera_models,
            frame_index=42,
        )
        assert result.frame_index == 42

    def test_returns_frame_associations_type(
        self, three_camera_models: dict[str, RefractiveProjectionModel]
    ) -> None:
        """Return type should always be FrameAssociations."""
        result = ransac_centroid_cluster(
            detections_per_camera={},
            models=three_camera_models,
        )
        assert isinstance(result, FrameAssociations)
