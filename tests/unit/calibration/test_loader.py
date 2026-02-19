"""Tests for calibration loader module."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from aquacal.config.schema import (
    CalibrationResult,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    InterfaceParams,
)

from aquapose.calibration import (
    CalibrationData,
    CameraData,
    UndistortionMaps,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)


def create_synthetic_calibration(
    num_ring_cameras: int = 2,
    num_auxiliary_cameras: int = 1,
    water_z: float = 0.978,
) -> CalibrationResult:
    """Create a synthetic CalibrationResult for testing.

    Args:
        num_ring_cameras: Number of ring cameras (non-auxiliary).
        num_auxiliary_cameras: Number of auxiliary cameras.
        water_z: Z-coordinate of water surface.

    Returns:
        Synthetic CalibrationResult with known values.
    """
    cameras = {}

    for i in range(num_ring_cameras):
        name = f"ring{i}"
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0.05])
        R = np.eye(3)
        t = np.array([i * 0.1, 0.0, 0.0])

        cameras[name] = CameraCalibration(
            name=name,
            intrinsics=CameraIntrinsics(
                K=K,
                dist_coeffs=dist_coeffs,
                image_size=(640, 480),
                is_fisheye=False,
            ),
            extrinsics=CameraExtrinsics(R=R, t=t),
            water_z=water_z,
            is_auxiliary=False,
        )

    for i in range(num_auxiliary_cameras):
        name = f"aux{i}"
        K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.15, -0.25, 0.03, 0.04])
        R = np.eye(3)
        t = np.array([0.0, i * 0.1, 0.0])

        cameras[name] = CameraCalibration(
            name=name,
            intrinsics=CameraIntrinsics(
                K=K,
                dist_coeffs=dist_coeffs,
                image_size=(640, 480),
                is_fisheye=True,
            ),
            extrinsics=CameraExtrinsics(R=R, t=t),
            water_z=water_z,
            is_auxiliary=True,
        )

    interface = InterfaceParams(
        normal=np.array([0.0, 0.0, -1.0]),
        n_air=1.0,
        n_water=1.333,
    )

    return CalibrationResult(
        cameras=cameras,
        interface=interface,
        board=None,  # type: ignore[arg-type]
        diagnostics=None,  # type: ignore[arg-type]
        metadata=None,  # type: ignore[arg-type]
    )


class TestCameraData:
    """Tests for CameraData dataclass."""

    def test_creation(self):
        """Test CameraData creation with all required fields."""
        K = torch.eye(3, dtype=torch.float32)
        dist_coeffs = torch.zeros(5, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float32)
        t = torch.zeros(3, dtype=torch.float32)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=R,
            t=t,
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        assert cam.name == "test_cam"
        assert cam.K.shape == (3, 3)
        assert cam.dist_coeffs.shape == (5,)
        assert cam.R.shape == (3, 3)
        assert cam.t.shape == (3,)
        assert cam.image_size == (640, 480)
        assert not cam.is_fisheye
        assert not cam.is_auxiliary


class TestCalibrationData:
    """Tests for CalibrationData dataclass."""

    def test_ring_cameras_returns_sorted_non_auxiliary(self):
        """Test ring_cameras property returns sorted non-auxiliary camera names."""
        cam1 = CameraData(
            name="cam_b",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )
        cam2 = CameraData(
            name="cam_a",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )
        cam3 = CameraData(
            name="aux_cam",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(4, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )

        calib = CalibrationData(
            cameras={"cam_b": cam1, "cam_a": cam2, "aux_cam": cam3},
            water_z=0.978,
            interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            n_air=1.0,
            n_water=1.333,
        )

        ring = calib.ring_cameras
        assert ring == ["cam_a", "cam_b"]
        assert "aux_cam" not in ring

    def test_auxiliary_cameras_returns_sorted_auxiliary(self):
        """Test auxiliary_cameras property returns sorted auxiliary camera names."""
        cam1 = CameraData(
            name="ring_cam",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )
        cam2 = CameraData(
            name="aux_b",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(4, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )
        cam3 = CameraData(
            name="aux_a",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(4, dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )

        calib = CalibrationData(
            cameras={"ring_cam": cam1, "aux_b": cam2, "aux_a": cam3},
            water_z=0.978,
            interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            n_air=1.0,
            n_water=1.333,
        )

        aux = calib.auxiliary_cameras
        assert aux == ["aux_a", "aux_b"]
        assert "ring_cam" not in aux

    def test_camera_positions_identity_rotation(self):
        """Test camera_positions computes C = -R^T @ t with identity rotation."""
        R = torch.eye(3, dtype=torch.float32)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        cam = CameraData(
            name="test_cam",
            K=torch.eye(3, dtype=torch.float32),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            R=R,
            t=t,
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        calib = CalibrationData(
            cameras={"test_cam": cam},
            water_z=0.978,
            interface_normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            n_air=1.0,
            n_water=1.333,
        )

        positions = calib.camera_positions()
        assert "test_cam" in positions
        expected = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        assert torch.allclose(positions["test_cam"], expected, atol=1e-6)


class TestLoadCalibrationData:
    """Tests for load_calibration_data function."""

    def test_basic_loading(self):
        """Test basic loading with synthetic calibration data."""
        synthetic_result = create_synthetic_calibration(
            num_ring_cameras=2, num_auxiliary_cameras=1
        )

        with patch(
            "aquapose.calibration.loader.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            assert len(calib.cameras) == 3
            assert len(calib.ring_cameras) == 2
            assert len(calib.auxiliary_cameras) == 1
            assert calib.water_z == 0.978
            assert calib.n_air == 1.0
            assert calib.n_water == 1.333
            assert torch.allclose(
                calib.interface_normal,
                torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            )

    def test_tensor_shapes_and_dtypes(self):
        """Test that tensors have correct shapes and dtypes after loading."""
        synthetic_result = create_synthetic_calibration(
            num_ring_cameras=1, num_auxiliary_cameras=1
        )

        with patch(
            "aquapose.calibration.loader.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            ring_cam = calib.cameras["ring0"]
            assert ring_cam.K.shape == (3, 3)
            assert ring_cam.K.dtype == torch.float32
            assert ring_cam.dist_coeffs.shape == (5,)
            assert ring_cam.dist_coeffs.dtype == torch.float64
            assert ring_cam.R.shape == (3, 3)
            assert ring_cam.R.dtype == torch.float32
            assert ring_cam.t.shape == (3,)
            assert ring_cam.t.dtype == torch.float32
            assert not ring_cam.is_fisheye
            assert not ring_cam.is_auxiliary

            aux_cam = calib.cameras["aux0"]
            assert aux_cam.K.shape == (3, 3)
            assert aux_cam.K.dtype == torch.float32
            assert aux_cam.dist_coeffs.shape == (4,)
            assert aux_cam.dist_coeffs.dtype == torch.float64
            assert aux_cam.R.shape == (3, 3)
            assert aux_cam.R.dtype == torch.float32
            assert aux_cam.t.shape == (3,)
            assert aux_cam.t.dtype == torch.float32
            assert aux_cam.is_fisheye
            assert aux_cam.is_auxiliary

            assert calib.interface_normal.shape == (3,)
            assert calib.interface_normal.dtype == torch.float32

    def test_all_tensors_on_cpu(self):
        """Test that all tensors are on CPU by default."""
        synthetic_result = create_synthetic_calibration(num_ring_cameras=1)

        with patch(
            "aquapose.calibration.loader.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            cam = calib.cameras["ring0"]
            assert cam.K.device.type == "cpu"
            assert cam.dist_coeffs.device.type == "cpu"
            assert cam.R.device.type == "cpu"
            assert cam.t.device.type == "cpu"
            assert calib.interface_normal.device.type == "cpu"

    def test_camera_classification(self):
        """Test that cameras are correctly classified as ring or auxiliary."""
        synthetic_result = create_synthetic_calibration(
            num_ring_cameras=3, num_auxiliary_cameras=2
        )

        with patch(
            "aquapose.calibration.loader.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            ring = calib.ring_cameras
            aux = calib.auxiliary_cameras

            assert len(ring) == 3
            assert len(aux) == 2
            assert set(ring) == {"ring0", "ring1", "ring2"}
            assert set(aux) == {"aux0", "aux1"}
            assert ring == sorted(ring)
            assert aux == sorted(aux)

    def test_water_z_extracted_from_first_camera(self):
        """Test that water_z is extracted from the first camera entry."""
        water_z_test = 1.234
        synthetic_result = create_synthetic_calibration(water_z=water_z_test)

        with patch(
            "aquapose.calibration.loader.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            assert calib.water_z == water_z_test

    def test_t_vector_squeezed_from_2d(self):
        """Test that (3, 1) shaped t vectors are squeezed to (3,)."""
        cameras = {}
        name = "test_cam"
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0.05])
        R = np.eye(3)
        t_2d = np.array([[1.0], [2.0], [3.0]])  # shape (3, 1)

        cameras[name] = CameraCalibration(
            name=name,
            intrinsics=CameraIntrinsics(
                K=K, dist_coeffs=dist_coeffs, image_size=(640, 480), is_fisheye=False
            ),
            extrinsics=CameraExtrinsics(R=R, t=t_2d),
            water_z=0.978,
            is_auxiliary=False,
        )
        interface = InterfaceParams(
            normal=np.array([0.0, 0.0, -1.0]), n_air=1.0, n_water=1.333
        )
        synthetic_result = CalibrationResult(
            cameras=cameras,
            interface=interface,
            board=None,  # type: ignore[arg-type]
            diagnostics=None,  # type: ignore[arg-type]
            metadata=None,  # type: ignore[arg-type]
        )

        with patch(
            "aquapose.calibration.loader.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            cam = calib.cameras["test_cam"]
            assert cam.t.shape == (3,)
            assert torch.allclose(
                cam.t, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            )

    def test_interface_normal_squeezed_from_2d(self):
        """Test that (3, 1) shaped interface normals are squeezed to (3,)."""
        cameras = {}
        name = "test_cam"
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0.05])
        R = np.eye(3)
        t = np.array([0.0, 0.0, 0.0])

        cameras[name] = CameraCalibration(
            name=name,
            intrinsics=CameraIntrinsics(
                K=K, dist_coeffs=dist_coeffs, image_size=(640, 480), is_fisheye=False
            ),
            extrinsics=CameraExtrinsics(R=R, t=t),
            water_z=0.978,
            is_auxiliary=False,
        )
        interface = InterfaceParams(
            normal=np.array([[0.0], [0.0], [-1.0]]), n_air=1.0, n_water=1.333
        )
        synthetic_result = CalibrationResult(
            cameras=cameras,
            interface=interface,
            board=None,  # type: ignore[arg-type]
            diagnostics=None,  # type: ignore[arg-type]
            metadata=None,  # type: ignore[arg-type]
        )

        with patch(
            "aquapose.calibration.loader.aquacal_load_calibration",
            return_value=synthetic_result,
        ):
            calib = load_calibration_data("dummy_path.json")

            assert calib.interface_normal.shape == (3,)
            assert torch.allclose(
                calib.interface_normal,
                torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            )


class TestUndistortionMaps:
    """Tests for UndistortionMaps dataclass."""

    def test_creation(self):
        """Test UndistortionMaps creation."""
        K_new = torch.eye(3, dtype=torch.float32)
        map_x = np.zeros((480, 640), dtype=np.float32)
        map_y = np.zeros((480, 640), dtype=np.float32)

        undist = UndistortionMaps(K_new=K_new, map_x=map_x, map_y=map_y)

        assert undist.K_new.shape == (3, 3)
        assert undist.K_new.dtype == torch.float32
        assert undist.map_x.shape == (480, 640)
        assert undist.map_x.dtype == np.float32
        assert undist.map_y.shape == (480, 640)
        assert undist.map_y.dtype == np.float32


class TestComputeUndistortionMaps:
    """Tests for compute_undistortion_maps function."""

    def test_pinhole_zero_distortion(self):
        """Test pinhole undistortion with zero distortion coefficients."""
        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.zeros(5, dtype=torch.float64)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        undist = compute_undistortion_maps(cam)

        assert undist.K_new.shape == (3, 3)
        assert undist.K_new.dtype == torch.float32
        assert undist.map_x.shape == (480, 640)
        assert undist.map_y.shape == (480, 640)
        assert undist.map_x.dtype == np.float32
        assert undist.map_y.dtype == np.float32
        assert torch.allclose(undist.K_new, K, atol=1.0)

    def test_pinhole_with_distortion(self):
        """Test pinhole undistortion with non-zero distortion coefficients."""
        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.tensor([0.1, -0.2, 0.001, 0.002, 0.05], dtype=torch.float64)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        undist = compute_undistortion_maps(cam)

        assert undist.K_new.shape == (3, 3)
        assert undist.map_x.shape == (480, 640)
        assert undist.map_y.shape == (480, 640)
        assert undist.K_new[0, 0] > 0
        assert undist.K_new[1, 1] > 0

    def test_fisheye_undistortion_maps(self):
        """Test fisheye undistortion map generation."""
        K = torch.tensor(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.tensor([0.15, -0.25, 0.03, 0.04], dtype=torch.float64)

        cam = CameraData(
            name="fisheye_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )

        undist = compute_undistortion_maps(cam)

        assert undist.K_new.shape == (3, 3)
        assert undist.K_new.dtype == torch.float32
        assert undist.map_x.shape == (480, 640)
        assert undist.map_y.shape == (480, 640)
        assert undist.map_x.dtype == np.float32
        assert undist.map_y.dtype == np.float32
        assert undist.K_new[0, 0] > 0
        assert undist.K_new[1, 1] > 0

    def test_fisheye_and_pinhole_produce_different_results(self):
        """Test that is_fisheye flag dispatches to different undistortion paths."""
        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        pinhole_cam = CameraData(
            name="pinhole",
            K=K,
            dist_coeffs=torch.tensor(
                [0.1, -0.2, 0.001, 0.002, 0.05], dtype=torch.float64
            ),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        fisheye_cam = CameraData(
            name="fisheye",
            K=K,
            dist_coeffs=torch.tensor([0.1, -0.2, 0.001, 0.002], dtype=torch.float64),
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=True,
            is_auxiliary=True,
        )

        pinhole_undist = compute_undistortion_maps(pinhole_cam)
        fisheye_undist = compute_undistortion_maps(fisheye_cam)

        assert pinhole_undist.K_new.shape == (3, 3)
        assert fisheye_undist.K_new.shape == (3, 3)
        assert not torch.allclose(pinhole_undist.K_new, fisheye_undist.K_new)


class TestUndistortImage:
    """Tests for undistort_image function."""

    def test_shape_and_dtype_preserved(self):
        """Test that undistort_image preserves input shape and dtype."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.zeros(5, dtype=torch.float64)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        undist = compute_undistortion_maps(cam)
        undistorted = undistort_image(image, undist)

        assert undistorted.shape == image.shape
        assert undistorted.dtype == image.dtype

    def test_zero_distortion_near_identity(self):
        """Test that zero distortion produces nearly identical output."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.zeros(5, dtype=torch.float64)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        undist = compute_undistortion_maps(cam)
        undistorted = undistort_image(image, undist)

        diff = np.abs(undistorted.astype(np.float32) - image.astype(np.float32))
        assert np.mean(diff) < 5.0

    def test_nonzero_distortion_modifies_image(self):
        """Test that non-zero distortion actually modifies the image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(0, 480, 40):
            for j in range(0, 640, 40):
                if (i // 40 + j // 40) % 2 == 0:
                    image[i : i + 40, j : j + 40] = 255

        K = torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        dist_coeffs = torch.tensor([0.3, -0.5, 0.002, 0.001, 0.1], dtype=torch.float64)

        cam = CameraData(
            name="test_cam",
            K=K,
            dist_coeffs=dist_coeffs,
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
            image_size=(640, 480),
            is_fisheye=False,
            is_auxiliary=False,
        )

        undist = compute_undistortion_maps(cam)
        undistorted = undistort_image(image, undist)

        assert not np.array_equal(undistorted, image)


@pytest.mark.parametrize("num_cameras", [1, 2, 13])
def test_load_correct_camera_count(num_cameras: int):
    """Test that the correct number of cameras is loaded."""
    synthetic_result = create_synthetic_calibration(
        num_ring_cameras=num_cameras, num_auxiliary_cameras=0
    )

    with patch(
        "aquapose.calibration.loader.aquacal_load_calibration",
        return_value=synthetic_result,
    ):
        calib = load_calibration_data("dummy_path.json")
        assert len(calib.cameras) == num_cameras
