"""Tests for RefractiveCamera and RefractiveSilhouetteRenderer."""

from __future__ import annotations

import pytest
import torch

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.mesh.builder import build_fish_mesh
from aquapose.mesh.state import FishState
from aquapose.optimization.renderer import (
    RefractiveCamera,
    RefractiveSilhouetteRenderer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_side_camera() -> RefractiveProjectionModel:
    """Synthetic side-view camera at (0.635, 0, 0) looking toward origin.

    K intrinsics: fx=fy=1400, cx=800, cy=600 (full image 1600x1200).
    Water surface at world Z=0.978. Fish at depth ~1.5m.
    """
    K = torch.tensor(
        [[1400.0, 0.0, 800.0], [0.0, 1400.0, 600.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.tensor([-0.635, 0.0, 0.0], dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(K, R, t, 0.978, normal, 1.0, 1.333)


@pytest.fixture
def camera_model() -> RefractiveProjectionModel:
    """Synthetic side-view camera for renderer tests."""
    return _make_side_camera()


@pytest.fixture
def camera_model_2() -> RefractiveProjectionModel:
    """Second synthetic camera for multi-camera tests (offset in Y)."""
    K = torch.tensor(
        [[1400.0, 0.0, 800.0], [0.0, 1400.0, 600.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    # Rotate 90 degrees around Z to place camera along Y axis.
    R = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    t = torch.tensor([0.0, -0.635, 0.0], dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return RefractiveProjectionModel(K, R, t, 0.978, normal, 1.0, 1.333)


@pytest.fixture
def fish_state() -> FishState:
    """A simple horizontal fish at depth 1.5 m, 15 cm body length."""
    return FishState(
        p=torch.tensor([0.0, 0.0, 1.5], dtype=torch.float32),
        psi=torch.tensor(0.0, dtype=torch.float32),
        theta=torch.tensor(0.0, dtype=torch.float32),
        kappa=torch.tensor(0.0, dtype=torch.float32),
        s=torch.tensor(0.15, dtype=torch.float32),
    )


@pytest.fixture
def renderer() -> RefractiveSilhouetteRenderer:
    """Standard renderer at 200x200 resolution for tests."""
    return RefractiveSilhouetteRenderer(image_size=(200, 200))


# Camera native image size (1200 x 1600) derived from K cx=800, cy=600.
_CAM_IMAGE_SIZE = (1200, 1600)


# ---------------------------------------------------------------------------
# RefractiveCamera tests
# ---------------------------------------------------------------------------


class TestRefractiveCamera:
    """Unit tests for the RefractiveCamera PyTorch3D wrapper."""

    def test_project_to_ndc_shape(self, camera_model: RefractiveProjectionModel):
        """project_to_ndc returns (N, 3) tensor for (N, 3) input."""
        cam = RefractiveCamera(camera_model, camera_image_size=_CAM_IMAGE_SIZE)
        pts = torch.tensor(
            [[0.0, 0.0, 1.5], [0.1, 0.05, 1.6], [-0.1, 0.0, 1.4]],
            dtype=torch.float32,
        )
        ndc = cam.project_to_ndc(pts)
        assert ndc.shape == (3, 3), f"Expected (3, 3), got {ndc.shape}"

    def test_project_to_ndc_range(self, camera_model: RefractiveProjectionModel):
        """NDC x and y outputs are in [-1, 1] for underwater points near camera axis."""
        cam = RefractiveCamera(camera_model, camera_image_size=_CAM_IMAGE_SIZE)
        pts = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32)
        ndc = cam.project_to_ndc(pts)
        # Fish near axis should project to NDC near center.
        assert ndc[0, 0].abs() < 1.0, f"NDC x={ndc[0, 0]:.3f} out of expected range"
        assert ndc[0, 1].abs() < 1.0, f"NDC y={ndc[0, 1]:.3f} out of expected range"

    def test_project_to_ndc_z_from_world(self, camera_model: RefractiveProjectionModel):
        """Z component of NDC output matches world-space Z of input points."""
        cam = RefractiveCamera(camera_model, camera_image_size=_CAM_IMAGE_SIZE)
        pts = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 2.0]], dtype=torch.float32)
        ndc = cam.project_to_ndc(pts)
        assert torch.allclose(ndc[:, 2], pts[:, 2]), (
            "Z should equal world Z for depth sort"
        )

    def test_project_to_ndc_gradient_flows(
        self, camera_model: RefractiveProjectionModel
    ):
        """Gradient flows from NDC output back to world-point input."""
        cam = RefractiveCamera(camera_model, camera_image_size=_CAM_IMAGE_SIZE)
        pts = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32, requires_grad=True)
        ndc = cam.project_to_ndc(pts)
        ndc.sum().backward()
        assert pts.grad is not None, "No gradient to world points"
        assert torch.isfinite(pts.grad).all(), "NaN gradient to world points"

    def test_get_image_size(self, camera_model: RefractiveProjectionModel):
        """get_image_size returns (H, W) tuple."""
        cam = RefractiveCamera(camera_model, camera_image_size=(1200, 1600))
        assert cam.get_image_size() == (1200, 1600)

    def test_image_size_derived_from_K(self, camera_model: RefractiveProjectionModel):
        """When camera_image_size is None, image size is derived from K (cx, cy)."""
        cam = RefractiveCamera(camera_model)  # K has cx=800, cy=600
        assert cam.H == 1200, f"Expected H=1200, got {cam.H}"
        assert cam.W == 1600, f"Expected W=1600, got {cam.W}"


# ---------------------------------------------------------------------------
# RefractiveSilhouetteRenderer tests
# ---------------------------------------------------------------------------


class TestRefractiveSilhouetteRenderer:
    """Unit tests for RefractiveSilhouetteRenderer."""

    def test_renderer_produces_nonempty_silhouette(
        self,
        renderer: RefractiveSilhouetteRenderer,
        fish_state: FishState,
        camera_model: RefractiveProjectionModel,
    ):
        """Rendered alpha map has nonzero pixels for a fish within view frustum."""
        meshes = build_fish_mesh([fish_state])
        result = renderer.render(meshes, [camera_model], ["cam0"])
        alpha = result["cam0"]
        assert alpha.shape == (200, 200), f"Expected (200, 200), got {alpha.shape}"
        assert alpha.max() > 0.0, "Alpha map is all-zero — fish not visible in render"

    def test_renderer_alpha_range(
        self,
        renderer: RefractiveSilhouetteRenderer,
        fish_state: FishState,
        camera_model: RefractiveProjectionModel,
    ):
        """All alpha values are in [0, 1]."""
        meshes = build_fish_mesh([fish_state])
        result = renderer.render(meshes, [camera_model], ["cam0"])
        alpha = result["cam0"]
        assert alpha.min() >= 0.0, f"Alpha below 0: min={alpha.min()}"
        assert alpha.max() <= 1.0 + 1e-6, f"Alpha above 1: max={alpha.max()}"

    def test_renderer_gradient_flow_all_params(
        self,
        renderer: RefractiveSilhouetteRenderer,
        camera_model: RefractiveProjectionModel,
    ):
        """Gradients flow from rendered silhouette through all 5 FishState params."""
        state = FishState(
            p=torch.tensor([0.0, 0.0, 1.5], dtype=torch.float32, requires_grad=True),
            psi=torch.tensor(0.0, dtype=torch.float32, requires_grad=True),
            theta=torch.tensor(0.0, dtype=torch.float32, requires_grad=True),
            kappa=torch.tensor(0.0, dtype=torch.float32, requires_grad=True),
            s=torch.tensor(0.15, dtype=torch.float32, requires_grad=True),
        )
        meshes = build_fish_mesh([state])
        result = renderer.render(meshes, [camera_model], ["cam0"])
        alpha = result["cam0"]
        loss = alpha.sum()
        loss.backward()

        assert state.p.grad is not None, "No gradient to p (position)"
        assert state.psi.grad is not None, "No gradient to psi (yaw)"
        assert state.theta.grad is not None, "No gradient to theta (pitch)"
        assert state.kappa.grad is not None, "No gradient to kappa (curvature)"
        assert state.s.grad is not None, "No gradient to s (scale)"

        assert torch.isfinite(state.p.grad).all(), "NaN gradient to p"
        assert torch.isfinite(state.psi.grad), "NaN gradient to psi"
        assert torch.isfinite(state.theta.grad), "NaN gradient to theta"
        assert torch.isfinite(state.kappa.grad), "NaN gradient to kappa"
        assert torch.isfinite(state.s.grad), "NaN gradient to s"

    def test_renderer_multiple_cameras(
        self,
        renderer: RefractiveSilhouetteRenderer,
        fish_state: FishState,
        camera_model: RefractiveProjectionModel,
        camera_model_2: RefractiveProjectionModel,
    ):
        """Render with 2+ cameras returns dict with correct keys and shapes."""
        meshes = build_fish_mesh([fish_state])
        result = renderer.render(
            meshes, [camera_model, camera_model_2], ["cam0", "cam1"]
        )
        assert set(result.keys()) == {"cam0", "cam1"}, (
            f"Unexpected keys: {result.keys()}"
        )
        assert result["cam0"].shape == (200, 200)
        assert result["cam1"].shape == (200, 200)

    def test_renderer_raises_on_mismatched_lengths(
        self,
        renderer: RefractiveSilhouetteRenderer,
        fish_state: FishState,
        camera_model: RefractiveProjectionModel,
    ):
        """Raises ValueError when cameras and camera_ids have different lengths."""
        meshes = build_fish_mesh([fish_state])
        with pytest.raises(ValueError, match="must have the same length"):
            renderer.render(meshes, [camera_model], ["cam0", "cam1"])
