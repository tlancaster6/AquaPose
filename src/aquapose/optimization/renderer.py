"""Differentiable silhouette renderer using refractive projection and PyTorch3D."""

from __future__ import annotations

import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    FoVOrthographicCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes

from aquapose.calibration.projection import RefractiveProjectionModel


def _image_size_from_K(K: torch.Tensor) -> tuple[int, int]:
    """Estimate camera image size (H, W) from intrinsic matrix K.

    Assumes the principal point lies at the image center, so
    W ≈ 2 * cx and H ≈ 2 * cy.

    Args:
        K: Camera intrinsic matrix, shape (3, 3), float32.

    Returns:
        (H, W) as integer pixel dimensions.
    """
    cx = float(K[0, 2].item())
    cy = float(K[1, 2].item())
    return round(cy * 2), round(cx * 2)


class RefractiveCamera:
    """PyTorch3D-compatible camera wrapping refractive projection into NDC coords.

    Projects world-space mesh vertices through
    ``RefractiveProjectionModel.project()`` (10-iteration fixed Newton-Raphson
    Snell's-law solver, fully differentiable) and converts to PyTorch3D NDC space.

    This class is used internally by ``RefractiveSilhouetteRenderer``. The NDC
    output is then rendered with an identity ``FoVOrthographicCameras``, keeping
    the refractive projection inside the autograd graph.

    Args:
        model: Refractive projection model for air-water interface.
        camera_image_size: Native (H, W) dimensions of the camera sensor in pixels,
            used to normalize pixel coordinates to NDC. Derived from K if not given.
    """

    def __init__(
        self,
        model: RefractiveProjectionModel,
        camera_image_size: tuple[int, int] | None = None,
    ) -> None:
        self.model = model
        if camera_image_size is None:
            camera_image_size = _image_size_from_K(model.K)
        self.H, self.W = camera_image_size

    def project_to_ndc(self, world_pts: torch.Tensor) -> torch.Tensor:
        """Project world points to PyTorch3D NDC space via the refractive model.

        Projects world-space vertices through ``RefractiveProjectionModel.project()``,
        which performs 10 fixed Newton-Raphson iterations to find the Snell's-law
        refraction point at the air-water interface. Results are converted from pixel
        coordinates to NDC space in the convention expected by ``MeshRasterizer``
        (x: left-to-right in [-1, 1], y: bottom-to-top in [-1, 1]).

        Gradients flow from the NDC coordinates back through the refractive
        projection to the input world points.

        Args:
            world_pts: World-space vertex coordinates, shape (N, 3), float32.

        Returns:
            ndc_pts: NDC coordinates, shape (N, 3), float32. The Z component
                carries the world-space Z for depth sorting by the rasterizer.
                Invalid points (above the water surface) are mapped to NDC (0, 0).
        """
        pixels, valid = self.model.project(world_pts)  # (N, 2), (N,)

        # Replace NaN from invalid pixels with 0 before converting, keeping in graph.
        safe_pixels = torch.where(
            valid.unsqueeze(-1),
            pixels,
            torch.zeros_like(pixels),
        )

        u = safe_pixels[:, 0]  # (N,) horizontal, in [0, W]
        v = safe_pixels[:, 1]  # (N,) vertical, in [0, H]

        # PyTorch3D NDC: x in [-1, 1] left-to-right, y in [-1, 1] bottom-to-top.
        ndc_x = (u / self.W) * 2.0 - 1.0
        ndc_y = -((v / self.H) * 2.0 - 1.0)  # flip Y: pixel +v down -> NDC +y up

        # Use world Z as the depth proxy for rasterizer depth sorting.
        z = world_pts[:, 2]

        return torch.stack([ndc_x, ndc_y, z], dim=-1)  # (N, 3)

    def get_image_size(self) -> tuple[int, int]:
        """Return camera (H, W) for compatibility."""
        return (self.H, self.W)


def _make_identity_ortho_camera(
    device: torch.device | str = "cpu",
) -> FoVOrthographicCameras:
    """Create a PyTorch3D FoVOrthographicCameras that acts as an identity transform.

    The camera maps NDC input coordinates directly to NDC output without any
    additional transformation. This allows pre-projecting mesh vertices with
    ``RefractiveCamera.project_to_ndc()`` and rendering with no further change.

    Args:
        device: Target device for camera tensors.

    Returns:
        An orthographic camera spanning exactly [-1, 1] in X and Y.
    """
    return FoVOrthographicCameras(
        znear=0.0,
        zfar=100.0,
        max_y=1.0,
        min_y=-1.0,
        max_x=1.0,
        min_x=-1.0,
        device=device,
    )


class RefractiveSilhouetteRenderer:
    """Differentiable multi-view silhouette renderer using refractive projection.

    Implements a vertex pre-projection approach for differentiable refractive
    silhouette rendering:

    1. Build the fish mesh in world space (via ``build_fish_mesh``).
    2. For each camera, project all mesh vertices from world space to NDC space
       using ``RefractiveCamera.project_to_ndc()``, which wraps the refractive
       projection model.
    3. Replace the mesh's vertex buffer with the NDC-projected vertices to create
       a pre-projected mesh.
    4. Render the pre-projected mesh with an identity ``FoVOrthographicCameras``
       (which applies no further transformation) using PyTorch3D's
       ``MeshRasterizer`` + ``SoftSilhouetteShader``.

    This approach keeps the refractive projection inside the autograd graph so
    that gradients flow from the silhouette loss back through the renderer to the
    FishState parameters.

    Args:
        image_size: (H, W) output render resolution in pixels. This controls
            the rasterization grid and alpha map shape. May differ from the
            camera's native sensor resolution.
        sigma: BlendParams sigma — controls silhouette edge softness.
            Larger values give softer edges and stronger boundary gradients.
            Default 1e-4 (PyTorch3D standard).
        gamma: BlendParams gamma — controls opacity blending. Default 1e-4.
        faces_per_pixel: Number of faces to accumulate per pixel during
            rasterization. Must be >> 1 for smooth silhouette gradients.
            Default 100.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        sigma: float = 1e-4,
        gamma: float = 1e-4,
        faces_per_pixel: int = 100,
    ) -> None:
        self.image_size = image_size
        self.blend_params = BlendParams(sigma=sigma, gamma=gamma)
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=float(np.log(1.0 / 1e-4 - 1.0) * sigma),
            faces_per_pixel=faces_per_pixel,
        )

    def render(
        self,
        meshes: Meshes,
        cameras: list[RefractiveProjectionModel],
        camera_ids: list[str],
        camera_image_sizes: list[tuple[int, int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Render differentiable silhouettes into each camera view.

        For each camera, reprojects all mesh vertices from world space to NDC space
        via the refractive model, builds a pre-projected mesh, and renders with an
        identity orthographic camera. The alpha channel of the rendered RGBA image is
        the differentiable silhouette.

        Args:
            meshes: PyTorch3D Meshes object with gradient-tracked vertex tensors
                (e.g., from ``build_fish_mesh``). Batch size 1 (single fish).
            cameras: One ``RefractiveProjectionModel`` per camera view.
            camera_ids: String identifiers for each camera (used as dict keys
                in the returned result).
            camera_image_sizes: Optional list of (H, W) native sensor dimensions
                for each camera, used for pixel-to-NDC normalisation. If None,
                each camera's image size is derived from its K matrix (cx, cy).

        Returns:
            Dict mapping camera_id to alpha map tensor of shape (H, W), float32,
            values in [0, 1]. Differentiable with respect to mesh vertex positions
            (and therefore FishState parameters via ``build_fish_mesh``).

        Raises:
            ValueError: If ``len(cameras) != len(camera_ids)``.
        """
        if len(cameras) != len(camera_ids):
            raise ValueError(
                f"cameras ({len(cameras)}) and camera_ids ({len(camera_ids)}) "
                "must have the same length"
            )

        # Get mesh device from vertex data.
        device = meshes.verts_list()[0].device

        # Build a shared identity orthographic camera for rasterisation.
        ortho_cam = _make_identity_ortho_camera(device=device)
        rasterizer = MeshRasterizer(
            cameras=ortho_cam,
            raster_settings=self.raster_settings,
        )
        shader = SoftSilhouetteShader(blend_params=self.blend_params)
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

        result: dict[str, torch.Tensor] = {}
        for idx, (model, cam_id) in enumerate(zip(cameras, camera_ids, strict=True)):
            cam_size = (
                camera_image_sizes[idx] if camera_image_sizes is not None else None
            )
            refractive_cam = RefractiveCamera(model, camera_image_size=cam_size)

            # Pre-project each mesh in the batch from world space to NDC.
            ndc_verts_list = []
            for verts in meshes.verts_list():
                # verts: (V, 3) world-space
                ndc_verts = refractive_cam.project_to_ndc(verts)  # (V, 3) NDC
                ndc_verts_list.append(ndc_verts)

            # Rebuild meshes with NDC-projected vertices, same face topology.
            ndc_meshes = Meshes(
                verts=ndc_verts_list,
                faces=meshes.faces_list(),
            )

            # Render: identity ortho camera maps NDC -> NDC.
            images = renderer(ndc_meshes)  # (N, H, W, 4) RGBA
            alpha = images[..., 3]  # (N, H, W)
            result[cam_id] = alpha.squeeze(0)  # (H, W)

        return result
