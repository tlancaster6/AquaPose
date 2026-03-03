"""Unit tests for confidence-weighted triangulation.

Covers:
- weighted_triangulate_rays: uniform weights == triangulate_rays output
- weighted_triangulate_rays: high-weight cameras bias the result
"""

from __future__ import annotations

import torch

from aquapose.calibration.projection import triangulate_rays
from aquapose.core.reconstruction.utils import weighted_triangulate_rays

# ---------------------------------------------------------------------------
# Tests: weighted_triangulate_rays
# ---------------------------------------------------------------------------


class TestWeightedTriangulateRays:
    """Tests for the weighted_triangulate_rays helper."""

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights should produce identical output to triangulate_rays."""
        # Build two non-parallel rays
        origins = torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=torch.float32)
        # Rays pointing roughly toward (0, 0, 1)
        d0 = torch.tensor([-0.3, 0.0, 1.0], dtype=torch.float32)
        d1 = torch.tensor([0.3, 0.0, 1.0], dtype=torch.float32)
        d0 = d0 / d0.norm()
        d1 = d1 / d1.norm()
        directions = torch.stack([d0, d1])

        weights = torch.ones(2, dtype=torch.float32)
        pt_weighted = weighted_triangulate_rays(origins, directions, weights)
        pt_unweighted = triangulate_rays(origins, directions)

        assert torch.allclose(pt_weighted, pt_unweighted, atol=1e-5), (
            f"Uniform-weighted result {pt_weighted} differs from "
            f"unweighted result {pt_unweighted}"
        )

    def test_high_weight_biases_result(self):
        """High weight on one camera should produce a different result than low weight.

        We set up three cameras: two close together (almost parallel, conflicting)
        and one orthogonal (good geometry). We compare:
        - w=[1,1,1]: uniform result is pulled toward the conflicting pair
        - w=[0.01,0.01,1]: result is dominated by the orthogonal camera's ray

        The two results must differ, confirming weight scaling is effective.
        """
        # Three camera origins
        o0 = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32)
        o1 = torch.tensor([0.05, 0.5, 0.0], dtype=torch.float32)  # near-duplicate of o0
        o2 = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)  # orthogonal

        # True point at (0.1, 0.1, 0.8)
        true_pt = torch.tensor([0.1, 0.1, 0.8], dtype=torch.float32)

        # Cameras 0 and 1 point toward a different (wrong) target (0.3, 0.3, 0.8)
        wrong_pt = torch.tensor([0.3, 0.3, 0.8], dtype=torch.float32)

        d0 = wrong_pt - o0
        d0 = d0 / d0.norm()
        d1 = wrong_pt - o1
        d1 = d1 / d1.norm()

        # Camera 2 points toward true_pt
        d2 = true_pt - o2
        d2 = d2 / d2.norm()

        origins = torch.stack([o0, o1, o2])
        directions = torch.stack([d0, d1, d2])

        # Uniform weights -- conflicting pair has more votes
        w_uniform = torch.ones(3, dtype=torch.float32)
        pt_uniform = weighted_triangulate_rays(origins, directions, w_uniform)

        # Near-zero weight on conflicting cameras -- orthogonal camera dominates
        w_biased = torch.tensor([0.01, 0.01, 1.0], dtype=torch.float32)
        pt_biased = weighted_triangulate_rays(origins, directions, w_biased)

        # Results must differ
        diff = float((pt_uniform - pt_biased).norm().item())
        assert diff > 0.01, (
            f"Biased and uniform results should differ (diff={diff:.4f})"
        )

        # Biased result should be closer to true_pt (camera 2 observes it)
        dist_biased = float((pt_biased - true_pt).norm().item())
        dist_uniform = float((pt_uniform - true_pt).norm().item())
        assert dist_biased < dist_uniform, (
            f"High-weight camera 2 observes true_pt; biased result (dist={dist_biased:.4f}) "
            f"should be closer to true_pt than uniform (dist={dist_uniform:.4f})"
        )

    def test_three_cameras_uniform_weights(self):
        """Three cameras with uniform weights should match unweighted."""
        origins = torch.tensor(
            [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=torch.float32,
        )
        true_pt = torch.tensor([0.0, 0.0, 0.8], dtype=torch.float32)
        directions = []
        for i in range(3):
            d = true_pt - origins[i]
            directions.append(d / d.norm())
        dirs = torch.stack(directions)

        w = torch.ones(3, dtype=torch.float32)
        pt_weighted = weighted_triangulate_rays(origins, dirs, w)
        pt_unweighted = triangulate_rays(origins, dirs)

        assert torch.allclose(pt_weighted, pt_unweighted, atol=1e-5)
