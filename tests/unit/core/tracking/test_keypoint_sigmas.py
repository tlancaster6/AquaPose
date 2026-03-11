"""Unit tests for keypoint_sigmas module."""

from __future__ import annotations

import numpy as np

from aquapose.core.tracking.keypoint_sigmas import (
    DEFAULT_SIGMAS,
    compute_keypoint_sigmas,
)


class TestDefaultSigmas:
    def test_default_sigmas_has_six_entries(self) -> None:
        assert len(DEFAULT_SIGMAS) == 6

    def test_default_sigmas_are_positive(self) -> None:
        assert np.all(DEFAULT_SIGMAS > 0)

    def test_endpoints_larger_than_midbody(self) -> None:
        # nose (0) and tail (5) should have larger sigmas than mid-body (2, 3)
        assert DEFAULT_SIGMAS[0] > DEFAULT_SIGMAS[2]
        assert DEFAULT_SIGMAS[5] > DEFAULT_SIGMAS[3]

    def test_default_sigmas_reasonable_range(self) -> None:
        # OKS sigmas are typically in [0.025, 0.15]
        assert np.all(DEFAULT_SIGMAS >= 0.01)
        assert np.all(DEFAULT_SIGMAS <= 0.25)


class TestComputeKeypointSigmas:
    def _make_annotations(
        self,
        n: int = 20,
        n_kpts: int = 6,
        rng: np.random.Generator | None = None,
    ) -> list[dict]:
        """Generate synthetic annotation records."""
        if rng is None:
            rng = np.random.default_rng(42)
        records = []
        for _ in range(n):
            kpts = rng.uniform(100.0, 900.0, size=(n_kpts, 2)).astype(np.float64)
            area = float(rng.uniform(1000.0, 50000.0))
            records.append({"keypoints": kpts, "obb_area": area})
        return records

    def test_returns_correct_shape(self) -> None:
        annotations = self._make_annotations(n=30)
        sigmas = compute_keypoint_sigmas(annotations)
        assert sigmas.shape == (6,)

    def test_returns_positive_floats(self) -> None:
        annotations = self._make_annotations(n=30)
        sigmas = compute_keypoint_sigmas(annotations)
        assert np.all(sigmas > 0)

    def test_returns_array(self) -> None:
        annotations = self._make_annotations(n=10)
        sigmas = compute_keypoint_sigmas(annotations)
        assert isinstance(sigmas, np.ndarray)

    def test_sigma_scales_with_variance(self) -> None:
        """Higher keypoint variance → larger sigma."""
        rng = np.random.default_rng(0)
        # Tight annotations: small variance
        n = 40
        tight = []
        noisy = []
        for _ in range(n):
            kpts_tight = rng.normal(500.0, 1.0, size=(6, 2))
            kpts_noisy = rng.normal(500.0, 50.0, size=(6, 2))
            area = 10000.0
            tight.append({"keypoints": kpts_tight, "obb_area": area})
            noisy.append({"keypoints": kpts_noisy, "obb_area": area})
        sigma_tight = compute_keypoint_sigmas(tight)
        sigma_noisy = compute_keypoint_sigmas(noisy)
        # Every keypoint sigma should be smaller for tight annotations
        assert np.all(sigma_tight < sigma_noisy)
