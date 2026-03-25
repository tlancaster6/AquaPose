"""Unit tests for ReID training module (projection head, group split, AUC, cache)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class TestProjectionHead:
    """Tests for ProjectionHead forward pass shape and normalization."""

    def test_projection_head_output_shape(self) -> None:
        """ProjectionHead(768, 256, 128) on (16, 768) input produces (16, 128)."""
        from aquapose.training.reid_training import ProjectionHead

        head = ProjectionHead(in_dim=768, hidden_dim=256, out_dim=128)
        head.eval()
        x = torch.randn(16, 768)
        out = head(x)
        assert out.shape == (16, 128)

    def test_projection_head_l2_normalized(self) -> None:
        """Output rows have unit L2 norm (within 1e-5)."""
        from aquapose.training.reid_training import ProjectionHead

        head = ProjectionHead(in_dim=768, hidden_dim=256, out_dim=128)
        head.eval()
        x = torch.randn(16, 768)
        out = head(x)
        norms = torch.norm(out, p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones(16), atol=1e-5, rtol=0.0)

    def test_projection_head_single_sample_eval(self) -> None:
        """Forward pass works with batch_size=1 in eval mode (BatchNorm eval)."""
        from aquapose.training.reid_training import ProjectionHead

        head = ProjectionHead(in_dim=768, hidden_dim=256, out_dim=128)
        head.eval()
        x = torch.randn(1, 768)
        out = head(x)
        assert out.shape == (1, 128)
        norm = torch.norm(out, p=2, dim=1)
        torch.testing.assert_close(norm, torch.ones(1), atol=1e-5, rtol=0.0)


class TestGroupTemporalSplit:
    """Tests for temporal group-based train/val splitting."""

    def test_group_temporal_split(self) -> None:
        """80% split puts groups 0-7 in train, 8-9 in val with no overlap."""
        from aquapose.training.reid_training import split_by_group

        # 40 samples, 10 groups (4 samples per group), group_ids in order
        group_ids = np.array([i // 4 for i in range(40)])
        cache = {"group_ids": group_ids, "features": np.zeros((40, 768))}
        train_idx, val_idx = split_by_group(cache, val_fraction=0.2)

        train_groups = set(group_ids[train_idx])
        val_groups = set(group_ids[val_idx])

        # No overlap
        assert train_groups & val_groups == set()
        # All samples accounted for
        assert len(train_idx) + len(val_idx) == 40
        # Last 20% of groups (8, 9) in val
        assert val_groups == {8, 9}
        assert train_groups == {0, 1, 2, 3, 4, 5, 6, 7}

    def test_group_split_by_window_start(self) -> None:
        """Groups are sorted numerically (matching temporal order from miner)."""
        from aquapose.training.reid_training import split_by_group

        # 20 samples, 5 groups (4 per group)
        group_ids = np.array([i // 4 for i in range(20)])
        cache = {"group_ids": group_ids, "features": np.zeros((20, 768))}
        train_idx, val_idx = split_by_group(cache, val_fraction=0.2)

        train_groups = set(group_ids[train_idx])
        val_groups = set(group_ids[val_idx])

        # Last 20% = last 1 group (group 4)
        assert val_groups == {4}
        assert train_groups == {0, 1, 2, 3}


class TestFemaleAUC:
    """Tests for female-female AUC computation."""

    def test_compute_female_auc_perfect(self) -> None:
        """Perfectly separable embeddings for 6 female fish give AUC == 1.0."""
        from aquapose.training.reid_training import compute_female_auc

        # 6 female fish, 4 samples each = 24 samples, 128-dim
        female_ids = [0, 1, 2, 3, 4, 8]
        fish_ids = np.array([fid for fid in female_ids for _ in range(4)])
        # Perfectly separable: each fish gets a distinct one-hot direction
        embeddings = np.zeros((24, 128), dtype=np.float32)
        for i, _fid in enumerate(female_ids):
            embeddings[i * 4 : (i + 1) * 4, i * 10 : (i + 1) * 10] = 1.0
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        auc = compute_female_auc(embeddings, fish_ids)
        assert auc == 1.0

    def test_compute_female_auc_random(self) -> None:
        """Random embeddings give AUC near 0.5 (within 0.2 tolerance)."""
        from aquapose.training.reid_training import compute_female_auc

        rng = np.random.RandomState(42)
        female_ids = [0, 1, 2, 3, 4, 8]
        fish_ids = np.array([fid for fid in female_ids for _ in range(10)])
        embeddings = rng.randn(60, 128).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        auc = compute_female_auc(embeddings, fish_ids)
        assert abs(auc - 0.5) < 0.2

    def test_compute_per_pair_auc(self) -> None:
        """Per-pair breakdown returns dict keyed by (fish_i, fish_j) tuples."""
        from aquapose.training.reid_training import compute_per_pair_auc

        female_ids = {0, 1, 2, 3, 4, 8}
        rng = np.random.RandomState(42)
        fish_id_list = [0, 1, 2, 3, 4, 8]
        fish_ids = np.array([fid for fid in fish_id_list for _ in range(10)])
        embeddings = rng.randn(60, 128).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        result = compute_per_pair_auc(embeddings, fish_ids, female_ids)
        assert isinstance(result, dict)
        # 6 choose 2 = 15 pairs
        assert len(result) == 15
        # Keys are (int, int) tuples
        for key in result:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert key[0] < key[1]


class TestBuildFeatureCache:
    """Tests for feature cache construction from reid_crops directory."""

    def test_build_feature_cache_structure(self, tmp_path: Path) -> None:
        """build_feature_cache returns dict with correct keys and shape."""
        from aquapose.training.reid_training import (
            ReidTrainingConfig,
            build_feature_cache,
        )

        # Create a fake reid_crops directory with 2 groups
        crops_dir = tmp_path / "reid_crops"
        for gid in range(2):
            group_dir = crops_dir / f"group_{gid:03d}"
            crops = []
            for fid in [0, 1]:
                fish_dir = group_dir / f"fish_{fid}"
                fish_dir.mkdir(parents=True)
                for frame in range(2):
                    fname = f"f{frame:06d}_cam0.jpg"
                    # Write a tiny valid image (3x3 BGR)
                    img = np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)
                    import cv2

                    cv2.imwrite(str(fish_dir / fname), img)
                    crops.append(
                        {
                            "fish_id": fid,
                            "frame": frame,
                            "camera": "cam0",
                            "n_cameras": 1,
                            "mean_residual": 0.01,
                            "detection_confidence": 0.9,
                            "filename": f"fish_{fid}/{fname}",
                        }
                    )
            manifest = {
                "group_id": gid,
                "window_start": gid * 100,
                "window_end": gid * 100 + 99,
                "fish_ids": [0, 1],
                "crops": crops,
            }
            group_dir.mkdir(parents=True, exist_ok=True)
            (group_dir / "manifest.json").write_text(json.dumps(manifest))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = ReidTrainingConfig(
            reid_crops_dir=crops_dir,
            output_dir=output_dir,
            device="cpu",
        )

        # Mock FishEmbedder to avoid GPU dependency
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.embed_batch.return_value = np.random.randn(
            4, 768
        ).astype(np.float32)

        with patch(
            "aquapose.training.reid_training.FishEmbedder",
            return_value=mock_embedder_instance,
        ):
            cache = build_feature_cache(crops_dir, config)

        assert "features" in cache
        assert "labels" in cache
        assert "group_ids" in cache
        assert "paths" in cache
        assert cache["features"].shape == (8, 768)  # 2 groups * 2 fish * 2 frames
        assert cache["features"].dtype == np.float32
        assert len(cache["labels"]) == 8
        assert len(cache["group_ids"]) == 8
        assert len(cache["paths"]) == 8


@pytest.mark.slow
class TestTrainingLoopSmoke:
    """Smoke test for the full training loop on synthetic data."""

    def test_training_loop_smoke(self) -> None:
        """Full train_reid_head on tiny synthetic data runs without error."""
        from aquapose.training.reid_training import ReidTrainingConfig, train_reid_head

        rng = np.random.RandomState(42)
        n_groups = 3
        n_fish = 5
        n_samples_per = 4
        n_total = n_groups * n_fish * n_samples_per

        features = rng.randn(n_total, 768).astype(np.float32)
        labels = np.array(
            [
                fid
                for _ in range(n_groups)
                for fid in range(n_fish)
                for _ in range(n_samples_per)
            ]
        )
        group_ids = np.array(
            [gid for gid in range(n_groups) for _ in range(n_fish * n_samples_per)]
        )
        paths = [
            f"group_{g:03d}/fish_{f}/frame_{s}.jpg"
            for g in range(n_groups)
            for f in range(n_fish)
            for s in range(n_samples_per)
        ]

        cache = {
            "features": features,
            "labels": labels,
            "group_ids": group_ids,
            "paths": paths,
        }

        config = ReidTrainingConfig(
            reid_crops_dir=Path("/tmp/fake"),
            output_dir=Path("/tmp/fake_output"),
            epochs=5,
            patience=10,
            num_classes=n_fish,
            device="cpu",
            samples_per_class=n_samples_per,
        )

        result = train_reid_head(cache, config)
        assert isinstance(result, dict)
        assert "best_auc" in result
        assert isinstance(result["best_auc"], float)
