"""Unit tests for SampleStore CRUD operations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aquapose.training.store import SampleStore


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    """Return a temporary root directory for the store."""
    return tmp_path / "store"


@pytest.fixture
def store(store_root: Path) -> SampleStore:
    """Create a SampleStore with a temporary root directory."""
    db_path = store_root / "samples.db"
    return SampleStore(db_path)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a minimal 1x1 JPEG file for testing."""
    # Minimal valid JPEG: SOI + APP0 + minimal data + EOI
    # Using raw bytes for a tiny valid JPEG
    img = tmp_path / "test_image.jpg"
    # Minimal JPEG bytes (1x1 white pixel)
    img.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05"
        b"\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06"
        b'\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br'
        b"\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghij"
        b"stuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96"
        b"\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3"
        b"\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9"
        b"\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5"
        b"\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\xa8\xa0\x03\xa5\x14"
        b"\x00\x1f\xff\xd9"
    )
    return img


@pytest.fixture
def sample_image2(tmp_path: Path) -> Path:
    """Create a second distinct image file for dedup testing."""
    img = tmp_path / "test_image2.jpg"
    # Different content so hash differs
    img.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    )
    return img


@pytest.fixture
def sample_label(tmp_path: Path) -> Path:
    """Create a minimal YOLO label file for testing."""
    lbl = tmp_path / "test_label.txt"
    lbl.write_text("0 0.5 0.5 0.1 0.1\n")
    return lbl


@pytest.fixture
def sample_label2(tmp_path: Path) -> Path:
    """Create a second label file for testing."""
    lbl = tmp_path / "test_label2.txt"
    lbl.write_text("0 0.3 0.3 0.2 0.2\n")
    return lbl


class TestCreateStore:
    def test_create_store_initializes_schema(self, store: SampleStore) -> None:
        """Create SampleStore with tmp path, verify tables exist and version == 1."""
        conn = store._connect()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        assert "samples" in tables
        assert "datasets" in tables
        assert "models" in tables

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 1

    def test_context_manager(self, store_root: Path) -> None:
        """Use with statement, verify connection closed."""
        db_path = store_root / "samples.db"
        with SampleStore(db_path) as s:
            s._connect()
            assert s._conn is not None
        assert s._conn is None


class TestImport:
    def test_import_new_sample(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Import image+label, verify files copied with UUID names, verify DB row."""
        sample_id, action = store.import_sample(sample_image, sample_label, "manual")
        assert action == "imported"
        assert len(sample_id) == 36  # UUID format

        # Verify files exist in managed directories
        images_dir = store.images_dir
        labels_dir = store.labels_dir
        assert (images_dir / f"{sample_id}.jpg").exists()
        assert (labels_dir / f"{sample_id}.txt").exists()

        # Verify DB row
        row = store.get(sample_id)
        assert row is not None
        assert row["source"] == "manual"
        assert row["content_hash"] is not None
        provenance = json.loads(row["provenance"])
        assert len(provenance) == 1
        assert provenance[0]["action"] == "imported"

    def test_import_duplicate_higher_priority_upserts(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Import pseudo, then import manual with same image -> upserted."""
        sid1, act1 = store.import_sample(sample_image, sample_label, "pseudo")
        assert act1 == "imported"

        sid2, act2 = store.import_sample(sample_image, sample_label2, "manual")
        assert act2 == "upserted"
        assert sid2 == sid1  # Same sample ID

        row = store.get(sid1)
        assert row["source"] == "manual"

    def test_import_duplicate_lower_priority_skipped(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Import manual, then import pseudo with same image -> skipped."""
        sid1, act1 = store.import_sample(sample_image, sample_label, "manual")
        assert act1 == "imported"

        sid2, act2 = store.import_sample(sample_image, sample_label2, "pseudo")
        assert act2 == "skipped"
        assert sid2 == sid1

        row = store.get(sid1)
        assert row["source"] == "manual"  # Unchanged

    def test_import_duplicate_equal_priority_upserts(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Import manual, then import manual again -> upserted."""
        sid1, _ = store.import_sample(sample_image, sample_label, "manual")
        sid2, act2 = store.import_sample(sample_image, sample_label2, "manual")
        assert act2 == "upserted"
        assert sid2 == sid1

    def test_content_hash_deterministic(self, sample_image: Path) -> None:
        """Same file content produces same hash."""
        h1 = SampleStore.compute_content_hash(sample_image)
        h2 = SampleStore.compute_content_hash(sample_image)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


class TestUpsertCascade:
    def test_upsert_cascades_augmented_children(
        self,
        store: SampleStore,
        store_root: Path,
        sample_image: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Import sample, add augmented child, re-import with higher priority -> child deleted."""
        sid, _ = store.import_sample(sample_image, sample_label, "pseudo")

        # Create augmented child
        aug_img = store_root / "aug.jpg"
        aug_lbl = store_root / "aug.txt"
        aug_img.write_bytes(b"\xff\xd8\xff\xd9")
        aug_lbl.write_text("0 0.5 0.5 0.1 0.1\n")
        child_id = store.add_augmented(sid, aug_img, aug_lbl)

        # Verify child exists
        assert store.get(child_id) is not None
        assert (store.images_dir / f"{child_id}.jpg").exists()

        # Upsert parent with higher priority
        _, act = store.import_sample(sample_image, sample_label2, "manual")
        assert act == "upserted"

        # Child should be gone
        assert store.get(child_id) is None
        assert not (store.images_dir / f"{child_id}.jpg").exists()


class TestAugmented:
    def test_add_augmented_links_parent(
        self,
        store: SampleStore,
        store_root: Path,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Add augmented variant, verify parent_id set, 'augmented' tag present."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")

        aug_img = store_root / "aug.jpg"
        aug_lbl = store_root / "aug.txt"
        aug_img.write_bytes(b"\xff\xd8\xff\xd9")
        aug_lbl.write_text("0 0.5 0.5 0.1 0.1\n")

        child_id = store.add_augmented(sid, aug_img, aug_lbl)
        child = store.get(child_id)
        assert child is not None
        assert child["parent_id"] == sid
        tags = json.loads(child["tags"])
        assert "augmented" in tags


class TestQuery:
    def test_query_by_source(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_image2: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Import mixed sources, query by source."""
        store.import_sample(sample_image, sample_label, "manual")
        store.import_sample(sample_image2, sample_label2, "pseudo")

        results = store.query(source="manual")
        assert len(results) == 1
        assert results[0]["source"] == "manual"

    def test_query_excludes_excluded_by_default(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Exclude a sample, verify query omits it."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        store.exclude([sid])

        results = store.query()
        assert len(results) == 0

    def test_query_tags_include_and_semantics(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_image2: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Import samples with tags, query tags_include uses AND semantics."""
        sid1, _ = store.import_sample(sample_image, sample_label, "manual")
        sid2, _ = store.import_sample(sample_image2, sample_label2, "manual")

        # Manually set tags via DB
        conn = store._connect()
        conn.execute(
            "UPDATE samples SET tags = ? WHERE id = ?",
            (json.dumps(["a", "b"]), sid1),
        )
        conn.execute(
            "UPDATE samples SET tags = ? WHERE id = ?",
            (json.dumps(["a", "c"]), sid2),
        )
        conn.commit()

        results = store.query(tags_include=["a", "b"])
        assert len(results) == 1
        assert results[0]["id"] == sid1

    def test_query_min_confidence(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_image2: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Query with min_confidence returns only high-confidence samples."""
        store.import_sample(
            sample_image,
            sample_label,
            "pseudo",
            metadata={"confidence": 0.9},
        )
        store.import_sample(
            sample_image2,
            sample_label2,
            "pseudo",
            metadata={"confidence": 0.3},
        )

        results = store.query(min_confidence=0.5)
        assert len(results) == 1
        meta = json.loads(results[0]["metadata"])
        assert meta["confidence"] == 0.9

    def test_query_min_confidence_includes_no_confidence(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_image2: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Manual sample without confidence key passes min_confidence filter."""
        store.import_sample(sample_image, sample_label, "manual")
        store.import_sample(
            sample_image2,
            sample_label2,
            "pseudo",
            metadata={"confidence": 0.3},
        )

        results = store.query(min_confidence=0.5)
        assert len(results) == 1
        assert results[0]["source"] == "manual"


class TestExcludeInclude:
    def test_exclude_soft_deletes(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Exclude sample, verify 'excluded' tag added, files still exist."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        store.exclude([sid])

        row = store.get(sid)
        assert row is not None
        tags = json.loads(row["tags"])
        assert "excluded" in tags
        assert (store.images_dir / f"{sid}.jpg").exists()

    def test_exclude_cascades_to_children(
        self,
        store: SampleStore,
        store_root: Path,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Exclude parent, verify augmented children also excluded."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")

        aug_img = store_root / "aug.jpg"
        aug_lbl = store_root / "aug.txt"
        aug_img.write_bytes(b"\xff\xd8\xff\xd9")
        aug_lbl.write_text("0 0.5 0.5 0.1 0.1\n")
        child_id = store.add_augmented(sid, aug_img, aug_lbl)

        store.exclude([sid])

        child = store.get(child_id)
        tags = json.loads(child["tags"])
        assert "excluded" in tags

    def test_include_reverses_exclude(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Exclude then include, verify tag removed and sample in default query."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        store.exclude([sid])
        store.include([sid])

        row = store.get(sid)
        tags = json.loads(row["tags"])
        assert "excluded" not in tags

        results = store.query()
        assert len(results) == 1

    def test_include_cascades_to_children(
        self,
        store: SampleStore,
        store_root: Path,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Include parent, verify augmented children also included."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")

        aug_img = store_root / "aug.jpg"
        aug_lbl = store_root / "aug.txt"
        aug_img.write_bytes(b"\xff\xd8\xff\xd9")
        aug_lbl.write_text("0 0.5 0.5 0.1 0.1\n")
        child_id = store.add_augmented(sid, aug_img, aug_lbl)

        store.exclude([sid])
        store.include([sid])

        child = store.get(child_id)
        tags = json.loads(child["tags"])
        assert "excluded" not in tags


class TestRemove:
    def test_remove_hard_deletes(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Remove sample, verify DB row gone, files gone."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        count = store.remove([sid])
        assert count == 1

        assert store.get(sid) is None
        assert not (store.images_dir / f"{sid}.jpg").exists()
        assert not (store.labels_dir / f"{sid}.txt").exists()

    def test_remove_cascades_children(
        self,
        store: SampleStore,
        store_root: Path,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Remove parent, verify children also removed."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")

        aug_img = store_root / "aug.jpg"
        aug_lbl = store_root / "aug.txt"
        aug_img.write_bytes(b"\xff\xd8\xff\xd9")
        aug_lbl.write_text("0 0.5 0.5 0.1 0.1\n")
        child_id = store.add_augmented(sid, aug_img, aug_lbl)

        store.remove([sid])

        assert store.get(child_id) is None
        assert not (store.images_dir / f"{child_id}.jpg").exists()


class TestProvenance:
    def test_provenance_history_tracked(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Import, upsert, verify provenance array has both entries."""
        sid, _ = store.import_sample(sample_image, sample_label, "pseudo")
        store.import_sample(sample_image, sample_label2, "manual")

        row = store.get(sid)
        provenance = json.loads(row["provenance"])
        assert len(provenance) == 2
        assert provenance[0]["action"] == "imported"
        assert provenance[1]["action"] == "replaced"


class TestDeleteFilesTolerance:
    def test_delete_files_tolerates_missing(
        self,
        store: SampleStore,
    ) -> None:
        """Delete files for nonexistent sample, no error raised."""
        # Should not raise
        store._delete_files("nonexistent-uuid")


def _make_image(tmp_path: Path, name: str, salt: int) -> Path:
    """Create a minimal JPEG with unique content via salt byte."""
    img = tmp_path / f"{name}.jpg"
    # Minimal JPEG header with a unique byte to vary content hash
    img.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + bytes([salt & 0xFF])
        + b"\xff\xd9"
    )
    return img


def _make_label(tmp_path: Path, name: str) -> Path:
    """Create a minimal YOLO label file."""
    lbl = tmp_path / f"{name}.txt"
    lbl.write_text("0 0.5 0.5 0.1 0.1\n")
    return lbl


class TestSaveDataset:
    def test_save_dataset_stores_manifest(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Save dataset with name, query recipe, and sample IDs. Retrieve by name."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        recipe = {"source": "manual"}
        store.save_dataset("ds1", recipe, [sid], split_seed=42)

        ds = store.get_dataset("ds1")
        assert ds is not None
        assert ds["name"] == "ds1"
        assert ds["query_recipe"] == recipe
        assert ds["sample_ids"] == [sid]
        assert ds["split_seed"] == 42

    def test_save_dataset_overwrites_existing(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_image2: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Save dataset with same name twice. Second call replaces first."""
        sid1, _ = store.import_sample(sample_image, sample_label, "manual")
        sid2, _ = store.import_sample(sample_image2, sample_label2, "manual")

        store.save_dataset("ds1", {"v": 1}, [sid1], split_seed=1)
        store.save_dataset("ds1", {"v": 2}, [sid2], split_seed=2)

        ds = store.get_dataset("ds1")
        assert ds["sample_ids"] == [sid2]
        assert ds["query_recipe"] == {"v": 2}

    def test_get_dataset_returns_none_for_missing(
        self,
        store: SampleStore,
    ) -> None:
        """Query nonexistent dataset returns None."""
        store._connect()
        assert store.get_dataset("nonexistent") is None

    def test_list_datasets_returns_all(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Save two datasets, list returns both."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        store.save_dataset("ds_a", {}, [sid], split_seed=1)
        store.save_dataset("ds_b", {}, [sid], split_seed=2)

        datasets = store.list_datasets()
        names = {d["name"] for d in datasets}
        assert names == {"ds_a", "ds_b"}


class TestAssemble:
    def test_assemble_creates_symlinks(
        self,
        store: SampleStore,
        tmp_path: Path,
    ) -> None:
        """Call assemble method, verify symlinks in YOLO directory structure."""
        # Import 5 samples so we have enough for a split
        sids = []
        for i in range(5):
            img = _make_image(tmp_path, f"img{i}", salt=i)
            lbl = _make_label(tmp_path, f"lbl{i}")
            sid, _ = store.import_sample(img, lbl, "manual")
            sids.append(sid)

        ds_path = store.assemble("test_ds", {}, val_fraction=0.2, seed=42)

        # Directory structure exists
        assert (ds_path / "images" / "train").is_dir()
        assert (ds_path / "images" / "val").is_dir()
        assert (ds_path / "labels" / "train").is_dir()
        assert (ds_path / "labels" / "val").is_dir()
        assert (ds_path / "dataset.yaml").exists()

        # Symlinks created
        train_imgs = list((ds_path / "images" / "train").iterdir())
        val_imgs = list((ds_path / "images" / "val").iterdir())
        assert len(train_imgs) + len(val_imgs) == 5
        assert len(val_imgs) == 1  # 20% of 5 = 1

        # All are symlinks
        for f in train_imgs + val_imgs:
            assert f.is_symlink()

    def test_assemble_val_split_excludes_pseudo_by_default(
        self,
        store: SampleStore,
        tmp_path: Path,
    ) -> None:
        """Import manual + pseudo, assemble default, val set has only manual."""
        # 5 manual, 5 pseudo
        for i in range(5):
            img = _make_image(tmp_path, f"manual_{i}", salt=i)
            lbl = _make_label(tmp_path, f"manual_lbl_{i}")
            store.import_sample(img, lbl, "manual")
        for i in range(5):
            img = _make_image(tmp_path, f"pseudo_{i}", salt=100 + i)
            lbl = _make_label(tmp_path, f"pseudo_lbl_{i}")
            store.import_sample(img, lbl, "pseudo")

        ds_path = store.assemble("split_test", {}, val_fraction=0.2, seed=42)

        val_imgs = list((ds_path / "images" / "val").iterdir())
        # val should be 20% of 5 manual = 1 sample (pseudo excluded from val)
        assert len(val_imgs) == 1

        # Verify val sample is from manual source (check that symlink target is in store)
        train_imgs = list((ds_path / "images" / "train").iterdir())
        assert len(train_imgs) == 9  # 4 manual-train + 5 pseudo

    def test_assemble_val_split_include_pseudo_override(
        self,
        store: SampleStore,
        tmp_path: Path,
    ) -> None:
        """Assemble with pseudo_in_val=True, verify pseudo samples can appear in val."""
        for i in range(5):
            img = _make_image(tmp_path, f"manual_{i}", salt=i)
            lbl = _make_label(tmp_path, f"manual_lbl_{i}")
            store.import_sample(img, lbl, "manual")
        for i in range(5):
            img = _make_image(tmp_path, f"pseudo_{i}", salt=100 + i)
            lbl = _make_label(tmp_path, f"pseudo_lbl_{i}")
            store.import_sample(img, lbl, "pseudo")

        ds_path = store.assemble(
            "override_test", {}, val_fraction=0.2, seed=42, pseudo_in_val=True
        )

        val_imgs = list((ds_path / "images" / "val").iterdir())
        # 20% of 10 = 2
        assert len(val_imgs) == 2

    def test_assemble_creates_deterministic_split(
        self,
        store: SampleStore,
        tmp_path: Path,
    ) -> None:
        """Assemble twice with same seed, verify identical train/val split."""
        for i in range(10):
            img = _make_image(tmp_path, f"det_{i}", salt=i)
            lbl = _make_label(tmp_path, f"det_lbl_{i}")
            store.import_sample(img, lbl, "manual")

        ds1 = store.assemble("det1", {}, val_fraction=0.2, seed=99)
        val1 = sorted(f.name for f in (ds1 / "images" / "val").iterdir())

        ds2 = store.assemble("det2", {}, val_fraction=0.2, seed=99)
        val2 = sorted(f.name for f in (ds2 / "images" / "val").iterdir())

        assert val1 == val2

    def test_assemble_with_min_confidence(
        self,
        store: SampleStore,
        tmp_path: Path,
    ) -> None:
        """Assemble with min_confidence=0.5, verify low-confidence samples excluded."""
        # 2 high confidence, 2 low confidence
        for i in range(2):
            img = _make_image(tmp_path, f"hi_{i}", salt=i)
            lbl = _make_label(tmp_path, f"hi_lbl_{i}")
            store.import_sample(img, lbl, "pseudo", metadata={"confidence": 0.9})
        for i in range(2):
            img = _make_image(tmp_path, f"lo_{i}", salt=50 + i)
            lbl = _make_label(tmp_path, f"lo_lbl_{i}")
            store.import_sample(img, lbl, "pseudo", metadata={"confidence": 0.2})

        ds_path = store.assemble(
            "conf_test", {"min_confidence": 0.5}, val_fraction=0.0, seed=42
        )

        train_imgs = list((ds_path / "images" / "train").iterdir())
        assert len(train_imgs) == 2  # only high-confidence

    def test_symlinks_are_relative(
        self,
        store: SampleStore,
        tmp_path: Path,
    ) -> None:
        """Verify created symlinks use relative targets, not absolute."""
        img = _make_image(tmp_path, "rel_img", salt=42)
        lbl = _make_label(tmp_path, "rel_lbl")
        store.import_sample(img, lbl, "manual")

        ds_path = store.assemble("rel_test", {}, val_fraction=0.0, seed=42)

        train_imgs = list((ds_path / "images" / "train").iterdir())
        assert len(train_imgs) == 1
        target = os.readlink(str(train_imgs[0]))
        assert not os.path.isabs(target), (
            f"Symlink target should be relative, got: {target}"
        )


class TestRegisterModel:
    def test_register_model_inserts_row(
        self,
        store: SampleStore,
    ) -> None:
        """Register model with run_id, weights_path, model_type, metrics. Verify row in models table."""
        store._connect()
        store.register_model(
            run_id="run_20260306_100000",
            weights_path="/path/to/best.pt",
            model_type="obb",
            metrics={"mAP50": 0.8, "mAP50-95": 0.5},
        )
        conn = store._connect()
        row = conn.execute(
            "SELECT * FROM models WHERE run_id = ?", ("run_20260306_100000",)
        ).fetchone()
        assert row is not None
        assert row["weights_path"] == "/path/to/best.pt"
        assert row["model_type"] == "obb"
        import json

        metrics = json.loads(row["metrics"])
        assert metrics["mAP50"] == 0.8

    def test_register_model_with_dataset(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Register model linked to a dataset name. Verify FK."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        store.save_dataset("my_dataset", {"source": "manual"}, [sid], split_seed=42)

        store.register_model(
            run_id="run_20260306_100001",
            weights_path="/path/to/best.pt",
            model_type="obb",
            dataset_name="my_dataset",
        )
        model = store.get_model("run_20260306_100001")
        assert model is not None
        assert model["dataset_name"] == "my_dataset"

    def test_list_models_returns_all(
        self,
        store: SampleStore,
    ) -> None:
        """Register two models, list returns both ordered by created_at desc."""
        store._connect()
        store.register_model(
            run_id="run_a",
            weights_path="/a/best.pt",
            model_type="obb",
        )
        store.register_model(
            run_id="run_b",
            weights_path="/b/best.pt",
            model_type="pose",
        )
        models = store.list_models()
        assert len(models) == 2
        # Most recent first
        assert models[0]["run_id"] == "run_b"
        assert models[1]["run_id"] == "run_a"

    def test_get_model_by_run_id(
        self,
        store: SampleStore,
    ) -> None:
        """Register model, get by run_id returns correct record."""
        store._connect()
        store.register_model(
            run_id="run_x",
            weights_path="/x/best.pt",
            model_type="pose",
            metrics={"mAP50": 0.9},
            tag="test-tag",
        )
        model = store.get_model("run_x")
        assert model is not None
        assert model["run_id"] == "run_x"
        assert model["weights_path"] == "/x/best.pt"
        assert model["model_type"] == "pose"
        assert model["metrics"]["mAP50"] == 0.9
        assert model["tag"] == "test-tag"

    def test_get_model_returns_none_for_missing(
        self,
        store: SampleStore,
    ) -> None:
        """Get model for nonexistent run_id returns None."""
        store._connect()
        assert store.get_model("nonexistent") is None


class TestExcludeWithReason:
    def test_exclude_with_reason(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Exclude with reason adds both 'excluded' and reason tags."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        store.exclude([sid], reason="bad_crop")

        row = store.get(sid)
        tags = json.loads(row["tags"])
        assert "excluded" in tags
        assert "bad_crop" in tags

    def test_exclude_without_reason_backward_compat(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Exclude without reason adds only 'excluded' (backward compat)."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        store.exclude([sid])

        row = store.get(sid)
        tags = json.loads(row["tags"])
        assert tags == ["excluded"]

    def test_include_keeps_reason_tags(
        self,
        store: SampleStore,
        sample_image: Path,
        sample_label: Path,
    ) -> None:
        """Include removes 'excluded' but keeps reason tags as audit trail."""
        sid, _ = store.import_sample(sample_image, sample_label, "manual")
        store.exclude([sid], reason="bad_crop")
        store.include([sid])

        row = store.get(sid)
        tags = json.loads(row["tags"])
        assert "excluded" not in tags
        assert "bad_crop" in tags


class TestAssembleTaggedSplit:
    def test_assemble_tagged_split_uses_val_tag(
        self,
        store: SampleStore,
        tmp_path: Path,
    ) -> None:
        """Assemble with split_mode='tagged' puts val-tagged samples in val set."""
        sids = []
        for i in range(5):
            img = _make_image(tmp_path, f"tagged_{i}", salt=i)
            lbl = _make_label(tmp_path, f"tagged_lbl_{i}")
            sid, _ = store.import_sample(img, lbl, "manual")
            sids.append(sid)

        # Tag last 2 samples with "val"
        conn = store._connect()
        for sid in sids[3:]:
            conn.execute(
                "UPDATE samples SET tags = ? WHERE id = ?",
                (json.dumps(["val"]), sid),
            )
        conn.commit()

        ds_path = store.assemble(
            "tagged_test", {}, val_fraction=0.2, seed=42, split_mode="tagged"
        )

        val_imgs = list((ds_path / "images" / "val").iterdir())
        train_imgs = list((ds_path / "images" / "train").iterdir())
        assert len(val_imgs) == 2
        assert len(train_imgs) == 3

    def test_assemble_random_default_backward_compat(
        self,
        store: SampleStore,
        tmp_path: Path,
    ) -> None:
        """Assemble with default split_mode='random' preserves existing behavior."""
        for i in range(5):
            img = _make_image(tmp_path, f"compat_{i}", salt=i)
            lbl = _make_label(tmp_path, f"compat_lbl_{i}")
            store.import_sample(img, lbl, "manual")

        ds_path = store.assemble("compat_test", {}, val_fraction=0.2, seed=42)

        val_imgs = list((ds_path / "images" / "val").iterdir())
        train_imgs = list((ds_path / "images" / "train").iterdir())
        assert len(val_imgs) + len(train_imgs) == 5

    def test_assemble_val_candidates_tag(
        self,
        store: SampleStore,
        tmp_path: Path,
    ) -> None:
        """Assemble with val_candidates_tag filters val-eligible samples in random mode."""
        sids = []
        for i in range(10):
            img = _make_image(tmp_path, f"cand_{i}", salt=i)
            lbl = _make_label(tmp_path, f"cand_lbl_{i}")
            sid, _ = store.import_sample(img, lbl, "manual")
            sids.append(sid)

        # Tag 4 samples as "curated"
        conn = store._connect()
        for sid in sids[:4]:
            conn.execute(
                "UPDATE samples SET tags = ? WHERE id = ?",
                (json.dumps(["curated"]), sid),
            )
        conn.commit()

        ds_path = store.assemble(
            "cand_test",
            {},
            val_fraction=0.5,
            seed=42,
            val_candidates_tag="curated",
        )

        val_imgs = list((ds_path / "images" / "val").iterdir())
        train_imgs = list((ds_path / "images" / "train").iterdir())
        # val should be 50% of 4 curated = 2
        assert len(val_imgs) == 2
        # train should be remaining 2 curated + 6 non-curated = 8
        assert len(train_imgs) == 8


class TestSummary:
    def test_summary_returns_counts(
        self,
        store: SampleStore,
        store_root: Path,
        tmp_path: Path,
        sample_image: Path,
        sample_image2: Path,
        sample_label: Path,
        sample_label2: Path,
    ) -> None:
        """Import mixed samples, call summary, verify source breakdown and totals."""
        store.import_sample(sample_image, sample_label, "manual")
        store.import_sample(sample_image2, sample_label2, "pseudo")

        # Add an augmented child
        aug_img = store_root / "aug.jpg"
        aug_lbl = store_root / "aug.txt"
        aug_img.write_bytes(b"\xff\xd8\xff\xd9")
        aug_lbl.write_text("0 0.5 0.5 0.1 0.1\n")
        sid = store.query(source="manual")[0]["id"]
        store.add_augmented(sid, aug_img, aug_lbl)

        # Exclude one sample
        pseudo_sid = store.query(source="pseudo")[0]["id"]
        store.exclude([pseudo_sid])

        summary = store.summary()
        assert summary["total"] == 3  # 1 manual + 1 pseudo + 1 augmented
        assert (
            summary["by_source"]["manual"] == 2
        )  # parent + augmented child inherits source
        assert summary["by_source"]["pseudo"] == 1
        assert summary["augmented_count"] == 1
        assert summary["excluded_count"] == 1
        assert summary["dataset_count"] == 0
        assert summary["model_count"] == 0
