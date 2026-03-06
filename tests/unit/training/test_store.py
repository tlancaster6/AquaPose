"""Unit tests for SampleStore CRUD operations."""

from __future__ import annotations

import json
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
