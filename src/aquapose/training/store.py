"""SampleStore: SQLite-backed training data management with dedup and provenance."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .store_schema import SCHEMA_SQL, SCHEMA_VERSION, SOURCE_PRIORITY

logger = logging.getLogger(__name__)


class SampleStore:
    """SQLite-backed store for training data samples.

    Manages training images and labels with content-hash deduplication,
    source-priority upsert, provenance tracking, augmentation lineage,
    and tag-based querying.

    Args:
        db_path: Path to the SQLite database file. Parent directory is
            used as the store root containing ``images/`` and ``labels/``
            subdirectories.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.root = self.db_path.parent
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        """Create or return the database connection.

        Initialises WAL mode, foreign keys, busy timeout, and runs
        schema DDL on first connect. Validates schema version.
        """
        if self._conn is not None:
            return self._conn

        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")

        existing_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if existing_version > SCHEMA_VERSION:
            conn.close()
            msg = (
                f"Database schema version {existing_version} is newer than "
                f"expected {SCHEMA_VERSION}. Upgrade this code."
            )
            raise RuntimeError(msg)

        conn.executescript(SCHEMA_SQL)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

        self._conn = conn
        return conn

    def close(self) -> None:
        """Close the database connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SampleStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def compute_content_hash(path: Path) -> str:
        """Compute SHA-256 hex digest of file contents.

        Args:
            path: Path to the file to hash.

        Returns:
            64-character lowercase hex string.
        """
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def import_sample(
        self,
        image_path: Path,
        label_path: Path,
        source: str,
        metadata: dict | None = None,
        import_batch_id: str | None = None,
    ) -> tuple[str, str]:
        """Import a training sample into the store.

        Computes content hash of the image for deduplication. If a sample
        with the same hash already exists, applies source-priority logic:
        higher or equal priority upserts, lower priority is skipped.

        Args:
            image_path: Path to source image file.
            label_path: Path to source label file.
            source: One of ``"manual"``, ``"corrected"``, ``"pseudo"``.
            metadata: Optional metadata dict (e.g. confidence, run_id).
            import_batch_id: Optional batch identifier for grouped imports.

        Returns:
            Tuple of ``(sample_id, action)`` where action is one of
            ``"imported"``, ``"upserted"``, or ``"skipped"``.
        """
        conn = self._connect()
        content_hash = self.compute_content_hash(image_path)
        meta = metadata or {}
        now = datetime.now(tz=UTC).isoformat()

        # Check for existing sample with same hash
        existing = conn.execute(
            "SELECT id, source, provenance, image_path FROM samples WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()

        if existing is not None:
            incoming_priority = SOURCE_PRIORITY.get(source, 0)
            existing_priority = SOURCE_PRIORITY.get(existing["source"], 0)

            if incoming_priority >= existing_priority:
                # Upsert: cascade delete children, replace label, update row
                self._cascade_delete_children(existing["id"])

                # Replace label file
                existing_label_rel = (
                    existing["image_path"]
                    .replace("images/", "labels/")
                    .replace(Path(existing["image_path"]).suffix, ".txt")
                )
                new_label_dest = self.root / existing_label_rel
                shutil.copy2(str(label_path), str(new_label_dest))

                new_provenance = self._append_provenance(
                    existing["provenance"], "replaced", source, import_batch_id
                )
                conn.execute(
                    """UPDATE samples
                       SET source = ?, label_path = ?, metadata = ?,
                           provenance = ?, updated_at = ?, import_batch_id = ?
                       WHERE id = ?""",
                    (
                        source,
                        existing_label_rel,
                        json.dumps(meta),
                        new_provenance,
                        now,
                        import_batch_id,
                        existing["id"],
                    ),
                )
                conn.commit()
                return (existing["id"], "upserted")

            return (existing["id"], "skipped")

        # New sample
        sample_id = str(uuid.uuid4())
        img_ext = image_path.suffix
        img_rel = f"images/{sample_id}{img_ext}"
        lbl_rel = f"labels/{sample_id}.txt"

        shutil.copy2(str(image_path), str(self.root / img_rel))
        shutil.copy2(str(label_path), str(self.root / lbl_rel))

        provenance = json.dumps(
            [{"action": "imported", "source": source, "timestamp": now}]
        )

        conn.execute(
            """INSERT INTO samples
               (id, content_hash, source, image_path, label_path, parent_id,
                import_batch_id, tags, provenance, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, NULL, ?, '[]', ?, ?, ?, ?)""",
            (
                sample_id,
                content_hash,
                source,
                img_rel,
                lbl_rel,
                import_batch_id,
                provenance,
                json.dumps(meta),
                now,
                now,
            ),
        )
        conn.commit()
        return (sample_id, "imported")

    def _cascade_delete_children(self, parent_id: str) -> int:
        """Delete all augmented children of a sample (files and DB rows).

        Args:
            parent_id: ID of the parent sample.

        Returns:
            Number of children deleted.
        """
        conn = self._connect()
        children = conn.execute(
            "SELECT id, image_path, label_path FROM samples WHERE parent_id = ?",
            (parent_id,),
        ).fetchall()

        for child in children:
            self._delete_files(child["id"])

        conn.execute("DELETE FROM samples WHERE parent_id = ?", (parent_id,))
        conn.commit()
        return len(children)

    def add_augmented(
        self,
        parent_id: str,
        image_path: Path,
        label_path: Path,
        metadata: dict | None = None,
    ) -> str:
        """Import an augmented variant linked to a parent sample.

        Source is inherited from the parent. The ``"augmented"`` tag is
        automatically added.

        Args:
            parent_id: ID of the parent sample.
            image_path: Path to the augmented image.
            label_path: Path to the augmented label.
            metadata: Optional metadata dict.

        Returns:
            The new sample's UUID.
        """
        conn = self._connect()
        parent = conn.execute(
            "SELECT source FROM samples WHERE id = ?", (parent_id,)
        ).fetchone()
        if parent is None:
            msg = f"Parent sample {parent_id} not found"
            raise ValueError(msg)

        sample_id = str(uuid.uuid4())
        img_ext = image_path.suffix
        img_rel = f"images/{sample_id}{img_ext}"
        lbl_rel = f"labels/{sample_id}.txt"
        meta = metadata or {}
        now = datetime.now(tz=UTC).isoformat()

        shutil.copy2(str(image_path), str(self.root / img_rel))
        shutil.copy2(str(label_path), str(self.root / lbl_rel))

        content_hash = self.compute_content_hash(image_path)
        provenance = json.dumps(
            [{"action": "imported", "source": parent["source"], "timestamp": now}]
        )

        conn.execute(
            """INSERT INTO samples
               (id, content_hash, source, image_path, label_path, parent_id,
                import_batch_id, tags, provenance, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)""",
            (
                sample_id,
                content_hash,
                parent["source"],
                img_rel,
                lbl_rel,
                parent_id,
                json.dumps(["augmented"]),
                provenance,
                json.dumps(meta),
                now,
                now,
            ),
        )
        conn.commit()
        return sample_id

    def query(
        self,
        source: str | None = None,
        tags_include: list[str] | None = None,
        tags_exclude: list[str] | None = None,
        exclude_excluded: bool = True,
        min_confidence: float | None = None,
    ) -> list[dict]:
        """Query samples with optional filters.

        Args:
            source: Filter by source type.
            tags_include: AND semantics -- sample must have ALL listed tags.
            tags_exclude: OR semantics -- exclude sample if it has ANY listed tag.
            exclude_excluded: If True, filter out samples with ``"excluded"`` tag.
            min_confidence: Minimum confidence threshold. Samples without a
                ``confidence`` key in metadata are included (assumed high quality).

        Returns:
            List of sample dicts matching all filters.
        """
        conn = self._connect()
        clauses: list[str] = []
        params: list[object] = []

        if source is not None:
            clauses.append("source = ?")
            params.append(source)

        if exclude_excluded:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = 'excluded')"
            )

        if tags_include:
            for tag in tags_include:
                clauses.append(
                    "EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = ?)"
                )
                params.append(tag)

        if tags_exclude:
            for tag in tags_exclude:
                clauses.append(
                    "NOT EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = ?)"
                )
                params.append(tag)

        if min_confidence is not None:
            clauses.append(
                "(json_extract(metadata, '$.confidence') IS NULL "
                "OR json_extract(metadata, '$.confidence') >= ?)"
            )
            params.append(min_confidence)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM samples WHERE {where} ORDER BY created_at"

        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get(self, sample_id: str) -> dict | None:
        """Get a single sample by ID.

        Args:
            sample_id: The sample UUID.

        Returns:
            Sample dict or None if not found.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM samples WHERE id = ?", (sample_id,)
        ).fetchone()
        return dict(row) if row else None

    def count(self, source: str | None = None) -> int:
        """Count samples matching an optional source filter.

        Args:
            source: Optional source type filter.

        Returns:
            Number of matching samples.
        """
        conn = self._connect()
        if source is not None:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM samples WHERE source = ?", (source,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) as cnt FROM samples").fetchone()
        return row["cnt"]

    def exclude(self, sample_ids: list[str]) -> int:
        """Soft-delete samples by adding the ``"excluded"`` tag.

        Also cascades to augmented children.

        Args:
            sample_ids: List of sample UUIDs to exclude.

        Returns:
            Total number of samples modified (parents + children).
        """
        conn = self._connect()
        count = 0
        all_ids = list(sample_ids)

        # Collect children
        for sid in sample_ids:
            children = conn.execute(
                "SELECT id FROM samples WHERE parent_id = ?", (sid,)
            ).fetchall()
            all_ids.extend(row["id"] for row in children)

        for sid in all_ids:
            row = conn.execute(
                "SELECT tags FROM samples WHERE id = ?", (sid,)
            ).fetchone()
            if row is None:
                continue
            tags = json.loads(row["tags"])
            if "excluded" not in tags:
                tags.append("excluded")
                now = datetime.now(tz=UTC).isoformat()
                conn.execute(
                    "UPDATE samples SET tags = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(tags), now, sid),
                )
                count += 1

        conn.commit()
        return count

    def include(self, sample_ids: list[str]) -> int:
        """Reverse exclusion by removing the ``"excluded"`` tag.

        Also cascades to augmented children.

        Args:
            sample_ids: List of sample UUIDs to include.

        Returns:
            Total number of samples modified (parents + children).
        """
        conn = self._connect()
        count = 0
        all_ids = list(sample_ids)

        # Collect children
        for sid in sample_ids:
            children = conn.execute(
                "SELECT id FROM samples WHERE parent_id = ?", (sid,)
            ).fetchall()
            all_ids.extend(row["id"] for row in children)

        for sid in all_ids:
            row = conn.execute(
                "SELECT tags FROM samples WHERE id = ?", (sid,)
            ).fetchone()
            if row is None:
                continue
            tags = json.loads(row["tags"])
            if "excluded" in tags:
                tags.remove("excluded")
                now = datetime.now(tz=UTC).isoformat()
                conn.execute(
                    "UPDATE samples SET tags = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(tags), now, sid),
                )
                count += 1

        conn.commit()
        return count

    def remove(self, sample_ids: list[str]) -> int:
        """Hard-delete samples: remove DB rows and files.

        Children are cascade-deleted via foreign key. Their files are
        also removed.

        Args:
            sample_ids: List of sample UUIDs to remove.

        Returns:
            Number of samples removed.
        """
        conn = self._connect()
        count = 0
        for sid in sample_ids:
            # Delete children files first (FK cascade handles DB rows)
            children = conn.execute(
                "SELECT id FROM samples WHERE parent_id = ?", (sid,)
            ).fetchall()
            for child in children:
                self._delete_files(child["id"])

            row = conn.execute("SELECT id FROM samples WHERE id = ?", (sid,)).fetchone()
            if row is None:
                continue

            self._delete_files(sid)
            conn.execute("DELETE FROM samples WHERE id = ?", (sid,))
            count += 1

        conn.commit()
        return count

    def _delete_files(self, sample_id: str, image_ext: str = ".jpg") -> None:
        """Remove image and label files from disk.

        Tolerates missing files (logs warning, does not raise).

        Args:
            sample_id: The sample UUID.
            image_ext: Image file extension (default ``.jpg``).
        """
        img = self.images_dir / f"{sample_id}{image_ext}"
        lbl = self.labels_dir / f"{sample_id}.txt"
        for path in (img, lbl):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete %s", path)

    def save_dataset(
        self,
        name: str,
        query_recipe: dict,
        sample_ids: list[str],
        split_seed: int,
    ) -> None:
        """Persist a dataset manifest in the database.

        Args:
            name: Dataset name (unique key).
            query_recipe: Dict describing the query used to select samples.
            sample_ids: List of sample UUIDs included in the dataset.
            split_seed: Random seed used for the train/val split.
        """
        conn = self._connect()
        now = datetime.now(tz=UTC).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO datasets (name, query_recipe, sample_ids, split_seed, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                name,
                json.dumps(query_recipe),
                json.dumps(sample_ids),
                split_seed,
                now,
            ),
        )
        conn.commit()

    def get_dataset(self, name: str) -> dict | None:
        """Retrieve a dataset manifest by name.

        Args:
            name: Dataset name.

        Returns:
            Dict with parsed JSON fields, or None if not found.
        """
        conn = self._connect()
        row = conn.execute("SELECT * FROM datasets WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["query_recipe"] = json.loads(result["query_recipe"])
        result["sample_ids"] = json.loads(result["sample_ids"])
        return result

    def list_datasets(self) -> list[dict]:
        """List all persisted dataset manifests.

        Returns:
            List of dataset dicts with parsed JSON fields.
        """
        conn = self._connect()
        rows = conn.execute("SELECT * FROM datasets ORDER BY created_at").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["query_recipe"] = json.loads(d["query_recipe"])
            d["sample_ids"] = json.loads(d["sample_ids"])
            results.append(d)
        return results

    def assemble(
        self,
        name: str,
        query: dict,
        val_fraction: float = 0.2,
        seed: int = 42,
        pseudo_in_val: bool = False,
        split_mode: str = "random",
        val_candidates_tag: str | None = None,
    ) -> Path:
        """Assemble a training dataset with symlinks in YOLO directory structure.

        Queries the store, splits into train/val, creates relative symlinks,
        writes ``dataset.yaml``, and persists the manifest.

        Args:
            name: Dataset name (becomes subdirectory under ``datasets/``).
            query: Keyword arguments passed to :meth:`query`.
            val_fraction: Fraction of val-eligible samples for validation.
            seed: Random seed for deterministic splitting.
            pseudo_in_val: If False (default), pseudo-label samples are
                excluded from the validation set.
            split_mode: Split strategy. ``"random"`` (default) uses seeded
                random shuffle. ``"tagged"`` puts samples with ``"val"``
                tag into the validation set directly.
            val_candidates_tag: When set in random mode, only samples with
                this tag are eligible for the validation set. Others go
                straight to train.

        Returns:
            Path to the assembled dataset directory.
        """
        samples = self.query(**query)

        if split_mode == "tagged":
            # Use "val" tag for splitting
            val_samples = [s for s in samples if "val" in json.loads(s["tags"])]
            train_samples = [s for s in samples if "val" not in json.loads(s["tags"])]
        elif val_candidates_tag is not None:
            # In random mode with val_candidates_tag: only tagged samples
            # are val-eligible
            val_eligible = [
                s for s in samples if val_candidates_tag in json.loads(s["tags"])
            ]
            train_only: list[dict] = [
                s for s in samples if val_candidates_tag not in json.loads(s["tags"])
            ]

            rng = random.Random(seed)
            rng.shuffle(val_eligible)
            n_val = int(len(val_eligible) * val_fraction)
            val_samples = val_eligible[:n_val]
            train_samples = val_eligible[n_val:] + train_only
        else:
            # Original random split logic
            if pseudo_in_val:
                val_eligible = list(samples)
                train_only = []
            else:
                val_eligible = [
                    s for s in samples if s["source"] in ("manual", "corrected")
                ]
                train_only = [
                    s for s in samples if s["source"] not in ("manual", "corrected")
                ]

            # Seeded shuffle for deterministic split
            rng = random.Random(seed)
            rng.shuffle(val_eligible)

            n_val = int(len(val_eligible) * val_fraction)
            val_samples = val_eligible[:n_val]
            train_samples = val_eligible[n_val:] + train_only

        # Create dataset directory (clean if exists)
        ds_dir = self.root / "datasets" / name
        if ds_dir.exists():
            shutil.rmtree(ds_dir)

        for split in ("train", "val"):
            (ds_dir / "images" / split).mkdir(parents=True)
            (ds_dir / "labels" / split).mkdir(parents=True)

        # Create relative symlinks
        for split_name, split_samples in [
            ("train", train_samples),
            ("val", val_samples),
        ]:
            for sample in split_samples:
                img_src = self.root / sample["image_path"]
                lbl_src = self.root / sample["label_path"]

                img_link = ds_dir / "images" / split_name / img_src.name
                lbl_link = ds_dir / "labels" / split_name / lbl_src.name

                img_rel = os.path.relpath(img_src, img_link.parent)
                lbl_rel = os.path.relpath(lbl_src, lbl_link.parent)

                img_link.symlink_to(img_rel)
                lbl_link.symlink_to(lbl_rel)

        # Write dataset.yaml
        dataset_yaml = {
            "path": str(ds_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "fish"},
            "nc": 1,
        }
        (ds_dir / "dataset.yaml").write_text(
            yaml.dump(dataset_yaml, default_flow_style=False)
        )

        # Persist manifest
        all_ids = [s["id"] for s in train_samples + val_samples]
        self.save_dataset(name, query, all_ids, split_seed=seed)

        return ds_dir

    def summary(self) -> dict:
        """Return summary statistics for the store.

        Returns:
            Dict with keys: ``total``, ``by_source``, ``augmented_count``,
            ``excluded_count``, ``dataset_count``, ``model_count``.
        """
        conn = self._connect()

        total = conn.execute("SELECT COUNT(*) as cnt FROM samples").fetchone()["cnt"]

        # By source breakdown
        rows = conn.execute(
            "SELECT source, COUNT(*) as cnt FROM samples GROUP BY source"
        ).fetchall()
        by_source = {row["source"]: row["cnt"] for row in rows}

        # Augmented count
        augmented = conn.execute(
            "SELECT COUNT(*) as cnt FROM samples WHERE parent_id IS NOT NULL"
        ).fetchone()["cnt"]

        # Excluded count
        excluded = conn.execute(
            "SELECT COUNT(*) as cnt FROM samples WHERE "
            "EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = 'excluded')"
        ).fetchone()["cnt"]

        # Dataset count
        ds_count = conn.execute("SELECT COUNT(*) as cnt FROM datasets").fetchone()[
            "cnt"
        ]

        # Model count
        model_count = conn.execute("SELECT COUNT(*) as cnt FROM models").fetchone()[
            "cnt"
        ]

        return {
            "total": total,
            "by_source": by_source,
            "augmented_count": augmented,
            "excluded_count": excluded,
            "dataset_count": ds_count,
            "model_count": model_count,
        }

    def register_model(
        self,
        run_id: str,
        weights_path: str,
        model_type: str,
        metrics: dict | None = None,
        dataset_name: str | None = None,
        tag: str | None = None,
    ) -> None:
        """Register a trained model in the store.

        Args:
            run_id: Unique identifier for the training run.
            weights_path: Path to the best model weights file.
            model_type: Model type string (e.g. ``"obb"``, ``"pose"``).
            metrics: Optional dict of training metrics (e.g. mAP50, mAP50-95).
            dataset_name: Optional dataset name linking to the datasets table.
            tag: Optional human-readable tag for this model.
        """
        conn = self._connect()
        now = datetime.now(tz=UTC).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO models
               (run_id, dataset_name, weights_path, model_type, metrics, tag, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                dataset_name,
                weights_path,
                model_type,
                json.dumps(metrics or {}),
                tag,
                now,
            ),
        )
        conn.commit()

    def list_models(self) -> list[dict]:
        """List all registered models.

        Returns:
            List of model dicts with parsed JSON metrics, ordered by
            created_at descending (most recent first).
        """
        conn = self._connect()
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["metrics"] = json.loads(d["metrics"])
            results.append(d)
        return results

    def get_model(self, run_id: str) -> dict | None:
        """Retrieve a model record by run_id.

        Args:
            run_id: The training run identifier.

        Returns:
            Model dict with parsed JSON metrics, or None if not found.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM models WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"])
        return d

    def _append_provenance(
        self,
        existing_json: str,
        action: str,
        source: str,
        batch_id: str | None = None,
    ) -> str:
        """Append an entry to the provenance JSON array.

        Args:
            existing_json: Current provenance JSON string.
            action: Action name (e.g. ``"imported"``, ``"replaced"``).
            source: Source type.
            batch_id: Optional batch identifier.

        Returns:
            Updated provenance JSON string.
        """
        entries = json.loads(existing_json)
        entry: dict[str, str] = {
            "action": action,
            "source": source,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
        if batch_id is not None:
            entry["batch_id"] = batch_id
        entries.append(entry)
        return json.dumps(entries)
