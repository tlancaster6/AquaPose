"""Assemble OBB dataset excluding original manual annotations (batch-manual).

Includes corrected pseudo-labels, hard cases, and wall-aug images whose source
was NOT a batch-manual sample. Traces wall-aug lineage via content hash matching
through the wall_augmented/ intermediate directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
STORE_DB = Path("~/aquapose/projects/YH/training_data/obb/store.db").expanduser()
STORE_ROOT = STORE_DB.parent
WALLAUG_IMG_DIR = STORE_ROOT / "wall_augmented" / "images"
DATASETS_DIR = STORE_ROOT / "datasets"
DATASET_NAME = "no_manual"


def sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_conn() -> sqlite3.Connection:
    """Open a connection to the store database."""
    conn = sqlite3.connect(str(STORE_DB))
    conn.row_factory = sqlite3.Row
    return conn


def query_samples(
    conn: sqlite3.Connection,
    tags_include: list[str] | None = None,
    tags_exclude: list[str] | None = None,
) -> list[dict]:
    """Query samples with tag filters (AND for include, OR-exclude)."""
    clauses: list[str] = [
        "NOT EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = 'excluded')"
    ]
    params: list[str] = []

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

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT * FROM samples WHERE {where} ORDER BY created_at", params
    ).fetchall()
    return [dict(r) for r in rows]


def make_symlink(src: Path, link: Path) -> None:
    """Create a relative symlink from link -> src."""
    rel = os.path.relpath(src, link.parent)
    link.symlink_to(rel)


def create_dataset_dir(name: str) -> Path:
    """Create a clean dataset directory with train/val structure."""
    ds_dir = DATASETS_DIR / name
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    for split in ("train", "val"):
        (ds_dir / "images" / split).mkdir(parents=True)
        (ds_dir / "labels" / split).mkdir(parents=True)
    return ds_dir


def write_dataset_yaml(ds_dir: Path) -> None:
    """Write YOLO OBB dataset.yaml."""
    cfg = {
        "names": {0: "fish"},
        "nc": 1,
        "path": str(ds_dir),
        "train": "images/train",
        "val": "images/val",
    }
    (ds_dir / "dataset.yaml").write_text(yaml.dump(cfg, default_flow_style=False))


def link_store_sample(sample: dict, ds_dir: Path, split: str) -> None:
    """Symlink a store sample's image and label into the dataset."""
    img_src = STORE_ROOT / sample["image_path"]
    lbl_src = STORE_ROOT / sample["label_path"]
    make_symlink(img_src, ds_dir / "images" / split / img_src.name)
    make_symlink(lbl_src, ds_dir / "labels" / split / lbl_src.name)


def register_dataset(
    conn: sqlite3.Connection,
    name: str,
    train_samples: list[dict],
    val_samples: list[dict],
    description: str,
) -> None:
    """Register a dataset manifest in the store DB."""
    all_ids = [s["id"] for s in train_samples + val_samples]
    now = (
        __import__("datetime")
        .datetime.now(tz=__import__("datetime").timezone.utc)
        .isoformat()
    )
    conn.execute(
        """INSERT OR REPLACE INTO datasets (name, query_recipe, sample_ids, split_seed, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            name,
            json.dumps({"description": description}),
            json.dumps(all_ids),
            42,
            now,
        ),
    )
    conn.commit()


def find_manual_wallaug_ids(conn: sqlite3.Connection) -> set[str]:
    """Identify wall-aug store sample IDs whose source was a batch-manual image.

    Traces lineage through the wall_augmented/ intermediate directory:
    1. The intermediate dir preserves original source filenames (UUIDs)
    2. Content hash of intermediate file matches the wall-aug store sample
    3. Filename stem matches a batch-manual source sample
    """
    # Build set of batch-manual filename stems
    manual_samples = query_samples(conn, tags_include=["batch-manual"])
    manual_stems = set()
    for s in manual_samples:
        manual_stems.add(os.path.splitext(os.path.basename(s["image_path"]))[0])

    # Build content_hash -> wall-aug store sample ID
    wallaug_samples = query_samples(conn, tags_include=["wall-aug"])
    hash_to_wallaug_id: dict[str, str] = {}
    for s in wallaug_samples:
        hash_to_wallaug_id[s["content_hash"]] = s["id"]

    # Match intermediate files to store samples and check source batch
    manual_wallaug_ids: set[str] = set()
    for img_path in sorted(WALLAUG_IMG_DIR.iterdir()):
        stem = img_path.stem
        if stem not in manual_stems:
            continue
        h = sha256(img_path)
        wallaug_id = hash_to_wallaug_id.get(h)
        if wallaug_id is not None:
            manual_wallaug_ids.add(wallaug_id)

    return manual_wallaug_ids


def main() -> None:
    """Assemble the no-manual OBB dataset."""
    print("OBB No-Manual Dataset Assembly")
    print("=" * 60)

    conn = get_conn()

    # ── Identify wall-aug samples to exclude ──
    print("\nTracing wall-aug lineage...")
    manual_wallaug_ids = find_manual_wallaug_ids(conn)
    print(f"  Wall-aug from batch-manual: {len(manual_wallaug_ids)} (excluding)")

    # ── Query samples by batch ──
    pseudo_train = query_samples(
        conn, tags_include=["batch-pseudo"], tags_exclude=["val"]
    )
    hard_train = query_samples(
        conn, tags_include=["batch-hard-cases"], tags_exclude=["val"]
    )
    wallaug_all = query_samples(conn, tags_include=["wall-aug"])
    wallaug_train = [s for s in wallaug_all if s["id"] not in manual_wallaug_ids]

    pseudo_val = query_samples(conn, tags_include=["batch-pseudo", "val"])
    hard_val = query_samples(conn, tags_include=["batch-hard-cases", "val"])

    train_samples = pseudo_train + hard_train + wallaug_train
    val_samples = pseudo_val + hard_val

    print(
        f"\nTrain: {len(pseudo_train)} pseudo + {len(hard_train)} hard-cases "
        f"+ {len(wallaug_train)} wall-aug = {len(train_samples)}"
    )
    print(
        f"Val: {len(pseudo_val)} pseudo + {len(hard_val)} hard-cases = {len(val_samples)}"
    )

    # ── Assemble dataset ──
    print(f"\nAssembling {DATASET_NAME}...")
    ds_dir = create_dataset_dir(DATASET_NAME)
    for s in train_samples:
        link_store_sample(s, ds_dir, "train")
    for s in val_samples:
        link_store_sample(s, ds_dir, "val")
    write_dataset_yaml(ds_dir)
    register_dataset(
        conn,
        DATASET_NAME,
        train_samples,
        val_samples,
        "Corrected pseudo + hard cases + wall-aug (no original manual annotations)",
    )

    # ── Verify ──
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    actual_train = len(list((ds_dir / "images" / "train").iterdir()))
    actual_val = len(list((ds_dir / "images" / "val").iterdir()))
    print(f"  Train images on disk: {actual_train} (expected {len(train_samples)})")
    print(f"  Val images on disk:   {actual_val} (expected {len(val_samples)})")

    # Check no batch-manual samples leaked
    manual_samples = query_samples(conn, tags_include=["batch-manual"])
    manual_img_names = {os.path.basename(s["image_path"]) for s in manual_samples}
    leaked = []
    for split in ("train", "val"):
        for link in (ds_dir / "images" / split).iterdir():
            target = os.path.basename(os.path.realpath(str(link)))
            if target in manual_img_names:
                leaked.append(f"{split}/{link.name}")
    if leaked:
        print(f"  LEAK DETECTED: {len(leaked)} batch-manual images found!")
        for p in leaked[:5]:
            print(f"    {p}")
    else:
        print("  No batch-manual leaks detected.")

    ok = (
        actual_train == len(train_samples)
        and actual_val == len(val_samples)
        and not leaked
    )
    print(f"\n{'PASSED' if ok else 'FAILED'}")

    conn.close()


if __name__ == "__main__":
    main()
