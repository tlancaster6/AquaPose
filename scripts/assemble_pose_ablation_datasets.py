"""Assemble 4 pose ablation datasets to measure impact of each training data type."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
STORE_DB = Path("~/aquapose/projects/YH/training_data/pose/store.db").expanduser()
STORE_ROOT = STORE_DB.parent
ROUND1_DIR = Path(
    "~/aquapose/projects/YH/training_data/round1_selected/pose"
).expanduser()
DATASETS_DIR = STORE_ROOT / "datasets"

# ── Batch date ranges ─────────────────────────────────────────────────────
BATCH_DATES = {
    "batch-manual": "2026-03-07",
    "batch-pseudo": "2026-03-09",
}

# ── Pose-specific dataset.yaml fields ─────────────────────────────────────
KPT_NAMES = ["nose", "head", "spine1", "spine2", "spine3", "tail"]
KPT_SHAPE = [6, 3]
FLIP_IDX = list(range(6))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(STORE_DB))
    conn.row_factory = sqlite3.Row
    return conn


# ── Step 1: Tag store samples by batch ────────────────────────────────────
def tag_batches() -> None:
    """Add batch-identifying tags to store samples based on import date."""
    conn = get_conn()
    for tag, date in BATCH_DATES.items():
        rows = conn.execute(
            "SELECT id, tags FROM samples WHERE substr(created_at,1,10) = ?",
            (date,),
        ).fetchall()
        updated = 0
        for row in rows:
            tags = json.loads(row["tags"])
            if tag not in tags:
                tags.append(tag)
                conn.execute(
                    "UPDATE samples SET tags = ? WHERE id = ?",
                    (json.dumps(tags), row["id"]),
                )
                updated += 1
        conn.commit()
        print(f"  Tagged {updated}/{len(rows)} samples with '{tag}'")
    conn.close()


# ── Step 2: Build content-hash mapping ────────────────────────────────────
def build_round1_hash_map() -> dict[str, Path]:
    """Map content_hash → round1_selected label path for all round1 images."""
    hash_to_label: dict[str, Path] = {}
    for split in ("train", "val"):
        img_dir = ROUND1_DIR / "images" / split
        lbl_dir = ROUND1_DIR / "labels" / split
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                h = sha256(img_path)
                hash_to_label[h] = lbl_path
    return hash_to_label


# ── Step 3: Query helpers ─────────────────────────────────────────────────
def query_samples(
    conn: sqlite3.Connection,
    tags_include: list[str] | None = None,
    tags_exclude: list[str] | None = None,
    require_parent: bool | None = None,
) -> list[dict]:
    """Query samples with tag filters (AND for include, OR-exclude).

    Args:
        require_parent: If True, only augmented (parent_id IS NOT NULL).
            If False, only originals (parent_id IS NULL). If None, no filter.
    """
    clauses: list[str] = []
    params: list[str] = []

    # Always exclude 'excluded' samples
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

    if require_parent is True:
        clauses.append("parent_id IS NOT NULL")
    elif require_parent is False:
        clauses.append("parent_id IS NULL")

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT * FROM samples WHERE {where} ORDER BY created_at", params
    ).fetchall()
    return [dict(r) for r in rows]


# ── Step 4: Dataset assembly ──────────────────────────────────────────────
def make_symlink(src: Path, link: Path) -> None:
    """Create a relative symlink from link → src."""
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
    """Write YOLO pose dataset.yaml."""
    cfg = {
        "names": {0: "fish"},
        "nc": 1,
        "path": str(ds_dir),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": KPT_SHAPE,
        "flip_idx": FLIP_IDX,
        "kpt_names": KPT_NAMES,
    }
    (ds_dir / "dataset.yaml").write_text(yaml.dump(cfg, default_flow_style=False))


def link_store_sample(sample: dict, ds_dir: Path, split: str) -> None:
    """Symlink a store sample's image and label into the dataset."""
    img_src = STORE_ROOT / sample["image_path"]
    lbl_src = STORE_ROOT / sample["label_path"]
    make_symlink(img_src, ds_dir / "images" / split / img_src.name)
    make_symlink(lbl_src, ds_dir / "labels" / split / lbl_src.name)


def link_raw_pseudo_sample(
    sample: dict, raw_label: Path, ds_dir: Path, split: str
) -> None:
    """Symlink store image but raw round1_selected label."""
    img_src = STORE_ROOT / sample["image_path"]
    make_symlink(img_src, ds_dir / "images" / split / img_src.name)
    make_symlink(raw_label, ds_dir / "labels" / split / Path(sample["label_path"]).name)


def register_dataset(
    conn: sqlite3.Connection,
    name: str,
    train_samples: list[dict],
    val_samples: list[dict],
    description: str,
) -> None:
    """Register a dataset manifest in the store DB so model registration FK succeeds."""
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


def assemble_all() -> dict[str, tuple[int, int]]:
    """Assemble all 4 pose ablation datasets. Returns {name: (train_count, val_count)}."""
    conn = get_conn()

    # ── Shared val set: all 128 val-tagged samples ──
    val_samples = query_samples(conn, tags_include=["val"])
    print(f"\nShared val set: {len(val_samples)} samples")

    # ── Per-batch train samples (excluding val, originals only) ──
    manual_train = query_samples(
        conn, tags_include=["batch-manual"], tags_exclude=["val"], require_parent=False
    )
    pseudo_train = query_samples(
        conn, tags_include=["batch-pseudo"], tags_exclude=["val"], require_parent=False
    )

    # ── Augmented samples (children) per batch ──
    manual_aug = query_samples(
        conn, tags_include=["batch-manual"], tags_exclude=["val"], require_parent=True
    )
    pseudo_aug = query_samples(
        conn, tags_include=["batch-pseudo"], tags_exclude=["val"], require_parent=True
    )

    print(f"Manual train originals: {len(manual_train)}")
    print(f"Pseudo train originals: {len(pseudo_train)}")
    print(f"Manual augmented: {len(manual_aug)}")
    print(f"Pseudo augmented: {len(pseudo_aug)}")

    # ── Build content-hash → raw label mapping ──
    print("\nBuilding content-hash mapping for round1_selected...")
    hash_to_raw_label = build_round1_hash_map()
    print(f"  Found {len(hash_to_raw_label)} round1_selected images")

    # Map pseudo_train samples to raw labels
    pseudo_raw_map: dict[str, Path] = {}  # sample_id → raw_label_path
    missing_raw = 0
    for s in pseudo_train:
        raw_lbl = hash_to_raw_label.get(s["content_hash"])
        if raw_lbl is not None:
            pseudo_raw_map[s["id"]] = raw_lbl
        else:
            missing_raw += 1
    print(
        f"  Mapped {len(pseudo_raw_map)}/{len(pseudo_train)} pseudo samples to raw labels"
    )
    if missing_raw > 0:
        print(f"  WARNING: {missing_raw} pseudo samples have no raw label match!")

    sizes: dict[str, tuple[int, int]] = {}

    # ── Model A: Manual originals only ──
    print("\nAssembling pose_ablation_a_manual...")
    ds_a = create_dataset_dir("pose_ablation_a_manual")
    for s in manual_train:
        link_store_sample(s, ds_a, "train")
    for s in val_samples:
        link_store_sample(s, ds_a, "val")
    write_dataset_yaml(ds_a)
    register_dataset(
        conn,
        "pose_ablation_a_manual",
        manual_train,
        val_samples,
        "Manual originals only",
    )
    sizes["A: manual"] = (len(manual_train), len(val_samples))

    # ── Model B: Manual + raw pseudo-labels ──
    print("Assembling pose_ablation_b_raw_pseudo...")
    ds_b = create_dataset_dir("pose_ablation_b_raw_pseudo")
    for s in manual_train:
        link_store_sample(s, ds_b, "train")
    for s in pseudo_train:
        raw_lbl = pseudo_raw_map.get(s["id"])
        if raw_lbl is not None:
            link_raw_pseudo_sample(s, raw_lbl, ds_b, "train")
        else:
            # Fallback to store label if no raw match
            link_store_sample(s, ds_b, "train")
    for s in val_samples:
        link_store_sample(s, ds_b, "val")
    write_dataset_yaml(ds_b)
    register_dataset(
        conn,
        "pose_ablation_b_raw_pseudo",
        manual_train + pseudo_train,
        val_samples,
        "Manual + raw pseudo-labels",
    )
    sizes["B: raw pseudo"] = (len(manual_train) + len(pseudo_train), len(val_samples))

    # ── Model C: Manual + corrected pseudo-labels ──
    print("Assembling pose_ablation_c_corrected_pseudo...")
    ds_c = create_dataset_dir("pose_ablation_c_corrected_pseudo")
    for s in manual_train:
        link_store_sample(s, ds_c, "train")
    for s in pseudo_train:
        link_store_sample(s, ds_c, "train")
    for s in val_samples:
        link_store_sample(s, ds_c, "val")
    write_dataset_yaml(ds_c)
    register_dataset(
        conn,
        "pose_ablation_c_corrected_pseudo",
        manual_train + pseudo_train,
        val_samples,
        "Manual + corrected pseudo-labels",
    )
    sizes["C: corrected pseudo"] = (
        len(manual_train) + len(pseudo_train),
        len(val_samples),
    )

    # ── Model D: C + elastic augmentation ──
    print("Assembling pose_ablation_d_augmented...")
    ds_d = create_dataset_dir("pose_ablation_d_augmented")
    all_aug = manual_aug + pseudo_aug
    for s in manual_train + pseudo_train + all_aug:
        link_store_sample(s, ds_d, "train")
    for s in val_samples:
        link_store_sample(s, ds_d, "val")
    write_dataset_yaml(ds_d)
    register_dataset(
        conn,
        "pose_ablation_d_augmented",
        manual_train + pseudo_train + all_aug,
        val_samples,
        "Manual + corrected pseudo + elastic augmentation",
    )
    sizes["D: + augmented"] = (
        len(manual_train) + len(pseudo_train) + len(all_aug),
        len(val_samples),
    )

    conn.close()
    return sizes


# ── Step 5: Verification ─────────────────────────────────────────────────
def verify_datasets(sizes: dict[str, tuple[int, int]]) -> bool:
    """Verify dataset sizes and val-set consistency."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    ok = True

    # Check sizes
    expected = {
        "A: manual": (258, 128),
        "B: raw pseudo": (514, 128),
        "C: corrected pseudo": (514, 128),
        "D: + augmented": (2554, 128),
    }

    print("\nDataset sizes:")
    print(f"  {'Model':<25} {'Train':>6} {'Val':>5}  {'Expected':>15}  {'OK?':>5}")
    print(f"  {'-' * 25} {'-' * 6} {'-' * 5}  {'-' * 15}  {'-' * 5}")
    for name, (train, val) in sizes.items():
        exp = expected.get(name, (0, 0))
        match = train == exp[0] and val == exp[1]
        if not match:
            ok = False
        status = "PASS" if match else "FAIL"
        print(
            f"  {name:<25} {train:>6} {val:>5}  "
            f"({exp[0]:>4}/{exp[1]:>3})       {status}"
        )

    # Check val set consistency (same symlink targets across all 4)
    print("\nVal set consistency:")
    datasets = [
        "pose_ablation_a_manual",
        "pose_ablation_b_raw_pseudo",
        "pose_ablation_c_corrected_pseudo",
        "pose_ablation_d_augmented",
    ]
    ref_targets: set[str] | None = None
    for ds_name in datasets:
        val_dir = DATASETS_DIR / ds_name / "images" / "val"
        targets = set()
        for link in sorted(val_dir.iterdir()):
            targets.add(os.readlink(str(link)))
        if ref_targets is None:
            ref_targets = targets
            print(f"  {ds_name}: {len(targets)} val images (reference)")
        elif targets == ref_targets:
            print(f"  {ds_name}: MATCH")
        else:
            print(f"  {ds_name}: MISMATCH!")
            ok = False

    # Spot-check: Model B raw labels vs Model C corrected labels differ
    print("\nRaw vs corrected label spot-check (Model B vs C):")
    b_labels = DATASETS_DIR / "pose_ablation_b_raw_pseudo" / "labels" / "train"
    c_labels = DATASETS_DIR / "pose_ablation_c_corrected_pseudo" / "labels" / "train"
    b_files = {f.name for f in b_labels.iterdir()}
    c_files = {f.name for f in c_labels.iterdir()}
    common = sorted(b_files & c_files)
    diffs_found = 0
    checked = 0
    for fname in common[:10]:
        b_target = os.readlink(str(b_labels / fname))
        c_target = os.readlink(str(c_labels / fname))
        if b_target != c_target:
            diffs_found += 1
            if checked < 3:
                print(f"  {fname}:")
                print(f"    B → {b_target}")
                print(f"    C → {c_target}")
        checked += 1
    print(f"  {diffs_found}/{checked} checked labels point to different targets")
    if diffs_found == 0:
        print("  WARNING: No label differences found — raw/corrected may be identical?")

    # Check Model D includes augmented samples
    print("\nModel D augmented check:")
    d_train = DATASETS_DIR / "pose_ablation_d_augmented" / "images" / "train"
    d_count = len(list(d_train.iterdir()))
    print(f"  Train images: {d_count} (expect 2554)")

    return ok


def main() -> None:
    print("Pose Ablation Dataset Assembly")
    print("=" * 60)

    print("\nStep 1: Tagging store samples by batch...")
    tag_batches()

    print("\nStep 2-4: Assembling datasets...")
    sizes = assemble_all()

    all_ok = verify_datasets(sizes)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<8} {'Dataset':<45} {'Train':>6} {'Val':>5}")
    print(f"{'-' * 8} {'-' * 45} {'-' * 6} {'-' * 5}")
    ds_names = [
        "pose_ablation_a_manual",
        "pose_ablation_b_raw_pseudo",
        "pose_ablation_c_corrected_pseudo",
        "pose_ablation_d_augmented",
    ]
    for (name, (train, val)), ds_name in zip(sizes.items(), ds_names, strict=False):
        print(f"{name[0]:<8} {ds_name:<45} {train:>6} {val:>5}")

    if all_ok:
        print("\nAll verifications PASSED.")
    else:
        print("\nSome verifications FAILED — check output above.")


if __name__ == "__main__":
    main()
