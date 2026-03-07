# Phase 71: Data Store Bootstrap - Research

**Researched:** 2026-03-07
**Domain:** Training data management, COCO-to-YOLO conversion, temporal splitting, YOLO training configuration
**Confidence:** HIGH

## Summary

Phase 71 exercises the existing data management infrastructure end-to-end while filling two specific gaps: temporal train/val splitting and reason-tagged exclusions. The codebase already has nearly everything needed -- SampleStore with full CRUD, CLI commands (import, convert, assemble, status, list, exclude, include, remove), COCO conversion, training wrappers, and model registration. The required changes are well-scoped modifications to existing code rather than new subsystems.

The main implementation work falls into four categories: (1) adding `--split-mode temporal` to `data convert` with frame-index-based splitting, (2) adding val-tagging during import and `--split-mode tagged` to `data assemble`, (3) adding `--reason` to `data exclude` and reason-breakdown to `data status`, and (4) updating training CLI defaults (mosaic, imgsz, rect) and running the actual baseline training.

**Primary recommendation:** Implement the temporal split and exclusion reason features first (pure code changes with tests), then run the end-to-end convert-import-assemble-train workflow as a validation step.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Temporal split strategy: contiguous block holdout, last 20% of unique frame indices as val
- Split at frame-index level (all cameras from a given frame stay in same split)
- Frame index parsed from filename: last section between final underscore and `.png` extension
- `--split-mode temporal|random` flag on `data convert` (default: `random` for backward compat)
- `data import` tags samples from `val/` subdirectory with `"val"` tag
- `store.assemble()` gets `--split-mode` option: `random` (current default) or `tagged` (val = samples with "val" tag)
- `--val-candidates` tag filter for random mode
- Exclusion: `data exclude --reason TAG` adds both `"excluded"` and reason string as separate tags
- Free-text reason (no predefined enum)
- `SampleStore.exclude()` accepts optional `reason: str | None` parameter -- no schema changes
- `data status` shows breakdown by reason tags
- OBB: 100 epochs, yolo26n-obb, mosaic=0.3, patience=100, imgsz=640
- Pose: 100 epochs, yolo26n-pose, mosaic=0.1, imgsz=128, rect=True, patience=100
- Elastic augmentation on pose import (4 variants, 5-15 degree range)
- No elastic augmentation for OBB
- Both models tagged as "baseline" in the store

### Claude's Discretion
- Exact implementation of frame index parsing from filenames
- How to surface rect training mode in the YOLO training wrappers
- Error handling for edge cases in temporal split (single frame index, etc.)

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BOOT-01 | `aquapose data convert` CLI subcommand converts COCO-JSON annotations to YOLO-OBB and YOLO-pose formats | Existing `convert_cmd` and `coco_convert.py` handle this fully. Need to add `--split-mode temporal` flag. |
| BOOT-02 | Manual annotations imported into data store as `source=manual` with correct provenance | Existing `import_cmd` handles this. Need to add val-tagging for samples from `val/` subdirectory. |
| BOOT-03 | Baseline OBB and pose models trained from store-assembled datasets and registered with model lineage | Existing `train obb` and `train pose` CLI commands with auto-registration. Need default updates (mosaic, imgsz, rect). |
| BOOT-04 | Temporal split convention -- train/val split respects temporal holdout | New feature: frame index parsing + temporal split logic in `coco_convert.py`, plus `--split-mode tagged` in `assemble`. |
| BOOT-05 | `aquapose data exclude --reason TAG` adds reason tag alongside "excluded" tag; `data status` shows breakdown | Modify `SampleStore.exclude()` and `exclude_cmd`, enhance `status_cmd` output. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLite (via stdlib) | 3.x | SampleStore database | Already in use, no schema changes needed |
| Click | installed | CLI framework | Already used for all CLI commands |
| ultralytics | installed | YOLO training | Already wrapped in `yolo_obb.py` and `yolo_pose.py` |
| PyYAML | installed | Config and dataset.yaml | Already used throughout |
| OpenCV | installed | Image I/O for augmentation | Already used in import and convert |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | installed | Curvature computation | Already used in import pipeline |

### Alternatives Considered
None -- all decisions are locked. This phase uses only existing dependencies.

## Architecture Patterns

### Existing Project Structure (relevant files)
```
src/aquapose/training/
├── store.py              # SampleStore class (modify: exclude() reason param)
├── store_schema.py       # SQL schema (no changes needed)
├── data_cli.py           # CLI commands (modify: convert, import, exclude, status, assemble)
├── coco_convert.py       # COCO-to-YOLO conversion (modify: temporal split)
├── cli.py                # Training CLI (modify: default values, add rect flag)
├── yolo_pose.py          # Pose training wrapper (modify: add rect param)
├── yolo_obb.py           # OBB training wrapper (no changes)
├── run_manager.py        # Run management + registration (no changes)
├── elastic_deform.py     # Elastic augmentation (no changes)
└── geometry.py           # OBB/pose geometry (no changes)
```

### Pattern 1: Frame Index Parsing
**What:** Extract temporal frame index from image filenames for temporal splitting
**When to use:** During `data convert --split-mode temporal`
**Example:**
```python
import re

def parse_frame_index(filename: str) -> int:
    """Extract frame index from filename like 'e3v82e0-..._657000.png'.

    The frame index is the last numeric section between the final underscore
    and the file extension.
    """
    stem = Path(filename).stem  # strip .png
    # Last segment after final underscore
    parts = stem.rsplit("_", 1)
    if len(parts) < 2:
        raise ValueError(f"Cannot parse frame index from {filename}")
    return int(parts[-1])
```

**Verified from actual data:** Filenames follow pattern `{camera_id}-{timestamp}_{frame_index}.png` where frame indices are: 612000, 621000, 630000, 639000, 648000, 657000, 666000, 675000, 684000, 693000 (10 unique values, step ~9000).

### Pattern 2: Temporal Split Implementation
**What:** Split images into train/val by frame index, keeping last 20% of temporal indices as val
**When to use:** In `generate_obb_dataset` and `generate_pose_dataset` when `split_mode="temporal"`
**Example:**
```python
def temporal_split(
    image_ids: list[int],
    image_lookup: dict[int, dict],
    val_fraction: float = 0.2,
) -> tuple[set[int], set[int]]:
    """Split image IDs by temporal frame index.

    All images sharing the same frame index go to the same split.
    The last val_fraction of unique frame indices become val.
    """
    # Extract frame indices
    frame_map: dict[int, list[int]] = {}  # frame_idx -> [image_ids]
    for img_id in image_ids:
        filename = image_lookup[img_id]["file_name"]
        frame_idx = parse_frame_index(filename)
        frame_map.setdefault(frame_idx, []).append(img_id)

    sorted_frames = sorted(frame_map.keys())
    n_val_frames = max(1, int(len(sorted_frames) * val_fraction))
    val_frames = set(sorted_frames[-n_val_frames:])

    val_ids = set()
    train_ids = set()
    for frame_idx, ids in frame_map.items():
        target = val_ids if frame_idx in val_frames else train_ids
        target.update(ids)

    return train_ids, val_ids
```

### Pattern 3: Val Tagging on Import
**What:** During `data import`, detect samples from `val/` subdirectory and add "val" tag
**When to use:** In `import_cmd` after importing each sample
**Example:**
```python
# In import_cmd, after scanning image_files:
for img_path in image_files:
    rel = img_path.relative_to(input_path / "images")
    is_val = rel.parts[0] == "val" if len(rel.parts) > 1 else False

    # ... import sample ...

    if is_val:
        # Add "val" tag to the sample
        conn = sample_store._connect()
        row = conn.execute("SELECT tags FROM samples WHERE id = ?", (sample_id,)).fetchone()
        tags = json.loads(row["tags"])
        if "val" not in tags:
            tags.append("val")
            conn.execute(
                "UPDATE samples SET tags = ? WHERE id = ?",
                (json.dumps(tags), sample_id),
            )
            conn.commit()
```

### Pattern 4: Tagged Split Mode in Assemble
**What:** Add `--split-mode tagged` to assemble that uses "val" tag instead of random split
**When to use:** When user wants deterministic temporal split from pre-tagged samples
**Example:**
```python
# In store.assemble(), add split_mode parameter:
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
    samples = self.query(**query)

    if split_mode == "tagged":
        val_samples = [s for s in samples if "val" in json.loads(s["tags"])]
        train_samples = [s for s in samples if "val" not in json.loads(s["tags"])]
    else:  # random
        # existing logic, with optional val_candidates_tag filter
        ...
```

### Pattern 5: Reason-Tagged Exclusion
**What:** Add optional `reason` parameter to `SampleStore.exclude()` that adds both "excluded" and the reason string as separate tags
**When to use:** `data exclude --reason TAG`
**Example:**
```python
def exclude(self, sample_ids: list[str], reason: str | None = None) -> int:
    # ... existing logic ...
    for sid in all_ids:
        row = conn.execute("SELECT tags FROM samples WHERE id = ?", (sid,)).fetchone()
        if row is None:
            continue
        tags = json.loads(row["tags"])
        if "excluded" not in tags:
            tags.append("excluded")
            if reason and reason not in tags:
                tags.append(reason)
            # ... update ...
            count += 1
    # ...
```

### Anti-Patterns to Avoid
- **Schema changes for reason tags:** The decision is explicit -- no schema changes. Reasons are tags in the existing JSON array.
- **Predefined reason enum:** Reasons are free-text. Do not constrain them.
- **Breaking backward compatibility on convert:** `--split-mode` defaults to `random`, preserving current behavior.
- **Modifying assemble's default behavior:** `split_mode` defaults to `"random"`, existing callers unaffected.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YOLO training | Custom training loops | `ultralytics` YOLO API | Already wrapped, battle-tested |
| Data store | File-based tracking | Existing `SampleStore` | Already built with dedup, provenance, tags |
| CLI framework | argparse | Click (already in use) | Consistent with codebase |
| Augmentation | New deformation code | Existing `elastic_deform.py` | Already integrated and tested |

## Common Pitfalls

### Pitfall 1: Tags Field Is JSON String, Not List
**What goes wrong:** Treating `tags` as a Python list when it's stored as a JSON string in SQLite
**Why it happens:** `store.query()` returns raw row dicts where `tags` is a JSON string
**How to avoid:** Always `json.loads(sample["tags"])` before checking tag membership
**Warning signs:** `"val" in sample["tags"]` returns True because "val" is a substring of the JSON string

### Pitfall 2: Frame Index Parsing Edge Cases
**What goes wrong:** Files without underscore-separated frame index break the parser
**Why it happens:** Augmented or renamed files may not follow the convention
**How to avoid:** Validate parsing and raise clear errors; only apply to `data convert` (raw COCO data always has this format)
**Warning signs:** `ValueError` on `int()` conversion

### Pitfall 3: Ultralytics rect Training Mode
**What goes wrong:** `rect=True` in Ultralytics uses non-square batches, which can interact with mosaic
**Why it happens:** Mosaic requires square images; rect mode pads to batch-max aspect ratio
**How to avoid:** Pass `rect=True` to `model.train()` -- Ultralytics handles the mosaic/rect interaction internally. When mosaic is very low (0.1), this is safe.
**Warning signs:** Warning messages from Ultralytics about rect+mosaic conflict

### Pitfall 4: Import From val/ Subdirectory Detection
**What goes wrong:** The relative path logic for detecting `val/` vs `train/` subdirectory fails
**Why it happens:** `img_path.relative_to(input_path / "images")` may have 1 or 2 parts depending on directory depth
**How to avoid:** Check `rel.parts[0]` only when `len(rel.parts) > 1`, otherwise it's a flat directory (no split subdirs)
**Warning signs:** All or no samples get tagged as "val"

### Pitfall 5: Existing Exclude Tests
**What goes wrong:** Adding `reason` parameter breaks existing tests that call `exclude(sample_ids)`
**Why it happens:** If reason becomes a required parameter
**How to avoid:** Make `reason` optional with `None` default. Existing callers pass no reason and behavior is unchanged.
**Warning signs:** Test failures in `test_store.py`

### Pitfall 6: Training With imgsz=128 + rect=True
**What goes wrong:** Ultralytics may warn or behave oddly with very small imgsz
**Why it happens:** Pose crops are 128x64, training at imgsz=128 with rect=True trains near native resolution
**How to avoid:** This is the intended behavior per user decision. The `rect=True` parameter tells Ultralytics to use rectangular batches matching the aspect ratio. Verify by checking training output dimensions.

## Code Examples

### Temporal Split in generate_obb_dataset
```python
# In coco_convert.py, modify generate_obb_dataset signature:
def generate_obb_dataset(
    coco: dict,
    images_dir: Path,
    output_dir: Path,
    median_arc: float,
    lateral_ratio: float,
    edge_factor: float,
    val_split: float,
    seed: int,
    n_keypoints: int = N_KEYPOINTS,
    split_mode: str = "random",  # NEW
) -> tuple[int, int]:
    # ...
    if split_mode == "temporal":
        train_ids, val_ids = temporal_split(
            all_image_ids, image_lookup, val_fraction=val_split
        )
    else:
        # existing random shuffle logic
        rng = random.Random(seed)
        rng.shuffle(all_image_ids)
        n_val = max(1, int(len(all_image_ids) * val_split))
        val_ids = set(all_image_ids[:n_val])
        train_ids = set(all_image_ids[n_val:])
```

### Reason-Aware Status Output
```python
# In status_cmd, after getting summary:
# Query for reason breakdown
conn = store._connect()
excluded_rows = conn.execute(
    "SELECT tags FROM samples WHERE "
    "EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = 'excluded')"
).fetchall()

reason_counts: dict[str, int] = {}
for row in excluded_rows:
    tags = json.loads(row["tags"])
    for tag in tags:
        if tag != "excluded":
            reason_counts[tag] = reason_counts.get(tag, 0) + 1

if reason_counts:
    reason_parts = ", ".join(f"{c} {r}" for r, c in sorted(reason_counts.items()))
    click.echo(f"  Excluded: {s['excluded_count']} ({reason_parts})")
```

### Adding rect to Pose Training
```python
# In yolo_pose.py, add rect parameter:
def train_yolo_pose(
    ...,
    rect: bool = True,  # NEW default True per user decision
) -> Path:
    # ...
    results = yolo_model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        device=device,
        project=str(output_dir / "_ultralytics"),
        name="train",
        imgsz=imgsz,
        patience=patience,
        mosaic=mosaic,
        rect=rect,  # NEW
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Random split in convert | Temporal split option | This phase | Prevents near-duplicate leakage |
| Flat exclusion tags | Reason-tagged exclusions | This phase | Better exclusion auditing |
| imgsz=640 for pose | imgsz=128 + rect=True | This phase | Train at native resolution |
| mosaic=1.0 default | mosaic=0.3 (OBB) / 0.1 (pose) | This phase | Better for small targets |

## Open Questions

1. **How should `include()` handle reason tags?**
   - What we know: `exclude()` adds both "excluded" and reason tag. `include()` currently only removes "excluded".
   - What's unclear: Should `include()` also remove reason tags, or leave them as audit trail?
   - Recommendation: `include()` should remove "excluded" but KEEP reason tags as audit trail. A sample that was excluded for "bad_crop" then re-included still has history. This doesn't affect query logic since `exclude_excluded` only checks for "excluded" tag.

2. **val-candidates tag filter in random mode**
   - What we know: User wants `--val-candidates` tag filter so only matching samples are candidates for val
   - What's unclear: Exact interaction with `pseudo_in_val` flag
   - Recommendation: `val_candidates_tag` is an additional filter on top of existing pseudo_in_val logic. If both are set, sample must match both criteria to be val-eligible.

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `store.py`, `data_cli.py`, `coco_convert.py`, `cli.py`, `yolo_pose.py`, `yolo_obb.py`, `run_manager.py`, `store_schema.py`
- Direct inspection of raw data filenames at `~/aquapose/projects/YH/training_data/raw/images/`
- Existing test files: `test_store.py`, `test_data_cli.py`

### Secondary (MEDIUM confidence)
- Ultralytics `rect` training parameter behavior (based on training knowledge and CONTEXT.md specifics)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in use, no new dependencies
- Architecture: HIGH - extending existing patterns with well-defined modifications
- Pitfalls: HIGH - identified from direct code reading and known patterns
- Temporal split: HIGH - verified filename pattern against actual data (10 frame indices confirmed)

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable -- internal codebase, no external API changes expected)
