# Phase 73: Round 1 Pseudo-Labels & Retraining - Research

**Researched:** 2026-03-07
**Domain:** Pseudo-label generation, diversity-based subset selection, data store management, YOLO model training, A/B curation comparison
**Confidence:** HIGH

## Summary

Phase 73 is a workflow-heavy phase that orchestrates existing CLI tools through a specific sequence: generate pseudo-labels, select a diverse subset, visually audit, import into stores, assemble datasets (curated vs uncurated arms), train 4 models, and compare results. The codebase already has all the building blocks -- `pseudo-label generate`, `data import`, `data exclude`, `data assemble`, `train obb/pose`, and `train compare`. The primary new work is a **diversity selection script** that picks ~50 OBB images and ~320 pose crops from the full pseudo-label output using camera coverage, temporal spread, and curvature stratification.

The store's `import_sample()` already accepts `source="pseudo"` and `metadata` dict (for confidence, round, etc.). The `assemble()` method supports `split_mode="tagged"` and `pseudo_in_val=False` by default, which correctly routes pseudo-labels to training only. The `exclude()` method supports `--reason TAG` for typed exclusion tracking. The `train compare` command reads `summary.json` from all run directories and formats a side-by-side table.

**Primary recommendation:** Structure this as 3 plans: (1) pseudo-label generation + diversity selection script + CLI enhancements (`--input-dir` for inspect, `--by-filename` for exclude) + visual audit + exclusion marking, (2) `--include-excluded` flag + dataset assembly, (3) training + secondary val evaluation + comparison and checkpoint. Plan 1 requires a new `select_diverse_subset.py` script and two CLI enhancements. Plan 2 requires one small CLI change. Plan 3 is pure workflow using existing CLI.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Target ~50 full-frame OBB images and ~320 pose crop images (mirroring manual annotation set sizes)
- OBB selection uses 3-axis diversity: camera coverage (~4 per camera with 2 flex slots), temporal spread within each camera (divide into 4 temporal bins, pick one per bin), fish count tiebreaker (prefer underrepresented fish counts)
- Pose selection uses 2-axis diversity: camera coverage and 3D curvature stratification (from confidence.json curvature_3d field)
- Exhaustive review of the diversity-selected subset (not sampling)
- Reject criteria: obvious failures only (wrong fish labeled, grossly misaligned keypoints, label on background)
- Accept imprecise-but-reasonable labels
- Both A/B arms use the same diversity-selected subset; Arm A (curated) has bad labels excluded, Arm B (uncurated) does not
- Exclusion tracking via freeform failure-typed reason tags using `data exclude --reason TAG`
- Primary comparison metric: training val metrics via `train compare`
- Elastic augmentation on manual annotations only (pseudo-labels have natural curvature diversity)
- Consensus-only pseudo-labels for pose (no gap-fill labels in round 1)
- OBB pseudo-labels include both consensus and gap sources
- Import with source=pseudo, round=1, with confidence scores stored as metadata
- Primary val set: Phase 71 manual val set (unchanged)
- Secondary val set: ~20% holdout from curated pseudo-labels, split temporally
- Train from scratch (no transfer learning from baseline weights)
- Same hyperparameters as Phase 71 baseline: epochs=100, patience=100, mosaic defaults (0.3 OBB, 0.1 pose)
- Run tags: `round1-curated` / `round1-uncurated`
- 4 training runs total: OBB curated, OBB uncurated, pose curated, pose uncurated

### Claude's Discretion
- Exact implementation of the diversity selection script (stratification bins, tiebreaker logic)
- How confidence scores are stored as metadata during store import (may need minor store schema extension)
- Order of operations for the secondary val set evaluation

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ITER-02 | Pseudo-labels generated from baseline run, visually audited, and imported into store as `source=pseudo, round=1` | Existing `pseudo-label generate` CLI produces YOLO-format outputs with confidence.json sidecars. `data import --store {obb,pose} --source pseudo` handles import with metadata. Diversity selection script is the only new code needed. |
| ITER-03 | Round 1 models trained on manual + pseudo-label datasets (elastic augmentation on manual only) and registered | `data assemble` with default `pseudo_in_val=False` routes pseudo-labels to train-only. `--augment` flag on import only applies to manual samples (already imported with augmentation in Phase 71). `train obb/pose --tag` handles training and auto-registration. |
| ITER-06 | A/B comparison -- model trained with human-curated exclusions vs model trained with full uncurated pseudo-labels, quantifying curation value | `data exclude --reason TAG` marks bad samples. Two `data assemble` calls (one with exclusions honored, one with `--tags-exclude` removed or reversed) produce curated vs uncurated datasets. `train compare --model-type {obb,pose}` shows side-by-side metrics. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| aquapose pseudo-label CLI | current | Generate OBB+pose pseudo-labels from diagnostic caches | Already implemented, tested in Phase 72 context |
| aquapose data CLI | current | Import, exclude, assemble training data | Full CRUD with provenance tracking |
| aquapose train CLI | current | Train YOLO OBB/pose models with auto-registration | Handles summary.json, model registration, config update |
| aquapose train compare | current | Side-by-side run comparison | Reads summary.json, formats table with best-value highlighting |
| SampleStore | current | SQLite-backed data management | Content-hash dedup, source-priority upsert, tag-based querying |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | installed | Diversity selection statistics | Curvature binning, stratified sampling |
| json | stdlib | Read confidence.json sidecars | Parse curvature_3d, confidence scores |

## Architecture Patterns

### Recommended Workflow Structure
```
Phase 73 Execution Flow:
1. aquapose pseudo-label generate [run]          # Full pseudo-label generation
2. python select_diverse_subset.py [pseudo_dir]  # New: diversity selection
3. aquapose pseudo-label inspect [run]            # Visual audit of subset
4. aquapose data exclude --store {type} --ids ... --reason TAG  # Mark failures
5. aquapose data import --store obb --source pseudo --input-dir [selected/obb]
6. aquapose data import --store pose --source pseudo --input-dir [selected/pose/consensus]
7. aquapose data assemble (x4: curated/uncurated x obb/pose)
8. aquapose train obb/pose (x4 runs with --tag)
9. aquapose train compare --model-type obb/pose
```

### Pattern 1: Diversity Selection Script
**What:** Standalone Python script that reads pseudo-label directories and confidence.json sidecars, applies multi-axis diversity sampling, and copies selected files to a new subset directory.
**When to use:** After `pseudo-label generate`, before import.

**OBB Selection Algorithm:**
```python
# Input: pseudo_dir/obb/confidence.json + image filenames
# Parse filenames: {frame_idx:06d}_{cam_id}.jpg
# 1. Group by cam_id (12 cameras)
# 2. Within each camera, divide frames into 4 temporal bins
# 3. From each bin, pick 1 frame preferring underrepresented fish counts
#    (fish count from confidence.json "tracked_fish_count" field)
# 4. Target: ~4 per camera = ~48, plus 2 flex slots for edge cases
# Output: Copy selected images+labels to subset directory
```

**Pose Selection Algorithm:**
```python
# Input: pseudo_dir/pose/consensus/confidence.json + image filenames
# Parse filenames: {frame_idx:06d}_{cam_id}_{fish_idx:03d}.jpg
# 1. Read curvature_3d from confidence.json labels[].curvature_3d
# 2. Stratify into curvature bins (e.g., quartiles)
# 3. Within each bin, sample proportionally across cameras
# 4. Target: ~320 crops total
# Output: Copy selected images+labels to subset directory
```

### Pattern 2: A/B Dataset Assembly
**What:** Create two parallel datasets from the same pseudo-label pool -- one excluding marked-bad samples, one including all.
**When to use:** After audit and exclusion marking.

```bash
# Curated arm: assembles with default exclude_excluded=True
aquapose data assemble --store obb --name round1-curated --split-mode tagged

# Uncurated arm: needs all pseudo-labels including excluded ones
# Option A: Use --tags-exclude to skip exclusion filtering
# Option B: Temporarily include all, assemble, then re-exclude
```

**Important finding:** The `assemble()` method's `exclude_excluded` parameter defaults to `True` in `query()`, meaning excluded samples are automatically filtered. For the uncurated arm, the assemble CLI does NOT expose an `--include-excluded` flag. This is a gap that needs addressing -- either:
1. Add `--include-excluded` flag to assemble CLI (small code change), or
2. Use `data include` to temporarily un-exclude, assemble uncurated, then re-exclude

**Recommendation:** Option 1 is cleaner -- add a `--include-excluded` flag to `assemble_cmd` that passes `exclude_excluded=False` to the query dict.

### Pattern 3: Secondary Val Set from Pseudo-Labels (Pose Only)
**What:** Hold out ~20% of curated pseudo-label pose crops as a secondary validation set for post-training evaluation. OBB budget (~50) is too small for a meaningful holdout.
**When to use:** During assembly of the curated dataset.

The `assemble()` method supports `pseudo_in_val=True` which allows pseudo-labels in the val split. For the secondary val set:
- Tag later-temporal pseudo-label samples with "val" before assembly
- Use `split_mode="tagged"` to route tagged samples to val
- Or use `val_candidates_tag` parameter in random mode

**Recommended approach:** After import, tag the last ~20% of pseudo-label samples (by frame index, later frames) with "val" tag. Then assemble with `split_mode="tagged"`. This ensures the primary manual val samples AND secondary pseudo val samples are both in the val set.

### Anti-Patterns to Avoid
- **Augmenting pseudo-labels:** Do NOT use `--augment` when importing pseudo-labels. Elastic augmentation is for manual annotations only (they already have augmented variants from Phase 71 import).
- **Transfer learning from baseline:** Decision is train from scratch. Do NOT use `--weights` flag.
- **Using gap-fill pose labels:** Phase 73 uses consensus-only for pose. OBB includes both gap and consensus (they're already merged by the generate CLI).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pseudo-label generation | Custom reprojection script | `aquapose pseudo-label generate` | Already handles spline evaluation, refractive projection, confidence scoring, completeness filtering |
| Dataset assembly | Manual symlink creation | `aquapose data assemble` | Handles train/val split, symlinks, dataset.yaml, manifest persistence |
| Model comparison | Custom metric extraction | `aquapose train compare` | Reads summary.json, formats table, supports CSV export |
| Sample exclusion | Manual DB edits | `aquapose data exclude --reason TAG` | Cascades to augmented children, tracks reason tags |

## Common Pitfalls

### Pitfall 1: pose dataset.yaml missing kpt_shape/flip_idx
**What goes wrong:** `store.assemble()` has a known bug where it may not write `kpt_shape` and `flip_idx` in pose dataset.yaml.
**Why it happens:** The `_detect_n_keypoints` method only runs when `self.root.name == "pose"` but the assembled dataset directory may not satisfy this check.
**How to avoid:** After every pose dataset assembly, manually verify dataset.yaml contains `kpt_shape: [6, 3]` and `flip_idx: [0, 1, 2, 3, 4, 5]`. Add them if missing.
**Warning signs:** Ultralytics training will fail with keypoint-related errors.

**Update:** Looking at the code more carefully, `assemble()` checks `if self.root.name == "pose"` (line 738 of store.py). The store root IS `training_data/pose/`, so `self.root.name` is `"pose"`. The issue is that `_detect_n_keypoints` looks at the assembled dataset's *store* samples, not the assembled directory. This should actually work correctly now. BUT the MEMORY.md explicitly notes this as a known bug that required manual fix in Phase 71. **Always verify after assembly.**

### Pitfall 2: Uncurated arm assembly excluding samples
**What goes wrong:** The uncurated arm accidentally filters out excluded samples because `exclude_excluded=True` is the default.
**Why it happens:** `assemble()` calls `query()` which defaults to `exclude_excluded=True`.
**How to avoid:** Need to add `--include-excluded` flag to assemble CLI, or temporarily un-exclude samples.
**Warning signs:** Uncurated dataset has fewer samples than expected.

### Pitfall 3: Confidence.json sidecar key mismatch
**What goes wrong:** Import CLI reads `confidence.json` keyed by image stem, but pose crops have `{frame_idx:06d}_{cam_id}_{fish_idx:03d}` stems while OBB images have `{frame_idx:06d}_{cam_id}` stems.
**Why it happens:** The pseudo-label CLI writes confidence.json with the `{frame_idx:06d}_{cam_id}` key for both OBB and pose, but pose crop files include the fish_idx suffix.
**How to avoid:** The import CLI already handles this -- it looks up by `img_path.stem` which matches the filename. For pose, the confidence.json entry key is `{frame_idx:06d}_{cam_id}` (without fish_idx), so it maps to the frame+cam level, with per-fish entries in the `labels` array. The import CLI takes `labels[0]` metadata, which works for single-fish crops but may lose fish-specific info for multi-fish frames. This is acceptable for round 1.

### Pitfall 4: Round metadata not stored
**What goes wrong:** No way to distinguish round 1 pseudo-labels from future round 2 pseudo-labels in the store.
**Why it happens:** The store schema has `source` field but no `round` field.
**How to avoid:** Pass `--metadata-json '{"round": 1}'` during import. The `round` will be stored in the sample's metadata JSON field. For querying, use `min_confidence` or batch_id based filtering.

### Pitfall 5: Temporal secondary val set requires manual tagging
**What goes wrong:** No built-in way to tag "later temporal" pseudo-label samples as val after import.
**Why it happens:** The store assigns UUID-based sample IDs, losing the original filename's temporal info.
**How to avoid:** Either (a) tag samples during import by placing val candidates in a `val/` subdirectory within the input, or (b) query samples after import and tag based on metadata. Option (a) is simpler -- the diversity selection script should place the last ~20% temporal samples in a `val/` subdirectory, and the import CLI already auto-tags `val/` subdirectory samples with the "val" tag (line 208-221 of data_cli.py).

### Pitfall 6: Four training runs consume significant GPU time
**What goes wrong:** Waiting for sequential GPU training (4 runs x ~1-2 hours each).
**Why it happens:** Only one GPU available; each run is ~100 epochs.
**How to avoid:** Use `TaskCreate` for long-running training. Plan tasks to allow non-blocking execution. Consider running OBB runs first (faster) then pose runs.

## Code Examples

### Pseudo-label generation
```bash
# Generate from latest run (Phase 72 baseline)
aquapose --project ~/aquapose/projects/YH pseudo-label generate --viz
```

### Diversity selection (new script pattern)
```python
# Read confidence.json
import json
from pathlib import Path

conf_path = pseudo_dir / "obb" / "confidence.json"
conf_data = json.loads(conf_path.read_text())

# Parse filename -> (frame_idx, cam_id)
for stem, entry in conf_data.items():
    parts = stem.split("_")
    frame_idx = int(parts[0])
    cam_id = "_".join(parts[1:])  # cam_id may contain underscores
    fish_count = entry.get("tracked_fish_count", 0)
    # entry["labels"][i]["curvature_3d"] for curvature
```

### Store import with metadata
```bash
aquapose --project ~/aquapose/projects/YH data import \
    --store obb --source pseudo \
    --input-dir /path/to/selected/obb \
    --metadata-json '{"round": 1}' \
    --batch-id "round1-obb"
```

### Dataset assembly for A/B
```bash
# Curated (excluded samples filtered out automatically)
aquapose --project ~/aquapose/projects/YH data assemble \
    --store obb --name round1-curated --split-mode tagged

# Uncurated (needs --include-excluded or workaround)
aquapose --project ~/aquapose/projects/YH data assemble \
    --store obb --name round1-uncurated --split-mode tagged --include-excluded
```

### Training with tags
```bash
aquapose --project ~/aquapose/projects/YH train obb \
    --data-dir ~/aquapose/projects/YH/training_data/obb/datasets/round1-curated \
    --tag round1-curated --epochs 100 --patience 100 --mosaic 0.3

aquapose --project ~/aquapose/projects/YH train pose \
    --data-dir ~/aquapose/projects/YH/training_data/pose/datasets/round1-curated \
    --tag round1-curated --epochs 100 --patience 100 --mosaic 0.1
```

### Model comparison
```bash
aquapose --project ~/aquapose/projects/YH train compare --model-type obb
aquapose --project ~/aquapose/projects/YH train compare --model-type pose
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual-only training data | Manual + pseudo-labels from 3D reconstruction | Phase 73 (now) | Larger training set, natural diversity |
| Random data selection | Diversity-maximizing subset selection | Phase 73 (now) | Better coverage of cameras, temporal ranges, curvatures |
| Single dataset per round | A/B curated vs uncurated comparison | Phase 73 (now) | Quantifies value of human curation |

## Open Questions

1. **Uncurated arm assembly mechanism**
   - What we know: `assemble()` defaults to `exclude_excluded=True`. The CLI does not expose `--include-excluded`.
   - What's unclear: Whether to add a CLI flag or use a workaround.
   - Recommendation: Add `--include-excluded` flag to `assemble_cmd`. It's a 2-line change (add option, pass to query). This is cleaner than temporary include/exclude toggling which risks leaving the store in an inconsistent state.

2. **Pose confidence.json multi-fish entries**
   - What we know: Pose confidence.json is keyed by `{frame_idx}_{cam_id}` with a `labels` array. Import CLI takes `labels[0]` for metadata.
   - What's unclear: Whether fish_idx-specific metadata (like per-fish curvature) is needed for selection.
   - Recommendation: The diversity selection script should read the full `labels` array and use per-fish `curvature_3d` for stratification. The import metadata loss is acceptable since we only need confidence score, not full per-fish breakdown.

3. **cam_id parsing from filenames**
   - What we know: Filenames are `{frame_idx:06d}_{cam_id}`. Camera IDs in AquaPose may contain underscores (e.g., `cam_01`).
   - What's unclear: Exact camera ID format in the YH project.
   - Recommendation: Split on first `_` after the 6-digit frame index. The frame_idx is always exactly 6 digits, so `stem[:6]` is frame_idx and `stem[7:]` is cam_id.

## Sources

### Primary (HIGH confidence)
- `src/aquapose/training/pseudo_label_cli.py` - Full pseudo-label generation and inspection CLI
- `src/aquapose/training/pseudo_labels.py` - Core label generation and confidence scoring
- `src/aquapose/training/store.py` - SampleStore with import, exclude, assemble, register_model
- `src/aquapose/training/data_cli.py` - CLI commands for import, exclude, assemble, status
- `src/aquapose/training/cli.py` - Training CLI with obb, pose, compare commands
- `src/aquapose/training/compare.py` - Run comparison table formatting
- `src/aquapose/training/run_manager.py` - Run directory management, model registration
- `src/aquapose/training/store_schema.py` - SQLite schema, source priority

### Secondary (MEDIUM confidence)
- `.planning/phases/73-round-1-pseudo-labels-retraining/73-CONTEXT.md` - User decisions
- MEMORY.md - Known bugs (kpt_shape/flip_idx in pose dataset.yaml)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools exist and are well-understood from source code review
- Architecture: HIGH - Workflow follows established CLI patterns with one new script
- Pitfalls: HIGH - Identified from direct source code analysis and prior phase experience

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable internal codebase)
