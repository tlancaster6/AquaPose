# Phase 77: Training Module Code Quality - Research

**Researched:** 2026-03-09
**Domain:** Python refactoring, code deduplication, test coverage (src/aquapose/training/)
**Confidence:** HIGH

## Summary

This phase is a pure refactoring and test-coverage phase with no behavioral changes (except fixing the `seg` CLI missing `register_trained_model` bug). All changes are confined to `src/aquapose/training/` and `tests/unit/training/`. The audit report (quick task 24) identified 14 findings; this phase addresses findings 1-5, 8, 10, and 12.

The codebase is well-understood -- all source files have been read and the exact duplication points verified. The three YOLO training wrappers (`yolo_obb.py`, `yolo_seg.py`, `yolo_pose.py`) are 90%+ identical, differing only in the model default string and pose's extra `rect` parameter. The three CLI commands in `cli.py` follow the same pattern, with the `seg` command missing the `register_trained_model` call (a concrete bug). The duplicated functions in `coco_convert.py` are byte-for-byte identical to their `geometry.py` originals. The `_LutConfigFromDict` classes in `prep.py` and `pseudo_label_cli.py` are identical. `select_diverse_subset.py` has 330 LOC of complex selection logic with zero test coverage.

**Primary recommendation:** Execute in two waves -- Wave 1 consolidates all duplicated code (findings 1-5, 12) and fixes the seg registration bug (finding 2). Wave 2 adds test coverage (findings 8, 10).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Create a single `train_yolo()` function in a new `yolo_training.py` file
- `train_yolo()` takes a `model_type` parameter ("obb", "seg", "pose") that controls defaults
- Pose-specific parameters (e.g. `rect`) are explicit optional kwargs on `train_yolo()`, not a defaults dict
- Delete `yolo_obb.py`, `yolo_seg.py`, `yolo_pose.py` entirely -- no thin wrapper files
- Provide `train_yolo_obb`, `train_yolo_seg`, `train_yolo_pose` as convenience aliases in the same file
- Update `__init__.py` to export from `yolo_training.py`
- Extract a shared `_run_training()` orchestrator function for CLI commands
- Keep 3 separate Click commands (`train obb`, `train seg`, `train pose`) -- no CLI interface change
- Each CLI command becomes ~10 lines of option parsing + one `_run_training()` call
- This automatically fixes the `seg` command missing `register_trained_model`
- `affine_warp_crop` and `transform_keypoints`: keep canonical versions in `geometry.py`, delete duplicates from `coco_convert.py`, import from `geometry.py`
- `compute_arc_length`: consolidate into `geometry.py`, import from both `coco_convert.py` and `pseudo_labels.py`
- `_LutConfigFromDict`: move to `common.py`, import from both `pseudo_label_cli.py` and `prep.py`
- Inline pose label parsing in `data_cli.py`: replace with call to `parse_pose_label()` from `elastic_deform.py`
- Canonical `compute_arc_length()` in `geometry.py` returns `0.0` (not `None`) when fewer than `min_visible` keypoints are visible
- Return type is `float`, not `float | None`

### Claude's Discretion
- Internal organization of `yolo_training.py` (helper placement, constant naming)
- Exact parameter forwarding structure in `_run_training()`
- Test fixture design details

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| click | (existing) | CLI framework | Already used for all CLI commands |
| pytest | (existing) | Test framework | Project standard via `hatch run test` |
| shutil | stdlib | File copy in weight-copying logic | Used in existing training wrappers |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock | stdlib | Mock YOLO training calls in tests | Weight-copying tests need mock `model.train()` |
| numpy | (existing) | Array operations in subset selection tests | Test fixtures for curvature/confidence data |

No new dependencies required. This phase only reorganizes existing code and adds tests.

## Architecture Patterns

### Current Source Structure (before refactoring)
```
src/aquapose/training/
  yolo_obb.py          # 108 LOC -- DELETE
  yolo_seg.py          # 108 LOC -- DELETE
  yolo_pose.py         # 112 LOC -- DELETE
  cli.py               # 465 LOC -- refactor CLI commands
  geometry.py          # 355 LOC -- add compute_arc_length()
  coco_convert.py      # 596 LOC -- remove duplicate functions
  pseudo_labels.py     # ~500 LOC -- remove _compute_arc_length()
  common.py            # 187 LOC -- add _LutConfigFromDict
  prep.py              # ~300 LOC -- remove _LutConfigFromDict
  pseudo_label_cli.py  # ~630 LOC -- remove _LutConfigFromDict
  data_cli.py          # ~500 LOC -- replace inline parsing
  __init__.py          # 137 LOC -- update imports
```

### Target Structure (after refactoring)
```
src/aquapose/training/
  yolo_training.py     # NEW -- consolidated train_yolo() + convenience aliases
  cli.py               # SLIMMED -- _run_training() orchestrator + thin commands
  geometry.py          # EXPANDED -- canonical compute_arc_length()
  coco_convert.py      # SLIMMED -- imports from geometry.py
  pseudo_labels.py     # SLIMMED -- imports compute_arc_length from geometry.py
  common.py            # EXPANDED -- _LutConfigFromDict added
  prep.py              # SLIMMED -- imports _LutConfigFromDict from common.py
  pseudo_label_cli.py  # SLIMMED -- imports _LutConfigFromDict from common.py
  data_cli.py          # SLIMMED -- uses parse_pose_label()
  __init__.py          # UPDATED -- imports from yolo_training.py
```

### Pattern 1: Unified YOLO Training Function
**What:** Single `train_yolo()` with model_type-controlled defaults + convenience aliases
**When to use:** All YOLO training calls

```python
# yolo_training.py
_MODEL_DEFAULTS: dict[str, dict[str, object]] = {
    "obb": {"model": "yolo26n-obb"},
    "seg": {"model": "yolo26n-seg"},
    "pose": {"model": "yolo26n-pose", "rect": True},
}

def train_yolo(
    data_dir: Path,
    output_dir: Path,
    model_type: str,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    val_split: float = 0.2,
    imgsz: int = 640,
    model: str | None = None,
    weights: Path | None = None,
    patience: int = 100,
    mosaic: float = 1.0,
    rect: bool | None = None,
) -> Path:
    ...

def train_yolo_obb(...) -> Path:
    return train_yolo(..., model_type="obb")

def train_yolo_seg(...) -> Path:
    return train_yolo(..., model_type="seg")

def train_yolo_pose(...) -> Path:
    return train_yolo(..., model_type="pose")
```

### Pattern 2: CLI Training Orchestrator
**What:** Shared `_run_training()` function in `cli.py` that encapsulates the run lifecycle
**When to use:** All `train obb/seg/pose` CLI commands

```python
def _run_training(
    ctx: click.Context,
    model_type: str,
    data_dir: str | None,
    tag: str | None,
    cli_args: dict,
    *,
    train_kwargs: dict | None = None,
) -> None:
    """Shared training orchestrator: run dir, logging, train, summary, register."""
    config_path = get_config_path(ctx)
    project_dir = get_project_dir(ctx)

    if data_dir is None:
        data_dir = str(project_dir / "training_data" / model_type)

    run_dir = create_run_dir(config_path, model_type)
    setup_file_logging(run_dir, f"train-{model_type}")
    # ... snapshot, train, summary, register (with try/except), next_steps
```

### Pattern 3: Arc Length Return Convention
**What:** Canonical `compute_arc_length()` returns `0.0` for degenerate cases, never `None`
**When to use:** All arc-length callers

```python
# geometry.py
def compute_arc_length(
    coords: np.ndarray,
    visible: np.ndarray,
    min_visible: int = 2,
) -> float:
    """Sum Euclidean distances between consecutive visible keypoints.

    Returns 0.0 if fewer than min_visible keypoints are visible.
    """
```

### Anti-Patterns to Avoid
- **Changing CLI interface:** The 3 Click commands must remain separate (`train obb`, `train seg`, `train pose`). Users should not see any change in CLI behavior.
- **Breaking import boundary:** `training/` must not import from `aquapose.engine`. The `_LutConfigFromDict` exists specifically to avoid this. The boundary test in `test_training_cli.py` enforces this.
- **Touching `common.py` legacy classes:** `EarlyStopping`, `MetricsLogger`, `save_best_and_last`, `make_loader` in `common.py` may be legacy (pre-Ultralytics). Adding `_LutConfigFromDict` is fine; do not refactor existing contents.
- **Forgetting lazy imports:** `from ultralytics import YOLO` must remain inside function bodies, not at module level. This prevents import-time GPU initialization.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pose label parsing | Inline token splitting in `data_cli.py` | `parse_pose_label()` from `elastic_deform.py` | Already handles edge cases, raises `ValueError` for multi-line labels |
| LUT config protocol | New class per module | Single `_LutConfigFromDict` in `common.py` | Satisfies `LutConfigLike` protocol, DRY |
| Weight copying logic | Per-model-type copy code | Single implementation in `train_yolo()` | Same logic for all model types |

## Common Pitfalls

### Pitfall 1: Breaking the `__init__.py` re-exports
**What goes wrong:** After deleting `yolo_obb.py`, `yolo_seg.py`, `yolo_pose.py`, imports like `from aquapose.training import train_yolo_obb` break.
**Why it happens:** `__init__.py` imports from the old module paths.
**How to avoid:** Update `__init__.py` to import all three convenience aliases from `yolo_training.py`. Keep the same public names in `__all__`.
**Warning signs:** Import errors in existing tests.

### Pitfall 2: `coco_convert.py` re-exports `compute_arc_length`
**What goes wrong:** `__init__.py` imports `compute_arc_length` from `coco_convert`. After moving the function to `geometry.py`, the `coco_convert` module must still re-export it OR `__init__.py` must be updated.
**Why it happens:** The function is a public API exported from the training package.
**How to avoid:** Keep `compute_arc_length` importable from `coco_convert` (via re-import from `geometry`) and update `__init__.py` to import from `geometry.py` directly. Both approaches work; the key is to not break any caller.
**Warning signs:** `ImportError` in tests or external code.

### Pitfall 3: Arc length return type change breaks callers
**What goes wrong:** `coco_convert.compute_arc_length` currently returns `float | None`. Changing to `float` requires updating all callers that check `is None`.
**Why it happens:** The caller `compute_median_arc_length` checks `if arc is not None`.
**How to avoid:** After changing the return type to always `float`, update `compute_median_arc_length` to check `if arc > 0.0` instead of `if arc is not None`. Also check `pseudo_labels.py` callers.
**Warning signs:** `None` comparisons on the new `float` return.

### Pitfall 4: `seg` CLI default mosaic differs from `obb`
**What goes wrong:** The `seg` command has `mosaic=1.0` default while `obb` has `mosaic=0.3`. These model-type-specific CLI defaults must be preserved.
**Why it happens:** Easy to accidentally unify CLI defaults when consolidating.
**How to avoid:** CLI option defaults remain on the Click decorators (they don't change). The `_run_training()` orchestrator just forwards whatever values the CLI decorators provide.
**Warning signs:** Changed default behavior for a model type.

### Pitfall 5: `pose` CLI has `imgsz=320` default, others have `imgsz=640`
**What goes wrong:** Losing the pose-specific `imgsz=320` default.
**Why it happens:** Same unification temptation as mosaic.
**How to avoid:** Keep model-type-specific defaults on Click decorators, not in `_run_training()`.

## Code Examples

### Weight-Copying Logic (to be tested)
```python
# Verified from yolo_obb.py lines 88-108 (identical in all 3 wrappers)
save_dir = (
    results.save_dir
    if results is not None and results.save_dir is not None
    else output_dir / "_ultralytics" / "train"
)
weights_dir = Path(str(save_dir)) / "weights"

best_src = weights_dir / "best.pt"
last_src = weights_dir / "last.pt"
best_dst = output_dir / "best_model.pt"
last_dst = output_dir / "last_model.pt"

if best_src.exists():
    shutil.copy2(best_src, best_dst)
if last_src.exists():
    shutil.copy2(last_src, last_dst)
if not best_dst.exists() and last_dst.exists():
    shutil.copy2(last_dst, best_dst)
```

### Inline Pose Parsing to Replace (data_cli.py:174-196)
```python
# CURRENT (to be replaced):
first_line = label_text.split("\n")[0].split()
kp_tokens = first_line[5:]
if len(kp_tokens) >= 6:
    pts = []
    for ki in range(0, len(kp_tokens), 3):
        if ki + 2 < len(kp_tokens):
            vis = float(kp_tokens[ki + 2])
            if vis > 0:
                pts.append([float(kp_tokens[ki]), float(kp_tokens[ki + 1])])

# REPLACEMENT:
from .elastic_deform import parse_pose_label
try:
    kps, vis = parse_pose_label(label_text)
    vis_pts = kps[vis]
    if len(vis_pts) >= 3:
        sample_meta["curvature"] = compute_curvature(vis_pts)
except ValueError:
    pass  # multi-line label (multi-fish crop), skip curvature
```

### Test Structure for Weight Copying
```python
# Test scenarios for weight-copying logic:
# 1. Both best.pt and last.pt exist -> both copied
# 2. Only best.pt exists -> best copied, last not
# 3. Only last.pt exists -> last copied, also copied as best_model.pt (fallback)
# 4. Neither exists -> no files copied, best_model.pt doesn't exist
# Mock YOLO().train() to return a mock results object with save_dir
```

### Test Structure for select_diverse_subset
```python
# Test scenarios for select_obb_subset:
# 1. Basic: 3 cameras, 10 entries each, target=9 -> 3 per camera
# 2. Camera balance: unequal camera counts -> proportional selection
# 3. Temporal spread: verify selections span time range
# 4. Edge case: target > available -> selects all
# 5. Edge case: single camera -> all picks from that camera
# 6. val_fraction=0 -> no val split

# Test scenarios for select_pose_subset:
# 1. Curvature stratification: known quartiles -> verify bin coverage
# 2. Camera + curvature: cross-product grouping works
# 3. Edge case: fewer entries than target -> selects all
# 4. val_fraction: verify temporal val split (later frames)
```

## State of the Art

No external library changes or deprecations relevant to this phase. All changes are internal refactoring.

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 3 separate YOLO wrapper files | Single `yolo_training.py` | This phase | Eliminates 200+ LOC duplication |
| Copy-paste CLI handlers | Shared `_run_training()` | This phase | Fixes seg registration bug, reduces ~250 LOC |
| `compute_arc_length` returns `None` | Returns `0.0` | This phase | Simpler caller logic, consistent with `pseudo_labels.py` |

## Open Questions

1. **Convenience alias signature**
   - What we know: `train_yolo_obb`, `train_yolo_seg`, `train_yolo_pose` should be convenience aliases
   - What's unclear: Should they duplicate the full parameter list (with model-type defaults) or use `**kwargs` forwarding?
   - Recommendation: Duplicate the full parameter list for IDE autocompletion and type checking. Use model-type-specific defaults in the signature (e.g., `model="yolo26n-obb"` for obb). Each alias is ~5 lines calling `train_yolo()`.

2. **`compute_arc_length` location in `__init__.py` exports**
   - What we know: Currently exported from `coco_convert`. After moving to `geometry.py`, `__init__.py` needs updating.
   - Recommendation: Import from `geometry.py` in `__init__.py`. Keep a re-import in `coco_convert.py` for backward compatibility of direct imports.

## Sources

### Primary (HIGH confidence)
- Direct source code inspection of all files listed above
- Audit report: `.planning/quick/24-audit-recent-pseudo-label-and-training-d/24-AUDIT-REPORT.md`
- CONTEXT.md decisions from user discussion session

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new libraries, pure internal refactoring
- Architecture: HIGH - all source files read, duplication verified byte-for-byte
- Pitfalls: HIGH - identified from direct code inspection of callers and import chains

**Research date:** 2026-03-09
**Valid until:** Indefinite (internal refactoring, not dependent on external versions)
