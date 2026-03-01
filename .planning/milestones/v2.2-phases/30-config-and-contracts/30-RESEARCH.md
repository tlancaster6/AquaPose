# Phase 30: Config and Contracts - Research

**Researched:** 2026-02-28
**Domain:** Python frozen dataclasses, YAML config loading, Click CLI, dataclass field extension
**Confidence:** HIGH

## Summary

Phase 30 is a pure refactoring and extension phase — no new pipeline stages, no new ML backends. It touches five distinct areas: (1) promoting `device` and `stop_frame` to top-level `PipelineConfig` and propagating them through `build_stages()`; (2) making `n_sample_points` configurable end-to-end (replacing the `n_points=15` default in `MidlineConfig` with a config-driven value, and routing it through all downstream modules); (3) strengthening `_filter_fields()` from silent-drop to strict-reject with field-rename hints; (4) replacing the existing `init-config` command (which currently just dumps a raw `PipelineConfig` as sorted YAML) with a proper project-directory scaffolding command; and (5) adding optional fields to the `Detection` and `Midline2D` dataclasses.

The codebase is already well-structured for this work. All config dataclasses are frozen, `load_config()` handles layered YAML + CLI overrides, and there is an existing `_filter_fields()` inner function (currently only applied to `AssociationConfig`, `TrackingConfig`, and `LutConfig`, NOT to `DetectionConfig`, `MidlineConfig`, `ReconstructionConfig`, or `SyntheticConfig`). The `Detection` dataclass lives in `src/aquapose/segmentation/detector.py` and `Midline2D` lives in `src/aquapose/reconstruction/midline.py`. Both are plain (non-frozen) dataclasses.

The hardcoded `15` problem is widespread: the literal appears in `reconstruction/triangulation.py` (as the module constant `N_SAMPLE_POINTS`), `reconstruction/midline.py` (default argument), `core/synthetic.py` (inline assignment), `visualization/midline_viz.py` and `visualization/triangulation_viz.py` (inline comparisons and calls), `io/midline_writer.py` (imports and uses `N_SAMPLE_POINTS`), `reconstruction/curve_optimizer.py` (imports and uses `N_SAMPLE_POINTS`), and many tests. The strategy is to route the value from `MidlineConfig.n_points` (renamed / aliased as `n_sample_points` at the top-level or kept as `n_points` in the sub-config) through `build_stages()` to every module that currently hard-codes it.

**Primary recommendation:** Implement changes in dependency order: dataclass extensions first (safest, no logic change), then `_filter_fields()` universalization, then device/stop_frame promotion, then `n_sample_points` propagation, then `init-config` rewrite. Test after each step.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### init-config Scaffolding
- `aquapose init-config <name>` creates `~/aquapose/projects/<name>/` with config YAML plus four empty subdirectories:
  - `runs/` — pipeline output
  - `models/` — ML model weights and training data
  - `geometry/` — calibration file, LUTs
  - `videos/` — input videos or subdirectories of input videos
- Generated YAML is minimal: only essential fields active, with brief comments pointing to docs for advanced config (stage-specific params, observer config are NOT included as commented-out blocks)
- `--synthetic` flag adds a `synthetic:` section to the YAML; this section does NOT include its own fish count — it relies on the top-level `n_animals` (in real data = expected count, in synthetic = generate-and-expect count)

#### YAML Field Ordering
- Fields ordered by what a user edits first when setting up a new project:
  1. `project_dir` (root for all relative paths)
  2. Other paths (video_dir, calibration_path, output_dir, etc.)
  3. `n_animals` and `device`
  4. Stage-specific parameters (below the fold)

#### Config Defaults
- `device`: Auto-detect — `torch.cuda.is_available()` → `cuda:0`, otherwise `cpu`. No explicit default in YAML; auto-detect is the behavior when field is absent.
- `n_sample_points`: Default 10. Intentionally lower than the current hardcoded 15 to flush out any remaining hardcoded references throughout the codebase.
- `n_animals`: **Required** — no default. Config validation fails if missing.
- `stop_frame`: Default `None` (process all frames). User sets it only for dev/debug runs.

#### Backward Compatibility
- `_filter_fields()` uses **strict reject**: unknown YAML fields cause an error listing the unrecognized fields. No silent dropping.
- Renamed fields (e.g. `expect_fish_count` → `n_animals`) are rejected with a hint: error message says "did you mean `n_animals`?" so the user knows exactly what to change. No auto-rename.
- Removed fields get a generic "unknown field: X" error — no deprecation registry or removal explanations. Keep it simple.

#### Dataclass Extensions
- `Detection.angle`: Radians from x-axis, standard math convention (0 = right, π/2 = up), range [-π, π]. Consistent with YOLO-OBB output format. `None` when produced by non-OBB detectors.
- `Detection.obb_points`: `np.ndarray` of shape (4, 2) — 4 corner points as (x, y), ordered clockwise from top-left of the oriented box. `None` when produced by non-OBB detectors.
- `Midline2D.point_confidence`: `np.ndarray` of shape (N,), values in [0, 1] probability range. Directly usable as triangulation weights. **Always an array** — backends that don't produce per-point confidence (e.g. segment-then-extract) fill with 1.0s. No None checks downstream.

### Claude's Discretion
- Whether `_filter_fields()` is a standalone utility or per-class classmethod
- Exact subdirectory structure within `_filter_fields()` implementation
- Project directory scaffolding details beyond the four named subdirectories
- Config validation error message formatting

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CFG-01 | Pipeline accepts a single top-level `device` parameter that propagates to all stages, eliminating per-stage device configuration | `device` currently lives only in `DetectionConfig`; `build_stages()` passes `config.detection.device` to both `DetectionStage` and `MidlineStage`. Must add `device: str` to `PipelineConfig`, add auto-detect logic, and update `build_stages()` to use `config.device`. |
| CFG-02 | `n_sample_points` configurable via pipeline config, no hardcoded constants remaining | `N_SAMPLE_POINTS = 15` in `triangulation.py` is imported by `midline_writer.py`, `curve_optimizer.py`, `synthetic/fish.py`. `MidlineConfig.n_points` already exists (default 15). Must add `n_sample_points` to `PipelineConfig` (default 10), wire through `build_stages()`, remove/replace all hardcoded uses of the constant. |
| CFG-03 | Fish count is a single unified parameter | `AssociationConfig.expected_fish_count` and `SyntheticConfig.fish_count` are already auto-populated from `n_animals` in `load_config()`. The `n_animals` field already exists on `PipelineConfig`. This requirement is mostly already satisfied — verify and document the propagation. |
| CFG-04 | `stop_frame` is a top-level pipeline parameter | Currently `DetectionConfig.stop_frame`. Must move to `PipelineConfig.stop_frame`, update `build_stages()` to pass `config.stop_frame` to `DetectionStage`. |
| CFG-05 | All stage configs use `_filter_fields()` so YAML files load without error after schema changes | Currently only `AssociationConfig`, `TrackingConfig`, and `LutConfig` are filtered. `DetectionConfig`, `MidlineConfig`, `ReconstructionConfig`, `SyntheticConfig` are NOT. Must universalize. Also change from silent-drop to strict-reject with rename hints. |
| CFG-06 | `init-config` creates project directory scaffold under `~/aquapose/projects/<name>/` | Current `init-config` only writes a YAML file. Must add `<name>` argument, create directory structure, write minimal commented YAML. Requires CLI signature change from `--output` flag to positional `<name>` arg. |
| CFG-07 | Config paths resolve relative to `project_dir`, absolute paths override | `project_dir` field does not exist yet on `PipelineConfig`. Must add it and implement path resolution in `load_config()`. |
| CFG-08 | `init-config --synthetic` includes the synthetic config section | New flag on the new `init-config` command. |
| CFG-09 | `init-config` orders YAML fields by user relevance | Current `serialize_config()` uses `sort_keys=True` (alphabetical). Must write custom ordered dict for `init-config` output specifically (serialization roundtrip doesn't need to change). |
| CFG-10 | `Detection` dataclass carries optional rotation angle | Add `angle: float | None = None` to `Detection` in `segmentation/detector.py`. All existing code paths create `Detection` without `angle`, so default of `None` is backward compatible. |
| CFG-11 | `Midline2D` dataclass carries optional per-point confidence | Add `point_confidence: np.ndarray | None = None` to `Midline2D` in `reconstruction/midline.py`. Locked decision: always an array (fill 1.0s), but stored as optional to allow None-check for absence from older code. Wait — CONTEXT says "Always an array — backends that don't produce per-point confidence fill with 1.0s. No None checks downstream." This means field should default to None but segment_then_extract backend must fill it with 1.0s. |
| CFG-12 | Pipeline runs E2E in both CPU and CUDA device modes, verified by E2E tests | Extend `TestSyntheticSmoke` in `tests/e2e/test_smoke.py` to parameterize over `device: cpu` (CUDA only if available). CPU mode should work without GPU. |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python `dataclasses` | stdlib | Frozen config dataclasses, field extension | Already in use throughout; no migration needed |
| `PyYAML` (`yaml`) | already a dep | YAML loading and ordered-dict dumping | Already used; `yaml.dump()` accepts OrderedDict for field ordering |
| `click` | already a dep | CLI commands | Already used for `run` and `init-config` commands |
| `torch` | already a dep | `torch.cuda.is_available()` for device auto-detect | Lazily importable to avoid import overhead in pure-config paths |
| `numpy` | already a dep | Array fields in `Detection` and `Midline2D` | Already in use in both dataclasses |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pathlib.Path` | stdlib | Path resolution relative to `project_dir` | Path manipulation in `load_config()` |
| `collections.OrderedDict` | stdlib | Field-ordered YAML output for `init-config` | Only in the `init-config` YAML template generation, not in `serialize_config()` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled `_filter_fields()` | Pydantic | Pydantic explicitly ruled out in REQUIREMENTS.md Out of Scope section — frozen dataclasses decided in v2.0 |
| `yaml.dump(sort_keys=False)` for ordering | ruamel.yaml | ruamel.yaml adds a dep; PyYAML with an OrderedDict achieves the same result |

---

## Architecture Patterns

### Current Config Structure (Before Phase 30)

```
PipelineConfig
├── run_id, output_dir, video_dir, calibration_path, mode, n_animals
├── stop_after: str | None
├── detection: DetectionConfig      # has: device, stop_frame (WRONG level)
├── midline: MidlineConfig          # has: n_points = 15 (default)
├── association: AssociationConfig  # has: expected_fish_count (auto-propagated)
├── tracking: TrackingConfig
├── reconstruction: ReconstructionConfig
├── synthetic: SyntheticConfig      # has: fish_count (auto-propagated)
└── lut: LutConfig
```

### Target Config Structure (After Phase 30)

```
PipelineConfig
├── project_dir: str = ""           # NEW: root for relative path resolution
├── run_id, output_dir, video_dir, calibration_path, mode
├── n_animals: int                  # REQUIRED (no default)
├── device: str                     # NEW: auto-detected, propagated to all stages
├── n_sample_points: int = 10       # NEW: top-level, replaces all hardcoded 15s
├── stop_frame: int | None = None   # MOVED from DetectionConfig
├── stop_after: str | None = None
├── detection: DetectionConfig      # device, stop_frame fields REMOVED
├── midline: MidlineConfig          # n_points fed from top-level n_sample_points
├── association: AssociationConfig
├── tracking: TrackingConfig
├── reconstruction: ReconstructionConfig
├── synthetic: SyntheticConfig
└── lut: LutConfig
```

### Pattern 1: Universalized `_filter_fields()` with Strict Reject

**What:** A shared utility (either standalone function or method) that rejects unknown fields with an informative error, rather than silently dropping them.

**Current behavior:** `_filter_fields()` is a closure inside `load_config()`, only called for `AssociationConfig`, `TrackingConfig`, and `LutConfig`. Unknown fields are silently dropped for those three; other configs crash with `TypeError` on unknown fields.

**Target behavior:** All stage configs go through `_filter_fields()`. Unknown fields raise `ValueError` with a message listing the unknown field(s) and, for known renames, a "did you mean X?" hint.

**Implementation options (Claude's discretion):**

Option A — Standalone function with a rename registry:
```python
_FIELD_RENAMES: dict[str, str] = {
    "expect_fish_count": "n_animals",
}

def _filter_fields(
    dc_type: type,
    kwargs: dict[str, Any],
    rename_hints: dict[str, str] | None = None,
) -> dict[str, Any]:
    valid = {f.name for f in dataclasses.fields(dc_type)}
    unknown = [k for k in kwargs if k not in valid]
    if unknown:
        hints = rename_hints or _FIELD_RENAMES
        msgs = []
        for k in unknown:
            if k in hints:
                msgs.append(f"unknown field {k!r} — did you mean {hints[k]!r}?")
            else:
                msgs.append(f"unknown field {k!r}")
        raise ValueError(
            f"{dc_type.__name__}: " + "; ".join(msgs)
        )
    return {k: v for k, v in kwargs.items() if k in valid}
```

Option B — Per-call with an inline rename dict. Simpler but no central registry.

**Recommendation:** Option A — standalone function at module level. Easier to test and extend.

### Pattern 2: Device Auto-Detection

**What:** `device` on `PipelineConfig` is optional in YAML (not a required field). When absent, auto-detect using torch.

**Implementation:**
```python
def _default_device() -> str:
    """Auto-detect compute device."""
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
```

Call `_default_device()` in `load_config()` when `device` is not present in YAML or CLI overrides. Store the resolved value on `PipelineConfig.device`.

**Important:** `PipelineConfig` must have `device` as a field with a dataclass default — `dataclasses.field(default_factory=_default_device)` is the right pattern for a computed default.

### Pattern 3: `n_sample_points` Propagation

**What:** `PipelineConfig.n_sample_points` (default 10) must feed into every module that currently hard-codes 15.

**Affected modules and the fix:**
1. `engine/config.py`: Add `n_sample_points: int = 10` to `PipelineConfig`. Feed it to `MidlineConfig.n_points` (or pass directly).
2. `engine/pipeline.py` (`build_stages()`): Pass `config.n_sample_points` to `MidlineStage(n_points=...)`, `ReconstructionStage(n_sample_points=...)`, and `SyntheticDataStage(n_points=...)`.
3. `reconstruction/triangulation.py`: `N_SAMPLE_POINTS = 15` is a module constant imported elsewhere. It should remain as a module-level fallback constant but the actual call sites must prefer the passed-in value. The comment "matches Phase 6 output" is stale — update to "default; prefer config.n_sample_points".
4. `core/synthetic.py`: `n_points = 15` at line 166 is hardcoded. Must read from `self._config.n_points` or be passed in.
5. `visualization/midline_viz.py`, `visualization/triangulation_viz.py`: These use `15` in inline comparisons (`if n_skel < 15`) and calls (`_resample_arc_length(..., 15)`). These are visualization utilities — they can keep the literal 15 as a local default since they are not part of the pipeline config chain. Flag these as acceptable (visualization is not a config-driven stage).
6. `io/midline_writer.py`: Uses `N_SAMPLE_POINTS` imported from `triangulation.py` to size HDF5 datasets. This is the trickiest: the writer creates fixed-size arrays at dataset creation time. Must accept `n_sample_points` at construction time.

### Pattern 4: `init-config` Project Directory Scaffolding

**What:** Replace the current `init-config` (writes a YAML file to a path) with a project-directory scaffolding command.

**Current signature:** `aquapose init-config [-o OUTPUT] [--force]`
**New signature:** `aquapose init-config <name> [--synthetic]`

**Existing tests in `test_cli.py::TestInitConfig`** test the old behavior (no positional arg, `--output` flag). These tests will need updating. Key tests: `test_init_config_creates_default_file`, `test_init_config_custom_output`, `test_init_config_refuses_overwrite`, `test_init_config_force_overwrites`.

**New behavior:**
1. Resolve `~/aquapose/projects/<name>/` using `pathlib.Path("~/aquapose/projects").expanduser() / name`
2. Create `runs/`, `models/`, `geometry/`, `videos/` subdirectories
3. Write `config.yaml` using an ordered template (NOT `dataclasses.asdict()` + `yaml.dump(sort_keys=True)`)
4. If `--synthetic`, include `synthetic:` section in YAML

**Ordered YAML generation:** Use `ruamel.yaml` or `yaml.dump` with a dict subclass that preserves insertion order. Since Python 3.7+ dicts preserve insertion order and `yaml.dump` with `default_flow_style=False, sort_keys=False` will respect insertion order, this is straightforward.

### Pattern 5: Path Resolution via `project_dir`

**What:** `project_dir` is a new top-level field. Relative paths in YAML (`video_dir`, `calibration_path`, `output_dir`, `detection.model_path`, `midline.weights_path`) are resolved relative to `project_dir` in `load_config()`.

**Implementation in `load_config()`:**
```python
project_dir_str = top_kwargs.get("project_dir", "")
if project_dir_str:
    project_dir = Path(project_dir_str).expanduser().resolve()
    for key in ("video_dir", "calibration_path", "output_dir"):
        if key in top_kwargs and not Path(top_kwargs[key]).is_absolute():
            top_kwargs[key] = str(project_dir / top_kwargs[key])
    # Also for sub-config path fields
    for kwargs, field_names in [
        (det_kwargs, ["model_path"]),
        (mid_kwargs, ["weights_path"]),
    ]:
        for fname in field_names:
            if fname in kwargs and kwargs[fname] and not Path(kwargs[fname]).is_absolute():
                kwargs[fname] = str(project_dir / kwargs[fname])
```

### Anti-Patterns to Avoid

- **Modifying `serialize_config()`:** The `sort_keys=True` in `serialize_config()` is intentional for reproducibility of run artifacts. Only the `init-config` template generation needs ordered output. Do not change `serialize_config()`.
- **Adding `n_sample_points` to `ReconstructionConfig` instead of top-level:** The value must be top-level because it couples the output of `MidlineStage` with the input of `ReconstructionStage`. If it lives in `ReconstructionConfig`, MidlineStage wouldn't see it.
- **Making `Detection` and `Midline2D` frozen:** These are runtime data containers (not config), they are non-frozen by design. `angle`, `obb_points`, and `point_confidence` should stay non-frozen.
- **Requiring `obb_points` when `angle` is set (or vice versa):** Both are independently optional `None` for non-OBB detectors. No co-validation needed.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ordered YAML output | Custom YAML serializer | `yaml.dump(sort_keys=False)` on an ordered dict | Python 3.7+ dict insertion order + PyYAML sort_keys=False |
| Device detection | Custom CUDA check | `torch.cuda.is_available()` | Handles all CUDA cases including multi-GPU |
| Path expansion | Custom string manipulation | `Path(...).expanduser().resolve()` | Handles `~`, relative paths, symlinks |

---

## Common Pitfalls

### Pitfall 1: `n_animals` Has No Default — Required Validation
**What goes wrong:** `PipelineConfig.n_animals` is currently `int = 9` (has a default). The locked decision is to make it required (no default). However, frozen dataclasses require field ordering: fields without defaults must precede fields with defaults. This means `n_animals` (required, no default) must come before all optional fields.
**Why it happens:** Python dataclass inheritance rules — required fields cannot follow optional fields.
**How to avoid:** Order fields carefully. Move `n_animals` early in the field list. Use `dataclasses.field()` with no default and add a `__post_init__` validation that raises `ValueError` if `n_animals <= 0`, or enforce it in `load_config()` after YAML loading.
**Warning signs:** `TypeError: non-default argument follows default argument` at class definition time.

**Note on implementation:** The cleaner approach is to keep `n_animals` as a field with a sentinel default (e.g., `n_animals: int = 0`) and validate in `load_config()` — since `PipelineConfig()` is never called directly by users (always through `load_config()`), validation in the factory is correct. This avoids dataclass field ordering constraints.

### Pitfall 2: `_filter_fields()` Strict Mode Breaks `load_config()` on `**top_kwargs`
**What goes wrong:** `top_kwargs` currently passes `**top_kwargs` directly to `PipelineConfig(...)` at the end of `load_config()`. If `_filter_fields()` is applied to all sub-configs but not to `top_kwargs`, unknown top-level keys will still crash with `TypeError`.
**How to avoid:** Apply `_filter_fields(PipelineConfig, top_kwargs)` before passing `**top_kwargs` to the constructor. Be careful: `top_kwargs` at that point has already had `run_id` and `output_dir` popped. The filter will catch truly unknown keys.

### Pitfall 3: `DetectionConfig.device` and `DetectionConfig.stop_frame` Must Be Removed
**What goes wrong:** When `device` and `stop_frame` move to top-level `PipelineConfig`, the old fields must be removed from `DetectionConfig`. Any existing YAML files that have `detection.device` or `detection.stop_frame` will hit the strict `_filter_fields()` and raise an error. The decision is: `detection.stop_frame` is a renamed/moved field — treat with a hint: "did you mean top-level `stop_frame`?".
**How to avoid:** Add rename hints for `detection.stop_frame` → `stop_frame` (top-level) and `detection.device` → `device` (top-level).

### Pitfall 4: `MidlineConfig.n_points` vs `PipelineConfig.n_sample_points`
**What goes wrong:** `MidlineConfig.n_points` (the sub-config field) must be kept in sync with `PipelineConfig.n_sample_points`. In `load_config()`, if the user sets `midline.n_points` in YAML, that should be respected over the top-level `n_sample_points`. If neither is set, use `n_sample_points` default (10).
**How to avoid:** In `load_config()`, after YAML parsing: if `"n_points"` not in `mid_kwargs`, set `mid_kwargs["n_points"] = top_kwargs.get("n_sample_points", 10)`. The sub-config field name `n_points` stays as `n_points` (not renamed) since it's a per-stage config detail.

### Pitfall 5: `Midline2D.point_confidence` Must Be Set by `SegmentThenExtractBackend`
**What goes wrong:** The locked decision says "Always an array — backends that don't produce per-point confidence fill with 1.0s." If `SegmentThenExtractBackend` creates `Midline2D(...)` without setting `point_confidence`, the field remains `None`. All downstream code that uses `point_confidence` as triangulation weights would need a None check.
**How to avoid:** When adding `point_confidence` to `Midline2D`, update all call sites in `SegmentThenExtractBackend` that construct `Midline2D` to populate `point_confidence = np.ones(n_points, dtype=np.float32)`. Also update `core/synthetic.py` which constructs `Midline2D` objects.
**Warning signs:** Check all `Midline2D(...)` construction sites:
- `reconstruction/midline.py` (line ~440, inside `MidlineExtractor.extract_midlines`)
- `core/midline/backends/segment_then_extract.py`
- `core/synthetic.py`

### Pitfall 6: `init-config` CLI Signature Change Breaks Existing Tests
**What goes wrong:** `test_cli.py::TestInitConfig` tests the old interface (`--output`, no positional arg). Changing to `init-config <name>` breaks all four tests.
**How to avoid:** Rewrite the `TestInitConfig` tests as part of this phase. New tests: `test_init_config_creates_project_directory`, `test_init_config_creates_subdirectories`, `test_init_config_with_synthetic_flag`, `test_init_config_yaml_has_correct_field_order`, `test_init_config_refuses_existing_dir`.

### Pitfall 7: `SyntheticConfig.fish_count` Removal
**What goes wrong:** `SyntheticConfig` has `fish_count: int = 3`. The locked decision says the synthetic section should NOT include its own fish count. But `SyntheticConfig.fish_count` is referenced throughout the codebase (`core/synthetic.py` uses `self._config.fish_count`).
**Resolution:** Do NOT remove `SyntheticConfig.fish_count`. It remains as an internal field that gets auto-populated from `n_animals` in `load_config()` (this already happens today). What changes is that the YAML generated by `init-config --synthetic` does NOT include `synthetic.fish_count` — users set `n_animals` at the top level and the loader propagates it. The `--synthetic` YAML section only shows fields specific to synthetic mode: `frame_count`, `noise_std`, `seed`.

---

## Code Examples

### Device Auto-Detection Pattern
```python
# Source: Python stdlib + torch API
def _default_device() -> str:
    """Return the best available compute device."""
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# In PipelineConfig:
device: str = dataclasses.field(default_factory=_default_device)
```

### Strict `_filter_fields()` with Rename Hints
```python
# Source: project code pattern, extended
_RENAME_HINTS: dict[str, str] = {
    "expect_fish_count": "n_animals",       # old AssociationConfig field
    "stop_frame": "stop_frame (top-level)", # moved from detection sub-config
    "device": "device (top-level)",         # moved from detection sub-config
}

def _filter_fields(
    dc_type: type,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    valid = {f.name for f in dataclasses.fields(dc_type)}
    unknown = [k for k in kwargs if k not in valid]
    if unknown:
        msgs = []
        for k in unknown:
            if k in _RENAME_HINTS:
                msgs.append(f"{k!r} — did you mean {_RENAME_HINTS[k]!r}?")
            else:
                msgs.append(f"unknown field {k!r}")
        raise ValueError(
            f"Config error in {dc_type.__name__}: " + "; ".join(msgs)
        )
    return {k: v for k, v in kwargs.items() if k in valid}
```

### Ordered YAML for `init-config` Template
```python
# Source: PyYAML + Python dict insertion order
import yaml

def _make_yaml_template(name: str, include_synthetic: bool) -> str:
    data: dict = {}
    data["project_dir"] = f"~/aquapose/projects/{name}"
    data["video_dir"] = "videos/"
    data["calibration_path"] = "geometry/calibration.json"
    data["output_dir"] = "runs/"
    data["n_animals"] = "???"  # required — must be set
    data["detection"] = {"detector_kind": "yolo", "model_path": "models/best.pt"}
    data["midline"] = {"weights_path": "models/unet_best.pth"}
    if include_synthetic:
        data["synthetic"] = {"frame_count": 30, "noise_std": 0.0, "seed": 42}
    return yaml.dump(data, default_flow_style=False, sort_keys=False)
```

### Detection Dataclass Extension
```python
# Source: aquapose/segmentation/detector.py (current + additions)
@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None
    area: int
    confidence: float
    # Phase 30 additions (CFG-10):
    angle: float | None = None          # radians, standard math convention
    obb_points: np.ndarray | None = None  # shape (4, 2), clockwise from top-left
```

### Midline2D Dataclass Extension
```python
# Source: aquapose/reconstruction/midline.py (current + additions)
@dataclass
class Midline2D:
    points: np.ndarray       # (N, 2), float32
    half_widths: np.ndarray  # (N,), float32
    fish_id: int
    camera_id: str
    frame_index: int
    is_head_to_tail: bool = False
    # Phase 30 addition (CFG-11):
    point_confidence: np.ndarray | None = None  # (N,), float32, [0, 1]
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `sort_keys=True` in `serialize_config()` | Keep for run artifacts | N/A | No change — sorted is correct for reproducibility |
| `_filter_fields()` silently drops unknown keys | Strict reject with hints | Phase 30 | Users get actionable errors instead of silent config drift |
| `device` per-stage in `DetectionConfig` | Top-level `device` on `PipelineConfig` | Phase 30 | One parameter to rule them all; no per-stage overrides needed |
| Hardcoded `N_SAMPLE_POINTS = 15` | Configurable `n_sample_points = 10` | Phase 30 | Unlocks CFG-02, unblocks EVAL-01 regression tests |

---

## Open Questions

1. **`n_animals` required-vs-sentinel implementation**
   - What we know: Making it a true required field (no default) breaks dataclass field ordering rules if any field before it has a default.
   - What's unclear: Whether to use sentinel value approach (default `n_animals: int = 0`, validate in `load_config()`) or move to end of required-field group.
   - Recommendation: Use sentinel `0` and validate in `load_config()` — cleanest for the frozen dataclass pattern. Add `if top_kwargs.get("n_animals", PipelineConfig.n_animals) == 0: raise ValueError("n_animals is required")`.

2. **`MidlineConfig.n_points` vs `PipelineConfig.n_sample_points` naming**
   - What we know: Sub-config uses `n_points`, top-level will use `n_sample_points`. These must be kept in sync by `load_config()`.
   - What's unclear: Whether `MidlineConfig.n_points` should be renamed to `n_sample_points` for consistency.
   - Recommendation: Keep `MidlineConfig.n_points` as-is (it's the per-stage name, matches `MidlineExtractor` constructor). Auto-propagate `n_sample_points` → `n_points` in `load_config()` when `n_points` is not explicitly set.

3. **`io/midline_writer.py` HDF5 dataset sizing**
   - What we know: `MidlineWriter` uses `N_SAMPLE_POINTS` imported from `triangulation.py` to size HDF5 datasets at creation time (shape `(max_fish, N_SAMPLE_POINTS)`).
   - What's unclear: Whether `MidlineWriter` needs to accept `n_sample_points` as a constructor argument (breaking change to `MidlineWriter.__init__`).
   - Recommendation: Add `n_sample_points: int = N_SAMPLE_POINTS` parameter to `MidlineWriter.__init__()`. `build_stages()` (or the `HDF5ExportObserver`) passes `config.n_sample_points`. The module-level constant `N_SAMPLE_POINTS` becomes the default fallback.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `hatch run test` |
| Full suite command | `hatch run test-all` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CFG-01 | `config.device` auto-detected and propagated to all stages | unit | `pytest tests/unit/engine/test_config.py -x` | ✅ (extend) |
| CFG-02 | `n_sample_points=10` used throughout, no hardcoded 15 remaining | unit | `pytest tests/unit/engine/test_config.py tests/unit/test_midline.py -x` | ✅ (extend) |
| CFG-03 | `n_animals` propagates to `expected_fish_count` and `fish_count` | unit | `pytest tests/unit/engine/test_config.py -x` | ✅ (already tested) |
| CFG-04 | `stop_frame` at top-level propagates to DetectionStage | unit | `pytest tests/unit/engine/test_config.py tests/unit/engine/test_build_stages.py -x` | ✅ (extend) |
| CFG-05 | Unknown YAML fields raise with hint, not silently drop | unit | `pytest tests/unit/engine/test_config.py::test_unknown_field_raises -x` | ❌ Wave 0 |
| CFG-06 | `init-config <name>` creates directory scaffold | unit | `pytest tests/unit/engine/test_cli.py::TestInitConfig -x` | ✅ (rewrite) |
| CFG-07 | Relative paths resolve from `project_dir` | unit | `pytest tests/unit/engine/test_config.py::test_project_dir_path_resolution -x` | ❌ Wave 0 |
| CFG-08 | `--synthetic` flag adds synthetic section to YAML | unit | `pytest tests/unit/engine/test_cli.py::TestInitConfig::test_synthetic_flag -x` | ❌ Wave 0 |
| CFG-09 | YAML field order matches user-relevance order | unit | `pytest tests/unit/engine/test_cli.py::TestInitConfig::test_yaml_field_order -x` | ❌ Wave 0 |
| CFG-10 | `Detection.angle` and `Detection.obb_points` fields exist and default None | unit | `pytest tests/unit/segmentation/test_detector.py -x` | ✅ (extend) |
| CFG-11 | `Midline2D.point_confidence` exists; segment_then_extract fills 1.0s | unit | `pytest tests/unit/test_midline.py tests/unit/core/midline/test_midline_stage.py -x` | ✅ (extend) |
| CFG-12 | Synthetic pipeline E2E runs on CPU | e2e | `pytest tests/e2e/test_smoke.py::TestSyntheticSmoke -x` | ✅ (extend) |

### Sampling Rate
- **Per task commit:** `hatch run test`
- **Per wave merge:** `hatch run test`
- **Phase gate:** `hatch run test` full green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/engine/test_config.py::test_unknown_field_raises` — covers CFG-05 strict reject
- [ ] `tests/unit/engine/test_config.py::test_rename_hint_in_error` — covers CFG-05 hint message
- [ ] `tests/unit/engine/test_config.py::test_project_dir_path_resolution` — covers CFG-07
- [ ] `tests/unit/engine/test_cli.py::TestInitConfig` — full rewrite of 4 existing tests + 3 new tests for CFG-06, CFG-08, CFG-09

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `src/aquapose/engine/config.py`, `pipeline.py`, `cli.py`, `segmentation/detector.py`, `reconstruction/midline.py`, `reconstruction/triangulation.py`, `core/synthetic.py`, `io/midline_writer.py`
- Test files — `tests/unit/engine/test_config.py`, `tests/unit/engine/test_cli.py`, `tests/e2e/test_smoke.py`
- `.planning/phases/30-config-and-contracts/30-CONTEXT.md` — locked decisions
- `.planning/REQUIREMENTS.md` — CFG-01 through CFG-12 descriptions

### Secondary (MEDIUM confidence)
- Python dataclasses docs (stdlib knowledge) — field ordering rules, `dataclasses.field(default_factory=...)` patterns
- PyYAML `sort_keys=False` behavior (stdlib knowledge) — respects dict insertion order in Python 3.7+

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are already in use; no new dependencies required
- Architecture: HIGH — based on direct code reading of all affected modules
- Pitfalls: HIGH — identified from concrete code locations and dataclass field ordering rules

**Research date:** 2026-02-28
**Valid until:** 2026-03-30 (codebase is stable; no fast-moving external dependencies)
