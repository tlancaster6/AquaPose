# Phase 85: Code Quality Audit & CLI Smoke Test - Research

**Researched:** 2026-03-11
**Domain:** Python static analysis, dead code removal, type error triage, CLI validation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**BoxMot Removal**
- Remove BoxMot entirely — delete the dependency from pyproject.toml, the ocsort_wrapper module, and its test file (test_ocsort_wrapper.py)
- Remove all config keys related to OC-SORT/BoxMot (tracker_kind "ocsort" option, iou_threshold for ocsort, boxmot references in docstrings)
- Clean break — no backward-compatible aliases, no fallback
- Verify no orphaned transitive dependencies remain after removal (run dependency check)

**Type Error Triage**
- Fix all 25 type errors — not just v3.7 regressions, bring the count to zero
- Fix the LutConfig protocol mismatch properly (align the LutConfigLike protocol with the frozen dataclass pattern)
- Use targeted `# type: ignore[attr]` comments for third-party stub gaps (cv2.VideoWriter_fourcc etc.)
- No CI enforcement gate — just fix the current batch; pre-commit hooks already run typecheck

**CLI Smoke Test**
- Use YH project config with chunk-size=100, max-chunks=2 (real data, exercises chunk plumbing)
- Success = exit code 0 AND expected output artifacts exist (run directory, frozen config, stage outputs)
- Verify the new tracker's CLI flags parse correctly (the 6 tunable params from Phase 84.1)
- Tracker params stay as YAML config only — do NOT add individual CLI flags to `aquapose run`
- Tracker config documentation is satisfied by existing TrackingConfig docstrings — no separate doc file

**Audit Reporting**
- Fix inline + write audit report to phase directory (85-AUDIT-REPORT.md)
- Full dead code scan across the entire codebase, not just v3.7 remnants
- Fix obviously dead code (unused imports, unreachable functions) inline if safe
- Document ambiguous findings for Phase 86 review
- Report structure: what was found, what was fixed, what remains for Phase 86

### Claude's Discretion
- Dead code scanning approach (manual inspection, vulture, or combination)
- Exact order of audit tasks (type fixes, dead code removal, BoxMot removal, smoke test)
- Report format and level of detail

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INTEG-03 | Code quality audit: no dead code, broken cross-references, or type errors from the overhaul | Inventory of all 25 type errors by file/fix strategy; vulture scan results; complete list of ocsort/boxmot reference sites |
| INTEG-04 | BoxMot dependency removal decision documented | Decision locked (remove); all removal sites catalogued |
</phase_requirements>

---

## Summary

Phase 85 is a quality gate, not a feature phase. The v3.7 overhaul (segmentation removal, pipeline reorder, custom tracker, association upgrade) introduced a known 25-error type backlog and left BoxMot as an unresolved dependency. This phase closes both issues and confirms the full pipeline runs cleanly.

The type errors divide into five distinct fix strategies: (1) protocol alignment for LutConfigLike (make fields `Final` or annotate as `@property`), (2) `# type: ignore[attr]` for cv2 stubs, (3) narrow the `context.get()` return type in AssociationStage, (4) fix orchestrator/pipeline attribute access on `object` types, and (5) address YOLO private import usage. Dead code scanning with vulture (run at 80% confidence) reveals four legitimate local-variable leaks and a large population of 60%-confidence false positives (CLI click commands registered dynamically, intentional synthetic stubs). The BoxMot removal touch list is well-contained: 6 source files plus 2 test files.

The CLI smoke test uses `aquapose run --config <YH_config> --set chunk_size=100 --max-chunks 2` with exit-code-0 plus artifact presence as the success criterion. The YH config at `~/aquapose/projects/YH/config.yaml` already has production models and the correct tracker settings; only chunk_size needs overriding via `--set`.

**Primary recommendation:** Run BoxMot removal first (eliminates entire ocsort code path), then fix type errors (easier with ocsort gone), then run vulture audit, then smoke test.

---

## Standard Stack

### Core (already present in project)
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| basedpyright | (hatch env) | Static type checking | Project toolchain — `hatch run typecheck` |
| ruff | (hatch env) | Lint + unused import detection (F401) | Project toolchain — `hatch run lint` |
| vulture | install on demand | Dead code detection (unreachable functions, unused variables) | Best Python dead-code scanner; fast; 60-100% confidence levels |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `hatch run test` | Run unit tests excluding slow/e2e | After each destructive change (BoxMot removal, type fixes) to confirm no regressions |
| `hatch run check` | lint + typecheck combined | Final gate before smoke test |

**Installation (vulture — not in pyproject.toml, install in hatch env for audit only):**
```bash
hatch run pip install vulture
hatch run vulture src/aquapose/ --min-confidence 80
```

---

## Architecture Patterns

### Type Error Catalogue (25 errors, 8 files)

Run: `hatch run typecheck` to verify. Current error locations and fix strategies:

#### `src/aquapose/cli.py` (5 errors, lines 641-650)
h5py `Group | Dataset | Datatype` union — `__delitem__`, `__getitem__`, `__setitem__`, `create_dataset` not on all variants. Fix: narrow the union with `isinstance(group, h5py.Group)` guard before operations.

#### `src/aquapose/core/association/stage.py` (3 errors, lines 89, 97, 98)
`context.get("tracks_2d")` returns `object`. Fix: cast the result — `tracks_2d = cast("dict[str, list[Tracklet2D]]", context.get("tracks_2d"))` — or add a typed accessor method to `PipelineContext`.

#### `src/aquapose/core/detection/backends/yolo.py` (1 error, line 86)
`YOLO` is not exported from the top-level `ultralytics` module. Fix: `from ultralytics.models import YOLO` (correct export path) or `# type: ignore[attr]` if the lazy import pattern must remain.

#### `src/aquapose/core/detection/backends/yolo_obb.py` (1 error, line 104)
Same YOLO import issue. Same fix.

#### `src/aquapose/core/pose/backends/pose_estimation.py` (3 errors, lines 89, 263, 265)
YOLO import + `result.keypoints` attribute access on `object`. Fix: YOLO import as above; for keypoints, cast the result to the correct Ultralytics result type.

#### `src/aquapose/core/reconstruction/stage.py` (1 error, line 312)
`backend.reconstruct_frame` not on `object`. Fix: type the `_backend` attribute with a Protocol or Union of concrete backends.

#### `src/aquapose/engine/orchestrator.py` (2 errors, lines 232, 252)
`object | VideoFrameSource` is not `Sized`. Fix: assert or cast to `VideoFrameSource` after isinstance check; the `object` comes from an untyped context attribute.

#### `src/aquapose/engine/pipeline.py` (3 errors, lines 366, 371, 431)
- Lines 366/371: LutConfigLike protocol mismatch. The frozen dataclass `LutConfig` has writable fields in the protocol. **Fix**: add `Final` annotations to protocol fields using `@property` style or annotate the protocol fields as `Final[float]`.
- Line 431: `tuple(config.detection.crop_size)` returns `tuple[int, ...]` but PoseStage expects `tuple[int, int]`. Fix: `cast(tuple[int, int], tuple(config.detection.crop_size))`.

#### `src/aquapose/evaluation/viz/overlay.py` (1 error, line 433)
`cv2.VideoWriter_fourcc` not in stubs. Fix: `# type: ignore[attr]`.

#### `src/aquapose/evaluation/viz/trails.py` (2 errors, lines 625, 702)
Same cv2 stub issue. Fix: `# type: ignore[attr]` on each line.

#### `src/aquapose/logging.py` (1 error, line 88)
`exc_info=(exc_type, exc_value, exc_tb)` — `exc_tb` is typed as `object` but stdlib expects `TracebackType | None`. Fix: `# type: ignore[arg-type]` (already has it on the wrong line — move to line 88), or cast `exc_tb` to `TracebackType | None`.

### LutConfigLike Protocol Fix — Recommended Pattern

The `LutConfigLike` protocol in `calibration/luts.py` declares writable fields. basedpyright infers that a `frozen=True` dataclass (whose fields are read-only) does not satisfy a protocol with writable fields.

Fix: make the protocol fields read-only by converting them to `@property` abstract members in the Protocol:

```python
# Source: basedpyright protocol compatibility rules
@runtime_checkable
class LutConfigLike(Protocol):
    @property
    def tank_diameter(self) -> float: ...
    @property
    def tank_height(self) -> float: ...
    @property
    def voxel_resolution_m(self) -> float: ...
    @property
    def margin_fraction(self) -> float: ...
    @property
    def forward_grid_step(self) -> int: ...
```

A frozen dataclass satisfies a protocol with read-only `@property` members because frozen fields are effectively read-only. This is the correct fix — not adding `@property` to the dataclass itself.

### BoxMot Removal Touch List

Complete list of files requiring changes (confirmed by grep):

**Files to delete:**
- `src/aquapose/core/tracking/ocsort_wrapper.py`
- `tests/unit/tracking/test_ocsort_wrapper.py`

**Files requiring edits:**
| File | What to Change |
|------|----------------|
| `pyproject.toml` | Remove `"boxmot>=11.0"` from dependencies |
| `src/aquapose/engine/config.py` | Remove `"ocsort"` from `valid_kinds`, remove `iou_threshold` field, remove all "boxmot"/"ocsort" docstring references |
| `src/aquapose/core/tracking/stage.py` | Remove the `else` branch (OcSortTracker path), update module docstring, remove ocsort reference in default fallback at line 98 |
| `src/aquapose/core/tracking/__init__.py` | Update module docstring (remove OC-SORT mention) |
| `src/aquapose/core/context.py` | Update `tracks_2d_state` field comment (lines 189, 201) — remove OcSortTracker reference |
| `src/aquapose/synthetic/stubs.py` | Update docstring (lines 7, 61, 65) — remove OcSortTracker mention |
| `src/aquapose/synthetic/detection.py` | Update docstring (line 5, 150) — remove OcSortTracker mention |
| `tests/unit/core/tracking/test_keypoint_tracker.py` | Remove `test_config_ocsort_still_valid` (line 752) and `test_tracking_stage_ocsort_unchanged` (line 877); remove `tracker_kind: str = "ocsort"` from fixture at line 34 |
| `tests/unit/core/tracking/test_tracking_stage.py` | Remove `tracker_kind: str = "ocsort"` default (line 34) |

### Dead Code Findings

**vulture at 80% confidence — 4 legitimate issues to fix:**
| Location | Finding | Action |
|----------|---------|--------|
| `evaluation/viz/overlay.py:302` | `show_fish_id` variable set but never used | Remove assignment |
| `io/midline_writer.py:230` | `exc_val` variable in exception handler never used | Replace with `_` |
| `synthetic/stubs.py:45` | `args` variable never used | Replace with `_` |
| `training/data_cli.py:76` | `augment_count` variable set but never used | Remove or wire up |

**vulture at 60% confidence — documented for Phase 86 review (do NOT fix inline):**
The 60% findings are nearly all false positives due to vulture's inability to detect:
- Click CLI commands registered via `@cli.command()` decorators
- Protocol/ABC method stubs
- Dataclass fields used via attribute access
- Methods called dynamically (e.g., `get_state()`, `from_state()`)

Notable exceptions worth Phase 86 investigation:
- `evaluation/viz/overlay.py:140` — `_reproject_3d_midline` unused function (may be dead)
- `core/reconstruction/backends/dlt.py:671` — `_triangulate_body_point` unused method
- `core/tracking/types.py` — multiple `FishTrack`/`TrackState` members (types.py was OC-SORT era; may be obsolete)
- `synthetic/trajectory.py` — `loose_school`, `milling`, `streaming` methods, `FishTrajectoryState` class (synthetic module may be partially orphaned)

### CLI Smoke Test Pattern

```bash
# Smoke test invocation (override chunk_size, limit to 2 chunks)
aquapose --config ~/aquapose/projects/YH/config.yaml \
  run \
  --set chunk_size=100 \
  --max-chunks 2

# Success criteria check
echo "Exit code: $?"
# Then check for run directory + frozen config artifact
ls ~/aquapose/projects/YH/runs/ | tail -3
```

**Artifact structure to verify:**
- `runs/run_YYYYMMDD_HHMMSS/` directory created
- `runs/run_.../config_frozen.yaml` exists (frozen config written by orchestrator)
- `runs/run_.../chunk_000/` and `chunk_001/` directories exist with stage outputs

**Tracker config YAML keys to verify parse correctly (6 params from Phase 84.1):**
```yaml
tracking:
  tracker_kind: keypoint_bidi
  max_coast_frames: 30
  base_r: 10.0
  lambda_ocm: 0.2
  match_cost_threshold: 1.2
  ocr_threshold: 0.5
```

The YH config at `~/aquapose/projects/YH/config.yaml` does not currently have a `tracking:` section — the defaults from `TrackingConfig` apply. The smoke test should verify these defaults load without error; no YAML edit required unless testing a specific override.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Finding dead code | Manual grep for unused imports | `vulture` | Handles class methods, unused attributes, unreachable branches |
| Finding unused imports | Manual inspection | `ruff check --select F401` | Already in project; zero false positives for imports |
| Protocol compatibility analysis | Manual field comparison | basedpyright error messages | Already identifies exactly which fields are incompatible |

---

## Common Pitfalls

### Pitfall 1: vulture 60% false positives
**What goes wrong:** Treating all vulture findings as dead code and deleting active CLI commands registered via Click decorators.
**Why it happens:** vulture cannot detect dynamic dispatch — Click registers commands by decorating functions, not calling them.
**How to avoid:** Only fix 80%+ confidence findings inline. Document 60% findings in the audit report for human review.
**Warning signs:** Deleting a function named `*_cmd` or `*_group` in cli.py.

### Pitfall 2: Incomplete BoxMot removal breaking test suite
**What goes wrong:** Deleting ocsort_wrapper.py but leaving `tracker_kind="ocsort"` tests that import it, causing import errors at test collection.
**Why it happens:** There are two separate test locations — `tests/unit/tracking/test_ocsort_wrapper.py` AND backward-compat tests in `tests/unit/core/tracking/test_keypoint_tracker.py` (lines 752 and 877).
**How to avoid:** Search for ALL ocsort references in tests before deleting the module.
**Warning signs:** `ImportError: cannot import name 'OcSortTracker'` during `hatch run test`.

### Pitfall 3: Protocol fix breaking runtime_checkable isinstance checks
**What goes wrong:** Converting LutConfigLike to `@property` style breaks `isinstance(obj, LutConfigLike)` checks if any exist.
**Why it happens:** `@runtime_checkable` Protocol with property members works differently than with plain attributes.
**How to avoid:** Search for `isinstance.*LutConfigLike` before changing the protocol. (Grep confirms: no isinstance checks exist — the protocol is only used as a type annotation.)

### Pitfall 4: iou_threshold removal breaking downstream configs
**What goes wrong:** Removing `iou_threshold` from `TrackingConfig` causes YAML parse failures for any saved configs that include `tracking.iou_threshold`.
**Why it happens:** Frozen dataclass `__init__` rejects unknown keyword args by default.
**How to avoid:** Check if `iou_threshold` is filtered by the YAML loader's `_filter_fields()` function. It is — `_filter_fields` drops unknown keys. So removal is safe for existing YAMLs. Remove the field cleanly.

### Pitfall 5: Smoke test exits 0 but artifacts missing
**What goes wrong:** Treating exit code 0 as sufficient smoke test pass.
**Why it happens:** Some code paths may silently swallow errors and write no output.
**How to avoid:** Explicitly check for run directory + `config_frozen.yaml` presence after the run.

### Pitfall 6: cv2 type: ignore placement
**What goes wrong:** Adding `# type: ignore` on the wrong line (e.g., the import line rather than the usage line).
**Why it happens:** Inattention. The error is on `cv2.VideoWriter_fourcc(...)` call sites, not the `import cv2` statement.
**How to avoid:** Place `# type: ignore[attr]` on the same line as the `cv2.VideoWriter_fourcc` call.

---

## Code Examples

### LutConfigLike protocol fix
```python
# calibration/luts.py
# Source: basedpyright documentation on protocol read-only properties
@runtime_checkable
class LutConfigLike(Protocol):
    """Structural protocol for LUT configuration objects (read-only view).

    Satisfied by frozen dataclasses with these fields without explicit
    inheritance. @property members match frozen dataclass fields because
    both are read-only from the caller's perspective.
    """

    @property
    def tank_diameter(self) -> float: ...
    @property
    def tank_height(self) -> float: ...
    @property
    def voxel_resolution_m(self) -> float: ...
    @property
    def margin_fraction(self) -> float: ...
    @property
    def forward_grid_step(self) -> int: ...
```

### AssociationStage tracks_2d narrowing
```python
# core/association/stage.py — after context.get() call
from typing import cast
from aquapose.core.tracking.types import Tracklet2D

tracks_2d = cast("dict[str, list[Tracklet2D]]", context.get("tracks_2d"))
```

### cv2 type: ignore pattern
```python
# evaluation/viz/overlay.py
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr]
```

### Smoke test artifact verification
```python
# Pattern for checking smoke test success
import subprocess, pathlib, yaml, datetime

result = subprocess.run(
    ["aquapose", "--config", str(config_path), "run",
     "--set", "chunk_size=100", "--max-chunks", "2"],
    capture_output=True, text=True
)
assert result.returncode == 0, f"aquapose run failed:\n{result.stderr}"

runs_dir = pathlib.Path("~/aquapose/projects/YH/runs").expanduser()
run_dirs = sorted(runs_dir.glob("run_*"))
latest = run_dirs[-1]
assert (latest / "config_frozen.yaml").exists(), "No frozen config written"
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| OC-SORT via BoxMot as primary tracker | Custom KeypointTracker (keypoint_bidi) | Phase 83 | BoxMot no longer needed |
| Segmentation backend registered for use | Segmentation backend removed from registry | Phase 81 | Only docstring references remain |
| 25 type errors in backlog | Target: 0 errors | Phase 85 | Clean baseline for future development |

---

## Open Questions

1. **Does `iou_threshold` need a migration shim in the YAML loader?**
   - What we know: `_filter_fields()` in `engine/config.py` drops unknown keys during YAML load, so existing configs with `tracking.iou_threshold` will silently ignore the field after removal.
   - What's unclear: Whether any currently active configs explicitly set this field.
   - Recommendation: Safe to remove cleanly. If smoke test uses YH config (which has no `tracking:` section), there is no risk.

2. **Are the 60% vulture findings in `synthetic/trajectory.py` genuinely dead?**
   - What we know: `loose_school`, `milling`, `streaming` are unused trajectory factory methods; `FishTrajectoryState` is an unused class.
   - What's unclear: Whether they're used by tests or evaluation scripts not scanned.
   - Recommendation: Document in audit report for Phase 86. Do not remove in Phase 85 (out of scope for safe inline fixes).

3. **Should `tests/unit/segmentation/` be renamed?**
   - What we know: The directory is misnamed — it actually contains tests for `core/pose/crop.py` and `core/detection/backends/yolo.py`, not any segmentation module.
   - What's unclear: Whether renaming to `tests/unit/pose/` would break anything in CI paths.
   - Recommendation: Document in audit report for Phase 86. Do not rename in Phase 85.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `hatch run typecheck` output (25 errors, exact files and lines)
- Direct codebase inspection — `grep -rn "boxmot\|ocsort"` across all source and test files
- Direct codebase inspection — `hatch run vulture src/ --min-confidence 80` results
- `src/aquapose/engine/config.py` — TrackingConfig, LutConfig definitions
- `src/aquapose/calibration/luts.py` — LutConfigLike protocol definition
- `pyproject.toml` — dependency declarations, hatch scripts

### Secondary (MEDIUM confidence)
- basedpyright protocol compatibility: `@property` members in Protocol satisfy frozen dataclass fields — standard Python typing behavior, cross-verified with the actual error messages

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Type error inventory: HIGH — directly from `hatch run typecheck` output
- BoxMot removal sites: HIGH — direct grep of entire codebase
- Dead code findings: HIGH (80% vulture) / MEDIUM (60% vulture — known false positive rate)
- CLI smoke test pattern: HIGH — CLI code read directly, YH config verified present
- LutConfigLike fix strategy: HIGH — standard basedpyright protocol behavior

**Research date:** 2026-03-11
**Valid until:** 2026-04-10 (stable domain — type errors and file inventory won't change without new commits)
