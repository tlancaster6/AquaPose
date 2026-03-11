# Phase 92: Parameter Tuning Pass - Research

**Researched:** 2026-03-11
**Domain:** Association parameter grid sweep, v3.7 vs v3.8 baseline comparison, end-to-end pipeline validation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Tuning grid:**
- Sweep `keypoint_confidence_floor` at 3 points: 0.2, 0.3, 0.4 (narrow range around default)
- Re-sweep `ray_distance_threshold` and `score_min` alongside the new param (multi-keypoint scoring may shift optimal values)
- Hold all other params at current defaults (validation, recovery, leiden, etc.)
- Validation and recovery params (min_segment_length, eviction_reproj_threshold, recovery thresholds) stay at implementation defaults — not swept

**Baseline comparison:**
- Run v3.7 config through the same tuning harness on the same cached data (apples-to-apples)
- Add a config toggle (e.g. `use_multi_keypoint_scoring: bool`) to disable multi-keypoint scoring for v3.7 baseline
- Generate a fresh diagnostic cache before tuning (current code with keypoint data in tracklets)
- Use the same short clip as Phase 72 (~1 min, --max-chunks 6, YH project)

**Metrics and targets:**
- Singleton rate is the primary metric for ranking grid candidates
- Reproj error and guardrails (must not degrade)
- Reproj error guardrail: must stay under 5px mean
- Singleton rate target: ~15%, floor: must beat 27% (v3.7 baseline)

**Tuning harness scope:**
- Harness replays the full association pipeline: scoring -> clustering -> validation -> recovery
- Both association and singleton recovery are included since recovery affects singleton rate
- Same tuning CLI entry point: `aquapose tune --stage association`
- Extend DEFAULT_GRID in association evaluator with the new params

**Final validation:**
- After selecting best config, run one full end-to-end pipeline run with tuned params (satisfies EVAL-02)
- Use the same short clip for the E2E run

### Claude's Discretion
- Exact grid values for ray_distance_threshold and score_min re-sweep
- Results document format and level of detail
- Whether to add additional metrics to the comparison beyond what's already computed
- Implementation details of the v3.7 scoring toggle

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | Parameter tuning pass on real data measuring singleton rate, reprojection error, and grouping quality vs v3.7 baseline | Existing `TuningOrchestrator.sweep_association()` covers singleton_rate, reproj error, and yield. Grid extension with `keypoint_confidence_floor` and re-sweep of `ray_distance_threshold` + `score_min` satisfies this. The v3.7 baseline toggle enables apples-to-apples comparison. |
| EVAL-02 | End-to-end pipeline run with tuned parameters confirms improvement over v3.7 | After grid selection, one `aquapose run --max-chunks 6` with tuned YAML defaults satisfies this. Phase 72 baseline run metrics are already documented for comparison. |
</phase_requirements>

---

## Summary

Phase 92 is a calibration phase that runs the existing tuning harness (`TuningOrchestrator`) on fresh v3.8 diagnostic cache data to empirically select optimal values for the new `keypoint_confidence_floor` parameter alongside a re-sweep of `ray_distance_threshold` and `score_min`. The existing machinery in `evaluation/tuning.py` and `evaluation/stages/association.py` is nearly ready — the main code changes are: (1) extend `DEFAULT_GRID` with the new param, (2) add a `use_multi_keypoint_scoring` toggle to `AssociationConfig` for the v3.7 baseline path, and (3) fix a pre-existing blocker in the tune CLI where it requires `config_exhaustive.yaml` (never written by the diagnostic observer). Once unblocked, the tuning workflow is: generate fresh cache → run sweep → apply winner → run E2E → document results.

The v3.7 baseline reproj error was 35-275px before the LUT coordinate fix and ~2.8-3.5px after. Current v3.7 association tuned params are `ray_distance_threshold=0.01`, `score_min=0.3`. These were chosen on centroid-only scoring. Multi-keypoint scoring uses a per-keypoint confidence intersection mask (`keypoint_confidence_floor`) which may shift the optimal `ray_distance_threshold` if keypoint rays are geometrically tighter than centroids. The sweep will determine whether these need adjustment.

The singleton rate target (15% vs 27% baseline) is aggressive. Phase 91 added singleton recovery; whether it achieves the target on real data is the key empirical question. The tuning harness replays the full pipeline including recovery, so the measured singleton rate will reflect the combined effect of scoring, validation, and recovery.

**Primary recommendation:** Fix the `config_exhaustive.yaml` blocker first (Phase 92 Task 1), then generate the fresh diagnostic cache (Task 2), then run the extended grid sweep (Task 3), then do the E2E run (Task 4), then commit tuned defaults and results doc (Task 5).

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python dataclasses | stdlib | AssociationConfig frozen fields | Already used throughout codebase for all config |
| PyYAML | existing dep | Serialize tuned config to YAML for project config | Used by `serialize_config()` and YAML loading |
| itertools | stdlib | Grid combination generation in `sweep_association()` | Already used in `TuningOrchestrator` |
| NumPy | existing dep | Metric computation in `evaluate_association()` | Already used throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hatch | existing | `hatch run test` for verification after code changes | Any code change in Phase 92 |
| click | existing | CLI tune command already registered | No CLI changes needed — existing command |

---

## Architecture Patterns

### Existing Tuning Harness Flow

```
aquapose run -p YH --max-chunks 6 --add-observer diagnostic
  → writes diagnostics/chunk_NNN/cache.pkl files

aquapose tune -p YH --stage association [--run latest]
  → TuningOrchestrator(config_path)       # loads config + caches
  → sweep_association()                   # grid search
    → Phase 1: joint 2D grid (ray_dist x score_min)
    → Phase 2: sequential carry-forward (eviction, leiden, early_k)
    → Phase 3 NEW: include keypoint_confidence_floor in Phase 1 joint grid
  → format_comparison_table()             # prints baseline vs winner
  → format_config_diff()                  # prints YAML diff
```

### DEFAULT_GRID Extension Pattern

Current `DEFAULT_GRID` in `/src/aquapose/evaluation/stages/association.py`:

```python
DEFAULT_GRID: dict[str, list[float]] = {
    "ray_distance_threshold": [0.01, 0.02, 0.03],
    "score_min": [0.03, 0.15, 0.30],
    "eviction_reproj_threshold": [0.02, 0.03, 0.05],
    "leiden_resolution": [0.5, 1.0, 2.0],
    "early_k": [5.0, 10.0, 30.0],
}
```

Extension for v3.8 adds `keypoint_confidence_floor` with the 3-point sweep [0.2, 0.3, 0.4]. The joint 2D grid in `sweep_association()` currently sweeps `ray_distance_threshold x score_min`. This becomes a 3D joint grid or a sequential add.

**Decision needed (Claude's discretion):** Expand the joint grid to 3D (ray_dist x score_min x keypoint_confidence_floor = 3×3×3 = 27 combos) OR add `keypoint_confidence_floor` as the first carry-forward parameter after the 2D joint grid. The 3D joint approach is cleaner but 3× slower per combo (each combo re-runs full association).

**Recommendation:** 3D joint grid for the first 3 params (27 combos), then sequential carry-forward for eviction, leiden, early_k. This is consistent with the "scoring params should be co-optimized" rationale for the existing 2D joint grid.

### v3.7 Baseline Toggle

The user decision is to add `use_multi_keypoint_scoring: bool = True` to `AssociationConfig`. When `False`, the scorer falls back to centroid-only scoring (the v3.7 behavior before Phase 88). This gives an apples-to-apples comparison using the same cache data.

Implementation sketch for `AssociationConfig` in `engine/config.py`:
```python
use_multi_keypoint_scoring: bool = True  # True = v3.8 multi-kpt, False = v3.7 centroid fallback
```

In `scoring.py`, `score_tracklet_pair()` already has the no-keypoint branch:
```python
if tracklet_a.keypoints is None or tracklet_b.keypoints is None:
    return 0.0
```
The toggle would add:
```python
if not config.use_multi_keypoint_scoring:
    # centroid-only path (Phase 87 removed centroid-to-centroid, need to restore it here for baseline)
    return _score_pair_centroid_only(...)
```

**Important:** The centroid-only baseline requires the centroid arrays in the tracklets, which are always present (not gated behind `keypoints is not None`). The centroid-only scorer from v3.7 used a single ray per tracklet from the centroid pixel. This logic must be reconstructed in `scoring.py` or inlined as a private helper for the toggle path.

### config_exhaustive.yaml Blocker

**Problem (HIGH confidence):** The `aquapose tune` CLI requires `run_dir / "config_exhaustive.yaml"` to exist. Searching all YH runs and archived runs confirms this file is **never written** by the diagnostic observer or orchestrator. The diagnostic observer only writes `diagnostics/chunk_NNN/cache.pkl` and `diagnostics/manifest.json`. The `config_exhaustive.yaml` name was introduced in commit `72b8f78` (feat(69-02)) but the writer was never implemented.

**Fix options (Claude's discretion):**
1. **Simplest:** Change the tune CLI to fall back to `config.yaml` in the run dir. The file `run_dir/config.yaml` is always written by the orchestrator (verified in multiple real runs). This is a 2-line fix in `cli.py`.
2. **Correct per error message:** Have the diagnostic observer write `config_exhaustive.yaml` as a copy of `serialize_config(config)` in the run output dir. This matches the error message that says "Run the pipeline with --add-observer diagnostic to generate it."

**Recommendation:** Option 1 (fall back to `config.yaml`) is the lowest-risk fix that unblocks Phase 92. Option 2 would be more correct documentation-wise but adds complexity. Either works — decide at plan time.

**Note on config.yaml in run dirs:** Each run writes `config.yaml` which is the **project config** (not a run-specific exhaustive dump). The `TuningOrchestrator.__init__` calls `load_config(config_path)` which needs a loadable config with `n_animals`, `calibration_path`, and `association` fields. The project `config.yaml` always has these. This confirms option 1 is safe.

### Anti-Patterns to Avoid
- **Re-running the full video pipeline per grid combo:** The harness already avoids this by replaying from tracking cache. Each combo only re-runs `AssociationStage` (~40s on first run, not 12× camera video processing).
- **Scoring params independent of centroid toggle:** When `use_multi_keypoint_scoring=False`, `keypoint_confidence_floor` is irrelevant — don't include it in the v3.7 baseline sweep.
- **Over-sweeping params that are "held at defaults":** Context.md is explicit: validation/recovery params are NOT swept. Only `keypoint_confidence_floor`, `ray_distance_threshold`, `score_min` are swept.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Grid evaluation | Custom grid runner | `TuningOrchestrator.sweep_association()` | Already handles caching, patching, metrics, output formatting |
| Metric computation | Custom metric functions | `evaluate_association()` + `_compute_centroid_reprojection()` | Already computes singleton_rate, yield, reproj error |
| Config patching | Reconstruct config from YAML | `dataclasses.replace(config.association, **params)` | Already used in `_patch_association_config()` |
| Output formatting | Custom table formatter | `format_comparison_table()`, `format_config_diff()` | Already implemented and tested |
| Cache loading | Re-implement cache discovery | `load_run_context()` | Already handles both chunk-based and legacy formats |

---

## Common Pitfalls

### Pitfall 1: DEFAULT_GRID vs sweep_association() coupling
**What goes wrong:** Adding `keypoint_confidence_floor` to `DEFAULT_GRID` doesn't automatically include it in the `sweep_association()` joint grid — the joint grid is hardcoded as `joint_params = ["ray_distance_threshold", "score_min"]`. The carry-forward loop reads from `DEFAULT_GRID` but only iterates `carry_forward_params = ["eviction_reproj_threshold", "leiden_resolution", "early_k"]`.
**Why it happens:** The joint vs carry-forward split is explicit code, not data-driven from the grid dict.
**How to avoid:** When adding `keypoint_confidence_floor` to `DEFAULT_GRID`, also update `joint_params` list in `sweep_association()` (or the carry-forward list, depending on chosen approach).

### Pitfall 2: v3.7 baseline scoring uses centroid arrays
**What goes wrong:** When `use_multi_keypoint_scoring=False`, the scorer falls back to centroid-based rays. But `score_tracklet_pair()` currently returns 0.0 immediately if `tracklet_a.keypoints is None`. Since v3.8 always populates keypoints (Phase 87), the centroid path is never reached.
**How to avoid:** The toggle must branch BEFORE the keypoints-None check, not rely on it as a fallback.

### Pitfall 3: config_exhaustive.yaml fails silently in automated testing
**What goes wrong:** If a test invokes `aquapose tune` and `config_exhaustive.yaml` doesn't exist, it raises `ClickException` rather than the expected tuning output.
**How to avoid:** Fix the blocker in Task 1 before any integration test or manual E2E run.

### Pitfall 4: Cache staleness — v3.8 tracklets have keypoints, v3.7 caches don't
**What goes wrong:** If the Phase 72 diagnostic cache (used for v3.7 baseline) was generated before Phase 87 (keypoint pass-through in tracker), the `Tracklet2D.keypoints` field will be None. Trying to run multi-keypoint scoring on that cache will return 0.0 for all pairs.
**Why it happens:** Phase 87 adds keypoint pass-through to the tracker. Old caches predate this.
**How to avoid:** Generate a fresh diagnostic cache with current code before running any sweep. Context.md explicitly says "Generate a fresh diagnostic cache before tuning." The Phase 72 cache at `~/aquapose/projects/YH/runs/run_20260307_140127/` should NOT be used directly for v3.8 scoring.

### Pitfall 5: YH config.yaml uses `midline.backend: pose_estimation` (deprecated key)
**What goes wrong:** The YH project config.yaml uses `midline:` as the YAML key for what is now `pose:`. The `load_config()` function accepts both via backward-compat aliasing (`pose_kwargs = _merge_stage_config(pose_kwargs, yaml_nested.get("midline", {}))`). This is transparent — but running `load_config(run_dir/"config.yaml")` for a run generated by the YH project will work correctly.

### Pitfall 6: score_min values in DEFAULT_GRID include 0.03 (very permissive)
**What goes wrong:** With multi-keypoint scoring, a score of 0.03 may allow noisy cross-camera associations that centroid scoring would reject. This could increase reproj error above the 5px guardrail.
**How to avoid:** Monitor reproj error during the joint grid sweep. If 0.03 consistently degrades guardrails, document this finding in the results doc and discard it from consideration.

---

## Code Examples

### Extending DEFAULT_GRID

```python
# Source: /src/aquapose/evaluation/stages/association.py
DEFAULT_GRID: dict[str, list[float]] = {
    "ray_distance_threshold": [0.01, 0.02, 0.03],
    "score_min": [0.03, 0.15, 0.30],
    "keypoint_confidence_floor": [0.2, 0.3, 0.4],  # NEW for Phase 92
    "eviction_reproj_threshold": [0.02, 0.03, 0.05],
    "leiden_resolution": [0.5, 1.0, 2.0],
    "early_k": [5.0, 10.0, 30.0],
}
```

### Extending the joint grid in sweep_association()

```python
# In TuningOrchestrator.sweep_association():
joint_params = ["ray_distance_threshold", "score_min", "keypoint_confidence_floor"]  # was 2-element
joint_values = [grid[p] for p in joint_params]
joint_combos = list(itertools.product(*joint_values))  # 3x3x3 = 27 combos
```

### Adding use_multi_keypoint_scoring to AssociationConfig

```python
# In /src/aquapose/engine/config.py AssociationConfig
use_multi_keypoint_scoring: bool = True
```

With docstring addition:
```
use_multi_keypoint_scoring: Toggle to select scoring method. ``True`` (default) uses
    multi-keypoint ray distances (v3.8). ``False`` uses single centroid rays
    (v3.7 behavior) for baseline comparison.
```

### fix for config_exhaustive.yaml in tune CLI (option 1)

```python
# In /src/aquapose/cli.py, tune_cmd:
config_path = run_dir / "config_exhaustive.yaml"
if not config_path.exists():
    config_path = run_dir / "config.yaml"  # fallback to project config
if not config_path.exists():
    raise click.ClickException(
        f"No config found in {run_dir}. "
        "Run the pipeline first to generate a run directory."
    )
```

### Generating fresh diagnostic cache for tuning

```bash
# Generate cache for --max-chunks 6 (~1 min of video):
aquapose run -p YH --max-chunks 6 --add-observer diagnostic

# Then tune from that run:
aquapose tune -p YH --stage association
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Centroid-only ray scoring | Multi-keypoint ray scoring | Phase 88 | Richer geometric signal; keypoint_confidence_floor is new tunable |
| No post-clustering validation | Group validation + changepoint detection | Phase 90 | Eviction/split reduces over-grouped singletons |
| No singleton recovery | Greedy whole-assign + split-assign recovery | Phase 91 | Singletons get second chance; affects singleton rate target |
| Manual config tuning | Grid sweep via TuningOrchestrator | Phase XX | Systematic parameter selection |

---

## Open Questions

1. **Whether 27-combo joint grid is too slow for interactive use**
   - What we know: Each combo re-runs `AssociationStage.run()` on the tracking cache. With 6 chunks × ~200 frames, that's ~1200 frames. Previous tune runs took ~40s per full association run (from tune.log in archived2).
   - What's unclear: Recovery pass in Phase 91 adds extra computation. 27 combos × ~40-60s each ≈ 18-27 min sweep.
   - Recommendation: Acceptable — this is a one-time calibration. Use `TaskCreate` to run it as a background task per CLAUDE.md instructions.

2. **Whether the centroid-only scoring toggle needs to be a full implementation or just a parameter skip**
   - What we know: When `use_multi_keypoint_scoring=False`, `score_tracklet_pair()` must return centroid-based scores. The centroid-only logic was removed when Phase 88 implemented multi-keypoint scoring.
   - What's unclear: Whether enough centroid logic exists in the codebase to reconstruct it cleanly, or if the baseline comparison can be done differently.
   - Recommendation: Implement a minimal centroid-only path in `score_tracklet_pair()`. The centroid array is always present on `Tracklet2D`. The core logic is a single ray per tracklet per frame (using `centroids[i]`), followed by the same soft kernel. This is ~20 lines.

3. **Where v3.7 baseline metrics (singleton_rate=27%, reproj_error=?) live**
   - What we know: Phase 72 baseline run at `~/aquapose/projects/YH/runs/run_20260307_140127/` has `eval_results.json` and `diagnostics/`. Memory says "v3.7 baseline (target: ~15%, floor: better than 27%)."
   - What's unclear: The exact reproj error from v3.7 baseline (memory says 2.8-3.5px after LUT fix). This gives the "not degraded" guardrail context.
   - Recommendation: Record the v3.7 baseline values before the sweep starts (read eval_results.json or re-run baseline combo in the harness). Document them in the results doc.

---

## Implementation Task Structure (for Planner)

Based on the research, Phase 92 naturally decomposes into sequential tasks:

**Task 1 — Unblock tune CLI + extend grid (Code)**
- Fix `config_exhaustive.yaml` fallback in `cli.py`
- Add `use_multi_keypoint_scoring: bool = True` to `AssociationConfig`
- Implement centroid-only scoring path in `score_tracklet_pair()` (toggle branch)
- Add `keypoint_confidence_floor` to `DEFAULT_GRID`
- Update `joint_params` in `sweep_association()` to include `keypoint_confidence_floor`
- Run `hatch run test` to verify no regressions

**Task 2 — Generate fresh v3.8 diagnostic cache (Execution)**
- Run `aquapose run -p YH --max-chunks 6 --add-observer diagnostic`
- Verify cache.pkl files exist in diagnostics/ with Tracklet2D.keypoints populated

**Task 3 — Run grid sweep (Execution)**
- Run `aquapose tune -p YH --stage association`
- Capture comparison table output
- Select winner params

**Task 4 — E2E validation run (Execution)**
- Apply winner params to YH config.yaml
- Run `aquapose run -p YH --max-chunks 6`
- Verify singleton_rate < 27% and reproj error < 5px

**Task 5 — Commit tuned defaults + results doc (Code + Docs)**
- Update `AssociationConfig` defaults to winner values
- Write brief tuning results document in `.planning/phases/92-parameter-tuning-pass/92-RESULTS.md`
- Commit

---

## Sources

### Primary (HIGH confidence)
- `/src/aquapose/evaluation/stages/association.py` — `DEFAULT_GRID`, `AssociationMetrics`, `evaluate_association()` — direct code inspection
- `/src/aquapose/evaluation/tuning.py` — `TuningOrchestrator.sweep_association()`, `_patch_association_config()`, `format_comparison_table()` — direct code inspection
- `/src/aquapose/engine/config.py` — `AssociationConfig` dataclass, all current fields, defaults — direct code inspection
- `/src/aquapose/core/association/scoring.py` — `score_tracklet_pair()`, `_batch_score_frames_kpt()`, `AssociationConfigLike` — direct code inspection
- `/src/aquapose/core/association/stage.py` — `AssociationStage.run()`, pipeline steps 1-5 — direct code inspection
- `/src/aquapose/core/association/recovery.py` — `recover_singletons()` — direct code inspection
- `/src/aquapose/cli.py` — `tune_cmd` — config_exhaustive.yaml blocker confirmed by direct inspection
- `/home/tlancaster6/aquapose/projects/YH/runs/archived2/run_20260303_204845/logs/tune.log` — confirms tune ran previously, helps understand timing

### Secondary (MEDIUM confidence)
- `.planning/memory/association-tuning.md` (referenced in MEMORY.md) — v3.7 tuned params and results; explains why ray_dist=0.01 was chosen

### Tertiary (LOW confidence)
- Phase timing estimate: 40-60s per combo inferred from archived tune.log timing (single association run visible in log). Recovery adds overhead not measured.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies already in codebase, no new libraries
- Architecture: HIGH — existing tuning harness fully inspected, precise change locations identified
- Pitfalls: HIGH for config blocker (confirmed by filesystem search); MEDIUM for timing estimates

**Research date:** 2026-03-11
**Valid until:** Stable — 30 days (no external dependencies or fast-moving APIs)
