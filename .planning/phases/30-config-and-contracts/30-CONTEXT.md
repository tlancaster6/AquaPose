# Phase 30: Config and Contracts - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Unify pipeline configuration (device propagation, configurable constants, backward compatibility), extend data contracts (Detection and Midline2D optional fields for v2.2 backends), and add `init-config` CLI scaffolding. No new pipeline stages or backends — this phase prepares the config and type foundations that Phases 31-34 build on.

</domain>

<decisions>
## Implementation Decisions

### init-config Scaffolding
- `aquapose init-config <name>` creates `~/aquapose/projects/<name>/` with config YAML plus four empty subdirectories:
  - `runs/` — pipeline output
  - `models/` — ML model weights and training data
  - `geometry/` — calibration file, LUTs
  - `videos/` — input videos or subdirectories of input videos
- Generated YAML is minimal: only essential fields active, with brief comments pointing to docs for advanced config (stage-specific params, observer config are NOT included as commented-out blocks)
- `--synthetic` flag adds a `synthetic:` section to the YAML; this section does NOT include its own fish count — it relies on the top-level `n_animals` (in real data = expected count, in synthetic = generate-and-expect count)

### YAML Field Ordering
- Fields ordered by what a user edits first when setting up a new project:
  1. `project_dir` (root for all relative paths)
  2. Other paths (video_dir, calibration_path, output_dir, etc.)
  3. `n_animals` and `device`
  4. Stage-specific parameters (below the fold)

### Config Defaults
- `device`: Auto-detect — `torch.cuda.is_available()` → `cuda:0`, otherwise `cpu`. No explicit default in YAML; auto-detect is the behavior when field is absent.
- `n_sample_points`: Default 10. Intentionally lower than the current hardcoded 15 to flush out any remaining hardcoded references throughout the codebase.
- `n_animals`: **Required** — no default. Config validation fails if missing.
- `stop_frame`: Default `None` (process all frames). User sets it only for dev/debug runs.

### Backward Compatibility
- `_filter_fields()` uses **strict reject**: unknown YAML fields cause an error listing the unrecognized fields. No silent dropping.
- Renamed fields (e.g. `expect_fish_count` → `n_animals`) are rejected with a hint: error message says "did you mean `n_animals`?" so the user knows exactly what to change. No auto-rename.
- Removed fields get a generic "unknown field: X" error — no deprecation registry or removal explanations. Keep it simple.

### Dataclass Extensions
- `Detection.angle`: Radians from x-axis, standard math convention (0 = right, π/2 = up), range [-π, π]. Consistent with YOLO-OBB output format. `None` when produced by non-OBB detectors.
- `Detection.obb_points`: `np.ndarray` of shape (4, 2) — 4 corner points as (x, y), ordered clockwise from top-left of the oriented box. `None` when produced by non-OBB detectors.
- `Midline2D.point_confidence`: `np.ndarray` of shape (N,), values in [0, 1] probability range. Directly usable as triangulation weights. **Always an array** — backends that don't produce per-point confidence (e.g. segment-then-extract) fill with 1.0s. No None checks downstream.

### Claude's Discretion
- Whether `_filter_fields()` is a standalone utility or per-class classmethod
- Exact subdirectory structure within `_filter_fields()` implementation
- Project directory scaffolding details beyond the four named subdirectories
- Config validation error message formatting

</decisions>

<specifics>
## Specific Ideas

- n_sample_points default of 10 serves dual purpose: sufficient for fish shape capture AND acts as a litmus test to surface any remaining hardcoded `15` literals in the codebase
- n_animals unification: single parameter serves both real (expected count) and synthetic (generate count) modes — the synthetic block should never have its own fish count

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 30-config-and-contracts*
*Context gathered: 2026-02-28*
