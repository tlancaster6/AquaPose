---
phase: 30-config-and-contracts
verified: 2026-02-28T21:00:00Z
status: passed
score: 12/12 must-haves verified
re_verification: null
gaps: []
human_verification: []
---

# Phase 30: Config and Contracts Verification Report

**Phase Goal:** Pipeline config is unified and backward-compatible, device propagates to all stages from one top-level parameter, and Detection/Midline2D dataclasses carry the optional fields that v2.2 backends require
**Verified:** 2026-02-28T21:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                      | Status     | Evidence                                                                                                  |
|----|--------------------------------------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------------|
| 1  | Detection has `angle` and `obb_points` optional fields defaulting to None                  | VERIFIED   | `detector.py` lines 38-39: `angle: float \| None = None`, `obb_points: np.ndarray \| None = None`       |
| 2  | Midline2D has `point_confidence` optional field defaulting to None                         | VERIFIED   | `midline.py` line 63: `point_confidence: np.ndarray \| None = None`                                     |
| 3  | SegmentThenExtractBackend fills `point_confidence` with 1.0s on every Midline2D            | VERIFIED   | `segment_then_extract.py` line 288: `point_confidence=np.ones(len(xy_frame), dtype=np.float32)`         |
| 4  | Synthetic module fills `point_confidence` with 1.0s on every Midline2D                    | VERIFIED   | `synthetic.py` line 254: `point_confidence=np.ones(n_points, dtype=np.float32)`                         |
| 5  | Unknown YAML fields in any stage config raise ValueError with field name listed            | VERIFIED   | `config.py` lines 438-464: `_filter_fields()` raises `ValueError` for unknown keys; applied to all 8 config types |
| 6  | Renamed fields produce "did you mean?" hints in errors                                     | VERIFIED   | `config.py` lines 431-435: `_RENAME_HINTS` covers `expect_fish_count`, `device`, `stop_frame`           |
| 7  | `PipelineConfig.device` auto-detects CUDA/CPU and propagates to all stages                 | VERIFIED   | `config.py` lines 249-260: `_default_device()` helper; `pipeline.py` lines 322, 331: `device=config.device` |
| 8  | `PipelineConfig.n_sample_points=10` feeds into MidlineConfig.n_points and downstream      | VERIFIED   | `config.py` line 606: `mid_kwargs["n_points"] = top_kwargs.get("n_sample_points", 10)`; `pipeline.py` line 303: `n_points=config.n_sample_points` |
| 9  | `PipelineConfig.stop_frame` replaces `DetectionConfig.stop_frame`                         | VERIFIED   | `DetectionConfig` in `config.py` has no `device` or `stop_frame` fields; `pipeline.py` line 320: `stop_frame=config.stop_frame` |
| 10 | `n_animals=0` sentinel raises ValueError when not set in YAML via `load_config()`         | VERIFIED   | `config.py` lines 593-595: `if resolved_n_animals <= 0: raise ValueError("n_animals is required...")` |
| 11 | `aquapose init-config <name>` creates project scaffold with config.yaml and 4 subdirs      | VERIFIED   | `cli.py` lines 153-184: creates `runs/`, `models/`, `geometry/`, `videos/` and `config.yaml` with ordered YAML |
| 12 | Relative paths in YAML resolve relative to `project_dir`; absolute paths are unchanged    | VERIFIED   | `config.py` lines 564-584: project_dir resolution block; 3 tests cover relative, absolute, and empty cases |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact                                                                    | Provides                                           | Status     | Details                                                                 |
|-----------------------------------------------------------------------------|----------------------------------------------------|------------|-------------------------------------------------------------------------|
| `src/aquapose/segmentation/detector.py`                                     | Detection with angle and obb_points fields         | VERIFIED   | Lines 38-39: both fields present, optional, default None                |
| `src/aquapose/reconstruction/midline.py`                                    | Midline2D with point_confidence field              | VERIFIED   | Line 63: field present, optional, default None; module docstring updated |
| `src/aquapose/engine/config.py`                                             | Module-level `_filter_fields` with `_RENAME_HINTS` | VERIFIED   | Lines 431-464: both present; applied to all 8 config types in `load_config()` |
| `src/aquapose/engine/config.py`                                             | PipelineConfig with device, n_sample_points, stop_frame, project_dir | VERIFIED | Lines 311-314: all 4 fields present with correct defaults |
| `src/aquapose/engine/pipeline.py`                                           | `build_stages()` propagates device and n_sample_points | VERIFIED | Lines 303, 320, 322, 331: `config.device` and `config.n_sample_points` used |
| `src/aquapose/core/midline/backends/segment_then_extract.py`               | Midline2D filled with point_confidence=1.0s        | VERIFIED   | Line 288: `np.ones(len(xy_frame), dtype=np.float32)` set on every construction |
| `src/aquapose/core/synthetic.py`                                            | SyntheticDataStage uses n_points and fills point_confidence | VERIFIED | Lines 127, 254: `self._n_points` used; `np.ones(n_points)` on every midline |
| `src/aquapose/reconstruction/triangulation.py`                              | `N_SAMPLE_POINTS=15` is fallback, not hardcoded    | VERIFIED   | Line 35-36: comment updated; pipeline modules accept `n_sample_points` param |
| `src/aquapose/io/midline_writer.py`                                         | `Midline3DWriter` accepts n_sample_points           | VERIFIED   | Line 60: `n_sample_points: int = N_SAMPLE_POINTS` param added          |
| `src/aquapose/reconstruction/curve_optimizer.py`                            | `CurveOptimizer` accepts n_sample_points            | VERIFIED   | Line 856: `n_sample_points: int = N_SAMPLE_POINTS` param added         |
| `src/aquapose/cli.py`                                                       | Rewritten init-config with `<name>` arg and `--synthetic` flag | VERIFIED | Lines 139-184: positional name, --synthetic flag, scaffold creation |
| `tests/unit/engine/test_config.py`                                          | Tests covering all new config behaviors             | VERIFIED   | 16+ new tests: strict reject, rename hints, auto-detect, n_sample_points, n_animals, path resolution |
| `tests/unit/engine/test_cli.py`                                             | Tests for init-config scaffold                      | VERIFIED   | 5 new tests: directory creation, field order, --synthetic, duplicate prevention, no-synthetic default |
| `tests/e2e/test_smoke.py`                                                   | CPU device E2E test                                 | VERIFIED   | Parametrized with `device="cpu"` (and `"cuda:0"` when available)        |

### Key Link Verification

| From                                  | To                          | Via                                              | Status   | Details                                                    |
|---------------------------------------|-----------------------------|--------------------------------------------------|----------|------------------------------------------------------------|
| `config.py` `_filter_fields`          | All 8 stage configs         | Applied in `load_config()` to every sub-config   | WIRED    | All 8 types wrapped: DetectionConfig, MidlineConfig, AssociationConfig, TrackingConfig, ReconstructionConfig, SyntheticConfig, LutConfig, PipelineConfig |
| `segment_then_extract.py`             | `Midline2D`                 | `point_confidence=np.ones(...)` on construction  | WIRED    | Line 288: every successful extraction path sets the field  |
| `config.py` PipelineConfig.device     | `pipeline.py` build_stages  | `device=config.device` passed to DetectionStage and MidlineStage | WIRED | Lines 322, 331 |
| `config.py` PipelineConfig.n_sample_points | `pipeline.py` build_stages | `n_points=config.n_sample_points` to SyntheticDataStage; propagated to midline via load_config | WIRED | Lines 303, 606 |
| `cli.py` init_config                  | `~/aquapose/projects/<name>/` | Creates directory structure including 4 subdirs | WIRED    | Lines 153-162: mkdir + 4 subdirs created                   |
| `config.py` project_dir               | Path fields in PipelineConfig | Resolved in load_config() before dataclass construction | WIRED | Lines 564-584: resolves video_dir, calibration_path, output_dir, model_path, weights_path |

### Requirements Coverage

| Requirement | Source Plan | Description                                                     | Status    | Evidence                                                             |
|-------------|-------------|-----------------------------------------------------------------|-----------|----------------------------------------------------------------------|
| CFG-01      | 30-02       | Single top-level `device` propagates to all stages              | SATISFIED | `PipelineConfig.device` field + `build_stages()` propagation verified |
| CFG-02      | 30-02       | `n_sample_points` configurable; no hardcoded constants          | SATISFIED | `n_sample_points=10` in PipelineConfig; downstream modules parameterized; 2 remaining `= 15` are function default fallbacks in legacy `MidlineExtractor` (not in pipeline path) |
| CFG-03      | 30-02       | Single unified fish count parameter (`n_animals`)               | SATISFIED | `n_animals` in `PipelineConfig`, propagated to `association.expected_fish_count` and `synthetic.fish_count`; `expect_fish_count` removed from AssociationConfig |
| CFG-04      | 30-02       | `stop_frame` is top-level, not stage-nested                     | SATISFIED | `stop_frame` in `PipelineConfig`; removed from `DetectionConfig`; rename hint added |
| CFG-05      | 30-01       | All stage configs use `_filter_fields()` (strict reject)        | SATISFIED | Module-level `_filter_fields()` applied to all 8 config types in `load_config()` |
| CFG-06      | 30-03       | `init-config` creates `~/aquapose/projects/<name>/` scaffold    | SATISFIED | CLI creates project dir + `runs/`, `models/`, `geometry/`, `videos/` + `config.yaml` |
| CFG-07      | 30-03       | Config paths resolve relative to `project_dir`                  | SATISFIED | `load_config()` resolves 5 path fields when `project_dir` is set; absolute paths left unchanged |
| CFG-08      | 30-03       | `--synthetic` flag adds synthetic section to generated YAML     | SATISFIED | `cli.py` line 176-177: `if synthetic: data["synthetic"] = {...}` |
| CFG-09      | 30-03       | YAML fields ordered by user relevance                           | SATISFIED | `cli.py` lines 164-177: paths first, core params, then stage configs; `sort_keys=False` |
| CFG-10      | 30-01       | Detection carries optional `angle` field for OBB orientation    | SATISFIED | `detector.py` line 38: `angle: float \| None = None` |
| CFG-11      | 30-01       | Midline2D carries optional `point_confidence` for weighting     | SATISFIED | `midline.py` line 63: `point_confidence: np.ndarray \| None = None` |
| CFG-12      | 30-02       | Pipeline runs E2E on CPU and CUDA device modes                  | SATISFIED | `test_smoke.py` parametrized with `device=["cpu", "cuda:0"]`; both pass |

**Note on REQUIREMENTS.md tracking table:** The tracking table in REQUIREMENTS.md shows CFG-05, CFG-10, and CFG-11 as "Pending" — this is a stale tracking table entry. The requirement description checklist at the top of REQUIREMENTS.md correctly shows all 12 CFG requirements checked `[x]`. The code confirms all three are fully implemented.

### Anti-Patterns Found

| File                             | Line(s) | Pattern             | Severity | Impact                                                                  |
|----------------------------------|---------|---------------------|----------|-------------------------------------------------------------------------|
| `reconstruction/midline.py`      | 235, 353 | `n_points: int = 15` | INFO    | Default parameter value in `_resample_arc_length()` and legacy `MidlineExtractor.__init__()`. These are not in the active pipeline path — `SegmentThenExtractBackend` uses `self._n_points` which is config-driven. No action required. |

No blocker or warning-level anti-patterns found.

### Human Verification Required

None. All phase 30 behaviors are verifiable programmatically via unit tests and code inspection.

### Gaps Summary

No gaps. All 12 CFG requirements are satisfied. The phase goal is fully achieved:

- **Unified config surface:** Single `device`, `n_sample_points`, and `stop_frame` at top-level `PipelineConfig`. All three propagate from one location to all stages via `build_stages()`.
- **Strict config validation:** `_filter_fields()` applied to all 8 config types with rename hints for moved fields. Unknown fields raise `ValueError` with the field name.
- **v2.2 dataclass contracts:** `Detection` has `angle` and `obb_points` optional fields. `Midline2D` has `point_confidence` optional field. Both active backends (SegmentThenExtract, Synthetic) fill `point_confidence` with `np.ones(N)`.
- **Project scaffolding:** `aquapose init-config <name>` creates a ready-to-use project layout with ordered YAML.
- **Path resolution:** Relative paths in YAML resolve from `project_dir`; absolute paths and empty `project_dir` are left unchanged.
- **Test coverage:** 582 tests pass (0 failures). 30+ new tests cover all new behaviors.

---

_Verified: 2026-02-28T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
