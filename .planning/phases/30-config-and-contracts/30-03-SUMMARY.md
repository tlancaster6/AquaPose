---
phase: 30-config-and-contracts
plan: "03"
subsystem: engine
tags: [cli, config, yaml, pathlib, click]

# Dependency graph
requires:
  - phase: 30-config-and-contracts
    provides: PipelineConfig with project_dir field, load_config() with strict field validation

provides:
  - init-config CLI command creates ~/aquapose/projects/<name>/ scaffold with config.yaml
  - project_dir-based path resolution in load_config() for relative paths
  - Generated YAML has user-relevant field ordering (paths first, then core params)

affects: [31-training-infra, 32-yolo-obb, 33-keypoint-midline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Project scaffold pattern: aquapose init-config <name> creates a ready-to-use project layout"
    - "Path resolution pattern: relative paths in config resolved from project_dir before dataclass construction"

key-files:
  created: []
  modified:
    - src/aquapose/cli.py
    - src/aquapose/engine/config.py
    - tests/unit/engine/test_cli.py
    - tests/unit/engine/test_config.py

key-decisions:
  - "init-config uses positional <name> arg (not --output/-o) — creates named project directories"
  - "Generated YAML omits device (auto-detected) and stop_frame (defaults None) — reduces noise for new users"
  - "n_animals placeholder is the string 'SET_ME' (not null/comment) — fails load_config() validation, forcing user action"
  - "project_dir resolution uses Path.resolve() which adds drive letter on Windows — tests use tmp_path (OS-native absolute path) to avoid platform issues"
  - "synthetic section in generated YAML excludes fish_count — propagated from top-level n_animals at runtime"

patterns-established:
  - "project_dir resolution runs before run_id/output_dir resolution in load_config() so output_dir can be relative to project_dir"
  - "test_init_config tests use monkeypatch.setenv HOME/USERPROFILE to redirect ~ without platform-specific Path.home() patching"

requirements-completed: [CFG-06, CFG-07, CFG-08, CFG-09]

# Metrics
duration: 15min
completed: "2026-02-28"
---

# Phase 30 Plan 03: CLI Project Scaffold and Path Resolution Summary

**`aquapose init-config <name>` creates project directories with ordered config.yaml; relative paths in config resolve from project_dir in load_config()**

## Performance

- **Duration:** 15 min
- **Started:** 2026-02-28T20:07:01Z
- **Completed:** 2026-02-28T20:22:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Rewrote `init-config` CLI from `--output` file writer to `<name>` project scaffold creator
- Generated `config.yaml` uses user-relevant field ordering (project_dir, paths, core params, stage configs)
- `--synthetic` flag adds synthetic section without `fish_count` (propagated from `n_animals`)
- `load_config()` now resolves relative path fields (video_dir, calibration_path, output_dir, model_path, weights_path) relative to `project_dir`
- Absolute paths and missing `project_dir` are left unchanged (no resolution)
- 5 new CLI tests + 3 new config path-resolution tests; all 582 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite init-config CLI command with project scaffolding** - `9c0beec` (feat)
2. **Task 2: Implement project_dir path resolution in load_config() and rewrite CLI tests** - `a2df1d5` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `src/aquapose/cli.py` - Rewrote init_config command: positional name arg, --synthetic flag, directory scaffold, ordered YAML
- `src/aquapose/engine/config.py` - Added project_dir path resolution block before run_id/output_dir resolution in load_config()
- `tests/unit/engine/test_cli.py` - Replaced 4 old TestInitConfig tests with 5 new scaffold-focused tests; uses monkeypatch.setenv for HOME/USERPROFILE
- `tests/unit/engine/test_config.py` - Added 3 new tests for project_dir path resolution (relative, absolute, empty)

## Decisions Made

- **`SET_ME` placeholder string for `n_animals`**: Chosen over YAML comment (`n_animals: # REQUIRED`) because it produces a string that fails `load_config()` validation cleanly with an actionable error. Comment would produce `n_animals: null`, which also fails — but `SET_ME` is more explicit in the file.
- **`device` omitted from generated YAML**: Intentional — auto-detected at runtime. Reduces cognitive load for new users who don't need to know about CUDA.
- **`Path.resolve()` in project_dir resolution**: Makes the path absolute and canonical. On Windows this adds the drive letter, which the tests account for by using `tmp_path` as the project dir (guaranteed OS-native absolute path).
- **`monkeypatch.setenv` for test home dir**: Cross-platform approach (patches both `HOME` and `USERPROFILE`) rather than `monkeypatch.setattr(Path, "home", ...)` which doesn't affect `expanduser()`.

## Deviations from Plan

None - plan executed exactly as written.

The only issue encountered was a test failure due to Windows-specific path behavior (`Path("/fake/project").resolve()` adds drive letter `C:`). Fixed by updating the test to use `tmp_path` (an OS-native absolute path) instead of a Unix-style path literal. This is a test correctness fix, not a production code change.

## Issues Encountered

- **Windows path resolution**: Test `test_project_dir_resolves_relative_paths` initially used `/fake/project` as the project dir. On Windows, `Path("/fake/project").resolve()` becomes `C:\fake\project` (adds current drive). Fixed by using `tmp_path` as the project dir so expected path computation uses the same `.resolve()` call as the implementation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `init-config` command is complete and user-ready for project onboarding documentation
- `project_dir` path resolution enables portable project layouts (relative paths in YAML)
- Phase 31 (Training Infra) can use this project layout when documenting training workflows
- CFG-06 through CFG-09 requirements fulfilled

---
*Phase: 30-config-and-contracts*
*Completed: 2026-02-28*
