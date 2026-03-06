---
phase: 13-engine-core
plan: 02
subsystem: engine
tags: [frozen-dataclasses, yaml, config, pipeline, pyyaml]

# Dependency graph
requires:
  - phase: 13-engine-core plan 01
    provides: engine package skeleton (Stage Protocol, PipelineContext)
provides:
  - Frozen dataclass config hierarchy (DetectionConfig, SegmentationConfig, TrackingConfig, TriangulationConfig, PipelineConfig)
  - load_config() factory with defaults -> YAML -> CLI override precedence
  - serialize_config() YAML serialization for run reproducibility
  - 11 unit tests covering all config behaviors
affects: [13-engine-core/03, 13-engine-core/04, any plan that runs the pipeline]

# Tech tracking
tech-stack:
  added: [pyyaml (already available), dataclasses.FrozenInstanceError]
  patterns:
    - frozen=True dataclasses for immutable config
    - loading precedence: defaults -> YAML -> CLI -> freeze
    - dot-notation CLI overrides ("detection.detector_kind")
    - serialize_config() as reproducibility contract (first artifact of every run)

key-files:
  created:
    - src/aquapose/engine/config.py
    - tests/unit/engine/test_config.py
    - tests/unit/engine/__init__.py
  modified:
    - src/aquapose/engine/__init__.py

key-decisions:
  - "Frozen dataclasses (not Pydantic) — consistent with CONTEXT.md decision, enforced via frozen=True"
  - "CLI overrides accept both dot-notation and nested dict forms for flexibility"
  - "output_dir default uses expanduser() so ~/aquapose/runs/ expands at load time"
  - "TriangulationConfig left as empty frozen dataclass — placeholder for future params per plan spec"

patterns-established:
  - "Stage configs as separate frozen dataclasses composed into PipelineConfig"
  - "load_config() factory isolates construction logic from caller"
  - "serialize_config() = dataclasses.asdict + yaml.dump — simple, dependency-free serialization"

requirements-completed: [ENG-05]

# Metrics
duration: 5min
completed: 2026-02-25
---

# Phase 13 Plan 02: Config Hierarchy Summary

**Frozen dataclass config hierarchy for the AquaPose pipeline with YAML + CLI override loading and post-freeze mutation guard**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-25T21:24:53Z
- **Completed:** 2026-02-25T21:30:03Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Implemented 4 stage-specific frozen dataclasses (DetectionConfig, SegmentationConfig, TrackingConfig, TriangulationConfig) and top-level PipelineConfig
- Implemented load_config() factory with layered precedence: defaults -> YAML file -> CLI kwargs -> freeze
- Implemented serialize_config() returning YAML string via dataclasses.asdict for run reproducibility
- Wrote 11 unit tests covering all config behaviors (defaults, YAML override, CLI override, precedence, freeze guard, run_id, serialization roundtrip)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement frozen config dataclass hierarchy with YAML and CLI override loading** - `14245bf` (feat)
2. **Task 2: Write tests for config defaults, YAML override, CLI override, freeze, and serialization** - `4ff6c06` (feat)

## Files Created/Modified

- `src/aquapose/engine/config.py` - Frozen config dataclass hierarchy, load_config(), serialize_config()
- `src/aquapose/engine/__init__.py` - Updated to export all config symbols (sorted __all__ for ruff RUF022 compliance)
- `tests/unit/engine/test_config.py` - 11 unit tests for all config behaviors
- `tests/unit/engine/__init__.py` - Test package init

## Decisions Made

- CLI overrides accept both dot-notation strings ("detection.detector_kind") and nested dicts ({"detection": {"detector_kind": ...}}) — both are common CLI/config patterns and trivial to support
- output_dir default uses Path.expanduser() at load_config() time so the home directory is resolved immediately
- TriangulationConfig is an empty frozen dataclass as specified — placeholder for future triangulation parameters

## Deviations from Plan

None - plan executed exactly as written.

Minor: Added 3 extra test cases beyond the 8 specified (test_frozen_stage_config_raises_on_mutation, test_load_config_cli_overrides_nested_dict, test_serialize_config_is_string) for additional coverage. All 11 tests pass.

## Issues Encountered

- Pre-commit ruff RUF022 rule requires `__all__` to be sorted alphabetically. The existing `__init__.py` from prior plans used grouped/commented ordering. Fixed by sorting all symbols alphabetically.

## Next Phase Readiness

- Config system complete and tested — ready for 13-03 (events) and 13-04 (orchestrator)
- load_config() and serialize_config() are the primary interfaces the orchestrator (plan 13-04) will call
- No blockers

---
*Phase: 13-engine-core*
*Completed: 2026-02-25*
