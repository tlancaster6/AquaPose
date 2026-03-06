---
phase: 18-cli-and-execution-modes
plan: 03
status: complete
---

## Summary

Implemented synthetic execution mode with SyntheticDataStage that generates known-geometry 3D fish splines, projects them through the real refractive calibration model, and replaces Detection + Midline stages. Made build_stages() mode-aware.

## What was built

- **SyntheticDataStage** (`src/aquapose/core/synthetic.py`): Stage protocol compliant. Generates configurable 3D fish splines, projects to 2D per camera, produces Detection + AnnotatedDetection with Midline2D objects. Deterministic with seed.
- **SyntheticConfig** (`src/aquapose/engine/config.py`): Added to PipelineConfig with fish_count, frame_count, noise_std, seed fields. Processed in load_config YAML and CLI override layers.
- **Mode-aware build_stages** (`src/aquapose/engine/pipeline.py`): Synthetic mode returns 4-stage list (SyntheticDataStage + Association + Tracking + Reconstruction). All other modes return standard 5-stage list.
- **CLI synthetic wiring** (`src/aquapose/cli.py`): Synthetic mode uses production-mode observers. No stage logic in CLI.

## Key files

### Created
- `src/aquapose/core/synthetic.py`
- `tests/unit/core/test_synthetic.py`
- `tests/unit/engine/test_build_stages.py`

### Modified
- `src/aquapose/core/__init__.py`
- `src/aquapose/engine/config.py`
- `src/aquapose/engine/__init__.py`
- `src/aquapose/engine/pipeline.py`
- `tests/unit/engine/test_cli.py`

## Verification

- 12 SyntheticDataStage unit tests pass (protocol, output, determinism, noise)
- 5 build_stages tests pass (production 5-stage, synthetic 4-stage, stage ordering)
- 18 CLI tests pass including 3 synthetic mode tests
- 576 total unit tests pass

## Commits

- `62e94df` feat(18-03): add synthetic execution mode with SyntheticDataStage
