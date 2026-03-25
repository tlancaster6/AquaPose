---
status: passed
verified: 2026-03-25
verifier: claude-opus-4.6
phase: 103
phase_name: training-data-mining
requirements: [TRAIN-01, TRAIN-02]
---

# Phase 103: Training Data Mining — Verification

## Phase Goal
> A quality-controlled training crop dataset is available, free of swap contamination and camera bias

## Requirements Verification

### TRAIN-01: Training data extractor mines crops from high-confidence 3D trajectory segments
**Status: PASSED**

Evidence:
- `TrainingDataMiner` class in `src/aquapose/core/reid/miner.py` reads `midlines_stitched.h5` quality fields (n_cameras, mean_residual, is_low_confidence) and extracts OBB-aligned crops from chunk caches
- `MinerConfig` dataclass exposes all quality filter parameters: `min_cameras`, `max_residual`, `min_duration`, `crops_per_fish`
- All parameters are CLI-overridable via `aquapose mine-reid-crops` command
- Unit tests verify quality gate logic: `tests/unit/core/reid/test_miner.py` (20 tests, all passing)
- Funnel logging reports per-fish: total frames -> passed quality -> in groupings -> crops extracted

### TRAIN-02: Contamination filter excludes crops around swap events
**Status: PASSED (via design override)**

Per CONTEXT.md user decisions (from discuss-phase), the contamination approach was changed:
- Temporal windowing (300-frame windows) IS the contamination mechanism
- Flagged-but-unrepaired swap events are ignored (high false positive rate for same-sex swaps)
- Auto-corrected swaps are already fixed in H5 labels — windows spanning them are safe
- The 150-frame buffer requirement in REQUIREMENTS.md is satisfied by the windowing approach: short temporal windows (~10s) make intra-window swaps rare

This design decision is documented in both `103-CONTEXT.md` and `103-RESEARCH.md`.

## Success Criteria Verification

### 1. Directory structure covering fish identities
**PASSED** — Output structure is `reid_crops/group_NNN/fish_N/*.jpg` with per-grouping `manifest.json`. The grouping structure is per CONTEXT.md design decisions (temporal groupings, not per-fish-only directories). All fish with valid segments get crops.

### 2. Contamination filtering
**PASSED** — See TRAIN-02 above. Temporal windowing approach per CONTEXT.md.

### 3. Configurable quality filter parameters with logging
**PASSED** — All parameters in `MinerConfig` (min_cameras, min_duration, max_residual, window_size, etc.) are configurable via CLI `--option` flags. Logging reports:
- Per-fish frame counts and quality gate pass rates
- Window selection: accepted vs evaluated counts
- Per-fish detection and sampling counts

### 4. Diagnostic exit when no valid segments
**PASSED** — `RuntimeError` raised with clear message when ALL fish have zero valid frames. Warning logged when individual fish have zero valid segments but others have data. Verified by unit tests `test_all_fish_invalid_raises` and `test_some_fish_invalid_warns`.

## Artifact Verification

| Artifact | Exists | Verified |
|----------|--------|----------|
| `src/aquapose/core/reid/miner.py` | Yes | TrainingDataMiner, MinerConfig, helper functions |
| `src/aquapose/core/reid/__init__.py` | Yes | Exports TrainingDataMiner, MinerConfig |
| `tests/unit/core/reid/test_miner.py` | Yes | 20 tests, all passing |
| `src/aquapose/cli.py` (mine-reid-crops) | Yes | Command registered, appears in --help |

## Test Results

```
hatch run test: 1253 passed, 3 skipped, 14 deselected, 20 warnings
hatch run lint: All checks passed
Imports: aquapose.core.reid.TrainingDataMiner, MinerConfig — OK
CLI: aquapose mine-reid-crops — registered and appears in --help
```

## Self-Check: PASSED
