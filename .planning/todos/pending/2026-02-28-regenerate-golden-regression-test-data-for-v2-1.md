---
created: 2026-02-28T12:44:13.357Z
title: Regenerate golden regression test data for v2.1
area: testing
files:
  - tests/golden/test_stage_harness.py
  - tests/golden/conftest.py
  - scripts/generate_golden_data.py
---

## Problem

The 8 golden regression tests in `tests/golden/test_stage_harness.py` are permanently skip-marked because their `.pt` fixture files were pickled from v1.0 modules (`aquapose.tracking.tracker.FishTrack`, `aquapose.segmentation.CropRegion`, etc.) that were deleted or restructured in v2.1. Python's unpickler fails on import, so the tests can't even collect.

## Solution

1. Rewrite `scripts/generate_golden_data.py` to use v2.1 stage boundaries (Detection → 2D Tracking → Association → Midline → Reconstruction)
2. Run on reference data to produce new `.pt` fixtures
3. Update `test_stage_harness.py` assertions for new stage outputs
4. Remove the `pytest.mark.skip` from the test module
