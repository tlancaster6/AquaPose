---
status: resolved
trigger: "9fish-coldstart-failure: Cold-start initialization catastrophically fails when n-fish=9 but works fine for 4-8 fish"
created: 2026-02-23T00:00:00Z
updated: 2026-02-23T00:00:00Z
---

## Current Focus

hypothesis: Unknown - gathering initial evidence from source code
test: Read curve_optimizer.py cold-start path and initialization module
expecting: Find hardcoded limit, threshold, or combinatorial issue that breaks at 9 fish
next_action: Read curve_optimizer.py, initialization module, and check RANSAC clustering logic

## Symptoms

expected: 9-fish cold-start initialization should work similar to 4-8 fish cases, with most/all fish getting triangulation-seeded and reasonable initial losses
actual: Only 1/9 fish gets triangulation-seeded (fish 8). Fine stage losses are very high (49-570 range). The optimizer "converges" at step 13 but results are bad.
errors: No errors thrown, but log shows "triangulation-seeded 1/9 cold-start fish: [8]" — RANSAC triangulation only succeeded for 1 fish.
reproduction: python scripts/diagnose_pipeline.py --method curve --stop-frame 1 --output-dir output/synthetic/realrig/curve/nfish9 --synthetic --n-fish 9
started: Discovered during synthetic data testing. Works fine with n-fish 4-8, fails at 9.

## Eliminated

## Evidence

## Resolution

root_cause:
fix:
verification:
files_changed: []
