# Phase 28: e2e testing - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the full pipeline end-to-end on real and synthetic data. Fix blocking bugs as encountered. Log non-blocking bugs for later triage. Update tests/e2e/ to current pipeline and confirm passing. Final output: at least some fish yield reasonable 3D splines with trajectories spanning contiguous frames. Full optimization and tuning is out of scope — this phase validates that the pipeline is basically functional.

</domain>

<decisions>
## Implementation Decisions

### Test data sources
- 4-camera subset using videos from `C:\Users\tucke\Desktop\Aqua\Videos\test_videos\`
- ~100 frames (~3-4 seconds) — enough to test tracking continuity without long runtimes
- Both real video and synthetic data: synthetic for fast CI, real for local validation
- Calibration: standard release calibration from `C:\Users\tucke\Desktop\Aqua\AquaCal\release_calibration\calibration.json`

### Success criteria
- Visual inspection of reprojection overlays for real data runs — splines should visibly follow fish
- At least 1 fish must produce valid 3D splines spanning 3+ contiguous frames
- No strict numeric threshold (e.g., reprojection error) — visual judgment for this phase

### Visual artifacts
- Real data tests: generate reprojection overlay videos by default (diagnostic mode)
- Synthetic data tests: data output only, no visualization artifacts

### Bug triage policy
- **Blocking:** Pipeline crashes/hangs OR zero fish get reconstructed (no 3D output at all)
- **Non-blocking:** Degraded quality, some fish fail but others succeed, unexpected warnings
- Straightforward blocking bugs: fix immediately in-phase
- Complex/risky blocking bugs: pause and report for user direction
- Non-blocking bugs: listed (no severity rating) in a .planning/ document for later triage

### E2E test structure
- Two assertion levels: smoke test (pipeline completes without exception) + output validation (3D splines exist, have expected shape, span multiple frames)
- Real-data tests marked @slow — only run with `hatch run test-all`; synthetic tests run in normal test suite
- Test artifacts (reprojection videos, spline data) saved to `tests/e2e/output/` (gitignored contents) for human review
- Pytest-only execution — no separate validation script

### Synthetic data approach
- Use existing synthetic mode (`--mode synthetic`) with default SyntheticConfig (3 fish, ~30 frames, seed=42)
- One simple smoke test — not scenario-based (crossing_paths, tight_schooling, etc. are available but not part of e2e suite)
- Exercises Tracking → Association → Reconstruction on known-geometry inputs
- Fabricated 3x3 camera rig with Snell's law projection (no real calibration needed)

### Claude's Discretion
- Whether to update existing e2e tests or rewrite from scratch (assess salvageability)
- Exact output validation assertions (spline shape, frame count thresholds)
- Artifact directory structure within tests/e2e/output/

</decisions>

<specifics>
## Specific Ideas

- User wants to be able to review real-data visual artifacts after pytest runs — save to tests/e2e/output/ so they're easy to find
- Synthetic tests serve as fast CI guard; real-data tests are the true validation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 28-e2e-testing*
*Context gathered: 2026-02-27*
