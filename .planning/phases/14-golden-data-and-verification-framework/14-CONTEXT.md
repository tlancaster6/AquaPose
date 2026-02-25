# Phase 14: Golden Data and Verification Framework - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Frozen reference outputs from the v1.0 pipeline exist on disk as a committed snapshot, and an interface test harness can assert that a Stage produces correct output from a given context. Golden data must be committed as a standalone commit BEFORE any stage migration begins (Phase 15).

</domain>

<decisions>
## Implementation Decisions

### Golden data scope
- Single representative clip with good fish visibility across cameras
- 30-60 consecutive frames to cover temporal tracking behavior
- All stage outputs frozen: detections, masks, midlines, associations, 3D splines, tracks
- One .pt fixture file per stage (e.g., golden_detection.pt, golden_segmentation.pt)
- Estimated total size: ~7-10MB — comfortable for direct repo commit

### Determinism strategy
- GPU allowed for generation (faster), with tolerance-based comparison for regression tests
- Fixed global seed (torch/numpy/random) set once at script start; seed value documented
- Default numerical tolerance: atol=1e-3 (moderate — allows GPU floating-point variance)
- Environment metadata recorded alongside golden data (GPU model, CUDA version, PyTorch version)

### Test harness design
- Serialization format: PyTorch .pt files (torch.save/torch.load) — native to the codebase, no conversion
- One fixture file per stage, stored in tests/golden/
- Assertions are structural + numerical:
  - Structural: correct keys, shapes, dtypes in output
  - Numerical: torch.allclose within tolerance (atol=1e-3)
- Harness tests instantiate a Stage, call stage.run(context), and assert output fields in PipelineContext

### Storage and CI
- Golden data committed directly to repo in tests/golden/ (version-controlled, ~7-10MB)
- Golden data regression tests marked @slow — skipped in CI, run locally or on GPU runners
- Generation script: standalone scripts/generate_golden_data.py (not a CLI subcommand)
- Raw video frames required for generation stay on local disk (not committed)

### Regression test retention (from doctrine)
- After Phase 16 equivalence is established, per-stage decision on keeping regression tests:
  - Detection: keep (quality results)
  - Segmentation: evaluate case-by-case (middling results)
  - Triangulation/optimization: may not keep (still under development)
- This is a Phase 16 decision, noted here for planner awareness

### Claude's Discretion
- Exact generation script structure and CLI arguments
- How PipelineContext fixtures are constructed for harness tests
- Per-stage tolerance adjustments if 1e-3 proves too tight/loose for specific outputs
- Golden data directory structure within tests/golden/

</decisions>

<specifics>
## Specific Ideas

- Doctrine mandates determinism: same inputs + config + seeds = identical outputs
- Stages are pure computation with no side effects — golden data captures only structured outputs, not filesystem artifacts
- Generation script must be re-runnable: running on the same clip with the same seed produces equivalent outputs (within GPU tolerance)
- "Complexity is allowed. Entanglement is not." — test harness should be simple and direct

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-golden-data-and-verification-framework*
*Context gathered: 2026-02-25*
