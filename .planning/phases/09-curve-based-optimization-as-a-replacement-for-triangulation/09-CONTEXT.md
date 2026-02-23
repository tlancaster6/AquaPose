# Phase 9: Curve-Based Optimization as a Replacement for Triangulation - Context

**Gathered:** 2026-02-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the interior of `triangulate_midlines()` with a correspondence-free B-spline optimizer that fits 3D splines directly against observed 2D skeletons via refractive reprojection + chamfer distance. The public API changes are acceptable — prefer clean code over backwards compatibility. Cross-view identity (RANSAC clustering) and half-width estimation remain unchanged.

Detailed design proposal: `.planning/inbox/curve_optimization_proposal.md`

</domain>

<decisions>
## Implementation Decisions

### Migration strategy
- Keep both old triangulation and new curve optimizer side-by-side during validation
- Code organization at Claude's discretion — prefer clean separation (e.g., new module alongside old)
- Wire both methods into `scripts/diagnose_pipeline.py` so user can compare by running the script with either method
- No need to write new diagnostics — just ensure both methods are accessible from the existing script
- After validation, old triangulation code will be deleted — design for eventual removal, don't over-invest in compatibility layers
- Breaking API changes are fine — prefer clean code

### Species priors & tuning
- Fish body length is 70–100mm (not 45mm as in proposal draft)
- All regularization weights (length, curvature, smoothness) exposed via a `CurveOptimizerConfig` dataclass
- Global species prior only for Phase 9 — no per-identity length refinement yet
- Per-identity length prior deferred to a future iteration if needed

### Curvature limits
- Claude's discretion on starting curvature limit value — pick something reasonable from literature, expose in config

### Performance targets
- Must be faster than current triangulation pipeline (~76s for 30 frames)
- CUDA GPU available — optimizer should leverage GPU for batched optimization
- Implement warm-start from previous frame's solution with cold-start fallback (as described in proposal)
- Implement adaptive early stopping (per-fish convergence masking) — important for 9-fish batches

### Claude's Discretion
- Curvature limit starting value
- Code organization (new module structure, naming)
- Coarse-to-fine stage count and control point counts
- L-BFGS hyperparameters (learning rate, max iterations per stage)
- Convergence threshold for early stopping
- How to handle the multi-start flip mitigation (if needed)

</decisions>

<specifics>
## Specific Ideas

- Full design proposal with loss formulation, coarse-to-fine strategy, initialization, and risk analysis is in `.planning/inbox/curve_optimization_proposal.md` — this should be the primary reference for research and planning
- User will compare methods by running `scripts/diagnose_pipeline.py` with each approach
- B-spline evaluation should use precomputed basis matrices (matrix multiply, not iterative de Boor)
- Chamfer distance is correspondence-free — the optimizer discovers point associations implicitly
- Huber loss for per-camera aggregation to downweight outlier cameras

</specifics>

<deferred>
## Deferred Ideas

- Per-identity length prior (running average of past reconstructions per fish) — add if global prior proves insufficient
- Multi-start optimization for head-tail flip resolution — add if flip rate is too high with coarse-to-fine alone
- Velocity-based warm-start extrapolation — add if simple previous-frame copy doesn't converge fast enough

</deferred>

---

*Phase: 09-curve-based-optimization-as-a-replacement-for-triangulation*
*Context gathered: 2026-02-22*
