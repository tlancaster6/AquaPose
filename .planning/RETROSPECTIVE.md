# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — MVP

**Shipped:** 2026-02-25
**Phases:** 12 | **Plans:** 28 | **Timeline:** 11 days

### What Was Built
- Complete 3D fish midline reconstruction from multi-view silhouettes
- Two reconstruction methods: direct triangulation (fast) and curve optimization (experimental)
- Full detection-to-output pipeline: YOLO/MOG2 → SAM2 pseudo-labels → U-Net → skeletonization → triangulation → B-spline → HDF5
- Synthetic data generation system for controlled testing
- Diagnostic tooling: CLEAR MOT metrics, 2D overlays, 3D animations, markdown reports

### What Worked
- **Architecture pivot was decisive:** Shelving analysis-by-synthesis (Phase 4) mid-milestone and switching to direct triangulation was the right call — saved weeks of optimization dead-end
- **Decimal phase insertion:** Urgent work (02.1, 02.1.1, 04.1) was cleanly inserted without disrupting the main roadmap
- **Quick tasks for targeted fixes:** 8 quick tasks handled bug fixes and enhancements without full phase overhead
- **Batch-first APIs from day one:** All modules accepted lists, making multi-fish extension straightforward
- **GSD workflow:** Planning → research → execution cycle kept each phase focused and dependency-ordered

### What Was Inefficient
- **U-Net IoU 0.623 accepted too early:** Should have invested more in segmentation quality before building 6 downstream phases — noisy 2D midlines are now the primary bottleneck
- **Phase 2 was planned and then entirely replanned:** Original 4 plans were superseded by troubleshooting phases (02.1, 02.1.1), then the module was rewritten again. Better upfront real-data testing would have caught issues earlier
- **Phase 8 plan 03 summary gap:** E2E integration test was committed but summary not written, creating an audit gap

### Patterns Established
- `.cpu().numpy()` always (never bare `.numpy()`) for GPU tensors
- Differential LR for encoder/decoder in U-Net training
- XY-only cost matrices for tracking (Z uncertainty too high)
- Population-constrained tracking with fish ID recycling
- Exhaustive pairwise triangulation for ≤7 cameras
- Chamfer distance (not index-aligned residual) for curve optimization

### Key Lessons
1. **Segmentation quality gates reconstruction quality** — investing in better masks earlier would have prevented weeks of debugging noisy midlines downstream
2. **Architecture pivots are cheaper than sunk-cost optimization** — the ABS-to-triangulation pivot saved the project from a 30-min/sec dead end
3. **Z-reconstruction is fundamentally limited by top-down geometry** — 132x anisotropy means XY-only approaches are often superior to full 3D
4. **Synthetic data is essential for debugging** — real data has too many confounds; synthetic data isolated individual failure modes

### Cost Observations
- Model mix: ~70% sonnet, ~25% opus, ~5% haiku (balanced profile)
- Sessions: ~15-20 across 11 days
- Notable: Quick tasks (8 total) handled ~30% of functional improvements with minimal overhead

---

## Milestone: v2.0 — Alpha

**Shipped:** 2026-02-27
**Phases:** 10 | **Plans:** 34 | **Timeline:** 3 days

### What Was Built
- Event-driven 3-layer architecture: Core Computation → PosePipeline → Observers
- 5 pure Stage implementors (Detection, Midline, Association, Tracking, Reconstruction)
- 5 Observers (timing, HDF5 export, 2D overlay, 3D animation, diagnostic)
- CLI entrypoint (`aquapose run`) with 4 execution modes
- Golden data verification framework and regression test suite
- AST-based import boundary checker with pre-commit enforcement
- Comprehensive architectural audit and full remediation

### What Worked
- **Guidebook-driven design:** The `alpha_refactor_guidebook.md` provided a clear architectural vision that was faithfully implemented — 9/9 DoD criteria passed at audit
- **Audit-then-remediate pattern:** Phase 19 diagnosed without fixing, Phase 20 fixed systematically — produced higher-quality remediation than fix-as-you-go
- **Golden data before migration:** Committing v1.0 outputs before porting any stages (VER-01) gave confidence during Stage Migration phase
- **Structural typing for protocols:** `typing.Protocol` with `runtime_checkable` eliminated inheritance requirements — stages are plain classes
- **3-day velocity:** 10 phases and 34 plans in 3 days shows the refactor scope was well-defined and the GSD workflow scaled

### What Was Inefficient
- **Phase 14.1 insertion:** The 7-to-5 stage mismatch should have been caught during Phase 13 discussion, not after Phase 14 shipped — missed context alignment cost an extra phase
- **SUMMARY frontmatter underutilized:** `requirements_completed` and `one_liner` fields were empty in most SUMMARY.md files, making milestone tooling (summary-extract) less useful
- **VER-03 not executable in CI:** Regression tests can only run on the dev machine with real data — no automated gate for numerical equivalence
- **STATE.md milestone field stale:** The init tool reported v1.0 as the milestone because STATE.md wasn't properly updated between milestones

### Patterns Established
- Import boundary: `core/` never imports `engine/` at runtime (AST-enforced)
- Stage Protocol: `run(context: PipelineContext) -> PipelineContext` — no inheritance
- Observer dispatch: fault-tolerant, observers cannot affect pipeline execution
- Config hierarchy: frozen dataclasses with defaults → YAML → CLI → freeze
- `build_stages()` and `build_observers()` factories in engine layer

### Key Lessons
1. **Align on canonical models before coding** — the 7-to-5 stage mismatch cost an entire inserted phase; discuss-phase should validate domain models against reference docs
2. **Audit phases pay for themselves** — Phase 19's structured findings made Phase 20 remediation systematic instead of ad-hoc
3. **Import boundaries need enforcement from day one** — 7 IB-003 violations accumulated across 5 phases before the checker existed
4. **Populate SUMMARY frontmatter consistently** — milestone tooling depends on `requirements_completed` and `one_liner` fields
5. **Environment-dependent tests need env var configuration** — `AQUAPOSE_VIDEO_DIR` pattern works well for data-dependent tests

### Cost Observations
- Model mix: ~60% sonnet, ~30% opus, ~10% haiku (balanced profile)
- Sessions: ~8-10 across 3 days
- Notable: Phase 20 was the most intensive (5 plans, 37 files touched in Plan 02 alone)

---

## Milestone: v3.0 — Ultralytics Unification

**Shipped:** 2026-03-02
**Phases:** 5 | **Plans:** 14 | **Timeline:** 2 days

### What Was Built
- Removed all custom U-Net, SAM2, MOG2, and legacy training code
- YOLO-seg and YOLO-pose training wrappers with CLI and COCO seg data converter
- SegmentationBackend and PoseEstimationBackend as selectable midline backends
- Standard YOLO txt+yaml training data format across all model types
- Consolidated config (single weights_path), correct init-config defaults
- core/types/ shared type package; legacy dirs reorganized into core/ submodules

### What Worked
- **Incremental migration strategy:** Wave 1 created new files, Wave 2 rewired imports, Wave 3 updated docs — clean dependency ordering
- **Parallel plan execution:** Plans 39-02 and 39-03 (src/ and test/ import rewiring) ran in parallel with no conflicts
- **Deferred-then-completed pattern:** STAB-04 deferred from Phase 38 to Phase 39 was the right call — docstring cleanup made more sense after module reorganization
- **Comprehensive audit:** 3-source cross-reference (VERIFICATION + SUMMARY + traceability) caught stale metadata

### What Was Inefficient
- **NDJSON format churn:** Phase 36 built NDJSON conversion, Phase 38 replaced it with standard txt+yaml — could have gone directly to txt+yaml
- **SUMMARY frontmatter still underutilized:** `one_liner` fields still null in all SUMMARYs, making automated accomplishment extraction fail
- **Phase 38 plan count confusion:** 38-03 deferred to Phase 39 but plan count shows 3/4 — ambiguous without reading the deferral note

### Patterns Established
- core/types/ as shared cross-stage type layer (stdlib + numpy only, no implementation imports)
- Dual-path migration: create new files → rewire imports → delete old — never break intermediate states
- Backend selection via config string → registry factory → lazy import

### Key Lessons
1. **Go directly to the right format** — NDJSON→txt+yaml churn was avoidable; research should validate format compatibility before building
2. **Module reorganization is easier after stabilization** — Phase 39 went smoothly because Phase 38 had already cleaned up dead code and config
3. **Populate SUMMARY frontmatter** — still not happening consistently; milestone tooling remains degraded

### Cost Observations
- Model mix: ~70% sonnet (executors), ~20% opus (orchestrator), ~10% sonnet (verifiers)
- Sessions: ~3-4 across 2 days
- Notable: Phase 39 was the most efficient — 4 plans in 3 waves completed in a single session

---

## Milestone: v3.1 — Reconstruction

**Shipped:** 2026-03-03
**Phases:** 7 | **Plans:** 13 | **Timeline:** 2 days

### What Was Built
- Diagnostic fixture system (MidlineFixture + NPZ serialization) for offline evaluation
- Evaluation harness with CalibBundle, Tier 1 reprojection error, Tier 2 leave-one-out stability
- Confidence-weighted DLT triangulation backend with outlier rejection
- Association parameter tuning infrastructure with grid sweep
- Empirical threshold tuning via evaluation harness (50.0 → 10.0)
- Dead code cleanup: removed ~3,200 lines of old reconstruction code

### What Worked
- **Evaluation-first approach:** Building the harness before the new backend meant every change was measurable — no guesswork
- **Empirical threshold tuning:** Grid search on real data outperformed manual parameter selection
- **Clean backend replacement:** Old triangulation and curve optimizer removed in a single phase with no orphaned imports
- **Inserted phase (43.1) for association tuning:** Quick insertion to investigate a real concern, properly scoped and concluded with clear findings
- **Quick tasks for targeted improvements:** 3 quick tasks (soft scoring kernel, grid sweep restructure, config unification) handled cross-cutting improvements efficiently

### What Was Inefficient
- **SUMMARY one_liner fields still not populated:** Automated accomplishment extraction continues to fail; manual extraction required at milestone completion
- **Traceability table in REQUIREMENTS.md went stale:** 8 requirements showed "Pending" status despite checkboxes being checked — dual bookkeeping creates drift
- **Association tuning yielded marginal gains:** Two full phases of work (43.1 + quick tasks 15-16) confirmed that association params aren't the bottleneck — earlier data analysis might have revealed this sooner

### Patterns Established
- NPZ fixture pattern: versioned (v1.0/v2.0) with flat slash-separated keys for numpy.load compatibility
- Evaluation harness as gatekeeper: no reconstruction change ships without Tier 1/Tier 2 measurement
- Single-backend simplicity: one reconstruction backend (DLT) instead of registry with multiple backends
- Parameter tuning via scripts + eval harness, not manual experimentation

### Key Lessons
1. **Build evaluation infrastructure first** — the harness paid for itself immediately; every subsequent phase used it for validation
2. **Data analysis before parameter sweeps** — the ~70% singleton rate meant association tuning had a low ceiling; analyzing the data distribution first would have saved effort
3. **Single-backend simplicity beats multi-backend flexibility** — removing the curve optimizer and backend registry simplified the codebase with no capability loss
4. **Populate SUMMARY frontmatter** — this is the fourth milestone where automated extraction fails; needs to be enforced in the plan execution workflow

### Cost Observations
- Model mix: ~60% sonnet (executors), ~30% opus (orchestrator), ~10% haiku (verifiers)
- Sessions: ~5-6 across 2 days
- Notable: Quick tasks 15-17 handled 3 improvements in a single session with minimal overhead

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Timeline | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 11 days | 12 | Initial development; architecture pivot mid-milestone |
| v2.0 | 3 days | 10 | Full architectural refactor; audit-then-remediate pattern |
| v3.0 | 2 days | 5 | Ultralytics migration; incremental file relocation strategy |
| v3.1 | 2 days | 7 | Reconstruction rebuild with evaluation-first approach |

### Cumulative Quality

| Milestone | LOC | Tests | Quick Tasks |
|-----------|-----|-------|-------------|
| v1.0 | 50,802 | ~300 | 8 |
| v2.0 | 18,660 + 14,826 test | 514 | 1 |
| v3.0 | 22,087 + 18,829 test | 656 | 3 |
| v3.1 | 19,493 source | - | 3 |

### Top Lessons (Verified Across Milestones)

1. Invest in input quality (segmentation) before building downstream pipelines
2. Pivot early when runtime measurements invalidate an approach
3. Align on canonical domain models before coding — mismatches compound
4. Audit phases produce better remediation than fix-as-you-go
5. Go directly to the right format — intermediate format churn wastes effort (v3.0 NDJSON→txt lesson)
6. Build evaluation infrastructure before making changes — every change should be measurable (v3.1)
