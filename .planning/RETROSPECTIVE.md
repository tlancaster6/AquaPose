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
3. **Z-reconstruction is fundamentally limited by top-down geometry** — ~11x Z/XY anisotropy (revised from early 132x estimate) means XY-only approaches are often superior to full 3D
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

## Milestone: v3.2 — Evaluation Ecosystem

**Shipped:** 2026-03-03
**Phases:** 5 | **Plans:** 11 | **Timeline:** 1 day

### What Was Built
- Per-stage pickle caching system with StaleCacheError and ContextLoader for sweep isolation
- Five typed stage evaluators (detection, tracking, association, midline, reconstruction) with frozen metric dataclasses
- `aquapose eval` CLI for multi-stage quality reports (human-readable + JSON)
- `aquapose tune` CLI with grid sweeps, two-tier validation, and config diff output
- Partial pipeline execution via `--resume-from` and `initial_context`
- Full removal of legacy evaluation machinery (harness.py, NPZ export, 3 standalone scripts)

### What Worked
- **Clean dependency chain:** Phase ordering (46→47→48→49→50) had perfect dependency isolation — each phase consumed only what the previous produced
- **Evaluation-first pattern continued:** Building stage evaluators (Phase 47) before the runner (Phase 48) and tuner (Phase 49) meant each layer had tested foundations
- **Single-day velocity:** 5 phases and 11 plans completed in one day — tight scope and well-understood domain from v3.1 foundations
- **Zero engine imports in evaluators:** The constraint that stage evaluators take explicit params (not config objects) paid off — evaluators are testable with synthetic data and reusable
- **Aggressive cleanup in final phase:** Deleting legacy code immediately (not deprecating with shims) eliminated maintenance burden

### What Was Inefficient
- **SUMMARY one_liner fields still not populated:** Fifth consecutive milestone where automated accomplishment extraction fails — pattern is now endemic
- **NPZ fixture system built in v3.1 then removed in v3.2:** The MidlineFixture + NPZ serialization pattern (Phase 41) was replaced by per-stage pickle caches — could have gone directly to pickle caches if v3.2 had been planned first
- **STATE.md velocity metrics incomplete:** Only 2 of 5 phases had timing data — metric tracking degraded mid-milestone

### Patterns Established
- Per-stage pickle cache envelope format: `{run_id, timestamp, stage_name, version_fingerprint, context}`
- ContextLoader shallow copy for sweep isolation (immutable-by-convention stage outputs)
- Stage evaluator pattern: pure function + frozen dataclass metrics + DEFAULT_GRID colocation
- Two-tier validation: fast sweep (few frames) → thorough top-N (many frames) → config diff
- Inline imports in CLI commands to avoid top-level engine coupling

### Key Lessons
1. **Plan evaluation infrastructure across milestones** — NPZ fixtures built in v3.1 were immediately replaced; coordinating v3.1+v3.2 design upfront would have avoided the throwaway work
2. **Aggressive deletion is better than deprecation** — removing legacy code entirely (not shimming) resulted in a cleaner codebase with no confusion about which path to use
3. **Per-stage caching enables efficient sweeps** — pickle caches as the single evaluation data source eliminated the monolithic NPZ bottleneck and enabled partial pipeline execution
4. **Enforce SUMMARY frontmatter in tooling** — five milestones of unfilled one_liner fields means the tooling should either auto-fill or block plan completion until populated

### Cost Observations
- Model mix: ~60% sonnet (executors), ~30% opus (orchestrator), ~10% haiku (verifiers)
- Sessions: 1-2 across 1 day
- Notable: Most efficient milestone yet — 11 plans in a single day with full audit passing 22/22 requirements

---

## Milestone: v3.3 — Chunk Mode

**Shipped:** 2026-03-05
**Phases:** 5 | **Plans:** 11 | **Timeline:** 2 days

### What Was Built
- FrameSource protocol + VideoFrameSource — injectable frame source replacing VideoSet
- ChunkOrchestrator processing video in fixed-size temporal chunks above PosePipeline
- ChunkHandoff frozen dataclass carrying tracker state + identity map across chunk boundaries
- Identity stitching mapping chunk-local fish IDs to globally consistent IDs via track ID continuity
- Per-chunk diagnostic caches (chunk_NNN/cache.pkl + manifest.json) with chunk-aware eval/tuning
- Visualization migrated from engine observers to `aquapose viz` CLI in evaluation suite

### What Worked
- **Layered architecture:** Frame source (51) → orchestrator (52) → CLI wiring (53) → diagnostics/eval (54) → gap closure (55) had clean dependency ordering with each phase building on the previous
- **Gap closure phase pattern:** Phase 55 explicitly created from audit findings worked well — targeted fixes with clear scope
- **Diagnostic + chunk mode co-existence:** Removing mutual exclusion in Phase 54 was the right call — per-chunk cache layout made it natural
- **Single-cache-per-chunk simplification:** Replacing per-stage cache files with one cache.pkl per chunk reduced complexity for both writing and reading

### What Was Inefficient
- **SUMMARY one_liner fields still not populated:** Sixth consecutive milestone where automated accomplishment extraction fails — now firmly an endemic process gap
- **Phase 53 scope underestimated:** Originally planned as 3 success criteria but only got 1 plan; Phase 54 and 55 had to close the remaining work
- **Phase 53 missing VERIFICATION.md:** Had to be covered retroactively by Phase 55 — verification should happen per-phase
- **54-VERIFICATION.md inaccurate about viz CLI shape:** Described subcommands but actual implementation uses flags — verification report quality control needed

### Patterns Established
- ChunkOrchestrator as layer above PosePipeline — owns chunk loop, identity stitching, HDF5 output
- ChunkHandoff in core/ (not engine/) to avoid circular imports between core/tracking and engine
- Per-chunk single cache layout: diagnostics/chunk_NNN/cache.pkl + manifest.json
- Dual loader pattern: load_run_context() for merged eval context; load_all_chunk_caches() for per-chunk viz
- Post-run visualization via evaluation suite CLI instead of in-pipeline observers

### Key Lessons
1. **Scope all integration work upfront** — Phase 53 was too thinly planned; diagnostic/eval migration should have been anticipated in the original 3-phase scope
2. **Verify each phase as it completes** — skipping Phase 53 verification created audit gaps that required a gap-closure phase
3. **Design for evolution** — INTEG-02 (mutual exclusion) was correct at Phase 53 but wrong by Phase 54; requirements should note when they may be superseded
4. **Populate SUMMARY frontmatter** — six milestones in, this is still not happening; enforce in tooling or accept it as a known gap

### Cost Observations
- Model mix: ~60% sonnet (executors), ~30% opus (orchestrator), ~10% haiku (verifiers)
- Sessions: ~3-4 across 2 days
- Notable: Phase 54 was the largest (4 plans) — visualization migration was the most complex subsystem change

---

## Milestone: v3.4 — Performance Optimization

**Shipped:** 2026-03-05
**Phases:** 5 | **Plans:** 8 | **Timeline:** 1 day

### What Was Built
- Vectorized association scoring via NumPy broadcasting (3.8x speedup)
- Vectorized DLT reconstruction via batched `torch.linalg.lstsq` (7.0x speedup)
- Background-thread frame prefetch in ChunkFrameSource
- Batched YOLO inference for detection (11.5x) and midline (8.1x) with OOM retry
- End-to-end performance validation script and report (8.2x total speedup)

### What Worked
- **Profile-driven prioritization:** Targeting the four profiled bottlenecks by impact-to-complexity ratio ensured maximum ROI per phase
- **Correctness-neutral optimization:** All changes verified against existing `aquapose eval` harness — no regressions introduced
- **OOM retry pattern:** Automatic batch halving with state persistence handled GPU memory limits gracefully without pipeline crashes
- **Transparent optimizations:** Phases 56 and 57 were internal refactors with zero API surface changes — no downstream code modifications needed
- **Fastest milestone yet:** 5 phases and 8 plans completed in a single day

### What Was Inefficient
- **SUMMARY one_liner fields still not populated:** Seventh consecutive milestone — this is now a permanent process gap
- **Deferred real-data eval comparisons:** SC-2 (Phase 56) and SC-5 (Phase 59) deferred real-data comparisons due to missing pre-optimization baselines — should have captured baselines before starting optimization work
- **REQUIREMENTS.md checkboxes stale:** ASSOC-01/ASSOC-02 unchecked despite being satisfied — dual bookkeeping continues to drift

### Patterns Established
- Scalar-to-batch vectorization: element-wise `sum(axis=1)` replaces `np.dot` for batched dot products
- OOM retry pattern: catch CUDA OOM, halve batch, retry from scratch, persist in BatchState
- Prefetch pattern: daemon thread + bounded queue + sentinel + stop_event for cooperative shutdown
- Collect-predict-redistribute: CPU crop extraction separated from GPU batch inference for clean retry boundaries

### Key Lessons
1. **Capture baselines before optimizing** — deferred real-data comparisons could have been avoided by running eval before starting the optimization work
2. **Transparent optimizations are the best optimizations** — Phases 56/57 changed internal implementations with zero API surface changes, requiring no downstream modifications
3. **Profile data is essential for prioritization** — without the py-spy profiling data, phase ordering would have been guesswork
4. **GPU non-determinism is inherent to batching** — batched vs serial YOLO inference produces slightly different results; this is expected, not a bug

### Cost Observations
- Model mix: ~60% sonnet (executors), ~30% opus (orchestrator), ~10% haiku (verifiers)
- Sessions: 1-2 across 1 day
- Notable: Most compact milestone — 8 plans in a single day with 8.2x speedup delivered

---

## Milestone: v3.5 — Pseudo-Labeling

**Shipped:** 2026-03-06
**Phases:** 9 | **Plans:** 22 | **Timeline:** 2 days

### What Was Built
- Z-denoising pipeline: centroid z-flattening + temporal Gaussian smoothing for clean 3D reprojections
- Pseudo-label generation: Source A (consensus reprojection) and Source B (gap-fill) with confidence scoring
- Gap detection via inverse LUT visibility, tagged by failure reason (no-detection, no-tracklet, failed-midline)
- Elastic midline deformation augmentation: TPS-based C-curve/S-curve variants reducing curvature bias
- SQLite sample store: content-hash dedup, provenance tracking, symlink-based assembly, model lineage
- Workflow-oriented CLI: project-aware resolution, run shorthand, deprecated command removal

### What Worked
- **Scope expansion handled cleanly:** Milestone grew from 6 phases (61-66) to 9 phases (61-69) by adding augmentation, sample store, and CLI cleanup — each addition was properly planned and executed without disrupting the original scope
- **Quick tasks for cross-cutting fixes:** 4 quick tasks (18-21) fixed pseudo-label output format, frame selection wiring, COCO interchange, and metadata ingestion without full phase overhead
- **TDD for complex modules:** SampleStore (Phase 68) was built test-first, catching edge cases in dedup, upsert, and cascade delete before integration
- **Elastic augmentation A/B experiment:** Validated augmentation effectiveness (OKS slope -0.71 to -0.30) before committing to infrastructure
- **CLI cleanup as final phase:** Reworking commands after all functionality was built avoided rework during the cleanup

### What Was Inefficient
- **SUMMARY one_liner fields still not populated:** Eighth consecutive milestone — confirmed as permanent process gap
- **Early phases (61-63) lack VERIFICATION.md:** Verification workflow wasn't established until Phase 64 — 14 requirements marked unsatisfied/partial in audit
- **Z-denoising implementation diverged from spec:** Requirements specified IRLS plane fit + plane normals; actual implementation used simpler centroid z-flattening — SUMMARYs describe the spec, not the implementation
- **Frame selection functions orphaned:** `frame_selection.py` functions built in Phase 65 were orphaned when Phase 69 deleted `dataset_assembly.py` — the replacement path doesn't use them
- **AUG requirements missing from traceability:** Phase 67 requirements (AUG-01..06) defined in ROADMAP but never added to REQUIREMENTS.md traceability table

### Patterns Established
- Source A/B pseudo-label separation: consensus reprojections vs gap-fill with distinct metadata and confidence thresholds
- Confidence composite scoring: 50% residual + 30% camera count + 20% per-camera variance
- Content-hash dedup in sample store: SHA-256 of image content, not path, for reliable deduplication
- Source priority upsert: manual(2) > corrected(1) > pseudo(0) — higher-quality labels always win
- Project-aware CLI: `--project` flag with CWD walk-up fallback for automatic project detection
- Run shorthand resolution: `latest`, timestamp suffix, negative index, or full path

### Key Lessons
1. **Establish verification workflow before first phase** — missing VERIFICATION.md for Phases 61-63 created 14 audit gaps that could have been avoided
2. **Keep SUMMARYs aligned with implementation, not spec** — Phase 61 SUMMARYs describe IRLS plane fit but implementation uses centroid z-flattening; misleading for future readers
3. **Validate augmentation before building infrastructure** — the A/B experiment in Phase 67 confirmed value before committing; this should be standard for experimental features
4. **Plan for scope growth** — milestones that add phases mid-execution work fine if each addition is properly scoped; don't resist scope growth when it's well-motivated
5. **Delete dead code immediately** — orphaned frame_selection.py functions survived because Phase 69 didn't audit consumers; cleanup phases should check imports

### Cost Observations
- Model mix: ~60% sonnet (executors), ~30% opus (orchestrator), ~10% haiku (verifiers)
- Sessions: ~6-8 across 2 days
- Notable: Largest milestone by phase count (9 phases, 22 plans) completed in 2 days — scope expansion didn't slow velocity

---

## Milestone: v3.6 — Model Iteration & QA

**Shipped:** 2026-03-10
**Phases:** 8 (1 skipped) | **Plans:** 13 | **Timeline:** 5 days

### What Was Built
- Extended evaluation metrics: percentiles, per-keypoint breakdown, curvature-stratified quality, track fragmentation
- Data store bootstrap: temporal split, tagged exclusions, baseline OBB and pose models trained and registered
- Full pseudo-label iteration loop: generate, diversity-select, CVAT correct, retrain, evaluate
- A/B curation comparison quantifying the value of light human curation (+9.2pts mAP50-95)
- eval-compare CLI for round-over-round pipeline metric comparison
- Training module consolidation: unified train_yolo(), seg registration fix, test coverage

### What Worked
- **Decision checkpoint pattern:** Phase 74's go/no-go on round 2 was the right structure — quantitative comparison with clear rationale prevented unnecessary work
- **A/B curation comparison:** Quantifying curation value (+9.2pts) made the CVAT review step data-driven rather than faith-based
- **Phase 77 as code quality phase:** Running independently of the iteration loop kept the main workflow unblocked while improving the training module
- **Quick task for NMS fix:** Replacing Ultralytics probiou with geometric polygon NMS (quick task 23) was cleanly scoped and immediately beneficial
- **Diversity selection for pseudo-labels:** Curvature-stratified and camera-balanced selection produced representative training subsets without manual curation of the selection process

### What Was Inefficient
- **SUMMARY one_liner fields still not populated:** Ninth consecutive milestone — confirmed as permanent process gap that milestone tooling cannot rely on
- **Phase 73 executed manually without SUMMARYs:** CVAT curation workflow was human-driven; plan execution artifacts (SUMMARYs) were never written. Verification still passed but audit required manual reconstruction of what happened
- **REQUIREMENTS.md checkboxes stale:** 8 requirements unchecked despite being satisfied — dual bookkeeping between checkboxes and traceability table continues to drift
- **Store provenance deviated from plan:** Phase 73 used source=corrected instead of source=pseudo with round=1 metadata — the manual execution skipped the planned import commands. Future iterations can't filter by round
- **correction_report.json never created:** Phase 73 planned per-sample correction magnitude quantification (IoU, displacement) but the artifact was never produced during CVAT curation

### Patterns Established
- Decision checkpoint after each iteration round: quantitative comparison → documented rationale → go/no-go
- Diversity selection via curvature-stratified + camera-balanced sampling for pseudo-label subsets
- eval-compare as standard post-run comparison tool (distinct from train compare for training metrics)
- Consolidated YOLO training: single train_yolo() entry point with model_type dispatch
- Shared _run_training() CLI orchestrator handling registration for all model types

### Key Lessons
1. **Human-in-the-loop phases need explicit artifact requirements** — Phase 73 CVAT curation was done manually but plan execution infrastructure (SUMMARY writing, store provenance) was skipped; manual phases should have a "re-entry checklist" for returning to GSD workflow
2. **Quantify human curation value before committing to review workflows** — the A/B comparison justified CVAT review; without it, the effort would have been faith-based
3. **Conditional phases work well with decision checkpoints** — skipping Phase 75 was clean because Phase 74 had a structured decision framework
4. **Code quality phases pay for themselves** — Phase 77 fixed the seg registration bug that would have caused silent data loss in future training runs
5. **SUMMARY frontmatter is permanently unfilled** — nine milestones of empty one_liner fields; either enforce in tooling or remove from templates

### Cost Observations
- Model mix: ~60% sonnet (executors), ~30% opus (orchestrator), ~10% haiku (verifiers)
- Sessions: ~8-10 across 5 days (includes significant human CVAT time)
- Notable: Phase 73 was the most human-intensive — CVAT curation was the bottleneck, not AI execution

---

## Milestone: v3.7 — Improved Tracking

**Shipped:** 2026-03-11
**Phases:** 11 (1 skipped, 2 inserted) | **Plans:** 18 | **Timeline:** 2 days

### What Was Built
- Custom OKS-based keypoint tracker replacing OC-SORT/BoxMot (24-dim KF, OCM, ORU/OCR recovery)
- Pipeline reordered: Detection → Pose → Tracking → Association → Reconstruction
- Segmentation midline backend fully removed
- Production OBB and Pose models retrained with all-source stratified data
- BoxMot dependency completely removed
- Tracker tuned to 27 tracks (vs OC-SORT 30) with 95% coverage

### What Worked
- **Investigation-before-action pattern:** Phase 78 occlusion investigation gave GO recommendation, avoiding unnecessary remediation work
- **Production retrain as inserted phase:** 78.1 was cleanly scoped and delivered immediately useful models
- **Complete dependency removal:** Removing BoxMot entirely (not just wrapping) eliminated a class of maintenance problems

### What Was Inefficient
- **ASSOC-01 implemented then deleted:** Keypoint centroid association was built for ocsort_wrapper, then deleted when BoxMot was removed — wasted effort
- **Phase count inflation:** 11 phases for what was essentially tracker replacement + model retrain + cleanup — could have been tighter

### Key Lessons
1. **Don't build features for code you're about to delete** — ASSOC-01 targeting ocsort_wrapper was wasted when BoxMot was removed one phase later
2. **Complete dependency removal is worth the extra effort** — leaving BoxMot as fallback would have created long-term maintenance burden

---

## Milestone: v3.8 — Improved Association

**Shipped:** 2026-03-12
**Phases:** 7 (1 inserted) | **Plans:** 12 | **Timeline:** 2 days

### What Was Built
- Multi-keypoint pairwise scoring replacing single-centroid ray casting
- Group validation with temporal changepoint detection
- Singleton recovery with swap-aware split-and-assign (27% → 5.4%)
- Association wall-time 452s → <30s via batch ray casting vectorization
- Fragment merging removed; cleaner pipeline

### What Worked
- **Architecture-first approach:** Gains came from structural changes (multi-keypoint scoring, changepoint detection), not parameter tuning — 27-combo sweep confirmed defaults already optimal
- **Bottleneck investigation as inserted phase (91.1):** Identified vectorization opportunity for 15x speedup
- **Remove-before-add pattern:** Deleting fragment merging (Phase 89) before adding validation (Phase 90) simplified the pipeline first

### What Was Inefficient
- **Phase 88 missing VERIFICATION.md and SUMMARY frontmatter:** Bookkeeping gap that required manual audit reconstruction

### Key Lessons
1. **Architectural improvements beat parameter tuning** — 80% singleton reduction came from richer signals, not tweaked thresholds
2. **Profile bottlenecks during development** — the 452s→30s speedup was discovered via an inserted investigation phase; profiling earlier would have caught it sooner

---

## Milestone: v3.9 — Reconstruction Modernization

**Shipped:** 2026-03-14
**Phases:** 4 | **Plans:** 5 | **Timeline:** 2 days

### What Was Built
- `n_sample_points` config wired end-to-end, default changed from 15 to 6
- Dead scalar triangulation code (~170 lines) removed from DltBackend
- Raw-keypoint reconstruction as primary output; B-spline fitting optional via `spline_enabled` toggle
- HDF5 dual-dataset layout (points + control_points, NaN-filled when unused)
- Z-denoising CLI fixed for raw-keypoint mode (NaN-safe dual shift)
- All reconstruction docstrings updated for keypoint-native variable-point-count output

### What Worked
- **Tight scope:** 4 phases, 5 plans, all completed in ~36 minutes total execution time — the most focused milestone yet
- **TDD discipline:** Every phase started with failing tests, then implementation — zero regressions
- **Clean phase ordering:** Independent phases (93, 94) followed by dependent phases (95→96) allowed natural progression
- **Auto-fix pattern:** Type errors from making Midline3D fields optional were caught by typecheck and fixed in-task — no separate bug-fix phases needed
- **Backward compatibility by default:** NaN-fill dual-dataset and None-returning legacy readers ensured no existing code broke

### What Was Inefficient
- **Tech debt items identified late:** INT-01/INT-02 (hardcoded n_points=15 in evaluation) were found by audit after all phases completed — could have been caught by reading downstream consumers during Phase 93 planning
- **Audit identified doc inconsistency in config.py** that was already stale from Phase 93's change — the phase should have caught its own docstring

### Patterns Established
- Optional-field dataclass migration: make fields Optional with None defaults, add None guards at all access sites
- NaN-fill dual-dataset HDF5 pattern: both representations always present, unused one NaN-filled
- Backward-compat reader pattern: check dataset existence, return None for legacy files
- Config toggle pattern: `spline_enabled` flows from config through constructor chain to algorithm branch

### Key Lessons
1. **Read downstream consumers when changing defaults** — changing n_sample_points default from 15 to 6 should have triggered a grep for hardcoded `15` across the entire codebase, not just the reconstruction module
2. **Small, focused milestones execute cleanly** — 4 phases with clear dependencies and tight scope resulted in zero rework and fast execution
3. **Audit before completion catches integration gaps** — INT-02 (evaluation hardcoded n_points) would have been a production issue without the milestone audit

### Cost Observations
- Model mix: ~70% sonnet (executors), ~20% opus (orchestrator), ~10% haiku (verifiers)
- Sessions: 2 across 2 days
- Notable: Most efficient per-phase execution — ~8 min average per plan, all plans zero-deviation

---

## Milestone: v3.10 — Publication Metrics

**Shipped:** 2026-03-15
**Phases:** 5 | **Plans:** 5

### What Was Built
- Full 9,450-frame diagnostic pipeline run (32 chunks, 12 cameras, 9 fish) with production models
- Per-stage timing extraction: 1.14 fps end-to-end, detection+pose dominate at 59.6%
- Reconstruction quality metrics: 3.41px mean reprojection error, p99=14.41px, 3.60 mean cameras/fish
- Tracking and association metrics: 1,932 tracklets, 12.1% singleton rate, 53 unique IDs across chunks
- Consolidated results document with 11 sections and 11 supporting CSVs

### What Worked
- **Existing eval infrastructure carried the milestone** — Phases 98-100 ran `aquapose eval` on cached data without writing new code (only 2 source files changed, +70 lines). The v3.2 evaluation ecosystem investment paid off.
- **Parallel phase execution** — Phases 98, 99, 100 all depended only on Phase 97, enabling independent execution without ordering constraints.
- **Auto-advance pipeline** — Plan → verify → execute → verify chain ran smoothly with `--auto` flag, reducing orchestration overhead.
- **CSV + markdown dual output** — Raw data in CSVs alongside narrative in performance-accuracy.md makes metrics both machine-readable and human-readable.

### What Was Inefficient
- **Phase 97 missing VERIFICATION.md** — Execution-only phase (no code changes) wasn't formally verified. The run output was consumed by all downstream phases, confirming success empirically, but the missing artifact created an audit flag.
- **Uncommitted runner.py change** — A Phase 99 code change (n_sample_points config read) was left unstaged. Should have been committed during Phase 99 execution.
- **Section numbering mismatch** — Phase 100 plan specified Section 10 but Phase 98 had already claimed it. Not a real problem (executor adapted correctly) but the plan was written with stale assumptions.

### Patterns Established
- Metrics-only milestones require minimal code changes — invest in evaluation infrastructure early
- Full-run metrics should be attributed to a specific run directory for reproducibility
- CSV companion files for every metrics section enable downstream analysis

### Key Lessons
1. **Execution-only phases still need verification artifacts** — even if "no code changed," the verification step catches missing documentation
2. **Metrics milestones validate prior infrastructure decisions** — v3.10 proved the v3.2 eval ecosystem, v3.3 chunk caching, and v3.4 performance investments were sound
3. **Auto-advance works well for small, sequential milestones** — 5 simple phases with clear dependencies executed without manual intervention

### Cost Observations
- Model mix: ~60% sonnet (executors/verifiers), ~40% opus (orchestrator)
- Sessions: 1 (all 5 phases in single session)
- Notable: Fastest milestone relative to scope — mostly eval + documentation, minimal code

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Timeline | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 11 days | 12 | Initial development; architecture pivot mid-milestone |
| v2.0 | 3 days | 10 | Full architectural refactor; audit-then-remediate pattern |
| v3.0 | 2 days | 5 | Ultralytics migration; incremental file relocation strategy |
| v3.1 | 2 days | 7 | Reconstruction rebuild with evaluation-first approach |
| v3.2 | 1 day | 5 | Evaluation ecosystem; per-stage caching + CLI tools |
| v3.3 | 2 days | 5 | Chunk processing; frame source abstraction + viz migration |
| v3.4 | 1 day | 5 | Performance optimization; 8.2x speedup via batching + vectorization |
| v3.5 | 2 days | 9 | Pseudo-labeling infrastructure; sample store; CLI cleanup |
| v3.6 | 5 days | 8 | Model iteration loop; A/B curation comparison; code quality |
| v3.7 | 2 days | 11 | Custom keypoint tracker; segmentation removal; pipeline reorder |
| v3.8 | 2 days | 7 | Multi-keypoint association; singleton rate 27%→5.4% |
| v3.9 | 2 days | 4 | Keypoint-native reconstruction; spline optional; dead code removal |
| v3.10 | 29 days | 5 | Publication metrics; eval-only milestone; auto-advance pipeline |

### Cumulative Quality

| Milestone | LOC | Tests | Quick Tasks |
|-----------|-----|-------|-------------|
| v1.0 | 50,802 | ~300 | 8 |
| v2.0 | 18,660 + 14,826 test | 514 | 1 |
| v3.0 | 22,087 + 18,829 test | 656 | 3 |
| v3.1 | 19,493 source | - | 3 |
| v3.2 | 20,789 source | ~788 | 0 |
| v3.3 | 21,634 source | ~807 | 0 |
| v3.4 | 22,754 source | ~840 | 0 |
| v3.5 | 28,033 source | - | 4 |
| v3.6 | 30,480 source | ~1142 | 3 |
| v3.7 | 29,525 source | ~1159 | 0 |
| v3.8 | 31,066 source | ~1200 | 0 |
| v3.9 | 31,188 source | ~1208 | 0 |
| v3.10 | 31,268 source | ~1208 | 0 |

### Top Lessons (Verified Across Milestones)

1. Invest in input quality (segmentation) before building downstream pipelines
2. Pivot early when runtime measurements invalidate an approach
3. Align on canonical domain models before coding — mismatches compound
4. Audit phases produce better remediation than fix-as-you-go
5. Go directly to the right format — intermediate format churn wastes effort (v3.0 NDJSON→txt lesson)
6. Build evaluation infrastructure before making changes — every change should be measurable (v3.1)
7. Plan infrastructure across milestones to avoid throwaway work (v3.1 NPZ → v3.2 pickle replacement)
8. Aggressive deletion beats deprecation — remove legacy code entirely, don't shim (v3.2)
9. Profile before optimizing — target bottlenecks by measured impact, not intuition (v3.4)
10. Capture baselines before starting optimization work — deferred comparisons create verification gaps (v3.4)
11. Establish verification workflow before first phase — missing VERIFICATION.md creates audit gaps (v3.5)
12. Validate experimental features with A/B experiments before building infrastructure (v3.5)
13. Human-in-the-loop phases need explicit artifact re-entry checklists — manual execution skips GSD workflow artifacts (v3.6)
14. Quantify human curation value before committing to review workflows — A/B comparison justified CVAT review (v3.6)
15. Don't build features for code you're about to delete — wasted effort on soon-to-be-removed modules (v3.7)
16. Architectural improvements beat parameter tuning — richer signals outperform tweaked thresholds (v3.8)
17. Read downstream consumers when changing defaults — grep for hardcoded values across the entire codebase (v3.9)
18. Small focused milestones execute cleanly — tight scope with clear dependencies results in zero rework (v3.9)
19. Execution-only phases still need verification artifacts — missing docs create audit flags even when output is correct (v3.10)
20. Metrics milestones validate prior infrastructure decisions — they prove (or disprove) that eval/caching/perf investments were sound (v3.10)
