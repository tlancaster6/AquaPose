# v2.0 Alpha Retrospective: AquaPose

**Period:** 2026-02-24 to 2026-02-27
**Scope:** Phases 13-20 — the full v2.0 Alpha refactor
**Author:** Phase 21 executor, 2026-02-27

---

## Executive Summary

The v2.0 Alpha refactor transformed AquaPose from a script-driven scientific pipeline into an event-driven computation engine with strict architectural layering. Over approximately three days and 138 commits, Phases 13 through 20 replaced the v1.0 `pipeline/orchestrator.py`-centric design with a 5-stage canonical pipeline governed by `PosePipeline`, enforced import boundaries between engine and computation layers, a typed event system for observer-based side effects, a frozen dataclass config hierarchy, and a working CLI. The refactor closed with a structured audit (Phase 19) and a remediation pass (Phase 20) that resolved all critical and warning findings. The codebase now has 514 passing unit tests, 80 source files across 10 modules, and approximately 18,660 lines of source code. All four execution modes (production, diagnostic, synthetic, benchmark) and both reconstruction backends (triangulation, curve optimizer) were verified against real video data. The architecture is sound and publication-ready from a software engineering standpoint; the primary remaining quality bottleneck is U-Net segmentation accuracy (IoU 0.623 vs. the 0.90 target), which constrains downstream 2D midline quality and thus 3D reconstruction precision.

---

## Architecture Assessment

The v2.0 Alpha was designed against a detailed architectural guidebook. The table below rates each major concern against its implementation outcome.

### Stage Protocol and Structural Typing

**Rating: Fully Achieved**

The `Stage` protocol is defined in `src/aquapose/core/context.py` using `typing.Protocol` with `runtime_checkable`. Any class with a `run(context: PipelineContext) -> None` method satisfies it without inheritance. Five production stages (`DetectionStage`, `MidlineStage`, `AssociationStage`, `TrackingStage`, `ReconstructionStage`) and the `SyntheticDataStage` adapter all implement it through structural typing. The protocol-over-inheritance pattern was adopted cleanly; no stage inherits from a base class.

### PipelineContext as Typed Accumulator

**Rating: Fully Achieved**

`PipelineContext` is a dataclass in `core/context.py` that accumulates typed results as each stage writes its output. Fields are strictly typed: `detections` (`list[dict[str, list[Detection]]]`), `annotated_detections`, `associated_bundles`, `tracks`, `midlines_3d`. No stage reads fields it did not produce or that a prior stage was not responsible for populating. The original placement in `engine/stages.py` was identified as an IB-003 violation during the Phase 19 audit and corrected in Phase 20-01 by moving `PipelineContext` and `Stage` to `core/context.py`, where they belong as pure data contracts.

### Event System and Observer Pattern

**Rating: Fully Achieved**

The event system uses typed dataclasses (`PipelineStart`, `PipelineComplete`, `PipelineFailed`, `StageStart`, `StageComplete`, and domain-level events) with an `EventBus` that dispatches synchronously along the `__mro__` chain. Observers subscribe by event type; subscribing to the `Event` base receives all subtypes. Dispatch is fault-tolerant — observer exceptions are logged but never re-raised, preserving pipeline determinism. All five observers (timing, HDF5 export, 2D reprojection overlay, 3D animation, diagnostics) are implemented as purely additive side effects. The observer assembly was extracted from `cli.py` to `engine/observer_factory.py` during Phase 20-03, thinning the CLI to 161 LOC.

### Import Boundary Enforcement

**Rating: Fully Achieved (with documented exceptions)**

The core rule — engine imports computation modules, never reverse — is enforced by an automated import boundary checker (`tools/import_boundary_checker.py`) wired as a pre-commit hook. The checker implements four rules:
- IB-001: no runtime `core/` → `engine/` imports — **0 violations**
- IB-002: no `engine/` → outside `core/` — **0 violations**
- IB-003: no `TYPE_CHECKING` backdoors in core — **0 violations** (fixed in Phase 20-01)
- IB-004: no legacy computation dirs importing `engine/` — **0 violations**

The seven IB-003 violations present in the Phase 19 audit snapshot (all five core stage files importing `PipelineContext` from `engine/stages.py` under `TYPE_CHECKING`) were resolved by moving `PipelineContext` to `core/context.py`. Two accepted exceptions in observer files (`animation_observer.py` and `overlay_observer.py` importing `SPLINE_K, SPLINE_KNOTS` constants from `reconstruction/triangulation`) are documented in the checker's allowlist and classified SR-002 rather than IB violations.

### Config Hierarchy

**Rating: Fully Achieved**

The config hierarchy follows the guidebook's flow: defaults → YAML file → CLI overrides → freeze. All config objects are frozen dataclasses. `load_config()` in `engine/config.py` (436 LOC) accepts a YAML path and keyword overrides using both old key names (backward compatibility with pre-v2.0 YAML files) and new names. The config is serialized to `config.yaml` as the very first artifact written before any stage runs, satisfying ENG-08. An `aquapose init-config` CLI command was added (Phase 21 quick task) to generate a default template YAML.

### 5-Stage Canonical Pipeline Model

**Rating: Fully Achieved**

The canonical 5-stage model (Detection → Midline → Association → Tracking → Reconstruction) replaced the 7-stage model from early planning. The correction was formalized in Phase 14.1, which aligned all planning documents and the engine code to the guidebook's Section 6 definition. The Midline stage subsumes segmentation and skeletonization; the Reconstruction stage subsumes triangulation and B-spline fitting. Stage requirements were correspondingly reduced from STG-01..07 to STG-01..05, reducing total v2.0 requirements from 24 to 22.

A significant architectural improvement landed in Phase 20-04: the Tracking stage was rewritten to consume `context.associated_bundles` from Stage 3 rather than re-running cross-view association internally via `discover_births()`. This eliminated a full O(N²) RANSAC association pass per frame that Stage 4 had been running redundantly. The `FishTracker.update_from_bundles()` method was added to support this, preserving full population lifecycle logic (probationary, confirmed, coasting, dead ID recycling) while accepting the cleaner bundle-based input.

---

## DoD Gate Assessment

The guidebook's Section 16 defines 9 DoD criteria. Results as of Phase 20 completion (all remediations applied):

| # | Criterion | Pre-Phase-20 | Post-Phase-20 |
|---|-----------|-------------|--------------|
| 1 | Exactly one canonical pipeline entrypoint | PASS | PASS |
| 2 | All scripts invoke PosePipeline (not stage functions directly) | PASS | PASS |
| 3 | `aquapose run` produces 3D midlines | PASS | PASS |
| 4 | Diagnostic functionality via observers | PASS | PASS |
| 5 | Synthetic mode runs through the pipeline (stage adapter) | PASS | PASS |
| 6 | Timing, HDF5, viz, diagnostics as observers | PASS | PASS |
| 7 | No stage imports dev tooling (IB-003) | FAIL (7 violations) | PASS |
| 8 | CLI is a thin wrapper | FAIL (244 LOC) | PASS (161 LOC) |
| 9 | No script calls stage functions directly | PASS | PASS |

All 9 criteria pass as of Phase 20 completion. The two failures from the Phase 19 audit snapshot were fully remediated: IB-003 violations by moving `PipelineContext` to `core/`, and CLI bloat by extracting observer assembly to `engine/observer_factory.py`.

---

## Code Health Metrics

### Source Codebase

| Metric | Value |
|--------|-------|
| Source modules (top-level) | 10 (`calibration`, `core`, `engine`, `io`, `reconstruction`, `segmentation`, `synthetic`, `tracking`, `visualization`, `cli.py`) |
| Source files (`.py`) | 80 |
| Source lines of code | ~18,660 |
| New in v2.0 (`core/`, `engine/`) | ~30 files, ~6,500 LoC |

Per-module file distribution:

| Module | Files | Role |
|--------|-------|------|
| `core/` | 30 | Stage implementations, context, synthetic adapter, domain types |
| `engine/` | 12 | PosePipeline, config, events, observers, observer factory |
| `segmentation/` | 7 | YOLO/MOG2 detectors, U-Net inferrer, SAM pseudo-labeler |
| `synthetic/` | 7 | Synthetic data generation for testing |
| `visualization/` | 6 | Reprojection overlays, 3D animation, diagnostics |
| `calibration/` | 4 | AquaCal loader, refractive projection, ray casting |
| `reconstruction/` | 4 | RANSAC triangulation, B-spline fitting, curve optimizer |
| `tracking/` | 4 | FishTracker, Hungarian matching, cross-view association |
| `io/` | 4 | HDF5 writer, video set loader, camera discovery |

### Test Suite

| Metric | Value |
|--------|-------|
| Total collected tests (non-slow) | 514 |
| Total including `@slow` | 548 |
| Test files | 68 |
| Test lines of code | ~14,826 |
| Unit tests | 511 |
| Integration tests | 3 |
| e2e tests | 0 (smoke tests via `smoke_test.py` script, not pytest) |
| Regression tests | 7 (all skip when video data absent; infra correct) |
| Failures | 0 |
| Known xfail | 1 (midline regression: golden data requires PosePipeline re-run to regenerate) |

Test infrastructure additions in v2.0: import boundary checker as pre-commit hook, `smoke_test.py` CLI script that exercises all 4 modes and both backends against real video data (verified Phase 19-02), regression conftest using `AQUAPOSE_VIDEO_DIR` and `AQUAPOSE_CALIBRATION_PATH` env vars for path portability.

### Pre-Commit Hook Stack

Three categories of automated checks run on every commit:
1. **Ruff** — lint (`--fix` on commit, strict check on push) and format
2. **Pre-commit-hooks** — trailing whitespace, EOF newline, YAML validity, large file guard
3. **detect-secrets** — prevents credential commits
4. **Import boundary checker** — enforces IB-001 through IB-004 on any changed `src/aquapose/` file

The import boundary checker was introduced in Phase 19-01 and became the enforcement mechanism that made the Phase 20-01 IB-003 fix auditable — the hook would now fail on any regression.

### Known Technical Debt

Items from the Phase 19 audit that were accepted rather than remediated:

| Item | Decision | Rationale |
|------|----------|-----------|
| RANSAC non-determinism (reproducibility test) | Accepted | Inherent to RANSAC; not a defect |
| `MidlineSet` assembly bridge pattern | Accepted | Intentional design for Stage 2 → Stage 4 data path |
| `CurveOptimizer` statefulness (warm-start across frames) | Accepted | Required for optimizer convergence efficiency |
| SR-002: constants from `reconstruction/` in observers | Accepted, allowlisted | Constants (`SPLINE_K`, `SPLINE_KNOTS`) are legitimate dependencies for visualization |
| `diagnostics.py` backward-compat shim (29 LOC) | Accepted | Preserves imports from `aquapose.visualization.diagnostics` after split into `midline_viz.py` and `triangulation_viz.py` |
| `cli.py` vs `cli/` (single file vs subdirectory) | Accepted | Cosmetic; functionally identical |
| Regression tests skip without video data | Documented | The video path is machine-specific; conftest skips cleanly with env var instructions |

---

## Phase-by-Phase Highlights

### Phase 13: Engine Core

Established the architectural skeleton: `Stage` protocol (structural typing, no inheritance), `PipelineContext` dataclass accumulator, typed event system with `EventBus`, `Observer` protocol, frozen dataclass config hierarchy with YAML+CLI override support, and `PosePipeline` orchestrator. The critical ENG-07 import boundary rule was encoded from the start, though the checker to enforce it came later in Phase 19. This phase produced the foundation that all subsequent phases built on — its four plans were complete in under a day.

### Phase 14: Golden Data and Verification Framework

Generated a frozen snapshot of v1.0 pipeline outputs on a real video clip and committed it as golden data. Wrote an interface test harness that can assert `stage.run(context)` output correctness against any stage. The golden data generation script (`generate_golden_data.py`) uses `PosePipeline` and `build_stages()` — it was already using the v2.0 API, which was a good sign of coherence between the old and new systems.

### Phase 14.1: Pipeline Structure Correction (INSERTED)

Discovered a mismatch between the guidebook's 5-stage model and the 7-stage model encoded in planning documents during Phase 14. Rather than paper over it, this inserted phase corrected the planning foundation: aligned ROADMAP, REQUIREMENTS, `PipelineContext`, and the config schema to the canonical model before any stage migrations began. This was the right call — starting Phase 15 from a wrong foundation would have caused cascading rework.

### Phase 15: Stage Migrations

The core porting work: five stages migrated as pure `Stage` implementors with no side effects. The most structurally interesting decision was Stage 4 (Tracking), where the port preserved `FishTracker` as a stateful object constructed once and persisted across frames — pure in the sense that it produces context fields with no I/O, but stateful in its internal population model. The most technically complex was Stage 2 (Midline), which subsumes U-Net inference, skeletonization, and BFS pruning in a single `segment_then_extract` backend. A bug ledger was maintained throughout Phase 15 for deferred coupling issues that became Phase 20 targets.

### Phase 16: Numerical Verification and Legacy Cleanup

Regression tests were written against golden data and all legacy v1.0 pipeline scripts were archived to `scripts/legacy/`. The regression tests discovered that the midline golden data was incompatible with the new pipeline's pre-tracking midline extraction order — documented as an `xfail(strict=False)` pending golden data regeneration with `PosePipeline`. The legacy cleanup removed the v1.0 execution path from all active import paths.

### Phase 17: Observers

All five observers were implemented as standalone classes subscribing to pipeline events: `TimingObserver` (per-stage and total wall clock), `HDF5ExportObserver` (spline control points and metadata), `Overlay2DObserver` (per-camera reprojection overlays), `Animation3DObserver` (3D midline video), `DiagnosticObserver` (per-stage snapshot capture). The observer pattern held cleanly — no stage was modified to accommodate any observer's needs.

### Phase 18: CLI and Execution Modes

Delivered `aquapose run` as a Click-based CLI entrypoint with four modes: production, diagnostic, synthetic, and benchmark. Smoke testing across all modes verified end-to-end correctness. The CLI was 244 LOC at phase completion — over the "thin wrapper" threshold — due to inline observer assembly logic, which was flagged in Phase 19 and fixed in Phase 20.

### Phase 19: Alpha Refactor Audit

A structured compliance audit against the guidebook produced `19-AUDIT.md` with 22 findings (1 critical root cause, 9 warnings, 10 info). Introduced automated tooling: the import boundary checker (`tools/import_boundary_checker.py`) as a pre-commit hook and a `smoke_test.py` CLI script for mode-level integration testing. The audit's value was not in finding unknown defects (most issues were already on the bug ledger) but in formally cataloging and prioritizing them for Phase 20.

### Phase 20: Post-Refactor Loose Ends

Resolved all critical and warning audit findings in five plans: moved `PipelineContext`/`Stage` to `core/`, deleted four dead modules (~45 files), removed 10 hardcoded `skip_camera_id` constants, extracted observer assembly from CLI to `engine/observer_factory.py`, rewired Stage 4 to consume Stage 3 bundles, and split `diagnostics.py` (2,203 LOC) into `midline_viz.py` and `triangulation_viz.py` with a backward-compat shim. The final test count of 514 passing was maintained throughout all remediations.

---

## Gaps Discovered

These are higher-level architectural or capability gaps surfaced by the retrospective analysis. They are distinct from the Phase 19/20 audit findings (which were granular compliance items), and they feed directly into the Phase 21 prospective document.

### Gap 1: Segmentation Quality Ceiling (Primary Bottleneck)

U-Net achieves IoU 0.623 on the fish segmentation task, substantially below the 0.90 target set during v1.0. This is the single largest quality bottleneck in the pipeline: poor masks → noisy 2D midlines → inaccurate 3D reconstructions. The v2.0 refactor correctly preserved the segmentation architecture as-is (the doctrine was "port behavior, not rewrite"), but the underlying quality limitation was inherited from v1.0 and not improved. Publication-quality results depend on closing this gap. Addressing it likely requires: higher-quality training data (more frames, better diversity), augmentation strategies (underwater lighting, occlusion), a larger encoder (MobileNetV3-Medium or ResNet-18), or a different architecture (e.g., segment-anything fine-tuning).

### Gap 2: Regression Suite Operationally Blocked

The 7 regression tests exist and are well-structured, but they skip in CI and on any machine without `AQUAPOSE_VIDEO_DIR` pointing to the production video dataset. This means numerical equivalence between v2.0 and v1.0 has never been formally verified on the full pipeline — only smoke-tested via `smoke_test.py` using real data. The xfail on midline regression requires golden data regeneration with `PosePipeline`. Making the regression suite operationally green would provide an ongoing numerical safety net for future changes.

### Gap 3: Association Stage Inefficiency Remains

Despite the Phase 20-04 improvement (Stage 4 now consumes Stage 3 bundles), the association stage (`AssociationStage`) still runs a full RANSAC centroid clustering over all detections per frame. This is O(N² C) in detections and cameras. For a 12-camera rig with multiple fish, this is the most expensive stage per frame. The v2.0 port preserved v1.0's RANSAC-based association as-is, but a more efficient approach (e.g., graph-based association, learned embeddings) could significantly reduce per-frame latency and improve accuracy in dense scenes.

### Gap 4: No CI/CD Integration

The test suite runs locally via `hatch run test`, but there is no continuous integration pipeline (GitHub Actions, GitLab CI, etc.) that runs tests on push. This means the 514-test suite is never automatically run on new changes in a clean environment. The pre-commit hooks enforce code quality at commit time, but not test correctness. A CI setup would also provide the environment to run regression tests consistently.

### Gap 5: Curve Optimizer Backend Incomplete for Production Use

The `curve_optimizer` backend exists and passes smoke tests, but it was ported from v1.0 with known limitations: it warm-starts from the triangulation backend's output (requiring triangulation to run first) and is ~3x slower (80s vs. 24s per clip). The optimizer's convergence properties in multi-fish scenes with occlusion are untested at scale. For publication results, the curve optimizer may produce better 3D midlines than direct triangulation — but this has not been benchmarked.

### Gap 6: HDF5 Output Schema Not Finalized

The `HDF5ExportObserver` writes spline control points and metadata, but the HDF5 schema was not formally versioned or documented in the v2.0 refactor. Downstream analysis scripts (not part of this codebase) will need a stable schema to read the output. The current schema is functional but undocumented, creating a risk of silent breaking changes if the output format evolves.

### Gap 7: "Port Behavior, Not Rewrite" Preserved v1.0 Limitations

The v2.0 doctrine was numerical equivalence with v1.0 — not improvement. This was the correct choice for a refactor phase, but it means all algorithmic limitations of v1.0 are present in v2.0:
- View-angle weighting in triangulation uses a fixed formula without learned or calibrated weights
- B-spline fitting uses hardcoded `SPLINE_K` and `SPLINE_KNOTS` constants
- The Hungarian tracker uses a fixed `reprojection_threshold` without any online adaptation
- MOG2 background subtraction is available as a backend but untested in v2.0 context

These are not bugs — they are preserved design choices — but they are candidates for the next milestone's improvement scope.

---

## GSD Process Retrospective

### What Worked Well

**The discuss-phase → plan-phase → execute-phase cycle** was highly effective for this project. The discussion phase consistently surfaced critical decisions before planning began, which prevented mid-execution pivots. The clearest example was Phase 14.1: the mismatch between the 7-stage planning model and the 5-stage guidebook was caught during discussion and corrected with a targeted inserted phase before any stage code was written.

**Phase sizing** was generally well-calibrated. Phases in the 2-5 plan range completed within a session without running into context degradation. Phase 20's 5-plan structure handled a broad remediation scope by assigning one concern per plan (IB-003, dead modules, CLI, Stage 3/4, info cleanup), which made each plan's scope clear and executable.

**Audit-then-remediate as two separate phases** (Phase 19 → Phase 20) was the right pattern for a completed refactor. Producing `19-AUDIT.md` as a standalone artifact before fixing anything created a clear decision record and prevented scope creep during remediation. The alternative — inline "fix as you find" during the audit — would have made the audit findings harder to trace.

**The bug ledger pattern** (introduced in Phase 15 to track deferred coupling issues) translated smoothly into Phase 19 audit items, which in turn translated into Phase 20 plans. This three-phase triage chain (ledger → audit → remediation) was an effective way to handle issues that were known but not immediately blocking.

**Quantitative verification at checkpoints** — counting tests, checking LOC, running the import boundary checker — was a reliable signal that plans completed what they claimed. The 514 test count became a stable invariant that executors verified after each Phase 20 plan.

### What Caused Friction

**Context management across sessions** was the primary pain point. State.md decisions accumulated rapidly (the Decisions section grew to ~175 lines) and became difficult to scan for relevant context when resuming. The `@` context references in plan files helped orient each execution session, but there was no efficient mechanism for surfacing only the decisions relevant to the current plan.

**Plan file context references sometimes pointed to documents not yet finalized** when a plan was written. Plans referencing `21-CONTEXT.md` or `19-AUDIT.md` work well when those files exist; if planning moved faster than execution, stale references were confusing. The practice of capturing context before creating plans (CONTEXT.md files) mostly prevented this.

**Checkpoint handling for human verification steps** added latency. Several Phase 19 plans paused for smoke test verification that could have been automated given the smoke test script already existed. The auto-approve mode (`workflow.auto_advance`) would have been appropriate for these steps but was not used.

**Large module `diagnostics.py`** (2,203 LOC) caused friction during Phase 20-05 because the split had to be carefully managed to preserve backward compatibility. Module splitting at that scale is inherently risky — a backward-compat shim was the right mitigation, but the need for it was a signal that module size should be controlled earlier in development.

### What to Change for the Next Milestone

1. **Tighter module size bounds**: Introduce a soft limit (e.g., 500 LOC) as a planning guideline. Files approaching the limit should be split before they become problematic. This is a culture change, not a tooling change.

2. **CI integration as a Phase 1 deliverable**: The next milestone should establish GitHub Actions or equivalent in the first plan, not as an afterthought. Running `hatch run test` in CI on every push is a minimal but high-value step.

3. **Decisions section pruning**: STATE.md decisions should be periodically archived to PROJECT.md or phase-specific context files. The growing decisions list in STATE.md becomes a navigation burden. A "decisions active" vs. "decisions archived" split would help.

4. **Regression test operationality**: The next milestone should include a plan to make the regression suite runnable in CI with a small synthetic dataset (not requiring the production video data), so numerical equivalence is automatically verified.

---

## Lessons Learned

**Structural typing via Protocol is the right choice for stage interfaces.** The absence of inheritance requirements made it straightforward to compose stages in tests without mock base classes and to introduce `SyntheticDataStage` as a drop-in adapter without changing the protocol definition.

**Moving shared data contracts to the lowest layer they belong in** eliminates the most common class of import boundary violations. The IB-003 violations (core importing engine for `PipelineContext`) were architecturally obvious in retrospect — a data accumulator that stages write to is definitionally a core concern, not an engine concern. Starting with `PipelineContext` in `core/` would have avoided 7 violations and a Phase 20-01 plan.

**Pre-commit automation scales well for a solo researcher.** The combination of Ruff (lint + format), detect-secrets, and the import boundary checker catches the most common classes of defects before they reach the commit log. The overhead per commit is low (under 5 seconds); the benefit is permanent.

**The "port behavior, not rewrite" doctrine was the right call for this refactor** — it kept scope bounded, provided a clear acceptance bar (numerical equivalence), and prevented gold-plating. The cost is that v1.0 algorithmic limitations are preserved in v2.0. Accepting that cost explicitly (by labeling it a doctrine, not an oversight) sets clear expectations for the next milestone's scope.

**Audit + remediation as distinct phases is superior to inline cleanup** for a refactor of this scope. The audit produces a structured finding catalog with severity ratings; remediation plans can then be scoped to specific findings by severity. This separation also makes the audit a standalone reference document useful beyond the remediation work.

**Dead code removal benefits from an explicit consumption trace.** The Phase 19 audit's dead code inventory (Section 5.1) traced each dead module from the pipeline entrypoint to confirm zero consumers before recommending deletion. This trace-first approach prevented accidental deletion of modules that had non-obvious consumers via dynamic imports or test fixtures.

---

*v2.0 Alpha retrospective complete.*
*Generated: 2026-02-27*
*Phase: 21-retrospective-prospective, Plan 01*
