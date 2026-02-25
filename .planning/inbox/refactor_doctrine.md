AquaPose Refactor Doctrine
Architectural Vision for Alpha Stabilization
1. Purpose of This Refactor

AquaPose is no longer a research script collection.
It is evolving into a scientific computation engine for multi-view 3D pose inference.

This refactor is not about cleaning code.

It is about:

Restoring architectural authority to the pipeline

Separating computation from observation

Enabling safe extensibility

Preserving research velocity without sacrificing structural integrity

The goal is to formalize complexity — not reduce it.

2. Core Identity of AquaPose

AquaPose must become:

Deterministic

Replayable

Fully introspectable

Stage-extensible

Observer-driven

CLI-executable

Script-agnostic

If a change violates one of these, it is architecturally suspect.

3. The Fundamental Reframe

We are transitioning from:

Script-driven scientific pipeline

To:

Event-driven scientific computation engine

The pipeline is the canonical execution path.
Scripts are disposable.

This inversion is non-negotiable.

4. Architectural Hierarchy

The system must be layered with strict directional dependencies.

Layer 1: Core Computation

Calibration

Detection

Segmentation

Tracking

Midline extraction

Triangulation / Optimization

These are pure computation modules.

They:

Accept structured inputs

Return structured outputs

Have no side effects

Do not write files

Do not generate visualizations

Do not import dev tooling

They are blind to diagnostics.

Layer 2: Execution Engine (PosePipeline)

This is the only orchestrator.

It:

Defines stage order

Manages execution state

Emits structured lifecycle events

Handles execution modes

Coordinates observers

Owns artifact management

All execution flows through this layer.

There must be exactly one canonical entrypoint.

Layer 3: Observability System

Observability is external to computation.

Observers may:

Record intermediate data

Generate diagnostics

Write reports

Save artifacts

Collect timing

Capture optimizer snapshots

Observers may not:

Mutate pipeline state

Change stage logic

Control execution flow

Observers are passive.

5. The Sacred Boundary

Stages must never depend on:

Visualization modules

Synthetic generators

Reporting utilities

File output logic

Experiment scripts

Dev tooling may depend on core.

Core may never depend on dev tooling.

This boundary must be enforced structurally.

6. Execution Modes Are Configuration, Not Branching

The system may support:

Production

Diagnostic

Synthetic

Benchmark

Modes must alter behavior via configuration and observer selection.

Modes must not introduce branching inside stage logic.

Synthetic execution is a stage adapter, not a pipeline bypass.

7. Artifacts Are First-Class Citizens

Artifacts are not incidental byproducts.

They must be:

Structured

Named consistently

Versioned

Associated with a run ID

Reproducible from inputs

File writing is not allowed inside stage functions.

All artifacts are managed centrally by the pipeline.

8. Determinism Is Mandatory

Given:

Identical inputs

Identical configuration

Identical random seeds

The pipeline must produce identical outputs.

Observers must not alter execution behavior.

Reproducibility is a core design requirement.

9. Extensibility Model

The system must support:

Swapping triangulation methods

Adding new optimizers

Adding new tracking algorithms

Adding new segmentation backends

Adding new diagnostic observers

Without modifying existing execution logic.

Extensibility is achieved through:

Stage interfaces

Dependency injection

Observer attachment

Configuration

Not through branching scripts.

10. The Pipeline Is the Only Truth

No script may:

Call stage functions directly

Reimplement orchestration logic

Manually loop tracking for snapshots

Bypass stage sequencing

All such behavior must be expressed through:

Pipeline configuration

Observers

Execution modes

If a script requires internal access, the pipeline is incomplete.

11. Alpha Definition of Done

Alpha stabilization is complete when:

There is exactly one canonical pipeline entrypoint.

All scripts invoke PosePipeline.

Diagnostic functionality is implemented via observers.

Synthetic mode runs through the pipeline.

Timing is implemented as an observer.

Raw export is implemented as an observer.

Visualization is implemented as an observer.

No stage imports dev tooling.

Dev tooling depends on core, never vice versa.

The CLI is a thin wrapper over PosePipeline.

12. Non-Goals

This refactor is not about:

Reducing feature count

Simplifying research tools

Removing diagnostics

Freezing experimentation

Rewriting working algorithms

It is about formalizing structure.

13. Long-Term Architectural Intention

AquaPose should become:

A modular, event-driven scientific inference engine capable of supporting automated experimentation, benchmarking, and analysis workflows.

Future capabilities enabled by this structure:

Automated benchmarking suites

Reproducible experiment reports

LLM-based analysis observers

Distributed execution

Plug-and-play research modules

Formal experiment tracking

This refactor is the enabling move.

14. Governing Principle

Complexity is allowed.
Entanglement is not.

Observability is encouraged.
Pipeline duplication is forbidden.

Research velocity must increase — not decrease — after this refactor.

15. Final Directive

Preserve scientific flexibility.

Enforce architectural discipline.

The pipeline is sacred.
Everything else is modular.

Build the system so that new developers extend it by attaching observers — not by writing new scripts.

End of doctrine.
