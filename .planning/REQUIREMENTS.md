# Requirements: AquaPose

**Defined:** 2026-02-25
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v2.0 Requirements

Requirements for the pre-alpha clean-room refactor. Each maps to roadmap phases.

### Engine Infrastructure

- [ ] **ENG-01**: Stage Protocol defined via `typing.Protocol` with structural typing
- [ ] **ENG-02**: PipelineContext dataclass accumulates typed results across stages
- [ ] **ENG-03**: Event system with typed dataclasses (pipeline lifecycle, stage lifecycle, frame-level, domain events)
- [ ] **ENG-04**: Observer protocol — subscribe to specific event types, synchronous dispatch
- [ ] **ENG-05**: Frozen dataclass config hierarchy (defaults → YAML → CLI overrides → freeze)
- [ ] **ENG-06**: PosePipeline orchestrator wires stages, emits events, coordinates observers
- [ ] **ENG-07**: Import boundary enforced: engine/ imports computation modules, never reverse
- [ ] **ENG-08**: Full serialized config logged as first artifact of every run

### Stage Migrations

- [ ] **STG-01**: Detection stage ported (YOLO/MOG2) — pure computation, no side effects
- [ ] **STG-02**: Segmentation stage ported (U-Net inference) — pure computation, no side effects
- [ ] **STG-03**: Midline extraction stage ported (skeletonization + BFS pruning)
- [ ] **STG-04**: Cross-view association stage ported (RANSAC centroid clustering)
- [ ] **STG-05**: Triangulation stage ported (RANSAC + view-angle weighting + B-spline fitting)
- [ ] **STG-06**: Tracking stage ported (Hungarian 3D tracking with population constraint)
- [ ] **STG-07**: Curve optimizer stage ported (correspondence-free B-spline via chamfer distance)

### Observers

- [ ] **OBS-01**: Timing observer records per-stage and total execution time
- [ ] **OBS-02**: HDF5 export observer writes spline control points and metadata
- [ ] **OBS-03**: 2D reprojection overlay visualization observer
- [ ] **OBS-04**: 3D midline animation visualization observer
- [ ] **OBS-05**: Diagnostic observer captures intermediate data per stage

### CLI & Modes

- [ ] **CLI-01**: `aquapose run` CLI entrypoint as thin wrapper over PosePipeline
- [ ] **CLI-02**: Production mode — standard pipeline execution
- [ ] **CLI-03**: Diagnostic mode — activates diagnostic observer, extra artifacts
- [ ] **CLI-04**: Synthetic mode — stage adapter injects synthetic data, no pipeline bypass
- [ ] **CLI-05**: Benchmark mode — timing-focused, minimal observers

### Verification

- [ ] **VER-01**: Golden data generated as standalone commit before stage migrations
- [ ] **VER-02**: Each ported stage verified with interface tests (stage.run(context) correctness)
- [ ] **VER-03**: Numerical regression tests against golden data — pass means equivalent or improved results (bug fixes during port are expected)
- [ ] **VER-04**: Legacy scripts archived to scripts/legacy/ then removed

## Future Requirements

### Post-Alpha Improvements

- **PERF-01**: Distributed execution support
- **PERF-02**: Automated benchmarking suites
- **EXT-01**: LLM-based analysis observers
- **EXT-02**: Formal experiment tracking integration
- **SEG-01**: Improve U-Net segmentation quality beyond IoU 0.623

## Out of Scope

| Feature | Reason |
|---------|--------|
| Segmentation quality improvements | Separate milestone — v2.0 is architecture only |
| New reconstruction algorithms | Preserve existing algorithms, don't add new ones |
| Rewriting working algorithms | Doctrine: port behavior, not rewrite logic |
| Real-time processing | Batch only — not in scope for alpha |
| GUI or web interface | CLI-only as defined in doctrine |
| Pydantic for config | Frozen dataclasses already decided |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENG-01 | Phase 13 | Pending |
| ENG-02 | Phase 13 | Pending |
| ENG-03 | Phase 13 | Pending |
| ENG-04 | Phase 13 | Pending |
| ENG-05 | Phase 13 | Pending |
| ENG-06 | Phase 13 | Pending |
| ENG-07 | Phase 13 | Pending |
| ENG-08 | Phase 13 | Pending |
| VER-01 | Phase 14 | Pending |
| VER-02 | Phase 14 | Pending |
| STG-01 | Phase 15 | Pending |
| STG-02 | Phase 15 | Pending |
| STG-03 | Phase 15 | Pending |
| STG-04 | Phase 15 | Pending |
| STG-05 | Phase 15 | Pending |
| STG-06 | Phase 15 | Pending |
| STG-07 | Phase 15 | Pending |
| VER-03 | Phase 16 | Pending |
| VER-04 | Phase 16 | Pending |
| OBS-01 | Phase 17 | Pending |
| OBS-02 | Phase 17 | Pending |
| OBS-03 | Phase 17 | Pending |
| OBS-04 | Phase 17 | Pending |
| OBS-05 | Phase 17 | Pending |
| CLI-01 | Phase 18 | Pending |
| CLI-02 | Phase 18 | Pending |
| CLI-03 | Phase 18 | Pending |
| CLI-04 | Phase 18 | Pending |
| CLI-05 | Phase 18 | Pending |

**Coverage:**
- v2.0 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-25*
*Last updated: 2026-02-25 — traceability populated by roadmapper*
