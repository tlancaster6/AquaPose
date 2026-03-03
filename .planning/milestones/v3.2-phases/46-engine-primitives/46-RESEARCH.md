# Phase 46: Engine Primitives - Research

**Researched:** 2026-03-03
**Domain:** Python pickle serialization, pipeline context caching, CLI extension
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Cache file format & layout**
- One pickle file per stage (e.g. `detection_cache.pkl`, `tracking_cache.pkl`) — not cumulative
- Each cache stores the **full PipelineContext snapshot** at that stage — any single cache is self-contained through that stage
- Files stored in `diagnostics/<stage>_cache.pkl` alongside existing outputs (pipeline_diagnostics.npz, midline_fixtures.npz)
- Cache files wrapped in a metadata envelope dict containing: run_id, timestamp, stage_name, version fingerprint, and the PipelineContext data

**Stage-skipping behavior**
- PosePipeline.run() accepts an `initial_context` parameter
- Auto-detect which stages to skip by inspecting PipelineContext fields — if a stage's output fields are already populated (non-None), skip it
- Skipped stages still emit StageComplete events with `elapsed_seconds=0` and `summary={'skipped': True}` — observers see a complete timeline
- CarryForward state is included in the tracking stage cache — enables multi-batch resumption
- Strict validation on initial_context: verify that required upstream fields are populated before each stage runs; fail fast with a clear message if not

**Staleness detection**
- Staleness detected via pickle load failure: catch AttributeError, ModuleNotFoundError, etc. during deserialization and raise StaleCacheError
- StaleCacheError message includes: cache file path, suggestion to re-run the pipeline, and the original exception for debugging
- StaleCacheError defined in `core/context.py` alongside PipelineContext and ContextLoader
- Basic shape validation after successful deserialization (e.g., frame_count == len(detections))

**Scope of caching trigger**
- Pickle caching is automatic in diagnostic mode — DiagnosticObserver writes cache files whenever output_dir is set, no extra config flag needed
- Single-file loading only — since each cache is a full context snapshot, one file is sufficient to resume from any stage

**Loader API**
- Standalone function `load_stage_cache(path) -> PipelineContext` in `core/context.py` — usable from scripts, notebooks, and pipeline
- CLI flag `--resume-from path/to/cache.pkl` on `aquapose run` that loads context and skips upstream stages

### Claude's Discretion
- Exact metadata envelope structure (dict keys, version fingerprint format)
- Internal implementation of field-based skip detection (mapping from stage class to output field names)
- Whether to log skipped stages at INFO or DEBUG level

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | DiagnosticObserver writes per-stage pickle cache files on each StageComplete event | DiagnosticObserver.on_event() already handles StageComplete; extend with pickle write to `diagnostics/<stage>_cache.pkl` when output_dir is set |
| INFRA-02 | PosePipeline.run() accepts optional pre-populated PipelineContext via initial_context parameter | pipeline.py line 120 — add `initial_context: PipelineContext | None = None`, inject at line 155 (context initialization), add field-presence skip logic in the stage loop (lines 164-184) |
| INFRA-03 | ContextLoader deserializes per-stage pickle caches into a fresh PipelineContext for sweep isolation | `load_stage_cache(path) -> PipelineContext` function in `core/context.py` — unwrap envelope dict, reconstruct PipelineContext, catch pickle errors and raise StaleCacheError |
| INFRA-04 | StaleCacheError raised with clear message when pickle deserialization fails due to class evolution | `StaleCacheError(Exception)` in `core/context.py` — catch AttributeError, ModuleNotFoundError, pickle.UnpicklingError during deserialization |
</phase_requirements>

---

## Summary

Phase 46 implements three tightly coupled primitives that together enable the sweep workflow: (1) DiagnosticObserver writes per-stage pickle cache files on each StageComplete event in diagnostic mode; (2) PosePipeline.run() accepts a pre-populated context and auto-skips stages whose outputs are already present; (3) a standalone `load_stage_cache()` function deserializes cache files and wraps pickle failures in `StaleCacheError`.

All four requirements are pure Python standard library work — no new dependencies are needed. The existing codebase provides all the structural hooks: DiagnosticObserver already handles StageComplete events and writes artifacts on PipelineComplete; PipelineContext already uses a `get()` pattern for field validation; the stage loop in pipeline.py is a simple for-loop that accepts straightforward skip logic; and the CLI uses Click with well-established option patterns.

The most complex design decision left to Claude's discretion is the version fingerprint for the metadata envelope. A lightweight approach using `{class_name}:{__module__}:{id(cls)}` would be fragile; instead a content-based approach using the dataclass field names hashed with a stable algorithm is more robust and survives innocent internal refactors.

**Primary recommendation:** Implement all four items in a single phase with four clearly separated tasks: (1) StaleCacheError + load_stage_cache in context.py, (2) pickle write logic in DiagnosticObserver, (3) initial_context + skip logic in PosePipeline.run(), (4) --resume-from CLI flag + __init__.py exports.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `pickle` (stdlib) | Python 3.11 | Serialize/deserialize PipelineContext + arbitrary domain objects | Only practical option for Python object graphs with numpy arrays and custom dataclasses; no external dep |
| `pathlib.Path` (stdlib) | Python 3.11 | File path handling for cache output | Already used uniformly throughout the codebase |
| `datetime` (stdlib) | Python 3.11 | Timestamp in metadata envelope | Already used in diagnostic_observer.py for NPZ timestamps |
| `click` (existing dep) | project version | CLI --resume-from flag | Already used for all CLI commands; consistent UX |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `hashlib` (stdlib) | Python 3.11 | Version fingerprint of PipelineContext field names | Generates a stable short hash that changes when dataclass fields are added/removed/renamed — cheap staleness signal |
| `logging` (stdlib) | Python 3.11 | Log skipped stages | Already used throughout pipeline.py and observers |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pickle` | `msgpack`, `joblib` | pickle handles arbitrary Python objects (dataclasses, numpy arrays) with no schema; external libs require explicit encoding/decoding |
| `pickle` + envelope dict | `shelve` | shelve is file-based key-value; envelope dict in a single pickle file is simpler and matches the one-file-per-stage design |
| field-name hash | version string from `__version__` | package version changes on every release even when the cache format is unchanged; field-name hash changes only when structure changes |

**Installation:** No new dependencies required. All libraries are Python stdlib or already in the project.

---

## Architecture Patterns

### Recommended File Changes
```
src/aquapose/
├── core/
│   └── context.py          # ADD: StaleCacheError, load_stage_cache(), version fingerprint helper
├── engine/
│   ├── diagnostic_observer.py  # MODIFY: on_event() — write pickle cache on StageComplete
│   └── pipeline.py         # MODIFY: run() — accept initial_context, skip logic in stage loop
src/aquapose/cli.py          # MODIFY: run command — add --resume-from flag
```

### Pattern 1: Metadata Envelope Dict
**What:** Wrap PipelineContext in a dict before pickling to include provenance metadata.
**When to use:** On every StageComplete write in DiagnosticObserver; on every load in load_stage_cache.
**Example:**
```python
# Writing (in DiagnosticObserver.on_event())
import pickle, datetime, hashlib, dataclasses

def _context_fingerprint(ctx: PipelineContext) -> str:
    """Hash of PipelineContext field names — changes when dataclass structure changes."""
    names = sorted(f.name for f in dataclasses.fields(ctx))
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:12]

envelope = {
    "run_id": run_id,           # from config, passed via observer __init__
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    "stage_name": event.stage_name,
    "version_fingerprint": _context_fingerprint(event.context),
    "context": event.context,
    "carry": carry,             # None for all stages except TrackingStage
}
cache_path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
```

### Pattern 2: load_stage_cache with StaleCacheError
**What:** Deserialize a cache file, validate the fingerprint, return the PipelineContext.
**When to use:** In scripts, notebooks, and the CLI --resume-from path.
**Example:**
```python
# In core/context.py
import pickle
from pathlib import Path

class StaleCacheError(Exception):
    """Raised when a stage cache cannot be deserialized due to class evolution."""

def load_stage_cache(path: str | Path) -> PipelineContext:
    """Load a stage cache pickle file and return the embedded PipelineContext.

    Args:
        path: Path to a *_cache.pkl file written by DiagnosticObserver.

    Returns:
        The deserialized PipelineContext.

    Raises:
        StaleCacheError: If deserialization fails (AttributeError,
            ModuleNotFoundError, pickle.UnpicklingError) or basic shape
            validation fails.
        FileNotFoundError: If path does not exist.
    """
    p = Path(path)
    try:
        envelope = pickle.loads(p.read_bytes())
    except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError) as exc:
        raise StaleCacheError(
            f"Cache file '{p}' is incompatible with the current codebase. "
            f"Re-run the pipeline in diagnostic mode to regenerate it. "
            f"Original error: {exc}"
        ) from exc

    ctx: PipelineContext = envelope["context"]

    # Basic shape validation
    if ctx.frame_count is not None and ctx.detections is not None:
        if ctx.frame_count != len(ctx.detections):
            raise StaleCacheError(
                f"Cache '{p}': frame_count={ctx.frame_count} but "
                f"len(detections)={len(ctx.detections)}. Cache may be corrupted."
            )
    return ctx
```

### Pattern 3: initial_context + Skip Logic in PosePipeline.run()
**What:** Accept an optional pre-populated context; skip stages whose output fields are non-None.
**When to use:** In PosePipeline.run() when initial_context is provided (e.g. from --resume-from).
**Example:**
```python
# In pipeline.py — the stage loop

# Map each stage class to the PipelineContext field(s) it populates.
# A stage is skipped if ALL its output fields are already non-None.
_STAGE_OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "DetectionStage":     ("frame_count", "camera_ids", "detections"),
    "SyntheticDataStage": ("frame_count", "camera_ids", "detections", "annotated_detections"),
    "TrackingStage":      ("tracks_2d",),
    "AssociationStage":   ("tracklet_groups",),
    "MidlineStage":       ("annotated_detections",),
    "ReconstructionStage": ("midlines_3d",),
}

for i, stage in enumerate(self._stages):
    stage_name = type(stage).__name__
    output_fields = _STAGE_OUTPUT_FIELDS.get(stage_name, ())
    already_populated = all(
        getattr(context, f, None) is not None for f in output_fields
    ) if output_fields else False

    if already_populated:
        logger.info("Skipping %s — outputs already populated in context", stage_name)
        self._bus.emit(StageComplete(
            stage_name=stage_name,
            stage_index=i,
            elapsed_seconds=0.0,
            summary={"skipped": True},
            context=context,
        ))
        continue

    # ... existing stage execution ...
```

### Pattern 4: CarryForward in TrackingStage Cache
**What:** DiagnosticObserver must capture the `carry` state alongside context for the tracking cache.
**When to use:** In DiagnosticObserver when receiving a StageComplete for "TrackingStage".
**Challenge:** DiagnosticObserver currently has no access to the `carry` object — it is maintained entirely inside PosePipeline._stages loop. There are two options:
1. Include carry in the StageComplete event payload (requires modifying the event or attaching it as a context attribute).
2. Pass carry into DiagnosticObserver via a dedicated method called by PosePipeline after the TrackingStage runs.

**Recommended approach (Claude's discretion):** Store carry on PipelineContext as an optional field `carry_forward: CarryForward | None = None`. PosePipeline sets `context.carry_forward = carry` immediately after TrackingStage runs, before emitting StageComplete. This keeps carry in the context (where it belongs), makes it visible to the observer, and serializes automatically with the context pickle. No event schema change needed.

### Anti-Patterns to Avoid
- **Deep copy of context before pickling:** Unnecessary — pickle serializes by value. Using `pickle.dumps(context)` already produces a detached copy. Never call `copy.deepcopy()` before pickling.
- **Writing cache on PipelineComplete instead of StageComplete:** Violates INFRA-01. If the pipeline fails after stage 3, stages 1-3 caches must still exist for partial sweep resumption.
- **Importing domain types in load_stage_cache:** The function lives in `core/context.py` which already imports PipelineContext. Keep it free of engine imports (ENG-07 boundary applies here too).
- **Using `pickle.PROTOCOL 2` or lower:** Always use `pickle.HIGHEST_PROTOCOL` (protocol 5 in Python 3.11) for best performance and numpy array support.
- **Skipping StageComplete emission for skipped stages:** The user decided skipped stages MUST emit StageComplete with `{'skipped': True}`. Observers (e.g. ConsoleObserver, TimingObserver) depend on a complete event sequence.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| numpy array serialization in pickle | Custom numpy encoder | stdlib `pickle` | pickle handles numpy natively since Python 3.8+; numpy arrays round-trip cleanly |
| version pinning for cache compatibility | Custom version registry | Field-name hash in envelope | Simple, automatic, changes only when structure changes |
| context copy for sweep isolation | deepcopy before loading | Fresh pickle.loads() call per sweep | pickle.loads() always produces a fully independent object graph |

**Key insight:** Python's pickle handles the entire PipelineContext object graph (dataclasses, lists, dicts, numpy arrays, nested custom objects) with no encoding boilerplate. The only gap pickle cannot cover is class definition changes — which is exactly what the envelope fingerprint and StaleCacheError handle.

---

## Common Pitfalls

### Pitfall 1: carry_forward is not accessible to DiagnosticObserver
**What goes wrong:** DiagnosticObserver receives a StageComplete event for TrackingStage, but `carry` is a local variable inside `PosePipeline.run()` — not in the context or the event. The observer can't write a complete tracking cache.
**Why it happens:** TrackingStage has a non-standard `run()` signature; carry is managed separately from context.
**How to avoid:** Add `carry_forward: CarryForward | None = None` field to PipelineContext. After `context, carry = stage.run(context, carry)`, assign `context.carry_forward = carry` before emitting StageComplete. The pickle serializer then captures it automatically.
**Warning signs:** Test asserts `envelope['context'].carry_forward is not None` after loading tracking cache — fails if carry is not injected.

### Pitfall 2: Strict validation fires unexpectedly during normal (non-resumed) runs
**What goes wrong:** Adding strict upstream validation to `PosePipeline.run()` means that if any stage partially populates context fields (e.g. frame_count set but detections None), the validation logic misfires.
**Why it happens:** The skip detection checks "ALL output fields non-None". But validation before each stage checks "required upstream fields non-None". These are different checks serving different purposes.
**How to avoid:** Only enable strict upstream validation when `initial_context` is provided. Normal runs do not need it (they start fresh). Document this in code.
**Warning signs:** Tests for normal pipeline runs start failing after adding validation.

### Pitfall 3: File written on every StageComplete even in non-diagnostic mode
**What goes wrong:** If the output_dir check is accidentally absent or always True, pickle files are written in production mode too — wasting I/O and disk.
**Why it happens:** DiagnosticObserver already has a `_output_dir` guard in `_on_pipeline_complete`. The same guard must be applied to the new StageComplete branch.
**How to avoid:** Wrap pickle write with `if self._output_dir is not None:` — same pattern as existing NPZ writes.
**Warning signs:** Integration tests for production mode find unexpected `*_cache.pkl` files.

### Pitfall 4: load_stage_cache does not handle missing envelope keys
**What goes wrong:** A cache written by an older version of DiagnosticObserver (before envelope format was introduced) lacks expected keys like `version_fingerprint` — causing KeyError on load.
**Why it happens:** The envelope format is new; old caches (pre-Phase 46) are raw PipelineContext pickles.
**How to avoid:** In load_stage_cache, check `isinstance(envelope, dict)` and `"context" in envelope`. If not a dict (raw context) or missing keys, raise StaleCacheError with a message indicating the cache predates the envelope format.
**Warning signs:** Loading old pipeline_diagnostics companion caches or test fixtures raises KeyError.

### Pitfall 5: `--resume-from` passes path to PosePipeline but pipeline has no run_id for the resumed run
**What goes wrong:** When resuming from a cache, the pipeline still generates a fresh `run_id` from the current timestamp. The run_id in the output directory may differ from the original run.
**Why it happens:** run_id is generated in load_config() at CLI time, not derived from the cache.
**How to avoid:** This is acceptable behavior (each pipeline invocation — even resumed — gets its own run_id and output directory). Document it. Do not attempt to inherit run_id from the cache.
**Warning signs:** N/A — this is expected behavior, just document it clearly.

---

## Code Examples

Verified patterns from existing codebase:

### Existing DiagnosticObserver.on_event() pattern (to extend)
```python
# Source: src/aquapose/engine/diagnostic_observer.py line 136
def on_event(self, event: Event) -> None:
    if isinstance(event, PipelineComplete):
        self._on_pipeline_complete()
        return
    if not isinstance(event, StageComplete):
        return
    context = event.context
    if context is None:
        return
    # ... build snapshot ...
    self.stages[event.stage_name] = snapshot
```

The new pickle write goes in this same method, in the `StageComplete` branch, after snapshot capture:
```python
    if self._output_dir is not None:
        self._write_stage_cache(event, context)
```

### Existing PosePipeline stage loop (to modify)
```python
# Source: src/aquapose/engine/pipeline.py lines 164-184
for i, stage in enumerate(self._stages):
    from aquapose.core.tracking import TrackingStage
    stage_name = type(stage).__name__
    self._bus.emit(StageStart(stage_name=stage_name, stage_index=i))
    stage_start = time.monotonic()
    if isinstance(stage, TrackingStage):
        context, carry = stage.run(context, carry)
    else:
        context = stage.run(context)
    elapsed = time.monotonic() - stage_start
    context.stage_timing[stage_name] = elapsed
    self._bus.emit(StageComplete(...))
```

### Existing CLI run command (to extend with --resume-from)
```python
# Source: src/aquapose/cli.py lines 31-138
@cli.command()
@click.option("--config", "-c", ...)
@click.option("--mode", "-m", ...)
# ADD: @click.option("--resume-from", ...)
def run(config, mode, ..., resume_from):
    # After stages = build_stages(pipeline_config):
    # if resume_from:
    #     initial_context = load_stage_cache(resume_from)
    # else:
    #     initial_context = None
    # pipeline.run(initial_context=initial_context)
```

### Existing PipelineContext field validation pattern (reference for skip validation)
```python
# Source: src/aquapose/core/context.py line 121
def get(self, field_name: str) -> object:
    value = getattr(self, field_name)
    if value is None:
        raise ValueError(
            f"PipelineContext.{field_name} is None — the stage that produces "
            f"'{field_name}' has not run yet. Check stage ordering.",
        )
    return value
```

---

## Integration Map

The four requirements touch exactly five files. No new files need to be created.

| File | Change Type | What Changes |
|------|-------------|--------------|
| `src/aquapose/core/context.py` | Extend | Add `carry_forward` field, `StaleCacheError`, `load_stage_cache()`, `_context_fingerprint()` helper |
| `src/aquapose/engine/diagnostic_observer.py` | Extend | Add `_write_stage_cache()` private method; call it in `on_event()` on StageComplete when `_output_dir` is set; store `_run_id` if needed for envelope |
| `src/aquapose/engine/pipeline.py` | Extend | Add `initial_context` param to `run()`; inject at context init; add `_STAGE_OUTPUT_FIELDS` mapping; add skip logic in stage loop; inject carry into context before TrackingStage StageComplete |
| `src/aquapose/cli.py` | Extend | Add `--resume-from` option to `run` command; call `load_stage_cache()` if provided |
| `src/aquapose/core/__init__.py` | Extend | Export `StaleCacheError`, `load_stage_cache` |

DiagnosticObserver needs the `run_id` to write it into the envelope. Currently it receives `output_dir` and `calibration_path` in `__init__`. Either add `run_id` as a third `__init__` param, or read it from `PipelineStart` event (DiagnosticObserver already receives all events). **Recommended:** Store `_run_id` from `PipelineStart` event in `on_event()` — avoids changing `__init__` signature and `build_observers()`.

---

## Open Questions

1. **Should `diagnostics/` be a subdirectory of `output_dir`, or does it already exist?**
   - What we know: CONTEXT.md says files go in `diagnostics/<stage>_cache.pkl`; the observer_factory creates DiagnosticObserver with `output_dir=config.output_dir`. The existing NPZ artifacts are written directly to `output_dir`.
   - What's unclear: Whether `diagnostics/` is a new subdirectory to create, or whether the intent is to write to `output_dir` directly with `<stage>_cache.pkl` naming.
   - Recommendation: Create a `diagnostics/` subdirectory within `output_dir` (i.e. `output_dir / "diagnostics" / f"{stage_name}_cache.pkl"`). This matches the success criterion text and keeps cache files separate from HDF5 and NPZ outputs. Use `mkdir(parents=True, exist_ok=True)` before writing.

2. **Does `load_stage_cache` also need to return the `carry_forward` object separately, or is embedding it in PipelineContext sufficient?**
   - What we know: CONTEXT.md says "CarryForward state is included in the tracking stage cache — enables multi-batch resumption." If carry is stored on `context.carry_forward`, `load_stage_cache` returns context and carry comes along.
   - What's unclear: Whether the caller (e.g. the resumed pipeline) needs to extract carry separately, or can just read `context.carry_forward`.
   - Recommendation: Store carry on `context.carry_forward`. PosePipeline.run() with initial_context sets its local `carry` from `initial_context.carry_forward` before the stage loop starts. No separate return value needed.

3. **Should `_STAGE_OUTPUT_FIELDS` be defined in context.py or pipeline.py?**
   - What we know: It maps stage class names to context field names — bridging engine (stage names) and core (context fields).
   - What's unclear: The ENG-07 import boundary says `core/` must not import from `engine/`.
   - Recommendation: Define in `pipeline.py` (engine layer), not in `context.py`. Stage names are strings (no import required), and context fields are also referenced as strings. No import boundary violation.

---

## Sources

### Primary (HIGH confidence)
- Direct source code reading — `src/aquapose/engine/pipeline.py`, `src/aquapose/engine/diagnostic_observer.py`, `src/aquapose/core/context.py`, `src/aquapose/engine/events.py`, `src/aquapose/engine/observer_factory.py`, `src/aquapose/cli.py` — full code review of all integration points
- `.planning/phases/46-engine-primitives/46-CONTEXT.md` — locked decisions, existing code insights, integration points

### Secondary (MEDIUM confidence)
- Python 3.11 stdlib documentation for `pickle` module — HIGHEST_PROTOCOL is 5 in Python 3.8+, native numpy array support confirmed
- `.planning/REQUIREMENTS.md` — requirement text for INFRA-01 through INFRA-04

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pure stdlib; all libraries already in use in the project
- Architecture: HIGH — all integration points identified from direct source code reading; patterns derived from existing code
- Pitfalls: HIGH — derived from direct code analysis of the specific files being modified; no speculative claims

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (stable stdlib APIs; codebase-derived findings valid until files change)
