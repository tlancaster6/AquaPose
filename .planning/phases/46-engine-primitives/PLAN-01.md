# Plan 46.1: StaleCacheError, load_stage_cache, and PipelineContext.carry_forward

---
wave: 1
depends_on: []
files_modified:
  - src/aquapose/core/context.py
  - src/aquapose/core/__init__.py
  - tests/unit/core/test_stage_cache.py
requirements:
  - INFRA-03
  - INFRA-04
autonomous: true
---

## Goal

Add `StaleCacheError`, `load_stage_cache()`, and the `carry_forward` field to `core/context.py`. These are foundational primitives used by Plans 46.2 and 46.3.

## Locked Decisions (from CONTEXT.md)

- `StaleCacheError` defined in `core/context.py` alongside PipelineContext
- Standalone function `load_stage_cache(path) -> PipelineContext` in `core/context.py`
- Staleness detected via pickle load failure: catch `AttributeError`, `ModuleNotFoundError`, `pickle.UnpicklingError` during deserialization and raise `StaleCacheError`
- `StaleCacheError` message includes: cache file path, suggestion to re-run the pipeline, and the original exception for debugging
- Basic shape validation after successful deserialization (e.g., `frame_count == len(detections)`)
- Each cache is wrapped in a metadata envelope dict containing: `run_id`, `timestamp`, `stage_name`, `version_fingerprint`, and the PipelineContext data
- CarryForward state stored on `PipelineContext.carry_forward` field (optional, `CarryForward | None = None`)

## Tasks

<task id="46.1.1">
<title>Add carry_forward field to PipelineContext and define StaleCacheError</title>
<details>

**File:** `src/aquapose/core/context.py`

1. Add a new optional field to `PipelineContext`:
   ```python
   carry_forward: CarryForward | None = None
   ```
   Place it after `stage_timing` since it is cross-batch state, not a stage output. Update the class docstring to document it.

2. Define `StaleCacheError` as a new exception class:
   ```python
   class StaleCacheError(Exception):
       """Raised when a stage cache cannot be deserialized due to class evolution."""
   ```

3. Add a private helper function for version fingerprinting:
   ```python
   def _context_fingerprint(ctx: PipelineContext) -> str:
       """Hash of PipelineContext field names -- changes when dataclass structure changes."""
       import dataclasses
       import hashlib
       names = sorted(f.name for f in dataclasses.fields(ctx))
       return hashlib.sha256("|".join(names).encode()).hexdigest()[:12]
   ```
   This function is also used by DiagnosticObserver (Plan 46.2) when writing the envelope, so it must be importable from `core/context.py`. Make it public: `context_fingerprint()`.

</details>
</task>

<task id="46.1.2">
<title>Implement load_stage_cache function</title>
<details>

**File:** `src/aquapose/core/context.py`

Implement `load_stage_cache(path: str | Path) -> PipelineContext` following the pattern from RESEARCH.md:

```python
def load_stage_cache(path: str | Path) -> PipelineContext:
    """Load a stage cache pickle file and return the embedded PipelineContext.

    Args:
        path: Path to a *_cache.pkl file written by DiagnosticObserver.

    Returns:
        The deserialized PipelineContext.

    Raises:
        StaleCacheError: If deserialization fails (AttributeError,
            ModuleNotFoundError, pickle.UnpicklingError) or the envelope
            format is invalid or basic shape validation fails.
        FileNotFoundError: If path does not exist.
    """
```

Implementation requirements:
1. Read bytes from `Path(path)` and call `pickle.loads()`.
2. Wrap deserialization in `try/except` catching `(AttributeError, ModuleNotFoundError, pickle.UnpicklingError)` and re-raise as `StaleCacheError` with the message format: `"Cache file '{p}' is incompatible with the current codebase. Re-run the pipeline in diagnostic mode to regenerate it. Original error: {exc}"`.
3. After successful deserialization, validate the envelope is a `dict` with a `"context"` key. If not, raise `StaleCacheError` indicating the cache predates the envelope format (Pitfall 4 from RESEARCH.md).
4. Extract `ctx = envelope["context"]` and run basic shape validation: if `ctx.frame_count is not None and ctx.detections is not None`, check `ctx.frame_count == len(ctx.detections)`. Raise `StaleCacheError` on mismatch.
5. Return `ctx`.

Add required imports at top of file: `import pickle` and `from pathlib import Path`.

</details>
</task>

<task id="46.1.3">
<title>Update __init__.py exports and write unit tests</title>
<details>

**File:** `src/aquapose/core/__init__.py`

Add `StaleCacheError`, `load_stage_cache`, and `context_fingerprint` to the imports and `__all__` list.

**File:** `tests/unit/core/test_stage_cache.py` (new file)

Write unit tests covering:

1. **test_load_stage_cache_round_trip**: Create a PipelineContext with known field values (frame_count=5, detections=[{} for _ in range(5)]), wrap in an envelope dict with the expected keys (run_id, timestamp, stage_name, version_fingerprint, context), pickle it to a tmp file, call `load_stage_cache()`, assert the returned context has matching field values.

2. **test_load_stage_cache_stale_cache_error**: Write a pickle file containing a non-deserializable object (e.g., monkeypatch a module path that doesn't exist by writing raw bytes of a pickle that references a non-existent class). Alternatively: write a file that is valid pickle but not a dict (e.g., just a string), and assert `StaleCacheError` is raised with the expected message substring.

3. **test_load_stage_cache_invalid_envelope**: Pickle a raw `PipelineContext` (not wrapped in envelope dict) to a tmp file. Call `load_stage_cache()` and assert `StaleCacheError` is raised mentioning "envelope".

4. **test_load_stage_cache_shape_mismatch**: Create an envelope where `ctx.frame_count=5` but `ctx.detections` has 3 elements. Assert `StaleCacheError` is raised mentioning "frame_count" and "len(detections)".

5. **test_load_stage_cache_file_not_found**: Call `load_stage_cache("nonexistent.pkl")` and assert `FileNotFoundError` is raised.

6. **test_context_fingerprint_stable**: Call `context_fingerprint()` twice on the same PipelineContext and assert equal results.

7. **test_carry_forward_field_exists**: Create a `PipelineContext()`, assert `carry_forward is None`. Create one with `carry_forward=CarryForward()`, assert it is not None.

Use `tmp_path` pytest fixture for file creation. Import from `aquapose.core.context` directly.

Run `hatch run test` after writing tests to verify they pass.

</details>
</task>

## Verification

```bash
hatch run test tests/unit/core/test_stage_cache.py -v
hatch run check
```

- [ ] `StaleCacheError` is importable from `aquapose.core`
- [ ] `load_stage_cache` is importable from `aquapose.core`
- [ ] `context_fingerprint` is importable from `aquapose.core`
- [ ] `PipelineContext` has a `carry_forward` field defaulting to `None`
- [ ] All 7 unit tests pass
- [ ] `hatch run check` passes (lint + typecheck)

## must_haves

- `StaleCacheError` exception with cache path and re-run suggestion in message
- `load_stage_cache()` returns PipelineContext from envelope-wrapped pickle
- Envelope validation catches non-dict and missing "context" key
- Shape validation catches frame_count vs len(detections) mismatch
- `carry_forward: CarryForward | None = None` field on PipelineContext
- `context_fingerprint()` returns stable hash of PipelineContext field names
