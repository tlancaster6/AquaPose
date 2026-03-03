# Plan 46.3: CLI --resume-from Flag and Integration Test

---
wave: 2
depends_on:
  - PLAN-01
files_modified:
  - src/aquapose/cli.py
  - tests/unit/engine/test_resume_cli.py
requirements:
  - INFRA-01
  - INFRA-02
  - INFRA-03
  - INFRA-04
autonomous: true
---

## Goal

Add `--resume-from` CLI flag to `aquapose run` that loads a stage cache and passes it as `initial_context` to PosePipeline. Write an integration-style test verifying the full round-trip: write cache via DiagnosticObserver, load via `--resume-from`, and confirm stage skipping.

## Locked Decisions (from CONTEXT.md)

- CLI flag `--resume-from path/to/cache.pkl` on `aquapose run` that loads context and skips upstream stages
- Each pipeline invocation (even resumed) gets its own run_id and output directory -- do not inherit run_id from cache (Pitfall 5 from RESEARCH.md)
- Standalone function `load_stage_cache(path) -> PipelineContext` used by CLI

## Tasks

<task id="46.3.1">
<title>Add --resume-from flag to aquapose run CLI command</title>
<details>

**File:** `src/aquapose/cli.py`

1. Add a new `@click.option` decorator to the `run` command:
   ```python
   @click.option(
       "--resume-from",
       "resume_from",
       type=click.Path(exists=True),
       default=None,
       help="Path to a stage cache pickle file. Skips stages whose outputs are already populated.",
   )
   ```

2. Add `resume_from: str | None` to the `run()` function signature.

3. After building stages (step 5 in the current code) and before creating the pipeline (step 7), add the resume logic:
   ```python
   # 5.5. Load initial context from cache if --resume-from provided
   initial_context = None
   if resume_from is not None:
       from aquapose.core.context import load_stage_cache
       initial_context = load_stage_cache(resume_from)
       click.echo(f"Loaded stage cache from {resume_from}")
   ```

4. Pass `initial_context` to `pipeline.run()`:
   ```python
   pipeline.run(initial_context=initial_context)
   ```
   (replaces the existing bare `pipeline.run()`)

5. Wrap the `load_stage_cache` call in a try/except for `StaleCacheError` and `FileNotFoundError`, converting them to `click.ClickException` for user-friendly error messages:
   ```python
   from aquapose.core.context import StaleCacheError, load_stage_cache
   try:
       initial_context = load_stage_cache(resume_from)
   except StaleCacheError as exc:
       raise click.ClickException(str(exc)) from exc
   except FileNotFoundError:
       raise click.ClickException(f"Cache file not found: {resume_from}")
   ```

</details>
</task>

<task id="46.3.2">
<title>Write integration test for --resume-from round-trip</title>
<details>

**File:** `tests/unit/engine/test_resume_cli.py` (new file)

This test verifies the full workflow: DiagnosticObserver writes a cache, CLI loads it, pipeline skips populated stages. Since the CLI test can't easily invoke a real pipeline, test the loading path directly.

1. **test_resume_from_loads_and_returns_context**: Create a PipelineContext with `frame_count=3, detections=[{} for _ in range(3)]`. Wrap it in an envelope dict, pickle to `tmp_path / "test_cache.pkl"`. Call `load_stage_cache(tmp_path / "test_cache.pkl")`. Assert the returned context has `frame_count == 3`.

2. **test_resume_from_stale_cache_gives_click_exception**: Use `click.testing.CliRunner` to invoke the `run` command with `--resume-from` pointing to a corrupt pickle file. Assert the result exit code is non-zero and the output contains "incompatible" or "re-run".

3. **test_resume_from_nonexistent_file_gives_click_exception**: Use `click.testing.CliRunner` to invoke the `run` command with `--resume-from` pointing to a path that does not exist. Assert the exit code is non-zero (Click's `exists=True` on the option handles this automatically).

4. **test_end_to_end_cache_write_and_reload**: Create a DiagnosticObserver with `output_dir=tmp_path`. Send PipelineStart, then StageComplete for DetectionStage with a populated PipelineContext. Verify the cache file exists. Load it with `load_stage_cache()`. Create a stub pipeline with the loaded context as `initial_context`. Verify the DetectionStage would be skipped (by checking that its output fields are populated in the context, matching the `_STAGE_OUTPUT_FIELDS` logic).

Use `tmp_path` pytest fixture. Import `load_stage_cache`, `StaleCacheError` from `aquapose.core.context`.

Run `hatch run test` after writing tests to verify they pass.

</details>
</task>

## Verification

```bash
hatch run test tests/unit/engine/test_resume_cli.py -v
hatch run check
```

- [ ] `aquapose run --help` shows `--resume-from` option
- [ ] `--resume-from` with a valid cache file loads context and passes to pipeline.run()
- [ ] `--resume-from` with a corrupt file shows a user-friendly error message
- [ ] `--resume-from` with a nonexistent file shows a user-friendly error message
- [ ] All tests pass
- [ ] `hatch run check` passes

## must_haves

- `--resume-from` option on `aquapose run` CLI command
- Loading a cache via `--resume-from` passes `initial_context` to PosePipeline.run()
- StaleCacheError from load is converted to click.ClickException
- Each resumed run gets its own run_id (not inherited from cache)
