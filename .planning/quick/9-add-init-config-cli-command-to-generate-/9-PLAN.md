---
phase: quick
plan: 9
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/cli.py
  - tests/unit/engine/test_cli.py
autonomous: true
requirements: []

must_haves:
  truths:
    - "Running `aquapose init-config` writes a valid YAML config file"
    - "Running `aquapose init-config --output custom.yaml` writes to the specified path"
    - "The generated YAML contains all pipeline config sections with defaults"
    - "Running init-config when the file already exists refuses to overwrite unless --force is given"
  artifacts:
    - path: "src/aquapose/cli.py"
      provides: "init-config Click subcommand"
      contains: "def init_config"
    - path: "tests/unit/engine/test_cli.py"
      provides: "Tests for init-config subcommand"
      contains: "TestInitConfig"
  key_links:
    - from: "src/aquapose/cli.py"
      to: "aquapose.engine.config"
      via: "PipelineConfig() + serialize_config()"
      pattern: "serialize_config.*PipelineConfig"
---

<objective>
Add an `init-config` CLI subcommand that generates a default template YAML config file.

Purpose: Users need a way to scaffold a config file with all available options and their defaults, rather than writing YAML from scratch or reading source code.
Output: Working `aquapose init-config` command with tests.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/cli.py
@src/aquapose/engine/config.py
@tests/unit/engine/test_cli.py
</context>

<interfaces>
<!-- Key types and contracts the executor needs. -->

From src/aquapose/engine/config.py:
```python
@dataclass(frozen=True)
class PipelineConfig:
    run_id: str = ""
    output_dir: str = ""
    video_dir: str = ""
    calibration_path: str = ""
    mode: str = "production"
    detection: DetectionConfig = ...
    midline: MidlineConfig = ...
    association: AssociationConfig = ...
    tracking: TrackingConfig = ...
    reconstruction: ReconstructionConfig = ...
    synthetic: SyntheticConfig = ...

def serialize_config(config: PipelineConfig) -> str:
    """Serialize config to a YAML string via dataclasses.asdict + yaml.dump."""
```

From src/aquapose/cli.py:
```python
@click.group()
def cli() -> None: ...

def main() -> None:
    cli()
```

The CLI uses Click groups. `cli` is the top-level group; `run` is the existing subcommand.
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Add init-config subcommand and tests</name>
  <files>src/aquapose/cli.py, tests/unit/engine/test_cli.py</files>
  <action>
Add an `init-config` subcommand to the existing `cli` Click group in `src/aquapose/cli.py`:

1. Add `serialize_config` to the existing import from `aquapose.engine` (it is already exported there).

2. Create the subcommand registered on `cli`:
```python
@cli.command("init-config")
@click.option(
    "--output", "-o",
    default="aquapose.yaml",
    type=click.Path(),
    help="Output file path (default: aquapose.yaml).",
)
@click.option(
    "--force", is_flag=True, default=False,
    help="Overwrite existing file.",
)
def init_config(output: str, force: bool) -> None:
```

3. Implementation:
   - Create a default `PipelineConfig()` (no args = all defaults).
   - Call `serialize_config(config)` to produce YAML string.
   - Check if `output` path already exists; if so, abort with error message unless `--force`.
   - Write YAML to the output path.
   - Print confirmation message: `click.echo(f"Config written to {output}")`.

4. Add `TestInitConfig` class to `tests/unit/engine/test_cli.py` with these tests:
   - `test_init_config_creates_default_file`: invoke `["init-config"]` in an isolated filesystem, verify `aquapose.yaml` exists and contains expected YAML sections (detection, midline, association, tracking, reconstruction).
   - `test_init_config_custom_output`: invoke with `["init-config", "-o", "custom.yaml"]`, verify `custom.yaml` created.
   - `test_init_config_refuses_overwrite`: create file first, invoke without --force, assert exit_code != 0 and file unchanged.
   - `test_init_config_force_overwrites`: create file first, invoke with `--force`, assert exit_code == 0 and file updated.
   - `test_init_config_help`: invoke `["init-config", "--help"]`, assert exit_code == 0 and "--output" in output.

Use `runner.isolated_filesystem()` context manager for all file-writing tests to avoid polluting the real filesystem.
  </action>
  <verify>
    <automated>cd C:/Users/tucke/PycharmProjects/AquaPose && hatch run python -m pytest tests/unit/engine/test_cli.py -x -v -k "InitConfig or init_config" 2>&1 | tail -30</automated>
  </verify>
  <done>
    - `aquapose init-config` writes aquapose.yaml with all default config sections
    - `aquapose init-config -o custom.yaml` writes to custom path
    - Refuses overwrite without --force; overwrites with --force
    - All new tests pass, existing CLI tests still pass
  </done>
</task>

</tasks>

<verification>
```bash
cd C:/Users/tucke/PycharmProjects/AquaPose && hatch run python -m pytest tests/unit/engine/test_cli.py -x -v
```
All CLI tests (existing + new) pass.

```bash
cd C:/Users/tucke/PycharmProjects/AquaPose && hatch run check
```
Lint and typecheck pass.
</verification>

<success_criteria>
- `aquapose init-config` generates a valid YAML config with all pipeline defaults
- --output and --force flags work correctly
- All tests pass, lint and typecheck clean
</success_criteria>

<output>
After completion, create `.planning/quick/9-add-init-config-cli-command-to-generate-/9-SUMMARY.md`
</output>
