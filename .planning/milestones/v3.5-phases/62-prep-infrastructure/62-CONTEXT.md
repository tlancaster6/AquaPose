# Phase 62: Prep Infrastructure - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can prepare calibrated keypoint t-values and pre-generated LUTs before running the pipeline. The pipeline fails fast with actionable errors if prep steps haven't been run. CLI wiring and fail-fast enforcement only -- no new algorithms or data formats.

</domain>

<decisions>
## Implementation Decisions

### LUT CLI design
- Command: `aquapose prep generate-luts --config config.yaml` -- reads calibration_path and lut config from the pipeline config YAML
- Skip if LUTs already exist on disk, with `--force` flag to regenerate
- Per-camera progress output (e.g. "Generated forward LUT for cam_01 (3/12)")
- Print the LUT cache directory path on completion
- LUT paths continue to use the existing convention (derived from calibration_path + lut config) -- no new config fields

### Fail-fast behavior
- **Keypoint t-values (PREP-02):** PoseEstimationBackend.__init__ raises if keypoint_t_values is None. Only the pose_estimation backend requires this -- segment_then_extract is unaffected.
- **LUTs (PREP-04):** LUT existence check at pipeline construction time (when stages are built), not at association runtime. Actionable error message: "LUTs not found. Run: aquapose prep generate-luts --config <path>"
- LUT check is conditional on whether association will run -- if --stop-after is set before association, the check is skipped (allows detection+tracking without pre-generated LUTs)
- All lazy LUT generation code removed from AssociationStage.run()

### calibrate-keypoints refinements
- Replace --output flag with --config (required) -- reads the pipeline config YAML and updates the keypoint_t_values field under the midline section in place
- No separate snippet file output -- the config is the single source of truth
- Simple pyyaml load/update/dump (comments in existing config will be stripped; acceptable tradeoff)

### Config integration
- init-config scaffold updated with YAML comments near the midline section reminding users to run `aquapose prep calibrate-keypoints`
- init-config prints a console message after creating the project reminding users to run LUT generation and t-value calibration before analyzing real data
- No new config fields for LUT paths (existing convention is sufficient)

### Claude's Discretion
- Exact error message wording beyond the required actionable hint
- Where in the pipeline construction to hook the LUT check (build_stages or PosePipeline.__init__)
- init-config comment/message wording

</decisions>

<specifics>
## Specific Ideas

- User wants early failure: "we shouldn't waste time with the preceding pipeline stages if we're going to fail at association"
- The LUT check should be as early as possible in the pipeline lifecycle, gated on whether association is actually going to run

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `calibrate-keypoints` command: Already exists at `src/aquapose/training/prep.py:26-150`. Needs interface change (--output -> --config) and config-writing logic.
- LUT generation functions: `generate_forward_luts()` and `generate_inverse_lut()` in `src/aquapose/calibration/luts.py`. Already handle undistortion maps correctly.
- LUT load/save: `load_forward_luts()`, `load_inverse_luts()`, `save_forward_luts()`, `save_inverse_luts()` in calibration/luts.py. Path derivation logic is reusable for the CLI.
- `prep_group`: Click group already wired into CLI at `src/aquapose/cli.py:430`.

### Established Patterns
- CLI commands use Click with `--config` for pipeline config path (see `run` command at cli.py:27-34)
- Config loading via `load_config(yaml_path=...)` from `engine/config.py`
- Stage construction happens in `build_stages()` -- natural place for fail-fast checks
- `stop_after` parameter already controls which stages are built

### Integration Points
- `AssociationStage.run()` at `core/association/stage.py:72-105`: Remove lazy LUT generation block (lines 76-105 become just the load calls with a fail-fast check)
- `PoseEstimationBackend.__init__()` at `core/midline/backends/pose_estimation.py:138-141`: Replace linspace fallback with raise
- `init_config()` at `cli.py:143-186`: Add comments and post-creation console message
- Pipeline construction (build_stages or PosePipeline.__init__): Add early LUT existence check

</code_context>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 62-prep-infrastructure*
*Context gathered: 2026-03-05*
