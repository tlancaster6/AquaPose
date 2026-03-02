# Phase 38: Stabilization and Tech Debt Cleanup - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Training data and config infrastructure uses standard YOLO txt+yaml format (not NDJSON), config fields are consolidated and init-config generates correct defaults, stale docstrings are updated, GUIDEBOOK.md is audited for accuracy, and dead legacy code is removed or migrated. No new features — infrastructure polish to close out v3.0.

</domain>

<decisions>
## Implementation Decisions

### NDJSON → txt+yaml label format
- Remove all NDJSON generation and consumption code entirely — no fallback, no dual format
- All 3 build_yolo_training_data modes (obb, seg, pose) output standard YOLO txt labels + dataset.yaml
- flip_idx in dataset.yaml for pose mode only (identity mapping `[0, 1, 2, 3, 4, 5]` so Ultralytics enables fliplr augmentation)
- OBB and seg modes don't need flip_idx — Ultralytics handles their augmentation natively
- User will delete and regenerate existing NDJSON datasets externally — those files are outside codebase scope
- Files to modify: `scripts/build_yolo_training_data.py`, `tmp/convert_all_annotations.py`, `src/aquapose/training/yolo_obb.py`, `yolo_pose.py`, `yolo_seg.py`, `src/aquapose/training/cli.py`

### Config field consolidation
- Remove `keypoint_weights_path` from MidlineConfig — just delete the field, no deprecation logic (pre-release code)
- Both segmentation and pose_estimation backends read from the single `weights_path` field
- Rename `model_path` → `weights_path` in DetectionConfig for consistency across all stages
- Generic docstring on unified `weights_path`: "Path to model weights for the active midline backend (segmentation or pose estimation)"

### init-config defaults
- Default detection backend: `yolo_obb`
- Default midline backend: `pose_estimation`
- Do not hardcode keypoint count (6) in generated config — let it come from config or model metadata
- Show essential fields + all backend selections with brief comments; hide internal tuning params
- Sensible project-relative default paths:
  - `video_dir`: `"videos"`
  - `geometry_path`: `"geometry/calibration.json"`
  - `output_dir`: `"runs"`
  - Detection `weights_path`: `"models/yolo_obb.pt"`
  - Midline `weights_path`: `"models/yolo_pose.pt"`

### Stale docstring cleanup
- Targeted grep for known stale terms: U-Net, UNet, no-op stub, "Phase 37 pending", legacy backend names
- Fix all matches in src/ — update to reflect current Ultralytics-based implementation
- Known locations: `core/midline/stage.py`, `reconstruction/midline.py`, `core/reconstruction/stage.py`

### GUIDEBOOK.md audit
- Full accuracy pass against current codebase — every section checked
- Regenerate source layout tree from actual filesystem
- Allow restructuring if sections are unclear or poorly organized
- Preserve the document's role as the authoritative architecture reference
- Update milestone history to include v3.0 completion
- Fix any references to removed code (U-Net, MOG2, SAM2, custom models)

### Dead code cleanup
- Audit legacy top-level directories: `reconstruction/`, `segmentation/`, `tracking/`, `visualization/` (and any others found)
- Analysis-first approach: produce an import analysis report showing which files are unused, which are thin wrappers of core/ code, and which contain unique logic
- Report must present evidence before any deletion
- Files with unique logic not duplicated elsewhere: migrate into `core/`, `engine/`, or `io/` as appropriate, then delete the legacy file
- Purely dead or wrapper-only files: delete after report confirms no external imports

### Claude's Discretion
- Exact grep patterns for stale docstring scan
- Order of operations across the 6 work areas
- Whether to consolidate small changes into shared plans or keep them separate
- GUIDEBOOK.md restructuring decisions (section ordering, heading levels, content grouping)

</decisions>

<specifics>
## Specific Ideas

- "This code is effectively pre-release — avoid unnecessary deprecation logic" (applies to config field removal)
- "The agent must report and present evidence before deleting anything" (dead code cleanup)
- Detection `model_path` → `weights_path` rename was discovered during discussion and added to scope

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/build_yolo_training_data.py`: existing multi-mode (obb/seg/pose) script — modify output format in place
- `src/aquapose/training/yolo_*.py`: three training wrappers that currently consume NDJSON — update data parameter
- `src/aquapose/engine/config.py`: MidlineConfig has both `weights_path` (line 110) and `keypoint_weights_path` (line 120)
- `src/aquapose/cli.py`: contains `init-config` command implementation

### Established Patterns
- Backend registration via stage-level registry resolved from config at pipeline construction
- Config is frozen dataclasses with hierarchical stage subtrees
- Training wrappers follow the yolo_obb.py pattern (established in Phase 31)

### Integration Points
- `config.py` line 621: `mid_kwargs` resolution reads both weight fields — must update when consolidating
- Training CLI (`training/cli.py`): help text references NDJSON — update to txt+yaml
- `init-config` CLI command generates YAML template — must reflect all config changes

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 38-stabilization-and-tech-debt-cleanup*
*Context gathered: 2026-03-02*
