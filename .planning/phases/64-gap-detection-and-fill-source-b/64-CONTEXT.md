# Phase 64: Gap Detection and Fill (Source B) - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Identify cameras where a reconstructed fish should be visible (via inverse LUT) but was not detected, classify each gap by failure reason, and generate corrective pseudo-labels by reprojecting the 3D reconstruction into those gap cameras. Also restructure the pseudo-label output directory to support separate consensus (Source A) and gap (Source B) subsets with independent metadata.

</domain>

<decisions>
## Implementation Decisions

### Visibility Determination
- Use inverse LUT `ghost_point_lookup()` on the fish's 3D centroid (mean of control points) to determine which cameras should see each fish
- Single centroid query per fish per frame -- no multi-point spline sampling needed
- Gap cameras are those flagged visible by the LUT but NOT in the midline's `per_camera_residuals` (i.e., did not contribute to the reconstruction)
- Contributing cameras are never flagged as gaps, even if their residual is high (Source A handles per-camera residual filtering separately)
- Minimum camera floor: 3 contributing cameras required before gap detection activates (configurable via `--min-cameras`, default 3)

### Gap Classification
- 3-tier per-frame classification: `no-detection`, `no-tracklet`, `failed-midline`
- Classification logic checks pipeline stages in reverse order:
  1. Check if any detection bbox in the gap camera/frame contains the projected 3D centroid -- if none, classify as `no-detection`
  2. Check if a tracklet covers the matching detection -- if not, classify as `no-tracklet`
  3. Otherwise (detection and tracklet exist but camera didn't contribute to reconstruction), classify as `failed-midline`
- Classification is per-frame, not per-tracklet -- each gap frame gets its own reason
- Generate gap-fill labels for ALL gap reasons (no pre-filtering by reason; Phase 65 can filter by reason during dataset assembly)

### Confidence & Quality
- Gap labels use the same `compute_confidence_score()` as Source A, based on the fish-level reconstruction quality (mean_residual, n_cameras, per_camera_variance from contributing cameras)
- No discount factor or separate formula for gap labels -- the fish-level score reflects reconstruction trustworthiness
- Basic bounds check on reprojected labels: skip if the projected spline falls mostly outside image bounds or OBB area is degenerate
- Camera floor (contributing cameras >= 3) is the primary quality gate for gap labeling

### Output Structure
- Restructured pseudo-label directory layout:
  ```
  run_dir/pseudo_labels/
    consensus/{obb,pose}/{images,labels}/train/   (was: {obb,pose}/...)
    gap/{obb,pose}/{images,labels}/train/
  ```
- Source A output renamed from `pseudo_labels/{obb,pose}/` to `pseudo_labels/consensus/{obb,pose}/` (breaking change, done as part of Phase 64)
- Each subset has its own `dataset.yaml` and `confidence.json` sidecar
- Gap sidecar adds per-label fields: `gap_reason` (no-detection | no-tracklet | failed-midline) and `n_source_cameras` (contributing camera count)

### CLI Design
- `aquapose pseudo-label generate` refactored with `--consensus` and `--gaps` flags
- At least one flag required; both can be passed together: `aquapose pseudo-label generate --consensus --gaps --config path/to/config.yaml`
- `--consensus` generates Source A labels (same logic as current `generate`)
- `--gaps` generates Source B gap-fill labels
- `--min-cameras` flag (default 3) controls the contributing-camera floor for gap detection
- Existing `--lateral-pad` and `--max-camera-residual` flags apply to consensus labels; gap labels inherit `--lateral-pad` but skip per-camera residual filtering

### Claude's Discretion
- Exact centroid computation method (mean of control points vs spline midpoint evaluation)
- Detection bbox overlap check implementation details (AABB containment vs center distance)
- Whether to share video frame reads between consensus and gap generation when both flags are passed
- Internal iteration strategy over diagnostic cache chunks

</decisions>

<specifics>
## Specific Ideas

- The directory restructure from `{obb,pose}/` to `consensus/{obb,pose}/` is a deliberate breaking change -- no backward compatibility shim needed since pseudo-label generation is new and not yet in any workflow
- Gap classification should use the projected centroid (via RefractiveProjectionModel.project()) for the bbox overlap check, not the LUT pixel coords (which are voxel-resolution approximations)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ghost_point_lookup()` (`calibration/luts.py`): Takes 3D points, returns per-camera visibility and pixel coords from InverseLUT. Core of visibility determination.
- `generate_fish_labels()` (`training/pseudo_labels.py`): Generates OBB + pose labels for one fish in one camera. Can be adapted for gap cameras by bypassing the per_camera_residuals check.
- `compute_confidence_score()` (`training/pseudo_labels.py`): Fish-level confidence scoring. Reusable as-is for gap labels.
- `RefractiveProjectionModel.project()` (`calibration/projection.py`): Precise 3D-to-2D projection for centroid reprojection during gap classification.
- `InverseLUT` (`calibration/luts.py`): Has `visibility_mask`, `projected_pixels`, `camera_ids`, `voxel_centers`. Loaded from disk via `load_inverse_luts()`.
- `PseudoLabelConfig` (`training/pseudo_label_cli.py`): Existing config dataclass for pseudo-label generation parameters.

### Established Patterns
- CLI groups via Click with `pseudo_label_group` already registered
- Dynamic import of engine modules in training CLI to avoid import boundary violation
- Diagnostic cache iteration via `load_run_context()` from `evaluation/runner.py`
- Confidence sidecar as JSON file alongside labels

### Integration Points
- Diagnostic caches at `run_dir/diagnostics/` contain `context.detections`, `context.tracks_2d`, `context.tracklet_groups`, `context.midlines_3d` -- all needed for gap classification
- InverseLUT must be loaded separately (not in diagnostic cache) -- requires calibration path and LUT config from the frozen run config
- `pseudo_label_group` in `training/pseudo_label_cli.py` is the CLI entrypoint to modify
- Phase 65 will consume both `consensus/` and `gap/` directories with independent confidence thresholds

</code_context>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 64-gap-detection-and-fill-source-b*
*Context gathered: 2026-03-05*
