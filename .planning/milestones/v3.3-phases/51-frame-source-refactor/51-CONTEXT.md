# Phase 51: Frame Source Refactor - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Extract video I/O from DetectionStage and MidlineStage into an injectable frame source. Remove `stop_frame` from pipeline config. Delete VideoSet after migration. All existing pipeline behavior preserved — stages and observers consume frames from the source instead of opening videos themselves.

</domain>

<decisions>
## Implementation Decisions

### Frame source interface
- Protocol class using `typing.Protocol`, living in `core/types/`
- Replaces VideoSet entirely — VideoSet is deleted after all consumers migrate
- Yields `(frame_idx, dict[str, ndarray])` per iteration (same shape VideoSet currently yields)
- Metadata properties (camera_ids, frame_count) left to Claude's discretion

### Stage constructor changes
- Constructor injection: frame source passed to `__init__()` of DetectionStage and MidlineStage
- `build_stages()` creates the frame source and passes it to both stages
- Frame source owns: video discovery (glob *.avi/*.mp4, camera ID extraction), calibration loading, undistortion map computation, path validation
- Stages no longer receive `video_dir` or `calibration_path` for frame access (MidlineStage still needs `calibration_path` for LUT-based orientation resolution)

### Observer frame access
- Refactor overlay_observer and tracklet_trail_observer in this phase (not deferred)
- Frame source passed to observers via `observer_factory`
- Eliminates all direct VideoSet usage in one phase — no stragglers

### stop_frame removal
- Remove `stop_frame` from `PipelineConfig`
- Add `stop_frame` to `_RENAME_HINTS` so existing YAML configs get a clear error with migration hint
- Replacement: `max_frames` parameter on the frame source constructor
- Update eval/tune CLI commands that rely on stop_frame (via n_frames) to use frame source's max_frames
- User will manually update external project YAML configs (~/aquapose/projects/YH/config.yaml)

### Claude's Discretion
- Whether both stages share the same frame source instance or get separate instances
- Exact metadata properties on the protocol (camera_ids, __len__, etc.)
- Internal implementation of the concrete frame source (how it manages video handles, context manager pattern)
- Whether frame source uses a context manager or manages lifecycle internally

</decisions>

<specifics>
## Specific Ideas

- "Can't the new frame source protocol just replace the video set?" — Yes, VideoSet is deleted entirely. The concrete frame source absorbs all of VideoSet's responsibilities.
- The concrete implementation should handle everything VideoSet currently does: opening video captures, synchronized multi-camera reading, undistortion, EOF handling.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `VideoSet` (`io/video.py`): Already yields `(frame_idx, dict[str, ndarray])` with undistortion — the concrete frame source absorbs this implementation
- `discover_camera_videos()` (`io/discovery.py`): Video discovery utility, can be reused inside the frame source
- `compute_undistortion_maps()` and `undistort_image()` (`calibration/loader.py`): Undistortion utilities used by current VideoSet
- `_RENAME_HINTS` dict in `engine/config.py`: Existing pattern for migration hints on removed config fields

### Established Patterns
- Constructor injection: backends are already wired via `__init__()` parameters in both stages
- `typing.Protocol` for structural typing: used for Stage protocol in engine
- Frozen dataclass configs: all stage configs are frozen dataclasses
- `observer_factory.py`: centralized factory for observer construction, natural place to inject frame source

### Integration Points
- `build_stages()` in `engine/pipeline.py`: creates all stages, will create the frame source and pass it
- `observer_factory.py`: constructs observers, will receive and pass frame source to overlay/tracklet observers
- `DetectionStage.__init__()` and `MidlineStage.__init__()`: lose video_dir, calibration_path (for frames), stop_frame params
- `PipelineConfig`: loses stop_frame field
- `evaluation/harness.py` and tune CLI: currently use stop_frame indirectly via n_frames, need migration

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 51-frame-source-refactor*
*Context gathered: 2026-03-03*
