# Phase 27: Diagnostic Visualization - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Observer-based diagnostic outputs for inspecting 2D tracking and cross-camera association quality. Produces per-camera centroid trail videos and a cross-camera association mosaic, color-coded by global fish ID. This is a Layer 3 Observer — passive, no mutation of pipeline state.

</domain>

<decisions>
## Implementation Decisions

### Trail rendering
- Fading tail trails with configurable length (default 30 frames, ~1 second at 30fps)
- Coasted frames distinguished from detected frames via color shift (lighter/grayed shade of the fish's assigned color)
- Global fish ID label displayed at the trail head (current centroid position)
- No local track IDs shown — global fish ID is the authoritative identity

### Association overlay layout
- Grid mosaic of all cameras tiled in a single video
- Same global fish ID gets the same color across all camera tiles
- Original camera frame as background with trails/markers drawn on top
- Each camera tile downsampled before compositing to keep mosaic dimensions reasonable
- No special highlighting for split/merge events — color-coding is sufficient

### Color scheme
- Use the 22-color Paul Tol-based COLORS palette from the dissertation style module
- Palette defined in `C:/Users/tucke/PycharmProjects/DissertationFigures/src/dissertationfigures/core/style.py`
- Colors assigned to global fish IDs, wrapping if more fish than colors
- Exact hex values: blue (#003070), gold (#EAD34C), teal (#44AA99), cyan (#66CCEE), green (#228833), olive (#999933), coral (#EE6677), rose (#EE99AA), purple (#AA3377), indigo (#332288), orange (#EE7733), red (#CC3311), wine (#882255), magenta (#EE3377), plum (#AA4499), steel (#4477AA), sky (#77AADD), sand (#DDCC77), lime (#BBCC33), peach (#EE8866), gray (#BBBBBB), dark gray (#777777)

### Output format
- MP4 container with H.264 codec
- Frame rate matches source video FPS
- Files written to `observers/diagnostics/` under the run artifact directory
- Descriptive file names: `tracklet_trails_{camera_id}.mp4`, `association_mosaic.mp4`

### Activation & configuration
- Always enabled in diagnostic mode (no separate config flag)
- All active cameras included in trail videos and mosaic — no camera subset selection
- Processes full run scope (no frame range option)
- No snapshot images — video output only

### Claude's Discretion
- Mosaic grid arrangement (rows x cols for 12 cameras)
- Exact downsampling resolution per tile
- Trail line thickness and opacity curve for the fade
- Font size and positioning for ID labels
- How to handle cameras with zero detections in the mosaic (show empty tile vs skip)

</decisions>

<specifics>
## Specific Ideas

- Color palette must match dissertation figures for visual consistency — hardcode the COLORS hex values directly (don't import from DissertationFigures, that's an external project)
- Downsampling is important for mosaic — 12 full-resolution camera tiles would produce an unwieldy video

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 27-diagnostic-visualization*
*Context gathered: 2026-02-27*
