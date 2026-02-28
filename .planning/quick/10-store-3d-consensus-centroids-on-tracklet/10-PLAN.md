---
phase: 10-store-3d-consensus-centroids-on-tracklet
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/core/association/types.py
  - src/aquapose/core/association/refinement.py
  - src/aquapose/engine/diagnostic_observer.py
  - tests/unit/core/association/test_refinement.py
  - tests/unit/engine/test_diagnostic_observer.py
autonomous: true
requirements: [CENTROID-01, CENTROID-02]

must_haves:
  truths:
    - "TrackletGroup carries per-frame 3D consensus centroids after refinement"
    - "Groups that skip refinement (< min_cameras or disabled) have consensus_centroids=None"
    - "DiagnosticObserver can serialize 2D-to-3D centroid correspondences to disk as NPZ"
  artifacts:
    - path: "src/aquapose/core/association/types.py"
      provides: "TrackletGroup.consensus_centroids field"
      contains: "consensus_centroids"
    - path: "src/aquapose/core/association/refinement.py"
      provides: "Consensus centroids populated on refined TrackletGroups"
      contains: "consensus_centroids"
    - path: "src/aquapose/engine/diagnostic_observer.py"
      provides: "NPZ export of 2D-3D correspondences"
      contains: "export_centroid_correspondences"
    - path: "tests/unit/core/association/test_refinement.py"
      provides: "Tests for consensus_centroids on TrackletGroup"
      contains: "consensus_centroids"
  key_links:
    - from: "src/aquapose/core/association/refinement.py"
      to: "src/aquapose/core/association/types.py"
      via: "TrackletGroup constructor with consensus_centroids kwarg"
      pattern: "consensus_centroids="
    - from: "src/aquapose/engine/diagnostic_observer.py"
      to: "src/aquapose/core/association/types.py"
      via: "reads TrackletGroup.consensus_centroids + tracklets for export"
      pattern: "consensus_centroids"
---

<objective>
Add per-frame 3D consensus centroids to TrackletGroup and provide a DiagnosticObserver method to export 2D-to-3D centroid correspondences as NPZ files for calibration fine-tuning.

Purpose: High-fidelity 2D pixel centroid to 3D world point correspondences are needed for iterative calibration refinement. The refinement code already computes these consensus points but discards them.
Output: TrackletGroup with consensus_centroids field; DiagnosticObserver.export_centroid_correspondences() method.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/core/association/types.py
@src/aquapose/core/association/refinement.py
@src/aquapose/engine/diagnostic_observer.py
@src/aquapose/core/context.py
@tests/unit/core/association/test_refinement.py
@tests/unit/engine/test_diagnostic_observer.py

<interfaces>
<!-- TrackletGroup is frozen dataclass; new field must have default=None -->
From src/aquapose/core/association/types.py:
```python
@dataclass(frozen=True)
class TrackletGroup:
    fish_id: int
    tracklets: tuple
    confidence: float | None = None
    per_frame_confidence: tuple | None = None
```

From src/aquapose/core/tracking/types.py:
```python
@dataclass(frozen=True)
class Tracklet2D:
    camera_id: str
    track_id: int
    frames: tuple          # tuple[int, ...]
    centroids: tuple       # tuple[tuple[float, float], ...]
    bboxes: tuple
    frame_status: tuple
```

From src/aquapose/core/association/refinement.py:
```python
def _compute_frame_consensus(...) -> dict[int, np.ndarray | None]:
    # Already computes per-frame 3D consensus — currently discarded after use
```

From src/aquapose/engine/diagnostic_observer.py:
```python
class StageSnapshot:
    tracklet_groups: list | None = None  # stores ref to PipelineContext.tracklet_groups

class DiagnosticObserver:
    stages: dict[str, StageSnapshot]
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add consensus_centroids field to TrackletGroup and populate in refinement</name>
  <files>
    src/aquapose/core/association/types.py
    src/aquapose/core/association/refinement.py
    tests/unit/core/association/test_refinement.py
  </files>
  <action>
1. In `types.py`, add a new field to TrackletGroup:
   ```python
   consensus_centroids: tuple | None = None
   ```
   This stores per-frame 3D consensus centroids as a tuple of (frame_idx, ndarray|None) pairs. Use `tuple[tuple[int, np.ndarray | None], ...]` semantically but declared as generic `tuple` to match existing style (same as tracklets field). Add docstring explaining the field: mapping from frame index to 3D consensus point (shape (3,)) or None when fewer than 2 rays available. None when refinement was skipped or group is an evicted singleton.

2. In `refinement.py`, update all three TrackletGroup construction sites:

   a. **Lines 134-139** (evicted singletons): Keep `consensus_centroids=None` (singletons have no multi-view consensus).

   b. **Lines 165-172** (refined groups): Pass `consensus_centroids=tuple((f, cleaned_consensus.get(f)) for f in frame_list)` using the already-computed `cleaned_consensus` dict. The `cleaned_consensus` variable at line 152 already has exactly the data we need.

   c. **Lines 177-180** (evicted singleton ID reassignment): Carry through `singleton.consensus_centroids` (which is None from step a).

3. In `test_refinement.py`, add tests:
   - `test_consensus_centroids_populated_after_refinement`: Verify refined group has non-None consensus_centroids, is a tuple, entries are (frame_idx, ndarray_or_None) pairs, and 3D points have shape (3,).
   - `test_consensus_centroids_none_for_skipped_groups`: Verify groups below min_cameras have consensus_centroids=None.
   - `test_consensus_centroids_none_for_evicted_singletons`: Verify evicted singleton groups have consensus_centroids=None.
   - `test_consensus_centroids_none_when_disabled`: Verify refinement_enabled=False preserves None.
  </action>
  <verify>hatch run test -- tests/unit/core/association/test_refinement.py -x</verify>
  <done>
    - TrackletGroup has consensus_centroids field defaulting to None
    - Refined groups carry per-frame 3D consensus centroids as tuple of (frame, point) pairs
    - Skipped/evicted groups have consensus_centroids=None
    - All existing and new tests pass
  </done>
</task>

<task type="auto">
  <name>Task 2: Add export_centroid_correspondences method to DiagnosticObserver</name>
  <files>
    src/aquapose/engine/diagnostic_observer.py
    tests/unit/engine/test_diagnostic_observer.py
  </files>
  <action>
1. In `diagnostic_observer.py`, add a method `export_centroid_correspondences(self, output_path: Path | str) -> Path` to DiagnosticObserver:

   ```python
   def export_centroid_correspondences(self, output_path: Path | str) -> Path:
   ```

   Logic:
   - Get the AssociationStage snapshot from `self.stages`. If not present, raise ValueError with clear message.
   - Read `tracklet_groups` from the snapshot. If None or empty, raise ValueError.
   - For each TrackletGroup with non-None consensus_centroids:
     - For each (frame_idx, point_3d) pair where point_3d is not None:
       - For each tracklet in the group, if the tracklet has this frame:
         - Record a correspondence row: [fish_id, frame_idx, x3d, y3d, z3d, camera_id, u_px, v_px]
   - Save as NPZ with arrays:
     - `fish_ids`: int array, shape (N,)
     - `frame_indices`: int array, shape (N,)
     - `points_3d`: float64 array, shape (N, 3)
     - `camera_ids`: object array of strings, shape (N,)
     - `centroids_2d`: float64 array, shape (N, 2) — pixel (u, v)
   - Return the resolved output path.

   Add necessary imports: `from pathlib import Path` and `import numpy as np`.

2. In `test_diagnostic_observer.py`, add tests:
   - `test_export_centroid_correspondences_writes_npz`: Create a mock scenario with TrackletGroups that have consensus_centroids set, feed through DiagnosticObserver, call export, verify NPZ file exists and contains expected arrays with correct shapes.
   - `test_export_centroid_correspondences_raises_without_association`: Verify ValueError when no AssociationStage snapshot exists.
   - `test_export_centroid_correspondences_skips_none_consensus`: Groups with consensus_centroids=None produce no rows (not an error, just skipped).

   For the mock TrackletGroups in tests, construct them directly with known consensus_centroids values (no need to run actual refinement). Import TrackletGroup and Tracklet2D. Create simple tracklets with known frames/centroids and groups with known consensus tuples containing ndarray points.
  </action>
  <verify>hatch run test -- tests/unit/engine/test_diagnostic_observer.py -x</verify>
  <done>
    - DiagnosticObserver.export_centroid_correspondences() writes NPZ with fish_ids, frame_indices, points_3d, camera_ids, centroids_2d arrays
    - Each row is one camera observation of one 3D consensus point (2D pixel centroid paired with 3D world point)
    - Raises ValueError if no AssociationStage data available
    - All existing and new tests pass
  </done>
</task>

</tasks>

<verification>
```bash
# Run all affected test modules
hatch run test -- tests/unit/core/association/test_refinement.py tests/unit/engine/test_diagnostic_observer.py -x

# Type check the modified files
hatch run typecheck

# Lint check
hatch run lint
```
</verification>

<success_criteria>
- TrackletGroup.consensus_centroids field exists with None default
- refine_clusters() populates consensus_centroids on refined groups using cleaned_consensus
- DiagnosticObserver.export_centroid_correspondences() writes structured NPZ with 2D-3D pairs
- All unit tests pass (existing + new)
- Type checking and linting pass
</success_criteria>

<output>
After completion, create `.planning/quick/10-store-3d-consensus-centroids-on-tracklet/10-SUMMARY.md`
</output>
