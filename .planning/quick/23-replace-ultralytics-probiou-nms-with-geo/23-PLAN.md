---
phase: 23-replace-ultralytics-probiou-nms-with-geo
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - src/aquapose/core/detection/backends/yolo_obb.py
  - src/aquapose/engine/config.py
  - src/aquapose/engine/pipeline.py
  - tests/unit/core/detection/test_detection_stage.py
autonomous: true
requirements: [QUICK-23]

must_haves:
  truths:
    - "YOLO internal NMS is effectively disabled (iou=0.95) so probiou never suppresses"
    - "Geometric polygon IoU NMS removes duplicate OBBs on the same fish"
    - "iou_threshold is configurable from YAML via detection.iou_threshold"
  artifacts:
    - path: "src/aquapose/core/detection/backends/yolo_obb.py"
      provides: "Geometric polygon NMS using Shapely"
      contains: "polygon_nms"
    - path: "src/aquapose/engine/config.py"
      provides: "iou_threshold field on DetectionConfig"
      contains: "iou_threshold"
    - path: "pyproject.toml"
      provides: "Shapely dependency"
      contains: "shapely"
  key_links:
    - from: "src/aquapose/engine/pipeline.py"
      to: "DetectionStage constructor"
      via: "passes config.detection.iou_threshold as iou_threshold kwarg"
      pattern: "iou_threshold=config\\.detection\\.iou_threshold"
    - from: "src/aquapose/core/detection/backends/yolo_obb.py"
      to: "_parse_results"
      via: "geometric NMS applied after YOLO result parsing"
      pattern: "polygon_nms"
---

<objective>
Replace Ultralytics probiou-based NMS in YOLOOBBBackend with geometric polygon IoU NMS using Shapely. Probiou (Gaussian approximation) underestimates overlap for elongated fish OBBs, causing duplicate detections on the same fish at low conf thresholds.

Purpose: Eliminate duplicate OBB detections by using exact polygon intersection instead of Gaussian approximation for IoU.
Output: Modified YOLOOBBBackend with geometric NMS, configurable iou_threshold on DetectionConfig, Shapely added as dependency.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/core/detection/backends/yolo_obb.py
@src/aquapose/engine/config.py
@src/aquapose/engine/pipeline.py
@tests/unit/core/detection/test_detection_stage.py

<interfaces>
<!-- Key types and contracts the executor needs -->

From src/aquapose/core/types/detection.py:
```python
@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    mask: np.ndarray | None
    area: int
    confidence: float
    angle: float | None = None
    obb_points: np.ndarray | None = None  # (4, 2) corners
```

From src/aquapose/engine/config.py:
```python
@dataclass(frozen=True)
class DetectionConfig:
    detector_kind: str = "yolo"
    conf_threshold: float = 0.2
    weights_path: str | None = None
    crop_size: list[int] = field(default_factory=lambda: [128, 64])
    detection_batch_frames: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    # NOTE: no iou_threshold field yet — must be added
```

From src/aquapose/engine/pipeline.py (build_stages):
```python
detection_stage = DetectionStage(
    frame_source=frame_source,
    detector_kind=config.detection.detector_kind,
    detection_batch_frames=config.detection.detection_batch_frames,
    weights_path=config.detection.weights_path,
    conf_threshold=config.detection.conf_threshold,
    device=config.device,
)
# NOTE: iou_threshold is NOT passed — must be added
```

From src/aquapose/core/detection/backends/yolo_obb.py:
```python
class YOLOOBBBackend:
    def __init__(self, weights_path, conf_threshold=0.5, iou_threshold=0.45, device="cuda"):
        self._conf = conf_threshold
        self._iou = iou_threshold  # currently passed directly to YOLO predict()

    def detect(self, frame) -> list[Detection]: ...
    def detect_batch(self, frames) -> list[list[Detection]]: ...
    def _parse_results(self, results) -> list[list[Detection]]: ...
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add Shapely dependency and implement geometric polygon NMS</name>
  <files>pyproject.toml, src/aquapose/core/detection/backends/yolo_obb.py, tests/unit/core/detection/test_detection_stage.py</files>
  <behavior>
    - Test: polygon_nms with two overlapping rectangles (IoU > threshold) keeps only the higher-confidence one
    - Test: polygon_nms with two non-overlapping rectangles keeps both
    - Test: polygon_nms with empty list returns empty list
    - Test: polygon_nms with single detection returns it unchanged
    - Test: polygon_nms suppresses correctly with 3+ detections in a chain (A overlaps B overlaps C, A highest conf — B suppressed by A, C may or may not be suppressed depending on overlap with A)
  </behavior>
  <action>
    1. Add `"shapely>=2.0"` to `dependencies` in `pyproject.toml`.

    2. In `yolo_obb.py`, add a module-level helper function `polygon_nms(detections: list[Detection], iou_threshold: float) -> list[Detection]`:
       - Sort detections by confidence descending.
       - For each detection, compute Shapely Polygon from `det.obb_points` (4x2 array).
       - Greedy suppress: for each candidate, check IoU against all kept detections. IoU = intersection.area / union.area. If IoU > iou_threshold with any kept detection, suppress it.
       - Return kept detections list.
       - Import `from shapely.geometry import Polygon` at top of file.

    3. Modify `YOLOOBBBackend.__init__` to store `self._iou` as the geometric NMS threshold (it already does, just rename semantics in the docstring).

    4. Hardcode `iou=0.95` in both `self._model.predict()` calls (in `detect` and `detect_batch`) to effectively disable YOLO's internal probiou NMS.

    5. In `_parse_results`, after building `frame_dets` for each frame, apply `polygon_nms(frame_dets, self._iou)` before appending to `all_detections`.

    6. Write tests for the `polygon_nms` function in `test_detection_stage.py`. Use synthetic Detection objects with known obb_points rectangles to verify NMS behavior.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test -- tests/unit/core/detection/test_detection_stage.py -x</automated>
  </verify>
  <done>
    - `polygon_nms` function exists and is tested with 5+ test cases
    - YOLO predict calls use hardcoded `iou=0.95`
    - `_parse_results` applies geometric NMS per frame
    - Shapely is in pyproject.toml dependencies
    - All existing tests still pass
  </done>
</task>

<task type="auto">
  <name>Task 2: Add iou_threshold to DetectionConfig and wire through build_stages</name>
  <files>src/aquapose/engine/config.py, src/aquapose/engine/pipeline.py</files>
  <action>
    1. In `config.py`, add `iou_threshold: float = 0.45` to `DetectionConfig` (place it after `conf_threshold` for readability). Update the class docstring to document it: "IoU threshold for geometric polygon NMS. Detections with polygon IoU above this value are suppressed in favor of the higher-confidence detection."

    2. In `pipeline.py` `build_stages()`, add `iou_threshold=config.detection.iou_threshold` to the `DetectionStage(...)` constructor call (around line 416-422). This flows through `**detector_kwargs` to `YOLOOBBBackend.__init__`.

    3. Verify the YAML round-trip works: `detection.iou_threshold` in a YAML config file should be parsed into `DetectionConfig.iou_threshold` and flow to the backend. No code change needed for this — the existing `load_config` YAML parsing handles it automatically via `_filter_fields`.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test -- tests/unit/core/detection/test_detection_stage.py tests/unit/engine/ -x</automated>
  </verify>
  <done>
    - `DetectionConfig.iou_threshold` exists with default 0.45
    - `build_stages` passes `iou_threshold` to `DetectionStage`
    - `hatch run check` passes (lint + typecheck)
    - All tests pass
  </done>
</task>

</tasks>

<verification>
- `hatch run test` — all unit tests pass
- `hatch run check` — lint + typecheck clean
- Confirm `polygon_nms` is called in `_parse_results` by reading the code
- Confirm YOLO predict uses `iou=0.95` (not `self._iou`) by reading the code
</verification>

<success_criteria>
- Duplicate OBB detections on same fish are suppressed by exact polygon IoU
- YOLO's internal probiou NMS is effectively disabled (hardcoded 0.95)
- iou_threshold is YAML-configurable via `detection.iou_threshold` (default 0.45)
- Shapely is a declared dependency
- All tests pass, lint + typecheck clean
</success_criteria>

<output>
After completion, create `.planning/quick/23-replace-ultralytics-probiou-nms-with-geo/23-SUMMARY.md`
</output>
