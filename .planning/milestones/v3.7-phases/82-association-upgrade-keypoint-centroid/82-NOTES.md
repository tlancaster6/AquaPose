# Phase 82 Notes: Keypoint Centroid Selection

## Selected Keypoint

**spine1 (index 2)** from the 6-point model `["nose", "head", "spine1", "spine2", "spine3", "tail"]`.

Configured via `AssociationConfig.centroid_keypoint_index = 2` (YAML-tunable).

## Why spine1?

spine1 is the mid-body anatomical point located roughly equidistant from the fish's head and tail. It is the most geometrically stable keypoint under the two most common failure modes:

1. **Frame-edge clipping**: When a fish is partially outside the frame, the nose and tail are the first keypoints to disappear. The body center (spine1/spine2) remains visible even when ~25-30% of the fish is off-screen.

2. **Partial occlusion**: When another fish overlaps, extremity keypoints (nose, tail) are the first to be occluded. The anatomical midpoint stays visible as long as the fish body center is unoccluded.

In contrast, the OBB centroid drifts significantly when the bounding box is clipped at frame edges — a clipped OBB is smaller and its geometric center shifts toward the visible portion of the fish. This drift directly inflates ray-ray distances during cross-view association and degrades clustering quality.

## Confidence Behavior

The confidence floor is set to `centroid_confidence_floor = 0.3` (matching the pose backend default `keypoint_confidence_floor`).

Per Phase 78.1 production model evaluation (mAP50-95=0.974), interior keypoints (spine1, spine2) consistently have higher confidence than extremity keypoints (nose, tail) because:
- Interior body points are visible from more viewpoints simultaneously
- They are less frequently occluded by tank walls or other fish
- The pose model has more training examples with visible interior keypoints

This means spine1 will satisfy the confidence floor in the vast majority of detected frames, with fallback only during genuine occlusion events.

## Fallback Behavior

When spine1 confidence is below 0.3, or keypoints are unavailable (e.g., detection came from a frame before the pose stage ran, or detection has `keypoints=None`), the OBB geometric center is used. Coasted frames (Kalman predictions without a matched detection) always use the OBB centroid since no keypoint is available.

The fallback is silent and transparent to downstream association — `Tracklet2D.centroids` type is `tuple[tuple[float, float], ...]` regardless of source.

## Configuration

Both parameters are YAML-tunable via the `association:` section:

```yaml
association:
  centroid_keypoint_index: 2   # 0=nose, 1=head, 2=spine1, 3=spine2, 4=spine3, 5=tail
  centroid_confidence_floor: 0.3
```

To use the nose keypoint instead of spine1 (e.g., for head-tracking experiments):
```yaml
association:
  centroid_keypoint_index: 0
```
