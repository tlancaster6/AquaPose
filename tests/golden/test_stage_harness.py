"""Interface test harness for v1.0 stage output validation.

Each test validates the structural correctness and numerical sanity of the
golden data fixtures produced by scripts/generate_golden_data.py.

These tests serve as the regression safety net for Phase 15 stage migrations.
When stages are ported to the new engine, the same structural assertions will
be applied to the outputs of ported Stage.run() implementations.

All tests are marked @slow and skipped in normal CI runs. Run them with:
    hatch run test-all tests/golden/

Tests skip gracefully when golden data has not yet been generated.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Structural tests — verify data shape, types, and key membership
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_detection_structure(
    golden_detections: list,
    golden_metadata: dict,
) -> None:
    """Verify structural correctness of golden detection data.

    Asserts that detections is a list of per-frame dicts, that camera keys
    match metadata, and that Detection objects have the expected attributes.
    """
    frame_count = golden_metadata["frame_count"]
    camera_ids = golden_metadata["camera_ids"]

    assert isinstance(golden_detections, list), "detections must be a list"
    assert len(golden_detections) == frame_count, (
        f"Expected {frame_count} frames, got {len(golden_detections)}"
    )

    # Every frame must be a dict mapping str -> list
    for fi, frame in enumerate(golden_detections):
        assert isinstance(frame, dict), f"Frame {fi} must be a dict"
        for cam, det_list in frame.items():
            assert isinstance(cam, str), f"Camera key must be str, got {type(cam)}"
            assert isinstance(det_list, list), (
                f"Frame {fi} cam {cam}: value must be a list"
            )

    # Camera IDs in first frame must be a subset of (or equal to) metadata camera_ids
    first_frame_cameras = set(golden_detections[0].keys())
    assert first_frame_cameras.issubset(set(camera_ids)), (
        f"Frame 0 cameras {first_frame_cameras} not a subset of metadata cameras {camera_ids}"
    )

    # Find first camera with detections and validate Detection attributes
    found_detection = False
    for frame in golden_detections:
        for _cam, det_list in frame.items():
            if det_list:
                det = det_list[0]
                assert hasattr(det, "bbox"), "Detection must have a 'bbox' attribute"
                assert hasattr(det, "confidence"), (
                    "Detection must have a 'confidence' attribute"
                )
                assert len(det.bbox) == 4, (
                    f"Detection bbox must have 4 elements, got {len(det.bbox)}"
                )
                found_detection = True
                break
        if found_detection:
            break

    assert found_detection, "No detections found in any frame — expected at least one"


@pytest.mark.slow
def test_segmentation_structure(
    golden_masks: list,
    golden_metadata: dict,
) -> None:
    """Verify structural correctness of golden segmentation data.

    Asserts that masks is a list of per-frame dicts, and that each element
    is a (uint8 ndarray, CropRegion) tuple with expected attributes.
    """
    frame_count = golden_metadata["frame_count"]

    assert isinstance(golden_masks, list), "masks must be a list"
    assert len(golden_masks) == frame_count, (
        f"Expected {frame_count} frames, got {len(golden_masks)}"
    )

    for fi, frame in enumerate(golden_masks):
        assert isinstance(frame, dict), f"Frame {fi} must be a dict"
        for cam, mask_list in frame.items():
            assert isinstance(cam, str), f"Camera key must be str, got {type(cam)}"
            assert isinstance(mask_list, list), (
                f"Frame {fi} cam {cam}: value must be a list"
            )

    # Find first (mask, crop_region) tuple and validate types and attributes
    found_mask = False
    for _fi, frame in enumerate(golden_masks):
        for _cam, mask_list in frame.items():
            if mask_list:
                mask, crop_region = mask_list[0]
                # Mask assertions
                assert isinstance(mask, np.ndarray), "Mask must be a numpy ndarray"
                assert mask.dtype == np.uint8, (
                    f"Mask dtype must be uint8, got {mask.dtype}"
                )
                assert mask.ndim == 2, (
                    f"Mask must have 2 dimensions (H, W), got {mask.ndim}"
                )
                # CropRegion assertions — stored as x1/y1/x2/y2 in pixels
                assert hasattr(crop_region, "x1"), "CropRegion must have 'x1' attribute"
                assert hasattr(crop_region, "y1"), "CropRegion must have 'y1' attribute"
                assert hasattr(crop_region, "x2"), "CropRegion must have 'x2' attribute"
                assert hasattr(crop_region, "y2"), "CropRegion must have 'y2' attribute"
                assert isinstance(crop_region.x1, int), "CropRegion.x1 must be int"
                assert isinstance(crop_region.y1, int), "CropRegion.y1 must be int"
                assert crop_region.x2 > crop_region.x1, (
                    "CropRegion.x2 must be greater than x1"
                )
                assert crop_region.y2 > crop_region.y1, (
                    "CropRegion.y2 must be greater than y1"
                )
                found_mask = True
                break
        if found_mask:
            break

    assert found_mask, "No masks found in any frame — expected at least one"


@pytest.mark.slow
def test_tracking_structure(
    golden_tracks: list,
    golden_metadata: dict,
) -> None:
    """Verify structural correctness of golden tracking data.

    Asserts that tracks is a list of per-frame lists, and that FishTrack
    objects have expected attributes including fish_id and position history.
    """
    frame_count = golden_metadata["frame_count"]

    assert isinstance(golden_tracks, list), "tracks must be a list"
    assert len(golden_tracks) == frame_count, (
        f"Expected {frame_count} frames, got {len(golden_tracks)}"
    )

    for fi, frame_tracks in enumerate(golden_tracks):
        assert isinstance(frame_tracks, list), f"Frame {fi} tracks must be a list"

    # Find first non-empty frame and validate FishTrack attributes
    found_track = False
    for _fi, frame_tracks in enumerate(golden_tracks):
        if frame_tracks:
            track = frame_tracks[0]
            assert hasattr(track, "fish_id"), "FishTrack must have 'fish_id' attribute"
            assert isinstance(track.fish_id, int), (
                f"FishTrack.fish_id must be int, got {type(track.fish_id)}"
            )
            # Tracks store position history in 'positions' deque
            assert hasattr(track, "positions"), (
                "FishTrack must have 'positions' attribute"
            )
            assert len(track.positions) > 0, "FishTrack.positions must be non-empty"
            last_pos = track.positions[-1]
            assert isinstance(last_pos, np.ndarray), (
                "FishTrack position must be a numpy ndarray"
            )
            assert last_pos.shape == (3,), (
                f"FishTrack position must have shape (3,), got {last_pos.shape}"
            )
            found_track = True
            break

    assert found_track, (
        "No tracks found in any frame — expected confirmed tracks by frame ~5"
    )


@pytest.mark.slow
def test_midline_extraction_structure(
    golden_midlines: list,
    golden_metadata: dict,
) -> None:
    """Verify structural correctness of golden midline extraction data.

    Asserts that midline_sets is a list of per-frame dicts (MidlineSet), and
    that each Midline2D has a 'points' attribute with shape (N, 2).
    """
    frame_count = golden_metadata["frame_count"]

    assert isinstance(golden_midlines, list), "midline_sets must be a list"
    assert len(golden_midlines) == frame_count, (
        f"Expected {frame_count} frames, got {len(golden_midlines)}"
    )

    for fi, midline_set in enumerate(golden_midlines):
        # MidlineSet is dict[int, dict[str, Midline2D]]
        assert isinstance(midline_set, dict), f"Frame {fi} MidlineSet must be a dict"

    # Find first non-empty MidlineSet and validate Midline2D attributes
    found_midline = False
    for _fi, midline_set in enumerate(golden_midlines):
        if midline_set:
            fish_id = next(iter(midline_set.keys()))
            cam_midlines = midline_set[fish_id]
            assert isinstance(fish_id, int), (
                f"MidlineSet key (fish_id) must be int, got {type(fish_id)}"
            )
            assert isinstance(cam_midlines, dict), (
                f"Inner MidlineSet value must be dict, got {type(cam_midlines)}"
            )
            cam_id = next(iter(cam_midlines.keys()))
            midline2d = cam_midlines[cam_id]
            assert isinstance(cam_id, str), (
                f"Camera ID key must be str, got {type(cam_id)}"
            )
            assert hasattr(midline2d, "points"), (
                "Midline2D must have 'points' attribute"
            )
            assert isinstance(midline2d.points, np.ndarray), (
                "Midline2D.points must be a numpy ndarray"
            )
            assert midline2d.points.ndim == 2, (
                f"Midline2D.points must be 2D (N, 2), got ndim={midline2d.points.ndim}"
            )
            assert midline2d.points.shape[1] == 2, (
                f"Midline2D.points last dim must be 2 (x, y), got {midline2d.points.shape[1]}"
            )
            found_midline = True
            break

    assert found_midline, (
        "No midlines found in any frame — expected midlines by frame ~4"
    )


@pytest.mark.slow
def test_triangulation_structure(
    golden_triangulation: list,
    golden_metadata: dict,
) -> None:
    """Verify structural correctness of golden triangulation data.

    Asserts that midlines_3d is a list of per-frame dicts mapping int (fish_id)
    to Midline3D objects with control_points of shape (K, 3).
    """
    frame_count = golden_metadata["frame_count"]

    assert isinstance(golden_triangulation, list), "triangulation must be a list"
    assert len(golden_triangulation) == frame_count, (
        f"Expected {frame_count} frames, got {len(golden_triangulation)}"
    )

    for fi, frame_tri in enumerate(golden_triangulation):
        assert isinstance(frame_tri, dict), f"Frame {fi} triangulation must be a dict"

    # Find first non-empty frame and validate Midline3D attributes
    found_3d = False
    for _fi, frame_tri in enumerate(golden_triangulation):
        if frame_tri:
            fish_id = next(iter(frame_tri.keys()))
            midline3d = frame_tri[fish_id]
            assert isinstance(fish_id, int), (
                f"Triangulation key (fish_id) must be int, got {type(fish_id)}"
            )
            assert hasattr(midline3d, "control_points"), (
                "Midline3D must have 'control_points' attribute"
            )
            assert isinstance(midline3d.control_points, np.ndarray), (
                "Midline3D.control_points must be a numpy ndarray"
            )
            assert midline3d.control_points.ndim == 2, (
                f"Midline3D.control_points must be 2D (K, 3), got ndim={midline3d.control_points.ndim}"
            )
            assert midline3d.control_points.shape[1] == 3, (
                f"Midline3D.control_points last dim must be 3 (x, y, z), got shape {midline3d.control_points.shape}"
            )
            # n_cameras records how many views contributed to triangulation
            assert hasattr(midline3d, "n_cameras"), (
                "Midline3D must have 'n_cameras' attribute"
            )
            assert isinstance(midline3d.n_cameras, int), (
                f"Midline3D.n_cameras must be int, got {type(midline3d.n_cameras)}"
            )
            found_3d = True
            break

    assert found_3d, "No 3D midlines found in any frame — expected triangulated results"


# ---------------------------------------------------------------------------
# Numerical stability tests — verify finite values and sane ranges
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_detection_numerical_stability(golden_detections: list) -> None:
    """Verify that golden detections contain valid numerical data.

    All bbox coordinates must be finite, all confidence values must be in
    [0, 1], and at least one frame must have at least one detection.
    """
    found_detection = False

    for fi, frame in enumerate(golden_detections):
        for cam, det_list in frame.items():
            for det in det_list:
                found_detection = True
                # bbox values must be finite
                for coord in det.bbox:
                    assert np.isfinite(float(coord)), (
                        f"Frame {fi} cam {cam}: bbox coord {coord} is not finite"
                    )
                # confidence must be in [0, 1]
                conf = float(det.confidence)
                assert 0.0 <= conf <= 1.0, (
                    f"Frame {fi} cam {cam}: confidence {conf} not in [0, 1]"
                )

    assert found_detection, (
        "No detections found — golden data must have at least one detection"
    )


@pytest.mark.slow
def test_segmentation_numerical_stability(golden_masks: list) -> None:
    """Verify that golden masks contain valid data.

    All mask pixel values must be either 0 or 255 (binary), and at least
    one mask must have non-zero pixels.
    """
    found_nonzero = False

    for fi, frame in enumerate(golden_masks):
        for cam, mask_list in frame.items():
            for mask, _crop in mask_list:
                unique_vals = set(mask.flatten().tolist())
                assert unique_vals.issubset({0, 255}), (
                    f"Frame {fi} cam {cam}: mask has values outside {{0, 255}}: "
                    f"{unique_vals - {0, 255}}"
                )
                if 255 in unique_vals:
                    found_nonzero = True

    assert found_nonzero, (
        "All masks are empty — golden data must have at least one mask with non-zero pixels"
    )


@pytest.mark.slow
def test_triangulation_numerical_stability(golden_triangulation: list) -> None:
    """Verify that golden triangulation contains valid 3D data.

    High-confidence midlines must have finite control points within reasonable
    tank bounds (< 10m). Low-confidence midlines (is_low_confidence=True) are
    allowed to be degenerate — they arise from poor multi-view coverage or RANSAC
    failures and are expected to contain outliers in the v1.0 pipeline.

    At least one high-confidence triangulated midline must exist in the dataset.
    """
    found_3d = False
    found_high_conf = False
    tank_bound_m = 10.0  # 10 metres is an extreme upper bound for any aquarium tank

    for fi, frame_tri in enumerate(golden_triangulation):
        for fish_id, midline3d in frame_tri.items():
            found_3d = True
            pts = midline3d.control_points

            # Low-confidence triangulations may be degenerate (RANSAC failures,
            # poor multi-view coverage) — skip numerical checks for them.
            is_low_conf = getattr(midline3d, "is_low_confidence", False)
            if is_low_conf:
                continue

            # High-confidence midlines must have finite coordinates in tank bounds
            found_high_conf = True
            assert np.all(np.isfinite(pts)), (
                f"Frame {fi} fish {fish_id}: high-confidence control_points contains NaN or Inf"
            )
            max_abs = float(np.max(np.abs(pts)))
            assert max_abs <= tank_bound_m, (
                f"Frame {fi} fish {fish_id}: high-confidence max |coordinate| {max_abs:.3f}m "
                f"exceeds tank bound {tank_bound_m}m"
            )

    assert found_3d, (
        "No 3D midlines found — golden data must have at least one triangulated midline"
    )
    assert found_high_conf, (
        "No high-confidence 3D midlines found — expected some frames to yield "
        "confident triangulations with >= 3 cameras"
    )


# ---------------------------------------------------------------------------
# Metadata completeness test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_metadata_completeness(golden_metadata: dict) -> None:
    """Verify that the golden metadata dict contains all required keys.

    All required environment and generation parameters must be present so
    that tolerance differences across GPUs can be correctly interpreted.
    """
    required_keys = {
        "seed",
        "stop_frame",
        "detector_kind",
        "max_fish",
        "torch_version",
        "numpy_version",
        "camera_ids",
        "frame_count",
        "generation_timestamp",
    }

    missing = required_keys - set(golden_metadata.keys())
    assert not missing, f"metadata.pt is missing required keys: {sorted(missing)}"

    # Type and value assertions
    assert isinstance(golden_metadata["seed"], int), (
        f"metadata['seed'] must be int, got {type(golden_metadata['seed'])}"
    )
    assert golden_metadata["frame_count"] > 0, (
        f"metadata['frame_count'] must be > 0, got {golden_metadata['frame_count']}"
    )
    assert isinstance(golden_metadata["camera_ids"], list), (
        "metadata['camera_ids'] must be a list"
    )
    assert len(golden_metadata["camera_ids"]) > 0, (
        "metadata['camera_ids'] must be non-empty"
    )
    for cam_id in golden_metadata["camera_ids"]:
        assert isinstance(cam_id, str), (
            f"Each camera_id must be str, got {type(cam_id)}: {cam_id!r}"
        )
