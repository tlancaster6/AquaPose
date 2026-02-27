"""Per-stage numerical regression tests against v1.0 golden reference data.

DEFERRED (EVAL-01): The v2.1 pipeline reorder (Detection -> 2D Tracking ->
Association -> Midline -> Reconstruction) invalidates these tests. The old
golden data was generated with v1.0 tracking-driven reconstruction. New
regression tests will be written post-v2.1 stabilization.

These tests are retained as templates. All tests are skipped with EVAL-01 note.

Stage mapping from v1.0 golden data to new PipelineContext fields:
- golden_detection.pt       -> context.detections          (Stage 1)
- golden_midline_extraction.pt -> context.annotated_detections (Stage 4, partial)
- golden_tracking.pt        -> context.tracks_2d            (Stage 2, new format)
- golden_triangulation.pt   -> context.midlines_3d          (Stage 5)

All tests are marked @pytest.mark.regression and @pytest.mark.slow so they are
excluded from the fast test loop. Run with:
    hatch run test-regression
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.regression.conftest import DET_ATOL, MID_ATOL, RECON_ATOL, TRK_ATOL

pytestmark = pytest.mark.skip(
    reason=(
        "Regression tests deferred — pipeline reorder invalidates existing tests (EVAL-01). "
        "Golden data generated with v1.0 tracking-driven reconstruction is incompatible "
        "with the v2.1 pipeline order. Tests retained as templates for post-v2.1 rebuild."
    )
)

# ---------------------------------------------------------------------------
# Detection regression (Stage 1)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.slow
def test_detection_regression(
    pipeline_context: object,
    golden_detections: list,
) -> None:
    """Compare context.detections against golden_detection.pt.

    For each frame and camera, asserts that:
    - Detection count matches.
    - Bbox coordinates match within DET_ATOL.
    - Confidence values match within DET_ATOL.

    Args:
        pipeline_context: Session-scoped PipelineContext from the full pipeline run.
        golden_detections: Golden detection data loaded from golden_detection.pt.
    """
    new_detections: list = pipeline_context.detections  # type: ignore[attr-defined]
    assert new_detections is not None, (
        "context.detections is None — Stage 1 did not run"
    )

    n_frames = min(len(new_detections), len(golden_detections))
    assert n_frames > 0, "No frames to compare"

    total_compared = 0
    max_deviation = 0.0

    for fi in range(n_frames):
        new_frame = new_detections[fi]
        gold_frame = golden_detections[fi]

        for cam_id in gold_frame:
            if cam_id not in new_frame:
                # Camera present in golden but absent in new output — allow if
                # both have zero detections worth of data.
                gold_dets = gold_frame[cam_id]
                if gold_dets:
                    pytest.fail(
                        f"Frame {fi}: camera '{cam_id}' present in golden data "
                        f"with {len(gold_dets)} detections but missing from new output"
                    )
                continue

            gold_dets = gold_frame[cam_id]
            new_dets = new_frame[cam_id]

            assert len(new_dets) == len(gold_dets), (
                f"Frame {fi} cam '{cam_id}': detection count mismatch — "
                f"golden={len(gold_dets)} new={len(new_dets)}"
            )

            for di, (gold_det, new_det) in enumerate(
                zip(gold_dets, new_dets, strict=True)
            ):
                # Compare bbox
                gold_bbox = np.array(gold_det.bbox, dtype=float)
                new_bbox = np.array(new_det.bbox, dtype=float)
                bbox_diff = float(np.max(np.abs(new_bbox - gold_bbox)))
                max_deviation = max(max_deviation, bbox_diff)
                assert np.allclose(new_bbox, gold_bbox, atol=DET_ATOL), (
                    f"Frame {fi} cam '{cam_id}' det {di}: bbox mismatch — "
                    f"golden={gold_bbox} new={new_bbox} max_diff={bbox_diff:.2e}"
                )
                # Compare confidence
                gold_conf = float(gold_det.confidence)
                new_conf = float(new_det.confidence)
                conf_diff = abs(new_conf - gold_conf)
                max_deviation = max(max_deviation, conf_diff)
                assert abs(new_conf - gold_conf) <= DET_ATOL, (
                    f"Frame {fi} cam '{cam_id}' det {di}: confidence mismatch — "
                    f"golden={gold_conf:.6f} new={new_conf:.6f} diff={conf_diff:.2e}"
                )
                total_compared += 1

    assert total_compared > 0, (
        "No detections were compared — golden data and new output both empty"
    )


# ---------------------------------------------------------------------------
# Midline regression (Stage 2) — xfail due to structural divergence
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "v1.0 golden midlines are keyed by fish_id (post-tracking), but new Stage 2 "
        "produces annotated_detections before tracking (fish_id=-1 placeholder). "
        "Structural divergence makes direct array comparison unreliable. "
        "Resolve when golden data is regenerated with PosePipeline."
    ),
    strict=False,
)
def test_midline_regression(
    pipeline_context: object,
    golden_midlines: list,
) -> None:
    """Compare Stage 2 midline outputs against golden_midline_extraction.pt.

    The v1.0 golden midlines are MidlineSet objects keyed by fish_id, where
    fish_ids are assigned by the v1.0 tracker. The new pipeline's Stage 2
    (MidlineStage) extracts midlines for ALL detections before tracking, so
    fish_id is a placeholder (-1) at this stage.

    This test attempts a camera-level comparison of midline point arrays
    across frames. Because the ordering may differ, it compares sorted
    point-array norms as a proxy for equivalence. If the structures diverge
    irreconcilably, the test is marked xfail.

    Args:
        pipeline_context: Session-scoped PipelineContext.
        golden_midlines: Golden midline data from golden_midline_extraction.pt.
    """
    annotated = pipeline_context.annotated_detections  # type: ignore[attr-defined]
    assert annotated is not None, (
        "context.annotated_detections is None — Stage 2 did not run"
    )

    n_frames = min(len(annotated), len(golden_midlines))
    total_compared = 0
    max_deviation = 0.0

    for fi in range(n_frames):
        new_frame = annotated[fi]
        gold_midline_set = golden_midlines[fi]  # dict[int, dict[str, Midline2D]]

        # Collect all midline points from new stage output
        new_midline_points_by_cam: dict[str, list] = {}
        for cam_id, det_list in new_frame.items():
            cam_pts = []
            for det in det_list:
                midline = getattr(det, "midline", None)
                if midline is not None and hasattr(midline, "points"):
                    cam_pts.append(midline.points)
            if cam_pts:
                new_midline_points_by_cam[cam_id] = cam_pts

        # Collect all midline points from golden data
        gold_midline_points_by_cam: dict[str, list] = {}
        for _fish_id, cam_midlines in gold_midline_set.items():
            for cam_id, midline2d in cam_midlines.items():
                gold_midline_points_by_cam.setdefault(cam_id, []).append(
                    midline2d.points
                )

        # Compare per camera: sort by L2 norm of flattened points
        for cam_id in gold_midline_points_by_cam:
            if cam_id not in new_midline_points_by_cam:
                continue
            gold_pts_list = sorted(
                gold_midline_points_by_cam[cam_id],
                key=lambda p: float(np.linalg.norm(p)),
            )
            new_pts_list = sorted(
                new_midline_points_by_cam[cam_id],
                key=lambda p: float(np.linalg.norm(p)),
            )
            n_compare = min(len(gold_pts_list), len(new_pts_list))
            for gi in range(n_compare):
                gold_arr = gold_pts_list[gi]
                new_arr = new_pts_list[gi]
                if gold_arr.shape != new_arr.shape:
                    continue
                diff = float(np.max(np.abs(new_arr - gold_arr)))
                max_deviation = max(max_deviation, diff)
                assert np.allclose(new_arr, gold_arr, atol=MID_ATOL), (
                    f"Frame {fi} cam '{cam_id}': midline points mismatch "
                    f"max_diff={diff:.2e} (atol={MID_ATOL})"
                )
                total_compared += 1


# ---------------------------------------------------------------------------
# Tracking regression (Stage 4)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.slow
def test_tracking_regression(
    pipeline_context: object,
    golden_tracks: list,
) -> None:
    """Compare context.tracks against golden_tracking.pt.

    For each frame, asserts that:
    - The set of confirmed fish_ids matches (or is a superset — new tracker
      may emit more tracks in edge frames).
    - For each matching fish_id, positions[-1] matches within TRK_ATOL.

    Args:
        pipeline_context: Session-scoped PipelineContext.
        golden_tracks: Golden tracking data from golden_tracking.pt.
    """
    new_tracks: list = pipeline_context.tracks  # type: ignore[attr-defined]
    assert new_tracks is not None, "context.tracks is None — Stage 4 did not run"

    n_frames = min(len(new_tracks), len(golden_tracks))
    total_compared = 0
    max_deviation = 0.0

    for fi in range(n_frames):
        new_frame_tracks = new_tracks[fi]
        gold_frame_tracks = golden_tracks[fi]

        new_by_id = {t.fish_id: t for t in new_frame_tracks}
        gold_by_id = {t.fish_id: t for t in gold_frame_tracks}

        # Every fish_id in golden must appear in new output
        for fish_id in gold_by_id:
            assert fish_id in new_by_id, (
                f"Frame {fi}: fish_id {fish_id} present in golden tracks "
                f"but missing from new output. "
                f"Golden ids: {sorted(gold_by_id)} New ids: {sorted(new_by_id)}"
            )

            gold_pos = np.array(gold_by_id[fish_id].positions[-1], dtype=float)
            new_pos = np.array(new_by_id[fish_id].positions[-1], dtype=float)
            diff = float(np.max(np.abs(new_pos - gold_pos)))
            max_deviation = max(max_deviation, diff)
            assert np.allclose(new_pos, gold_pos, atol=TRK_ATOL), (
                f"Frame {fi} fish {fish_id}: position mismatch — "
                f"golden={gold_pos} new={new_pos} max_diff={diff:.2e}"
            )
            total_compared += 1

    assert total_compared > 0, (
        "No tracks were compared — golden data and new output both empty for all frames"
    )


# ---------------------------------------------------------------------------
# Reconstruction regression (Stage 5)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.slow
def test_reconstruction_regression(
    pipeline_context: object,
    golden_triangulation: list,
) -> None:
    """Compare context.midlines_3d against golden_triangulation.pt.

    For each frame and fish_id present in the golden data, asserts that the
    new pipeline produces a 3D midline with control_points matching within
    RECON_ATOL (1e-3, sub-millimeter).

    Args:
        pipeline_context: Session-scoped PipelineContext.
        golden_triangulation: Golden triangulation data from golden_triangulation.pt.
    """
    new_midlines_3d: list = pipeline_context.midlines_3d  # type: ignore[attr-defined]
    assert new_midlines_3d is not None, (
        "context.midlines_3d is None — Stage 5 did not run"
    )

    n_frames = min(len(new_midlines_3d), len(golden_triangulation))
    total_compared = 0
    max_deviation = 0.0

    for fi in range(n_frames):
        new_frame = new_midlines_3d[fi]
        gold_frame = golden_triangulation[fi]

        for fish_id, gold_m3d in gold_frame.items():
            assert fish_id in new_frame, (
                f"Frame {fi}: fish_id {fish_id} present in golden triangulation "
                f"but missing from new output. "
                f"Golden ids: {sorted(gold_frame)} New ids: {sorted(new_frame)}"
            )

            gold_pts = np.array(gold_m3d.control_points, dtype=float)
            new_m3d = new_frame[fish_id]
            new_pts = np.array(new_m3d.control_points, dtype=float)

            # Skip NaN control points (degenerate triangulation)
            if np.any(np.isnan(gold_pts)) or np.any(np.isnan(new_pts)):
                total_compared += 1
                continue

            assert gold_pts.shape == new_pts.shape, (
                f"Frame {fi} fish {fish_id}: control_points shape mismatch — "
                f"golden={gold_pts.shape} new={new_pts.shape}"
            )

            diff = float(np.max(np.abs(new_pts - gold_pts)))
            max_deviation = max(max_deviation, diff)
            assert np.allclose(new_pts, gold_pts, atol=RECON_ATOL), (
                f"Frame {fi} fish {fish_id}: control_points mismatch — "
                f"max_diff={diff:.2e} (atol={RECON_ATOL})"
            )
            total_compared += 1

    assert total_compared > 0, (
        "No 3D midlines were compared — golden triangulation and new output both empty"
    )
