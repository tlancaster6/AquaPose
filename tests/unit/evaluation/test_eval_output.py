"""Unit tests for format_eval_report and format_eval_json in evaluation/output.py."""

from __future__ import annotations

import json

import numpy as np

from aquapose.evaluation.stages.association import AssociationMetrics
from aquapose.evaluation.stages.detection import DetectionMetrics
from aquapose.evaluation.stages.midline import MidlineMetrics
from aquapose.evaluation.stages.reconstruction import ReconstructionMetrics
from aquapose.evaluation.stages.tracking import TrackingMetrics

# ---------------------------------------------------------------------------
# Helpers: build synthetic EvalRunnerResult objects
# ---------------------------------------------------------------------------


def _make_detection_metrics() -> DetectionMetrics:
    return DetectionMetrics(
        total_detections=120,
        mean_confidence=0.875,
        std_confidence=0.045,
        mean_jitter=0.5,
        per_camera_counts={"cam0": 60, "cam1": 60},
    )


def _make_tracking_metrics() -> TrackingMetrics:
    return TrackingMetrics(
        track_count=18,
        length_median=4.0,
        length_mean=4.0,
        length_min=4,
        length_max=4,
        coast_frequency=0.1,
        detection_coverage=0.9,
    )


def _make_association_metrics() -> AssociationMetrics:
    return AssociationMetrics(
        fish_yield_ratio=0.8,
        singleton_rate=0.25,
        camera_distribution={1: 10, 2: 30},
        total_fish_observations=40,
        frames_evaluated=5,
    )


def _make_midline_metrics() -> MidlineMetrics:
    return MidlineMetrics(
        mean_confidence=0.92,
        std_confidence=0.08,
        completeness=0.98,
        temporal_smoothness=1.5,
        total_midlines=80,
    )


def _make_reconstruction_metrics() -> ReconstructionMetrics:
    return ReconstructionMetrics(
        mean_reprojection_error=2.5,
        max_reprojection_error=8.0,
        fish_reconstructed=18,
        fish_available=20,
        inlier_ratio=0.9,
        low_confidence_flag_rate=0.1,
        tier2_stability=0.005,
        per_camera_error={
            "cam0": {"mean_px": 2.3, "max_px": 7.0},
            "cam1": {"mean_px": 2.7, "max_px": 8.0},
        },
        per_fish_error={
            0: {"mean_px": 2.4, "max_px": 7.5},
            1: {"mean_px": 2.6, "max_px": 8.0},
        },
    )


def _make_full_result():
    """Build an EvalRunnerResult with all stages present."""
    from aquapose.evaluation.runner import EvalRunnerResult

    return EvalRunnerResult(
        run_id="run_test_001",
        stages_present=frozenset(
            {"detection", "tracking", "association", "midline", "reconstruction"}
        ),
        detection=_make_detection_metrics(),
        tracking=_make_tracking_metrics(),
        association=_make_association_metrics(),
        midline=_make_midline_metrics(),
        reconstruction=_make_reconstruction_metrics(),
        frames_evaluated=5,
        frames_available=10,
    )


def _make_partial_result():
    """Build an EvalRunnerResult with only detection present."""
    from aquapose.evaluation.runner import EvalRunnerResult

    return EvalRunnerResult(
        run_id="run_partial_001",
        stages_present=frozenset({"detection"}),
        detection=_make_detection_metrics(),
        tracking=None,
        association=None,
        midline=None,
        reconstruction=None,
        frames_evaluated=5,
        frames_available=10,
    )


def _make_empty_result():
    """Build an EvalRunnerResult with all-None metrics."""
    from aquapose.evaluation.runner import EvalRunnerResult

    return EvalRunnerResult(
        run_id="run_empty_001",
        stages_present=frozenset(),
        detection=None,
        tracking=None,
        association=None,
        midline=None,
        reconstruction=None,
        frames_evaluated=0,
        frames_available=0,
    )


# ---------------------------------------------------------------------------
# Tests: format_eval_report
# ---------------------------------------------------------------------------


def test_format_eval_report_full_result_contains_header() -> None:
    """format_eval_report with full result contains 'Evaluation Report' header."""
    from aquapose.evaluation.output import format_eval_report

    result = _make_full_result()
    report = format_eval_report(result)

    assert "Evaluation Report" in report
    assert "run_test_001" in report
    assert "5" in report  # frames evaluated
    assert "10" in report  # frames available


def test_format_eval_report_full_result_contains_all_stages() -> None:
    """format_eval_report with full result contains all 5 stage sections."""
    from aquapose.evaluation.output import format_eval_report

    result = _make_full_result()
    report = format_eval_report(result)

    assert "Detection" in report
    assert "Tracking" in report
    assert "Association" in report
    assert "Midline" in report
    assert "Reconstruction" in report


def test_format_eval_report_full_result_contains_metric_values() -> None:
    """format_eval_report includes key metric values in the report."""
    from aquapose.evaluation.output import format_eval_report

    result = _make_full_result()
    report = format_eval_report(result)

    # Detection summary: total detections
    assert "120" in report
    # Reconstruction: mean reprojection error
    assert "2.5" in report or "2.50" in report


def test_format_eval_report_partial_result_only_detection_section() -> None:
    """format_eval_report with detection-only result: only Detection section, no others."""
    from aquapose.evaluation.output import format_eval_report

    result = _make_partial_result()
    report = format_eval_report(result)

    assert "Detection" in report
    # These stage names should not appear (no metrics to show)
    assert "Tracking" not in report
    assert "Association" not in report
    assert "Midline" not in report
    assert "Reconstruction" not in report


def test_format_eval_report_empty_result_header_only() -> None:
    """format_eval_report with empty result (all None): produces header only, no stage sections."""
    from aquapose.evaluation.output import format_eval_report

    result = _make_empty_result()
    report = format_eval_report(result)

    assert "Evaluation Report" in report
    # No stage sections expected
    assert "Detection" not in report
    assert "Tracking" not in report
    assert "Reconstruction" not in report


def test_format_eval_report_returns_string() -> None:
    """format_eval_report returns a string."""
    from aquapose.evaluation.output import format_eval_report

    result = _make_full_result()
    report = format_eval_report(result)

    assert isinstance(report, str)
    assert len(report) > 0


# ---------------------------------------------------------------------------
# Tests: format_eval_json
# ---------------------------------------------------------------------------


def test_format_eval_json_produces_valid_json() -> None:
    """format_eval_json produces a valid JSON string (json.loads succeeds)."""
    from aquapose.evaluation.output import format_eval_json

    result = _make_full_result()
    json_str = format_eval_json(result)

    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)


def test_format_eval_json_schema() -> None:
    """format_eval_json schema has run_id, stages_present, frames_evaluated, frames_available, stages."""
    from aquapose.evaluation.output import format_eval_json

    result = _make_full_result()
    parsed = json.loads(format_eval_json(result))

    assert "run_id" in parsed
    assert parsed["run_id"] == "run_test_001"
    assert "stages_present" in parsed
    assert "frames_evaluated" in parsed
    assert parsed["frames_evaluated"] == 5
    assert "frames_available" in parsed
    assert parsed["frames_available"] == 10
    assert "stages" in parsed
    assert isinstance(parsed["stages"], dict)


def test_format_eval_json_stages_keys() -> None:
    """format_eval_json stages dict contains correct stage keys for present stages."""
    from aquapose.evaluation.output import format_eval_json

    result = _make_full_result()
    parsed = json.loads(format_eval_json(result))

    assert "detection" in parsed["stages"]
    assert "tracking" in parsed["stages"]
    assert "association" in parsed["stages"]
    assert "midline" in parsed["stages"]
    assert "reconstruction" in parsed["stages"]


def test_format_eval_json_partial_result_absent_stages_omitted() -> None:
    """format_eval_json with partial result omits absent stages from 'stages' dict."""
    from aquapose.evaluation.output import format_eval_json

    result = _make_partial_result()
    parsed = json.loads(format_eval_json(result))

    assert "detection" in parsed["stages"]
    assert "tracking" not in parsed["stages"]
    assert "association" not in parsed["stages"]
    assert "midline" not in parsed["stages"]
    assert "reconstruction" not in parsed["stages"]


def test_format_eval_json_numpy_safe_encoding() -> None:
    """format_eval_json handles numpy scalar values without error."""
    from aquapose.evaluation.output import format_eval_json
    from aquapose.evaluation.runner import EvalRunnerResult

    # Build result with numpy float values in DetectionMetrics
    det = DetectionMetrics(
        total_detections=int(np.int64(50)),
        mean_confidence=float(np.float64(0.9)),
        std_confidence=float(np.float32(0.05)),
        mean_jitter=float(np.float64(0.2)),
        per_camera_counts={"cam0": int(np.int32(50))},
    )
    result = EvalRunnerResult(
        run_id="run_numpy_test",
        stages_present=frozenset({"detection"}),
        detection=det,
        tracking=None,
        association=None,
        midline=None,
        reconstruction=None,
        frames_evaluated=5,
        frames_available=10,
    )

    # Should not raise TypeError for numpy scalars
    json_str = format_eval_json(result)
    parsed = json.loads(json_str)
    assert parsed["run_id"] == "run_numpy_test"


def test_format_eval_json_empty_result() -> None:
    """format_eval_json with empty result still produces valid JSON with empty stages dict."""
    from aquapose.evaluation.output import format_eval_json

    result = _make_empty_result()
    parsed = json.loads(format_eval_json(result))

    assert parsed["run_id"] == "run_empty_001"
    assert parsed["stages"] == {}
    assert parsed["stages_present"] == []
