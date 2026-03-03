"""Offline evaluation harness for reconstruction quality metrics.

Provides run_evaluation, EvalResults, select_frames, flag_outliers,
format_baseline_report, and related types for loading a self-contained
MidlineFixture and computing Tier 1 and Tier 2 reconstruction metrics
without running the full pipeline.
"""

from aquapose.evaluation.harness import EvalResults, generate_fixture, run_evaluation
from aquapose.evaluation.metrics import Tier1Result, Tier2Result, select_frames
from aquapose.evaluation.output import flag_outliers, format_baseline_report
from aquapose.evaluation.stages import (
    ASSOCIATION_DEFAULT_GRID,
    RECONSTRUCTION_DEFAULT_GRID,
    AssociationMetrics,
    DetectionMetrics,
    MidlineMetrics,
    ReconstructionMetrics,
    TrackingMetrics,
    evaluate_association,
    evaluate_detection,
    evaluate_midline,
    evaluate_reconstruction,
    evaluate_tracking,
)

__all__ = [
    "ASSOCIATION_DEFAULT_GRID",
    "RECONSTRUCTION_DEFAULT_GRID",
    "AssociationMetrics",
    "DetectionMetrics",
    "EvalResults",
    "MidlineMetrics",
    "ReconstructionMetrics",
    "Tier1Result",
    "Tier2Result",
    "TrackingMetrics",
    "evaluate_association",
    "evaluate_detection",
    "evaluate_midline",
    "evaluate_reconstruction",
    "evaluate_tracking",
    "flag_outliers",
    "format_baseline_report",
    "generate_fixture",
    "run_evaluation",
    "select_frames",
]
