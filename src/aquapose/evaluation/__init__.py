"""Offline evaluation harness for reconstruction quality metrics.

Provides run_evaluation, EvalResults, select_frames, flag_outliers,
format_baseline_report, and related types for loading a self-contained
MidlineFixture and computing Tier 1 and Tier 2 reconstruction metrics
without running the full pipeline.
"""

from aquapose.evaluation.harness import EvalResults, generate_fixture, run_evaluation
from aquapose.evaluation.metrics import Tier1Result, Tier2Result, select_frames
from aquapose.evaluation.output import flag_outliers, format_baseline_report

__all__ = [
    "EvalResults",
    "Tier1Result",
    "Tier2Result",
    "flag_outliers",
    "format_baseline_report",
    "generate_fixture",
    "run_evaluation",
    "select_frames",
]
