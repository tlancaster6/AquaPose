"""Offline evaluation harness for reconstruction quality metrics.

Provides run_evaluation, EvalResults, select_frames, and related types for
loading a self-contained MidlineFixture and computing Tier 1 and Tier 2
reconstruction metrics without running the full pipeline.
"""

from aquapose.evaluation.harness import EvalResults, run_evaluation
from aquapose.evaluation.metrics import Tier1Result, Tier2Result, select_frames

__all__ = [
    "EvalResults",
    "Tier1Result",
    "Tier2Result",
    "run_evaluation",
    "select_frames",
]
