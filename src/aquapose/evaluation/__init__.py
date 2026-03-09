"""Offline evaluation and tuning for multi-stage pipeline quality metrics.

Provides select_frames, flag_outliers, format_baseline_report, format_eval_report,
format_eval_json, EvalRunner, EvalRunnerResult, and related types for loading
evaluation data and computing multi-stage quality metrics without running the
full pipeline.

Also exposes the viz sub-package for post-run visualization generation.
"""

from aquapose.evaluation.compare import (
    load_eval_results,
    write_comparison_json,
)
from aquapose.evaluation.metrics import Tier1Result, Tier2Result, select_frames
from aquapose.evaluation.output import (
    flag_outliers,
    format_baseline_report,
    format_eval_json,
    format_eval_report,
)
from aquapose.evaluation.runner import EvalRunner, EvalRunnerResult, load_run_context
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
from aquapose.evaluation.tuning import (
    TuningOrchestrator,
    TuningResult,
    format_comparison_table,
    format_config_diff,
    format_yield_matrix,
)
from aquapose.evaluation.viz import (
    generate_all,
    generate_animation,
    generate_overlay,
    generate_trails,
)

__all__ = [
    "ASSOCIATION_DEFAULT_GRID",
    "RECONSTRUCTION_DEFAULT_GRID",
    "AssociationMetrics",
    "DetectionMetrics",
    "EvalRunner",
    "EvalRunnerResult",
    "MidlineMetrics",
    "ReconstructionMetrics",
    "Tier1Result",
    "Tier2Result",
    "TrackingMetrics",
    "TuningOrchestrator",
    "TuningResult",
    "evaluate_association",
    "evaluate_detection",
    "evaluate_midline",
    "evaluate_reconstruction",
    "evaluate_tracking",
    "flag_outliers",
    "format_baseline_report",
    "format_comparison_table",
    "format_config_diff",
    "format_eval_json",
    "format_eval_report",
    "format_yield_matrix",
    "generate_all",
    "generate_animation",
    "generate_overlay",
    "generate_trails",
    "load_eval_results",
    "load_run_context",
    "select_frames",
    "write_comparison_json",
]
