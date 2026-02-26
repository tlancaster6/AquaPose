"""Importability check for the v1.0 pipeline.stages module.

The v1.0 batch functions (run_tracking, run_triangulation, etc.) are still
present in aquapose.pipeline.stages for backward compatibility with the v1.0
orchestrator. Functional testing of the canonical execution path is covered
by the regression suite in tests/regression/.

This file verifies that the module remains importable and that the public
symbols are present — nothing more.
"""

from __future__ import annotations

import aquapose.pipeline.stages as _stages


def test_pipeline_stages_importable() -> None:
    """aquapose.pipeline.stages is importable and exposes v1.0 public symbols."""
    assert callable(_stages.run_detection)
    assert callable(_stages.run_segmentation)
    assert callable(_stages.run_tracking)
    assert callable(_stages.run_midline_extraction)
    assert callable(_stages.run_triangulation)
