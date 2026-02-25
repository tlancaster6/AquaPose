"""Engine package — Stage Protocol, PipelineContext, and pipeline orchestration.

This package defines the foundational contracts for the v2.0 engine:

- :class:`Stage` — structural protocol every pipeline stage must satisfy
- :class:`PipelineContext` — typed dataclass accumulating inter-stage results

Import boundary (ENG-07): this package imports only stdlib and computation
modules. Computation modules (calibration, segmentation, etc.) must NOT
import from engine/.
"""

from aquapose.engine.stages import PipelineContext, Stage

__all__ = ["PipelineContext", "Stage"]
