"""AquaPose pipeline engine.

This package provides the architectural skeleton for the AquaPose pipeline:
Stage Protocol, PipelineContext, event system, observer base, config
hierarchy, and pipeline orchestrator.

Import boundary (ENG-07): engine/ may import from computation modules, but
computation modules must NEVER import from engine/. This is enforced at all
levels with no TYPE_CHECKING exceptions.
"""

from aquapose.engine.config import (
    DetectionConfig,
    PipelineConfig,
    SegmentationConfig,
    TrackingConfig,
    TriangulationConfig,
    load_config,
    serialize_config,
)
from aquapose.engine.events import (
    Event,
    FrameProcessed,
    PipelineComplete,
    PipelineFailed,
    PipelineStart,
    StageComplete,
    StageStart,
)
from aquapose.engine.observers import EventBus, Observer
from aquapose.engine.stages import PipelineContext, Stage

__all__ = [
    "DetectionConfig",
    "Event",
    "EventBus",
    "FrameProcessed",
    "Observer",
    "PipelineComplete",
    "PipelineConfig",
    "PipelineContext",
    "PipelineFailed",
    "PipelineStart",
    "SegmentationConfig",
    "Stage",
    "StageComplete",
    "StageStart",
    "TrackingConfig",
    "TriangulationConfig",
    "load_config",
    "serialize_config",
]
