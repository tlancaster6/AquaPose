"""AquaPose pipeline engine.

This package provides the architectural skeleton for the AquaPose pipeline:
Stage Protocol, PipelineContext, event system, observer base, config
hierarchy, and pipeline orchestrator.

Import boundary (ENG-07): engine/ may import from computation modules, but
computation modules must NEVER import from engine/. This is enforced at all
levels with no TYPE_CHECKING exceptions.
"""

from aquapose.engine.config import (
    AssociationConfig,
    DetectionConfig,
    MidlineConfig,
    PipelineConfig,
    ReconstructionConfig,
    TrackingConfig,
    load_config,
    serialize_config,
)
from aquapose.engine.diagnostic_observer import DiagnosticObserver, StageSnapshot
from aquapose.engine.events import (
    Event,
    FrameProcessed,
    PipelineComplete,
    PipelineFailed,
    PipelineStart,
    StageComplete,
    StageStart,
)
from aquapose.engine.hdf5_observer import HDF5ExportObserver
from aquapose.engine.observers import EventBus, Observer
from aquapose.engine.pipeline import PosePipeline
from aquapose.engine.stages import PipelineContext, Stage
from aquapose.engine.timing import TimingObserver

__all__ = [
    "AssociationConfig",
    "DetectionConfig",
    "DiagnosticObserver",
    "Event",
    "EventBus",
    "FrameProcessed",
    "HDF5ExportObserver",
    "MidlineConfig",
    "Observer",
    "PipelineComplete",
    "PipelineConfig",
    "PipelineContext",
    "PipelineFailed",
    "PipelineStart",
    "PosePipeline",
    "ReconstructionConfig",
    "Stage",
    "StageComplete",
    "StageSnapshot",
    "StageStart",
    "TimingObserver",
    "TrackingConfig",
    "load_config",
    "serialize_config",
]
