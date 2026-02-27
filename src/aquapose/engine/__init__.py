"""AquaPose pipeline engine.

This package provides the architectural skeleton for the AquaPose pipeline:
Stage Protocol, PipelineContext, event system, observer base, config
hierarchy, and pipeline orchestrator.

Import boundary (ENG-07): engine/ may import from computation modules, but
computation modules must NEVER import from engine/. This is enforced at all
levels with no TYPE_CHECKING exceptions.
"""

from aquapose.core.context import PipelineContext, Stage
from aquapose.engine.animation_observer import Animation3DObserver
from aquapose.engine.config import (
    AssociationConfig,
    DetectionConfig,
    MidlineConfig,
    PipelineConfig,
    ReconstructionConfig,
    SyntheticConfig,
    TrackingConfig,
    load_config,
    serialize_config,
)
from aquapose.engine.console_observer import ConsoleObserver
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
from aquapose.engine.observer_factory import build_observers
from aquapose.engine.observers import EventBus, Observer
from aquapose.engine.overlay_observer import Overlay2DObserver
from aquapose.engine.pipeline import PosePipeline
from aquapose.engine.timing import TimingObserver

__all__ = [
    "Animation3DObserver",
    "AssociationConfig",
    "ConsoleObserver",
    "DetectionConfig",
    "DiagnosticObserver",
    "Event",
    "EventBus",
    "FrameProcessed",
    "HDF5ExportObserver",
    "MidlineConfig",
    "Observer",
    "Overlay2DObserver",
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
    "SyntheticConfig",
    "TimingObserver",
    "TrackingConfig",
    "build_observers",
    "load_config",
    "serialize_config",
]
