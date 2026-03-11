"""AquaPose pipeline engine.

This package provides the architectural skeleton for the AquaPose pipeline:
Stage Protocol, PipelineContext, event system, observer base, config
hierarchy, and pipeline orchestrator.

Import boundary (ENG-07): engine/ may import from computation modules, but
computation modules must NEVER import from engine/. This is enforced at all
levels with no TYPE_CHECKING exceptions.
"""

from aquapose.core.context import PipelineContext, Stage, load_chunk_cache
from aquapose.engine.config import (
    AssociationConfig,
    DetectionConfig,
    PipelineConfig,
    PoseConfig,
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
from aquapose.engine.observer_factory import build_observers
from aquapose.engine.observers import EventBus, Observer
from aquapose.engine.orchestrator import ChunkHandoff, ChunkOrchestrator, write_handoff
from aquapose.engine.pipeline import PosePipeline
from aquapose.engine.timing import TimingObserver

__all__ = [
    "AssociationConfig",
    "ChunkHandoff",
    "ChunkOrchestrator",
    "ConsoleObserver",
    "DetectionConfig",
    "DiagnosticObserver",
    "Event",
    "EventBus",
    "FrameProcessed",
    "Observer",
    "PipelineComplete",
    "PipelineConfig",
    "PipelineContext",
    "PipelineFailed",
    "PipelineStart",
    "PoseConfig",
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
    "load_chunk_cache",
    "load_config",
    "serialize_config",
    "write_handoff",
]
