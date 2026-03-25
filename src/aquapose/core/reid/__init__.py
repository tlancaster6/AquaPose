"""Fish re-identification via appearance embeddings.

Provides a MegaDescriptor-T backbone wrapper for extracting L2-normalized
embedding vectors from fish crops, plus a batch runner for processing
completed pipeline runs, zero-shot evaluation utilities, a training
data miner for extracting grouped crop datasets, and appearance-based
swap detection and repair.
"""

from .embedder import FishEmbedder
from .eval import compute_reid_metrics, print_reid_report
from .miner import MinerConfig, TrainingDataMiner
from .runner import EmbedRunner
from .swap_detector import ReidEvent, SwapDetector, SwapDetectorConfig

__all__ = [
    "EmbedRunner",
    "FishEmbedder",
    "MinerConfig",
    "ReidEvent",
    "SwapDetector",
    "SwapDetectorConfig",
    "TrainingDataMiner",
    "compute_reid_metrics",
    "print_reid_report",
]
