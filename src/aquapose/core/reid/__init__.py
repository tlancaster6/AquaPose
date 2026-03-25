"""Fish re-identification via appearance embeddings.

Provides a MegaDescriptor-T backbone wrapper for extracting L2-normalized
embedding vectors from fish crops, plus a batch runner for processing
completed pipeline runs and zero-shot evaluation utilities.
"""

from .embedder import FishEmbedder

__all__ = ["FishEmbedder"]
