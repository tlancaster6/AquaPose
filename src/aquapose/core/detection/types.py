"""Stage-specific types for the Detection stage (Stage 1).

Re-exports the canonical Detection dataclass from segmentation.detector so
downstream code can import from a single location within the core package.
"""

from __future__ import annotations

from aquapose.segmentation.detector import Detection

__all__ = ["Detection"]
