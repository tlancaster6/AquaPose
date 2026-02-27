"""Legacy tracking package stub — v2.1 tracking lives in aquapose.core.tracking.

In v2.1, per-camera 2D tracking is performed by the TrackingStage in
aquapose.core.tracking. The cross-camera association lives in
aquapose.core.association. FishTrack and TrackState have moved to
aquapose.core.tracking.types.

This package is retained as an empty stub to avoid import errors in any
code that imports ``aquapose.tracking`` as a package (not specific symbols).
"""

__all__: list = []
