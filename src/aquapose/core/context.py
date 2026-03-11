"""Stage Protocol and PipelineContext — core data contracts for the pipeline.

Defines the structural typing contract that all pipeline stages must satisfy,
and the typed accumulator that flows data between stages. Also defines
ChunkHandoff for persisting cross-chunk state (tracker state and identity map)
across chunk boundaries. Provides StaleCacheError, load_stage_cache,
load_chunk_cache, and context_fingerprint for working with stage-output pickle
caches written by DiagnosticObserver.

These types live in core/ because they are pure data containers with no engine
logic. Placing them here eliminates all TYPE_CHECKING backdoors where core/
stage files previously needed to import from engine/.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class StaleCacheError(Exception):
    """Raised when a stage cache cannot be deserialized due to class evolution."""


def context_fingerprint(ctx: PipelineContext) -> str:
    """Return a stable hash of PipelineContext field names.

    The hash changes whenever the dataclass structure changes (fields added,
    renamed, or removed), making it suitable as a version fingerprint in cache
    envelopes written by DiagnosticObserver.

    Args:
        ctx: Any PipelineContext instance (only field names are hashed, not values).

    Returns:
        12-character hex string derived from SHA-256 of sorted field names.

    """
    names = sorted(f.name for f in dataclasses.fields(ctx))
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:12]


def load_stage_cache(path: str | Path) -> PipelineContext:
    """Load a stage cache pickle file and return the embedded PipelineContext.

    Args:
        path: Path to a *_cache.pkl file written by DiagnosticObserver.

    Returns:
        The deserialized PipelineContext.

    Raises:
        StaleCacheError: If deserialization fails (AttributeError,
            ModuleNotFoundError, pickle.UnpicklingError) or the envelope
            format is invalid or basic shape validation fails.
        FileNotFoundError: If path does not exist.

    """
    p = Path(path)
    raw = p.read_bytes()
    try:
        envelope = pickle.loads(raw)
    except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError) as exc:
        raise StaleCacheError(
            f"Cache file '{p}' is incompatible with the current codebase. "
            f"Re-run the pipeline in diagnostic mode to regenerate it. "
            f"Original error: {exc}"
        ) from exc

    if not isinstance(envelope, dict) or "context" not in envelope:
        raise StaleCacheError(
            f"Cache file '{p}' does not contain a valid envelope dict with a "
            f"'context' key. The cache predates the envelope format — "
            f"re-run the pipeline in diagnostic mode to regenerate it."
        )

    ctx: PipelineContext = envelope["context"]

    if (
        ctx.frame_count is not None
        and ctx.detections is not None
        and ctx.frame_count != len(ctx.detections)
    ):
        raise StaleCacheError(
            f"Cache file '{p}' has inconsistent shape: "
            f"frame_count={ctx.frame_count} but len(detections)={len(ctx.detections)}. "
            f"The cache may be corrupt — re-run the pipeline in diagnostic mode."
        )

    return ctx


def load_chunk_cache(path: str | Path) -> PipelineContext:
    """Load a chunk cache pickle file and return the embedded PipelineContext.

    Loads from the new per-chunk single-cache layout written by DiagnosticObserver
    (``diagnostics/chunk_NNN/cache.pkl``). On version fingerprint mismatch, logs a
    warning but returns the context anyway (no StaleCacheError for fingerprint issues).

    Args:
        path: Path to a ``cache.pkl`` file written by the new DiagnosticObserver.

    Returns:
        The deserialized PipelineContext.

    Raises:
        StaleCacheError: If deserialization fails or the envelope format is invalid.
        FileNotFoundError: If path does not exist.

    """
    p = Path(path)
    raw = p.read_bytes()
    try:
        envelope = pickle.loads(raw)
    except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError) as exc:
        raise StaleCacheError(
            f"Cache file '{p}' is incompatible with the current codebase. "
            f"Re-run the pipeline in diagnostic mode to regenerate it. "
            f"Original error: {exc}"
        ) from exc

    if not isinstance(envelope, dict) or "context" not in envelope:
        raise StaleCacheError(
            f"Cache file '{p}' does not contain a valid envelope dict with a "
            f"'context' key. Re-run the pipeline in diagnostic mode to regenerate it."
        )

    ctx: PipelineContext = envelope["context"]

    # Check version fingerprint and warn on mismatch (do NOT raise)
    stored_fp = envelope.get("version_fingerprint", "")
    current_fp = context_fingerprint(ctx)
    if stored_fp and stored_fp != current_fp:
        logger.warning(
            "Cache file '%s' has a stale version fingerprint "
            "(stored=%s, current=%s). Context may be incompatible with the "
            "current codebase — proceed with caution.",
            p,
            stored_fp,
            current_fp,
        )

    return ctx


@runtime_checkable
class Stage(Protocol):
    """Structural protocol for all pipeline stages.

    Any class that implements a ``run(context: PipelineContext) -> PipelineContext``
    method is automatically a Stage — no inheritance required.

    Example::

        class MyStage:
            def run(self, context: PipelineContext) -> PipelineContext:
                context.camera_ids = ["cam1", "cam2"]
                return context
    """

    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute this stage, read from context, populate output fields, return context.

        Args:
            context: Accumulated pipeline state from prior stages.

        Returns:
            The same context object with this stage's output fields populated.

        """
        ...


@dataclass(frozen=True)
class ChunkHandoff:
    """Cross-chunk state carried between chunk invocations.

    Frozen so it is replaced wholesale each chunk, never mutated.

    Attributes:
        tracks_2d_state: Per-camera opaque tracker state blobs.
            Keys are camera IDs; values are dicts from KeypointTracker.get_state().
            Used to restore tracker continuity at the start of the next chunk.
        identity_map: Maps chunk-local fish IDs to globally consistent fish IDs.
            Keys are chunk-local fish IDs (from TrackletGroup.fish_id in the
            just-completed chunk); values are global fish IDs.
            Built by the identity stitcher after each chunk.
        track_id_to_global: Maps (camera_id, track_id) tuples to global fish IDs.
            Used for track-continuity-based identity stitching across chunk boundaries.
        next_global_id: Next globally unique fish ID to assign to an unmatched
            fish. Monotonically increasing across chunks to prevent ID reuse.
    """

    tracks_2d_state: dict  # camera_id -> KeypointTracker.get_state() blob
    identity_map: dict  # local_fish_id -> global_fish_id
    track_id_to_global: dict  # (camera_id, track_id) -> global_fish_id
    next_global_id: int


@dataclass
class PipelineContext:
    """Typed accumulator for inter-stage data flow in the 5-stage pipeline.

    Stages populate their output field(s) and return the context. Fields are
    None until the producing stage has run. Use :meth:`get` to retrieve a
    field with a clear error if the upstream stage has not yet executed.

    Fields use generic stdlib types only to preserve the engine import
    boundary (ENG-07). Actual element types are documented below.

    The 5-stage data flow (v3.7) is:
    1. Detection     -> ``detections``
    2. Pose          -> (enriches detections in-place with keypoints)
    3. 2D Tracking   -> ``tracks_2d``
    4. Association   -> ``tracklet_groups``
    5. Reconstruction -> ``midlines_3d``

    Attributes:
        frame_count: Number of frames processed. Set by the Detection stage (Stage 1).
        camera_ids: Active camera IDs. Set by the Detection stage (Stage 1).
            Type: ``list[str]``
        detections: Stage 1 (Detection) output. Per-frame per-camera detection results.
            Indexed by frame_idx. Each entry is a dict mapping camera_id to list of
            Detection objects. After Stage 2 (Pose), each Detection also carries
            ``keypoints`` and ``keypoint_conf`` fields.
            Type: ``list[dict[str, list[Detection]]]``
        tracks_2d: Stage 3 (2D Tracking) output. Per-camera temporal tracklets.
            Keys are camera IDs; values are lists of Tracklet2D objects for that camera.
            Type: ``dict[str, list[Tracklet2D]]``
        tracklet_groups: Stage 4 (Association) output. Cross-camera identity clusters.
            Each entry is a TrackletGroup representing one fish whose per-camera
            tracklets have been matched across cameras.
            Type: ``list[TrackletGroup]``
        midlines_3d: Stage 5 (Reconstruction) output. Per-frame 3D midline results.
            Each entry maps fish_id to a Spline3D (or Midline3D) object.
            Type: ``list[dict[int, Spline3D]]``
        stage_timing: Wall-clock seconds per stage, keyed by stage class name.
        carry_forward: Cross-chunk state persisted between chunk invocations.
            Holds tracker state and identity map so tracking and identity are
            continuous across chunk boundaries. None when no prior chunk has
            been processed. Runtime type: ChunkHandoff | None.

    """

    frame_count: int | None = None
    camera_ids: list | None = None
    detections: list | None = None
    tracks_2d: dict | None = None
    tracklet_groups: list | None = None
    midlines_3d: list | None = None
    stage_timing: dict = field(default_factory=dict)
    carry_forward: ChunkHandoff | None = None

    def get(self, field_name: str) -> object:
        """Return the value of a field, raising ValueError if it is None.

        Args:
            field_name: Name of the PipelineContext field to retrieve.

        Returns:
            The field value (guaranteed non-None).

        Raises:
            ValueError: If the field is None, indicating the producing stage
                has not yet run.
            AttributeError: If ``field_name`` is not a valid field on this dataclass.

        """
        value = getattr(self, field_name)
        if value is None:
            raise ValueError(
                f"PipelineContext.{field_name} is None — the stage that produces "
                f"'{field_name}' has not run yet. Check stage ordering.",
            )
        return value
