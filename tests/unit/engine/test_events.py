"""Unit tests for the typed event dataclasses and EventBus/Observer system.

Tests cover:
- Frozen dataclass immutability
- Timestamp field auto-population
- Observer structural typing (no inheritance required)
- EventBus subscription, filtered dispatch, synchronous ordering
- Base-type subscription receiving all events
- Fault-tolerant dispatch (observer error does not stop others)
- Unsubscribe removes observer from future dispatches
"""

from __future__ import annotations

import time

import pytest

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

# ---------------------------------------------------------------------------
# Event dataclass tests
# ---------------------------------------------------------------------------


def test_event_dataclasses_frozen() -> None:
    """Mutating a frozen event field raises FrozenInstanceError."""
    from dataclasses import FrozenInstanceError

    event = PipelineStart(run_id="run_test", config=None)
    with pytest.raises(FrozenInstanceError):
        event.run_id = "mutated"  # type: ignore[misc]


def test_event_has_timestamp() -> None:
    """StageComplete event has a float timestamp within the last 5 seconds."""
    before = time.time()
    event = StageComplete(
        stage_name="detection", stage_index=0, elapsed_seconds=1.0, summary={}
    )
    after = time.time()

    assert isinstance(event.timestamp, float)
    assert before <= event.timestamp <= after


# ---------------------------------------------------------------------------
# Observer protocol tests
# ---------------------------------------------------------------------------


def test_observer_structural_typing() -> None:
    """A plain class with on_event satisfies Observer without inheritance."""

    class SimpleRecorder:
        def __init__(self) -> None:
            self.received: list[Event] = []

        def on_event(self, event: Event) -> None:
            self.received.append(event)

    recorder = SimpleRecorder()
    assert isinstance(recorder, Observer), (
        "SimpleRecorder should satisfy Observer protocol via structural typing"
    )


# ---------------------------------------------------------------------------
# EventBus tests
# ---------------------------------------------------------------------------


def test_eventbus_delivers_to_subscriber() -> None:
    """EventBus calls on_event exactly once for the subscribed event type."""
    received: list[Event] = []

    class Recorder:
        def on_event(self, event: Event) -> None:
            received.append(event)

    bus = EventBus()
    recorder = Recorder()
    bus.subscribe(StageStart, recorder)

    evt = StageStart(stage_name="detection", stage_index=0)
    bus.emit(evt)

    assert len(received) == 1
    assert received[0] is evt


def test_eventbus_filters_by_type() -> None:
    """Observer subscribed to StageStart does not receive PipelineComplete."""
    received_a: list[Event] = []
    received_b: list[Event] = []

    class A:
        def on_event(self, event: Event) -> None:
            received_a.append(event)

    class B:
        def on_event(self, event: Event) -> None:
            received_b.append(event)

    bus = EventBus()
    bus.subscribe(StageStart, A())
    bus.subscribe(PipelineComplete, B())

    evt = StageStart(stage_name="seg", stage_index=1)
    bus.emit(evt)

    assert len(received_a) == 1, "A should receive StageStart"
    assert len(received_b) == 0, "B should NOT receive StageStart"


def test_eventbus_synchronous_order() -> None:
    """Observers receive the event in subscription order (A, B, C)."""
    order: list[str] = []

    class MakeRecorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def on_event(self, event: Event) -> None:
            order.append(self.name)

    bus = EventBus()
    bus.subscribe(PipelineStart, MakeRecorder("A"))
    bus.subscribe(PipelineStart, MakeRecorder("B"))
    bus.subscribe(PipelineStart, MakeRecorder("C"))

    bus.emit(PipelineStart(run_id="run_test", config=None))

    assert order == ["A", "B", "C"]


def test_eventbus_base_type_subscription() -> None:
    """Subscribing to Event (base) receives all event subtypes."""
    received: list[Event] = []

    class CatchAll:
        def on_event(self, event: Event) -> None:
            received.append(event)

    bus = EventBus()
    bus.subscribe(Event, CatchAll())

    bus.emit(StageStart(stage_name="tracking", stage_index=2))
    bus.emit(PipelineComplete(run_id="run_test", elapsed_seconds=5.0))
    bus.emit(FrameProcessed(stage_name="seg", frame_index=0, frame_count=100))

    assert len(received) == 3, (
        "CatchAll subscribed to Event base should receive all 3 events"
    )


def test_eventbus_fault_tolerant() -> None:
    """A raising observer does not prevent delivery to subsequent observers."""
    received_b: list[Event] = []

    class RaisingObserver:
        def on_event(self, event: Event) -> None:
            raise RuntimeError("deliberate failure")

    class GoodObserver:
        def on_event(self, event: Event) -> None:
            received_b.append(event)

    bus = EventBus()
    bus.subscribe(PipelineFailed, RaisingObserver())
    bus.subscribe(PipelineFailed, GoodObserver())

    evt = PipelineFailed(run_id="run_test", error="oops", elapsed_seconds=1.0)
    # Should not raise — fault-tolerant dispatch continues past the error.
    bus.emit(evt)

    assert len(received_b) == 1, "GoodObserver should still receive the event"
    assert received_b[0] is evt


def test_unsubscribe() -> None:
    """After unsubscribe, observer receives no further events."""
    received: list[Event] = []

    class Recorder:
        def on_event(self, event: Event) -> None:
            received.append(event)

    bus = EventBus()
    recorder = Recorder()
    bus.subscribe(StageComplete, recorder)

    evt1 = StageComplete(stage_name="det", stage_index=0, elapsed_seconds=1.0)
    bus.emit(evt1)
    assert len(received) == 1, "Observer should receive first event"

    bus.unsubscribe(StageComplete, recorder)

    evt2 = StageComplete(stage_name="seg", stage_index=1, elapsed_seconds=0.5)
    bus.emit(evt2)
    assert len(received) == 1, "Observer should NOT receive event after unsubscribe"
