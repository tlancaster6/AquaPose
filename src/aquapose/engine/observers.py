"""Observer protocol and EventBus for typed synchronous event dispatch.

The observer pattern decouples pipeline execution from side effects.  Stages
perform pure computation; observers react to lifecycle events (timing,
logging, export, visualization) without mutating pipeline state.

Design invariants:
- Delivery is *synchronous* — the pipeline blocks on each ``on_event`` call.
  Determinism is mandatory; if an observer needs non-blocking behaviour it
  manages its own internal queue / worker thread.
- Observers are *passive* — they must not mutate pipeline state, change stage
  logic, or control execution flow.
- Subscription is *typed* — an observer subscribes to a specific ``Event``
  subclass (or the ``Event`` base to receive everything).
- Dispatch is *fault-tolerant* — if an observer raises, a warning is logged
  and delivery continues to the remaining observers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Protocol, runtime_checkable

from aquapose.engine.events import Event

logger = logging.getLogger(__name__)


@runtime_checkable
class Observer(Protocol):
    """Structural protocol for pipeline event observers.

    Any class that defines an ``on_event`` method with the correct signature
    satisfies this protocol — no inheritance required.

    Example::

        class MyLogger:
            def on_event(self, event: Event) -> None:
                print(f"[{type(event).__name__}] {event}")

        bus = EventBus()
        bus.subscribe(StageStart, MyLogger())
    """

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event.

        Args:
            event: The event instance. The concrete type is always one of the
                event subclasses defined in ``aquapose.engine.events``.

        Note:
            Observers must not mutate pipeline state, raise exceptions
            intentionally, or perform blocking I/O without managing their own
            internal thread.
        """
        ...


class EventBus:
    """Typed, synchronous event dispatcher.

    Observers register interest in a specific event *type*.  When an event is
    emitted via :meth:`emit`, it is delivered to all observers subscribed to
    the event's exact type **and** to all observers subscribed to any ancestor
    type in the ``Event`` hierarchy (most-specific → most-general, in
    subscription order within each tier).

    Example::

        bus = EventBus()
        bus.subscribe(StageStart, timing_observer)
        bus.subscribe(Event, audit_logger)   # receives everything
        bus.emit(StageStart(stage_name="detection", stage_index=0))
        # timing_observer.on_event called, then audit_logger.on_event called
    """

    def __init__(self) -> None:
        # Maps event type -> ordered list of subscribed observers.
        self._subscriptions: dict[type[Event], list[Observer]] = defaultdict(list)

    def subscribe(self, event_type: type[Event], observer: Observer) -> None:
        """Register *observer* to receive events of *event_type*.

        Args:
            event_type: The event class to subscribe to.  Use ``Event`` (the
                base class) to receive all events.
            observer: Any object satisfying the :class:`Observer` protocol.
        """
        self._subscriptions[event_type].append(observer)

    def unsubscribe(self, event_type: type[Event], observer: Observer) -> None:
        """Remove *observer* from the subscription list for *event_type*.

        If *observer* is not subscribed to *event_type* this is a no-op.

        Args:
            event_type: The event class to unsubscribe from.
            observer: The observer to remove.
        """
        observers = self._subscriptions.get(event_type)
        if observers and observer in observers:
            observers.remove(observer)

    def emit(self, event: Event) -> None:
        """Deliver *event* synchronously to all matching observers.

        Delivery order:
        1. Observers subscribed to the event's *exact* type (in subscription
           order).
        2. Observers subscribed to each *parent* type in MRO order, skipping
           ``object`` (in subscription order per parent).

        This means subscribing to ``StageStart`` receives only ``StageStart``
        events, while subscribing to ``Event`` (the base) receives every event.

        Fault tolerance: if an observer's ``on_event`` raises an exception, a
        warning is logged and delivery continues to the remaining observers.

        Args:
            event: The event to dispatch.
        """
        event_type = type(event)

        # Walk the MRO to find all Event ancestor types (excluding object).
        # We collect observers in MRO order so exact-type subscribers fire first.
        seen: set[int] = set()  # avoid duplicate delivery for diamond hierarchies
        for ancestor in event_type.__mro__:
            if ancestor is object:
                continue
            if not (isinstance(ancestor, type) and issubclass(ancestor, Event)):
                continue
            for obs in list(self._subscriptions.get(ancestor, [])):
                obs_id = id(obs)
                if obs_id in seen:
                    continue
                seen.add(obs_id)
                try:
                    obs.on_event(event)
                except Exception:
                    logger.warning(
                        "Observer %r raised an exception on event %r; "
                        "continuing delivery to remaining observers.",
                        obs,
                        event,
                        exc_info=True,
                    )


__all__ = ["EventBus", "Observer"]
