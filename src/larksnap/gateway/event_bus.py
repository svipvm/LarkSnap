"""Thread-safe event bus for inter-module communication.

Publishers and subscribers can run on different threads. Handlers
are invoked synchronously on the publisher's thread, so a handler
that does slow work (HTTP, disk I/O) will block the publisher.
Callers needing non-blocking dispatch should offload the slow work
to a ``WorkerPool`` themselves.

Concurrency contract:
  - subscribe / unsubscribe / clear / publish are all safe to call
    from any thread, including concurrently.
  - Handlers are invoked on the publisher's thread, holding NO lock
    for the duration of the call. This avoids the deadlock that
    would arise if a handler tried to subscribe/unsubscribe from
    inside its own callback.
"""

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Enumeration of system event types."""

    FRAME_CAPTURED = "frame_captured"
    DETECTION_COMPLETED = "detection_completed"
    # Emitted by the snapshot service every time a detection
    # snapshot is written to disk. ``data`` carries the absolute
    # file path (str). Fires whether or not the notifier is
    # enabled — the snapshot is owned by the detection path.
    SNAPSHOT_SAVED = "snapshot_saved"
    NOTIFICATION_SENT = "notification_sent"
    ERROR_OCCURRED = "error_occurred"
    CAMERA_FAILED = "camera_failed"
    CAMERA_READ_FAILED = "camera_read_failed"
    CAMERA_OPENED = "camera_opened"
    CAMERA_CLOSED = "camera_closed"
    DETECTION_STARTED = "detection_started"
    DETECTION_STOPPED = "detection_stopped"
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    CHAT_ID_OBTAINED = "chat_id_obtained"
    NOTIFICATION_ENABLED = "notification_enabled"
    NOTIFICATION_DISABLED = "notification_disabled"
    # Per-subsystem state change (Camera/Detector/Notifier). ``data``
    # carries a ``ComponentStatus`` instance from
    # ``larksnap.gateway.component_state``.
    COMPONENT_STATE_CHANGED = "component_state_changed"


@dataclass
class Event:
    """Event data class for inter-module communication."""

    type: EventType
    data: Any | None = None
    source: str | None = None


EventHandler = Callable[["Event"], None]


class EventBus:
    """Thread-safe event bus for publish/subscribe inter-module communication."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        # RLock (re-entrant) so a handler that calls subscribe/unsubscribe
        # on the same bus from inside its own callback doesn't deadlock.
        self._lock = threading.RLock()
        self._logger = logging.getLogger("larksnap.event_bus")

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        with self._lock:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        with self._lock:
            handlers = self._handlers.get(event_type)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                # Idempotent: unsubscribing an unknown handler is a no-op.
                pass

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers.

        Handlers are invoked on the publisher's thread. To avoid
        deadlocks, the lock is released before any handler runs.
        Exceptions in handlers are logged and swallowed so one bad
        handler can't stop the rest from running.
        """
        # Snapshot the handler list under the lock, then release the
        # lock before invoking them. This prevents a slow handler
        # from blocking subscribe/unsubscribe on other threads, and
        # also prevents a handler that calls subscribe/unsubscribe
        # from deadlocking.
        with self._lock:
            handlers = list(self._handlers.get(event.type, ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._logger.error(
                    "Error in event handler for %s: %s", event.type.value, e
                )

    def clear(self) -> None:
        """Remove all event subscriptions."""
        with self._lock:
            self._handlers.clear()

    def handler_count(self, event_type: EventType) -> int:
        """Return the number of handlers subscribed to an event type (for tests)."""
        with self._lock:
            return len(self._handlers.get(event_type, ()))
