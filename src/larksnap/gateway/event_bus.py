import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Enumeration of system event types."""

    FRAME_CAPTURED = "frame_captured"
    DETECTION_COMPLETED = "detection_completed"
    NOTIFICATION_SENT = "notification_sent"
    ERROR_OCCURRED = "error_occurred"
    CAMERA_FAILED = "camera_failed"
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    CHAT_ID_OBTAINED = "chat_id_obtained"
    NOTIFICATION_ENABLED = "notification_enabled"
    NOTIFICATION_DISABLED = "notification_disabled"


@dataclass
class Event:
    """Event data class for inter-module communication."""

    type: EventType
    data: Any | None = None
    source: str | None = None


EventHandler = Callable[[Event], None]


class EventBus:
    """Event bus for publish/subscribe inter-module communication."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._logger = logging.getLogger("larksnap.event_bus")

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers."""
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._logger.error(
                    "Error in event handler for %s: %s", event.type.value, e
                )

    def clear(self) -> None:
        """Remove all event subscriptions."""
        self._handlers.clear()
