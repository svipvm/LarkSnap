from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from larksnap.adapters.base import BaseAdapter


@dataclass
class NotificationMessage:
    """Message data class for notification delivery."""

    title: str
    content: str
    label: str
    confidence: float
    timestamp: str
    snapshot_path: str | None = None
    extra: dict[str, Any] | None = None


class NotifierAdapter(BaseAdapter):
    """Abstract notifier adapter interface for sending notifications."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the notification service."""
        pass

    @abstractmethod
    def send_message(self, message: NotificationMessage) -> bool:
        """Send a notification message."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the notification service."""
        pass

    def initialize(self) -> None:
        self.connect()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.disconnect()

    def release(self) -> None:
        pass
