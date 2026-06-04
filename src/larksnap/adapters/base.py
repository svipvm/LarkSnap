from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """Abstract base class for all adapters with lifecycle management."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the adapter resources."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the adapter."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the adapter."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release all adapter resources."""
        pass
