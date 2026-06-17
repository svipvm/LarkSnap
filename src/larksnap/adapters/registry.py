"""Adapter registry with factory pattern.

Provides a centralized registry for adapter types, eliminating if-else
chains in the controller. New adapters register themselves via decorator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

from larksnap.utils.exceptions import GatewayError

if TYPE_CHECKING:
    from larksnap.adapters.base import BaseAdapter

T = TypeVar("T", bound="BaseAdapter")

_logger = logging.getLogger("larksnap.adapters.registry")


class AdapterRegistry:
    """Generic adapter registry with decorator-based registration.

    Usage:
        registry = AdapterRegistry[DetectorAdapter]("detector")

        @registry.register("seg")
        class SegDetectorAdapter(DetectorAdapter): ...

        adapter = registry.create("seg", config)
    """

    def __init__(self, category: str) -> None:
        self._category = category
        self._registry: dict[str, type] = {}

    def register(self, name: str):
        """Decorator to register an adapter class under a given name."""
        def decorator(cls: type) -> type:
            if name in self._registry:
                _logger.warning(
                    "Overwriting %s adapter '%s': %s → %s",
                    self._category, name, self._registry[name].__name__, cls.__name__,
                )
            self._registry[name] = cls
            _logger.debug("Registered %s adapter: %s → %s", self._category, name, cls.__name__)
            return cls
        return decorator

    def create(self, name: str, *args, **kwargs):
        """Create an adapter instance by registered name."""
        if name not in self._registry:
            available = ", ".join(self._registry.keys()) or "(none)"
            raise GatewayError(
                f"Unknown {self._category} adapter: '{name}'. "
                f"Available: {available}"
            )
        return self._registry[name](*args, **kwargs)

    def available(self) -> list[str]:
        """Return list of registered adapter names."""
        return list(self._registry.keys())

    def has(self, name: str) -> bool:
        """Check if an adapter name is registered."""
        return name in self._registry


# ─── Global registries ────────────────────────────────────────────────

camera_registry = AdapterRegistry("camera")
detector_registry = AdapterRegistry("detector")
notifier_registry = AdapterRegistry("notifier")
