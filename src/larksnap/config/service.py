"""Live configuration service.

The :class:`ConfigService` is the single source of truth for
``AppConfig`` at runtime. It wraps the in-memory pydantic model with:

* Dotted-path access (``detector.target_classes``, ``camera.fps``)
  with strict type coercion via pydantic's ``TypeAdapter``.
* JSON-encoded values so chat commands can pass complex values
  (``["person","car"]``, ``true``, ``30.5``) without ambiguity.
* Automatic disk persistence after every successful ``set``.
* A small pub/sub for components that need to react to live
  changes (e.g. the seg detector updates its target-class filter
  set when the user runs ``/config set detector.target_classes …``).
* A static list of "restart-required" prefixes so the controller
  can flag fields like ``camera.*`` or ``recorder.*`` in chat
  replies — they get saved to disk but won't apply until the
  next process start.

Threading:
    Public methods are guarded by an internal ``RLock`` so the
    chat command thread, the ZMQ result thread, and the UI
    thread can all call ``get`` / ``set`` safely. Subscribers
    are invoked serially; a buggy listener is logged and
    skipped rather than aborting the change.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, TypeAdapter

from larksnap.config.loader import save_config
from larksnap.config.models import AppConfig
from larksnap.utils.exceptions import ConfigError


# Paths that require a process restart to take effect. ``set``
# still persists them to disk — the chat reply just annotates
# them so the user knows the change won't be live until the
# process is restarted. The list is intentionally explicit
# rather than auto-derived: a new field defaults to "live" and
# only moves to this list once we know it can't be hot-swapped.
RESTART_REQUIRED_PREFIXES: tuple[str, ...] = (
    "camera.",
    "detector.type",
    "detector.mock.",
    "detector.seg.",
    "recorder.",
    "logging.",
    "service.",
)


class ConfigService:
    """In-memory ``AppConfig`` with disk persistence and live-change hooks.

    The service is intentionally thin: it does not own the
    ``AppConfig`` instance — callers can read ``self.config`` for
    a direct reference. This lets existing components (the
    notification service, the controller) keep reading the same
    pydantic tree they always have, while a ``set`` call atomically
    mutates it and notifies subscribers.
    """

    def __init__(
        self,
        config: AppConfig,
        config_path: Path | str | None = None,
    ) -> None:
        self._config = config
        self._config_path = Path(config_path) if config_path else None
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[str, Any, Any], None]] = []
        self._logger = logging.getLogger("larksnap.config.service")

    # ── Public properties ────────────────────────────────────────

    @property
    def config(self) -> AppConfig:
        """Direct reference to the in-memory ``AppConfig``.

        Exposed so the controller and other components can keep
        reading fields like ``self._config.gateway.notification_interval``
        the same way they did before the service existed. Mutations
        should go through :py:meth:`set` so persistence and
        notifications are handled.
        """
        return self._config

    @property
    def config_path(self) -> Path | None:
        """Path to the YAML file the config was loaded from / saved to."""
        return self._config_path

    # ── Path access ──────────────────────────────────────────────

    def get(self, path: str) -> Any:
        """Return the value at ``path``.

        ``path`` is dotted, e.g. ``detector.target_classes``.
        Raises :class:`ConfigError` for unknown / malformed paths.
        """
        with self._lock:
            node = self._resolve_node(path)
            return node

    def set(
        self, path: str, json_value: str
    ) -> tuple[Any, Any, str]:
        """Set the value at ``path`` from a JSON-encoded ``json_value``.

        Returns ``(old_value, new_value, status)`` where ``status``
        is ``"applied"`` (live) or ``"restart_required"``.

        Raises :class:`ConfigError` for unknown paths, invalid JSON,
        type mismatches, or pydantic validation failures.
        """
        with self._lock:
            # 1. Parse the JSON. We accept any JSON literal so a
            #    user can pass booleans / numbers / lists / strings
            #    / objects unambiguously. A bare bareword like
            #    ``true`` without quotes would be invalid JSON
            #    here — that's the price of accuracy.
            try:
                parsed = json.loads(json_value)
            except json.JSONDecodeError as e:
                raise ConfigError(
                    f"值不是合法 JSON: {json_value!r} ({e.msg})"
                ) from e

            # 2. Walk to the parent + look up the leaf metadata.
            parent, leaf, leaf_type = self._walk(path)
            old = getattr(parent, leaf)

            # 3. Coerce using pydantic's TypeAdapter so a user
            #    passing ``"30"`` for an int field gets 30, not "30".
            try:
                new = self._coerce(parsed, leaf_type)
            except (ValueError, TypeError) as e:
                raise ConfigError(
                    f"无法将 {parsed!r} 转为 {leaf_type} (字段 {path}): {e}"
                ) from e

            # 4. Set the attribute; pydantic v2 re-validates.
            try:
                setattr(parent, leaf, new)
            except (ValueError, TypeError) as e:
                raise ConfigError(
                    f"字段 {path} 验证失败: {e}"
                ) from e

            # 5. Persist + notify. Either failure must not leave
            #    the in-memory model and the disk file in a
            #    divergent state, so we persist first and only
            #    notify on success.
            self._save()
            self._notify(path, old, new)

            status = (
                "restart_required" if self._needs_restart(path) else "applied"
            )
            return old, new, status

    def list_paths(self, prefix: str = "") -> list[str]:
        """List all available config paths, optionally filtered by ``prefix``."""
        with self._lock:
            paths = self._collect_paths(self._config)
        if not prefix:
            return paths
        return [p for p in paths if p.startswith(prefix)]

    # ── Subscriptions ────────────────────────────────────────────

    def subscribe(
        self, callback: Callable[[str, Any, Any], None]
    ) -> Callable[[], None]:
        """Register a listener for live updates.

        The callback receives ``(path, old_value, new_value)`` after
        every successful ``set``. Returns an ``unsubscribe`` function
        so callers can deregister cleanly.
        """
        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsubscribe

    def save(self) -> None:
        """Persist the current in-memory config to disk."""
        with self._lock:
            self._save()

    # ── Helpers ──────────────────────────────────────────────────

    def needs_restart(self, path: str) -> bool:
        """True if changing ``path`` requires a process restart to apply."""
        return self._needs_restart(path)

    def _resolve_node(self, path: str) -> Any:
        """Resolve a dotted path to its current value or sub-model."""
        if not path:
            raise ConfigError("路径不能为空")
        parts = path.split(".")
        if not all(parts):
            raise ConfigError(f"路径格式无效: {path!r}")
        current: Any = self._config
        for part in parts:
            if not isinstance(current, BaseModel):
                raise ConfigError(
                    f"路径 {path!r} 在 {part!r} 处穿过了非模型节点"
                )
            if part not in current.model_fields:
                raise ConfigError(
                    f"未知路径: {path!r} (字段 {part!r} 不存在)"
                )
            current = getattr(current, part)
        return current

    def _walk(self, path: str) -> tuple[BaseModel, str, Any]:
        """Walk to the parent of the leaf and return its metadata.

        Returns ``(parent_model, leaf_name, leaf_annotation)``. The
        annotation is suitable for ``pydantic.TypeAdapter`` so a
        caller can validate an arbitrary value against the leaf's
        declared type.
        """
        if not path:
            raise ConfigError("路径不能为空")
        parts = path.split(".")
        if not all(parts):
            raise ConfigError(f"路径格式无效: {path!r}")
        current: Any = self._config
        for part in parts[:-1]:
            if not isinstance(current, BaseModel):
                raise ConfigError(
                    f"路径 {path!r} 在 {part!r} 处穿过了非模型节点"
                )
            if part not in current.model_fields:
                raise ConfigError(
                    f"未知路径: {path!r} (字段 {part!r} 不存在)"
                )
            current = getattr(current, part)
        if not isinstance(current, BaseModel):
            raise ConfigError(f"路径 {path!r} 终点不是模型节点")
        leaf = parts[-1]
        if leaf not in current.model_fields:
            raise ConfigError(
                f"未知路径: {path!r} (字段 {leaf!r} 不存在)"
            )
        leaf_type = current.model_fields[leaf].annotation
        return current, leaf, leaf_type

    def _coerce(self, parsed: Any, target_type: Any) -> Any:
        """Coerce a JSON-parsed value to ``target_type``.

        ``pydantic.TypeAdapter`` is the workhorse here: it understands
        ``list[str]``, ``tuple[float, float]``, ``Optional[X]`` and
        every other annotation pydantic accepts.
        """
        try:
            adapter = TypeAdapter(target_type)
            return adapter.validate_python(parsed)
        except Exception as e:
            raise ConfigError(
                f"类型校验失败 ({target_type}): {e}"
            ) from e

    def _collect_paths(
        self, node: BaseModel, prefix: str = ""
    ) -> list[str]:
        """DFS through the pydantic tree to collect every leaf path."""
        paths: list[str] = []
        for name in node.model_fields:
            full = f"{prefix}.{name}" if prefix else name
            value = getattr(node, name)
            if isinstance(value, BaseModel):
                paths.extend(self._collect_paths(value, full))
            else:
                paths.append(full)
        return paths

    def _needs_restart(self, path: str) -> bool:
        for prefix in RESTART_REQUIRED_PREFIXES:
            if path == prefix.rstrip("."):
                return True
            if path.startswith(prefix):
                return True
        return False

    def _save(self) -> None:
        if self._config_path is None:
            # No path was supplied at construction time — the
            # service is running in-memory only. The chat reply
            # should already have told the user "需重启" so a
            # restart is required to re-load, but at least the
            # in-memory state stays consistent.
            self._logger.debug(
                "ConfigService has no config_path; skipping save"
            )
            return
        save_config(self._config, str(self._config_path))

    def _notify(self, path: str, old: Any, new: Any) -> None:
        # Iterate over a copy so a subscriber that unsubscribes
        # itself doesn't break the rest of the chain.
        for cb in list(self._subscribers):
            try:
                cb(path, old, new)
            except Exception as e:
                # A buggy listener must not abort the rest of
                # the chain — log and continue. The save has
                # already happened, so the on-disk state is
                # consistent regardless.
                self._logger.warning(
                    "Config subscriber for %r raised: %s", path, e
                )
