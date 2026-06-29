"""Shared service body used by every platform wrapper.

The Windows service and the Linux systemd service (and the generic
POSIX fallback) all want the same thing: load the config, set up
the logger, start a :class:`GatewayController`, block until a stop
signal, then tear the controller down. The *only* thing that varies
between platforms is the stop-signal mechanism. This module keeps the
common body in one place so the per-platform wrappers stay tiny and
the logic is testable in isolation.
"""

from __future__ import annotations

import logging
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Callable

from larksnap.config.loader import load_config
from larksnap.config.service import ConfigService
from larksnap.gateway.controller import GatewayController
from larksnap.utils.logger import setup_logger


@dataclass(frozen=True)
class ServiceRunnerConfig:
    """Configuration object for :class:`ServiceRunner`.

    ``config_path`` – YAML config to load. ``None`` uses the
    loader's default. ``console_output`` defaults to ``False`` because
    services normally log to a file – enabling it is useful when
    running in the foreground for debugging.
    """

    config_path: str | None = None
    console_output: bool = False


class ServiceRunner:
    """Encapsulates the long-lived service state.

    Public lifecycle:

    1. ``__init__`` – bind the controller but don't start it.
    2. ``start`` – ``controller.initialize()`` then
       ``controller.start()`` (camera + pipeline).
    3. ``wait_for_stop`` – block until :meth:`request_stop` is
       called from a signal handler / another thread.
    4. ``stop`` – graceful controller teardown (idempotent).
    """

    def __init__(
        self,
        config_path: str | None = None,
        controller_factory: Callable[[], GatewayController] | None = None,
    ) -> None:
        self._config_path = config_path
        self._controller_factory = controller_factory
        self._logger: logging.Logger | None = None
        self._controller: GatewayController | None = None
        self._stop_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Load config, set up logger, bring the controller up."""
        if self._controller is not None:
            raise RuntimeError("ServiceRunner already started")

        config = load_config(self._config_path)
        self._logger = setup_logger(
            level=config.logging.level,
            log_format=config.logging.format,
            file_path=config.logging.file_path,
            max_bytes=config.logging.max_bytes,
            backup_count=config.logging.backup_count,
            console_output=config.logging.console_output,
        )
        if self._logger is not None:
            self._logger.info("LarkSnap service starting...")

        if self._controller_factory is not None:
            self._controller = self._controller_factory()
        else:
            # Build a ConfigService so the service-mode gateway
            # also supports ``/config set`` from the Feishu chat.
            # Without this the command would still mutate the
            # in-memory AppConfig but would not persist to disk,
            # which would be a confusing regression vs. the
            # Qt / tray modes.
            config_service = ConfigService(config, config_path=self._config_path)
            self._controller = GatewayController(
                config, config_service=config_service
            )

        assert self._controller is not None  # for type checkers
        self._controller.initialize()
        self._controller.start()

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        """Block until :meth:`request_stop` is called.

        Returns ``True`` if a stop was requested, ``False`` if the
        timeout elapsed. Pass ``None`` (the default) to wait forever.
        """
        return self._stop_event.wait(timeout=timeout)

    def request_stop(self) -> None:
        """Signal the runner to shut down.

        Idempotent – safe to call from multiple signal handlers.
        The actual controller teardown is deferred to :meth:`stop`.
        """
        self._stop_event.set()

    def stop(self) -> None:
        """Tear the controller down. Idempotent."""
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        if self._controller is not None:
            try:
                self._controller.stop()
            except Exception as exc:  # noqa: BLE001
                if self._logger is not None:
                    self._logger.error("Controller stop failed: %s", exc)

        if self._logger is not None:
            self._logger.info("LarkSnap service stopped")

    # ------------------------------------------------------------------
    # Convenience accessors (used by the platform wrappers and tests)
    # ------------------------------------------------------------------
    @property
    def controller(self) -> GatewayController | None:
        """Return the underlying controller, or None if not started yet."""
        return self._controller

    @property
    def logger(self) -> logging.Logger | None:
        """Return the service logger, or None if not started yet."""
        return self._logger

    @property
    def config_path(self) -> str | None:
        """Return the config path that was used to start the runner."""
        return self._config_path


# ---------------------------------------------------------------------------
# Module-level helper used by every platform wrapper.
# ---------------------------------------------------------------------------
def run_service_blocking(
    config_path: str | None = None,
    *,
    on_ready: Callable[[ServiceRunner], None] | None = None,
) -> None:
    """Run the service body until a stop signal arrives.

    Args:
        config_path: YAML config path forwarded to the loader.
        on_ready: Optional hook invoked *after* the controller is up
            but *before* :meth:`wait_for_stop`. The Windows wrapper
            uses it to flip the SCM state to ``SERVICE_RUNNING``; the
            Linux wrapper uses it to emit the ``READY=1`` sd_notify
            message. Keeping the hook here means the body is
            platform-agnostic.
    """
    runner = ServiceRunner(config_path=config_path)
    runner.start()

    try:
        if on_ready is not None:
            on_ready(runner)
        runner.wait_for_stop()
    finally:
        runner.stop()


# ---------------------------------------------------------------------------
# POSIX signal-based wait used by Linux / macOS / generic POSIX.
# ---------------------------------------------------------------------------
def install_posix_stop_handlers(runner: ServiceRunner) -> None:
    """Wire SIGTERM / SIGINT to :meth:`ServiceRunner.request_stop`.

    On Windows, :func:`signal.signal` only supports ``SIGINT`` /
    ``SIGBREAK``; the Windows service uses its own event loop and
    does not call this helper.
    """
    def _handle(signum: int, _frame: FrameType | None) -> None:
        if runner.logger is not None:
            runner.logger.info("Received signal %d, shutting down", signum)
        runner.request_stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):
            # SIGTERM may not be installable on some platforms
            # (e.g. Windows); ignore – the service relies on other
            # stop mechanisms there.
            pass


# ---------------------------------------------------------------------------
# Re-export the default config path so other modules (e.g. systemd unit
# template) can reference the same default the loader uses.
# ---------------------------------------------------------------------------
def default_config_path() -> Path:
    """Return the loader's default config path."""
    from larksnap.config.loader import load_config  # noqa: F401 – re-import for typing

    # The loader is intentionally stateful only when actually called,
    # so we replicate the same default discovery here.
    return Path.cwd() / "config" / "config.yaml"
