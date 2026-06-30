"""Shared pytest fixtures for the LarkSnap test suite.

Centralises the boilerplate that every controller / snapshot / notification
test would otherwise duplicate. The fixtures here are deliberately small and
behaviour-focussed: each one builds a single piece of the system with the
absolute minimum of magic, so a failing test pinpoints its layer by the
fixture name alone.

The fixtures follow a few conventions:

* **State isolation** — every test that writes to the runtime state file
  gets its own ``tmp_path_factory``-backed path. This is the only way to
  prevent cross-test leakage through ``%APPDATA%/LarkSnap/runtime_state.json``
  on Windows.
* **No real hardware** — the camera adapter is replaced with a ``MagicMock``
  that returns a deterministic black frame. The detector type is set to
  ``"mock"`` so no onnx model is loaded.
* **Bounded timeouts** — the close-wait timeout is 3 s (see
  ``close_timeout`` below), not the 5 s used in production. Mock
  pipelines drain in milliseconds; the production timeout exists
  to bound real ZMQ teardown, not the test path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from larksnap.config.models import AppConfig
from larksnap.gateway.controller import GatewayController

if TYPE_CHECKING:
    from larksnap.adapters.detector.interface import DetectionResult
    from larksnap.adapters.notifier.interface import NotifierAdapter
    from larksnap.gateway.event_bus import EventBus
    from larksnap.gateway.notification_service import (
        NotificationService,
        NotificationServiceConfig,
    )
    from larksnap.gateway.snapshot_service import (
        SnapshotService,
        SnapshotServiceConfig,
    )
    from larksnap.utils.worker_pool import WorkerPool


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_config() -> AppConfig:
    """A default AppConfig tuned for unit tests.

    Sets ``detector.type = "mock"`` (so no onnx model is loaded) and
    ``camera.device_index = 999`` (so no real camera is ever probed).
    Other knobs (save_interval, message_template) keep their defaults
    so a test that doesn't override them exercises the real contract.
    """
    cfg = AppConfig()
    cfg.detector.type = "mock"
    cfg.camera.device_index = 999
    return cfg


@pytest.fixture
def mock_camera() -> MagicMock:
    """A drop-in camera adapter that returns a black frame forever.

    Use this anywhere a real ``CameraAdapter`` is required; it never
    blocks and never errors.
    """
    cam = MagicMock()
    cam.is_opened.return_value = True
    cam.read_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return cam


@pytest.fixture
def install_mock_adapters(
    mock_camera: MagicMock,
) -> Callable[[GatewayController], None]:
    """Return a callable that patches a controller with mock adapters.

    The lambda replaces the controller's ``_create_adapters`` method so
    a single line of setup wires up camera/detector/notifier/recorder:

        controller = _make_controller()
        install_mock_adapters(controller)
        controller.open_camera()

    Reusing the same lambda across tests gives them a uniform shape
    and makes the controller setup boilerplate disappear.
    """

    def _install(controller: GatewayController) -> None:
        def _populate() -> None:
            controller._camera = mock_camera
            controller._detector = MagicMock()
            controller._notifier = MagicMock()
            controller._recorder = MagicMock()

        # Bind _create_adapters to the controller instance, the same
        # way test_controller_lifecycle.py does it manually.
        controller._create_adapters = _populate  # type: ignore[method-assign]

    return _install


@pytest.fixture
def gateway_controller(
    app_config: AppConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> GatewayController:
    """A fresh GatewayController pointed at a per-test state file.

    The state file lives under ``tmp_path_factory.mktemp(...)`` so it
    cannot leak into another test or pollute ``%APPDATA%/LarkSnap``.
    Tests that don't drive the state machine can ignore the file
    entirely; the controller creates an empty one on first use and
    the temp directory is reaped on teardown.
    """
    state_path = tmp_path_factory.mktemp("runtime_state") / "runtime_state.json"
    return GatewayController(app_config, state_path=state_path)


@pytest.fixture
def close_timeout() -> float:
    """Timeout (seconds) for ``GatewayController.wait_closed``.

    Set to 3 s. The close worker briefly transitions through
    IDLE → publish events → signal ``_close_done``; tests that
    poll ``is_busy`` immediately after calling ``wait_closed``
    can fall into the gap where state is IDLE but the done
    event hasn't been set yet, which would falsely time out at
    2 s. 3 s absorbs the worst-case event-bus publish latency
    in the mock pipeline path while still surfacing real hangs
    quickly.
    """
    return 3.0


# ---------------------------------------------------------------------------
# Adapter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> "EventBus":
    """A real EventBus for tests that exercise pub/sub semantics."""
    from larksnap.gateway.event_bus import EventBus
    return EventBus()


@pytest.fixture
def worker_pool() -> "WorkerPool":
    """A small WorkerPool for tests that exercise async dispatch."""
    from larksnap.utils.worker_pool import WorkerPool
    pool = WorkerPool(max_workers=2, queue_size=32, name_prefix="test")
    yield pool
    pool.shutdown(timeout=2.0)


@pytest.fixture
def notification_service(
    event_bus: "EventBus",
    worker_pool: "WorkerPool",
) -> tuple["NotificationService", "NotifierAdapter"]:
    """A ``(service, notifier)`` pair with a fresh EventBus and WorkerPool.

    The notifier is a ``MagicMock`` so the test can assert on
    ``notifier.send_message`` directly. Use ``target_classes`` on
    the config (returned alongside, or via a custom factory) to
    exercise the filter layer.
    """
    from larksnap.adapters.notifier.interface import NotifierAdapter
    from larksnap.gateway.notification_service import (
        NotificationService,
        NotificationServiceConfig,
    )
    notifier = MagicMock(spec=NotifierAdapter)
    notifier.send_message = MagicMock(return_value=True)
    svc = NotificationService(
        config=NotificationServiceConfig(),
        notifier=notifier,
        event_bus=event_bus,
        worker_pool=worker_pool,
    )
    return svc, notifier


@pytest.fixture
def snapshot_service(
    event_bus: "EventBus",
    tmp_path: pytest.TmpPathFactory,
) -> tuple["SnapshotService", "SnapshotServiceConfig"]:
    """A ``(service, config)`` pair with a fresh temp snapshot dir.

    The directory under ``tmp_path`` is wiped between tests, so the
    per-class throttling state is *not* persisted across tests. Use
    ``reset_cooldowns()`` if a test needs a clean cooldown table
    without re-creating the service.
    """
    from larksnap.gateway.snapshot_service import (
        SnapshotService,
        SnapshotServiceConfig,
    )
    cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path))
    svc = SnapshotService(config=cfg, event_bus=event_bus)
    return svc, cfg


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers with the warning silencer.

    Markers added in ``pyproject.toml`` are still surfaced in the
    deprecation warning the first time pytest sees them; registering
    them at runtime suppresses that warning and gives us a single
    place to enumerate them.
    """
    config.addinivalue_line(
        "markers", "slow: long-running tests (>1 s) excluded from the default suite"
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip ``slow``-marked tests unless the user opts in with ``-m slow``.

    The slow tests are still collected (so ``--collect-only`` sees them)
    and can be re-enabled with ``-m slow`` for nightly runs. This is
    the same pattern pytest uses for the built-in ``slow`` marker.
    """
    if config.getoption("-m", default="") in ("slow",):
        return
    skip_slow = pytest.mark.skip(
        reason="slow test; run with `pytest -m slow` to enable",
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
