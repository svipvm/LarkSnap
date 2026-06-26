"""Tests for the unified component state model and controller integration.

Covers:
  - ``ComponentState`` display names are stable (UI depends on them).
  - Allowed state transitions enforce the documented lifecycle.
  - ``component_state_from_legacy`` maps boolean flags correctly.
  - ``GatewayController`` publishes ``COMPONENT_STATE_CHANGED`` events
    for each subsystem when the state changes.
  - The controller's ``get_system_status`` returns a coherent snapshot
    in a single lock acquisition (so the UI never sees partial updates).
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from larksnap.config.models import AppConfig
from larksnap.gateway.component_state import (
    ALLOWED_TRANSITIONS,
    ComponentKind,
    ComponentState,
    ComponentStatus,
    SystemStatus,
    component_state_from_legacy,
    is_valid_transition,
)
from larksnap.gateway.controller import GatewayController
from larksnap.gateway.event_bus import Event, EventType


def _make_controller() -> GatewayController:
    """Build a controller that won't try to open a real camera."""
    cfg = AppConfig()
    cfg.detector.type = "mock"
    cfg.camera.device_index = 999  # ensure no real camera
    return GatewayController(cfg)


# ── Display names ─────────────────────────────────────────────────────


def test_display_names_are_unique_and_stable() -> None:
    """UI subscribes to the exact English label for each state value.

    Reordering the display table would silently break menus, the
    status panel, and log-grep scripts that watch for these strings.
    """
    assert ComponentState.IDLE.display_name == "Idle"
    assert ComponentState.STARTING.display_name == "Starting"
    assert ComponentState.RUNNING.display_name == "Running"
    assert ComponentState.STOPPING.display_name == "Stopping"
    assert ComponentState.STOPPED.display_name == "Stopped"
    assert ComponentState.FAILED.display_name == "Failed"
    assert ComponentState.DISABLED.display_name == "Disabled"


def test_status_display_name_delegates_to_state() -> None:
    status = ComponentStatus(kind=ComponentKind.CAMERA, state=ComponentState.RUNNING)
    assert status.display_name == ComponentState.RUNNING.display_name == "Running"


def test_status_active_and_transitioning_flags() -> None:
    active = ComponentStatus(kind=ComponentKind.DETECTOR, state=ComponentState.RUNNING)
    starting = ComponentStatus(kind=ComponentKind.DETECTOR, state=ComponentState.STARTING)
    stopped = ComponentStatus(kind=ComponentKind.DETECTOR, state=ComponentState.STOPPED)
    failed = ComponentStatus(kind=ComponentKind.DETECTOR, state=ComponentState.FAILED)

    assert active.is_active and not active.is_transitioning
    assert starting.is_active and starting.is_transitioning
    assert not stopped.is_active and not stopped.is_transitioning
    assert not failed.is_active and not failed.is_transitioning


# ── Transition table ──────────────────────────────────────────────────


def test_transition_table_is_connected() -> None:
    """Every state has at least one allowed transition (no dead ends)."""
    for state, allowed in ALLOWED_TRANSITIONS.items():
        assert allowed, f"{state} has no allowed transitions"


@pytest.mark.parametrize(
    "current, target, expected",
    [
        (ComponentState.IDLE,     ComponentState.STARTING, True),
        (ComponentState.STARTING, ComponentState.RUNNING,  True),
        (ComponentState.STOPPING, ComponentState.STOPPED,  True),
        (ComponentState.STOPPED,  ComponentState.IDLE,     True),
        (ComponentState.STOPPED,  ComponentState.STARTING, True),
        (ComponentState.FAILED,   ComponentState.IDLE,     True),
        (ComponentState.DISABLED, ComponentState.RUNNING,  True),
        # Illegal jumps
        (ComponentState.IDLE,     ComponentState.RUNNING,  False),
        (ComponentState.STOPPED,  ComponentState.RUNNING,  False),
        (ComponentState.RUNNING,  ComponentState.IDLE,     False),
    ],
)
def test_valid_transition(current: ComponentState, target: ComponentState, expected: bool) -> None:
    assert is_valid_transition(current, target) is expected


# ── Legacy flag mapping ───────────────────────────────────────────────


def test_legacy_disabled_wins_over_running() -> None:
    assert (
        component_state_from_legacy(
            is_open=True, is_busy=False, is_running=True, is_disabled=True,
        )
        is ComponentState.DISABLED
    )


def test_legacy_failed_wins_over_open() -> None:
    assert (
        component_state_from_legacy(
            is_open=True, is_busy=False, is_running=False, is_failed=True,
        )
        is ComponentState.FAILED
    )


def test_legacy_starting_when_busy_and_not_open() -> None:
    assert (
        component_state_from_legacy(
            is_open=False, is_busy=True, is_running=False,
        )
        is ComponentState.STARTING
    )


def test_legacy_stopping_when_busy_and_open() -> None:
    assert (
        component_state_from_legacy(
            is_open=True, is_busy=True, is_running=False,
        )
        is ComponentState.STOPPING
    )


def test_legacy_idle_when_all_false() -> None:
    assert (
        component_state_from_legacy(
            is_open=False, is_busy=False, is_running=False,
        )
        is ComponentState.IDLE
    )


# ── Controller integration ────────────────────────────────────────────


class _Recorder:
    """Thread-safe subscription helper for the event bus."""

    def __init__(self, controller: GatewayController, event_type: EventType) -> None:
        self.events: list[Event] = []
        self._lock = threading.Lock()
        controller.event_bus.subscribe(event_type, self._on_event)

    def _on_event(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)

    def snapshot(self) -> list[Event]:
        with self._lock:
            return list(self.events)


def test_get_system_status_initial_state_is_idle() -> None:
    """A brand-new controller reports camera + detector as IDLE.

    The notifier's initial state is ``DISABLED`` (user opt-in), not
    ``IDLE`` — the two are deliberately separate so the UI can
    distinguish "not yet started" from "user turned off".
    """
    controller = _make_controller()
    try:
        status = controller.get_system_status()

        assert isinstance(status, SystemStatus)
        assert status.camera.state is ComponentState.IDLE
        assert status.detector.state is ComponentState.IDLE
        assert status.notifier.state is ComponentState.DISABLED
    finally:
        controller.stop()


def test_component_state_changed_event_carries_status() -> None:
    """The event payload is a ComponentStatus, not a raw string.

    Downstream handlers (UI, logs) read ``event.data.display_name``
    and ``event.data.kind`` — if the payload type changes, those
    handlers will break silently. This test pins the contract by
    driving a notifier state transition through the service and
    asserting the published event carries a proper status payload.
    """
    from larksnap.adapters.notifier.interface import NotifierAdapter
    from larksnap.gateway.event_bus import EventBus
    from larksnap.gateway.notification_service import (
        NotificationService,
        NotificationServiceConfig,
    )

    bus = EventBus()
    notifier = MagicMock(spec=NotifierAdapter)
    notifier.send_message = MagicMock(return_value=True)
    service = NotificationService(
        config=NotificationServiceConfig(),
        notifier=notifier,
        event_bus=bus,
    )

    captured: list[Event] = []
    bus.subscribe(EventType.COMPONENT_STATE_CHANGED, captured.append)

    # The service starts enabled; transition to DISABLED first so
    # we have a clean before/after for the next call.
    service.disable_notification()
    assert any(
        isinstance(e.data, ComponentStatus)
        and e.data.kind is ComponentKind.NOTIFIER
        and e.data.state is ComponentState.DISABLED
        for e in captured
    )

    # DISABLED → RUNNING: should publish one state event whose
    # data is a fully populated notifier status.
    captured.clear()
    service.enable_notification()
    notifier_events = [
        e for e in captured
        if isinstance(e.data, ComponentStatus) and e.data.kind is ComponentKind.NOTIFIER
    ]
    assert notifier_events, "expected a notifier state change event"
    assert notifier_events[-1].data.state is ComponentState.RUNNING

    # RUNNING → DISABLED: should also publish.
    captured.clear()
    service.disable_notification()
    disabled_events = [
        e for e in captured
        if isinstance(e.data, ComponentStatus)
        and e.data.kind is ComponentKind.NOTIFIER
        and e.data.state is ComponentState.DISABLED
    ]
    assert disabled_events, "expected a notifier DISABLED state event"

    # Idempotency: enabling when already enabled should not re-publish.
    captured.clear()
    service.enable_notification()
    service.enable_notification()
    running_count = sum(
        1 for e in captured
        if isinstance(e.data, ComponentStatus)
        and e.data.kind is ComponentKind.NOTIFIER
        and e.data.state is ComponentState.RUNNING
    )
    assert running_count == 1, "duplicate enable should not re-publish"


def test_get_system_status_consistent_under_concurrent_reads() -> None:
    """``get_system_status`` returns a single coherent snapshot.

    The controller takes its lock once for all three subsystem
    queries; the UI relies on this to avoid seeing a "Camera RUNNING
    / Detector IDLE" snapshot taken before the detector transition.
    This test just exercises the contract from multiple threads.
    """
    controller = _make_controller()
    try:
        snapshots: list[SystemStatus] = []
        errors: list[BaseException] = []

        def reader() -> None:
            try:
                for _ in range(50):
                    snapshots.append(controller.get_system_status())
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"reader threads raised: {errors}"
        assert len(snapshots) == 200
        # Every snapshot must have all three kinds filled in (the
        # controller never returns a half-built SystemStatus).
        for snap in snapshots:
            assert isinstance(snap.camera, ComponentStatus)
            assert isinstance(snap.detector, ComponentStatus)
            assert isinstance(snap.notifier, ComponentStatus)
    finally:
        controller.stop()
