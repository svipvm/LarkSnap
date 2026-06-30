"""Tests for the production-grade controller state machine and lifecycle.

Covers:
  * GatewayState transitions (legal and illegal)
  * Primary fix: open_camera() now starts the pipeline immediately
  * Non-blocking close_camera() (returns fast)
  * Event subscription dedup
  * Idempotent operations
  * Runtime state persistence across close/open

The shared ``install_mock_adapters`` and ``close_timeout`` fixtures in
``conftest.py`` replace the dozen ad-hoc ``_install`` lambdas that
previously opened every test, so each test reads as just the behaviour
under test.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from larksnap.gateway.controller import (
    GatewayController,
    GatewayState,
    _ALLOWED_TRANSITIONS,
)
from larksnap.gateway.event_bus import Event, EventType

if TYPE_CHECKING:
    from larksnap.gateway.event_bus import EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_idle(controller: GatewayController, close_timeout: float) -> None:
    """Wait for the close worker and assert the controller is back to IDLE.

    Pulled out as a helper because every lifecycle test ends with the
    same three lines: ``close → wait_closed → assert IDLE``. Keeping
    the shared body in one place makes the actual contract being
    tested in each test stand out.
    """
    assert controller.wait_closed(timeout=close_timeout), (
        f"close did not finish within {close_timeout}s "
        f"(state={controller.state})"
    )
    assert controller.state == GatewayState.IDLE


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    """Pure state-table checks. No controller, no I/O — fast."""

    def test_initial_state_is_idle(
        self, gateway_controller: GatewayController,
    ) -> None:
        """A brand-new controller must be in IDLE."""
        assert gateway_controller.state == GatewayState.IDLE
        assert gateway_controller.is_camera_open is False
        assert gateway_controller.is_running is False
        assert gateway_controller.is_detection_active is False
        assert gateway_controller.is_busy is False

    @pytest.mark.parametrize(
        "src,dst,legal",
        [
            # Legal forward transitions
            (GatewayState.IDLE,     GatewayState.OPENING,  True),
            (GatewayState.OPENING,  GatewayState.CAMERA_ON, True),
            (GatewayState.CAMERA_ON, GatewayState.DETECTING, True),
            (GatewayState.CAMERA_ON, GatewayState.CLOSING,  True),
            (GatewayState.DETECTING, GatewayState.CAMERA_ON, True),
            (GatewayState.CLOSING,  GatewayState.IDLE,      True),
            # Illegal jumps the controller must reject
            (GatewayState.IDLE,     GatewayState.DETECTING, False),
            (GatewayState.IDLE,     GatewayState.CAMERA_ON, False),
            (GatewayState.CAMERA_ON, GatewayState.IDLE,     False),
            (GatewayState.DETECTING, GatewayState.IDLE,     False),
        ],
    )
    def test_allowed_transition_table(
        self, src: GatewayState, dst: GatewayState, legal: bool,
    ) -> None:
        assert (dst in _ALLOWED_TRANSITIONS[src]) is legal


# ---------------------------------------------------------------------------
# Open / close lifecycle
# ---------------------------------------------------------------------------


class TestOpenCloseLifecycle:
    """Open → use → close. Each test exercises one slice of the path."""

    def test_open_starts_pipeline_immediately(
        self,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        """Primary fix: opening the camera starts the pipeline right away.

        Before the fix the pipeline was only *created* (not started) by
        open_camera(), so users in CAMERA_ON state saw no frames.
        """
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()
        try:
            assert gateway_controller.state == GatewayState.CAMERA_ON
            assert gateway_controller._pipeline is not None
            assert gateway_controller._pipeline.is_running
            assert gateway_controller.is_camera_open is True
        finally:
            gateway_controller.close_camera()
            _assert_idle(gateway_controller, close_timeout)

    def test_close_camera_is_non_blocking(
        self,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        """close_camera() must return in milliseconds, even with slow teardown."""
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()
        assert gateway_controller.is_camera_open

        t0 = time.monotonic()
        gateway_controller.close_camera()
        elapsed = time.monotonic() - t0

        # close_camera() should return in well under a second; the
        # actual ZMQ teardown happens in a background thread.
        assert elapsed < 1.0, f"close_camera() blocked for {elapsed:.2f}s"
        assert gateway_controller.state in (
            GatewayState.CLOSING, GatewayState.IDLE,
        )
        _assert_idle(gateway_controller, close_timeout)

    def test_double_open_is_noop(
        self,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        """A second open_camera() while running must not change state or crash."""
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()
        gateway_controller.open_camera()  # must be a no-op
        assert gateway_controller.state == GatewayState.CAMERA_ON
        gateway_controller.close_camera()
        _assert_idle(gateway_controller, close_timeout)

    def test_open_while_closing_is_rejected(
        self,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        """open_camera() during an in-flight close must not crash.

        The state is expected to remain in CLOSING (or already IDLE if
        the close worker finished by then). Either outcome is fine —
        what matters is that the second open is silently rejected
        rather than corrupting the state machine.
        """
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()
        gateway_controller.close_camera()
        gateway_controller.open_camera()  # must be rejected
        assert gateway_controller.state in (
            GatewayState.CLOSING, GatewayState.IDLE,
        )
        _assert_idle(gateway_controller, close_timeout)

    def test_close_during_close_is_safe(
        self,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        """Two close_camera() calls back-to-back must not crash."""
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()
        gateway_controller.close_camera()
        gateway_controller.close_camera()  # no-op
        _assert_idle(gateway_controller, close_timeout)

    def test_is_busy_clears_after_close(
        self,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        """is_busy must be False once the close worker has finished."""
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()
        assert gateway_controller.is_busy is False
        gateway_controller.close_camera()
        _assert_idle(gateway_controller, close_timeout)
        assert gateway_controller.is_busy is False

    def test_recover_from_fatal_pipeline_error(
        self,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        """A fatal pipeline error must take the gateway back to IDLE,
        and the user must be able to re-open after.
        """
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()
        gateway_controller.start_detection()
        assert gateway_controller.state == GatewayState.DETECTING

        # Simulate a fatal pipeline error (as would come from
        # FrameProducer via the event bus).
        gateway_controller._on_pipeline_fatal_error(
            Event(type=EventType.CAMERA_READ_FAILED, data={"error": "test"}),
        )
        _assert_idle(gateway_controller, close_timeout)
        assert gateway_controller.is_camera_open is False

        # Must be able to re-open.
        gateway_controller.open_camera()
        assert gateway_controller.is_camera_open
        gateway_controller.close_camera()
        _assert_idle(gateway_controller, close_timeout)


# ---------------------------------------------------------------------------
# Detection lifecycle
# ---------------------------------------------------------------------------


class TestDetectionLifecycle:

    def test_start_stop_detection_state_transitions(
        self,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()
        assert gateway_controller.state == GatewayState.CAMERA_ON

        gateway_controller.start_detection()
        assert gateway_controller.state == GatewayState.DETECTING
        assert gateway_controller.is_running is True

        gateway_controller.stop_detection()
        assert gateway_controller.state == GatewayState.CAMERA_ON
        assert gateway_controller.is_running is False

        gateway_controller.close_camera()
        _assert_idle(gateway_controller, close_timeout)


# ---------------------------------------------------------------------------
# Event subscription
# ---------------------------------------------------------------------------


class TestEventSubscription:

    def test_handler_dedup_across_reopen(
        self,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        """Repeated subscribe() calls for the same handler must not stack up.

        If a user opens/closes the camera repeatedly, the same handler
        would otherwise be registered N times and called N times per
        event — a quiet but nasty leak.
        """
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()

        bus: EventBus = gateway_controller.event_bus
        handlers_after_open = len(
            bus._handlers.get(EventType.CAMERA_READ_FAILED, []),
        )
        assert handlers_after_open > 0, "expected at least one handler"

        gateway_controller.close_camera()
        _assert_idle(gateway_controller, close_timeout)

        # Re-open and confirm the handler count didn't grow.
        gateway_controller.open_camera()
        handlers_after_reopen = len(
            bus._handlers.get(EventType.CAMERA_READ_FAILED, []),
        )
        assert handlers_after_reopen == handlers_after_open, (
            f"Handler duplicated: {handlers_after_open} → {handlers_after_reopen}"
        )
        gateway_controller.close_camera()
        _assert_idle(gateway_controller, close_timeout)


# ---------------------------------------------------------------------------
# Runtime state persistence across close/open
# ---------------------------------------------------------------------------


class TestRuntimeStatePersistence:

    def test_state_persisted_on_close_and_restored_on_open(
        self,
        tmp_path: pytest.TmpPathFactory,
        gateway_controller: GatewayController,
        install_mock_adapters: Callable[[GatewayController], None],
        close_timeout: float,
    ) -> None:
        """Detector + notifier state must survive a close/open cycle.

        User-facing behaviour: closing the camera while detection is
        active and notification is disabled, then re-opening, must
        bring the system back to that exact state — not the
        "first-run" defaults.
        """
        from larksnap.gateway.notification_service import (
            NotificationService,
            NotificationServiceConfig,
        )

        # Redirect persistence to a per-test location.
        gateway_controller._state_path = tmp_path / "runtime_state.json"
        install_mock_adapters(gateway_controller)

        gateway_controller.open_camera()
        gateway_controller.start_detection()
        assert gateway_controller.state == GatewayState.DETECTING

        # Install a real NotificationService so we can flip its
        # enabled flag and observe the restored value.
        notif = NotificationService(
            config=NotificationServiceConfig(
                message_template="[LarkSnap] {label}",
            ),
            notifier=gateway_controller._notifier,
            event_bus=gateway_controller._event_bus,
            worker_pool=None,
        )
        notif.disable_notification()
        gateway_controller._notification_service = notif

        # Close → capture state, release adapters, go IDLE.
        gateway_controller.close_camera()
        _assert_idle(gateway_controller, close_timeout)

        # Re-open and confirm the detector + notifier state is restored.
        install_mock_adapters(gateway_controller)
        gateway_controller.open_camera()
        try:
            assert gateway_controller.is_detection_active is True
            assert (
                gateway_controller._notification_service.is_notification_enabled
                is False
            )
        finally:
            gateway_controller.close_camera()
            _assert_idle(gateway_controller, close_timeout)
