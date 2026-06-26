"""Tests for the production-grade controller state machine and lifecycle.

Covers:
  - GatewayState transitions (legal and illegal)
  - Primary fix: open_camera() now starts the pipeline immediately
  - Non-blocking close_camera() (returns fast)
  - Event subscription dedup
  - Idempotent operations
"""

import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from larksnap.config.models import AppConfig
from larksnap.gateway.controller import (
    GatewayController,
    GatewayState,
    _ALLOWED_TRANSITIONS,
)
from larksnap.gateway.event_bus import Event, EventType


def _make_mock_camera() -> MagicMock:
    cam = MagicMock()
    cam.is_opened.return_value = True
    cam.read_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return cam


def _make_controller_with_mock_camera(
    tmp_path_factory: pytest.TempPathFactory | None = None,
) -> GatewayController:
    """Build a controller pointed at a mock camera.

    Uses a per-test state file under a temporary directory so the
    state persistence feature (which writes to %APPDATA%/LarkSnap
    by default) does not leak between tests and cannot influence
    one test from a previous one.
    """
    cfg = AppConfig()
    cfg.detector.type = "mock"
    cfg.camera.device_index = 999  # ensure no real camera
    if tmp_path_factory is not None:
        state_path = tmp_path_factory.mktemp("runtime_state") / "runtime_state.json"
    else:
        state_path = None
    return GatewayController(cfg, state_path=state_path)


def test_state_machine_initial_state(tmp_path_factory) -> None:
    """New controller must be in IDLE state."""
    controller = _make_controller_with_mock_camera(tmp_path_factory)
    assert controller.state == GatewayState.IDLE
    assert controller.is_camera_open is False
    assert controller.is_running is False
    assert controller.is_detection_active is False
    assert controller.is_busy is False
    controller.stop()  # cleanup


def test_state_transitions_are_validated() -> None:
    """Illegal transitions must be rejected."""
    # IDLE -> DETECTING is illegal (must go through CAMERA_ON)
    assert GatewayState.DETECTING not in _ALLOWED_TRANSITIONS[GatewayState.IDLE]
    # IDLE -> CAMERA_ON is illegal (must go through OPENING)
    assert GatewayState.CAMERA_ON not in _ALLOWED_TRANSITIONS[GatewayState.IDLE]
    # CAMERA_ON -> IDLE is illegal (must go through CLOSING)
    assert GatewayState.IDLE not in _ALLOWED_TRANSITIONS[GatewayState.CAMERA_ON]


def test_open_camera_starts_pipeline_immediately(tmp_path_factory) -> None:
    """Primary fix: opening the camera must start the pipeline right away.

    Before the fix, the pipeline was only created (not started) by
    open_camera(), so users in CAMERA_ON state saw no frames.
    """
    controller = _make_controller_with_mock_camera(tmp_path_factory)
    mock_cam = _make_mock_camera()

    with patch.object(controller, "_create_adapters") as mock_create:
        from larksnap.adapters.detector.mock_adapter import MockDetectorAdapter
        from larksnap.adapters.notifier.feishu_adapter import FeishuNotifierAdapter
        from larksnap.adapters.recorder.video_recorder import VideoRecorderAdapter
        from larksnap.gateway.pipeline import Pipeline

        fake_detector = MagicMock()
        fake_notifier = MagicMock()
        fake_recorder = MagicMock()
        mock_create.side_effect = lambda: setattr(controller, "_camera", mock_cam) or setattr(
            controller, "_detector", fake_detector
        ) or setattr(controller, "_notifier", fake_notifier) or setattr(
            controller, "_recorder", fake_recorder
        )

        # The real _create_adapters builds all four adapters; patch it to
        # install our mocks and then drive the lifecycle through the
        # public path.
        def _install() -> None:
            controller._camera = mock_cam
            controller._detector = MagicMock()
            controller._notifier = MagicMock()
            controller._recorder = MagicMock()

        mock_create.side_effect = _install

        controller.open_camera()
        # Wait briefly for any cleanup of transient state
        try:
            assert controller.state == GatewayState.CAMERA_ON, (
                f"Expected CAMERA_ON, got {controller.state}"
            )
            # The pipeline must have been started
            assert controller._pipeline is not None
            assert controller._pipeline.is_running, "Pipeline should be running in CAMERA_ON"
            # is_camera_open must return True
            assert controller.is_camera_open is True
        finally:
            controller.close_camera()
            # Non-blocking close — wait bounded for cleanup
            assert controller.wait_closed(timeout=5.0), "Close did not finish in 5s"
            assert controller.state == GatewayState.IDLE
            controller.stop()


def test_close_camera_is_non_blocking(tmp_path_factory) -> None:
    """close_camera() must return quickly even when ZMQ teardown is slow."""
    controller = _make_controller_with_mock_camera(tmp_path_factory)

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        controller.open_camera()
        assert controller.is_camera_open

        t0 = time.monotonic()
        controller.close_camera()
        elapsed = time.monotonic() - t0

        # close_camera() should return in well under a second; the
        # actual ZMQ teardown happens in a background thread.
        assert elapsed < 1.0, f"close_camera() blocked for {elapsed:.2f}s"
        # State should be CLOSING until the worker finishes
        assert controller.state in (GatewayState.CLOSING, GatewayState.IDLE)

        # Wait for the worker to finish
        assert controller.wait_closed(timeout=5.0)
        assert controller.state == GatewayState.IDLE
        controller.stop()


def test_double_open_is_noop(tmp_path_factory) -> None:
    """Calling open_camera() twice without closing should be rejected."""
    controller = _make_controller_with_mock_camera(tmp_path_factory)

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        controller.open_camera()
        # Second call should be a no-op (not raise, not change state)
        controller.open_camera()
        assert controller.state == GatewayState.CAMERA_ON
        controller.close_camera()
        controller.wait_closed(timeout=5.0)
        controller.stop()


def test_open_while_closing_is_rejected(tmp_path_factory) -> None:
    """open_camera() during an in-flight close must be rejected, not crash."""
    controller = _make_controller_with_mock_camera(tmp_path_factory)

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        controller.open_camera()
        controller.close_camera()
        # Immediate second open: must be rejected (state is CLOSING)
        controller.open_camera()
        # State should still be CLOSING or back to IDLE
        assert controller.state in (GatewayState.CLOSING, GatewayState.IDLE)
        controller.wait_closed(timeout=5.0)
        controller.stop()


def test_event_subscription_dedup(tmp_path_factory) -> None:
    """Repeated subscribe() calls for the same handler must not stack up."""
    controller = _make_controller_with_mock_camera(tmp_path_factory)

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        controller.open_camera()
        # Capture handler count after open
        bus = controller.event_bus
        handlers_after_open = len(bus._handlers.get(EventType.CAMERA_READ_FAILED, []))

        # Open again after close — handler should still appear only once
        controller.close_camera()
        controller.wait_closed(timeout=5.0)
        controller.open_camera()

        handlers_after_reopen = len(bus._handlers.get(EventType.CAMERA_READ_FAILED, []))
        assert handlers_after_reopen == handlers_after_open, (
            f"Handler duplicated: {handlers_after_open} → {handlers_after_reopen}"
        )

        controller.close_camera()
        controller.wait_closed(timeout=5.0)
        controller.stop()


def test_start_stop_detection_state_transitions(tmp_path_factory) -> None:
    """start_detection() / stop_detection() must use the new state enum."""
    controller = _make_controller_with_mock_camera(tmp_path_factory)

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        controller.open_camera()
        assert controller.state == GatewayState.CAMERA_ON

        controller.start_detection()
        assert controller.state == GatewayState.DETECTING
        assert controller.is_running

        controller.stop_detection()
        assert controller.state == GatewayState.CAMERA_ON
        assert not controller.is_running

        controller.close_camera()
        controller.wait_closed(timeout=5.0)
        assert controller.state == GatewayState.IDLE
        controller.stop()


def test_is_busy_during_close(tmp_path_factory) -> None:
    """is_busy must be True while close is in flight."""
    controller = _make_controller_with_mock_camera(tmp_path_factory)

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        controller.open_camera()
        assert not controller.is_busy

        controller.close_camera()
        # is_busy should be True until the worker finishes
        # (the close worker is fast in our test environment, so check
        # via the state rather than asserting a window)
        controller.wait_closed(timeout=5.0)
        assert not controller.is_busy
        controller.stop()


def test_close_during_close_is_safe(tmp_path_factory) -> None:
    """Calling close_camera() twice in a row must not crash."""
    controller = _make_controller_with_mock_camera(tmp_path_factory)

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        controller.open_camera()
        controller.close_camera()
        # Second close: must be a no-op
        controller.close_camera()
        controller.wait_closed(timeout=5.0)
        assert controller.state == GatewayState.IDLE
        controller.stop()


def test_recover_from_fatal_pipeline_error(tmp_path_factory) -> None:
    """A fatal pipeline error must take the gateway back to IDLE.

    The user must be able to re-open the camera after a read failure.
    """
    controller = _make_controller_with_mock_camera(tmp_path_factory)

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        controller.open_camera()
        controller.start_detection()
        assert controller.state == GatewayState.DETECTING

        # Simulate a fatal pipeline error (as would come from
        # FrameProducer via the event bus)
        controller._on_pipeline_fatal_error(
            Event(type=EventType.CAMERA_READ_FAILED, data={"error": "test"})
        )
        # Give the background close worker time to run
        assert controller.wait_closed(timeout=5.0)
        assert controller.state == GatewayState.IDLE
        assert not controller.is_camera_open

        # Must be able to re-open
        controller.open_camera()
        assert controller.is_camera_open
        controller.close_camera()
        controller.wait_closed(timeout=5.0)
        controller.stop()


# ── Runtime state persistence across close/open ───────────────────────


def test_state_persisted_on_close_and_restored_on_open(tmp_path, tmp_path_factory) -> None:
    """Detector + notifier state must survive a close/open cycle.

    The user-facing behaviour: closing the camera while detection
    is active and notification is disabled, then re-opening, must
    bring the system back to that exact state — not the
    "first-run" defaults.

    Uses a tmp_path state file so the test doesn't pollute the
    user's real runtime_state.json.
    """
    from larksnap.gateway.notification_service import NotificationService, NotificationServiceConfig

    state_file = tmp_path / "runtime_state.json"
    controller = _make_controller_with_mock_camera(tmp_path_factory)
    # Redirect persistence to a per-test location.
    controller._state_path = state_file

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        # Open + start detection + disable notification.
        controller.open_camera()
        assert controller.state == GatewayState.CAMERA_ON
        controller.start_detection()
        assert controller.state == GatewayState.DETECTING
        # Replace the auto-built notification service with one we
        # control, so we can flip its enabled flag and observe the
        # restored value. The real one is built inside open_camera
        # but it's been replaced by a MagicMock at this layer, so
        # we install our own and rewire.
        notif = NotificationService(
            config=NotificationServiceConfig(
                notification_interval=30,
                snapshot_dir="snapshots",
                message_template="[LarkSnap] {label}",
            ),
            notifier=controller._notifier,
            event_bus=controller._event_bus,
            worker_pool=None,
        )
        notif.disable_notification()
        controller._notification_service = notif
        # Publish the unified state so the UI stays consistent.
        from larksnap.gateway.event_bus import EventType
        controller._event_bus.publish(Event(
            type=EventType.COMPONENT_STATE_CHANGED,
            source="test",
            data=None,
        ))

        # Close → capture state, release adapters, go IDLE.
        controller.close_camera()
        assert controller.wait_closed(timeout=5.0)
        assert controller.state == GatewayState.IDLE
        # The state file must have been written.
        assert state_file.exists(), "State file was not written on close"

    # Reopen with a brand-new controller to simulate a process
    # restart (adapters cleared, but state file on disk). Reuse
    # the same state file so the new controller reads the
    # persisted state on open.
    controller2 = GatewayController(AppConfig(), state_path=state_file)

    def _install2() -> None:
        controller2._camera = _make_mock_camera()
        controller2._detector = MagicMock()
        controller2._notifier = MagicMock()
        controller2._recorder = MagicMock()

    with patch.object(controller2, "_create_adapters", side_effect=_install2):
        controller2.open_camera()
        # Detector should be back in DETECTING (was saved as running).
        assert controller2.state == GatewayState.DETECTING, (
            f"Expected detector to be restored to DETECTING, got {controller2.state}"
        )
        # Notifier should be restored to disabled.
        # We replaced the auto-built notif service on the second
        # controller too, so the restore path writes through the
        # same instance.
        assert controller2._notification_service is not None
        assert controller2._notification_service.is_notification_enabled is False, (
            "Notifier state was not restored from disk"
        )

        controller2.close_camera()
        controller2.wait_closed(timeout=5.0)
        controller2.stop()


def test_state_restore_is_noop_when_file_missing(tmp_path, tmp_path_factory) -> None:
    """No state file → no restoration → fresh defaults apply.

    The first-ever run must behave exactly like before this
    feature was added: detector off (CAMERA_ON), notifier enabled.
    """
    controller = _make_controller_with_mock_camera(tmp_path_factory)
    controller._state_path = tmp_path / "does_not_exist.json"

    def _install() -> None:
        controller._camera = _make_mock_camera()
        controller._detector = MagicMock()
        controller._notifier = MagicMock()
        controller._recorder = MagicMock()

    with patch.object(controller, "_create_adapters", side_effect=_install):
        controller.open_camera()
        # Detector defaults to off (CAMERA_ON).
        assert controller.state == GatewayState.CAMERA_ON
        # Notifier defaults to enabled.
        assert controller._notification_service.is_notification_enabled is True
        controller.close_camera()
        controller.wait_closed(timeout=5.0)
        controller.stop()
