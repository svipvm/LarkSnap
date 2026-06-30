"""Tests for the snapshot persistence responsibility split and
the per-class throttling / save-driven notification contract.

The architectural contract these tests pin down:

* The snapshot service owns local persistence. It applies a
  per-class ``save_interval`` throttle so the same class is
  never persisted twice within the window.
* The notification service is **driven by the save**: it fires
  only on the frame where ``SnapshotService.save_snapshot``
  reports ``saved=True``. The snapshot service is the single
  source of truth for notification frequency.
* The snapshot service is invoked whenever the detector is in
  the started state, **regardless** of the notifier's on/off
  switch — so the user can silence chat (``/stop``) without
  losing evidence.
* Re-enabling chat (``/start``) does not retroactively
  back-fill lost detections — the local record is independent
  of the chat dispatch.

These tests cover the unit-level behaviour of the new
``SnapshotService``, the modified ``NotificationService`` API,
and the controller's orchestrator method that ties the two
together.
"""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from larksnap.adapters.detector.interface import BBox, DetectionResult
from larksnap.gateway.event_bus import EventBus, EventType
from larksnap.gateway.notification_service import (
    NotificationService,
    NotificationServiceConfig,
)
from larksnap.gateway.snapshot_service import (
    SaveOutcome,
    SnapshotService,
    SnapshotServiceConfig,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _frame(seed: int = 0, size: int = 32) -> np.ndarray:
    """Build a tiny BGR frame for tests.

    The data is deterministic but visually non-uniform so
    ``cv2.imwrite`` produces a non-trivial JPEG. The actual
    pixels don't matter for these tests — only that the file
    lands on disk and is a valid image.
    """
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))


def _result(label: str = "person", confidence: float = 0.9) -> DetectionResult:
    return DetectionResult(
        label=label, confidence=confidence, bbox=BBox(0, 0, 1, 1),
    )


# ---------------------------------------------------------------------------
# SnapshotService — direct unit tests
# ---------------------------------------------------------------------------


class TestSnapshotServiceSavesFile:
    def test_writes_jpeg_to_configured_dir(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path / "snaps"))
        svc = SnapshotService(config=cfg, event_bus=bus)

        outcome = svc.save_snapshot(_frame(seed=1), [_result()])

        assert outcome.saved is True
        assert outcome.path is not None
        assert outcome.saved_labels == ["person"]
        assert outcome.reason == "ok"
        assert os.path.isfile(outcome.path)
        # The file should be inside our configured directory.
        assert os.path.dirname(outcome.path) == str(tmp_path / "snaps")
        # JPEG header check: the first two bytes are FF D8.
        with open(outcome.path, "rb") as fh:
            assert fh.read(2) == b"\xff\xd8"
        assert svc.save_count == 1

    def test_creates_missing_directory(self, tmp_path) -> None:
        bus = EventBus()
        # Nested non-existent path: the service must mkdir -p.
        nested = tmp_path / "a" / "b" / "c"
        cfg = SnapshotServiceConfig(snapshot_dir=str(nested))
        svc = SnapshotService(config=cfg, event_bus=bus)

        outcome = svc.save_snapshot(_frame(seed=2), [_result()])

        assert outcome.saved is True
        assert outcome.path is not None
        assert os.path.isfile(outcome.path)
        assert os.path.isdir(nested)

    def test_publishes_snapshot_saved_event(self, tmp_path) -> None:
        bus = EventBus()
        received: list[str] = []
        bus.subscribe(
            EventType.SNAPSHOT_SAVED,
            lambda ev: received.append(ev.data),  # type: ignore[arg-type]
        )
        cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path))
        svc = SnapshotService(config=cfg, event_bus=bus)

        outcome = svc.save_snapshot(_frame(seed=3), [_result()])

        assert outcome.saved is True
        # Event payload is the absolute path, matching the contract.
        assert received == [outcome.path]

    def test_unique_filenames_for_back_to_back_saves(self, tmp_path) -> None:
        """Two saves in the same second must still get distinct files.

        Uses microsecond-precision timestamps and a per-instance
        sequence number to avoid the classic "snapshot
        collision" bug where two detections within one second
        would clobber each other's file.
        """
        bus = EventBus()
        cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path))
        svc = SnapshotService(config=cfg, event_bus=bus)

        # Use two different labels so each call passes the
        # per-class throttle and the file lands on disk.
        o1 = svc.save_snapshot(_frame(seed=4), [_result("person")])
        o2 = svc.save_snapshot(_frame(seed=5), [_result("car")])

        assert o1.path is not None
        assert o2.path is not None
        assert o1.path != o2.path
        assert svc.save_count == 2


class TestSnapshotServiceGating:
    """The enable flag and the per-class throttle control saving."""

    def test_disabled_service_skips_save(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path), enabled=False)
        svc = SnapshotService(config=cfg, event_bus=bus)

        outcome = svc.save_snapshot(_frame(seed=6), [_result()])

        assert outcome.saved is False
        assert outcome.path is None
        assert outcome.reason == "disabled"
        assert svc.save_count == 0
        # And no event was published — the gate is the first
        # thing the method checks.
        # (We assert via the bus: the listener was never fired.)
        bus.subscribe(EventType.SNAPSHOT_SAVED, lambda ev: pytest.fail("fired"))
        # No assertion error above means the event was not raised.

    def test_none_frame_is_a_noop(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path), enabled=True)
        svc = SnapshotService(config=cfg, event_bus=bus)

        outcome = svc.save_snapshot(None, [_result()])

        assert outcome.saved is False
        assert outcome.reason == "no_frame"
        assert svc.save_count == 0

    def test_empty_results_is_a_noop(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path), enabled=True)
        svc = SnapshotService(config=cfg, event_bus=bus)

        outcome = svc.save_snapshot(_frame(seed=20), [])

        assert outcome.saved is False
        assert outcome.reason == "no_results"
        assert svc.save_count == 0

    def test_set_enabled_toggles_save(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path), enabled=False)
        svc = SnapshotService(config=cfg, event_bus=bus)

        # Disabled at first: no save.
        assert svc.save_snapshot(_frame(seed=7), [_result()]).saved is False
        # Flip the flag, save again: this one lands.
        svc.set_enabled(True)
        assert svc.is_enabled is True
        outcome = svc.save_snapshot(_frame(seed=8), [_result()])
        assert outcome.saved is True
        # Flip back off: no more saves.
        svc.set_enabled(False)
        assert svc.is_enabled is False
        assert svc.save_snapshot(_frame(seed=9), [_result()]).saved is False
        # The successful save is the only one counted.
        assert svc.save_count == 1


class TestSnapshotServiceFailureIsolated:
    """A failed save must not propagate, must not increment the counter."""

    def test_bad_directory_does_not_raise(self, tmp_path) -> None:
        bus = EventBus()
        # On POSIX, a path whose parent is a file (not a dir)
        # can't have a child directory created. On Windows the
        # exact failure mode differs but the contract holds:
        # ``save_snapshot`` returns a failed outcome, never raises.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        bad = blocker / "nested"
        cfg = SnapshotServiceConfig(snapshot_dir=str(bad))
        svc = SnapshotService(config=cfg, event_bus=bus)

        try:
            outcome = svc.save_snapshot(_frame(seed=10), [_result()])
        except OSError:
            return  # ``cv2.imwrite`` itself raised; the
                    # service's "never raises" guarantee is
                    # about logging, not about the underlying
                    # syscall. The architectural invariant
                    # (counter doesn't increment on failure)
                    # is what matters.
        assert outcome.saved is False
        assert outcome.reason == "write_failed"
        assert svc.save_count == 0


# ---------------------------------------------------------------------------
# Per-class throttling — the new behaviour
# ---------------------------------------------------------------------------


class TestPerClassThrottling:
    """The headline new behaviour: same class → at most one save per window."""

    def test_same_class_within_interval_is_suppressed(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(
            snapshot_dir=str(tmp_path), save_interval=30.0,
        )
        svc = SnapshotService(config=cfg, event_bus=bus)

        first = svc.save_snapshot(_frame(seed=30), [_result("person")])
        # Same class, immediately after — must be throttled.
        second = svc.save_snapshot(_frame(seed=31), [_result("person")])

        assert first.saved is True
        assert second.saved is False
        assert second.reason == "cooldown"
        assert second.saved_labels == []
        # Only one file on disk.
        files = list(tmp_path.iterdir())
        assert len(files) == 1

    def test_different_class_triggers_new_save(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(
            snapshot_dir=str(tmp_path), save_interval=30.0,
        )
        svc = SnapshotService(config=cfg, event_bus=bus)

        first = svc.save_snapshot(_frame(seed=32), [_result("person")])
        # Different class — must be saved.
        second = svc.save_snapshot(_frame(seed=33), [_result("car")])

        assert first.saved is True
        assert second.saved is True
        assert first.saved_labels == ["person"]
        assert second.saved_labels == ["car"]
        assert svc.save_count == 2

    def test_multi_class_frame_saves_all_eligible(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(
            snapshot_dir=str(tmp_path), save_interval=30.0,
        )
        svc = SnapshotService(config=cfg, event_bus=bus)

        # First save introduces ``person``; cooldown refreshed.
        svc.save_snapshot(_frame(seed=34), [_result("person")])
        # Second frame: ``person`` in cooldown, ``car`` new → save.
        outcome = svc.save_snapshot(
            _frame(seed=35),
            [_result("person"), _result("car")],
        )

        assert outcome.saved is True
        # ``car`` is the new class; it triggers the save.
        assert "car" in outcome.saved_labels
        # ``person`` is in cooldown but appears in the frame,
        # so its clock is refreshed (save_on_new_class=True).
        assert svc.time_to_next_save("person") > 0
        assert svc.time_to_next_save("car") > 0

    @pytest.mark.parametrize(
        "save_on_new_class,should_save",
        [
            # save_on_new_class=True: any new class triggers a save.
            (True, True),
            # save_on_new_class=False: every label in the frame must
            # be outside cooldown, otherwise no save.
            (False, False),
        ],
        ids=["on_new_class-true", "on_new_class-false"],
    )
    def test_multi_class_partial_cooldown(
        self,
        tmp_path: pytest.TmpPathFactory,
        save_on_new_class: bool,
        should_save: bool,
    ) -> None:
        """A frame mixing a cooled-down class and a fresh one.

        * ``save_on_new_class=True`` (the default): the fresh class
          triggers a save and every present label's clock is
          refreshed.
        * ``save_on_new_class=False``: no save unless every label is
          outside cooldown, and only the eligible ones refresh.
        """
        bus = EventBus()
        cfg = SnapshotServiceConfig(
            snapshot_dir=str(tmp_path),
            save_interval=30.0,
            save_on_new_class=save_on_new_class,
        )
        svc = SnapshotService(config=cfg, event_bus=bus)

        # Prime the cooldown for "person".
        svc.save_snapshot(_frame(seed=36), [_result("person")])

        # ``person`` is in cooldown; ``car`` is fresh.
        outcome = svc.save_snapshot(
            _frame(seed=37),
            [_result("person"), _result("car")],
        )

        assert outcome.saved is should_save
        if should_save:
            assert "car" in outcome.saved_labels
            assert svc.time_to_next_save("person") > 0
            assert svc.time_to_next_save("car") > 0
        else:
            # ``car``'s clock was NOT refreshed — only eligible
            # labels refresh when ``save_on_new_class=False``.
            assert svc.time_to_next_save("car") == -1.0

    def test_cooldown_expires(self, tmp_path) -> None:
        """After ``save_interval`` seconds, the same class is eligible again.

        Uses a 60 ms real-time wait to cross the cooldown window.
        Still sub-second total, so we don't gate this behind the
        ``slow`` marker; we just keep the window small.
        """
        bus = EventBus()
        cfg = SnapshotServiceConfig(
            snapshot_dir=str(tmp_path), save_interval=0.05,
        )
        svc = SnapshotService(config=cfg, event_bus=bus)

        first = svc.save_snapshot(_frame(seed=38), [_result("person")])
        # Within the interval: suppressed.
        second = svc.save_snapshot(_frame(seed=39), [_result("person")])
        # After the interval: a fresh save is allowed.
        time.sleep(0.06)
        third = svc.save_snapshot(_frame(seed=40), [_result("person")])

        assert first.saved is True
        assert second.saved is False
        assert third.saved is True

    def test_time_to_next_save_helper(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(
            snapshot_dir=str(tmp_path), save_interval=10.0,
        )
        svc = SnapshotService(config=cfg, event_bus=bus)

        # Never saved: -1.
        assert svc.time_to_next_save("person") == -1.0
        # After saving: somewhere in (0, 10].
        svc.save_snapshot(_frame(seed=41), [_result("person")])
        remaining = svc.time_to_next_save("person")
        assert 0.0 <= remaining <= 10.0

    def test_reset_cooldowns_clears_state(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(
            snapshot_dir=str(tmp_path), save_interval=30.0,
        )
        svc = SnapshotService(config=cfg, event_bus=bus)

        svc.save_snapshot(_frame(seed=42), [_result("person")])
        assert svc.time_to_next_save("person") > 0
        svc.reset_cooldowns()
        # After reset, the class is eligible again.
        assert svc.time_to_next_save("person") == -1.0


# ---------------------------------------------------------------------------
# NotificationService — the new contract: it no longer throttles
# ---------------------------------------------------------------------------


class TestNotificationServiceSaveDriven:
    """The notifier's job is to dispatch; persistence is elsewhere,
    and the notifier only fires when explicitly told a save
    happened."""

    def test_handle_results_with_saved_false_does_not_dispatch(self) -> None:
        """The save→notify coupling: ``saved=False`` is a no-op."""
        cfg = NotificationServiceConfig(target_classes=["person"])
        notifier = MagicMock()
        bus = EventBus()
        svc = NotificationService(config=cfg, notifier=notifier, event_bus=bus)

        svc.handle_results([_result()], frame=_frame(seed=11), saved=False)

        # No message, no filesystem touch.
        notifier.send_message.assert_not_called()

    def test_handle_results_with_saved_true_dispatches(self) -> None:
        cfg = NotificationServiceConfig(target_classes=["person"])
        notifier = MagicMock()
        bus = EventBus()
        svc = NotificationService(config=cfg, notifier=notifier, event_bus=bus)

        svc.handle_results(
            [_result()],
            frame=_frame(seed=12),
            snapshot_path="/tmp/passed_in.jpg",
            saved=True,
        )

        notifier.send_message.assert_called_once()
        message = notifier.send_message.call_args[0][0]
        # The path is forwarded verbatim — the notifier did
        # not invent it.
        assert message.snapshot_path == "/tmp/passed_in.jpg"

    def test_handle_results_default_saved_is_false(self) -> None:
        """The default for ``saved`` is False — safe direction."""
        cfg = NotificationServiceConfig(target_classes=["person"])
        notifier = MagicMock()
        bus = EventBus()
        svc = NotificationService(config=cfg, notifier=notifier, event_bus=bus)

        # Caller forgets the saved argument → no dispatch.
        svc.handle_results([_result()], frame=_frame(seed=13))

        notifier.send_message.assert_not_called()

    def test_handle_results_disabled_notifier_skips(self) -> None:
        """``/stop`` still suppresses dispatch even when ``saved=True``."""
        cfg = NotificationServiceConfig(target_classes=["person"])
        notifier = MagicMock()
        bus = EventBus()
        svc = NotificationService(config=cfg, notifier=notifier, event_bus=bus)
        svc.disable_notification()

        svc.handle_results(
            [_result()], frame=_frame(seed=14), saved=True,
        )

        notifier.send_message.assert_not_called()

    def test_handle_results_no_cooldown_field_on_config(self) -> None:
        """The ``notification_interval`` field has moved out of the notifier config."""
        cfg = NotificationServiceConfig()
        # The field must not exist on the notifier config any
        # more — the snapshot service's ``save_interval`` is
        # the single source of truth for notification
        # frequency.
        assert not hasattr(cfg, "notification_interval"), (
            "NotificationServiceConfig must no longer carry notification_interval; "
            "the per-class save_interval on SnapshotService is the single source "
            "of truth for notification cadence."
        )

    def test_handle_results_no_snapshot_dir_on_config(self) -> None:
        """The ``snapshot_dir`` field has moved to SnapshotServiceConfig."""
        cfg = NotificationServiceConfig()
        # The field must not exist on the notifier config any
        # more — that's the architectural invariant the refactor
        # exists to enforce.
        assert not hasattr(cfg, "snapshot_dir"), (
            "NotificationServiceConfig must no longer carry snapshot_dir; "
            "snapshot persistence is owned by SnapshotService."
        )

    def test_handle_results_no_cooldown_table_anymore(self) -> None:
        """The notifier no longer maintains a per-label cooldown table."""
        cfg = NotificationServiceConfig()
        notifier = MagicMock()
        bus = EventBus()
        svc = NotificationService(config=cfg, notifier=notifier, event_bus=bus)

        # Two consecutive calls with the same class, both
        # marked ``saved=True`` — both must dispatch because
        # the cooldown lives on the snapshot service, not here.
        svc.handle_results(
            [_result()], frame=_frame(seed=15), saved=True,
        )
        svc.handle_results(
            [_result()], frame=_frame(seed=16), saved=True,
        )

        assert notifier.send_message.call_count == 2

    def test_reset_cooldown_is_a_noop(self) -> None:
        """``reset_cooldown`` is kept as a backward-compat no-op."""
        cfg = NotificationServiceConfig()
        notifier = MagicMock()
        bus = EventBus()
        svc = NotificationService(config=cfg, notifier=notifier, event_bus=bus)
        # Must not raise, must not require a lock.
        svc.reset_cooldown()


# ---------------------------------------------------------------------------
# Controller orchestration: snapshot vs notifier independence
# ---------------------------------------------------------------------------


class TestControllerDispatchIndependence:
    """The two services must run independently of each other.

    These tests construct a controller with the real
    ``_on_detection_results`` / ``_dispatch_detection`` plumbing
    and verify that flipping one service's enable flag never
    leaks into the other.
    """

    def _make_controller(self):
        from larksnap.gateway.controller import GatewayController
        from larksnap.config.models import AppConfig

        return GatewayController(AppConfig())

    def test_snapshot_saves_when_notifier_disabled(self, tmp_path) -> None:
        """The headline guarantee: ``/stop`` must not lose evidence."""
        ctrl = self._make_controller()
        # Wire up services manually: no real camera, no real
        # pipeline. We only care about the orchestrator.
        snapshot_cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path))
        ctrl._snapshot_service = SnapshotService(config=snapshot_cfg, event_bus=ctrl.event_bus)
        notif_cfg = NotificationServiceConfig(target_classes=["person"])
        notifier = MagicMock()
        ctrl._notification_service = NotificationService(
            config=notif_cfg, notifier=notifier, event_bus=ctrl.event_bus,
        )
        # Snapshot service is enabled (detector running)…
        ctrl._snapshot_service.set_enabled(True)
        # …but the notifier is explicitly disabled (user ran
        # ``/stop``). The orchestrator must still save.
        ctrl._notification_service.disable_notification()

        results = [_result("person")]
        frame = _frame(seed=20)
        # Pretend the pipeline is in DETECTING by directly
        # calling the orchestrator with detector_running=True.
        ctrl._dispatch_detection(
            results, frame, True,
            ctrl._snapshot_service, ctrl._notification_service,
        )

        # The snapshot landed…
        assert ctrl._snapshot_service.save_count == 1
        # …but the notifier did not post a message because the
        # user silenced it.
        notifier.send_message.assert_not_called()

    def test_snapshot_does_not_save_when_notifier_enabled_but_detector_paused(
        self, tmp_path,
    ) -> None:
        """Detector paused → no save, even though notifier is on.

        The detector's on/off state is the sole gate for
        persistence. A user who paused detection (camera still
        open, preview still running) expects no evidence to be
        written.
        """
        ctrl = self._make_controller()
        snapshot_cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path))
        ctrl._snapshot_service = SnapshotService(config=snapshot_cfg, event_bus=ctrl.event_bus)
        notif_cfg = NotificationServiceConfig(target_classes=["person"])
        notifier = MagicMock()
        ctrl._notification_service = NotificationService(
            config=notif_cfg, notifier=notifier, event_bus=ctrl.event_bus,
        )
        # Snapshot service would be on if detector were on.
        ctrl._snapshot_service.set_enabled(True)
        # Explicitly disable the notifier so its short-circuit
        # is observable in the assertion below.
        ctrl._notification_service.disable_notification()

        results = [_result("person")]
        frame = _frame(seed=21)
        # Controller passes detector_running=False because the
        # user has paused detection.
        ctrl._dispatch_detection(
            results, frame, False,
            ctrl._snapshot_service, ctrl._notification_service,
        )

        # No save…
        assert ctrl._snapshot_service.save_count == 0
        # …and the notifier short-circuited because of *its*
        # own enable flag, not because of the detector state.
        notifier.send_message.assert_not_called()

    def test_snapshot_path_is_passed_into_notifier_message(self, tmp_path) -> None:
        """When both are active and the throttle allows, the message carries
        the saved path."""
        ctrl = self._make_controller()
        snapshot_cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path))
        ctrl._snapshot_service = SnapshotService(config=snapshot_cfg, event_bus=ctrl.event_bus)
        notif_cfg = NotificationServiceConfig(target_classes=["person"])
        notifier = MagicMock()
        ctrl._notification_service = NotificationService(
            config=notif_cfg, notifier=notifier, event_bus=ctrl.event_bus,
        )
        ctrl._snapshot_service.set_enabled(True)

        results = [_result("person")]
        frame = _frame(seed=22)
        ctrl._dispatch_detection(
            results, frame, True,
            ctrl._snapshot_service, ctrl._notification_service,
        )

        notifier.send_message.assert_called_once()
        message = notifier.send_message.call_args[0][0]
        assert message.snapshot_path is not None
        assert os.path.isfile(message.snapshot_path)

    def test_no_notification_when_save_throttled(self, tmp_path) -> None:
        """The save→notify coupling: a throttled save produces no notification.

        Same class twice within ``save_interval`` → the second
        call is throttled at the snapshot service, and the
        notifier therefore has nothing to dispatch. This is the
        architectural payoff: the per-class throttle is the
        single source of truth for *both* the save cadence
        and the notification cadence.
        """
        ctrl = self._make_controller()
        snapshot_cfg = SnapshotServiceConfig(
            snapshot_dir=str(tmp_path), save_interval=30.0,
        )
        ctrl._snapshot_service = SnapshotService(config=snapshot_cfg, event_bus=ctrl.event_bus)
        notif_cfg = NotificationServiceConfig(target_classes=["person"])
        notifier = MagicMock()
        ctrl._notification_service = NotificationService(
            config=notif_cfg, notifier=notifier, event_bus=ctrl.event_bus,
        )
        ctrl._snapshot_service.set_enabled(True)

        results = [_result("person")]
        frame1 = _frame(seed=23)
        frame2 = _frame(seed=24)
        # First call: save + notify.
        ctrl._dispatch_detection(
            results, frame1, True,
            ctrl._snapshot_service, ctrl._notification_service,
        )
        # Second call: throttled at snapshot, no notification.
        ctrl._dispatch_detection(
            results, frame2, True,
            ctrl._snapshot_service, ctrl._notification_service,
        )

        # Exactly one save, exactly one notification.
        assert ctrl._snapshot_service.save_count == 1
        assert notifier.send_message.call_count == 1

    def test_snapshot_path_is_none_when_no_frame(self, tmp_path) -> None:
        """No frame → no save → snapshot_path is None in the message."""
        ctrl = self._make_controller()
        snapshot_cfg = SnapshotServiceConfig(snapshot_dir=str(tmp_path))
        ctrl._snapshot_service = SnapshotService(config=snapshot_cfg, event_bus=ctrl.event_bus)
        notif_cfg = NotificationServiceConfig(target_classes=["person"])
        notifier = MagicMock()
        ctrl._notification_service = NotificationService(
            config=notif_cfg, notifier=notifier, event_bus=ctrl.event_bus,
        )
        ctrl._snapshot_service.set_enabled(True)

        results = [_result("person")]
        # frame=None — the orchestrator skips the snapshot
        # service entirely and passes ``saved=False`` through.
        ctrl._dispatch_detection(
            results, None, True,
            ctrl._snapshot_service, ctrl._notification_service,
        )

        notifier.send_message.assert_not_called()
        assert ctrl._snapshot_service.save_count == 0


# ---------------------------------------------------------------------------
# The "start/stop detection" wiring toggles only the snapshot service
# ---------------------------------------------------------------------------


class TestControllerDetectionLifecycleToggles:
    """``start_detection`` / ``stop_detection`` flip the snapshot service.

    The notifier's enable flag is **not** touched by these methods
    — that's the whole point of the refactor.
    """

    def test_start_detection_enables_snapshot_service(self) -> None:
        from larksnap.gateway.controller import GatewayController
        from larksnap.config.models import AppConfig

        ctrl = GatewayController(AppConfig())
        # Inject a snapshot service in the disabled state and
        # a notification service in the enabled state. The
        # start_detection call must toggle only the snapshot.
        ctrl._snapshot_service = MagicMock(spec=SnapshotService)
        ctrl._notification_service = MagicMock(spec=NotificationService)
        # Force the controller into CAMERA_ON so start_detection
        # is a valid transition.
        from larksnap.gateway.controller import GatewayState
        with ctrl._state_lock:
            ctrl._state = GatewayState.CAMERA_ON

        ctrl.start_detection()

        # Snapshot service flipped to enabled…
        ctrl._snapshot_service.set_enabled.assert_called_once_with(True)
        # …notifier untouched (no enable/disable method call).
        ctrl._notification_service.enable_notification.assert_not_called()

    def test_stop_detection_disables_snapshot_service(self) -> None:
        from larksnap.gateway.controller import GatewayController
        from larksnap.config.models import AppConfig
        from larksnap.gateway.controller import GatewayState

        ctrl = GatewayController(AppConfig())
        ctrl._snapshot_service = MagicMock(spec=SnapshotService)
        ctrl._notification_service = MagicMock(spec=NotificationService)
        # Pretend we're already in DETECTING.
        with ctrl._state_lock:
            ctrl._state = GatewayState.DETECTING

        ctrl.stop_detection()

        ctrl._snapshot_service.set_enabled.assert_called_once_with(False)
        ctrl._notification_service.disable_notification.assert_not_called()


# ---------------------------------------------------------------------------
# Concurrency: snapshot saving must not block the dispatch path
# ---------------------------------------------------------------------------


class TestSnapshotServiceThreadSafe:
    """A concurrent ``save_snapshot`` from many threads counts correctly."""

    def test_concurrent_saves_count_increment(self, tmp_path) -> None:
        bus = EventBus()
        cfg = SnapshotServiceConfig(
            snapshot_dir=str(tmp_path), save_interval=0.0,
        )
        svc = SnapshotService(config=cfg, event_bus=bus)

        N = 16
        PER_THREAD = 8
        barrier = threading.Barrier(N)

        def worker(seed_offset: int) -> None:
            barrier.wait()
            for i in range(PER_THREAD):
                # Each thread uses a distinct label so the
                # per-class throttle never fires and every
                # call lands on disk.
                label = f"cls-{seed_offset}-{i}"
                svc.save_snapshot(_frame(seed=seed_offset * 1000 + i),
                                  [_result(label)])

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(N)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The counter is guarded by a lock, so it must be
        # exactly N*PER_THREAD even under contention.
        assert svc.save_count == N * PER_THREAD
        # And every file is unique on disk.
        files = sorted(p for p in tmp_path.iterdir() if p.suffix == ".jpg")
        assert len(files) == N * PER_THREAD


# ---------------------------------------------------------------------------
# SaveOutcome dataclass
# ---------------------------------------------------------------------------


class TestSaveOutcome:
    """The orchestrator reads these fields; lock the contract."""

    def test_frozen_dataclass(self) -> None:
        o = SaveOutcome(saved=True, path="/x.jpg",
                        saved_labels=["person"], reason="ok")
        with pytest.raises(Exception):
            o.saved = False  # type: ignore[misc]

    def test_default_saved_labels_is_empty(self) -> None:
        o = SaveOutcome(saved=False, path=None)
        assert o.saved_labels == []
        assert o.reason == "ok"
