import pytest

from larksnap.config.models import AppConfig
from larksnap.gateway.controller import GatewayController
from larksnap.gateway.event_bus import Event, EventBus, EventType


class TestEventBus:
    def test_subscribe_and_publish(self) -> None:
        bus = EventBus()
        received = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.SYSTEM_STARTED, handler)
        bus.publish(Event(type=EventType.SYSTEM_STARTED, data="test"))

        assert len(received) == 1
        assert received[0].data == "test"

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        received = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.SYSTEM_STARTED, handler)
        bus.unsubscribe(EventType.SYSTEM_STARTED, handler)
        bus.publish(Event(type=EventType.SYSTEM_STARTED))

        assert len(received) == 0

    def test_multiple_handlers(self) -> None:
        bus = EventBus()
        count = [0, 0]

        def handler1(event: Event) -> None:
            count[0] += 1

        def handler2(event: Event) -> None:
            count[1] += 1

        bus.subscribe(EventType.SYSTEM_STARTED, handler1)
        bus.subscribe(EventType.SYSTEM_STARTED, handler2)
        bus.publish(Event(type=EventType.SYSTEM_STARTED))

        assert count[0] == 1
        assert count[1] == 1

    def test_handler_error_does_not_break_others(self) -> None:
        bus = EventBus()
        count = [0]

        def bad_handler(event: Event) -> None:
            raise ValueError("test error")

        def good_handler(event: Event) -> None:
            count[0] += 1

        bus.subscribe(EventType.SYSTEM_STARTED, bad_handler)
        bus.subscribe(EventType.SYSTEM_STARTED, good_handler)
        bus.publish(Event(type=EventType.SYSTEM_STARTED))

        assert count[0] == 1

    def test_clear(self) -> None:
        bus = EventBus()
        received = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.SYSTEM_STARTED, handler)
        bus.clear()
        bus.publish(Event(type=EventType.SYSTEM_STARTED))

        assert len(received) == 0


class TestGatewayController:
    @pytest.mark.slow
    def test_initialize_with_mock(self) -> None:
        config = AppConfig()
        config.detector.type = "mock"
        controller = GatewayController(config)
        controller.initialize()
        controller.stop()

    # Note: two legacy tests were removed in the refactor.
    # - ``test_start_and_stop`` patched ``_create_camera`` (private
    #   factory that no longer exists on ``GatewayController``).
    # - ``test_filter_results`` called ``_filter_results`` (the
    #   filtering logic moved to the detector adapter and the
    #   notification service; see ``test_target_class_filter`` and
    #   ``test_snapshot_service`` for the new contract tests).
    # The replacements are in ``test_controller_lifecycle`` and
    # ``test_target_class_filter``.
