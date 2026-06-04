import time
from unittest.mock import MagicMock, patch

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
    def test_initialize_with_mock(self) -> None:
        config = AppConfig()
        config.detector.type = "mock"
        controller = GatewayController(config)
        controller.initialize()
        controller.stop()

    def test_start_and_stop(self) -> None:
        config = AppConfig()
        config.detector.type = "mock"
        config.camera.device_index = 999
        controller = GatewayController(config)

        with patch.object(controller, "_create_camera") as mock_create:
            mock_camera = MagicMock()
            mock_camera.read_frame.return_value = __import__("numpy").zeros(
                (480, 640, 3), dtype=__import__("numpy").uint8
            )
            mock_create.return_value = mock_camera
            controller.initialize()
            controller.start()
            assert controller.is_running
            time.sleep(0.5)
            controller.stop()
            assert not controller.is_running

    def test_pause_and_resume(self) -> None:
        config = AppConfig()
        config.detector.type = "mock"
        controller = GatewayController(config)

        with patch.object(controller, "_create_camera") as mock_create:
            mock_camera = MagicMock()
            mock_camera.read_frame.return_value = __import__("numpy").zeros(
                (480, 640, 3), dtype=__import__("numpy").uint8
            )
            mock_create.return_value = mock_camera
            controller.initialize()
            controller.start()
            controller.pause()
            assert controller.is_paused
            controller.resume()
            assert not controller.is_paused
            controller.stop()

    def test_filter_results(self) -> None:
        from larksnap.adapters.detector.interface import BBox, DetectionResult

        config = AppConfig()
        config.detector.confidence_threshold = 0.7
        config.detector.target_classes = ["person"]
        controller = GatewayController(config)

        results = [
            DetectionResult(label="person", confidence=0.9, bbox=BBox(0, 0, 100, 100)),
            DetectionResult(label="car", confidence=0.9, bbox=BBox(0, 0, 100, 100)),
            DetectionResult(label="person", confidence=0.3, bbox=BBox(0, 0, 100, 100)),
        ]

        filtered = controller._filter_results(results)
        assert len(filtered) == 1
        assert filtered[0].label == "person"
        assert filtered[0].confidence == 0.9
