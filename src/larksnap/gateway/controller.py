"""Gateway controller — thin orchestrator.

Composes Pipeline, NotificationService, and Recorder.
Delegates all domain logic to specialized modules.

State machine:
  IDLE        → open_camera()   → CAMERA_ON
  CAMERA_ON   → close_camera()  → IDLE
  CAMERA_ON   → start_detection() → DETECTING
  DETECTING   → stop_detection()  → CAMERA_ON
  DETECTING   → pause()        → DET_PAUSED
  DET_PAUSED  → resume()       → DETECTING
  DET_PAUSED  → stop_detection()  → CAMERA_ON
  CAMERA_ON/DETECTING/DET_PAUSED → close_camera() → IDLE
"""

from __future__ import annotations

import logging

from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.adapters.detector.interface import DetectionResult, DetectorAdapter
from larksnap.adapters.notifier.feishu_adapter import FeishuNotifierAdapter
from larksnap.adapters.notifier.feishu_ws_client import CommandHandler, FeishuWSClient
from larksnap.adapters.notifier.interface import NotifierAdapter
from larksnap.adapters.recorder.video_recorder import VideoRecorderAdapter
from larksnap.adapters.registry import camera_registry, detector_registry, notifier_registry
from larksnap.config.models import AppConfig
from larksnap.gateway.event_bus import Event, EventBus, EventType
from larksnap.gateway.notification_service import NotificationService, NotificationServiceConfig
from larksnap.gateway.pipeline import Pipeline, PipelineConfig
from larksnap.utils.exceptions import CameraError, GatewayError


class GatewayController:
    """Thin orchestrator that composes Pipeline + NotificationService + Recorder.

    Supports decoupled camera and detection lifecycle:
      - Camera can be opened/closed independently
      - Detection can be started/stopped while camera is open
      - Closing camera automatically stops detection
    """

    def __init__(self, config: AppConfig, event_bus: EventBus | None = None) -> None:
        self._config = config
        self._event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("larksnap.gateway")

        # Adapters (created in _create_adapters())
        self._camera: CameraAdapter | None = None
        self._detector: DetectorAdapter | None = None
        self._notifier: NotifierAdapter | None = None
        self._recorder: VideoRecorderAdapter | None = None
        self._ws_client: FeishuWSClient | None = None

        # Composed services
        self._pipeline: Pipeline | None = None
        self._notification_service: NotificationService | None = None

        # State
        self._camera_open = False
        self._detection_running = False
        self._detection_paused = False
        self._camera_failed = False

        # Ensure adapter modules are imported
        self._ensure_adapters_registered()

    # ── Public Properties ─────────────────────────────────────────────

    @property
    def is_camera_open(self) -> bool:
        return self._camera_open

    @property
    def is_running(self) -> bool:
        """Detection is running (not paused)."""
        return self._detection_running and not self._detection_paused

    @property
    def is_detection_active(self) -> bool:
        """Detection pipeline is active (running or paused)."""
        return self._detection_running

    @property
    def is_paused(self) -> bool:
        return self._detection_paused

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None and self._recorder.is_recording

    @property
    def detection_count(self) -> int:
        return self._pipeline.detection_count if self._pipeline else 0

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def producer_fps(self) -> float:
        return self._pipeline.producer_fps if self._pipeline else 0.0

    @property
    def notification_enabled(self) -> bool:
        return self._notification_service.notification_enabled if self._notification_service else False

    def get_latest_frame(self):
        return self._pipeline.get_latest_frame() if self._pipeline else None

    def get_latest_results(self) -> list[DetectionResult]:
        return self._pipeline.get_latest_results() if self._pipeline else []

    # ── Camera Lifecycle ──────────────────────────────────────────────

    def open_camera(self, device_index: int | None = None) -> None:
        """Open camera and start preview. Optionally switch device index."""
        if self._camera_open:
            if device_index is not None and device_index != self._config.camera.device_index:
                # Switch camera: close current, reopen with new index
                self.close_camera()
            else:
                self._logger.warning("Camera is already open")
                return

        try:
            self._logger.info("Opening camera...")

            if device_index is not None:
                self._config.camera.device_index = device_index

            self._create_adapters()

            try:
                self._camera.initialize()
            except CameraError as e:
                self._logger.error("Camera initialization failed: %s", e)
                self._event_bus.publish(Event(
                    type=EventType.CAMERA_FAILED,
                    data={"error": str(e), "device_index": self._config.camera.device_index},
                    source="gateway",
                ))
                self._camera_failed = True
                self._release_adapters()
                raise

            self._camera_failed = False
            self._detector.initialize()
            self._notifier.initialize()
            self._recorder.initialize()

            # Compose Pipeline
            pipeline_config = PipelineConfig(
                frame_queue_hwm=self._config.gateway.frame_queue_hwm,
                frame_queue_policy=self._config.gateway.frame_queue_policy,
            )
            self._pipeline = Pipeline(
                camera=self._camera,
                detector=self._detector,
                config=pipeline_config,
                event_bus=self._event_bus,
            )
            self._event_bus.subscribe(EventType.CAMERA_READ_FAILED, self._on_pipeline_fatal_error)

            # Compose NotificationService
            notif_config = NotificationServiceConfig(
                notification_interval=self._config.gateway.notification_interval,
                snapshot_dir=self._config.gateway.snapshot_dir,
                message_template=self._config.notifier.message_template,
            )
            self._notification_service = NotificationService(
                config=notif_config,
                notifier=self._notifier,
                event_bus=self._event_bus,
            )
            self._pipeline.set_on_results(self._notification_service.handle_results)

            # Start WebSocket command listener
            if self._config.notifier.app_id and self._config.notifier.app_secret:
                self._init_ws_client()

            # If chat_id is already configured, publish event for UI
            if self._config.notifier.chat_id:
                self._event_bus.publish(Event(
                    type=EventType.CHAT_ID_OBTAINED,
                    data=self._config.notifier.chat_id,
                    source="gateway",
                ))

            self._camera_open = True
            self._event_bus.publish(Event(type=EventType.CAMERA_OPENED, source="gateway"))
            self._logger.info("Camera opened (device %d)", self._config.camera.device_index)

        except Exception as e:
            self._release_adapters()
            raise GatewayError(f"Failed to open camera: {e}") from e

    def close_camera(self) -> None:
        """Close camera and release all resources. Stops detection if running."""
        if not self._camera_open:
            return

        # Stop detection first if running
        if self._detection_running:
            self._stop_detection_internal()

        # Stop recording if active
        if self._recorder is not None and self._recorder.is_recording:
            self._recorder.stop_recording()

        # Stop WS client
        if self._ws_client is not None:
            self._ws_client.stop()
            self._ws_client = None

        self._camera_open = False
        self._detection_running = False
        self._detection_paused = False

        self._release_adapters()
        self._event_bus.publish(Event(type=EventType.CAMERA_CLOSED, source="gateway"))
        self._logger.info("Camera closed")

    # ── Detection Lifecycle ───────────────────────────────────────────

    def start_detection(self) -> None:
        """Start detection pipeline. Camera must be open."""
        if not self._camera_open:
            raise GatewayError("Camera must be open before starting detection")
        if self._detection_running:
            self._logger.warning("Detection is already running")
            return

        self._pipeline.start()
        self._detection_running = True
        self._detection_paused = False
        self._event_bus.publish(Event(type=EventType.DETECTION_STARTED, source="gateway"))
        self._logger.info("Detection started")

    def stop_detection(self) -> None:
        """Stop detection pipeline. Camera stays open for preview."""
        if not self._detection_running:
            return
        self._stop_detection_internal()

    def _stop_detection_internal(self) -> None:
        """Internal: stop detection pipeline and update state."""
        if self._pipeline is not None:
            self._pipeline.stop()

        # Recreate pipeline (it can't be restarted after stop)
        pipeline_config = PipelineConfig(
            frame_queue_hwm=self._config.gateway.frame_queue_hwm,
            frame_queue_policy=self._config.gateway.frame_queue_policy,
        )
        self._pipeline = Pipeline(
            camera=self._camera,
            detector=self._detector,
            config=pipeline_config,
            event_bus=self._event_bus,
        )
        self._event_bus.subscribe(EventType.CAMERA_READ_FAILED, self._on_pipeline_fatal_error)
        if self._notification_service is not None:
            self._pipeline.set_on_results(self._notification_service.handle_results)

        self._detection_running = False
        self._detection_paused = False
        self._event_bus.publish(Event(type=EventType.DETECTION_STOPPED, source="gateway"))
        self._logger.info("Detection stopped")

    def pause(self) -> None:
        if self._pipeline is not None and self._detection_running and not self._detection_paused:
            self._pipeline.pause()
            self._detection_paused = True
            self._event_bus.publish(Event(type=EventType.DETECTION_PAUSED, source="gateway"))

    def resume(self) -> None:
        if self._pipeline is not None and self._detection_running and self._detection_paused:
            self._pipeline.resume()
            self._detection_paused = False
            self._event_bus.publish(Event(type=EventType.DETECTION_RESUMED, source="gateway"))

    # ── Recording ─────────────────────────────────────────────────────

    def start_recording(self) -> None:
        if self._recorder is not None and self._detection_running:
            self._recorder.start_recording(context=None)

    def stop_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.stop_recording()

    # ── Notification ──────────────────────────────────────────────────

    def enable_notification(self) -> None:
        if self._notification_service is not None:
            self._notification_service.enable_notification()

    def disable_notification(self) -> None:
        if self._notification_service is not None:
            self._notification_service.disable_notification()

    # ── Legacy compatibility ──────────────────────────────────────────

    def initialize(self) -> None:
        """Legacy: open camera + start detection."""
        self.open_camera()
        self.start_detection()

    def start(self) -> None:
        """Legacy: start detection if camera is open."""
        if self._camera_open and not self._detection_running:
            self.start_detection()

    def stop(self) -> None:
        """Legacy: close camera (stops everything)."""
        self.close_camera()

    # ── Event handlers ────────────────────────────────────────────────

    def _on_pipeline_fatal_error(self, event: Event) -> None:
        """Handle pipeline fatal error — sync controller state."""
        self._logger.error("Pipeline stopped due to fatal error: %s", event.data)
        self._detection_running = False
        self._detection_paused = False
        self._camera_open = False
        self._event_bus.publish(Event(type=EventType.CAMERA_READ_FAILED, data=event.data, source="gateway"))

    # ── Private ────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_adapters_registered() -> None:
        """Import adapter modules to trigger @register decorators."""
        import larksnap.adapters.camera.opencv_adapter  # noqa: F401
        import larksnap.adapters.detector.mock_adapter  # noqa: F401
        import larksnap.adapters.detector.seg_adapter  # noqa: F401
        import larksnap.adapters.notifier.feishu_adapter  # noqa: F401

    def _create_adapters(self) -> None:
        """Create adapter instances via registry."""
        self._camera = camera_registry.create(
            self._config.camera.type, self._config.camera
        )
        self._detector = detector_registry.create(
            self._config.detector.type, self._config.detector
        )
        self._notifier = notifier_registry.create(
            self._config.notifier.type, self._config.notifier
        )
        self._recorder = VideoRecorderAdapter(
            output_dir=self._config.recorder.output_dir,
            fps=self._config.recorder.fps,
            codec=self._config.recorder.codec,
            frame_queue_url=Pipeline.FRAME_QUEUE_URL,
        )

    def _init_ws_client(self) -> None:
        try:
            self._ws_client = FeishuWSClient(
                config=self._config.notifier,
                on_command=self._handle_command,
            )
            self._ws_client.start()
        except Exception as e:
            self._logger.warning(
                "Feishu WS client failed to start (commands disabled): %s", e
            )
            self._ws_client = None

    def _handle_command(self, cmd: CommandHandler) -> None:
        self._logger.info("Processing command: /%s", cmd.name)

        if cmd.chat_id and isinstance(self._notifier, FeishuNotifierAdapter):
            self._notifier.set_chat_id(cmd.chat_id)

        if cmd.name == "init":
            self._logger.info("Init command received, chat_id obtained")
            self._event_bus.publish(
                Event(type=EventType.CHAT_ID_OBTAINED, data=cmd.chat_id, source="gateway")
            )
            if isinstance(self._notifier, FeishuNotifierAdapter):
                self._notifier.send_text("[LarkSnap] 初始化成功！发送 /start 开始通知，/stop 停止通知")
        elif cmd.name == "start":
            if self._notification_service is not None:
                self._notification_service.enable_notification()
            if isinstance(self._notifier, FeishuNotifierAdapter):
                self._notifier.send_text("[LarkSnap] 通知已开启，检测到目标将推送告警")
        elif cmd.name == "stop":
            if self._notification_service is not None:
                self._notification_service.disable_notification()
            if isinstance(self._notifier, FeishuNotifierAdapter):
                self._notifier.send_text("[LarkSnap] 通知已关闭，发送 /start 可重新开启")
        elif cmd.name == "pause":
            if self._detection_running and not self._detection_paused:
                self.pause()
        elif cmd.name == "resume":
            if self._detection_running and self._detection_paused:
                self.resume()
        elif cmd.name == "status":
            status = "running" if self._detection_running else "stopped"
            if self._detection_running and self._detection_paused:
                status = "paused"
            notif = "enabled" if (self._notification_service and self._notification_service.notification_enabled) else "disabled"
            self._logger.info("Gateway status: %s, notification: %s", status, notif)
        elif cmd.name == "help":
            self._logger.info(
                "Available commands: /init, /start, /stop, /pause, /resume, /status, /help"
            )

    def _release_adapters(self) -> None:
        """Release all adapter resources."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

        adapters = [self._notifier, self._detector, self._camera, self._recorder]
        for adapter in adapters:
            if adapter is not None:
                try:
                    adapter.stop()
                    adapter.release()
                except Exception as e:
                    self._logger.error("Error releasing adapter: %s", e)

        self._notifier = None
        self._detector = None
        self._camera = None
        self._recorder = None
        self._notification_service = None
