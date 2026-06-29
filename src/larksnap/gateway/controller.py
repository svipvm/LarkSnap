"""Gateway controller — thin orchestrator.

Composes Pipeline, NotificationService, and Recorder.
Delegates all domain logic to specialized modules.

State machine:
  IDLE        → open_camera()   → CAMERA_ON     (preview running, detection off)
  CAMERA_ON   → start_detection() → DETECTING   (preview running, detection active)
  DETECTING   → stop_detection()  → CAMERA_ON   (preview running, detection off)
  CAMERA_ON/DETECTING → close_camera() → IDLE

Concurrency contract:
  - State is mutated only under ``self._state_lock`` (an RLock so a
    state method may call another one).
  - Adapters, pipeline, and the WS client are created/destroyed only
    on the thread that calls ``open_camera`` / ``close_camera``. The
    main window always invokes these from the Qt main thread, so the
    "owning thread" for the pipeline is the Qt main thread.
  - The pipeline's background threads (FrameProducer, FrameConsumer,
    ResultSubscriber) run on their own Python threads and only
    communicate with the controller via ``EventBus`` events.
  - Long-running close operations are dispatched to a one-shot
    background thread so the Qt main loop stays responsive.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from typing import Any, Callable

from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.adapters.detector.interface import DetectionResult, DetectorAdapter
from larksnap.adapters.detector.seg_adapter import _SegWrapper
from larksnap.adapters.notifier.command_registry import CommandRegistry, CommandSpec
from larksnap.adapters.notifier.feishu_adapter import FeishuNotifierAdapter
from larksnap.adapters.notifier.feishu_ws_client import CommandHandler, FeishuWSClient
from larksnap.adapters.notifier.interface import NotifierAdapter
from larksnap.adapters.recorder.video_recorder import VideoRecorderAdapter
from larksnap.adapters.registry import camera_registry, detector_registry, notifier_registry
from larksnap.config.models import AppConfig
from larksnap.config.service import ConfigService
from larksnap.config.state_persistence import (
    PersistedRuntimeState,
    default_state_path,
    load_state,
    save_state,
)
from larksnap.gateway.component_state import (
    ComponentKind,
    ComponentState,
    ComponentStatus,
    SystemStatus,
    component_state_from_legacy,
)
from larksnap.gateway.event_bus import Event, EventBus, EventType
from larksnap.gateway.notification_service import NotificationService, NotificationServiceConfig
from larksnap.gateway.pipeline import Pipeline, PipelineConfig
from larksnap.utils.exceptions import CameraError, GatewayError
from larksnap.utils.worker_pool import WorkerPool

# Type alias for event handlers (mirrors the one in event_bus.py).
EventHandler = Callable[["Event"], None]


def _render_single_help(spec: CommandSpec) -> str:
    """Format the help block for a single :class:`CommandSpec`.

    Module-level so it can be reused by tests without instantiating a
    controller. Mirrors the layout used by
    :py:meth:`CommandRegistry.render_help` for the full catalogue.
    """
    lines = [
        f"/{spec.name} - {spec.description}",
        f"  语法: {spec.syntax}",
    ]
    if spec.examples:
        lines.append("  示例: " + "，".join(spec.examples))
    if spec.aliases:
        lines.append("  别名: " + "，".join(f"/{a}" for a in spec.aliases))
    return "\n".join(lines)


class GatewayState(enum.Enum):
    """Atomic gateway state.

    All public state queries (``is_camera_open``, ``is_running``, ...)
    read this enum under ``_state_lock``. All state transitions go
    through ``_set_state`` which validates the transition.
    """

    IDLE = "idle"               # no camera, no pipeline
    OPENING = "opening"         # camera init in progress (used internally)
    CLOSING = "closing"         # camera teardown in progress (used internally)
    CAMERA_ON = "camera_on"     # pipeline running, detection off
    DETECTING = "detecting"     # pipeline running, detection active


# State transitions that are allowed. ``CLOSING`` and ``OPENING`` are
# transient internal states — public methods never return them.
_ALLOWED_TRANSITIONS: dict[GatewayState, frozenset[GatewayState]] = {
    GatewayState.IDLE:        frozenset({GatewayState.OPENING}),
    GatewayState.OPENING:     frozenset({GatewayState.CAMERA_ON, GatewayState.IDLE}),
    GatewayState.CAMERA_ON:   frozenset({GatewayState.DETECTING, GatewayState.CLOSING}),
    GatewayState.DETECTING:   frozenset({GatewayState.CAMERA_ON, GatewayState.CLOSING}),
    GatewayState.CLOSING:     frozenset({GatewayState.IDLE}),
}


def _safe_release_adapter(adapter: Any, logger: logging.Logger) -> None:
    """Best-effort adapter stop+release. Never raises.

    Production-grade teardown must not propagate exceptions from one
    adapter's cleanup into another's. We log and continue.
    """
    if adapter is None:
        return
    try:
        if hasattr(adapter, "stop") and callable(adapter.stop):
            adapter.stop()
    except Exception as e:  # noqa: BLE001 — cleanup path, log all
        logger.error("Adapter stop failed: %s", e)
    try:
        if hasattr(adapter, "release") and callable(adapter.release):
            adapter.release()
    except Exception as e:  # noqa: BLE001
        logger.error("Adapter release failed: %s", e)


class GatewayController:
    """Thin orchestrator that composes Pipeline + NotificationService + Recorder.

    Supports decoupled camera and detection lifecycle:
      - Camera can be opened/closed independently
      - Detection can be started/stopped while camera is open
      - Closing camera automatically stops detection
    """

    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus | None = None,
        state_path: "Path | None" = None,
        config_service: ConfigService | None = None,
    ) -> None:
        # Defer the typing import — Path is in the local module
        # scope, no need to widen the module's typing imports.
        from pathlib import Path as _Path

        self._config = config
        # ``ConfigService`` is the gateway's window onto ``AppConfig``
        # for the Feishu ``/config`` command. It is optional in
        # __init__ so the existing tests that build a bare
        # ``GatewayController`` continue to work; when ``None``
        # the controller fabricates an in-memory one with no
        # backing file path, which means ``/config set`` will
        # still mutate the AppConfig but won't persist to disk.
        self._config_service = config_service or ConfigService(config)
        self._event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("larksnap.gateway")
        # Where the detector/notifier on/off state is persisted
        # across camera close/open cycles. ``None`` means "use the
        # default user-data path".
        self._state_path: _Path | None = (
            _Path(state_path) if state_path is not None else default_state_path()
        )

        # Pre-compute the per-instance inproc URLs so the recorder
        # (created in _create_adapters) and the pipeline (created
        # later) agree on the same endpoints. Using a uuid suffix
        # avoids the "Address in use" error that the fixed
        # ``Pipeline.FRAME_QUEUE_URL`` would produce on the second
        # open/close cycle when the previous producer's inproc
        # endpoint is still being torn down inside the singleton
        # ZMQ context.
        import uuid
        self._instance_id = uuid.uuid4().hex[:8]
        self._frame_queue_url = (
            f"inproc://larksnap_frame_queue_{self._instance_id}"
        )
        self._result_queue_url = (
            f"inproc://larksnap_detection_results_{self._instance_id}"
        )

        # Adapters (created in _create_adapters())
        self._camera: CameraAdapter | None = None
        self._detector: DetectorAdapter | None = None
        self._notifier: NotifierAdapter | None = None
        self._recorder: VideoRecorderAdapter | None = None
        self._ws_client: FeishuWSClient | None = None

        # Composed services
        self._pipeline: Pipeline | None = None
        self._notification_service: NotificationService | None = None

        # Bounded worker pool for offloading blocking I/O. The ZMQ
        # result-subscriber thread (which fires the notification
        # callback) and any other "fast loop" must never block on
        # HTTP calls or disk writes; those are pushed here.
        self._worker_pool: WorkerPool = WorkerPool(
            max_workers=4,
            queue_size=128,
            name_prefix="larksnap-gw",
        )

        # State — all access goes through _state_lock. We keep the
        # individual booleans as a *cache* of the state enum for
        # backwards-compatible property semantics.
        self._state_lock = threading.RLock()
        self._state: GatewayState = GatewayState.IDLE
        self._camera_failed = False

        # Tracked event-bus subscriptions so re-opening the camera
        # doesn't pile up duplicate handlers. Key: (EventType, handler_id).
        self._subscriptions: set[tuple[EventType, int]] = set()

        # One-shot thread used for non-blocking close. We protect the
        # start of a new close thread with a lock so concurrent
        # ``close_camera()`` calls (e.g. double-clicks) can't race
        # against each other and either spawn a second close thread
        # or clear the done event while another thread is waiting on
        # it. ``_close_thread_lock`` and ``_close_done`` are only
        # touched while holding ``_state_lock`` (to keep ordering
        # guarantees simple) but the lock is its own RLock so
        # ``wait_closed`` can take it independently.
        self._close_thread_lock = threading.RLock()
        self._close_thread: threading.Thread | None = None
        self._close_done = threading.Event()

        # Ensure adapter modules are imported
        self._ensure_adapters_registered()

        # Subscribe to live config changes so ``/config set`` can
        # propagate to the running components (detector target
        # classes, notification cooldown, message template, etc.).
        # The unsubscribe function is held for the controller's
        # lifetime — there's no public close path that needs it,
        # and the lambda captures only attributes that exist for
        # the whole process.
        self._config_unsubscribe = self._config_service.subscribe(
            self._on_config_changed
        )

    # ── Config change propagation ────────────────────────────────────

    def _on_config_changed(self, path: str, _old: Any, _new: Any) -> None:
        """Apply a config change to running components.

        Called by the ``ConfigService`` after every successful
        ``set``. Only paths that have a live effect are handled
        here — restart-required paths are persisted to disk by
        the service and the chat reply tells the user to restart.
        """
        try:
            if path == "detector.target_classes":
                # Hot-swap the seg detector's monitoring set.
                # Mock detector reads fresh from config so it
                # doesn't need a callback.
                if isinstance(self._detector, _SegWrapper):
                    self._detector.set_target_classes(_new)
                # Also keep the notification service's defense-in-depth
                # filter in sync — the user expects both layers to
                # reflect the new set.
                if self._notification_service is not None:
                    self._notification_service._config.target_classes = list(_new)
            elif path == "notifier.message_template":
                # ``NotificationService`` already re-reads its config
                # on every dispatch, so an in-place mutation is
                # enough. We touch the attribute explicitly to make
                # the propagation discoverable in code review.
                if self._notification_service is not None:
                    self._notification_service._config.message_template = _new
            elif path == "gateway.snapshot_dir":
                if self._notification_service is not None:
                    self._notification_service._config.snapshot_dir = _new
            elif path == "notifier.send_image":
                if self._notification_service is not None:
                    self._notification_service._config.send_image = _new
            # gateway.notification_interval is read from
            # self._config.notification_interval *via* the
            # NotificationServiceConfig snapshot the controller
            # built at camera-open time. Mirror the same trick
            # here so a chat /config set takes effect on the
            # next detection, not the next camera open.
            elif path == "gateway.notification_interval":
                if self._notification_service is not None:
                    self._notification_service._config.notification_interval = _new
        except Exception as e:
            # A subscriber that raises must not abort the chat
            # reply — log and move on. The disk file is already
            # updated; the live state will catch up next time the
            # user touches a related config field.
            self._logger.warning(
                "Failed to apply live config change %r: %s", path, e
            )

    # ── State helpers (must be called under _state_lock) ──────────────

    def _set_state(self, new_state: GatewayState) -> bool:
        """Atomic state transition with validation.

        Returns True if the transition was allowed, False if rejected
        (e.g. a duplicate open_camera() call while already opening).
        """
        with self._state_lock:
            allowed = _ALLOWED_TRANSITIONS.get(self._state, frozenset())
            if new_state not in allowed:
                self._logger.debug(
                    "Rejecting state transition %s → %s (not allowed)",
                    self._state.value, new_state.value,
                )
                return False
            old = self._state
            self._state = new_state
            if old is not new_state:
                self._logger.info("State: %s → %s", old.value, new_state.value)
            return True

    def _get_state(self) -> GatewayState:
        with self._state_lock:
            return self._state

    def _publish_component_state(self, *kinds: ComponentKind) -> None:
        """Publish ``COMPONENT_STATE_CHANGED`` for each given subsystem.

        Centralised so every transition (camera open, detection
        start/stop, notifier toggle) emits a single, consistent
        event with the freshest snapshot. UI listeners can rely on
        this event to keep the status panel, menu checkboxes, and
        tray icon in lock-step with the backend without polling.

        Exceptions are swallowed (and logged) because publishing is a
        best-effort notification — a failed publish must not undo a
        successful state transition.
        """
        for kind in kinds:
            try:
                status = self.get_component_status(kind)
                self._event_bus.publish(Event(
                    type=EventType.COMPONENT_STATE_CHANGED,
                    data=status,
                    source="gateway",
                ))
            except Exception as e:  # noqa: BLE001
                self._logger.error(
                    "Failed to publish component state for %s: %s",
                    kind.value, e,
                )

    def _subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe ``handler`` to ``event_type`` exactly once.

        Naive repeated ``event_bus.subscribe`` calls would stack the
        same handler N times after N open/close cycles, causing N
        callback invocations per event. Track subscriptions in a set
        and skip duplicates.

        The key uses ``id(underlying func)`` rather than ``id(handler)``
        because bound-method objects get a fresh ``id`` on every
        attribute access, which would defeat the dedup.
        """
        # For bound methods, use the underlying function's id; for
        # plain functions/lambdas, use the handler's id directly.
        func = getattr(handler, "__func__", handler)
        key = (event_type, id(func))
        if key in self._subscriptions:
            return
        self._event_bus.subscribe(event_type, handler)
        self._subscriptions.add(key)

    # ── Public Properties ─────────────────────────────────────────────

    @property
    def state(self) -> GatewayState:
        """Current gateway state (thread-safe)."""
        return self._get_state()

    @property
    def is_camera_open(self) -> bool:
        """True when the camera is open and the pipeline is running.

        Includes CAMERA_ON and DETECTING — anything where the camera
        is actively producing frames.
        """
        s = self._get_state()
        return s in (GatewayState.CAMERA_ON, GatewayState.DETECTING)

    @property
    def is_running(self) -> bool:
        """Detection is running (camera open and detection active)."""
        return self._get_state() == GatewayState.DETECTING

    @property
    def is_detection_active(self) -> bool:
        """Detection pipeline is active."""
        return self._get_state() == GatewayState.DETECTING

    @property
    def is_busy(self) -> bool:
        """True while a camera open/close operation is in flight.

        The UI uses this to disable menu items during transitions and
        to display a loading overlay so the window doesn't look frozen.
        """
        s = self._get_state()
        return s in (GatewayState.OPENING, GatewayState.CLOSING)

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
        """Deprecated alias for :py:meth:`is_notification_enabled`.

        Kept so existing callers and tests don't break; new code
        should use the ``is_*`` form to match the other state
        properties (``is_camera_open``, ``is_running`` etc.).
        """
        return self.is_notification_enabled

    @property
    def is_notification_enabled(self) -> bool:
        """True if the notification dispatch is currently enabled.

        The notifier can be enabled even when the camera is closed —
        the ``/start`` and ``/stop`` Feishu commands toggle this
        regardless of camera state.
        """
        ns = self._notification_service
        if ns is None:
            return False
        return ns.is_notification_enabled

    def get_component_status(
        self, kind: ComponentKind
    ) -> ComponentStatus:
        """Return the unified status of one subsystem.

        Centralises the mapping from the controller's existing
        booleans onto a single ``ComponentStatus`` so the UI doesn't
        have to know about each subsystem's internal flags.
        """
        # Snapshot everything under the state lock so the returned
        # status is consistent even if a transition is in flight.
        with self._state_lock:
            gateway_state = self._state
            camera_failed = self._camera_failed
            is_busy = gateway_state in (
                GatewayState.OPENING,
                GatewayState.CLOSING,
            )
            cam_open = gateway_state in (
                GatewayState.CAMERA_ON,
                GatewayState.DETECTING,
            )

        if kind is ComponentKind.CAMERA:
            camera_state = component_state_from_legacy(
                is_open=cam_open,
                is_busy=is_busy and gateway_state == GatewayState.OPENING,
                is_running=cam_open and gateway_state != GatewayState.OPENING,
                is_failed=camera_failed,
            )
            return ComponentStatus(
                kind=kind,
                state=camera_state,
                detail=("最近一次初始化失败" if camera_failed else None),
            )

        if kind is ComponentKind.DETECTOR:
            detector_state = component_state_from_legacy(
                is_open=gateway_state == GatewayState.DETECTING,
                is_busy=is_busy,
                is_running=gateway_state == GatewayState.DETECTING,
                is_failed=False,
            )
            return ComponentStatus(kind=kind, state=detector_state)

        if kind is ComponentKind.NOTIFIER:
            notif_on = self.is_notification_enabled
            notif_state = (
                ComponentState.DISABLED
                if not notif_on
                else component_state_from_legacy(
                    is_open=notif_on,
                    is_busy=False,
                    is_running=notif_on,
                )
            )
            return ComponentStatus(kind=kind, state=notif_state)

        raise ValueError(f"Unknown component kind: {kind!r}")

    def get_system_status(self) -> SystemStatus:
        """Return a snapshot of all three subsystem states at once."""
        return SystemStatus(
            camera=self.get_component_status(ComponentKind.CAMERA),
            detector=self.get_component_status(ComponentKind.DETECTOR),
            notifier=self.get_component_status(ComponentKind.NOTIFIER),
        )

    def get_latest_frame(self):
        return self._pipeline.get_latest_frame() if self._pipeline else None

    def get_latest_results(self) -> list[DetectionResult]:
        return self._pipeline.get_latest_results() if self._pipeline else []

    # ── Camera Lifecycle ──────────────────────────────────────────────

    def open_camera(self, device_index: int | None = None) -> None:
        """Open camera and start the preview pipeline.

        The pipeline is started immediately (in ``CAMERA_ON`` state) so
        the UI sees frames right away. Detection is controlled by
        ``start_detection()`` / ``stop_detection()``. This fixes the
        "camera open but no frames" issue.

        Reentrant: calling this while already open with the same
        device is a no-op; calling with a different device triggers a
        close+reopen cycle.

        Runs synchronously on the calling thread, but the expensive
        ZMQ/pipeline start is bounded — if it ever blocks, the UI
        thread will at worst experience a brief pause, not a hang,
        because every ZMQ operation has a timeout.
        """
        with self._state_lock:
            current = self._state
            if current in (GatewayState.OPENING, GatewayState.CLOSING):
                self._logger.warning(
                    "Camera operation already in progress (state=%s); ignoring open_camera",
                    current.value,
                )
                return
            if current in (GatewayState.CAMERA_ON, GatewayState.DETECTING):
                if device_index is not None and device_index != self._config.camera.device_index:
                    # Switch camera: close current, reopen with new index.
                    # Release the lock so close_camera() can take it.
                    target_device = device_index
                else:
                    self._logger.warning("Camera is already open")
                    return
            else:
                target_device = device_index

        # Switch path: close then reopen on the same caller's thread.
        if current in (GatewayState.CAMERA_ON, GatewayState.DETECTING) \
                and device_index is not None and device_index != self._config.camera.device_index:
            self._config.camera.device_index = target_device  # type: ignore[has-type]
            # Close blocks this thread; safe because it is the UI thread
            # and the close path is bounded by timeouts.
            self.close_camera()
            self.open_camera()
            return

        # Normal open path
        if not self._set_state(GatewayState.OPENING):
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
                self._set_state(GatewayState.IDLE)
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
            self._subscribe(EventType.CAMERA_READ_FAILED, self._on_pipeline_fatal_error)

            # Compose NotificationService
            notif_config = NotificationServiceConfig(
                notification_interval=self._config.gateway.notification_interval,
                snapshot_dir=self._config.gateway.snapshot_dir,
                message_template=self._config.notifier.message_template,
                # Defense-in-depth: the notification layer also
                # filters by target_classes so a detector adapter
                # that forgets the contract cannot leak non-target
                # detections to the user.
                target_classes=list(self._config.detector.target_classes),
            )
            self._notification_service = NotificationService(
                config=notif_config,
                notifier=self._notifier,
                event_bus=self._event_bus,
                worker_pool=self._worker_pool,
            )
            self._pipeline.set_on_results(self._notification_service.handle_results)

            # Start WebSocket command listener (process-lifetime singleton)
            if self._config.notifier.app_id and self._config.notifier.app_secret:
                self._init_ws_client()

            # If chat_id is already configured, publish event for UI
            if self._config.notifier.chat_id:
                self._event_bus.publish(Event(
                    type=EventType.CHAT_ID_OBTAINED,
                    data=self._config.notifier.chat_id,
                    source="gateway",
                ))

            # ── PRIMARY FIX: start the pipeline NOW so preview frames
            # flow in CAMERA_ON state. Detection runs from the moment
            # the camera opens, so no separate start step is needed.
            self._pipeline.start()

            if not self._set_state(GatewayState.CAMERA_ON):
                # Unexpected: someone else moved the state. Roll back.
                self._logger.error(
                    "State transition to CAMERA_ON rejected (was %s) — rolling back",
                    self._state.value,
                )
                self._release_adapters()
                self._set_state(GatewayState.IDLE)
                raise GatewayError("Failed to enter CAMERA_ON state")

            self._event_bus.publish(Event(type=EventType.CAMERA_OPENED, source="gateway"))
            # Publish the unified component state so the status
            # panel and menu checkboxes update in lock-step.
            self._publish_component_state(
                ComponentKind.CAMERA, ComponentKind.DETECTOR, ComponentKind.NOTIFIER,
            )
            # Restore the detector + notifier on/off state from the
            # previous session. Done after CAMERA_OPENED so the UI
            # has subscribed to the event bus and will see the
            # subsequent detector / notifier state events.
            self._restore_runtime_state()
            self._logger.info("Camera opened (device %d)", self._config.camera.device_index)

        except Exception as e:
            # Make sure we always end up in IDLE on failure
            self._release_adapters()
            with self._state_lock:
                if self._state in (GatewayState.OPENING, GatewayState.CAMERA_ON):
                    self._state = GatewayState.IDLE
            raise GatewayError(f"Failed to open camera: {e}") from e

    def close_camera(self) -> None:
        """Close camera and release all resources. Stops detection if running.

        Implemented as a non-blocking state transition: the calling
        thread (typically the Qt main thread) only flips the state to
        ``CLOSING`` and starts a background daemon to do the actual
        teardown. The background thread publishes ``CAMERA_CLOSED``
        when done, which the UI listens for via the event bridge.

        Concurrency: two near-simultaneous ``close_camera()`` calls
        (e.g. the user double-clicks the close button, or the fatal
        error handler and the user both try to close at once) are
        serialised by ``_close_thread_lock`` so exactly one close
        thread runs. The second call is a no-op that returns
        immediately; callers that need a "fully closed" signal must
        use ``wait_closed()``.

        This eliminates the multi-second UI freezes that occurred when
        ZMQ socket closes, thread joins, and context termination all
        ran synchronously on the main thread.

        Note: The Feishu WebSocket command client is intentionally NOT
        stopped here. The lark-oapi SDK's ``ws.Client.start()`` is a
        blocking call that owns an asyncio event loop with no public
        stop API. Restarting it on every camera open/close cycle causes
        ``RuntimeError: This event loop is already running`` and
        ``coroutine was never awaited`` warnings because the SDK has
        module-level singleton state that gets corrupted across
        instances. The WS client is started once (lazily, on first
        ``open_camera``) and torn down once on process exit via
        ``stop()``.
        """
        with self._close_thread_lock:
            with self._state_lock:
                # If we're already in CLOSING and a thread is still
                # alive, this call is a duplicate; do nothing.
                if self._state == GatewayState.CLOSING:
                    if self._close_thread is not None and self._close_thread.is_alive():
                        return
                    # Otherwise a previous close thread finished but
                    # the state was left at CLOSING (defensive).
                if self._state == GatewayState.IDLE:
                    return
                # Capture the on/off state of the detector + notifier
                # BEFORE flipping the state to CLOSING. Done under
                # the state lock and on the calling thread so we read
                # the user's current intent (DETECTING / CAMERA_ON),
                # not the transient CLOSING value. The background
                # worker runs later, by which time the state has
                # already moved on.
                self._capture_and_persist_runtime_state()
                if not self._set_state(GatewayState.CLOSING):
                    return

            # Reset the done event and start the background teardown.
            # Done under ``_close_thread_lock`` so a second
            # ``close_camera()`` can't race us into clearing an event
            # that another thread is about to wait on. Without the
            # lock, a waiter that started after the previous close
            # completed could see the stale "True" and miss the new
            # close entirely.
            self._close_done.clear()
            self._close_thread = threading.Thread(
                target=self._close_camera_worker,
                name="GatewayController-close",
                daemon=True,
            )
            self._close_thread.start()

    def _close_camera_worker(self) -> None:
        """Background worker: do the actual teardown.

        Uses timeouts on every blocking operation. Any exception is
        logged and swallowed — we MUST reach the IDLE state so the
        gateway can be reopened.
        """
        try:
            self._logger.info("Closing camera (background)...")

            # Stop recording first and wait for the recorder to
            # fully flush the in-flight writes to disk. We do this
            # on the close worker (NOT the UI thread) so the user
            # can keep interacting with the menu while the file
            # is being finalised. The recorder's stop_recording
            # itself is non-blocking, so this wait is the only
            # blocking step and it now runs off the UI thread.
            if self._recorder is not None and (
                self._recorder.is_recording or self._recorder.is_draining
            ):
                try:
                    self._recorder.stop_recording()
                    self._recorder.wait_drained(timeout=5.0)
                except Exception as e:
                    self._logger.error("Stop recording during close failed: %s", e)

            # Stop pipeline (this may take a few seconds for ZMQ cleanup)
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception as e:
                    self._logger.error("Pipeline stop during close failed: %s", e)

            # Release adapters (best-effort, never raises)
            try:
                self._release_adapters()
            except Exception as e:
                self._logger.error("Adapter release during close failed: %s", e)

            with self._state_lock:
                self._state = GatewayState.IDLE

            try:
                self._event_bus.publish(Event(type=EventType.CAMERA_CLOSED, source="gateway"))
                # Publish unified component state so the UI panel
                # and menu reflect the new camera/detector state.
                self._publish_component_state(
                    ComponentKind.CAMERA, ComponentKind.DETECTOR,
                )
            except Exception as e:
                self._logger.error("Publish CAMERA_CLOSED failed: %s", e)

            self._logger.info("Camera closed")
        finally:
            self._close_done.set()

    def _capture_and_persist_runtime_state(self) -> None:
        """Snapshot detector + notifier state and write it to disk.

        Called from the close-camera worker, BEFORE the pipeline
        and notification service are released, so the source of
        truth is still live. Best-effort: any error is logged but
        does not abort the close.
        """
        try:
            detector_running = self._state == GatewayState.DETECTING
            notifier_enabled = (
                self._notification_service.is_notification_enabled
                if self._notification_service is not None
                else True
            )
            save_state(
                PersistedRuntimeState(
                    detector_running=detector_running,
                    notifier_enabled=notifier_enabled,
                ),
                self._state_path,
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            self._logger.error("Failed to capture runtime state: %s", e)

    def _restore_runtime_state(self) -> None:
        """Read the persisted state and apply it to the live components.

        Called from open_camera() AFTER the pipeline and
        notification service are created. If no persisted state is
        available, or the file is unreadable, we fall back to the
        natural defaults (detection running, notifier enabled) so
        the user's first run is unchanged.

        The user-visible effect is seamless: the menu checkboxes
        and status panel jump straight to the saved state via the
        component-state event published by each method below, so
        no "re-initialising" flash appears.
        """
        persisted = load_state(self._state_path)
        if persisted is None:
            return
        # Apply detector state first (sets the gateway state enum),
        # then notifier state (independent of the pipeline).
        if persisted.detector_running and self._state == GatewayState.CAMERA_ON:
            try:
                self.start_detection()
            except Exception as e:  # noqa: BLE001 — best-effort
                self._logger.error("Failed to restore detector state: %s", e)
        elif not persisted.detector_running and self._state == GatewayState.DETECTING:
            try:
                self.stop_detection()
            except Exception as e:  # noqa: BLE001 — best-effort
                self._logger.error("Failed to restore detector state: %s", e)
        # Notifier state. We always have a notification service at
        # this point because open_camera() created one before calling
        # this method.
        if self._notification_service is not None:
            try:
                if persisted.notifier_enabled:
                    self._notification_service.enable_notification()
                else:
                    self._notification_service.disable_notification()
            except Exception as e:  # noqa: BLE001 — best-effort
                self._logger.error("Failed to restore notifier state: %s", e)

    def wait_closed(self, timeout: float = 5.0) -> bool:
        """Block until a pending close operation finishes.

        Returns True if the close completed within ``timeout`` seconds,
        False otherwise. Safe to call when no close is in progress
        (returns True immediately). Safe to call concurrently — the
        event is created once and never replaced.
        """
        if not self.is_busy:
            return True
        return self._close_done.wait(timeout)

    # ── Detection Lifecycle ───────────────────────────────────────────

    def start_detection(self) -> None:
        """Start detection (resume the paused pipeline). Camera must be open."""
        with self._state_lock:
            current = self._state
            if current == GatewayState.IDLE:
                raise GatewayError("Camera must be open before starting detection")
            if current in (GatewayState.OPENING, GatewayState.CLOSING):
                self._logger.warning(
                    "Cannot start detection while camera is %s", current.value,
                )
                return
            if current == GatewayState.DETECTING:
                self._logger.warning("Detection is already running")
                return
            if not self._set_state(GatewayState.DETECTING):
                return

        if self._pipeline is not None:
            self._pipeline.resume()
        self._event_bus.publish(Event(type=EventType.DETECTION_STARTED, source="gateway"))
        self._publish_component_state(ComponentKind.DETECTOR)
        self._logger.info("Detection started")

    def stop_detection(self) -> None:
        """Stop detection (pause the pipeline). Camera stays open for preview.

        Unlike the previous implementation, we do NOT tear down and
        recreate the pipeline. Detection is simply paused — the
        camera and preview keep running. This is much cheaper and
        avoids the ZMQ close/reopen cycle that caused re-open bugs.
        """
        with self._state_lock:
            current = self._state
            if current != GatewayState.DETECTING:
                return
            if not self._set_state(GatewayState.CAMERA_ON):
                return

        if self._pipeline is not None:
            self._pipeline.pause()
        self._event_bus.publish(Event(type=EventType.DETECTION_STOPPED, source="gateway"))
        self._publish_component_state(ComponentKind.DETECTOR)
        self._logger.info("Detection stopped")

    # ── Recording ─────────────────────────────────────────────────────

    def start_recording(self) -> None:
        if self._recorder is not None and self.is_detection_active:
            self._recorder.start_recording(context=None)

    def stop_recording(self) -> None:
        if self._recorder is not None:
            # Non-blocking: ``stop_recording`` only flips state and
            # dispatches the file release to a background worker.
            # The UI updates immediately because ``is_recording``
            # returns False right after this call. The file is
            # fully written a moment later by the recorder's
            # finalizer thread.
            self._recorder.stop_recording()

    @property
    def is_recording_draining(self) -> bool:
        """True while a previously stopped recording is still being
        flushed to disk. Used by the UI to disable re-recording
        briefly and to show a "saving" hint.
        """
        return self._recorder is not None and self._recorder.is_draining

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
        # Only start detection if open_camera succeeded
        if self.is_camera_open:
            self.start_detection()

    def start(self) -> None:
        """Legacy: start detection if camera is open."""
        if self.is_camera_open and not self.is_detection_active:
            self.start_detection()

    def stop(self) -> None:
        """Legacy: close camera (stops everything).

        On process exit this is the one place that tears down the WS
        client AND the background worker pool. Camera open/close
        cycles intentionally do not stop the WS client (see
        ``close_camera`` docstring).

        Waits for any in-flight close to finish (bounded) so callers
        that need to know "everything is stopped" get a deterministic
        answer.
        """
        if self.is_camera_open or self.is_busy:
            self.close_camera()
            self.wait_closed(timeout=5.0)
        if self._ws_client is not None:
            try:
                self._ws_client.stop()
            except Exception as e:
                self._logger.error("WS client stop failed: %s", e)
            self._ws_client = None
        # Shutdown the worker pool last so any straggler tasks
        # dispatched by the closing pipeline have a chance to run.
        try:
            self._worker_pool.shutdown(timeout=2.0)
        except Exception as e:  # noqa: BLE001
            self._logger.error("Worker pool shutdown failed: %s", e)

    # ── Event handlers ────────────────────────────────────────────────

    def _on_pipeline_fatal_error(self, event: Event) -> None:
        """Handle pipeline fatal error — initiate non-blocking close.

        The pipeline ran out of retries while reading frames. The
        correct response is to take the gateway to IDLE so the user
        can re-open the camera. We dispatch the actual teardown to
        the same background path as ``close_camera`` to keep the
        main thread responsive.
        """
        self._logger.error("Pipeline stopped due to fatal error: %s", event.data)
        self._event_bus.publish(Event(
            type=EventType.CAMERA_READ_FAILED,
            data=event.data,
            source="gateway",
        ))
        # Trigger a non-blocking close. If a close is already in
        # progress this is a no-op.
        if self.is_camera_open:
            self.close_camera()

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
            frame_queue_url=self._frame_queue_url,
            worker_pool=self._worker_pool,
        )

    def _init_ws_client(self) -> None:
        """Start the Feishu WebSocket command listener (idempotent).

        The lark-oapi SDK's ``ws.Client`` owns a private asyncio event
        loop with no public stop API, so we only ever create ONE instance
        per process. Subsequent calls are a no-op.
        """
        if self._ws_client is not None:
            # Already started (or start in progress) — don't restart.
            return
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

        # Unknown commands: the parser saw a leading "/" but the name
        # wasn't in the registry. Reply with a friendly error so the
        # user knows their input was received (rather than the bot
        # silently doing nothing) and point them at /help.
        if cmd.unknown:
            self._logger.info("Rejecting unknown command: /%s", cmd.name)
            if isinstance(self._notifier, FeishuNotifierAdapter):
                self._notifier.send_text(
                    f"[LarkSnap] 未知指令 /{cmd.name}。发送 /help 查看可用命令。"
                )
            return

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
        elif cmd.name == "status":
            s = self._get_state()
            if s == GatewayState.DETECTING:
                status = "running"
            else:
                status = s.value
            notif_on = (
                "enabled" if (self._notification_service
                              and self._notification_service.notification_enabled)
                else "disabled"
            )
            camera = "open" if self.is_camera_open else "closed"
            targets = ",".join(self._config.detector.target_classes) or "(none)"
            interval = self._config.gateway.notification_interval
            self._logger.info("Gateway status: %s, notification: %s", status, notif_on)
            # Reply to the chat so the user actually sees the result
            # of ``/status`` — a log line is not a reply.
            if isinstance(self._notifier, FeishuNotifierAdapter):
                self._notifier.send_text(
                    "[LarkSnap] 当前状态\n"
                    f"  网关: {status}\n"
                    f"  摄像头: {camera}\n"
                    f"  通知: {notif_on}\n"
                    f"  监控类别: {targets}\n"
                    f"  通知间隔: {interval}s"
                )
        elif cmd.name == "help":
            # Two modes:
            #   /help              — render the full command catalogue
            #   /help <command>    — render just the named command's
            #                        syntax and examples
            # The second form is the "drill-down" behaviour suggested
            # by the ``syntax = "/help [command]"`` in the registry.
            self._handle_help_command(cmd)
        elif cmd.name == "config":
            # /config [get|set|show|paths] [path] [json_value]
            # The ConfigService handles all the heavy lifting
            # (path resolution, type coercion, persistence, and
            # live propagation). The controller is just the
            # chat-facing dispatcher: turn the result into a
            # human-readable Feishu reply.
            self._handle_config_command(cmd)

    def _handle_help_command(self, cmd: CommandHandler) -> None:
        """Render help text in response to ``/help`` and push it to chat.

        With no args, the full command catalogue is rendered. With one
        arg, only the matching command's spec is shown (useful when a
        user forgets the exact syntax of a specific command). Unknown
        subcommands fall back to the full catalogue with a note.
        """
        if isinstance(self._notifier, FeishuNotifierAdapter):
            self._notifier.send_text(self._render_help(cmd.args))

    def _handle_config_command(self, cmd: CommandHandler) -> None:
        """Dispatch ``/config`` subcommands and reply to chat.

        The available subcommands are:

        * ``/config`` or ``/config show`` — full config tree
        * ``/config show <prefix>`` — sub-tree (e.g. ``detector``)
        * ``/config get <path>`` — single value
        * ``/config set <path> <json>`` — set + save + live apply
        * ``/config paths [prefix]`` — flat list of all paths

        Anything else returns a usage hint that lists the
        subcommands plus a couple of worked examples.
        """
        if isinstance(self._notifier, FeishuNotifierAdapter):
            self._notifier.send_text(self._render_config(cmd.args))

    def _render_config(self, args: list[str]) -> str:
        """Render the chat reply for a ``/config`` invocation.

        The reply covers every error path explicitly so a user
        can see *why* their command didn't work, not just that
        it didn't.
        """
        from larksnap.utils.exceptions import ConfigError

        # No args → show the full tree.
        if not args or args == ["show"]:
            return self._render_config_show(None)
        sub = args[0].lower()
        if sub == "show":
            prefix = args[1] if len(args) > 1 else None
            return self._render_config_show(prefix)
        if sub == "paths":
            prefix = args[1] if len(args) > 1 else ""
            return self._render_config_paths(prefix)
        if sub == "get":
            if len(args) < 2:
                return self._render_config_usage()
            return self._render_config_get(args[1])
        if sub == "set":
            if len(args) < 3:
                return self._render_config_usage()
            path = args[1]
            # The JSON value can be more than one token (e.g. a
            # JSON array ``["person","car"]`` arrives as multiple
            # tokens because shlex splits on the commas and quotes).
            # Join with a single space — the service uses
            # ``json.loads`` to parse the joined string, so a
            # well-formed JSON literal still round-trips.
            json_value = " ".join(args[2:])
            try:
                old, new, status = self._config_service.set(path, json_value)
            except ConfigError as e:
                return f"[LarkSnap] /config set 失败: {e}"
            marker = "（需重启）" if status == "restart_required" else "（已生效）"
            saved = self._config_service.config_path
            saved_str = str(saved) if saved is not None else "<内存>"
            return (
                f"[LarkSnap] 已更新: {path} {marker}\n"
                f"  旧值: {old!r}\n"
                f"  新值: {new!r}\n"
                f"  已保存到 {saved_str}"
            )
        return self._render_config_usage()

    def _render_config_show(self, prefix: str | None) -> str:
        """Render a sub-tree of the config as ``path = value`` lines."""
        from larksnap.utils.exceptions import ConfigError

        if prefix is not None:
            try:
                self._config_service.get(prefix)
            except ConfigError as e:
                return f"[LarkSnap] /config show 失败: {e}"
        lines = [f"[LarkSnap] 配置视图 {prefix or '<root>'}", ""]
        for path in self._config_service.list_paths(prefix or ""):
            try:
                value = self._config_service.get(path)
            except Exception:
                # Don't let one bad leaf block the whole view.
                continue
            restart = " [需重启]" if self._config_service.needs_restart(path) else ""
            lines.append(f"  {path} = {value!r}{restart}")
        if len(lines) == 2:
            return f"[LarkSnap] 前缀 {prefix!r} 下没有可显示的配置项"
        return "\n".join(lines)

    def _render_config_paths(self, prefix: str) -> str:
        """Render a flat list of all available config paths."""
        paths = self._config_service.list_paths(prefix)
        if not paths:
            return f"[LarkSnap] 前缀 {prefix!r} 下没有匹配的路径"
        header = f"[LarkSnap] 可配置路径 ({len(paths)}):"
        body = "\n".join(f"  {p}" for p in paths)
        return f"{header}\n{body}"

    def _render_config_get(self, path: str) -> str:
        """Render a single ``get`` reply."""
        from larksnap.utils.exceptions import ConfigError

        try:
            value = self._config_service.get(path)
        except ConfigError as e:
            return f"[LarkSnap] /config get 失败: {e}"
        return f"[LarkSnap] {path} = {value!r}"

    def _render_config_usage(self) -> str:
        """Friendly reminder of the subcommands and a few examples."""
        return (
            "[LarkSnap] /config 用法:\n"
            "  /config                          显示全部配置\n"
            "  /config show [prefix]            显示子树 (例: /config show detector)\n"
            "  /config paths [prefix]           列出所有路径\n"
            "  /config get <path>               获取单个值\n"
            "  /config set <path> <json>        设置单个值 (例: /config set gateway.notification_interval 60)\n"
            "  值使用 JSON 字面量: 数字 30、字符串 \"person\"、列表 [\"a\",\"b\"]、布尔 true"
        )

    def _render_help(self, args: list[str]) -> str:
        """Build the help text for ``/help [command]``.

        Pure function of ``args`` and the registry, which keeps it
        trivial to unit-test without spinning up a notifier.
        """
        if args:
            target = args[0].lstrip("/").lower()
            spec = CommandRegistry.get(target)
            if spec is not None:
                return _render_single_help(spec)
            # Unknown subcommand — surface the full catalogue plus a
            # short note so the user knows we didn't ignore them.
            return (
                f"[LarkSnap] 未找到指令 /{target}。\n\n"
                + CommandRegistry.render_help()
            )
        return CommandRegistry.render_help()

    def _release_adapters(self) -> None:
        """Release all adapter resources (best-effort, never raises).

        Uses ``_safe_release_adapter`` so a single adapter's failure
        can't stop the rest from being cleaned up. This is the
        production-grade guarantee: the gateway MUST end up with
        ``None`` references and the IDLE state, no matter what.
        """
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                self._logger.error("Pipeline stop during release failed: %s", e)
            self._pipeline = None

        for adapter in (self._notifier, self._detector, self._camera, self._recorder):
            _safe_release_adapter(adapter, self._logger)

        self._notifier = None
        self._detector = None
        self._camera = None
        self._recorder = None
        self._notification_service = None
