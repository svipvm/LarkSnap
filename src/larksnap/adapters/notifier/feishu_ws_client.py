"""Feishu WebSocket event subscription client for receiving commands.

Uses the lark-oapi SDK to establish a long-lived WebSocket connection
for receiving message events from Feishu. Supports commands like
start/stop/status to control the gateway.
"""

import json
import logging
import threading
from collections.abc import Callable

from larksnap.config.models import NotifierConfig
from larksnap.utils.exceptions import NotifierError

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

    _LARK_SDK_AVAILABLE = True
except ImportError:
    _LARK_SDK_AVAILABLE = False


class CommandHandler:
    """Parsed command from a Feishu message."""

    __slots__ = ("name", "args", "chat_id", "message_id")

    def __init__(
        self,
        name: str,
        args: list[str] | None = None,
        chat_id: str = "",
        message_id: str = "",
    ) -> None:
        self.name = name
        self.args = args or []
        self.chat_id = chat_id
        self.message_id = message_id


class FeishuWSClient:
    """Feishu WebSocket long-connection client for receiving commands.

    Subscribes to im.message.receive_v1 events via the lark-oapi SDK
    WebSocket transport. Parses incoming text messages as commands and
    dispatches them to a registered callback.
    """

    # Supported commands
    COMMANDS = ("init", "start", "stop", "status", "help")

    def __init__(
        self,
        config: NotifierConfig,
        on_command: Callable[[CommandHandler], None],
    ) -> None:
        if not _LARK_SDK_AVAILABLE:
            raise NotifierError(
                "lark-oapi package is required for WebSocket mode. "
                "Install it with: uv add lark-oapi"
            )

        self._config = config
        self._on_command = on_command
        self._logger = logging.getLogger("larksnap.notifier.feishu_ws")
        self._client: lark.ws.Client | None = None  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the WebSocket client in a background thread.

        Note: the lark-oapi ``ws.Client.start()`` is a blocking call that
        owns a private asyncio event loop. The SDK has no public stop
        API and no clean way to interrupt ``start()``. Callers should
        therefore treat the WS client as a process-lifetime singleton —
        start it once, leave it running across camera open/close cycles,
        and only tear it down on application exit.
        """
        if self._running:
            self._logger.warning("Feishu WS client is already running")
            return

        if not self._config.app_id or not self._config.app_secret:
            self._logger.warning(
                "Feishu app_id/app_secret not configured, "
                "WebSocket command listener disabled"
            )
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._logger.info("Feishu WS client starting...")

    def stop(self) -> None:
        """Best-effort shutdown.

        The lark-oapi SDK does not expose a public stop API, so we cannot
        guarantee that the internal ``start()`` call returns. The thread
        is a daemon — it will be killed when the process exits.

        We also silence the ``RuntimeWarning: coroutine 'Client._disconnect'
        was never awaited`` warning that the lark SDK's GC emits when the
        event loop is torn down without proper cleanup.
        """
        import warnings

        if not self._running and self._thread is None:
            return

        self._running = False
        # Drop our reference to the SDK client. lark's GC will close the
        # underlying WebSocket and emit the never-awaited coroutine warning.
        self._client = None

        if self._thread is not None and self._thread.is_alive():
            # Don't block the main thread waiting for the SDK to give up.
            # The thread is a daemon and will die with the process.
            self._thread.join(timeout=1.0)
        self._thread = None
        self._logger.info("Feishu WS client stopped")

    def _run(self) -> None:
        """Run the WebSocket client (blocking)."""
        import warnings

        # The lark-oapi SDK creates coroutines for connect / disconnect /
        # ping / receive that, when the SDK's internal loop is torn down
        # abruptly (process exit or restart), are never awaited and trigger
        # noisy RuntimeWarnings. We can't fix the SDK, so we suppress them
        # in this thread only.
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"lark.*")

        try:
            event_handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(self._on_message)
                .build()
            )

            self._client = lark.ws.Client(  # type: ignore[attr-defined]
                self._config.app_id,
                self._config.app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO,
            )

            self._logger.info("Feishu WS client connecting...")
            self._client.start()
            self._logger.info("Feishu WS client disconnected")
        except Exception as e:
            if self._running:
                self._logger.error("Feishu WS client error: %s", e)
            self._running = False

    def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message events from Feishu."""
        try:
            if data.event is None:
                return

            message = data.event.message
            if message is None:
                return

            # Only handle text messages
            msg_type = message.message_type
            if msg_type != "text":
                return

            chat_id = message.chat_id or ""
            message_id = message.message_id or ""

            # Parse text content
            content_str = message.content or "{}"
            content = json.loads(content_str)
            text = content.get("text", "").strip()

            if not text:
                return

            # Parse command
            cmd = self._parse_command(text, chat_id, message_id)
            if cmd is not None:
                self._logger.info(
                    "Received command: %s %s (chat=%s)",
                    cmd.name,
                    cmd.args,
                    cmd.chat_id,
                )
                self._on_command(cmd)

        except Exception as e:
            self._logger.error("Error handling Feishu message: %s", e)

    def _parse_command(
        self, text: str, chat_id: str, message_id: str
    ) -> CommandHandler | None:
        """Parse a text message into a CommandHandler.

        Supports formats:
          - /start
          - /stop
          - /status
          - /help
        """
        if not text.startswith("/"):
            return None

        parts = text.split()
        cmd_name = parts[0][1:].lower()  # strip leading "/"

        if cmd_name not in self.COMMANDS:
            self._logger.debug("Unknown command: /%s", cmd_name)
            return None

        return CommandHandler(
            name=cmd_name,
            args=parts[1:],
            chat_id=chat_id,
            message_id=message_id,
        )
