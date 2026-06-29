"""Feishu notifier adapter using app API for image and text messages.

Concurrency:
  - ``set_chat_id`` and ``_persist_chat_id`` are protected by a
    write lock so a UI-thread save can't race with a WS-client
    thread save (which used to be able to corrupt the YAML file).
  - The tenant access token cache is guarded by its own lock so
    concurrent notification dispatches on the worker pool don't
    each fire a token refresh and stampede the Feishu auth API.
  - The httpx client is shared across threads; httpx is documented
    as thread-safe for the ``Client`` class, so no extra lock is
    needed around ``self._client.post``.

Message content encoding:
  - Feishu's ``im/v1/messages`` endpoint requires the ``content``
    field to be a *JSON-encoded string* whose value matches the
    declared ``msg_type``. Naive f-string interpolation
    (``f'{{"text": "{text}"}}'``) silently breaks the moment the
    payload contains a ``"``, a newline, a backslash, or any other
    character that needs JSON escaping — the resulting string is
    not valid JSON and the API returns
    ``ext=content is not a string in json format``. We always
    build the content envelope with :py:func:`json.dumps` so the
    serializer handles escaping correctly.
"""

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import httpx

from larksnap.adapters.notifier.interface import NotificationMessage, NotifierAdapter
from larksnap.adapters.registry import notifier_registry
from larksnap.config.models import NotifierConfig
from larksnap.utils.exceptions import NotifierError

_FEISHU_BASE_URL = "https://open.feishu.cn/open-apis"


def _feishu_content(payload: dict) -> str:
    """Serialize a Feishu message payload to the JSON string the API wants.

    Feishu's ``im/v1/messages`` endpoint takes a ``content`` field
    that is *itself* a JSON string. Building it via f-string
    interpolation is unsafe — any ``"``, ``\\``, or control
    character in the payload corrupts the outer JSON and the API
    rejects the request. Routing through :py:func:`json.dumps`
    keeps the envelope valid no matter what the payload contains.
    """
    return json.dumps(payload, ensure_ascii=False)


@notifier_registry.register("feishu")
class FeishuNotifierAdapter(NotifierAdapter):
    """Feishu (Lark) notifier adapter using app API."""

    def __init__(self, config: NotifierConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("larksnap.notifier.feishu")
        self._client: httpx.Client | None = None
        self._tenant_token: str | None = None
        self._token_expires: float = 0.0
        # Serialise token refreshes so multiple workers don't all
        # hit the auth endpoint at once. RLock (not Lock) so a
        # caller that already holds the lock can re-enter.
        self._token_lock = threading.RLock()
        # Serialise config writes. The UI thread (settings dialog)
        # and the WS client thread (chat_id learned from /init) can
        # both invoke ``set_chat_id`` simultaneously; without this
        # lock, two YAML writes can interleave and corrupt the file.
        self._config_lock = threading.Lock()

    def connect(self) -> None:
        """Initialize the HTTP client for Feishu API."""
        try:
            self._client = httpx.Client(timeout=30.0)
            self._logger.info("Feishu notifier connected")
            # Auto-detect chat_id from existing bot chats if not configured
            if not self._config.chat_id:
                self._auto_detect_chat_id()
        except Exception as e:
            raise NotifierError(f"Failed to connect to Feishu: {e}") from e

    def _auto_detect_chat_id(self) -> None:
        """Try to auto-detect chat_id from bot's existing chats via API."""
        try:
            token = self._get_tenant_token()
            if not token:
                return
            resp = self._client.get(
                f"{_FEISHU_BASE_URL}/im/v1/chats",
                headers={"Authorization": f"Bearer {token}"},
                params={"page_size": 20},
            )
            data = resp.json()
            if data.get("code") != 0:
                self._logger.debug("Failed to list chats: %s", data.get("msg"))
                return
            items = data.get("data", {}).get("items", [])
            if items:
                # Use the first available P2P chat
                chat_id = items[0].get("chat_id", "")
                if chat_id:
                    self._config.chat_id = chat_id
                    self._persist_chat_id(chat_id)
                    self._logger.info(
                        "Auto-detected chat_id from bot chats: %s", chat_id
                    )
        except Exception as e:
            self._logger.debug("Auto-detect chat_id failed: %s", e)

    def set_chat_id(self, chat_id: str) -> None:
        """Update the target chat ID for notifications and persist to config.

        Holds ``_config_lock`` for the whole read-modify-write of
        the config file. The lock prevents a second ``set_chat_id``
        call (e.g. from the WS client thread receiving a fresh
        ``/init``) from racing the YAML read+modify+write here and
        producing a corrupted file with the keys out of order or
        the chat_id set to a stale value.
        """
        if not chat_id:
            return
        with self._config_lock:
            if chat_id == self._config.chat_id:
                return
            self._config.chat_id = chat_id
            self._logger.info("Notification target chat updated: %s", chat_id)
            self._persist_chat_id(chat_id)

    def _persist_chat_id(self, chat_id: str) -> None:
        """Save chat_id to config file so it persists across restarts."""
        try:
            from larksnap.config.loader import save_config
            from larksnap.config.models import AppConfig

            config_path = str(
                Path(__file__).parent.parent.parent.parent.parent
                / "config" / "config.yaml"
            )
            path = Path(config_path)
            if not path.exists():
                return

            with open(path, encoding="utf-8") as f:
                import yaml
                raw = yaml.safe_load(f) or {}

            if "notifier" not in raw:
                raw["notifier"] = {}
            raw["notifier"]["chat_id"] = chat_id

            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            self._logger.info("Chat ID persisted to config file")
        except Exception as e:
            self._logger.warning("Failed to persist chat_id to config: %s", e)

    def send_message(self, message: NotificationMessage) -> bool:
        """Send a notification message via Feishu app API.

        If send_image is enabled and snapshot_path is provided, sends the
        image to the configured chat. Otherwise sends a text message.
        Returns False if Feishu is not configured (not an error).
        """
        if self._client is None:
            self._logger.warning(
                "Feishu client not connected, skipping notification"
            )
            return False

        if not self._config.app_id or not self._config.app_secret:
            self._logger.warning(
                "Feishu not configured (no app_id/app_secret), "
                "skipping notification"
            )
            return False

        # Try image message first if configured
        if (
            self._config.send_image
            and message.snapshot_path
            and self._config.chat_id
        ):
            return self._send_image_message(message)

        # Fallback to text message
        if self._config.chat_id:
            return self._send_text_message(message)

        self._logger.warning(
            "Feishu chat_id not configured, skipping notification"
        )
        return False

    def disconnect(self) -> None:
        """Close the HTTP client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._tenant_token = None
            self._logger.info("Feishu notifier disconnected")

    def _get_tenant_token(self) -> str:
        """Get or refresh the tenant_access_token.

        Held under ``_token_lock`` so concurrent notification
        dispatches on the worker pool don't all fire the same
        auth request at once (the Feishu auth endpoint rate-limits
        and will return 99991663 if you hammer it). The lock is
        RLock so an outer caller that's already inside the critical
        section (e.g. while holding it to chain token-aware calls)
        can re-enter without deadlocking.
        """
        with self._token_lock:
            if (
                self._tenant_token
                and time.time() < self._token_expires
            ):
                return self._tenant_token

            if self._client is None:
                raise NotifierError("Feishu notifier not connected")

            resp = self._client.post(
                f"{_FEISHU_BASE_URL}/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self._config.app_id,
                    "app_secret": self._config.app_secret,
                },
            )
            data = resp.json()
            if data.get("code") != 0:
                raise NotifierError(
                    f"Failed to get tenant token: {data.get('msg', 'unknown')}"
                )

            self._tenant_token = data["tenant_access_token"]
            expire = data.get("expire", 7200)
            self._token_expires = time.time() + expire - 300  # refresh 5 min early
            self._logger.info("Feishu tenant token refreshed")
            return self._tenant_token

    def _upload_image(self, image_path: str) -> str:
        """Upload an image to Feishu and return the image_key."""
        token = self._get_tenant_token()

        path = Path(image_path)
        if not path.exists():
            raise NotifierError(f"Image file not found: {image_path}")

        with open(path, "rb") as f:
            image_data = f.read()

        checksum = hashlib.sha256(image_data).hexdigest()

        resp = self._client.post(  # type: ignore[union-attr]
            f"{_FEISHU_BASE_URL}/im/v1/images",
            headers={"Authorization": f"Bearer {token}"},
            data={"image_type": "message"},
            files={"image": (path.name, image_data, "image/jpeg")},
        )
        data = resp.json()
        if data.get("code") != 0:
            raise NotifierError(
                f"Failed to upload image: {data.get('msg', 'unknown')}"
            )

        image_key = data["data"]["image_key"]
        self._logger.info(
            "Image uploaded: %s (key=%s, sha256=%s)",
            path.name, image_key, checksum[:16],
        )
        return image_key

    def _send_image_message(self, message: NotificationMessage) -> bool:
        """Send an image message to a Feishu chat using app API."""
        token = self._get_tenant_token()

        try:
            image_key = self._upload_image(message.snapshot_path)  # type: ignore[arg-type]
        except NotifierError:
            self._logger.exception("Image upload failed, falling back to text")
            return self._send_text_message(message)

        for attempt in range(self._config.retry.max_retries):
            try:
                resp = self._client.post(  # type: ignore[union-attr]
                    f"{_FEISHU_BASE_URL}/im/v1/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "receive_id_type": "chat_id",
                        "receive_id": self._config.chat_id,
                    },
                    json={
                        "receive_id": self._config.chat_id,
                        "msg_type": "image",
                        "content": _feishu_content({"image_key": image_key}),
                    },
                )
                data = resp.json()
                if data.get("code") == 0:
                    self._logger.info("Feishu image message sent successfully")
                    self._send_text_to_chat(token, message)
                    return True
                self._logger.warning(
                    "Feishu image message API error: %s",
                    data.get("msg", "unknown"),
                )
            except httpx.RequestError as e:
                self._logger.warning(
                    "Feishu image message attempt %d failed: %s",
                    attempt + 1, e,
                )

            if attempt < self._config.retry.max_retries - 1:
                time.sleep(self._config.retry.retry_interval)

        self._logger.error(
            "Feishu image message failed after %d retries",
            self._config.retry.max_retries,
        )
        return False

    def _send_text_message(self, message: NotificationMessage) -> bool:
        """Send a text message to the configured Feishu chat."""
        token = self._get_tenant_token()
        content = self._config.message_template.format(
            label=message.label,
            confidence=message.confidence,
            timestamp=message.timestamp,
            snapshot_path=message.snapshot_path or "",
        )

        for attempt in range(self._config.retry.max_retries):
            try:
                resp = self._client.post(  # type: ignore[union-attr]
                    f"{_FEISHU_BASE_URL}/im/v1/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "receive_id_type": "chat_id",
                        "receive_id": self._config.chat_id,
                    },
                    json={
                        "receive_id": self._config.chat_id,
                        "msg_type": "text",
                        "content": _feishu_content({"text": content}),
                    },
                )
                data = resp.json()
                if data.get("code") == 0:
                    self._logger.info("Feishu text message sent successfully")
                    return True
                self._logger.warning(
                    "Feishu text message API error: %s",
                    data.get("msg", "unknown"),
                )
            except httpx.RequestError as e:
                self._logger.warning(
                    "Feishu text message attempt %d failed: %s",
                    attempt + 1, e,
                )

            if attempt < self._config.retry.max_retries - 1:
                time.sleep(self._config.retry.retry_interval)

        self._logger.error(
            "Feishu text message failed after %d retries",
            self._config.retry.max_retries,
        )
        return False

    def _send_text_to_chat(
        self, token: str, message: NotificationMessage
    ) -> None:
        """Send a text message to the configured chat as companion."""
        content = self._config.message_template.format(
            label=message.label,
            confidence=message.confidence,
            timestamp=message.timestamp,
            snapshot_path=message.snapshot_path or "",
        )
        try:
            self._client.post(  # type: ignore[union-attr]
                f"{_FEISHU_BASE_URL}/im/v1/messages",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "receive_id_type": "chat_id",
                    "receive_id": self._config.chat_id,
                },
                json={
                    "receive_id": self._config.chat_id,
                    "msg_type": "text",
                    "content": _feishu_content({"text": content}),
                },
            )
        except httpx.RequestError:
            self._logger.warning("Failed to send companion text message")

    def send_text(self, text: str) -> bool:
        """Send a plain text message to the configured chat (for command responses)."""
        if not self._client or not self._config.chat_id:
            return False
        try:
            token = self._get_tenant_token()
            resp = self._client.post(
                f"{_FEISHU_BASE_URL}/im/v1/messages",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "receive_id_type": "chat_id",
                    "receive_id": self._config.chat_id,
                },
                json={
                    "receive_id": self._config.chat_id,
                    "msg_type": "text",
                    "content": _feishu_content({"text": text}),
                },
            )
            data = resp.json()
            if data.get("code") == 0:
                self._logger.info("Feishu text sent: %s", text[:50])
                return True
            self._logger.warning("Feishu text send error: %s", data.get("msg"))
            return False
        except Exception as e:
            self._logger.warning("Failed to send text: %s", e)
            return False
