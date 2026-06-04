import hashlib
import logging
import time
from pathlib import Path

import httpx

from larksnap.adapters.notifier.interface import NotificationMessage, NotifierAdapter
from larksnap.config.models import NotifierConfig
from larksnap.utils.exceptions import NotifierError

_FEISHU_BASE_URL = "https://open.feishu.cn/open-apis"


class FeishuNotifierAdapter(NotifierAdapter):
    """Feishu (Lark) notifier adapter using webhook and app API."""

    def __init__(self, config: NotifierConfig) -> None:
        """Initialize the Feishu notifier with configuration."""
        self._config = config
        self._logger = logging.getLogger("larksnap.notifier.feishu")
        self._client: httpx.Client | None = None
        self._tenant_token: str | None = None
        self._token_expires: float = 0.0

    def connect(self) -> None:
        """Initialize the HTTP client for Feishu API."""
        try:
            self._client = httpx.Client(timeout=30.0)
            self._logger.info("Feishu notifier connected")
        except Exception as e:
            raise NotifierError(f"Failed to connect to Feishu: {e}") from e

    def send_message(self, message: NotificationMessage) -> bool:
        """Send a notification message via Feishu.

        If send_image is enabled and snapshot_path is provided, sends the
        image to the configured chat. Otherwise falls back to webhook text.
        Returns False if Feishu is not configured (not an error).
        """
        if self._client is None:
            self._logger.warning(
                "Feishu client not connected, skipping notification"
            )
            return False

        # Check if any Feishu channel is configured
        has_app_creds = bool(
            self._config.app_id and self._config.app_secret
        )
        has_webhook = bool(self._config.webhook_url)

        if not has_app_creds and not has_webhook:
            self._logger.warning(
                "Feishu not configured (no app_id/app_secret or webhook_url), "
                "skipping notification"
            )
            return False

        # Try image message first if configured
        if (
            self._config.send_image
            and message.snapshot_path
            and has_app_creds
            and self._config.chat_id
        ):
            return self._send_image_message(message)

        # Fallback to webhook text message
        return self._send_webhook_text(message)

    def disconnect(self) -> None:
        """Close the HTTP client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._tenant_token = None
            self._logger.info("Feishu notifier disconnected")

    def _get_tenant_token(self) -> str:
        """Get or refresh the tenant_access_token."""
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

        # Calculate SHA-256 checksum for validation
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
            return self._send_webhook_text(message)

        for attempt in range(self._config.retry.max_retries):
            try:
                resp = self._client.post(  # type: ignore[union-attr]
                    f"{_FEISHU_BASE_URL}/im/v1/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"receive_id": self._config.chat_id},
                    json={
                        "receive_id": self._config.chat_id,
                        "msg_type": "image",
                        "content": f'{{"image_key": "{image_key}"}}',
                    },
                )
                data = resp.json()
                if data.get("code") == 0:
                    self._logger.info("Feishu image message sent successfully")
                    # Also send a text notification as companion
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
                params={"receive_id": self._config.chat_id},
                json={
                    "receive_id": self._config.chat_id,
                    "msg_type": "text",
                    "content": f'{{"text": "{content}"}}',
                },
            )
        except httpx.RequestError:
            self._logger.warning("Failed to send companion text message")

    def _send_webhook_text(self, message: NotificationMessage) -> bool:
        """Send a text notification via Feishu webhook."""
        if not self._config.webhook_url:
            self._logger.warning(
                "Feishu webhook URL not configured, skipping notification"
            )
            return False

        payload = self._build_webhook_payload(message)

        for attempt in range(self._config.retry.max_retries):
            try:
                response = self._client.post(  # type: ignore[union-attr]
                    self._config.webhook_url, json=payload
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get("code", -1) == 0:
                        self._logger.info(
                            "Feishu webhook notification sent successfully"
                        )
                        return True
                    self._logger.warning(
                        "Feishu API returned error: %s",
                        result.get("msg", "unknown"),
                    )
                else:
                    self._logger.warning(
                        "Feishu API returned status %d", response.status_code
                    )
            except httpx.RequestError as e:
                self._logger.warning(
                    "Feishu notification attempt %d failed: %s", attempt + 1, e
                )

            if attempt < self._config.retry.max_retries - 1:
                time.sleep(self._config.retry.retry_interval)

        self._logger.error(
            "Feishu notification failed after %d retries",
            self._config.retry.max_retries,
        )
        return False

    def _build_webhook_payload(self, message: NotificationMessage) -> dict:
        """Build the Feishu webhook request payload from a notification message."""
        content = self._config.message_template.format(
            label=message.label,
            confidence=message.confidence,
            timestamp=message.timestamp,
            snapshot_path=message.snapshot_path or "",
        )
        return {
            "msg_type": "text",
            "content": f'{{"text": "{content}"}}',
        }
