"""Regression tests for Feishu ``content`` JSON encoding.

The Feishu ``im/v1/messages`` endpoint requires the ``content`` field
to be a *JSON string* whose decoded form matches the declared
``msg_type``. Building that string with f-string interpolation
silently breaks the moment the payload contains a ``"``, a
newline, a backslash, or any other character that needs JSON
escaping — the resulting string is no longer valid JSON and the
API returns ``ext=content is not a string in json format``.

The bug originally surfaced when a user sent ``/help`` from Feishu:
the rendered help text contains the example
``/config set detector.target_classes ["person","car","dog"]``,
whose embedded ``"`` corrupted the f-string envelope. The fix
introduces :py:func:`larksnap.adapters.notifier.feishu_adapter.
_feishu_content`, which always routes the payload through
:py:func:`json.dumps`.

These tests pin the behaviour for ``send_text`` and the
text-message path of ``send_message`` so a future f-string
regression cannot land unnoticed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from larksnap.adapters.notifier.feishu_adapter import (
    FeishuNotifierAdapter,
    _feishu_content,
)
from larksnap.adapters.notifier.interface import NotificationMessage
from larksnap.config.models import NotifierConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_notifier(
    *,
    chat_id: str = "oc_test_chat",
    app_id: str = "cli_test",
    app_secret: str = "secret_test",
) -> FeishuNotifierAdapter:
    """Construct a Feishu notifier stubbed with a fake HTTP client.

    The fake client captures the last ``post`` call so tests can
    inspect the exact ``content`` payload that would have been
    sent to the Feishu API. By default it replies with a success
    payload (``code == 0``); tests that need an error reply can
    reassign ``notifier._client.post.return_value``.
    """
    cfg = NotifierConfig(
        app_id=app_id,
        app_secret=app_secret,
        chat_id=chat_id,
        message_template=(
            "[LarkSnap] {label} {confidence:.0%} {timestamp} {snapshot_path}"
        ),
    )
    notifier = FeishuNotifierAdapter(cfg)
    # Skip the real ``connect`` (no real httpx.Client, no auto-detect).
    notifier._client = MagicMock()
    notifier._tenant_token = "test-tenant-token"
    notifier._token_expires = 9e18  # never expires during the test
    # Default reply: Feishu "code 0" success envelope. Tests that
    # exercise the error branch can override ``return_value``.
    notifier._client.post.return_value = MagicMock(
        json=lambda: {"code": 0, "msg": "success", "data": {}}
    )
    return notifier


def _last_post_json(mock_client: MagicMock) -> dict:
    """Return the ``json=`` body from the most recent ``post`` call."""
    assert mock_client.post.called, "expected Feishu client to be called"
    call = mock_client.post.call_args
    # ``json=`` may be passed as a positional or keyword arg.
    if "json" in call.kwargs:
        return call.kwargs["json"]
    return call.args[1] if len(call.args) > 1 else {}


# ---------------------------------------------------------------------------
# The _feishu_content helper itself
# ---------------------------------------------------------------------------


class TestFeishuContentHelper:
    def test_round_trips_simple_text(self) -> None:
        encoded = _feishu_content({"text": "hello"})
        assert json.loads(encoded) == {"text": "hello"}

    def test_escapes_double_quotes(self) -> None:
        # The exact failure mode from /help — embedded quotes must
        # be backslash-escaped, not echoed raw.
        encoded = _feishu_content(
            {"text": '/config set target ["person","car"]'}
        )
        assert json.loads(encoded) == {
            "text": '/config set target ["person","car"]'
        }
        # The raw encoded string must NOT contain the broken f-string
        # form (which would have a literal " inside a JSON value).
        # If the bug regresses the encoded form is invalid JSON
        # and the round-trip above would raise.
        assert encoded.startswith("{") and encoded.endswith("}")

    def test_escapes_backslashes(self) -> None:
        encoded = _feishu_content({"text": "path\\to\\thing"})
        assert json.loads(encoded) == {"text": "path\\to\\thing"}

    def test_escapes_newlines(self) -> None:
        # Multi-line text must survive intact after a decode/encode
        # round-trip.
        encoded = _feishu_content({"text": "line1\nline2\nline3"})
        assert json.loads(encoded) == {"text": "line1\nline2\nline3"}

    def test_preserves_unicode(self) -> None:
        # ``ensure_ascii=False`` keeps non-ASCII characters readable
        # in the Feishu UI (important for Chinese reply text).
        encoded = _feishu_content({"text": "检测到 person"})
        assert "检测到" in encoded
        assert json.loads(encoded) == {"text": "检测到 person"}


# ---------------------------------------------------------------------------
# send_text — the original failure site
# ---------------------------------------------------------------------------


class TestSendTextEncoding:
    def test_plain_text_round_trips(self) -> None:
        notifier = _build_notifier()
        ok = notifier.send_text("[LarkSnap] hello")
        assert ok is True
        body = _last_post_json(notifier._client)
        assert body["msg_type"] == "text"
        # The ``content`` must be a valid JSON string, not a raw
        # string with broken escapes.
        content = body["content"]
        assert isinstance(content, str)
        decoded = json.loads(content)
        assert decoded == {"text": "[LarkSnap] hello"}

    def test_help_text_with_embedded_quotes_does_not_break_json(self) -> None:
        """The original bug: ``/help`` produced text containing ``"``."""
        notifier = _build_notifier()
        help_like = (
            "/config - 修改配置\n"
            "  示例: /config set detector.target_classes "
            '["person","car","dog"]\n'
            "  别名: /state"
        )
        ok = notifier.send_text(help_like)
        assert ok is True
        body = _last_post_json(notifier._client)
        # Critical: the API only accepts valid JSON here. The naive
        # f-string envelope would have produced
        # ``{"text": "...["person","car","dog"]..."}`` which is
        # unparseable. json.loads on it must succeed.
        decoded = json.loads(body["content"])
        assert decoded["text"] == help_like

    def test_text_with_newlines_does_not_break_json(self) -> None:
        notifier = _build_notifier()
        multiline = "[LarkSnap] line 1\nline 2\nline 3"
        ok = notifier.send_text(multiline)
        assert ok is True
        body = _last_post_json(notifier._client)
        decoded = json.loads(body["content"])
        assert decoded["text"] == multiline

    def test_text_with_chinese_does_not_break_json(self) -> None:
        notifier = _build_notifier()
        text = "[LarkSnap] 已更新: detector.target_classes （已生效）"
        ok = notifier.send_text(text)
        assert ok is True
        body = _last_post_json(notifier._client)
        decoded = json.loads(body["content"])
        assert decoded["text"] == text

    def test_skips_when_chat_id_missing(self) -> None:
        notifier = _build_notifier(chat_id="")
        ok = notifier.send_text("anything")
        assert ok is False
        notifier._client.post.assert_not_called()

    def test_returns_false_on_api_error(self) -> None:
        notifier = _build_notifier()
        # Simulate a 200 OK with a Feishu-level error code.
        notifier._client.post.return_value = MagicMock(
            json=lambda: {"code": 230001, "msg": "invalid content"}
        )
        ok = notifier.send_text("hello")
        assert ok is False


# ---------------------------------------------------------------------------
# send_message text path — same encoding rules apply
# ---------------------------------------------------------------------------


class TestSendMessageTextEncoding:
    def _build(self) -> FeishuNotifierAdapter:
        return _build_notifier()

    def test_text_message_with_special_chars_round_trips(self) -> None:
        notifier = self._build()
        # Force the text path by disabling send_image.
        notifier._config.send_image = False
        notifier._config.message_template = "[LarkSnap] {label}"
        msg = NotificationMessage(
            title="",
            content="ignored",
            label='tricky "label" with [brackets]',
            confidence=0.9,
            timestamp="2024-01-01 00:00:00",
        )
        ok = notifier.send_message(msg)
        assert ok is True
        body = _last_post_json(notifier._client)
        assert body["msg_type"] == "text"
        decoded = json.loads(body["content"])
        assert decoded["text"] == "[LarkSnap] tricky \"label\" with [brackets]"
