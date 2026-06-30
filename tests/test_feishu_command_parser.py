"""Tests for the Feishu WebSocket command parser.

The parser is exposed as ``FeishuWSClient._parse_command``. We bypass
``__init__`` (which would otherwise require a live lark-oapi install
and configuration) and only stand up the bits the parser needs: a
logger attribute and a class-level reference to ``CommandRegistry``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from larksnap.adapters.notifier.command_registry import CommandRegistry
from larksnap.adapters.notifier.feishu_ws_client import (
    CommandHandler,
    FeishuWSClient,
    _strip_at_mention,
)


@pytest.fixture
def client() -> FeishuWSClient:
    """Bare WS client with just the bits the parser needs."""
    inst = FeishuWSClient.__new__(FeishuWSClient)
    inst._logger = logging.getLogger("larksnap.test.feishu_ws")
    return inst


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    """Snapshot the registry around each test (see registry tests)."""
    saved_specs = dict(CommandRegistry._specs)
    saved_aliases = dict(CommandRegistry._aliases)
    try:
        yield
    finally:
        CommandRegistry._specs = saved_specs
        CommandRegistry._aliases = saved_aliases


class TestStripAtMention:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("@bot /start", "/start"),
            ("@_user_1 /start", "/start"),
            ("@_user_1 /start extra", "/start extra"),
            ("/start", "/start"),
            ("", ""),
            ("hello /start", "hello /start"),  # mid-text mention not stripped
        ],
    )
    def test_strip_mentions(self, raw: str, expected: str) -> None:
        assert _strip_at_mention(raw) == expected


class TestParseCommandHappyPath:
    @pytest.mark.parametrize(
        "text,expected_name,expected_args",
        [
            ("/start", "start", []),
            ("/start extra args", "start", ["extra", "args"]),
            ("/stop", "stop", []),
            ("/init", "init", []),
            ("/status", "status", []),
            ("/help", "help", []),
            ("/help start", "help", ["start"]),
        ],
    )
    def test_basic_commands(
        self,
        client: FeishuWSClient,
        text: str,
        expected_name: str,
        expected_args: list[str],
    ) -> None:
        cmd = client._parse_command(text, "chat-1", "msg-1")
        assert cmd is not None
        assert cmd.name == expected_name
        assert cmd.args == expected_args
        assert cmd.unknown is False
        assert cmd.chat_id == "chat-1"
        assert cmd.message_id == "msg-1"

    def test_case_insensitive(self, client: FeishuWSClient) -> None:
        cmd = client._parse_command("/STATUS", "c", "m")
        assert cmd is not None
        assert cmd.name == "status"

    def test_alias_resolution(self, client: FeishuWSClient) -> None:
        # ``state`` is registered as an alias of ``status``.
        cmd = client._parse_command("/state", "c", "m")
        assert cmd is not None
        assert cmd.name == "state"
        assert cmd.unknown is False
        # The registry resolves it back to the primary spec.
        assert CommandRegistry.get(cmd.name).name == "status"

    def test_group_chat_mention_stripped(self, client: FeishuWSClient) -> None:
        cmd = client._parse_command("@_user_1 /start", "c", "m")
        assert cmd is not None
        assert cmd.name == "start"
        assert cmd.args == []

    def test_quoted_arg_preserved(self, client: FeishuWSClient) -> None:
        cmd = client._parse_command('/echo "hello world"', "c", "m")
        # ``echo`` isn't registered, so the command is reported as
        # unknown — but the quoted arg should still be one token.
        assert cmd is not None
        assert cmd.name == "echo"
        assert cmd.args == ["hello world"]
        assert cmd.unknown is True

    def test_extra_whitespace_collapsed(self, client: FeishuWSClient) -> None:
        cmd = client._parse_command("   /start    with   extra   spaces  ", "c", "m")
        assert cmd is not None
        assert cmd.name == "start"
        assert cmd.args == ["with", "extra", "spaces"]


class TestParseCommandEdgeCases:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "/",            # bare slash, no name
            "//",           # double slash
            "///",          # all slashes
            "not a command",
            "hello world",
            "1234",
        ],
    )
    def test_not_a_command(self, client: FeishuWSClient, text: str) -> None:
        # Anything that doesn't look like a command returns None so
        # the dispatch thread leaves it alone.
        assert client._parse_command(text, "c", "m") is None

    def test_unknown_command_flagged(self, client: FeishuWSClient) -> None:
        cmd = client._parse_command("/foobar arg1 arg2", "c", "m")
        assert cmd is not None
        assert cmd.name == "foobar"
        assert cmd.args == ["arg1", "arg2"]
        assert cmd.unknown is True

    def test_malformed_quote_does_not_crash(
        self, client: FeishuWSClient
    ) -> None:
        # An unmatched quote would normally make shlex raise; the
        # parser must fall back to plain splitting instead of taking
        # the dispatch thread down with it.
        cmd = client._parse_command("/start 'unterminated quote", "c", "m")
        assert cmd is not None
        assert cmd.name == "start"
        # Both tokens should be present (single quotes are literal
        # characters under plain split).
        assert len(cmd.args) == 2

    def test_unicode_command_name(self, client: FeishuWSClient) -> None:
        # The parser should not crash on non-ASCII names — they just
        # won't be in the registry, so they get the unknown treatment.
        cmd = client._parse_command("/测试", "c", "m")
        assert cmd is not None
        assert cmd.unknown is True


class TestCommandHandler:
    def test_default_unknown_false(self) -> None:
        h = CommandHandler(name="start")
        assert h.unknown is False
        assert h.args == []
        assert h.chat_id == ""
        assert h.message_id == ""

    def test_slots_prevent_extra_attrs(self) -> None:
        h = CommandHandler(name="start")
        with pytest.raises(AttributeError):
            h.not_a_real_field = 1  # type: ignore[attr-defined]


class TestFeishuWSClientCommandsBackwardsCompat:
    def test_commands_tuple_reflects_registry(self) -> None:
        # External code (and a couple of log messages) still introspect
        # ``FeishuWSClient.COMMANDS``. It must stay in sync with the
        # registry so the legacy view doesn't drift.
        assert set(FeishuWSClient.COMMANDS) >= {"init", "start", "stop", "status", "help"}
        # And the set is exactly the registered primary names.
        assert set(FeishuWSClient.COMMANDS) == {
            s.name for s in CommandRegistry.all_specs()
        }


class TestOnMessageLogging:
    """Regression: the per-command trace must NOT spam INFO logs.

    The controller already logs ``Processing command: /<name>`` at
    INFO when it dispatches the parsed command. Having the WS client
    ALSO log ``Received command: ...`` at INFO on every chat message
    means each ``/`` command produces two log lines carrying the
    same information. The WS-client trace belongs at DEBUG — useful
    for routing diagnostics, not for the default log stream.
    """

    @pytest.fixture
    def client_with_callback(
        self, client: FeishuWSClient
    ) -> tuple[FeishuWSClient, MagicMock]:
        """Wire a mock ``_on_command`` so ``_on_message`` dispatches."""
        cb = MagicMock()
        client._on_command = cb
        return client, cb

    def _make_event(
        self, text: str, chat_id: str = "oc_test", message_id: str = "om_1"
    ) -> MagicMock:
        """Build a ``P2ImMessageReceiveV1``-shaped mock the dispatcher can read."""
        event = MagicMock()
        event.event.message.message_type = "text"
        event.event.message.chat_id = chat_id
        event.event.message.message_id = message_id
        # ``message.content`` is a JSON string; ``_on_message`` parses
        # it and reads ``text``.
        import json
        event.event.message.content = json.dumps({"text": text})
        return event

    def test_received_command_log_is_debug_level(
        self, client_with_callback: tuple[FeishuWSClient, MagicMock], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = client_with_callback
        # The bare ``client`` fixture uses a dedicated test logger
        # name (``larksnap.test.feishu_ws``) so the parser tests
        # don't pollute the production module's log namespace.
        # The level of the actual call is what we're verifying, so
        # listen on the right name.
        with caplog.at_level(
            logging.DEBUG, logger="larksnap.test.feishu_ws"
        ):
            client._on_message(self._make_event("/start"))
        debug_records = [
            r for r in caplog.records
            if r.name == "larksnap.test.feishu_ws"
            and "Received command" in r.getMessage()
        ]
        assert len(debug_records) == 1
        assert debug_records[0].levelno == logging.DEBUG

    def test_received_command_not_emitted_at_info(
        self, client_with_callback: tuple[FeishuWSClient, MagicMock], caplog: pytest.LogCaptureFixture
    ) -> None:
        # The default log level for the WS module is INFO. At INFO
        # the verbose per-command trace must NOT appear — only the
        # controller-side ``Processing command`` line should.
        client, _ = client_with_callback
        with caplog.at_level(
            logging.INFO, logger="larksnap.test.feishu_ws"
        ):
            client._on_message(self._make_event("/start"))
        info_records = [
            r for r in caplog.records
            if r.name == "larksnap.test.feishu_ws"
            and "Received command" in r.getMessage()
        ]
        assert info_records == []

    def test_on_command_still_invoked(
        self, client_with_callback: tuple[FeishuWSClient, MagicMock]
    ) -> None:
        # Lowering the log level must not change the actual
        # dispatch behaviour — the controller still gets the
        # parsed command.
        client, cb = client_with_callback
        client._on_message(self._make_event("/start", chat_id="oc_xyz"))
        cb.assert_called_once()
        cmd = cb.call_args[0][0]
        assert cmd.name == "start"
        assert cmd.chat_id == "oc_xyz"

    def test_non_text_message_ignored_silently(
        self, client_with_callback: tuple[FeishuWSClient, MagicMock], caplog: pytest.LogCaptureFixture
    ) -> None:
        # Image / file / audio messages must not produce a trace
        # log either — the handler returns early on the message
        # type check.
        client, cb = client_with_callback
        event = self._make_event("/start")
        event.event.message.message_type = "image"
        with caplog.at_level(
            logging.DEBUG, logger="larksnap.test.feishu_ws"
        ):
            client._on_message(event)
        cb.assert_not_called()
        # No command trace, even at DEBUG, because we never parsed
        # a command in the first place.
        assert all(
            "Received command" not in r.getMessage() for r in caplog.records
        )
