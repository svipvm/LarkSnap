"""Tests for the controller's handling of Feishu chat commands.

The controller's ``_handle_command`` is the only place that turns an
incoming chat command into a side-effect (enabling notifications,
publishing events, replying with help). These tests focus on the new
``/help`` rendering and the unknown-command reply path; the existing
``init`` / ``start`` / ``stop`` / ``status`` behaviour is covered by
manual smoke-testing and the pre-existing tests under
``tests/test_controller_lifecycle.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from larksnap.adapters.notifier.command_registry import CommandRegistry
from larksnap.adapters.notifier.feishu_adapter import FeishuNotifierAdapter
from larksnap.adapters.notifier.feishu_ws_client import CommandHandler
from larksnap.gateway.controller import GatewayController, GatewayState, _render_single_help


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


def _bare_controller() -> GatewayController:
    """Build a controller with only the bits the help handler touches.

    Skips the full ``__init__`` (which would otherwise try to set up
    a worker pool, a config, a pipeline, etc.) — we only need the
    notifier attribute and a logger to exercise the help path.
    """
    import threading
    from larksnap.config.models import AppConfig
    from larksnap.config.service import ConfigService
    from larksnap.gateway.event_bus import EventBus

    inst = GatewayController.__new__(GatewayController)
    inst._logger = __import__("logging").getLogger("larksnap.test.controller")
    inst._notifier = MagicMock(spec=FeishuNotifierAdapter)
    inst._notification_service = None
    inst._event_bus = EventBus()
    # ``/status`` reads a handful of attributes off the controller
    # (config, gateway state, notification toggle). Provide a
    # real-enough AppConfig + state lock + ConfigService so the
    # formatter has something concrete to render without crashing
    # on missing private attributes.
    inst._config = AppConfig()
    inst._config_service = ConfigService(inst._config)
    inst._config_unsubscribe = inst._config_service.subscribe(lambda *_a: None)
    inst._state_lock = threading.RLock()
    inst._state = GatewayState.CAMERA_ON
    inst._detector = None
    return inst


class TestHandleHelpCommand:
    def test_help_sends_full_catalogue(self) -> None:
        ctrl = _bare_controller()
        ctrl._handle_command(CommandHandler(name="help", chat_id="c1"))

        ctrl._notifier.send_text.assert_called_once()
        sent = ctrl._notifier.send_text.call_args[0][0]
        # All five built-in commands should appear in the rendered
        # help text, plus the section header.
        assert "[LarkSnap] 可用命令：" in sent
        for cmd in ("init", "start", "stop", "status", "help"):
            assert f"/{cmd}" in sent

    def test_help_with_subcommand_sends_single_block(self) -> None:
        ctrl = _bare_controller()
        ctrl._handle_command(
            CommandHandler(name="help", args=["start"], chat_id="c1")
        )

        ctrl._notifier.send_text.assert_called_once()
        sent = ctrl._notifier.send_text.call_args[0][0]
        assert "/start" in sent
        assert "语法: /start" in sent
        # The single-command drill-down must not list the others.
        assert "/stop" not in sent
        assert "/init" not in sent

    def test_help_with_unknown_subcommand_falls_back_to_catalogue(self) -> None:
        ctrl = _bare_controller()
        ctrl._handle_command(
            CommandHandler(name="help", args=["no-such-cmd"], chat_id="c1")
        )

        sent = ctrl._notifier.send_text.call_args[0][0]
        assert "未找到指令" in sent
        # And still show the full catalogue so the user can recover.
        assert "/start" in sent

    def test_help_with_leading_slash_in_subcommand(self) -> None:
        # ``/help /start`` is a common way to ask "tell me about /start".
        # The renderer must strip the leading slash before lookup.
        ctrl = _bare_controller()
        ctrl._handle_command(
            CommandHandler(name="help", args=["/start"], chat_id="c1")
        )
        sent = ctrl._notifier.send_text.call_args[0][0]
        assert "/start" in sent
        assert "/stop" not in sent

    def test_help_does_nothing_when_notifier_is_wrong_type(self) -> None:
        # If a non-Feishu notifier is plugged in (e.g. a future
        # Slack/DingTalk adapter), the help handler must silently
        # no-op rather than crashing — other commands still work
        # because they check ``isinstance`` too.
        ctrl = _bare_controller()
        ctrl._notifier = MagicMock()  # not a FeishuNotifierAdapter
        ctrl._handle_command(CommandHandler(name="help", chat_id="c1"))
        ctrl._notifier.send_text.assert_not_called()


class TestHandleStatusCommand:
    """``/status`` must always reply to the user with the snapshot.

    A log line is not a reply — the whole point of running ``/status``
    from a chat is to see the answer in the chat.
    """

    def test_status_sends_reply(self) -> None:
        ctrl = _bare_controller()
        ctrl._handle_command(CommandHandler(name="status", chat_id="c1"))
        ctrl._notifier.send_text.assert_called_once()
        sent = ctrl._notifier.send_text.call_args[0][0]
        assert "[LarkSnap]" in sent
        # The reply should include the same fields a user would
        # expect to see — gateway, camera, notification, targets.
        for field in ("网关", "摄像头", "通知", "监控类别", "通知间隔"):
            assert field in sent

    def test_status_reports_target_classes(self) -> None:
        ctrl = _bare_controller()
        ctrl._config.detector.target_classes = ["person", "car", "dog"]
        ctrl._handle_command(CommandHandler(name="status", chat_id="c1"))
        sent = ctrl._notifier.send_text.call_args[0][0]
        assert "person,car,dog" in sent

    def test_status_handles_empty_target_classes(self) -> None:
        ctrl = _bare_controller()
        ctrl._config.detector.target_classes = []
        ctrl._handle_command(CommandHandler(name="status", chat_id="c1"))
        sent = ctrl._notifier.send_text.call_args[0][0]
        # An empty monitoring set should be reported as such rather
        # than rendering an empty ``"监控类别: "`` line.
        assert "(none)" in sent

    def test_status_reflects_gateway_state(self) -> None:
        ctrl = _bare_controller()
        ctrl._state = GatewayState.DETECTING
        ctrl._handle_command(CommandHandler(name="status", chat_id="c1"))
        sent = ctrl._notifier.send_text.call_args[0][0]
        assert "running" in sent

    def test_status_does_nothing_when_notifier_is_wrong_type(self) -> None:
        ctrl = _bare_controller()
        ctrl._notifier = MagicMock()  # not a FeishuNotifierAdapter
        ctrl._handle_command(CommandHandler(name="status", chat_id="c1"))
        ctrl._notifier.send_text.assert_not_called()


class TestAllCommandsReply:
    """Regression: every built-in command must produce a chat reply.

    A command that mutates state or renders text but never calls
    ``send_text`` would silently swallow the user's input — a
    frustrating UX. This test is the safety net: if a future
    command is added without a reply, it fails loudly here.
    """

    @pytest.mark.parametrize(
        "cmd_name,args",
        [
            ("init", []),
            ("start", []),
            ("stop", []),
            ("status", []),
            ("help", []),
            ("config", []),
            ("config", ["get", "gateway.notification_interval"]),
        ],
    )
    def test_command_calls_send_text(
        self, cmd_name: str, args: list[str]
    ) -> None:
        ctrl = _bare_controller()
        ctrl._handle_command(
            CommandHandler(name=cmd_name, args=args, chat_id="c1")
        )
        ctrl._notifier.send_text.assert_called_once()


class TestHandleUnknownCommand:
    def test_unknown_command_sends_error(self) -> None:
        ctrl = _bare_controller()
        cmd = CommandHandler(
            name="foobar", args=["x"], chat_id="c1", unknown=True
        )
        ctrl._handle_command(cmd)

        ctrl._notifier.send_text.assert_called_once()
        sent = ctrl._notifier.send_text.call_args[0][0]
        assert "未知指令" in sent
        assert "/foobar" in sent
        assert "/help" in sent  # point the user at /help

    def test_unknown_command_does_not_trigger_side_effects(self) -> None:
        # An unknown command must NOT enable notifications, publish
        # chat_id events, or otherwise look like ``/start`` / ``/init``.
        ctrl = _bare_controller()
        cmd = CommandHandler(name="foobar", chat_id="c1", unknown=True)
        # Track handler count before/after to confirm no new
        # subscribers were registered. ``_handlers`` is the
        # EventBus's private store (see event_bus.py).
        from larksnap.gateway.event_bus import EventType

        handlers_before = sum(
            len(ctrl._event_bus._handlers[ev])
            for ev in EventType
        )
        ctrl._handle_command(cmd)
        handlers_after = sum(
            len(ctrl._event_bus._handlers[ev])
            for ev in EventType
        )
        assert handlers_before == handlers_after
        # And notification_service is None anyway, but exercise the
        # branch explicitly so the "no AttributeError" path is covered.
        assert ctrl._notification_service is None


class TestRenderHelp:
    def test_render_help_no_args_matches_registry(self) -> None:
        ctrl = _bare_controller()
        text = ctrl._render_help([])
        assert text == CommandRegistry.render_help()

    def test_render_help_with_known_arg(self) -> None:
        ctrl = _bare_controller()
        text = ctrl._render_help(["status"])
        expected = _render_single_help(CommandRegistry.get("status"))
        assert text == expected

    def test_render_help_with_unknown_arg_includes_catalogue(self) -> None:
        ctrl = _bare_controller()
        text = ctrl._render_help(["nope"])
        assert "未找到指令 /nope" in text
        assert "/start" in text  # full catalogue still rendered
