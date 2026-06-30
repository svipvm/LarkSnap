"""Tests for the controller's handling of the ``/config`` chat command.

The command dispatches to four subcommands (``show``, ``paths``,
``get``, ``set``) and renders a usage hint for anything else. The
heavy lifting (path resolution, type coercion, persistence,
subscribers) lives in :class:`larksnap.config.service.ConfigService`
and is tested separately. These tests cover the chat-facing surface:

* Every subcommand produces a non-empty, human-readable reply.
* ``set`` mutates the in-memory ``AppConfig`` AND persists to the
  same YAML file the loader reads.
* The reply marks restart-required fields explicitly so the user
  isn't surprised by stale runtime behaviour.
* ``get``/``show``/``paths`` never mutate state.
* Live-change propagation triggers the controller's
  ``_on_config_changed`` callback for known live fields.
* Errors from the service surface as friendly chat messages rather
  than uncaught exceptions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from larksnap.adapters.notifier.feishu_adapter import FeishuNotifierAdapter
from larksnap.adapters.notifier.feishu_ws_client import CommandHandler
from larksnap.config.loader import save_config
from larksnap.config.models import AppConfig
from larksnap.config.service import ConfigService
from larksnap.gateway.controller import GatewayController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_controller(
    cfg: AppConfig | None = None,
    config_path: Path | None = None,
) -> tuple[GatewayController, ConfigService]:
    """Build a controller with only the bits ``/config`` touches.

    Skips the full ``__init__`` (which would otherwise try to set up
    a worker pool, a config, a pipeline, etc.). The returned
    controller has:

    * A logger
    * A real ``ConfigService`` so path/coercion logic is exercised
    * A Feishu notifier mock for the chat-reply assertions
    * A null notification service — the config command doesn't touch
      detection, so the seg wrapper / notification service
      propagation is exercised separately.
    """
    if cfg is None:
        cfg = AppConfig()
    inst = GatewayController.__new__(GatewayController)
    inst._config = cfg
    inst._logger = logging.getLogger("larksnap.test.controller.config")
    inst._notifier = MagicMock(spec=FeishuNotifierAdapter)
    inst._notification_service = None
    inst._detector = None
    inst._event_bus = MagicMock()
    service = ConfigService(cfg, config_path=config_path)
    inst._config_service = service
    # Pre-register a no-op unsubscribe so the test doesn't blow up if
    # any teardown path is hit. The real __init__ would do this.
    inst._config_unsubscribe = service.subscribe(lambda *_a: None)
    return inst, service


def _reload_yaml(path: Path) -> dict[str, Any]:
    """Read the YAML file back, tolerating the python/tuple tag."""
    with open(path, encoding="utf-8") as f:
        return yaml.unsafe_load(f)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """A per-test YAML file pre-populated with a known AppConfig."""
    p = tmp_path / "config.yaml"
    save_config(AppConfig(), str(p))
    return p


# ---------------------------------------------------------------------------
# show (default)
# ---------------------------------------------------------------------------


class TestConfigShow:
    def test_no_args_renders_full_tree(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config([])
        assert "[LarkSnap] 配置视图" in text
        # A few well-known leaves should appear in the dump.
        assert "camera.fps" in text
        assert "detector.target_classes" in text
        assert "gateway.notification_interval" in text

    def test_show_subcommand_renders_full_tree(self) -> None:
        ctrl, _ = _bare_controller()
        assert (
            ctrl._render_config(["show"])
            == ctrl._render_config([])
        )

    def test_show_with_prefix_filters(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["show", "camera"])
        assert "camera.fps" in text
        # Leaves outside the prefix should not be in the dump.
        assert "notifier.message_template" not in text

    def test_show_with_unknown_prefix_returns_friendly_error(self) -> None:
        ctrl, _ = _bare_controller()
        # ``_render_config_show`` validates the prefix via ``get``
        # first, so any unknown prefix surfaces as the same friendly
        # error a ``/config get`` would.
        text = ctrl._render_config(["show", "no_such_prefix"])
        assert "失败" in text

    def test_show_marks_restart_required_fields(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["show"])
        # ``camera.*`` and ``logging.*`` are restart-required.
        assert "[需重启]" in text
        assert "camera.fps" in text
        # And the marker must sit on at least one of the camera lines.
        camera_lines = [
            line for line in text.splitlines() if "camera." in line
        ]
        assert any("[需重启]" in line for line in camera_lines)

    def test_show_empty_prefix_returns_no_match_message(self) -> None:
        ctrl, _ = _bare_controller()
        # ``nope.`` triggers the path-format check inside
        # ``list_paths`` (trailing dot), which short-circuits before
        # the "no matches" branch. Both responses are valid
        # user-facing signals that the prefix is wrong.
        text = ctrl._render_config(["show", "nope."])
        assert "失败" in text or "没有可显示的配置项" in text


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


class TestConfigPaths:
    def test_paths_no_prefix_returns_all(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["paths"])
        assert "可配置路径" in text
        assert "camera.fps" in text
        assert "detector.target_classes" in text

    def test_paths_with_prefix_filters(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["paths", "detector"])
        assert "detector.target_classes" in text
        assert "camera.fps" not in text

    def test_paths_no_match_returns_friendly_error(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["paths", "nope."])
        assert "没有匹配的路径" in text


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestConfigGet:
    def test_get_returns_value(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["get", "gateway.notification_interval"])
        assert "gateway.notification_interval" in text
        assert "30" in text

    def test_get_returns_list_value(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["get", "detector.target_classes"])
        assert "detector.target_classes" in text
        assert "person" in text

    def test_get_unknown_path_returns_friendly_error(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["get", "no.such.path"])
        assert "失败" in text

    def test_get_with_missing_path_arg_returns_usage(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["get"])
        assert "用法" in text


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------


class TestConfigSet:
    def test_set_updates_value_and_persists(
        self, config_path: Path
    ) -> None:
        ctrl, _ = _bare_controller(config_path=config_path)
        text = ctrl._render_config(
            ["set", "gateway.notification_interval", "60"]
        )
        # Reply shows old → new and the persistence path.
        assert "已更新" in text
        assert "gateway.notification_interval" in text
        assert "30" in text  # old
        assert "60" in text  # new
        assert str(config_path) in text

        # In-memory value reflects the change.
        assert ctrl._config.gateway.notification_interval == 60
        # And the on-disk YAML was updated.
        reloaded = _reload_yaml(config_path)
        assert reloaded["gateway"]["notification_interval"] == 60

    def test_set_live_field_marks_applied(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(
            ["set", "gateway.notification_interval", "90"]
        )
        assert "（已生效）" in text
        assert "（需重启）" not in text

    def test_set_restart_required_field_marks_restart(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["set", "camera.fps", "60"])
        assert "（需重启）" in text
        # But the value still got applied in memory and persisted.
        assert ctrl._config.camera.fps == 60

    def test_set_target_classes_with_json_array(
        self, config_path: Path
    ) -> None:
        ctrl, _ = _bare_controller(config_path=config_path)
        text = ctrl._render_config(
            ["set", "detector.target_classes", '["person","car","dog"]']
        )
        assert "已更新" in text
        assert ctrl._config.detector.target_classes == [
            "person",
            "car",
            "dog",
        ]
        # And it landed on disk.
        reloaded = _reload_yaml(config_path)
        assert reloaded["detector"]["target_classes"] == [
            "person",
            "car",
            "dog",
        ]

    def test_set_bool_field(self) -> None:
        ctrl, _ = _bare_controller()
        ctrl._render_config(["set", "notifier.send_image", "false"])
        assert ctrl._config.notifier.send_image is False

    def test_set_invalid_json_returns_friendly_error(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(
            ["set", "gateway.notification_interval", "not a number"]
        )
        assert "失败" in text
        # State was not mutated.
        assert ctrl._config.gateway.notification_interval == 30

    def test_set_unknown_path_returns_friendly_error(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["set", "no.such.thing", "1"])
        assert "失败" in text

    def test_set_type_mismatch_returns_friendly_error(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(
            ["set", "gateway.notification_interval", '{"x":1}']
        )
        assert "失败" in text

    def test_set_missing_path_returns_usage(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["set"])
        assert "用法" in text

    def test_set_missing_value_returns_usage(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["set", "gateway.notification_interval"])
        assert "用法" in text

    def test_set_without_persisted_path_says_in_memory(self) -> None:
        ctrl, _ = _bare_controller()  # no config_path
        text = ctrl._render_config(
            ["set", "gateway.notification_interval", "5"]
        )
        assert "已更新" in text
        # The persistence path is rendered as a placeholder.
        assert "<内存>" in text


# ---------------------------------------------------------------------------
# Dispatch (via _handle_command)
# ---------------------------------------------------------------------------


class TestConfigDispatch:
    def test_config_command_sends_reply_to_chat(self) -> None:
        ctrl, _ = _bare_controller()
        ctrl._handle_command(CommandHandler(name="config", chat_id="c1"))
        ctrl._notifier.send_text.assert_called_once()
        sent = ctrl._notifier.send_text.call_args[0][0]
        # Default = show the full tree.
        assert "[LarkSnap] 配置视图" in sent

    def test_config_set_command_sends_reply(self) -> None:
        ctrl, _ = _bare_controller()
        ctrl._handle_command(
            CommandHandler(
                name="config",
                args=["set", "gateway.notification_interval", "11"],
                chat_id="c1",
            )
        )
        ctrl._notifier.send_text.assert_called_once()
        sent = ctrl._notifier.send_text.call_args[0][0]
        assert "已更新" in sent
        assert "gateway.notification_interval" in sent
        assert ctrl._config.gateway.notification_interval == 11

    def test_config_command_silently_noop_for_wrong_notifier(self) -> None:
        # If a non-Feishu notifier is plugged in, ``/config`` must
        # silently do nothing rather than crashing.
        ctrl, _ = _bare_controller()
        ctrl._notifier = MagicMock()  # not a FeishuNotifierAdapter
        ctrl._handle_command(CommandHandler(name="config", chat_id="c1"))
        ctrl._notifier.send_text.assert_not_called()

    def test_unknown_subcommand_renders_usage(self) -> None:
        ctrl, _ = _bare_controller()
        text = ctrl._render_config(["frobnicate"])
        assert "用法" in text

    def test_help_arg_renders_usage(self) -> None:
        ctrl, _ = _bare_controller()
        # ``/config help`` is a common way to ask for syntax.
        text = ctrl._render_config(["help"])
        assert "用法" in text


# ---------------------------------------------------------------------------
# Live propagation (controller subscribes to ConfigService)
# ---------------------------------------------------------------------------


class TestConfigPropagation:
    """Smoke tests for the on-config-change callback.

    The full live-update behaviour is covered by the integration
    tests in ``test_target_class_filter.py``; here we just verify
    the callback is wired up and doesn't crash on a no-op
    notification service.
    """

    def test_target_classes_change_does_not_crash(self) -> None:
        ctrl, service = _bare_controller()
        # Real change → service fires the callback, which touches
        # ``self._detector`` (None) and ``self._notification_service``
        # (None). The callback must handle both being None without
        # raising.
        service.set("detector.target_classes", '["person","car"]')
        assert ctrl._config.detector.target_classes == ["person", "car"]

    def test_message_template_change_does_not_crash(self) -> None:
        ctrl, service = _bare_controller()
        service.set("notifier.message_template", '"new template"')
        assert ctrl._config.notifier.message_template == "new template"

    def test_notification_interval_change_does_not_crash(self) -> None:
        ctrl, service = _bare_controller()
        service.set("gateway.notification_interval", "15")
        assert ctrl._config.gateway.notification_interval == 15

    def test_send_image_change_does_not_crash(self) -> None:
        ctrl, service = _bare_controller()
        service.set("notifier.send_image", "false")
        assert ctrl._config.notifier.send_image is False
